from __future__ import print_function
import sys
import os
import numpy as np
import time
import datetime
import json
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from Utils import utils
from Network.BigGAN import losses
from Dataset import datasets as dset, animal_hash
from Dataset import mini_datasets as mdset
from tqdm import tqdm
import h5py as h5


def SL_training_function(G, D, GD, z_, y_, ema, state_dict, config):
    def train(x, y):
        G.optim.zero_grad()
        D.optim.zero_grad()
        # How many chunks to split x and y into?
        x = torch.split(x, config['batch_size'])
        y = torch.split(y, config['batch_size'])
        counter = 0

        # Optionally toggle D and G's "require_grad"
        if config['toggle_grads']:
            utils.toggle_grad(D, True)
            utils.toggle_grad(G, False)

        for step_index in range(config['num_D_steps']):
            # If accumulating gradients, loop multiple times before an optimizer step
            D.optim.zero_grad()
            for accumulation_index in range(config['num_D_accumulations']):
                z_.sample_()
                D_fake, D_real = GD(z_[:config['batch_size']], y_[:config['batch_size']],
                                    x[counter], y[counter], train_G=False,
                                    split_D=config['split_D'])

                # Compute components of D's loss, average them, and divide by
                # the number of gradient accumulations
                D_loss_real, D_loss_fake = losses.discriminator_loss(D_fake, D_real)
                D_loss = (D_loss_real + D_loss_fake) / float(config['num_D_accumulations'])
                D_loss.backward()
                counter += 1

            # Optionally apply ortho reg in D
            if config['D_ortho'] > 0.0:
                # Debug print to indicate we're using ortho reg in D.
                print('using modified ortho reg in D')
                utils.ortho(D, config['D_ortho'])

            D.optim.step()

        # Optionally toggle "requires_grad"
        if config['toggle_grads']:
            utils.toggle_grad(D, False)
            utils.toggle_grad(G, True)

        # Zero G's gradients by default before training G, for safety
        G.optim.zero_grad()

        # If accumulating gradients, loop multiple times
        for accumulation_index in range(config['num_G_accumulations']):
            z_.sample_()
            D_fake = GD(z_, y_, train_G=True, split_D=config['split_D'])
            G_loss = losses.generator_loss(D_fake) / float(config['num_G_accumulations'])
            G_loss.backward()

        # Optionally apply modified ortho reg in G
        if config['G_ortho'] > 0.0:
            print('using modified ortho reg in G')  # Debug print to indicate we're using ortho reg in G
            # Don't ortho reg shared, it makes no sense. Really we should blacklist any embeddings for this
            utils.ortho(G, config['G_ortho'],
                        blacklist=[param for param in G.shared.parameters()])
        G.optim.step()

        # If we have an ema, update it, regardless of if we test with it or not
        if config['ema']:
            ema.update(state_dict['itr'])

        out = {'G_loss': float(G_loss.item()),
               'D_loss_real': float(D_loss_real.item()),
               'D_loss_fake': float(D_loss_fake.item())}
        # Return G's loss and the components of D's loss.
        return out

    return train


def get_minidata_loaders(dataset, data_root=None, augment=False, batch_size=64,
                     num_workers=8, shuffle=True, load_in_mem=False, hdf5=False,
                     pin_memory=True, drop_last=True, start_itr=0,
                     num_epochs=500, use_multiepoch_sampler=False,
                     index_dir='/gpub/temp/imagenet2012/hdf5',
                     **kwargs):
    # Append /FILENAME.hdf5 to root if using hdf5
    data_root += '/%s' % utils.root_dict[dataset]
    print('Using dataset root location %s' % data_root)

    which_dataset = mdset.MiniImagenet
    norm_mean = [0.5, 0.5, 0.5]
    norm_std = [0.5, 0.5, 0.5]
    image_size = 128
    # For image folder datasets, name of the file where we store the precomputed
    # image locations to avoid having to walk the dirs every time we load.
    dataset_kwargs = {'index_filename': '%s/%s_imgs.npz' % (index_dir, dataset)}

    # HDF5 datasets have their own inbuilt transform, no need to train_transform
    if 'hdf5' in dataset:
        train_transform = None
    else:
        if augment:
            print('Data will be augmented...')
            if dataset in ['C10', 'C100']:
                train_transform = [transforms.RandomCrop(32, padding=4),
                                   transforms.RandomHorizontalFlip()]
            else:
                train_transform = [utils.RandomCropLongEdge(),
                                   transforms.Resize(image_size),
                                   transforms.RandomHorizontalFlip()]
        else:
            print('Data will not be augmented...')
            if dataset in ['C10', 'C100']:
                train_transform = []
            else:
                train_transform = [utils.CenterCropLongEdge(), transforms.Resize(image_size)]
            # train_transform = [transforms.Resize(image_size), transforms.CenterCrop]
        train_transform = transforms.Compose(train_transform + [
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)])
    train_set = which_dataset(root=data_root, transform=train_transform,
                              load_in_mem=load_in_mem, **dataset_kwargs)

    # Prepare loader; the loaders list is for forward compatibility with
    # using validation / test splits.
    loaders = []
    if use_multiepoch_sampler:
        print('Using multiepoch sampler from start_itr %d...' % start_itr)
        loader_kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory, 'drop_last': drop_last}
        sampler = utils.MultiEpochSampler(train_set, num_epochs, start_itr, batch_size)
        train_loader = DataLoader(train_set, batch_size=batch_size,
                                  sampler=sampler, **loader_kwargs)
    else:
        loader_kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory,
                         'drop_last': drop_last}  # Default, drop last incomplete batch
        train_loader = DataLoader(train_set, batch_size=batch_size,
                                  shuffle=shuffle, **loader_kwargs)
    loaders.append(train_loader)
    return loaders


def load_weights(net_list, state_dict, weights_root, experiment_name,
                 name_suffix=None, ema_list=None, strict=True, load_optim=True):
    root = '/'.join([weights_root, experiment_name])
    if name_suffix:
        print('Loading %s weights from %s...' % (name_suffix, root))
    else:
        print('Loading weights from %s...' % root)
    for net in net_list:
        net.load_state_dict(
            torch.load('%s/%s.pth' % (root, utils.join_strings('_', [net.name, name_suffix]))),
            strict=strict)
        if load_optim:
            net.optim.load_state_dict(
                torch.load('%s/%s.pth' % (root, utils.join_strings('_', [net.name, 'optim', name_suffix]))))
    for item in state_dict:
        state_dict[item] = torch.load('%s/%s.pth' % (root,
                                                     utils.join_strings('_', ['state_dict', name_suffix])))[item]
    if ema_list is not None:
        for ema in ema_list:
            ema.load_state_dict(
                torch.load('%s/%s.pth' % (root, utils.join_strings('_', [ema.name, name_suffix]))),
                strict=strict)


def save_weights(net_list, state_dict, weights_root, experiment_name,
                 name_suffix=None, ema_list=None):
    root = '/'.join([weights_root, experiment_name])
    if not os.path.exists(root):
        os.mkdir(root)
    if name_suffix:
        print('Saving weights to %s/%s...' % (root, name_suffix))
    else:
        print('Saving weights to %s...' % root)
    for net in net_list:
        torch.save(net.state_dict(),
                   '%s/%s.pth' % (root, utils.join_strings('_', [net.name, name_suffix])))
        torch.save(net.optim.state_dict(),
                   '%s/%s.pth' % (root, utils.join_strings('_', [net.name, 'optim', name_suffix])))
    torch.save(state_dict,
               '%s/%s.pth' % (root, utils.join_strings('_', ['state_dict', name_suffix])))
    if ema_list is not None:
        for ema in ema_list:
            torch.save(ema.state_dict(),
                       '%s/%s.pth' % (root, utils.join_strings('_', [ema.name, name_suffix])))


class KNN(object):
    """
    FTWS: Need to finish
    """
    def __init__(self,
                 dataloader,
                 K=5,
                 sample_batch=10,
                 anchor_num=10,
                 device='cuda',
                 load_anchor_dir='/ghome/fengrl/home/FGAN/KNN_anchor.npy'):
        self.K = K
        assert K <= anchor_num
        self.device = device
        self.sample_batch = sample_batch
        dataset = dataloader.dataset
        if load_anchor_dir is not None:
            print('Loading KNN parameters from %s ...' % load_anchor_dir)
            file = np.load(load_anchor_dir).item()
            self.anchor, self.anchor_label, self.index = file['anchor'], file['anchor_label'], file['index']
            self.anchor = torch.tensor(self.anchor)
        else:
            self.anchor, self.anchor_label, self.index = make_anchor(dataset, anchor_num)
        self.anchor = self.anchor.to(device)
        self.dataloader = dataloader

    def __call__(self, encoder):
        """
        FTWS: return dict as out in train_fns.
        """
        precision = 0.0
        with torch.no_grad():
            anchor_v = encoder(self.anchor).split(1, 0)
            for i, (x, y) in tqdm(enumerate(self.dataloader)):
                x, y = x.to(self.device), y.to(self.device)
                if i > self.sample_batch:
                    break
                v = encoder(x)
                dist = []
                for anchor in anchor_v:
                    dist.append(torch.norm(v - anchor, 2, 1, keepdim=True))
                dist = torch.cat(dist, 1)    # [batch_size, anchor_num]
                y_arg = dist.argsort(1)
                for k in range(self.K):
                    y_ = self.anchor_label[y_arg[:, k]]
                    precision += (torch.tensor(y_) == torch.tensor(y)).sum().type_as(x) / y.shape[0]
                precision = float(precision / self.K)
                del x, y, v
        return precision / self.sample_batch


def make_anchor(dataset, anchor_num):
    with h5.File(dataset.root, 'r') as f:
        dlabel = f['labels'][:]
    anchor_list = []
    counter = 0
    label_record = 0
    print("Generating KNN anchor with %i anchors per class." % anchor_num)
    for index in tqdm(range(len(dlabel))):
        label = dlabel[index]
        if counter < anchor_num:
            anchor_list.append(index)
            counter += 1
        elif label != label_record:
            label_record = label
            anchor_list.append(index)
            counter = 1
    del dlabel
    return dataset[anchor_list] + (anchor_list,)


def make_index_per_class(dataset, anchor_num):
    anchor_list = []
    remain_list = []
    counter = 0
    label_record = 0
    print("Generating Few shot anchor with %d anchors per class." % anchor_num)
    for index in tqdm(range(len(dataset))):
        label = dataset[index]
        if counter < anchor_num:
            anchor_list.append(index)
            counter += 1
        elif label != label_record:
            label_record = label
            anchor_list.append(index)
            counter = 1
        else:
            remain_list.append(index)
    return anchor_list, remain_list


# Sample function for use with inception metrics
def sample(Invert, G, z_, y_, config):
    with torch.no_grad():
        z_.sample_()
        y_.sample_()
        if config['parallel']:
            v = nn.parallel.data_parallel(Invert, z_)
            G_z = nn.parallel.data_parallel(G, (v, G.shared(y_)))
        else:
            v = Invert(z_)
            G_z = G(v, G.shared(y_))
        return G_z, y_


def update_config_roots(config):
  if config['base_root']:
    print('Pegging all root folders to base root %s' % config['base_root'])
    for key in ['weights', 'logs', 'samples']:
      config['%s_root' % key] = '%s/%s' % (config['base_root'], key)
  return config


def accumulate_standing_stats(net_list, z, y, nclasses, num_accumulations=16):
    for net in net_list:
        utils.initiate_standing_stats(net)
        net.train()
        for i in range(num_accumulations):
            with torch.no_grad():
                z.normal_()
                y.random_(0, nclasses)
                x = net(z, net.shared(y)) # No need to parallelize here unless using syncbn
        # Set to eval mode
        net.eval()


def prepare_fixed_x(dataloader, G_batch_size, config, experiment_name, device='cuda'):
    x, y = dataloader.__iter__().__next__()
    x = x.to(device)
    y = y.to(device)
    x = torch.split(x, G_batch_size)[0]
    y = torch.split(y, G_batch_size)[0]
    if config['G_fp16']:
        x = x.half()
        y = y.half()
    if not os.path.isdir('%s/%s' % (config['samples_root'], experiment_name)):
        os.mkdir('%s/%s' % (config['samples_root'], experiment_name))
    image_filename = '%s/%s/fixed_x.jpg' % (config['samples_root'],
                                            experiment_name)
    torchvision.utils.save_image(x.float().cpu(), image_filename,
                                 nrow=int(x.shape[0] ** 0.5), normalize=True)
    return x, y


def sample_sheet(G, I, classes_per_sheet, num_classes, samples_per_class, parallel,
                 samples_root, experiment_name, folder_number, z_=None):
    # Prepare sample directory
    if not os.path.isdir('%s/%s' % (samples_root, experiment_name)):
        os.mkdir('%s/%s' % (samples_root, experiment_name))
    if not os.path.isdir('%s/%s/%d' % (samples_root, experiment_name, folder_number)):
        os.mkdir('%s/%s/%d' % (samples_root, experiment_name, folder_number))
    # loop over total number of sheets
    for i in range(num_classes // classes_per_sheet):
        ims = []
        y = torch.arange(i * classes_per_sheet, (i + 1) * classes_per_sheet, device='cuda')
        for j in range(samples_per_class):
            if (z_ is not None) and hasattr(z_, 'sample_') and classes_per_sheet <= z_.size(0):
                z_.sample_()
            else:
                z_ = torch.randn(classes_per_sheet, G.dim_z, device='cuda')
            with torch.no_grad():
                if parallel:
                    v = nn.parallel.data_parallel(I, z_[:classes_per_sheet])
                    o = nn.parallel.data_parallel(G, (v, G.shared(y)))
                else:
                    v = I(z_[:classes_per_sheet])
                    o = G(v, G.shared(y))

            ims += [o.data.cpu()]
        # This line should properly unroll the images
        out_ims = torch.stack(ims, 1).view(-1, ims[0].shape[1], ims[0].shape[2],
                                           ims[0].shape[3]).data.float().cpu()
        # The path for the samples
        image_filename = '%s/%s/%d/samples%d.jpg' % (samples_root, experiment_name,
                                                     folder_number, i)
        torchvision.utils.save_image(out_ims, image_filename,
                                     nrow=samples_per_class, normalize=True)


# interp sheet function
# Supports full, class-wise and intra-class interpolation
def interp_sheet(G, I, num_per_sheet, num_midpoints, num_classes, parallel,
                 samples_root, experiment_name, folder_number, sheet_number=0,
                 fix_z=False, fix_y=False, device='cuda'):
    # Prepare zs and ys
    if fix_z: # If fix Z, only sample 1 z per row
        zs = torch.randn(num_per_sheet, 1, G.dim_z, device=device)
        zs = zs.repeat(1, num_midpoints + 2, 1).view(-1, G.dim_z)
    else:
        zs = utils.interp(torch.randn(num_per_sheet, 1, G.dim_z, device=device),
                          torch.randn(num_per_sheet, 1, G.dim_z, device=device),
                          num_midpoints).view(-1, G.dim_z)
    if fix_y: # If fix y, only sample 1 z per row
        ys = utils.sample_1hot(num_per_sheet, num_classes)
        ys = G.shared(ys).view(num_per_sheet, 1, -1)
        ys = ys.repeat(1, num_midpoints + 2, 1).view(num_per_sheet * (num_midpoints + 2), -1)
    else:
        ys = utils.interp(G.shared(utils.sample_1hot(num_per_sheet, num_classes)).view(num_per_sheet, 1, -1),
                          G.shared(utils.sample_1hot(num_per_sheet, num_classes)).view(num_per_sheet, 1, -1),
                          num_midpoints).view(num_per_sheet * (num_midpoints + 2), -1)
    # Run the net--note that we've already passed y through G.shared.
    if G.fp16:
        zs = zs.half()
    with torch.no_grad():
        if parallel:
            vs = nn.parallel.data_parallel(I, zs).data.cpu()
            out_ims = nn.parallel.data_parallel(G, (vs, ys)).data.cpu()
        else:
            vs = I(zs).data.cpu()
            out_ims = G(vs, ys).data.cpu()
    interp_style = '' + ('Z' if not fix_z else '') + ('Y' if not fix_y else '')
    image_filename = '%s/%s/%d/interp%s%d.jpg' % (samples_root, experiment_name,
                                                  folder_number, interp_style,
                                                  sheet_number)
    torchvision.utils.save_image(out_ims, image_filename,
                                 nrow=num_midpoints + 2, normalize=True)


def general_one_hot(y, n_classes):
    """
    -1 marks samples which dont have labels, this function will convert them into [0,...,0]
    :param y:
    :param n_classes:
    :return:
    """
    y = y + 1
    y_one_hot = torch.nn.functional.one_hot(y, n_classes+1)
    y_one_hot = y_one_hot[:, 1:]
    return y_one_hot


def sample_for_SL(G, z_, y_, config):
  with torch.no_grad():
    z_.sample_()
    y_
    if config['parallel']:
      G_z =  nn.parallel.data_parallel(G, (z_, G.shared(y_)))
    else:
      G_z = G(z_, G.shared(y_))
    return G_z, y_


class dense_eval(nn.Module):
    def __init__(self, in_feature, n_classes=1000, steps=5):
        super(dense_eval, self).__init__()
        self.steps = steps
        self.dense = nn.Linear(in_feature, n_classes)
        self.softmax = nn.Sigmoid(dim=1)
        self.init_weight()

    def init_weight(self):
        torch.nn.init.zeros_(self.dense.weight)

    def forward(self, x):
        x = self.dense(x)
        x = self.softmax(x)
        return x


def eval_encoder(Encoder, loader, dense_eval: nn.Module, config, sample_batch=10):
    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    optim = torch.optim.Adam(params=dense_eval.parameters(), lr=0.001)

    def model(x, y):
        if config['parallel']:
            with torch.no_grad():
                x = nn.parallel.data_parallel(Encoder, x)
            x = nn.parallel.data_parallel(dense_eval, x)
            loss = torch.nn.parallel.data_parallel(loss_fn, (x, y))
        else:
            with torch.no_grad():
                x = Encoder(x)
            x = dense_eval(x)
            loss = loss_fn(x)
        return loss

    def train(x, y):
        optim.zero_grad()
        loss = model(x, y)
        loss.backward()
        optim.step()
        return loss

    for i, (x, y) in enumerate(loader):
        if i > dense_eval.steps:
            break
        train(x, y)
        del x, y
    del optim, loss_fn

    loss = 0.0
    for i, (x, y) in enumerate(loader):
        if i > sample_batch:
            break
        with torch.no_grad():
            loss += model(x, y) / x.shape[0]
        del x, y
    return loss






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
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from Utils import utils
from Dataset import datasets as dset, animal_hash


def load_weights(net_list, state_dict, weights_root, experiment_name, load_whole=False,
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
                 anchor_num=10):
        self.K = K
        assert K <= anchor_num
        self.sample_batch = sample_batch
        self.anchor, self.anchor_label = make_anchor(dataloader, anchor_num)
        self.dataloader = dataloader

    def __call__(self, encoder):
        """
        FTWS: return dict as out in train_fns.
        """
        precision = 0
        with torch.no_grad:
            anchor_v = encoder(self.anchor).split(1, 0)
            for i, (x, y) in enumerate(self.dataloader):
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
                    precision += 1.0 * (y_ == y) / y.shape[0]
                precision = float(precision / self.K)
            del x, y, v
        return precision / self.sample_batch


def make_anchor(dataloader, anchor_num):
    dataset = dataloader.dataset
    anchor_list = []
    counter = 0
    label_record = 0
    print("Generating KNN anchor with %i anchors per class." % anchor_num)
    for index in range(len(dataset)):
        _, label = dataset[index]
        if counter < anchor_num:
            anchor_list.append(index)
            counter += 1
        elif label != label_record:
            label_record = label
            anchor_list.append(index)
            counter = 1
    return dataset[anchor_list]


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
    x = torch.split(x, G_batch_size)[0]
    if config['G_fp16']:
        x = x.half()
    if not os.path.isdir('%s/%s' % (config['samples_root'], experiment_name)):
        os.mkdir('%s/%s' % (config['samples_root'], experiment_name))
    image_filename = '%s/%s/fixed_x.jpg' % (config['samples_root'],
                                            experiment_name)
    torchvision.utils.save_image(x.float().cpu(), image_filename,
                                 nrow=int(x.shape[0] ** 0.5), normalize=True)
    return x


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

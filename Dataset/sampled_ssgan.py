import h5py as h5
import torch
import torch.utils.data as data
import numpy as np
from torch.utils.data import DataLoader
import os
from Utils import utils, vae_utils


class SSGAN_HDF5(data.Dataset):
    def __init__(self, root, start=0, end=-1, transform=None, target_transform=None,
                 load_in_mem=False, train=True, download=False, validate_seed=0,
                 val_split=0, **kwargs):  # last four are dummies

        self.root = root
        self.num_imgs = end - start

        # self.transform = transform
        self.target_transform = target_transform

        # Set the transform here
        self.transform = transform

        # load the entire dataset into memory?
        self.load_in_mem = load_in_mem

        # If loading into memory, do so now
        if self.load_in_mem:
            print('Loading %s into memory, index range from %d to %d...' % (root, start, end))
            with h5.File(root, 'r') as f:
                self.img = f['img'][start: end]
                self.z = f['z'][start: end]
            with h5.File(root.replace('SSGAN128', 'wema'), 'r') as f:
                self.w = f['w'][start: end]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        # If loaded the entire dataset in RAM, get image from memory
        if self.load_in_mem:
            img = self.img[index]
            z = self.z[index]
            w = self.w[index]

        # Else load it from disk
        else:
            with h5.File(self.root, 'r') as f:
                img = f['img'][index]
                z = f['z'][index]
                w = f['w'][index]
            with h5.File(self.root.replace('SSGAN128', 'wema'), 'r') as f:
                self.w = f['w'][index]

        # if self.transform is not None:
        # img = self.transform(img)
        # Apply my own transform
        img = (torch.from_numpy(img).float() - 0.5) * 2
        img = img.permute([2, 0, 1])
        z = torch.from_numpy(z).float()
        w = torch.from_numpy(w).float()

        # return img, z, w
        return img, z, w

    def __len__(self):
        return self.num_imgs
        # return len(self.f['imgs'])


def make_dset_range(root, piece=6, batch_size=64):
    with h5.File(root, 'r') as f:
        num_samples = len(f['z'])
    batch_num = num_samples // batch_size
    per_set_batch = batch_num // piece
    per_set_num = per_set_batch * batch_size
    start = []
    end = []
    for i in range(piece):
        start.append(i * per_set_num)
        end.append((i+1) * per_set_num)
    end[piece - 1] = num_samples
    return start, end


def get_SSGAN_sample_loader(ssgan_sample_root, start, end, batch_size=64, shuffle=False, num_workers=8,
                            load_in_mem=True, pin_memory=True,
                            drop_last=True, **kwargs):
    print('Using SSGAN sample location % s' % ssgan_sample_root)
    dset = SSGAN_HDF5(ssgan_sample_root, start, end, load_in_mem=load_in_mem)
    loader_kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory, 'drop_last': drop_last}
    loader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, **loader_kwargs)
    return loader


def save_and_eavl(E, Out, E_ema, O_ema, state_dict, config, experiment_name, eval_fn=None, test_log=None):
    if config['num_save_copies'] > 0:
        vae_utils.save_weights([E, Out], state_dict, config['weights_root'],
                               experiment_name,
                               'copy%d' % state_dict['save_num'],
                               [E_ema, O_ema] if config['ema'] else [None])
        state_dict['save_num'] = (state_dict['save_num'] + 1) % config['num_save_copies']
    if eval_fn is not None:
        which_E = E_ema if config['ema'] and config['use_ema'] else E
        precise = eval_fn(which_E)
        if precise > state_dict['best_precise']:
            print('KNN precise improved over previous best, saving checkpoint...')
            vae_utils.save_weights([E], state_dict, config['weights_root'],
                                   experiment_name, 'best%d' % state_dict['save_best_num'],
                                   [E_ema if config['ema'] else None])
            state_dict['save_best_num'] = (state_dict['save_best_num'] + 1) % config['num_best_copies']
        state_dict['best_precise'] = max(state_dict['best_precise'], precise)

        test_log.log(itr=int(state_dict['itr']), precise=float(precise))



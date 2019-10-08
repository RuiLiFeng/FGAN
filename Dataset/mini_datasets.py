import pickle
import torch.utils.data as data
import torch
import numpy as np
from Utils import vae_utils, utils
import h5py as h5


class MiniImagenet(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, split='train', **kwargs):
        assert split in ['train', 'test', 'val']
        self.split = split
        self.root = root
        self.index_root = '/gdata/fengrl/fgan/data/mini-imagenet-cache-train.pkl'
        print('Loading %s into memory...' % self.root)
        with open(self.index_root, 'rb') as f:
            self.data = pickle.load(f)
            self.class_dict = self.data['class_dict']
        self.index = []
        for key in self.class_dict:
            self.index += self.class_dict[key]

        with h5.File(root, 'r') as f:
            self.data = f['imgs'][self.index]
            self.labels = f['labels'][self.index]

        self.num_imgs = len(self.labels)

        # self.transform = transform
        self.target_transform = target_transform

        # Set the transform here
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        img = self.data[index]
        target = self.labels[index]

        # if self.transform is not None:
        # img = self.transform(img)
        # Apply my own transform
        img = ((torch.from_numpy(img).float() / 255) - 0.5) * 2

        if self.target_transform is not None:
            target = self.target_transform(target)

        # return img, int(target)
        return img, target

    def __len__(self):
        return self.num_imgs
        # return len(self.f['imgs'])


class FImagenet(data.Dataset):
    def __init__(self, root, index_list, labeled_or_not='labeled', transform=None, target_transform=None, **kwargs):
        self.root = root
        self.num_imgs = len(index_list)
        self.labeled_or_not = labeled_or_not
        print('Loading %s with $s %d imgs into memory...' % (self.root, labeled_or_not, self.num_imgs))
        self.index_list = index_list

        with h5.File(root, 'r') as f:
            self.data = f['imgs'][index_list]
            self.labels = f['labels'][index_list]
        if self.labeled_or_not == 'unlabeled':
            self.Flabels = -1 * np.ones_like(self.labels)
        else:
            self.Flabels = self.labels

        # self.transform = transform
        self.target_transform = target_transform

        # Set the transform here
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        img = self.data[index]
        target = self.Flabels[index]

        # if self.transform is not None:
        # img = self.transform(img)
        # Apply my own transform
        img = ((torch.from_numpy(img).float() / 255) - 0.5) * 2

        if self.target_transform is not None:
            target = self.target_transform(target)

        # return img, int(target)
        return img, target

    def __len__(self):
        return self.num_imgs
        # return len(self.f['imgs'])


def make_fewshot_dataset(root, num_per_class, **dataset_kwargs):
    with h5.File(root, 'r') as f:
        labels = f['labels'][:]
    labeled_index, unlabeled_index = vae_utils.make_index_per_class(labels, num_per_class)
    labeled_dataset = FImagenet(root, labeled_index, 'labeled', **dataset_kwargs)
    unlabeled_dataset = FImagenet(root, unlabeled_index, 'unlabeled', **dataset_kwargs)
    return labeled_dataset, unlabeled_dataset


# Convenience function to centralize all data loaders
def get_data_loaders(dataset, B, num_per_class=50, data_root=None, augment=False, batch_size=64,
                     num_workers=8, shuffle=True, load_in_mem=False, hdf5=False,
                     pin_memory=True, drop_last=True, start_itr=0,
                     num_epochs=500, use_multiepoch_sampler=False,
                     index_dir='/gpub/temp/imagenet2012/hdf5',
                     **kwargs):
    # FTWS: Add index_dir to load index file.

    # Append /FILENAME.hdf5 to root if using hdf5
    data_root += '/%s' % utils.root_dict[dataset]
    print('Using dataset root location %s' % data_root)

    which_dataset = FImagenet
    norm_mean = [0.5, 0.5, 0.5]
    norm_std = [0.5, 0.5, 0.5]
    image_size = utils.imsize_dict[dataset]
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
                train_transform = [utils.transforms.RandomCrop(32, padding=4),
                                   utils.transforms.RandomHorizontalFlip()]
            else:
                train_transform = [utils.RandomCropLongEdge(),
                                   utils.transforms.Resize(image_size),
                                   utils.transforms.RandomHorizontalFlip()]
        else:
            print('Data will not be augmented...')
            if dataset in ['C10', 'C100']:
                train_transform = []
            else:
                train_transform = [utils.CenterCropLongEdge(), utils.transforms.Resize(image_size)]
            # train_transform = [transforms.Resize(image_size), transforms.CenterCrop]
        train_transform = utils.transforms.Compose(train_transform + [
            utils.transforms.ToTensor(),
            utils.transforms.Normalize(norm_mean, norm_std)])
    ldset, uldset = make_fewshot_dataset(root=data_root, num_per_class=num_per_class, transform=train_transform,
                                         load_in_mem=load_in_mem, **dataset_kwargs)

    # Prepare loader; the loaders list is for forward compatibility with
    # using validation / test splits.
    loaders = []
    if use_multiepoch_sampler:
        print('Using multiepoch sampler from start_itr %d...' % start_itr)
        loader_kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory}
        l_sampler = utils.MultiEpochSampler(ldset, num_epochs, start_itr, batch_size - B)
        l_loader = utils.DataLoader(ldset, batch_size=batch_size - B,
                                    sampler=l_sampler, **loader_kwargs)
        u_sampler = utils.MultiEpochSampler(uldset, num_epochs, start_itr, B)
        u_loader = utils.DataLoader(uldset, batch_size=B,
                                    sampler=u_sampler, **loader_kwargs)
    else:
        loader_kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory,
                         'drop_last': drop_last}  # Default, drop last incomplete batch
        l_loader = utils.DataLoader(ldset, batch_size=batch_size - B,
                                    shuffle=shuffle, **loader_kwargs)
        u_loader = utils.DataLoader(uldset, batch_size=B,
                                    shuffle=shuffle, **loader_kwargs)
    loaders.append(l_loader)
    loaders.append(u_loader)
    return loaders


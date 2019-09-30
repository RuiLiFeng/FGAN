import pickle
import torch.utils.data as data
import torch
import numpy as np
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

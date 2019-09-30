import pickle
import torch.utils.data as data
import torch


class MiniImagenet(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, split='train', **kwargs):
        assert split in ['train', 'test', 'val']
        self.split = split
        self.root = root + '/mini-imagenet-cache-%s.pkl' % split
        print('Loading %s into memory...' % self.root)
        with open(self.root, 'r') as f:
            self.data = pickle.load(f)
            self.img = self.data['image_data']
            self.class_dict = self.data['class_dict']
        self.labels = []
        for key in self.class_dict:
            self.labels += self.class_dict[key]

        self.num_imgs = len(self.img)

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
        img = self.img[index]
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

import functools

import torch
import torch.nn as nn
import numpy as np
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
import torchvision

from Network.BigGAN import layers
from Network.BigGAN import BigGAN
from Network.VaeGAN import Invert

NUM_ROTATIONS = 4


def merge_with_rotation_data(images, labels, num_rot_examples):
    """
    Returns the original data concatenated with the rotated versions.
    Put all rotation angles in a single batch, the first batch_size are the original up-right images, followed by
    rotated_batch_size * 3 rotated images with 3 different angles. For NUM_ROTATions = 4 and num_rot_examples=2 we have
    labels_rotated [0, 0, 1, 1, 2, 2, 3, 3]
    :param images: [NCHW]
    :param labels: [batch_size]
    :param num_rot_examples:
    :return:
    """
    img_to_rot = images[-num_rot_examples:]
    img_rotated = rotate_images(img_to_rot, rot90_scalars=(1, 2, 3))
    img_rotated_labels = labels[-num_rot_examples:].repeat(3)
    all_img = torch.cat([images, img_rotated], 0)
    all_labels = torch.cat([labels, img_rotated_labels], 0)
    labels_rotated = tile(torch.tensor([1, 2, 3], dtype=labels.dtype), 0, num_rot_examples)
    all_labels_rotated = torch.cat([torch.zeros_like(labels), labels_rotated], 0)
    return all_img, all_labels, all_labels_rotated


def rotate_images(images, rot90_scalars=(0, 1, 2, 3)):
    """
    Return the input image and its 90, 180, and 270 degree rotations.
    images: NCHW
    """
    images_rotated = [
        images,  # 0 degree
        torch.rot90(images, 1, (2, 3)),
        torch.rot90(images, 2, (2, 3)),
        torch.rot90(images, 3, (2, 3))
    ]
    results = torch.cat([images_rotated[i] for i in rot90_scalars], 0)
    results = torch.reshape(results, [-1] + list(images.shape)[1:])
    return results


def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index)

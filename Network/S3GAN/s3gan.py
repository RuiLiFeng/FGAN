"""
FTWS: This scripts functions as s3gan.
"""
import functools

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F

from Network.BigGAN import layers
from Network.BigGAN import BigGAN
from Network.S3GAN.s3_ops import *
from Network.VaeGAN import Invert

NUM_ROTATIONS = 4


class Generator(BigGAN.Generator):
    """
    FTWS: The default Generator, plays the role of decoder in the VAE arch.
    """
    def __init__(self, name=None, **kwargs):
        super(Generator, self).__init__(**kwargs)
        self.name = name if name is not None else "G"

    def forward(self, z, y):
        # If hierarchical, concatenate zs and ys
        if self.hier:
            zs = torch.split(z, self.z_chunk_size, 1)
            z = zs[0]
            ys = [torch.cat([y, item], 1) for item in zs[1:]]
        else:
            ys = [y] * len(self.blocks)

        # First linear layer
        z = self.linear(z)
        # Reshape
        z = z.view(z.size(0), -1, self.bottom_width, self.bottom_width)

        # Loop over blocks
        for index, blocklist in enumerate(self.blocks):
            # Second inner loop in case block has multiple layers
            for block in blocklist:
                z = block(z, ys[index])

        # Apply batchnorm-relu-conv-tanh at output
        del ys, y
        return torch.tanh(self.output_layer(z))


class Discriminator(BigGAN.Discriminator):
    """
    FTWS: The discriminator to the Generator, which helps the training of it.
    """
    def __init__(self, self_supervision='rotation', project_y=True,
                 use_soft_pred=False, use_predictor=True, name=None, **kwargs):
        super(Discriminator, self).__init__(**kwargs)
        assert self_supervision in ['none', 'rotation']
        self.self_supervision = self_supervision
        self.project_y = project_y
        self.use_predictor = use_predictor
        self.use_soft_pred = use_soft_pred
        if 'rotation' in self.self_supervision:
            self.rotation_predictor = self.which_linear(self.arch['out_channels'][-1], NUM_ROTATIONS)
        if self.project_y:
            if self.use_predictor:
                self.class_predictor = self.which_linear(self.arch['out_channels'][-1], self.n_classes),
        self.name = name if name is not None else "D"

    def forward(self, x, y=None):
        # Stick x into h for cleaner for loops without flow control
        h = x
        # Loop over blocks
        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                h = block(h)
        # Apply global sum pooling as in SN-GAN
        h = torch.sum(self.activation(h), [2, 3])
        # Get initial class-unconditional output
        out = self.linear(h)
        if 'rotation' in self.self_supervision:
            rotation_logits = self.rotation_predictor(h)
        else:
            rotation_logits = None
        if self.project_y:
            if self.use_predictor:
                aux_logits = self.class_predictor(h)
                with torch.no_grad():
                    if self.use_soft_pred:
                        y_predicted = torch.softmax(aux_logits, 1)
                    else:
                        y_predicted = F.one_hot(torch.argmax(aux_logits, 1), self.n_classes)
                    is_label_available = (torch.sum(y, 1, keepdim=True, dtype=torch.float32) > 0.5).type(torch.float32)
                    y = (1.0 - is_label_available) * y_predicted + is_label_available * y
            else:
                aux_logits = None
            y = self.embed(y)
        else:
            y = 0
        # Get projection of final featureset onto class vectors and add to evidence
        out = out + torch.sum(y * h, 1, keepdim=True)
        del h
        return out, rotation_logits, aux_logits
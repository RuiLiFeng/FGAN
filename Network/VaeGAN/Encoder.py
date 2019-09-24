import numpy as np
import math
import functools

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P
from torchvision.models.resnet import Bottleneck, ResNet
from Network.BigGAN import layers


class Encoder_inv(ResNet):
    """
    FTWS: The encoder which take real pictures and return its corresponding latent code, through a typical ResNet50 arch
    without the final linear layer.
    In further development, this Network will be adopted to the prop of labeled imgs and automatically reform its dept
    -hs.
    """
    def __init__(self, dim_z, skip_init=False, E_lr=2e-4, E_init='ortho',
                 adam_eps=1e-8, E_B1=0.0, E_B2=0.999, E_mixed_precision=False, name=None):
        super(Encoder_inv, self).__init__(Bottleneck, [3, 4, 6, 4])
        self.downsample = nn.Sequential(
            nn.Conv2d(512 * 4, dim_z, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(dim_z))
        self.out_layer = nn.Sequential(
            nn.Linear(dim_z, dim_z),
            nn.ReLU(inplace=True)
        )
        if not skip_init:
            self.init_weights()
        self.init = E_init
        # Set name used for save and load weights
        self.name = name if name is not None else "Encoder"

        # Set up optimizer
        self.lr, self.B1, self.B2, self.adam_eps = E_lr, E_B1, E_B2, adam_eps
        if E_mixed_precision:
            print('Using fp16 adam in D...')
            from Utils import utils
            self.optim = utils.Adam16(params=self.parameters(), lr=self.lr,
                                      betas=(self.B1, self.B2), weight_decay=0, eps=self.adam_eps)
        else:
            self.optim = optim.Adam(params=self.parameters(), lr=self.lr,
                                    betas=(self.B1, self.B2), weight_decay=0, eps=self.adam_eps)

    def init_weights(self):
        self.param_count = 0
        for module in self.modules():
            if (isinstance(module, nn.Conv2d)
                    or isinstance(module, nn.Linear)
                    or isinstance(module, nn.Embedding)):
                if self.init == 'ortho':
                    init.orthogonal_(module.weight)
                elif self.init == 'N02':
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ['glorot', 'xavier']:
                    init.xavier_uniform_(module.weight)
                else:
                    print('Init style not recognized...')
                self.param_count += sum([p.data.nelement() for p in module.parameters()])
        print('Param count for Encoder''s initialized parameters: %d' % self.param_count)

    def forward(self, x):
        """
        :param x: [NCHW]
        :return:
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = self.downsample(x)
        x = torch.flatten(x, 1)
        x = self.out_layer(x)
        return x


class Encoder_rd(ResNet):
    def __init__(self, dim_z, skip_init=False, E_lr=2e-4,
                 adam_eps=1e-8, E_B1=0.0, E_B2=0.999, E_mixed_precision=False, name=None):
        super(Encoder_rd, self).__init__(Bottleneck, [3, 4, 6, 4])
        self.resblock1 = ResBlock(512 * 4, dim_z * 2)
        self.resblock2 = ResBlock(dim_z * 2, dim_z * 2)
        if not skip_init:
            self.init_weights()
        # Set name for saving and loading weights
        self.name = name if name is not None else "Encoder"

        # Set up optimizer
        self.lr, self.B1, self.B2, self.adam_eps = E_lr, E_B1, E_B2, adam_eps
        if E_mixed_precision:
            print('Using fp16 adam in D...')
            from Utils import utils
            self.optim = utils.Adam16(params=self.parameters(), lr=self.lr,
                                      betas=(self.B1, self.B2), weight_decay=0, eps=self.adam_eps)
        else:
            self.optim = optim.Adam(params=self.parameters(), lr=self.lr,
                                    betas=(self.B1, self.B2), weight_decay=0, eps=self.adam_eps)

    def init_weights(self):
        self.param_count = 0
        for module in self.modules():
            if (isinstance(module, nn.Conv2d)
                    or isinstance(module, nn.Linear)
                    or isinstance(module, nn.Embedding)):
                if self.init == 'ortho':
                    init.orthogonal_(module.weight)
                elif self.init == 'N02':
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ['glorot', 'xavier']:
                    init.xavier_uniform_(module.weight)
                else:
                    print('Init style not recognized...')
                self.param_count += sum([p.data.nelement() for p in module.parameters()])
        print('Param count for Encoder''s initialized parameters: %d' % self.param_count)

    def forward(self, x):
        """
        :param x: [NCHW]
        :return:
        """
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)  # [batch_size, 2048, 1, 1]

        x = self.resblock1(x)
        x = self.resblock2(x)  # [batch_size, dim_z *2]
        mean, var = torch.split(x, x.shape[1] // 2, 1)
        epsilon = torch.rand(mean.shape)
        x = var * epsilon + mean
        return x


class ResBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(ResBlock, self).__init__()
        if inplanes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, planes, 1),
                nn.BatchNorm2d(planes))
        else:
            self.shortcut = nn.Identity()
        self.res_layer = nn.Sequential(
            nn.Conv2d(inplanes, planes, 1),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, 1),
            nn.BatchNorm2d(planes)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.res_layer(x)
        x += identity
        x = self.relu(x)
        return x


def Encoder(arch='ResBlock_inv', **kwargs):
    encoder_dict = {"ResBlock_inv": Encoder_inv, "ResBlock_rand": Encoder_rd}
    assert arch in encoder_dict.keys()
    return encoder_dict[arch](**kwargs)

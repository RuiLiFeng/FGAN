import numpy as np
import math
import functools

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P

from Network.BigGAN import layers


class Dense(nn.Linear):
    def __init__(self, in_channels, hidden,
                 gain=np.sqrt(2), use_wscale=True, mul_lrmul=0.01, bias_lrmul=0.01):
        """
        :param in_channels: feature shapes, batch_size is not contained, must be int.
        :param hidden: Dimension of hidden layers.
        """
        self._shape = [in_channels, hidden]
        self._gain = gain
        self._use_wscale = use_wscale
        self._mul_lrmul = mul_lrmul
        self._bias_lrmul = bias_lrmul
        super(Dense, self).__init__(in_channels, hidden, bias=True)

    def reset_parameters(self):
        fan_in = self._shape[0]
        he_std = self._gain / np.sqrt(fan_in)  # He init
        # Equalized learning rate and custom learning rate multiplier.
        if self._use_wscale:
            init_std = 1.0 / self._mul_lrmul
            self._runtime_coef = he_std * self._mul_lrmul
        else:
            init_std = he_std / self._mul_lrmul
            self._runtime_coef = self._mul_lrmul
        nn.init.normal_(self.weight, 0, init_std)
        nn.init.zeros_(self.bias)

    def forward(self, input):
        return F.linear(input, self.weight * self._runtime_coef, self.bias * self._bias_lrmul)


class phi(nn.Module):
    def __init__(self, in_channels, out_channels=None, alpha=0.2, **kwargs):
        super(phi, self).__init__()
        self._in_channels = in_channels
        self._out_channels = in_channels if out_channels is None else out_channels
        self.dense1 = Dense(in_channels, in_channels, **kwargs)
        self.act1 = torch.nn.LeakyReLU(alpha)
        self.dense2 = Dense(in_channels, self._out_channels, **kwargs)
        self.act2 = torch.nn.LeakyReLU(alpha)

    def forward(self, x):
        x = self.dense1(x)
        x = self.act1(x)
        x = self.dense2(x)
        x = self.act2(x)
        return x


class step(nn.Module):
    def __init__(self, in_channels, is_reverse=True, **kwargs):
        super(step, self).__init__()
        self._in_channels = in_channels
        self._is_reverse = is_reverse
        self.phi = phi(in_channels // 2, in_channels // 2, **kwargs)

    def _reverse(self, x, axis=1):
        indices = [slice(None)] * x.dim()
        indices[axis] = torch.arange(x.size(axis) - 1, -1, -1,
                                     dtype=torch.long, device=x.device)
        return x[tuple(indices)]

    def forward(self, x):
        assert len(x.shape) == 2
        assert x.shape[1] == self._in_channels
        x = self._reverse(x)
        # mid = x.shape[1] // 2
        x1 = x[:, :x.shape[1] // 2]
        x2 = x[:, x.shape[1] // 2:]
        x2 = torch.add(x2, self.phi(x1))
        x = torch.cat((x1, x2), dim=1)
        return x


class Invert(nn.Module):
    def __init__(self, dim_z, I_depth, is_reverse=True, I_lr=2e-4,
                 adam_eps=1e-8, I_B1=0.0, I_B2=0.999, I_mixed_precision=False, name=None, **kwargs):
        """
        Invertible network.
        :param dim_z: Must be int.
        :param depth: How many coupling layer are used.
        :param is_reverse:
        """
        super(Invert, self).__init__()
        self._z_dim = dim_z
        self._depth = I_depth
        self._is_reverse = is_reverse
        self.steps = []
        self.steps = nn.ModuleList([step(self._z_dim, is_reverse) for _ in range(self._depth)])
        # Set name, used for load and save weights
        self.name = name if name is not None else 'Invert'

        # Set up optimizer
        self.lr, self.B1, self.B2, self.adam_eps = I_lr, I_B1, I_B2, adam_eps
        if I_mixed_precision:
            print('Using fp16 adam in D...')
            from Utils import utils
            self.optim = utils.Adam16(params=self.parameters(), lr=self.lr,
                                      betas=(self.B1, self.B2), weight_decay=0, eps=self.adam_eps)
        else:
            self.optim = optim.Adam(params=self.parameters(), lr=self.lr,
                                    betas=(self.B1, self.B2), weight_decay=0, eps=self.adam_eps)

    def forward(self, x):
        for stp in self.steps:
            x = stp(x)
        return x


"""
FTWS: This scripts functions as the decoders in vae.
"""
import functools

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F

from Network.BigGAN import layers
from Network.BigGAN import BigGAN
from Network.VaeGAN import Invert
from Network.VaeGAN import Encoder


class Generator(BigGAN.Generator):
    """
    FTWS: The default Generator, plays the role of decoder in the VAE arch.
    """
    def __init__(self, name=None, **kwargs):
        super(Generator, self).__init__(**kwargs)
        self.name = name if name is not None else "G"


class Discriminator(BigGAN.Discriminator):
    """
    FTWS: The discriminator to the Generator, which helps the training of it.
    """
    def __init__(self, name=None, **kwargs):
        super(Discriminator, self).__init__(**kwargs)
        self.name = name if name is not None else "D"

    def forward(self, x, y=None):
        """
        FTWS: Unconditional discriminator.
        :param x:
        :param y:
        :return:
        """
        if y is not None:
            raise ValueError("This is the unconditional discriminator, where y must be none, got {}!".format(y))
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
        # Get projection of final featureset onto class vectors and add to evidence

        # Get y batch index
        # UNLABEL = -1
        # y_index = torch.nonzero(y == UNLABEL)
        # y = y[y_index]
        # out = out + torch.sum(self.embed(y) * h, 1, keepdim=True)
        return out


class LatentBinder(nn.Module):
    """
    FTWS: This class plays the role of binder for latent space induced by Invert Network and Encoder Network,
    through minimizing the Warsserstein Loss between the distribution of Invert(Z) and Encoder(X), where Z~N(0,1),
    X~Read Data distribution. It will process the latent code: v_inv and v_en produced by Invert Net and Encoder Net,
    through depth layers ResBlock arch, and return logits logits_inv, logits_en to the loss contributor.
    """
    def __init__(self, dim_z=120, L_depth=4, skip_init=False, L_init='ortho', L_lr=2e-4,
                 adam_eps=1e-8, L_B1=0.0, L_B2=0.999, L_mixed_precision=False, name=None, **kwargs):
        super(LatentBinder, self).__init__()
        self.layers = torch.nn.ModuleList([ResBlock() for _ in range(L_depth)])
        self.out = torch.nn.Sequential(torch.nn.Linear(dim_z, 1),
                                       torch.nn.ReLU(inplace=True))
        self.init = L_init
        if not skip_init:
            self.init_weights()
        self.name = name if name is not None else "LatentBinder"
        # Set up optimizer
        self.lr, self.B1, self.B2, self.adam_eps = L_lr, L_B1, L_B2, adam_eps
        if L_mixed_precision:
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
            if (isinstance(module, nn.Conv1d)
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
        print('Param count for D''s initialized parameters: %d' % self.param_count)

    def forward(self, z):
            z = z.reshape([z.shape[0], 1, z.shape[1]])
            for layer in self.layers:
                z = layer(z)
            z = z.reshape([z.shape[0], -1])
            out = self.out(z)
            return out


class ResBlock(nn.Module):
    """
    FTWS: This block is the base block of LatentBinder, it takes vectors with any dimensions, and process it with two
    fully-connected layers with residual link.
    Notice that, the input latents should have shape [batch_size, 1, latent_dimension]
    """
    def __init__(self, inplanes=1, planes=1):
        super(ResBlock, self).__init__()
        if inplanes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(inplanes, planes, 1),
                nn.BatchNorm1d(planes))
        else:
            self.shortcut = nn.Identity()
        self.res_layer = nn.Sequential(
            nn.Conv1d(inplanes, planes, 1),
            nn.BatchNorm1d(planes),
            nn.ReLU(inplace=True),
            nn.Conv1d(planes, planes, 1),
            nn.BatchNorm1d(planes)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.res_layer(x)
        x += identity
        x = self.relu(x)
        return x


class Decoder(nn.Module):
    """
    FTWS: This class handles the training of Decoder, take z, gy, x, dy (could be None) as inputs and produce logits for
    training.
    It will also offer API for accessing G, D, L components.
    There are four types of Decoder's training pipeline:
    1. View discriminator as to work for the joint distribution of latent and imgs
    2. View generator and discriminator as a new discriminator to distinguish between Invert(Z) and Encoder(X)
    3. Add a regularize term to the final loss to bond Invert(Z) and Encoder(X)
    4. Add a individual discriminator to distinguish Invert(Z) and Encoder(X)
    """
    def __init__(self, I, E, G, D, L, name=None, **kwargs):
        super(Decoder, self).__init__()
        self.Invert = I
        self.Encoder = E
        self.G = G
        self.D = D
        self.LatentBinder = L
        self.name = name if name is not None else "Decoder"

    def forward(self, z, iy, x, ey, train_G=False, split_D=False):
        """
        FTWS: This is the implementation of the 4-th case. In this mode, encoder, invert are seen as part of generator.
        Notice that there is no need for real labels, and the input to D must contain no labels.
        :param z: Input latent
        :param iy: Fake labels for inverter
        :param x: Real images [NCHW]
        :param ey: Fake labels for encoder
        :param train_G:
        :param split_D:
        :return: D_fake, D_real, D_inv, D_en, G_en, x
        """
        with torch.set_grad_enabled(train_G):   # If false all variable computed under the context will not have grad
            # enabled.
            v_inv = self.Invert(z)
            v_en = self.Encoder(x)
            G_inv = self.G(v_inv, self.G.shared(iy))
            G_en = self.G(v_en, self.G.shared(ey))
            # Cast if necessary
            if self.G.fp16 and not self.D.fp16:
                G_inv = G_inv.float()
                G_en = G_en.float()
            if self.D.fp16 and not self.G.fp16:
                G_inv = G_inv.half()
                G_en = G_en.half()
        # Split_D means to run D once with real data and once with fake,
        # rather than concatenating along the batch dimension.
        if split_D:
            D_fake = self.D(G_inv)
            D_real = self.D(x)
            D_inv = self.LatentBinder(v_inv)
            D_en = self.LatentBinder(v_en)
            del G_en, G_inv, v_en, v_inv
            return D_fake, D_real, D_inv, D_en, G_en, x
        else:
            D_input = torch.cat([G_inv, x], 0)
            v_input = torch.cat([v_inv, v_en], 0)
            # Get Discriminator output
            D_out = self.D(D_input)
            Dv_out = self.LatentBinder(v_input)
            D_fake, D_real = torch.split(D_out, [G_inv.shape[0], x.shape[0]])
            D_inv, D_en = torch.split(Dv_out, [v_inv.shape[0], v_en.shape[0]])
            del G_en, G_inv, v_en, v_inv, D_input, v_input, D_out, Dv_out
            
            return D_fake, D_real, D_inv, D_en, G_en, x
            # D_fake, D_real, D_inv, D_en, G_en, x

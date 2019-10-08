from Network.BigGAN import BigGAN
import torch
from Network.VaeGAN import Invert


class Generator(BigGAN.Generator):
    def __init__(self, **kwargs):
        super(Generator, self).__init__(**kwargs)
        self.Invert = Invert.Invert(**kwargs)
        self.name = 'G'

    def forward(self, z, y):
        z = self.Invert(z)
        net = super(Generator, self).forward(z, y)
        return net


class Discriminator(BigGAN.Discriminator):
    def __init__(self, **kwargs):
        super(Discriminator, self).__init__(**kwargs)
        self.name = 'D'

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
        # Get projection of final featureset onto class vectors and add to evidence

        # Get y batch index
        # UNLABEL = -1
        # y_index = torch.nonzero(y == UNLABEL)
        # y = y[y_index]
        out = out + torch.sum(self.embed(y) * h, 1, keepdim=True)
        return out


class G_D(BigGAN.G_D):
    def __init__(self, G, D):
        super(G_D, self).__init__(G, D)

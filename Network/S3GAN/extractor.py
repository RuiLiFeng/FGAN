import functools

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from Network.VaeGAN.ResNet import ResNet, Bottleneck
from Network.S3GAN.s3_ops import merge_with_rotation_data
from Utils import vae_utils, utils
from Network.BigGAN import layers

NUM_ROTATIONS = 4


class Extractor(nn.Module):
    def __init__(self, n_class=1000, Ex_init='ortho', name=None, Ex_lr=2e-4, adam_eps=1e-8,
                 Ex_B1=0.0, Ex_B2=0.999, skip_init=False, **kwargs):
        super(Extractor, self).__init__()
        self.ResNet = ResNet(Bottleneck, [3, 4, 6, 3], width_per_group=64*2)
        del self.ResNet.fc
        self.init = Ex_init
        self.c_r = nn.Linear(512*4, NUM_ROTATIONS)
        self.s2l = nn.Linear(512*4, n_class)
        self.name = 'Extractor' if name is None else name

        if not skip_init:
            self.init_weights()

        self.lr, self.B1, self.B2, self.adam_eps = Ex_lr, Ex_B1, Ex_B2, adam_eps
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
        print('Param count for Extractor''s initialized parameters: %d' % self.param_count)

    def forward(self, x, y):
        img_num = x.shape[0]
        x, y, ry = merge_with_rotation_data(x, y, img_num)
        x = self.ResNet.conv1(x)
        x = self.ResNet.bn1(x)
        x = self.ResNet.relu(x)
        x = self.ResNet.maxpool(x)

        x = self.ResNet.layer1(x)
        x = self.ResNet.layer2(x)
        x = self.ResNet.layer3(x)
        x = self.ResNet.layer4(x)

        x = self.ResNet.avgpool(x)
        x = torch.flatten(x, 1)
        rot_logits = self.c_r(x)
        s2l_logits = self.s2l(x)
        return rot_logits, s2l_logits, y, ry


def extractor_loss(rot_logits, s2l_logits, y, ry):
    n_classes = s2l_logits.shape[1]
    rot_prop = torch.nn.functional.log_softmax(rot_logits, 1)
    s2l_prop = torch.nn.functional.log_softmax(s2l_logits, 1)
    ry_one_hot = torch.nn.functional.one_hot(ry, NUM_ROTATIONS).type_as(rot_prop)
    # y_one_hot[i]=[0,....,0] if the i-th element in this batch
    # has no label, that is, y[i] = -1
    y_one_hot = vae_utils.general_one_hot(y, n_classes).type_as(s2l_prop)

    rot_loss = torch.sum(rot_prop * ry_one_hot / NUM_ROTATIONS)
    s2l_loss = torch.sum(s2l_prop * y_one_hot / NUM_ROTATIONS)
    return rot_loss, s2l_loss


def Extractor_training_function(Ex, ema, Ex_parallel, state_dict, config):
    def train(x, y):
        Ex.optim.zero_grad()
        rot_logits, s2l_logits, y, ry = Ex_parallel(x, y)
        rot_loss, s2l_loss = extractor_loss(rot_logits, s2l_logits, y, ry)
        loss = rot_loss + 0.5 * s2l_loss
        loss.backward()
        if config['Ex_ortho'] > 0.0:
            print('using modified ortho reg in Extractor')  # Debug print to indicate we're using ortho reg in G
            # Don't ortho reg shared, it makes no sense. Really we should blacklist any embeddings for this
            utils.ortho(Ex, config['G_ortho'])
        Ex.optim.step()

        # If we have an ema, update it, regardless of if we test with it or not
        if config['ema']:
            ema.update(state_dict['itr'])

        out = {'G_loss': float(loss.item()),
               'D_loss_real': float(rot_loss.item()),
               'D_loss_fake': float(s2l_loss.item())}
        # Return G's loss and the components of D's loss.
        return out

    return train

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.modules.loss as loss


# DCGAN loss
def loss_dcgan_dis(dis_fake, dis_real):
    L1 = torch.mean(F.softplus(-dis_real))
    L2 = torch.mean(F.softplus(dis_fake))
    return L1, L2


def loss_dcgan_gen(dis_fake):
    loss = torch.mean(F.softplus(-dis_fake))
    return loss


# Hinge Loss
def loss_hinge_dis(dis_fake, dis_real):
    loss_real = torch.mean(F.relu(1. - dis_real))
    loss_fake = torch.mean(F.relu(1. + dis_fake))
    return loss_real, loss_fake
    # def loss_hinge_dis(dis_fake, dis_real): # This version returns a single loss
    # loss = torch.mean(F.relu(1. - dis_real))
    # loss += torch.mean(F.relu(1. + dis_fake))
    # return loss


def loss_hinge_gen(dis_fake):
    loss = -torch.mean(dis_fake)
    return loss


def loss_hinge_recon(fakes, reals, vgg, r_loss_scale=0.001):
    pixel_err = torch.mean(F.mse_loss(fakes, reals, reduce=False), [index for index in range(1, len(fakes.shape[1:]))])
    fakes_vgg = vgg(fakes)
    reals_vgg = vgg(reals)
    vgg_err = torch.mean(F.mse_loss(fakes_vgg, reals_vgg, reduce=False), [index for index in
                                                                          range(1, len(fakes_vgg.shape[1:]))])
    loss_vgg = torch.mean(4.0 - F.relu(4.0 - vgg_err))
    loss_pixel = torch.mean(4.0 - F.relu(4.0 - pixel_err))
    return loss_pixel + r_loss_scale * loss_vgg


def loss_hinge_latent_dis(inv, en):
    loss_inv = torch.mean(F.relu(1. - inv))
    loss_en = torch.mean(F.relu(1. + en))
    return loss_inv + loss_en


def loss_hinge_latent_gen(inv, en):
    loss_inv = torch.mean(F.relu(1. + inv))
    loss_en = torch.mean(F.relu(1. - en))
    return loss_inv + loss_en


class ParallelLoss(loss._Loss):
    def __init__(self, vgg, config, size_average=None, reduce=None, reduction='mean'):
        super(ParallelLoss, self).__init__(size_average, reduce, reduction)
        self.vgg = vgg
        self.adv_loss_scale = config['adv_loss_scale']
        self.recon_loss_scale = config['recon_loss_scale']
        self.num_G_accumulations = config['num_G_accumulations']
        self.num_D_accumulations = config['num_D_accumulations']

    def forward(self, input_tuple, training_G=True):
        """

        :param input_tuple: D_fake, D_real, D_inv, D_en, G_en, reals
        :param training_G:
        :return:
        """
        D_fake, D_real, D_inv, D_en, G_en, reals = input_tuple
        if training_G:
            G_loss_fake = generator_loss(D_fake) * self.adv_loss_scale
            Latent_loss = latent_loss_gen(D_inv, D_en)
            Recon_loss = recon_loss(G_en, reals, self.vgg, self.recon_loss_scale)
            G_loss = (G_loss_fake + Latent_loss + Recon_loss) / float(self.num_G_accumulations)

            out_dict = {'Recon_loss': float(Recon_loss.item()),
                        'G_loss_fake': float(G_loss_fake.item()),
                        'Latent_loss': float(Latent_loss.item())}
            return G_loss, out_dict
        else:
            D_loss_real, D_loss_fake = discriminator_loss(D_fake, D_real)
            Latent_loss = latent_loss_dis(D_inv, D_en)
            D_loss = (D_loss_real + D_loss_fake + Latent_loss) / float(
                self.num_D_accumulations)

            out_dict = {'D_loss_real': float(D_loss_real.item()),
                        'D_loss_fake': float(D_loss_fake.item()),
                        'Latent_loss': float(Latent_loss.item())}
            return D_loss, out_dict


# Default to hinge loss
generator_loss = loss_hinge_gen
discriminator_loss = loss_hinge_dis
latent_loss_dis = loss_hinge_latent_dis
latent_loss_gen = loss_hinge_latent_gen
recon_loss = loss_hinge_recon

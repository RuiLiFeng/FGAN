import torch
import torch.nn.functional as F
from Metric.vggutils import load_vgg_from_local


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


def loss_hinge_recon(fakes, reals, r_loss_scale=0.001):
    vgg = load_vgg_from_local()
    pixel_err = torch.norm(fakes - reals, 'fro', [index for index in range(len(fakes.shape)) if index > 0])
    fakes_vgg = vgg(fakes)
    reals_vgg = vgg(reals)
    vgg_err = torch.norm(fakes_vgg - reals_vgg, 'fro', [index for index in range(len(fakes.shape)) if index > 0])
    vgg_err = vgg_err / torch.prod(vgg_err.shape[1:]).type_as(vgg_err)
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


# Default to hinge loss
generator_loss = loss_hinge_gen
discriminator_loss = loss_hinge_dis
latent_loss_dis = loss_hinge_latent_dis
latent_loss_gen = loss_hinge_latent_gen
recon_loss = loss_hinge_recon

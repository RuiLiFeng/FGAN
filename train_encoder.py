""" BigGAN: The Authorized Unofficial PyTorch release
    Code by A. Brock and A. Andonian
    This code is an unofficial reimplementation of
    "Large-Scale GAN Training for High Fidelity Natural Image Synthesis,"
    by A. Brock, J. Donahue, and K. Simonyan (arXiv 1809.11096).

    Let's go.
"""

import functools
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import my stuff
from Metric import inception_utils
from Utils import utils, vae_utils
from Training import train_fns
from sync_batchnorm import patch_replication_callback
from Network.BigGAN import BigGAN
from Network.VaeGAN.Encoder import Encoder
from Network.VaeGAN import losses
import os, torchvision


# The main training file. Config is a dictionary specifying the configuration
# of this training run.

"""
The utils for training encoder.
"""


def load_pretrained(net, path):
    pretrained_dict = torch.load(path)
    net_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in net_dict}
    net_dict.update(pretrained_dict)
    net.load_state_dict(net_dict)


def save_and_sample(G, E, E_ema, fixed_x, fixed_y, state_dict, config, experiment_name):
    vae_utils.save_weights([E], state_dict, config['weights_root'],
                           experiment_name, None, [E_ema if config['ema'] else None])
    if config['num_save_copies'] > 0:
        vae_utils.save_weights([E], state_dict, config['weights_root'],
                               experiment_name,
                               'copy%d' % state_dict['save_num'],
                               [E_ema if config['ema'] else None])
        state_dict['save_num'] = (state_dict['save_num'] + 1) % config['num_save_copies']
        G_batch_size = max(config['G_batch_size'], config['batch_size'])
        z_, y_ = utils.prepare_z_y(G_batch_size, G.dim_z, config['n_classes'],
                          device='cuda', fp16=config['G_fp16'])
        utils.accumulate_standing_stats(G, z_, y_, config['n_classes'], config['num_standing_accumulations'])
        del z_, y_
        which_E = E_ema if config['ema'] and config['use_ema'] else E
        with torch.no_grad():
            if config['parallel']:
                fixed_w = nn.parallel.data_parallel(which_E, fixed_x)
                fixed_Gz = nn.parallel.data_parallel(G, (fixed_w, G.shared(fixed_y)))
            else:
                fixed_w = which_E(fixed_x)
                fixed_Gz = G(fixed_w, G.shared(fixed_y))
        if not os.path.isdir('%s/%s' % (config['samples_root'], experiment_name)):
            os.mkdir('%s/%s' % (config['samples_root'], experiment_name))
        image_filename = '%s/%s/fixed_samples%d.jpg' % (config['samples_root'],
                                                        experiment_name,
                                                        state_dict['itr'])
        torchvision.utils.save_image(fixed_Gz.float().cpu(), image_filename,
                                     nrow=int(fixed_Gz.shape[0] ** 0.5), normalize=True)


def run(config):
    # Update the config dict as necessary
    # This is for convenience, to add settings derived from the user-specified
    # configuration into the config-dict (e.g. inferring the number of classes
    # and size of the images from the dataset, passing in a pytorch object
    # for the activation specified as a string)
    config['resolution'] = utils.imsize_dict[config['dataset']]
    config['n_classes'] = utils.nclass_dict[config['dataset']]
    config['G_activation'] = utils.activation_dict[config['G_nl']]
    config['D_activation'] = utils.activation_dict[config['D_nl']]
    # By default, skip init if resuming training.
    if config['resume']:
        print('Skipping initialization for training resumption...')
        config['skip_init'] = True
    config = vae_utils.update_config_roots(config)
    device = 'cuda'

    # Seed RNG
    utils.seed_rng(config['seed'])

    # Prepare root folders if necessary
    utils.prepare_root(config)

    # Setup cudnn.benchmark for free speed
    torch.backends.cudnn.benchmark = True

    experiment_name = (config['experiment_name'] if config['experiment_name']
                       else utils.name_from_config(config))
    print('Experiment name is %s' % experiment_name)

    # Next, build the model
    G = BigGAN.Generator(**{**config, 'skip_init': True, 'no_optim': True}).to(device)
    D = BigGAN.Discriminator(**{**config, 'skip_init': True, 'no_optim': True}).to(device)
    E = Encoder(**config).to(device)
    vgg_alter = Encoder(**{**config, 'skip_init': True, 'no_optim': True, 'name': 'Vgg_alter'}).to(device)
    load_pretrained(G, config['pretrained_G_dir'])
    load_pretrained(D, config['pretrained_D_dir'])
    load_pretrained(vgg_alter, config['pretrained_vgg_alter_dir'])

    # If using EMA, prepare it
    if config['ema']:
        print('Preparing EMA for G with decay of {}'.format(config['ema_decay']))
        E_ema = Encoder(**{**config, 'skip_init': True,
                                   'no_optim': True}).to(device)
        ema = utils.ema(E, E_ema, config['ema_decay'], config['ema_start'])
    else:
        E_ema, ema = None, None

    class TrainWarpper(nn.Module):
        def __init__(self):
            super(TrainWarpper, self).__init__()
            self.G = G
            self.D = D
            self.E = E
            self.vgg_alter = vgg_alter

        def forward(self, img, label):
            en_w = self.E(img)
            with torch.no_grad():
                fake = self.G(en_w, self.G.shared(label))
                logits = self.D(fake, label)
                vgg_logits = F.l1_loss(self.vgg_alter(img), self.vgg_alter(fake))
            return fake, logits, vgg_logits

    Wrapper = TrainWarpper()
    print(G)
    print(D)
    print(E)
    print(vgg_alter)
    print('Number of params in G: {} D: {} E: {} Vgg_alter: {}'.format(
        *[sum([p.data.nelement() for p in net.parameters()]) for net in [G, D, E, vgg_alter]]))
    # Prepare state dict, which holds things like epoch # and itr #
    state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0,
                  'best_IS': 0, 'best_FID': 999999, 'config': config}

    # If loading from a pre-trained model, load weights
    if config['resume']:
        print('Loading weights...')
        vae_utils.load_weights([E], state_dict,
                               config['weights_root'], experiment_name,
                               config['load_weights'] if config['load_weights'] else None,
                               [E_ema if config['ema'] else None])

    # If parallel, parallelize the GD module
    if config['parallel']:
        Wrapper = nn.DataParallel(Wrapper)
        if config['cross_replica']:
            patch_replication_callback(Wrapper)

    # Prepare loggers for stats; metrics holds test metrics,
    # lmetrics holds any desired training metrics.
    test_metrics_fname = '%s/%s_log.jsonl' % (config['logs_root'],
                                              experiment_name)
    train_metrics_fname = '%s/%s' % (config['logs_root'], experiment_name)
    print('Inception Metrics will be saved to {}'.format(test_metrics_fname))
    test_log = utils.MetricsLogger(test_metrics_fname,
                                   reinitialize=(not config['resume']))
    print('Training Metrics will be saved to {}'.format(train_metrics_fname))
    train_log = utils.MyLogger(train_metrics_fname,
                               reinitialize=(not config['resume']),
                               logstyle=config['logstyle'])
    # Write metadata
    utils.write_metadata(config['logs_root'], experiment_name, config, state_dict)
    # Prepare data; the Discriminator's batch size is all that needs to be passed
    # to the dataloader, as G doesn't require dataloading.
    # Note that at every loader iteration we pass in enough data to complete
    # a full D iteration (regardless of number of D steps and accumulations)
    D_batch_size = (config['batch_size'] * config['num_D_steps']
                    * config['num_D_accumulations'])

    loaders = utils.get_data_loaders(**{**config, 'batch_size': D_batch_size,
                                        'start_itr': state_dict['itr']})
    G_batch_size = max(config['G_batch_size'], config['batch_size'])
    fixed_x, fixed_y = vae_utils.prepare_fixed_x(loaders[0], G_batch_size, config, experiment_name, device)

    # Prepare inception metrics: FID and IS
    get_inception_metrics = inception_utils.prepare_inception_metrics(config['dataset'],
                                                                      config['parallel'],
                                                                      config['data_root'],
                                                                      config['no_fid'])

    # Prepare noise and randomly sampled label arrays

    def train(img, label):
        E.optim.zero_grad()
        img = torch.split(img, config['batch_size'])
        label = torch.split(label, config['batch_size'])
        counter = 0

        for step_index in range(config['num_D_steps']):
            E.optim.zero_grad()
            fake, logits, vgg_loss = Wrapper(img[counter], label[counter])
            vgg_loss = vgg_loss * config['vgg_loss_scale']
            d_loss = losses.generator_loss(logits) * config['adv_loss_scale']
            recon_loss = losses.recon_loss(fakes=fake, reals=img)
            loss = d_loss + recon_loss + vgg_loss
            loss.backward()
            counter += 1
            if config['E_ortho'] > 0.0:
                # Debug print to indicate we're using ortho reg in D.
                print('using modified ortho reg in D')
                utils.ortho(D, config['E_ortho'])
            E.optim.step()

        out = {'Vgg_loss': float(vgg_loss.item()),
               'D_loss': float(d_loss.item()),
               'pixel_loss': float(recon_loss.item())}
        return out

    print('Beginning training at epoch %d...' % state_dict['epoch'])
    # Train for specified number of epochs, although we mostly track G iterations.
    for epoch in range(state_dict['epoch'], config['num_epochs']):
        # Which progressbar to use? TQDM or my own?
        if config['pbar'] == 'mine':
            pbar = utils.progress(loaders[0], displaytype='s1k' if config['use_multiepoch_sampler'] else 'eta')
        else:
            pbar = tqdm(loaders[0])
        for i, (x, y) in enumerate(pbar):
            # Increment the iteration counter
            state_dict['itr'] += 1
            # Make sure G and D are in training mode, just in case they got set to eval
            # For D, which typically doesn't have BN, this shouldn't matter much.
            G.train()
            D.train()
            E.train()
            vgg_alter.train()
            if config['ema']:
                E_ema.train()
            if config['D_fp16']:
                x, y = x.to(device).half(), y.to(device)
            else:
                x, y = x.to(device), y.to(device)
            metrics = train(x, y)
            train_log.log(itr=int(state_dict['itr']), **metrics)

            # Every sv_log_interval, log singular values
            if (config['sv_log_interval'] > 0) and (not (state_dict['itr'] % config['sv_log_interval'])):
                train_log.log(itr=int(state_dict['itr']),
                              **{**utils.get_SVs(E, 'E')})

            # If using my progbar, print metrics.
            if config['pbar'] == 'mine':
                print(', '.join(['itr: %d' % state_dict['itr']]
                                + ['%s : %+4.3f' % (key, metrics[key])
                                   for key in metrics]), end=' ')

            # Save weights and copies as configured at specified interval
            if not (state_dict['itr'] % config['save_every']):
                if config['G_eval_mode']:
                    print('Switchin G to eval mode...')
                    G.eval()
                    E.eval()
                    if config['ema']:
                        E_ema.eval()
                save_and_sample(G, E, E_ema, fixed_x, fixed_y,
                                state_dict, config, experiment_name)
        # Increment epoch counter at end of epoch
        state_dict['epoch'] += 1


def main():
    # parse command line and run
    parser = utils.prepare_parser()
    config = vars(parser.parse_args())
    print(config)
    run(config)


if __name__ == '__main__':
    main()
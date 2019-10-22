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
from importlib import import_module
from Network.VaeGAN.Encoder import Encoder
from Network.BigGAN.BigGAN import Generator


# The main training file. Config is a dictionary specifying the configuration
# of this training run.
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

    # Import the model--this line allows us to dynamically select different files.
    experiment_name = (config['experiment_name'] if config['experiment_name']
                       else utils.name_from_config(config))
    print('Experiment name is %s' % experiment_name)

    # Next, build the model
    G = Generator(**config).to(device)
    print('Loading pretrained G for dir %s ...' % config['pretrained_G_dir'])
    pretrained_dict = torch.load(config['pretrained_G_dir'])
    G_dict = G.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in G_dict}
    G_dict.update(pretrained_dict)
    G.load_state_dict(G_dict)

    E = Encoder(**config).to(device)
    utils.toggle_grad(G, False)
    utils.toggle_grad(E, True)

    class G_E(nn.Module):
        def __init__(self):
            super(G_E, self).__init__()
            self.G = G
            self.E = E

        def forward(self, w, y):
            with torch.no_grad():
                net = self.G(w, self.G.shared(y))
            net = self.E(net)
            return net

    GE = G_E()

    # If using EMA, prepare it
    if config['ema']:
        print('Preparing EMA for E with decay of {}'.format(config['ema_decay']))
        E_ema = Encoder(**{**config, 'skip_init': True,
                           'no_optim': True}).to(device)
        e_ema = utils.ema(E, E_ema, config['ema_decay'], config['ema_start'])
    else:
        E_ema, e_ema = None, None

    print(G)
    print(E)
    print('Number of params in G: {} E: {}'.format(
        *[sum([p.data.nelement() for p in net.parameters()]) for net in [G, E]]))
    # Prepare state dict, which holds things like epoch # and itr #
    state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0,
                  'best_IS': 0, 'best_FID': 999999, 'config': config}

    # If loading from a pre-trained model, load weights
    if config['resume']:
        print('Loading weights...')
        vae_utils.load_weights([E], state_dict,
                               config['weights_root'], experiment_name,
                               config['load_weights'] if config['load_weights'] else None,
                               [e_ema] if config['ema'] else None)

    # If parallel, parallelize the GD module
    if config['parallel']:
        GE = nn.DataParallel(GE)
        if config['cross_replica']:
            patch_replication_callback(GE)

    # Prepare loggers for stats; metrics holds test metrics,
    # lmetrics holds any desired training metrics.
    train_metrics_fname = '%s/%s' % (config['logs_root'], experiment_name)
    print('Training Metrics will be saved to {}'.format(train_metrics_fname))
    train_log = utils.MyLogger(train_metrics_fname,
                               reinitialize=(not config['resume']),
                               logstyle=config['logstyle'])
    # Write metadata
    utils.write_metadata(config['logs_root'], experiment_name, config, state_dict)

    G_batch_size = max(config['G_batch_size'], config['batch_size'])
    z_, y_ = utils.prepare_z_y(G_batch_size, G.dim_z, config['n_classes'],
                               device=device, fp16=config['G_fp16'])

    def train():
        E.optim.zero_grad()
        z_.sample_()
        y_.sample_()

        net = GE(z_[:config['batch_size']], y_[:config['batch_size']])
        loss = F.l1_loss(z_[:config['batch_size']], net)
        loss.backward()
        if config["E_ortho"] > 0.0:
            print('using modified ortho reg in E')
            utils.ortho(E, config['E_ortho'])
        E.optim.step()
        out = {'loss': float(loss.item())}
        return out

    print('Beginning training at epoch %d...' % state_dict['epoch'])
    # Train for specified number of epochs, although we mostly track G iterations.
    for epoch in range(state_dict['epoch'], config['num_epochs']):
        for i in range(100000):
            # Increment the iteration counter
            state_dict['itr'] += 1
            # Make sure G and D are in training mode, just in case they got set to eval
            # For D, which typically doesn't have BN, this shouldn't matter much.
            G.train()
            E.train()
            if config['ema']:
                E_ema.train()
            metrics = train()
            train_log.log(itr=int(state_dict['itr']), **metrics)

            # Every sv_log_interval, log singular values
            if (config['sv_log_interval'] > 0) and (not (state_dict['itr'] % config['sv_log_interval'])):
                train_log.log(itr=int(state_dict['itr']),
                              **{**utils.get_SVs(G, 'G'), **utils.get_SVs(E, 'E')})

            # If using my progbar, print metrics.
            if config['pbar'] == 'mine':
                print(', '.join(['itr: %d' % state_dict['itr']]
                                + ['%s : %+4.3f' % (key, metrics[key])
                                   for key in metrics]), end=' ')

            # Save weights and copies as configured at specified interval
            if not (state_dict['itr'] % config['save_every']):
                vae_utils.save_weights([E], state_dict, config['weights_root'],
                                       experiment_name, 'copy%d' % state_dict['save_num'],
                                       E_ema if config['ema'] else None)
                state_dict['save_num'] = (state_dict['save_num'] + 1) % config['num_save_copies']
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
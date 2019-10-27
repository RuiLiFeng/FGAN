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
from sync_batchnorm import patch_replication_callback
from Network.VaeGAN.Encoder import Encoder
from Dataset import sampled_ssgan


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
  config = utils.update_config_roots(config)
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
  E = Encoder(**{**config, 'no_optim': True}).to(device)

  # If using EMA, prepare it
  if config['ema']:
    print('Preparing EMA for G with decay of {}'.format(config['ema_decay']))
    E_ema = Encoder(**{**config, 'skip_init':True,
                       'no_optim': True}).to(device)
    ema = utils.ema(E, E_ema, config['ema_decay'], config['ema_start'])
  else:
    E_ema, ema = None, None

  print(E)
  print('Number of params in E: {}'.format(
    *[sum([p.data.nelement() for p in net.parameters()]) for net in [E]]))
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
    E = nn.DataParallel(E)
    save_and_eval_E = E.module
    if config['cross_replica']:
      patch_replication_callback(E)
  else:
    save_and_eval_E = E

  # Prepare loggers for stats; metrics holds test metrics,
  # lmetrics holds any desired training metrics.
  test_metrics_fname = '%s/%s_log.jsonl' % (config['logs_root'],
                                            experiment_name)
  train_metrics_fname = '%s/%s' % (config['logs_root'], experiment_name)
  print('Inception Metrics will be saved to {}'.format(test_metrics_fname))
  test_log = utils.MetricsLogger(test_metrics_fname,
                                 reinitialize=(not config['resume']))
  KNN = None
  print('Training Metrics will be saved to {}'.format(train_metrics_fname))
  train_log = utils.MyLogger(train_metrics_fname,
                             reinitialize=(not config['resume']),
                             logstyle=config['logstyle'])
  # Write metadata
  utils.write_metadata(config['logs_root'], experiment_name, config, state_dict)

  E_optim = torch.optim.Adam(params=E.parameters(), lr=config['E_lr'],
                             betas=(config['E_B1'], config['E_B2']), weight_decay=0, eps=config['adam_eps'])

  def train(w, img):
    E_optim.zero_grad()
    w_ = E(img)
    loss = F.mse_loss(w_, w, reduction='sum')
    loss.backward()
    if config['E_ortho'] > 0.0:
      # Debug print to indicate we're using ortho reg in D.
      print('using modified ortho reg in E')
      utils.ortho(save_and_eval_E, config['E_ortho'])
    E_optim.step()
    out = {'loss': loss}
    return out

  start, end = sampled_ssgan.make_dset_range(config['ssgan_sample_root'], config['ssgan_piece'])
  print('Beginning training at epoch %d...' % state_dict['epoch'])
  # Train for specified number of epochs, although we mostly track G iterations.
  for epoch in range(state_dict['epoch'], config['num_epochs']):
    for piece in range(config['ssgan_piece']):
      print('Load %d-th piece of ssgan sample into memory...' % piece)
      loader = sampled_ssgan.get_SSGAN_sample_loader(**{**config, 'start_itr': state_dict['itr'],
                                                        'start': start[piece], 'end': end[piece]})
      # Which progressbar to use? TQDM or my own?
      if config['pbar'] == 'mine':
        pbar = utils.progress(loader, displaytype='eta')
      else:
        pbar = tqdm(loader)
      for i, (img, z, w) in enumerate(pbar):
        # Increment the iteration counter
        state_dict['itr'] += 1
        # Make sure G and D are in training mode, just in case they got set to eval
        # For D, which typically doesn't have BN, this shouldn't matter much.
        E.train()
        if config['ema']:
          E_ema.train()
        img, w = img.to(device), w.to(device)
        metrics = train(img, w)
        train_log.log(itr=int(state_dict['itr']), **metrics)

        # Every sv_log_interval, log singular values
        if (config['sv_log_interval'] > 0) and (not (state_dict['itr'] % config['sv_log_interval'])):
          train_log.log(itr=int(state_dict['itr']),
                        **{**utils.get_SVs(save_and_eval_E, 'E')})

        # If using my progbar, print metrics.
        if config['pbar'] == 'mine':
          print(', '.join(['itr: %d' % state_dict['itr']]
                          + ['%s : %+4.3f' % (key, metrics[key])
                             for key in metrics]), end=' ')

        # Save weights and copies as configured at specified interval
        if not (state_dict['itr'] % config['save_every']):
          if config['G_eval_mode']:
            print('Switchin e to eval mode...')
            save_and_eval_E.eval()
            if config['ema']:
              E_ema.eval()
          sampled_ssgan.save_and_eavl(save_and_eval_E, E_ema, state_dict, config, experiment_name, KNN, test_log)
      del loader, pbar
    #  Increment epoch counter at end of epoch
    state_dict['epoch'] += 1


def main():
  # parse command line and run
  parser = utils.prepare_parser()
  config = vars(parser.parse_args())
  print(config)
  run(config)

if __name__ == '__main__':
  main()
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

# Import my stuff
from Metric import inception_utils
from Utils import utils, vae_utils
from Dataset import mini_datasets
from Network.S3GAN import extractor as Ex
from Training import train_fns
from sync_batchnorm import patch_replication_callback
from importlib import import_module

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
  E = Ex.Extractor(**config).to(device)

  # If using EMA, prepare it
  if config['ema']:
    print('Preparing EMA for E with decay of {}'.format(config['ema_decay']))
    E_ema = Ex.Extractor(**{**config, 'skip_init':True,
                            'no_optim': True}).to(device)
    ema = utils.ema(E, E_ema, config['ema_decay'], config['ema_start'])
  else:
    E_ema, ema = None, None

  print(E)
  print('Number of params in E: {}'.format(
    sum([p.data.nelement() for p in E.parameters()])))
  # Prepare state dict, which holds things like epoch # and itr #
  state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0,
                'best_IS': 0, 'best_FID': 999999, 'config': config}

  # If loading from a pre-trained model, load weights
  if config['resume']:
    print('Loading weights...')
    vae_utils.load_weights([E], state_dict,
                           config['weights_root'], experiment_name,
                           config['load_weights'] if config['load_weights'] else None,
                           [E_ema] if config['ema'] else None)

  # If parallel, parallelize the GD module
  if config['parallel']:
    E_parallel = nn.DataParallel(E)
    if config['cross_replica']:
      patch_replication_callback(E_parallel)

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
  D_batch_size = (config['batch_size'] * 8)
  loaders = mini_datasets.get_data_loaders(**{**config, 'batch_size': D_batch_size,
                                              'start_itr': state_dict['itr']})

  # Prepare noise and randomly sampled label arrays
  # Allow for different batch sizes in G
  G_batch_size = max(config['G_batch_size'], config['batch_size'])
  # Loaders are loaded, prepare the training function
  if config['which_train_fn'] == 'GAN':
    train = Ex.Extractor_training_function(E, ema, E_parallel, state_dict, config)
  # Else, assume debugging and use the dummy train fn
  else:
    train = train_fns.dummy_training_function()

  print('Beginning training at epoch %d...' % state_dict['epoch'])
  # Train for specified number of epochs, although we mostly track G iterations.
  for epoch in range(state_dict['epoch'], config['num_epochs']):    
    # Which progressbar to use? TQDM or my own?
    if config['pbar'] == 'mine':
      pbar = utils.progress(zip(loaders[0], loaders[1]), displaytype='s1k' if config['use_multiepoch_sampler'] else 'eta')
    else:
      pbar = tqdm(zip(loaders[0], loaders[1]))
    for i, (lx, ly, ux, uy) in enumerate(pbar):
      # Increment the iteration counter
      state_dict['itr'] += 1
      # Make sure G and D are in training mode, just in case they got set to eval
      # For D, which typically doesn't have BN, this shouldn't matter much.
      E.train()
      ## Last night we process here!
      if config['ema']:
        G_ema.train()
      if config['D_fp16']:
        x, y = x.to(device).half(), y.to(device)
      else:
        x, y = x.to(device), y.to(device)
      metrics = train(x, y)
      train_log.log(itr=int(state_dict['itr']), **metrics)
      
      # Every sv_log_interval, log singular values
      if (config['sv_log_interval'] > 0) and (not (state_dict['itr'] % config['sv_log_interval'])):
        train_log.log(itr=int(state_dict['itr']),
                      **{**utils.get_SVs(G, 'G'), **utils.get_SVs(D, 'D')})

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
          if config['ema']:
            G_ema.eval()
        train_fns.save_and_sample(G, D, G_ema, z_, y_, fixed_z, fixed_y,
                                  state_dict, config, experiment_name)

      # Test every specified interval
      if not (state_dict['itr'] % config['test_every']):
        if config['G_eval_mode']:
          print('Switchin G to eval mode...')
          G.eval()
        train_fns.test(G, D, G_ema, z_, y_, state_dict, config, sample,
                       get_inception_metrics, experiment_name, test_log)
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
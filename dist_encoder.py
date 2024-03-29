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
  timer = vae_utils.Timer()

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
  E = Encoder(**{**config, 'arch': 'default'}).to(device)
  Out = Encoder(**{**config, 'arch': 'out'}).to(device)

  # If using EMA, prepare it
  if config['ema']:
    print('Preparing EMA for G with decay of {}'.format(config['ema_decay']))
    E_ema = Encoder(**{**config, 'skip_init':True,
                       'no_optim': True, 'arch': 'default'}).to(device)
    O_ema = Encoder(**{**config, 'skip_init':True,
                       'no_optim': True, 'arch': 'out'}).to(device)
    eema = utils.ema(E, E_ema, config['ema_decay'], config['ema_start'])
    oema = utils.ema(Out, O_ema, config['ema_decay'], config['ema_start'])
  else:
    E_ema, eema, O_ema, oema = None, None, None, None

  print(E)
  print(Out)
  print('Number of params in E: {}'.format(
    *[sum([p.data.nelement() for p in net.parameters()]) for net in [E, Out]]))
  # Prepare state dict, which holds things like epoch # and itr #
  state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0,
                'best_IS': 0, 'best_FID': 999999, 'config': config, 'best_precise': 0.0}

  # If loading from a pre-trained model, load weights
  if config['resume']:
    print('Loading weights...')
    vae_utils.load_weights([E, Out], state_dict,
                           config['weights_root'], experiment_name,
                           config['load_weights'] if config['load_weights'] else None,
                           [E_ema, O_ema] if config['ema'] else [None])

  class Wrapper(nn.Module):
    def __init__(self):
      super(Wrapper, self).__init__()
      self.E = E
      self.O = Out

    def forward(self, x):
      x = self.E(x)
      x = self.O(x)
      return x

  W = Wrapper()

  # If parallel, parallelize the GD module
  if config['parallel']:
    W = nn.DataParallel(W)
    if config['cross_replica']:
      patch_replication_callback(W)


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
  # Batch size for dataloader, prefetch 8 times batch
  batch_size = config['batch_size'] * config['num_D_steps'] * config['num_D_accumulations']

  eval_loader = utils.get_data_loaders(**{**config, 'load_in_mem': False, 'use_multiepoch_sampler': False})[0]
  dense_eval = vae_utils.dense_eval(2048, config['n_classes'], steps=5).to(device)
  eval_fn = functools.partial(vae_utils.eval_encoder, sample_batch=10,
                              config=config, loader=eval_loader,
                              dense_eval=dense_eval, device=device)

  E_scheduler = torch.optim.lr_scheduler.StepLR(E.optim, step_size=2, gamma=0.1)
  O_scheduler = torch.optim.lr_scheduler.StepLR(Out.optim, step_size=2, gamma=0.1)

  def train(w, img):
    E.optim.zero_grad()
    Out.optim.zero_grad()
    w_ = W(img)
    loss = F.mse_loss(w_, w, reduction='mean')
    loss.backward()
    if config['E_ortho'] > 0.0:
      # Debug print to indicate we're using ortho reg in D.
      print('using modified ortho reg in E')
      utils.ortho(E, config['E_ortho'])
      utils.ortho(Out, config['E_ortho'])
    E.optim.step()
    Out.optim.step()
    out = {' loss': float(loss.item())}
    if config['ema']:
      for ema in [eema, oema]:
        ema.update(state_dict['itr'])
    del w_, loss
    return out



  start, end = sampled_ssgan.make_dset_range(config['ssgan_sample_root'], config['ssgan_piece'], batch_size)
  timer.update()
  print('Beginning training at epoch %d (runing time %02d day %02d h %02d min %02d sec) ...' %
        ((state_dict['epoch'],) + timer.runing_time))
  # Train for specified number of epochs, although we mostly track G iterations.
  for epoch in range(state_dict['epoch'], config['num_epochs']):
    for piece in range(config['ssgan_piece']):
      timer.update()
      print('Load %d-th piece of ssgan sample into memory (runing time %02d day %02d h %02d min %02d sec)...'
            % ((piece,) + timer.runing_time))
      loader = sampled_ssgan.get_SSGAN_sample_loader(**{**config, 'batch_size': batch_size,
                                                        'start_itr': state_dict['itr'],
                                                        'start': start[piece], 'end': end[piece]})
      for _ in range(200):
        for i, (img, z, w) in enumerate(loader):
          # Increment the iteration counter
          state_dict['itr'] += 1
          # Make sure G and D are in training mode, just in case they got set to eval
          # For D, which typically doesn't have BN, this shouldn't matter much.
          E.train()
          Out.train()
          if config['ema']:
            E_ema.train()
            O_ema.train()

          img, w = img.to(device), w.to(device)
          counter = 0
          img = torch.split(img, config['batch_size'])
          w = torch.split(w, config['batch_size'])
          metrics = train(w[counter], img[counter])
          counter += 1
          del img, w

          train_log.log(itr=int(state_dict['itr']), **metrics)

          if not (state_dict['itr'] % 100):
            timer.update()
            print("Runing time %02d day %02d h %02d min %02d sec," %
                  timer.runing_time + ', '.join(['itr: %d' % state_dict['itr']]
                                                + ['%s : %+4.3f' % (key, metrics[key])
                                                   for key in metrics]))


          # Save weights and copies as configured at specified interval
          if not (state_dict['itr'] % config['save_every']):
            if config['G_eval_mode']:
              print('Switchin E to eval mode...')
              E.eval()
              if config['ema']:
                E_ema.eval()
            sampled_ssgan.save_and_eavl(E, Out, E_ema, O_ema, state_dict, config, experiment_name, eval_fn, test_log)
      E_scheduler.step()
      O_scheduler.step()
      del loader
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

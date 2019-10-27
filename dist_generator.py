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
import numpy as np
from sync_batchnorm import patch_replication_callback
from Network.BigGAN.BigGAN import Generator
from Dataset import sampled_ssgan
from Training import train_fns


def sample_with_embed(G, embed, z_, y_, config):
  with torch.no_grad():
    z_.sample_()
    y_.sample_()
    if config['parallel']:
      w = nn.parallel.data_parallel(embed, z_)
      G_z =  nn.parallel.data_parallel(G, (w, G.shared(y_)))
    else:
      w = embed(z_)
      G_z = G(w, G.shared(y_))
    return G_z, y_


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

  # If using EMA, prepare it
  if config['ema']:
    print('Preparing EMA for G with decay of {}'.format(config['ema_decay']))
    G_ema = Generator(**{**config, 'skip_init':True,
                       'no_optim': True}).to(device)
    ema = utils.ema(G, G_ema, config['ema_decay'], config['ema_start'])
  else:
    G_ema, ema = None, None

  print(G)
  print('Number of params in E: {}'.format(
    *[sum([p.data.nelement() for p in net.parameters()]) for net in [G]]))
  # Prepare state dict, which holds things like epoch # and itr #
  state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0,
                'best_IS': 0, 'best_FID': 999999, 'config': config, 'best_precision': 0.0}

  # If loading from a pre-trained model, load weights
  if config['resume']:
    print('Loading weights...')
    vae_utils.load_weights([G], state_dict,
                           config['weights_root'], experiment_name,
                           config['load_weights'] if config['load_weights'] else None,
                           [G_ema] if config['ema'] else [None])

  class Wrapper(nn.Module):
    def __init__(self):
      super(Wrapper, self).__init__()
      self.G = G

    def forward(self, w, y):
      x = self.G(w, self.G.shared(y))
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

  get_inception_metrics = inception_utils.prepare_inception_metrics(config['dataset'], config['parallel'],
                                                                    config['data_root'], config['no_fid'])
  z_, y_ = utils.prepare_z_y(config['batch_size'], G.dim_z,
                            config['n_classes'], device=device,
                            fp16=config['G_fp16'])
  fixed_w, fixed_y = utils.prepare_z_y(config['batch_size'], G.dim_z,
                                       config['n_classes'], device=device,
                                       fp16=config['G_fp16'])
  fixed_w.sample_()
  fixed_y.sample_()
  G_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(G.optim, mode='min', factor=0.5)
  MSE = torch.nn.MSELoss(reduction='mean')

  def train(w, img):
    y_.sample_()
    G.optim.zero_grad()
    x = W(w, y_)
    loss = MSE(x, img)
    loss.backward()
    if config['E_ortho'] > 0.0:
      # Debug print to indicate we're using ortho reg in D.
      print('using modified ortho reg in E')
      utils.ortho(G, config['G_ortho'])
    G.optim.step()
    G_scheduler.step(loss)
    out = {' loss': float(loss.item())}
    if config['ema']:
      ema.update(state_dict['itr'])
    del w, img, loss, x
    return out

  class Embed(nn.Module):
    def __init__(self):
      super(Embed, self).__init__()
      embed = np.load('/ghome/fengrl/home/FGAN/embed_ema.npy')
      self.embed = torch.tensor(embed, requires_grad=False).to(device)

    def forward(self, z):
      z = torch.matmul(z, self.embed)
      return z

  embedding = Embed().to(device)
  fixed_w = torch.matmul(fixed_w, embedding.embed)

  sample = functools.partial(sample_with_embed,
                             embed=embedding,
                             G=(G_ema if config['ema'] and config['use_ema']
                                else G),
                             z_=z_, y_=y_, config=config)

  start, end = sampled_ssgan.make_dset_range(config['ssgan_sample_root'], config['ssgan_piece'], config['batch_size'])
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
        G.train()
        if config['ema']:
          G_ema.train()
        img, w = img.to(device), w.to(device)
        metrics = train(w, img)
        train_log.log(itr=int(state_dict['itr']), **metrics)

        # Every sv_log_interval, log singular values
        if (config['sv_log_interval'] > 0) and (not (state_dict['itr'] % config['sv_log_interval'])):
          train_log.log(itr=int(state_dict['itr']),
                        **{**utils.get_SVs(G, 'G')})

        # If using my progbar, print metrics.
        if config['pbar'] == 'mine':
          print(', '.join(['itr: %d' % state_dict['itr']]
                          + ['%s : %+4.3f' % (key, metrics[key])
                             for key in metrics]), end=' ')

        # Save weights and copies as configured at specified interval
        if not (state_dict['itr'] % config['save_every']):
          if config['G_eval_mode']:
            print('Switchin e to eval mode...')
            G.eval()
            if config['ema']:
              G_ema.eval()
          train_fns.save_and_sample(G, None, G_ema, z_, y_, fixed_w, fixed_y,
                                    state_dict, config, experiment_name)
          # Test every specified interval
        if not (state_dict['itr'] % config['test_every']):
          if config['G_eval_mode']:
            print('Switchin G to eval mode...')
            G.eval()
          train_fns.test(G, None, G_ema, z_, y_, state_dict, config, sample,
                         get_inception_metrics, experiment_name, test_log)
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

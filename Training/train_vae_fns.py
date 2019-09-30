import os

import torch
import torch.nn as nn
import torchvision

from Network.VaeGAN import losses
from Utils import utils, vae_utils, parallel_utils


# Dummy training function for debugging
def dummy_training_function():
    def train(x, y):
        return {}

    return train


def VAE_training_function(G, D, E, I, L, Decoder, z_, y_, ey_, ema_list, state_dict, config):
    def train(x):
        G.optim.zero_grad()
        D.optim.zero_grad()
        I.optim.zero_grad()
        E.optim.zero_grad()
        L.optim.zero_grad()
        # How many chunks to split x and y into?
        x = torch.split(x, config['batch_size'])
        counter = 0

        # Optionally toggle D and G's "require_grad"
        if config['toggle_grads']:
            utils.toggle_grad(D, True)
            utils.toggle_grad(L, True)
            utils.toggle_grad(G, False)
            utils.toggle_grad(I, False)
            utils.toggle_grad(E, False)

        for step_index in range(config['num_D_steps']):
            # If accumulating gradients, loop multiple times before an optimizer step
            D.optim.zero_grad()
            L.optim.zero_grad()
            for accumulation_index in range(config['num_D_accumulations']):
                z_.sample_()
                y_.sample_()
                ey_.sample_()
                D_fake, D_real, D_inv, D_en, _, _ = Decoder(z_[:config['batch_size']], y_[:config['batch_size']],
                                                            x[counter], ey_[:config['batch_size']], train_G=False,
                                                            split_D=config['split_D'])

                # Compute components of D's loss, average them, and divide by
                # the number of gradient accumulations
                D_loss_real, D_loss_fake = losses.discriminator_loss(D_fake, D_real)
                Latent_loss = losses.latent_loss_dis(D_inv, D_en)
                D_loss = (D_loss_real + D_loss_fake + Latent_loss) / float(
                    config['num_D_accumulations'])
                D_loss.backward()
                counter += 1

            # Optionally apply ortho reg in D
            if config['D_ortho'] > 0.0:
                # Debug print to indicate we're using ortho reg in D.
                print('using modified ortho reg in D and Latent_Binder')
                utils.ortho(D, config['D_ortho'])
                utils.ortho(L, config['L_ortho'])

            D.optim.step()
            L.optim.step()

        # Optionally toggle "requires_grad"
        if config['toggle_grads']:
            utils.toggle_grad(D, False)
            utils.toggle_grad(L, False)
            utils.toggle_grad(G, True)
            utils.toggle_grad(I, True)
            utils.toggle_grad(E, True)

        # Zero G's gradients by default before training G, for safety
        G.optim.zero_grad()
        I.optim.zero_grad()
        E.optim.zero_grad()
        counter = 0

        # If accumulating gradients, loop multiple times
        for accumulation_index in range(config['num_G_accumulations']):
            z_.sample_()
            y_.sample_()
            ey_.sample_()
            D_fake, _, D_inv, D_en, G_en, reals = Decoder(z_, y_,
                                                          x[counter], ey_, train_G=True, split_D=config['split_D'])
            G_loss_fake = losses.generator_loss(D_fake) * config['adv_loss_scale']
            Latent_loss = losses.latent_loss_gen(D_inv, D_en)
            Recon_loss = losses.recon_loss(G_en, reals)
            G_loss = (G_loss_fake + Latent_loss + Recon_loss) / float(config['num_G_accumulations'])
            G_loss.backward()
            counter += 1

        # Optionally apply modified ortho reg in G
        if config['G_ortho'] > 0.0:
            print('using modified ortho reg in G, Invert, and Encoder')
            # Debug print to indicate we're using ortho reg in G
            # Don't ortho reg shared, it makes no sense. Really we should blacklist any embeddings for this
            utils.ortho(G, config['G_ortho'],
                        blacklist=[param for param in G.shared.parameters()])
            utils.ortho(E, config['E_ortho'])
            utils.ortho(I, config['I_ortho'])
        G.optim.step()
        I.optim.step()
        E.optim.step()

        # If we have an ema, update it, regardless of if we test with it or not
        if config['ema']:
            for ema in ema_list:
                ema.update(state_dict['itr'])

        out = {'G_loss': float(G_loss.item()),
               'D_loss_real': float(D_loss_real.item()),
               'D_loss_fake': float(D_loss_fake.item()),
               'Latent_loss': float(Latent_loss.item()),
               'Recon_loss': float(Recon_loss.item())}
        
        # Release GPU memory:
        del G_loss, D_loss_real, D_loss_fake, Latent_loss, Recon_loss
        del D_fake, D_real, D_inv, D_en, G_en, reals
        del x
        
        # Return G's loss and the components of D's loss.
        return out
    return train


def parallel_training_function(G, D, E, I, L, Decoder, z_, y_, ey_, ema_list, state_dict, config):
    parallel_loss = losses.ParallelLoss(config)
    parallel_loss = parallel_utils.DataParallelCriterion(parallel_loss)

    def train(x):
        G.optim.zero_grad()
        D.optim.zero_grad()
        I.optim.zero_grad()
        E.optim.zero_grad()
        L.optim.zero_grad()
        # How many chunks to split x and y into?
        x = torch.split(x, config['batch_size'])
        counter = 0

        # Optionally toggle D and G's "require_grad"
        if config['toggle_grads']:
            utils.toggle_grad(D, True)
            utils.toggle_grad(L, True)
            utils.toggle_grad(G, False)
            utils.toggle_grad(I, False)
            utils.toggle_grad(E, False)

        for step_index in range(config['num_D_steps']):
            # If accumulating gradients, loop multiple times before an optimizer step
            D.optim.zero_grad()
            L.optim.zero_grad()
            for accumulation_index in range(config['num_D_accumulations']):
                z_.sample_()
                y_.sample_()
                ey_.sample_()
                out_tuple = Decoder(z_[:config['batch_size']], y_[:config['batch_size']],
                                    x[counter], ey_[:config['batch_size']], train_G=False,
                                    split_D=config['split_D'])

                # Compute components of D's loss, average them, and divide by
                # the number of gradient accumulations
                D_loss, out_dict_d = parallel_loss(out_tuple, training_G=False)
                D_loss.backward()
                counter += 1

            # Optionally apply ortho reg in D
            if config['D_ortho'] > 0.0:
                # Debug print to indicate we're using ortho reg in D.
                print('using modified ortho reg in D and Latent_Binder')
                utils.ortho(D, config['D_ortho'])
                utils.ortho(L, config['L_ortho'])

            D.optim.step()
            L.optim.step()

        # Optionally toggle "requires_grad"
        if config['toggle_grads']:
            utils.toggle_grad(D, False)
            utils.toggle_grad(L, False)
            utils.toggle_grad(G, True)
            utils.toggle_grad(I, True)
            utils.toggle_grad(E, True)

        # Zero G's gradients by default before training G, for safety
        G.optim.zero_grad()
        I.optim.zero_grad()
        E.optim.zero_grad()
        counter = 0

        # If accumulating gradients, loop multiple times
        for accumulation_index in range(config['num_G_accumulations']):
            z_.sample_()
            y_.sample_()
            ey_.sample_()
            out_tuple = Decoder(z_, y_,
                                x[counter], ey_, train_G=True, split_D=config['split_D'])
            G_loss, out_dict_g = parallel_loss(out_tuple, training_G=True)
            G_loss.backward()
            counter += 1

        # Optionally apply modified ortho reg in G
        if config['G_ortho'] > 0.0:
            print('using modified ortho reg in G, Invert, and Encoder')
            # Debug print to indicate we're using ortho reg in G
            # Don't ortho reg shared, it makes no sense. Really we should blacklist any embeddings for this
            utils.ortho(G, config['G_ortho'],
                        blacklist=[param for param in G.shared.parameters()])
            utils.ortho(E, config['E_ortho'])
            utils.ortho(I, config['I_ortho'])
        G.optim.step()
        I.optim.step()
        E.optim.step()

        # If we have an ema, update it, regardless of if we test with it or not
        if config['ema']:
            for ema in ema_list:
                ema.update(state_dict['itr'])

        out = {'G_loss': float(G_loss.item()),
               'D_loss': float(D_loss.item())}
        out.update(out_dict_g)
        out.update(out_dict_d)

        # Release GPU memory:
        del G_loss, D_loss
        del x

        # Return G's loss and the components of D's loss.
        return out

    return train


''' This function takes in the model, saves the weights (multiple copies if 
    requested), and prepares sample sheets: one consisting of samples given
    a fixed noise seed (to show how the model evolves throughout training),
    a set of full conditional sample sheets, and a set of interp sheets. '''


def save_and_sample(G, D, E, I ,L , G_ema, I_ema, E_ema, z_, y_, fixed_z, fixed_y, fixed_x,
                    state_dict, config, experiment_name):
    vae_utils.save_weights([G, D, E, I, L], state_dict, config['weights_root'],
                           experiment_name, None, [G_ema, I_ema, E_ema] if config['ema'] else None)
    # Save an additional copy to mitigate accidental corruption if process
    # is killed during a save (it's happened to me before -.-)
    if config['num_save_copies'] > 0:
        vae_utils.save_weights([G, D, E, I, L], state_dict, config['weights_root'],
                               experiment_name,
                               'copy%d' % state_dict['save_num'],
                               [G_ema, I_ema, E_ema] if config['ema'] else None)
        state_dict['save_num'] = (state_dict['save_num'] + 1) % config['num_save_copies']

    # Use EMA G for samples or non-EMA?
    if config['ema'] and config['use_ema']:
        which_G, which_E, which_I = G_ema, E_ema, I_ema
    else:
        which_G, which_E, which_I = G, E, I

    # Accumulate standing statistics?
    if config['accumulate_stats']:
        vae_utils.accumulate_standing_stats([which_G, which_I, which_E],
                                            z_, y_, config['n_classes'],
                                            config['num_standing_accumulations'])

    # Save a random sample sheet with fixed z and y
    with torch.no_grad():
        if config['parallel']:
            fixed_inv = nn.parallel.data_parallel(which_I, fixed_z)
            fixed_Gz = nn.parallel.data_parallel(which_G, (fixed_inv, which_G.shared(fixed_y)))
            fixed_en = nn.parallel.data_parallel(which_E, fixed_x)
            fixed_Gx = nn.parallel.data_parallel(which_G, (fixed_en, which_G.shared(fixed_y)))
        else:
            fixed_inv = which_I(fixed_z)
            fixed_Gz = which_G(fixed_inv, which_G.shared(fixed_y))
            fixed_en = which_E(fixed_x)
            fixed_Gx = which_G(fixed_en, which_G.shared(fixed_y))
    if not os.path.isdir('%s/%s' % (config['samples_root'], experiment_name)):
        os.mkdir('%s/%s' % (config['samples_root'], experiment_name))
    image_filename = '%s/%s/fixed_samples%d.jpg' % (config['samples_root'],
                                                    experiment_name,
                                                    state_dict['itr'])
    vae_image_filename = '%s/%s/fixed_vae%d.jpg' % (config['samples_root'],
                                                    experiment_name,
                                                    state_dict['itr'])
    torchvision.utils.save_image(fixed_Gz.float().cpu(), image_filename,
                                 nrow=int(fixed_Gz.shape[0] ** 0.5), normalize=True)
    torchvision.utils.save_image(fixed_Gx.float().cpu(), vae_image_filename,
                                 nrow=int(fixed_Gx.shape[0] ** 0.5), normalize=True)
    # For now, every time we save, also save sample sheets
    vae_utils.sample_sheet(which_G, which_I,
                           classes_per_sheet=utils.classes_per_sheet_dict[config['dataset']],
                           num_classes=config['n_classes'],
                           samples_per_class=10, parallel=config['parallel'],
                           samples_root=config['samples_root'],
                           experiment_name=experiment_name,
                           folder_number=state_dict['itr'],
                           z_=z_)
    # Also save interp sheets
    for fix_z, fix_y in zip([False, False, True], [False, True, False]):
        vae_utils.interp_sheet(which_G, which_I,
                               num_per_sheet=16,
                               num_midpoints=8,
                               num_classes=config['n_classes'],
                               parallel=config['parallel'],
                               samples_root=config['samples_root'],
                               experiment_name=experiment_name,
                               folder_number=state_dict['itr'],
                               sheet_number=0,
                               fix_z=fix_z, fix_y=fix_y, device='cuda')


''' This function runs the inception metrics code, checks if the results
    are an improvement over the previous best (either in IS or FID, 
    user-specified), logs the results, and saves a best_ copy if it's an 
    improvement. '''


def test(G, D, E, I, L, KNN, G_ema, I_ema, E_ema, z_, y_, state_dict, config, sample, get_inception_metrics,
         experiment_name, test_log):
    print('Gathering inception metrics...')
    if config['accumulate_stats']:
        vae_utils.accumulate_standing_stats([G_ema, I_ema, E_ema] if config['ema'] and config['use_ema'] else
                                            [G, I, E],
                                            z_, y_, config['n_classes'],
                                            config['num_standing_accumulations'])
    IS_mean, IS_std, FID = get_inception_metrics(sample,
                                                 config['num_inception_images'],
                                                 num_splits=10)
    print('Itr %d: PYTORCH UNOFFICIAL Inception Score is %3.3f +/- %3.3f, PYTORCH UNOFFICIAL FID is %5.4f' % (
        state_dict['itr'], IS_mean, IS_std, FID))
    # If improved over previous best metric, save approrpiate copy
    if ((config['which_best'] == 'IS' and IS_mean > state_dict['best_IS'])
            or (config['which_best'] == 'FID' and FID < state_dict['best_FID'])):
        print('%s improved over previous best, saving checkpoint...' % config['which_best'])
        vae_utils.save_weights([G, D, E, I, L], state_dict, config['weights_root'],
                               experiment_name, 'best%d' % state_dict['save_best_num'],
                               [G_ema, I_ema, E_ema] if config['ema'] else None)
        state_dict['save_best_num'] = (state_dict['save_best_num'] + 1) % config['num_best_copies']
    state_dict['best_IS'] = max(state_dict['best_IS'], IS_mean)
    state_dict['best_FID'] = min(state_dict['best_FID'], FID)
    KNN_precision = KNN(E_ema if config['ema'] and config['use_ema'] else E)
    # Log results to file
    test_log.log(itr=int(state_dict['itr']), IS_mean=float(IS_mean),
                 IS_std=float(IS_std), FID=float(FID), KNN_precision=float(KNN_precision))

import warnings
from VAD_models import MaskedDecoder, MaskedVAD2,MaskedVAD, MaskedVAD_free
import sys
import math
import os
import argparse
import time
import itertools
import json

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.decomposition import PCA
import yaml
from easydict import EasyDict
from torchvision import transforms
import torchvision


def create_model(config):
    latent_size = config['n_latent_params']
    n_hidden = config['n_hidden']
    if type(n_hidden) == int:
        hidden_sizes = [n_hidden] * config['n_hidden_layers']

    output_size = config['output_size']
    layer_norm = config['layer_norm']
    dropout = config['dropout']
    activation = config['activation_fn']


    if config['model'] == 'ae':
        if config['kl']:
            warnings.warn('Adding KL term for non-variational autoencoder...', UserWarning)

        print("making MaskedDecoder", file=sys.stderr)
        return MaskedDecoder(latent_size, hidden_sizes, output_size, layer_norm,
                    dropout, activation)
    elif config['model'] == 'vae':
        if config['log_var'] is None:
            error_msg = "log_var cannot be None for vae. "
            error_msg += "Use --model vae_free if you want to optimize the log-variance."
            raise ValueError(error_msg)

        if not config['sfm_transform']:
            print("making MaskedVAD2", file=sys.stderr)
            output_size = 84*3
            return MaskedVAD2(latent_size, hidden_sizes, output_size, layer_norm,
                        dropout, activation, log_var=config['log_var'])
        
        else:
            print("making MaskedVAD", file=sys.stderr)
            return MaskedVAD(latent_size, hidden_sizes, output_size, layer_norm,
                        dropout, activation, log_var=config['log_var'])
    elif config['model'] == 'vae_free':
        if not config['kl']:
            warnings.warn("Training vae_free without KL term...", UserWarning)

        return MaskedVAD_free(latent_size, hidden_sizes, output_size, layer_norm,
                        dropout, activation)

    raise 

def torch_mse_mask(y_true, y_pred, mask):
    """
    Returns the mean MSE
    """
    sq_errors = ((y_pred - y_true) * mask) ** 2
    mean_error = sq_errors.sum() / mask.sum()
    return mean_error

def optimize_network(config, model, y, mask, mode):
    assert mode in ['train', 'test']

    # load appropriate hyper-parameters
    if mode == 'train':
        n_epochs = config['n_train_epochs']
        batch_size = config['batch_size']
        param_init = config['latent_param_init']

    elif mode == 'test':
        n_epochs = config['n_test_epochs']
        if config.get('test_batch_size') is not None:
            batch_size = config['test_batch_size']
        else:
            batch_size = config['batch_size']
        param_init = config['test_latent_param_init']

    print(f"Mode: {mode}")
    print(f"Batch size: {batch_size}")

    n_points = y.shape[0]
    

    # initialize latent variables
    if param_init == 'pca':
        pca = PCA(model.latent_size)
        pca.fit(y.cpu())

        latents = torch.tensor(pca.explained_variance_ratio_, dtype=torch.float, device=config['device'])
        latents = latents.repeat(n_points, 1)
        print(latents.size())

    elif param_init == 'train':
        assert mode != 'train'

        print("Initializing latents using training latents as mean", file=sys.stderr)
        train_latents = config['train_latents']
        train_means = torch.mean(train_latents, 0)
        train_std = torch.std(train_latents, 0)
        latents = torch.tensor(np.random.normal(train_means, train_std, size=(n_points, model.latent_size)), device=config['device'])

    else:
        latents = model.init_latents(n_points, config['device'], param_init)

    # latent parameters to update
    latents.requires_grad = True
    latent_params = [latents]
    if config['model'] == 'vae_free':
        # randomly init log_var
        latent_log_var = torch.randn_like(latents, device=config['device'])
        latent_log_var.requires_grad = True
        latent_params.append(latent_log_var)

    epoch = 0
    if mode == 'test':
        # freeze the network weights
        model.freeze_hiddens()

    if mode == 'train':
        lr = config['net_lr']
        latent_lr = config['latent_param_lr']
        if config['use_adam']:
            net_optimizer = optim.Adam(model.parameters(), lr=lr)
            latent_optimizer = optim.Adam(latent_params, lr=latent_lr)
        else:
            net_optimizer = optim.SGD(model.parameters(), lr=lr)
            latent_optimizer = optim.SGD(latent_params, lr=latent_lr)
        # for reduce lr on plateau
        net_scheduler = optim.lr_scheduler.ReduceLROnPlateau(net_optimizer, mode='min',
                factor=0.5, patience=10, verbose=True)
        latent_scheduler = optim.lr_scheduler.ReduceLROnPlateau(latent_optimizer, mode='min',
                factor=0.5, patience=10, verbose=True)

        optimizers = [net_optimizer, latent_optimizer]
        schedulers = [net_scheduler, latent_scheduler]
        print(f"Optimizer: {net_optimizer}, {latent_optimizer}", file=sys.stderr)

    elif mode == 'test':
        latent_lr = config['test_latent_param_lr']

        if config['use_adam']:
            optimizer = optim.Adam(latent_params, lr=latent_lr)
        else:
            optimizer = optim.SGD(latent_params, lr=latent_lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                factor=0.5, patience=10, verbose=True)

        optimizers = [optimizer]
        schedulers = [scheduler]
        print(f"Test optimizer: {optimizer}", file=sys.stderr)

    # start optimization loop
    start_time = time.time()
    losses = []

    while True:
        epoch += 1

        order = np.random.permutation(n_points)
        cumu_loss = 0
        cumu_total_loss = 0
        cumu_kl_loss = 0

        n_batches = n_points // batch_size
        # model.set_verbose(False)
        for i in range(n_batches):
            # model.zero_grad()
            for op in optimizers:
                op.zero_grad()
            # net_optimizer.zero_grad()
            # latent_optimizer.zero_grad()

            idxes = order[i * batch_size: (i + 1) * batch_size]

            if config['model'] == 'vae_free':
                pred_y = model(latents[idxes], latent_log_var[idxes])
            elif config['sfm_transform']:
                pred_y, transform_mat = model(latents[idxes])
            else:
                pred_y = model(latents[idxes])
                # print(pred_y)

            # model.set_verbose(False)
            # print(y.shape)
            # print(mask.shape)
            # print(idxes)

            masked_train = y[idxes] * mask[idxes]

            # loss with masking
            loss = torch_mse_mask(y[idxes], pred_y, mask[idxes])
            
            if config['kl']:
                if config['model'] == 'vae':
                    z_var = torch.full_like(latents[idxes], config['log_var'])
                    kl_loss = 0.5 * torch.sum(torch.exp(z_var) + latents[idxes]**2 - 1. - z_var) / batch_size
                    total_loss = loss + config['ratio_kl'] * kl_loss

                elif config['model'] == 'vae_free':
                    kl_loss = 0.5 * torch.sum(torch.exp(latent_log_var[idxes]) + latents[idxes]**2 - 1. - latent_log_var[idxes])
                    kl_loss /= batch_size
                    total_loss = loss + config['ratio_kl'] * kl_loss

                else:
                    raise NotImplementedError
            else:
                kl_loss = 0.

            total_loss = loss

            # loss = loss_fn(pred_y, train_y[idxes]
            # loss *= train_mask[idxes]
            cumu_total_loss += float(total_loss)
            cumu_loss += float(loss)
            cumu_kl_loss += float(kl_loss)

            total_loss.backward()

            # for name, param in model.named_parameters():
            #     print(param.grad)
            #     break

            for op in optimizers:
                op.step()
            # net_optimizer.step()
            # latent_optimizer.step()

        curr_time = time.time() - start_time
        avg_loss = cumu_loss / n_batches

        avg_kl_loss = cumu_kl_loss / n_batches
        avg_total_loss = cumu_total_loss / n_batches

        print("Epoch {} - Average loss: {:.6f}, Cumulative loss: {:.6f}, KL loss: {:.6f}, Average total loss: {:.6f} ({:.2f} s)".format(epoch, avg_loss, cumu_loss, avg_kl_loss, avg_total_loss, curr_time),
                file=sys.stderr)
        losses.append([float(avg_loss), float(avg_kl_loss), float(avg_total_loss)])

        # early stopping etc.
        if epoch >= n_epochs:
            print("Max number of epochs reached!", file=sys.stderr)
            break

        if not config['reduce']:
            for sch in schedulers:
                sch.step(cumu_loss)
            # net_scheduler.step(cumu_loss)
            # latent_scheduler.step(cumu_loss)

        sys.stderr.flush()
        sys.stdout.flush()

    if mode == 'train':
        # return final latent variables, to possibly initialize during testing
        if config['model'] == 'vae_free':
            train_latents = latents, latent_log_var
        else:
            train_latents = latents
        return train_latents, losses

    elif mode == 'test':
        print("Final test loss: {}".format(losses[-1]), file=sys.stderr)

        # get final predictions to get loss wrt unmasked test data
        all_pred = []
        with torch.no_grad():
            idxes = np.arange(n_points)
            n_batches = math.ceil(n_points / batch_size)

            for i in range(n_batches):
                idx = idxes[i*batch_size : (i+1)*batch_size]
                if config['model'] == 'vae_free':
                    pred_y = model(latents[idx], latent_log_var[idx])
                elif config['sfm_transform']:
                    pred_y, transform_mat = model(latents[idx])
                else:
                    pred_y = model(latents[idx])
                all_pred.append(pred_y)

        all_pred = torch.cat(all_pred, dim=0)

        if config['clean_y'] is not None:
            clean_y = config['clean_y']
            #final_test_loss = float(loss_fn(all_pred * test_mask, clean_y * test_mask))
            #final_clean_loss = float(loss_fn(all_pred, clean_y))
            final_test_loss = float(torch_mse_mask(clean_y, all_pred, mask))
            final_clean_loss = float(torch_mse_mask(clean_y, all_pred, torch.ones_like(all_pred)))
            print("Masked test loss: {}".format(final_test_loss), file=sys.stderr)
            print("Clean test loss: {}".format(final_clean_loss), file=sys.stderr)

            mse = torch.mean(torch.mean((all_pred - clean_y) ** 2, -1), -1)
            print("Manual calculation: {}".format(mse), file=sys.stderr)

        if config['model'] == 'vae_free':
            test_latents = latents, latent_log_var
        else:
            test_latents = latents

        return losses, (final_test_loss, final_clean_loss), all_pred, test_latents

def save_results(config, model_folder, results):
    train_latents = results['train_latents']
    train_loss = results['train_loss']
    model = results['model']

    # write the latents to a file
    train_latents_file = os.path.join(model_folder, 'final_latents.h5')
    print("saving final train and test latents to {}".format(train_latents_file), file=sys.stderr)
    with h5py.File(train_latents_file, 'w') as f:
        if config['model'] == 'vae_free':
            # save both mean and log variance
            g = f.create_group('train_latents')
            g.create_dataset('mu', data=train_latents[0].detach().cpu().numpy())
            g.create_dataset('log_var', data=train_latents[1].detach().cpu().numpy())
        else:
            f.create_dataset('train_latents', data=train_latents.detach().cpu().numpy())

    results_file = os.path.join(model_folder, 'results.json')
    print("writing to {}".format(results_file), file=sys.stderr)

    # convert to 2d list
    if type(train_loss[0]) is not list:
        train_loss = [[x] for x in train_loss]

    with open(results_file, 'w') as f:
        json.dump({
            'train_loss': train_loss
        }, f, indent=4)

    # save model dict
    model_file = os.path.join(model_folder, 'model.pth')
    print("saving model state_dict to {}".format(model_file), file=sys.stderr)
    torch.save(model.state_dict(), model_file)

class MNIST():
    def __init__(self, data_path):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5), (0.5))])

        self.dataset = torchvision.datasets.MNIST(root=data_path, train=True,
                                                download=False, transform=transform)
        
    
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, item):
        img, label = self.dataset[item]
        img_flat = img.view(-1, 1)
        return img_flat

def main():

    config_file = './config.yaml'

    with open(config_file, "r") as file:
        config = EasyDict(yaml.safe_load(file))

    config['cuda'] = torch.cuda.is_available()

    if config['cuda']:
        config['device'] = torch.device('cuda')
    else:
        config['device'] = torch.device('cpu')

    model = create_model(config)
    model = model.to(config['device'])

    here = os.path.join(os.path.dirname(__file__)) 
    data_path = os.path.join(here, '../../dataset/')

    dataset = MNIST(data_path) 
    y = []
    for i in range(len(dataset)):
        sample = dataset[i]
        y.append(np.array(sample))

    train_y = np.array(y).astype(np.float)
    train_mask = np.ones_like(train_y).astype(np.float)

    train_y = np.squeeze(train_y)
    train_mask = np.squeeze(train_mask)

    train_y = torch.tensor(train_y).to(config['device'])
    train_mask = torch.tensor(train_mask).to(config['device'])

    train_latents, train_loss = optimize_network(config, model, train_y, train_mask, 'train')

    basedir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    model_folder = os.path.join(basedir, 'model_saves')

    if not os.path.isdir(model_folder):
        os.makedirs(model_folder)

    save_results(config, model_folder, {
        'train_latents': train_latents,
        'train_loss': train_loss,
        'model': model,
    })


if __name__== '__main__':
    main()
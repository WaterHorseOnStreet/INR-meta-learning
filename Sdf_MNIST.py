import os
import torch

import numpy as np
import matplotlib.pyplot as plt

import torchvision
from torchvision import transforms
import scipy.ndimage
from torch import nn
from collections import OrderedDict, Mapping
from torch.utils.data import DataLoader, Dataset

from torch.nn.init import _calculate_correct_fan

from torchmeta.modules import (MetaModule, MetaSequential, MetaLinear)
from Siren_meta import BatchLinear, MetaFC, MAML, plot_sample_image, dict_to_gpu, lin2img

def get_mgrid(sidelen):
    # Generate 2D pixel coordinates from an image of sidelen x sidelen
    pixel_coords = np.stack(np.mgrid[:sidelen,:sidelen], axis=-1)[None,...].astype(np.float32)
    pixel_coords /= sidelen    
    pixel_coords -= 0.5
    pixel_coords = torch.Tensor(pixel_coords).view(-1, 2)
    return pixel_coords

class SignedDistanceTransform:
    def __call__(self, img_tensor):
        # Threshold.
        img_tensor[img_tensor<0.5] = 0.
        img_tensor[img_tensor>=0.5] = 1.

        # Compute signed distances with distance transform
        img_tensor = img_tensor.numpy()

        neg_distances = scipy.ndimage.morphology.distance_transform_edt(img_tensor)
        sd_img = img_tensor - 1.
        sd_img = sd_img.astype(np.uint8)
        signed_distances = scipy.ndimage.morphology.distance_transform_edt(sd_img) - neg_distances
        signed_distances /= float(img_tensor.shape[1])
        signed_distances = torch.Tensor(signed_distances)

        return signed_distances, torch.Tensor(img_tensor)


class MNISTSDFDataset(torch.utils.data.Dataset):
    def __init__(self, split, size=(256,256)):
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            SignedDistanceTransform(),
        ])
        self.img_dataset = torchvision.datasets.MNIST('./data/MNIST', train=True if split == 'train' else False,
                                                download=True)
        self.meshgrid = get_mgrid(size[0])
        self.im_size = size

    def __len__(self):
        return len(self.img_dataset)

    def __getitem__(self, item):
        img, digit_class = self.img_dataset[item]

        signed_distance_img, binary_image = self.transform(img)
        
        coord_values = self.meshgrid.reshape(-1, 2)
        signed_distance_values = signed_distance_img.reshape((-1, 1))
        
        indices = torch.randperm(coord_values.shape[0])
        support_indices = indices[:indices.shape[0]//2]
        query_indices = indices[indices.shape[0]//2:]

        meta_dict = {'context': {'x':coord_values[support_indices], 'y':signed_distance_values[support_indices]}, 
                     'query': {'x':coord_values[query_indices], 'y':signed_distance_values[query_indices]}, 
                     'all': {'x':coord_values, 'y':signed_distance_values}}

        return meta_dict

def sal_init(m):
    if type(m) == BatchLinear or nn.Linear:
        if hasattr(m, 'weight'):
            std = np.sqrt(2) / np.sqrt(_calculate_correct_fan(m.weight, 'fan_out'))

            with torch.no_grad():
                m.weight.normal_(0., std)
        if hasattr(m, 'bias'):
            m.bias.data.fill_(0.0)


def sal_init_last_layer(m):
    if hasattr(m, 'weight'):
        val = np.sqrt(np.pi) / np.sqrt(_calculate_correct_fan(m.weight, 'fan_in'))
        with torch.no_grad():
            m.weight.fill_(val)
    if hasattr(m, 'bias'):
        m.bias.data.fill_(0.0)

def sdf_loss(predictions, gt):
    return ((predictions - gt)**2).mean()


def inner_maml_sdf_loss(predictions, gt):
    return ((predictions - gt)**2).sum(0).mean()

def main():
    train_dataset = MNISTSDFDataset('train', size=(64, 64))
    val_dataset = MNISTSDFDataset('val', size=(64, 64))

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=16)

    hypo_module = MetaFC(in_features=2, out_features=1, 
                        num_hidden_layers=2, hidden_features=256, 
                        outermost_linear=True)
    hypo_module.apply(sal_init)
    hypo_module.net[-1].apply(sal_init_last_layer)
    model = MAML(num_meta_steps=3, hypo_module=hypo_module, loss=inner_maml_sdf_loss, init_lr=1e-5, 
                lr_type='global').cuda()


    optim = torch.optim.Adam(lr=1e-4, params=model.parameters())

    train_losses = []
    val_losses = []

    for epoch in range(3):
        for step, meta_batch in enumerate(train_dataloader):
            model.train()        
            meta_batch = dict_to_gpu(meta_batch)

            # Adapt model using context examples
            fast_params, _ = model.generate_params(meta_batch['context'])
            
            # Use the adapted examples to make predictions on query
            pred_sd = model.forward_with_params(meta_batch['query']['x'], fast_params)
            
            # Calculate loss on query examples
            loss = sdf_loss(pred_sd, meta_batch['query']['y'].cuda())
            train_losses.append(loss.item())

            optim.zero_grad()
            loss.backward()
            optim.step()

            if step % 100 == 0:
                # Assemble validation input
                meta_batch['context']['x'] = meta_batch['query']['x'] = meta_batch['all']['x']
                meta_batch['context']['y'] = meta_batch['query']['y'] = meta_batch['all']['y']

                with torch.no_grad():
                    model_output = model(meta_batch)

                print("Step %d, Total loss %0.6f" % (step, loss))
                fig, axes = plt.subplots(1,5, figsize=(30,6))
                ax_titles = ['Learned Initialization', 'Inner step 1 output', 
                            'Inner step 2 output', 'Inner step 3 output', 
                            'Ground Truth']
                for i, inner_step_out in enumerate(model_output['intermed_predictions']):
                    plot_sample_image(inner_step_out, ax=axes[i])
                    axes[i].set_axis_off()
                    axes[i].set_title(ax_titles[i], fontsize=25)
                plot_sample_image(model_output['model_out'], ax=axes[-2])
                axes[-2].set_axis_off()
                axes[-2].set_title(ax_titles[-2], fontsize=25)

                plot_sample_image(meta_batch['query']['y'], ax=axes[-1])
                axes[-1].set_axis_off()
                axes[-1].set_title(ax_titles[-1], fontsize=25)
                plt.show()

    with torch.no_grad():
        model.eval()
        for step, meta_batch in enumerate(val_dataloader):
            # Instead of explicitly calling generate_params and forward_with_params,
            # we can pass the meta_batch dictionary to the model's forward method
            pred_sd = model(meta_batch)
            val_loss = sdf_loss(pred_sd['model_out'], meta_batch['query']['y'].cuda())
            val_losses.append(val_loss.item())
            
            if step % 1000 == 0:
                pred_image = model.forward_with_params(meta_batch['all']['x'].cuda(), fast_params)
                print(f"Val Image -- Epoch: {epoch} \t step: {step} \t loss: {loss.item()}")
                plt.imshow(lin2img(pred_image).cpu().numpy()[0][0])
                plt.show()


if __name__ == "__main__":
    main()
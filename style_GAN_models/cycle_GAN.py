import argparse
import enum
import os
import numpy as np
import math
import itertools
import datetime
import time
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable



from utils import ReplayBuffer, LambdaLR 
import yaml
from easydict import EasyDict

import torch.nn as nn
import torchvision.transforms.functional as F
import torch
from Kitti import Kitti
from style_GAN_model import MixVisionTransformer
from networks import SyntheticNet
from cycleGANModel import *
import torchvision.transforms.functional as F

class MyGenerator(nn.Module):
    def __init__(self,img_size):
        super(MyGenerator, self).__init__()

        self.img_size = img_size

        channel_table = []
        self.latent_dim = 512
        fmap_base = 8192
        for i in range(1,8):
            channel_table.append(min(int(fmap_base / (2.0 ** (i * 1))), 512))

        self.channel_table = channel_table
        self.MixViT = MixVisionTransformer(img_size=self.img_size)
        self.Synthetic = SyntheticNet(in_dim=self.latent_dim,out_channel=2*self.latent_dim,channel_table=self.channel_table)
        input_tensor = torch.rand([1,512,4,4])
        self.input_tensor = nn.Parameter(input_tensor,requires_grad=False)

    def forward(self,x):
        result = self.MixViT(x.float())
        result = torch.cat(result,dim=1)

        result = self.Synthetic(self.input_tensor,result)

        return result.float()

config_file = './config.yaml'
config = None
with open(config_file, "r") as file:
    config = EasyDict(yaml.safe_load(file))
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

cuda = torch.cuda.is_available()

input_shape = (3,256,256)

n_residual_blocks = 9

# our images in two domains have different size, pay attention
G_AB = MyGenerator(256)
G_BA = MyGenerator(256)

D_A = Discriminator(input_shape)
D_B = Discriminator(input_shape)

if cuda:
    G_AB = G_AB.cuda()
    G_BA = G_BA.cuda()
    D_A = D_A.cuda()
    D_B = D_B.cuda()
    criterion_cycle.cuda()
    criterion_GAN.cuda()
    criterion_identity.cuda()

# G_AB.apply(weights_init_normal)
# G_BA.apply(weights_init_normal)
D_A.apply(weights_init_normal)
D_B.apply(weights_init_normal)

lr = 0.001
betas = (0.0,0.9)
optimizer_G = torch.optim.Adam(
    itertools.chain(G_AB.parameters(),G_BA.parameters()), lr=lr, betas=betas
)
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=lr, betas=betas)
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=lr, betas=betas)

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(10,0,2).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(10,0,2).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(10,0,2).step)

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

fake_A_Buffer = ReplayBuffer()
fake_B_Buffer = ReplayBuffer()


# transforms_ = transforms.Compose([
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])]
# )

def kitti_collate(data):
    """
       data: is a list of tuples with (example, label, length)
             where 'example' is a tensor of arbitrary shape
             and label/length are scalars
    """
    images, labels = zip(*data)

    imgs = []
    for img in images:
        img = np.array(img)
        img = torch.tensor(img)
        img = img.permute(2,0,1)
        
        img = F.crop(img,0,0,384,1240)
        img = img.unsqueeze(0)
        img = torch.nn.functional.interpolate(img, size=(256,256))
        img = img.squeeze()
        imgs.append(img)

    images = torch.stack(imgs,dim=0)

    return images,labels


def Cityscapes_collate(data):
    """
       data: is a list of tuples with (example, label, length)
             where 'example' is a tensor of arbitrary shape
             and label/length are scalars
    """
    images, labels = zip(*data)

    imgs = []
    for img in images:
        img = np.array(img)
        img = torch.tensor(img)
        img = img.permute(2,0,1)
        img = F.crop(img,0,0,384,1240)
        img = img.unsqueeze(0)
        img = torch.nn.functional.interpolate(img, size=(256,256))
        img = img.squeeze()
        imgs.append(img)

    images = torch.stack(imgs,dim=0)

    return images,labels


def sample_images(dataloader_A,dataloader_B,batches_done):
    """Saves a generated sample from the test set"""
    sample_A = next(iter(dataloader_A))
    sample_B = next(iter(dataloader_B))
    G_AB.eval()
    G_BA.eval()
    real_A = Variable(sample_A[0].float().cuda())
    fake_B = G_AB(real_A)
    real_B = Variable(sample_B[0].float().cuda())
    fake_A = G_BA(real_B)
    # Arange images along x-axis
    real_A = make_grid(real_A, nrow=5, normalize=True)
    real_B = make_grid(real_B, nrow=5, normalize=True)
    fake_A = make_grid(fake_A, nrow=5, normalize=True)
    fake_B = make_grid(fake_B, nrow=5, normalize=True)
    # Arange images along y-axis
    image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
    save_image(image_grid, "images/%s/%s.png" % (config['dataset_name'], batches_done), normalize=False)
    
here = os.getcwd()

kitti_path = os.path.join(here,'../../../dataset/kitti/')

kitti = Kitti(root=kitti_path)

cityscapes_path = os.path.join(here,'../../../dataset/cityscapes')
Cityscapes_dataset = torchvision.datasets.Cityscapes(root=cityscapes_path, split='train', mode='fine',target_type='semantic')

batch_size = 1
# mixed_synth = 0.4
dataloader_A = DataLoader(kitti,batch_size=batch_size,
                        shuffle=True, num_workers=0,collate_fn=kitti_collate)

dataloader_B = DataLoader(Cityscapes_dataset,batch_size=batch_size,
                        shuffle=True, num_workers=0,collate_fn=Cityscapes_collate)

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad
    
def train():

    len_dataloader= len(dataloader_A) if len(dataloader_A) < len(dataloader_B) else len(dataloader_B)
    print('length of dataset is {}'.format(len_dataloader))
    for epoch in range(100):
        for iter, (k,c) in enumerate(zip(dataloader_A, dataloader_B)):

            # rea = Variable(k[0].type(Tensor))
            # syn = Variable(c[0].type(Tensor))
            rea = k[0].float().cuda()
            syn = c[0].float().cuda()
            valid = Variable(Tensor(np.ones((rea.size(0),*D_A.output_shape))), requires_grad=False)
            fake = Variable(Tensor(np.ones((rea.size(0),*D_A.output_shape))), requires_grad=False)

            G_AB.train()
            G_BA.train()
            set_requires_grad([D_A,D_B],False)
            optimizer_G.zero_grad()

            loss_id_A = criterion_identity(G_BA(rea), rea)
            loss_id_B = criterion_identity(G_AB(syn), syn)

            loss_identity = (loss_id_A + loss_id_B) / 2

            fake_B = G_AB(rea)
            loss_GAN_AB = criterion_GAN(D_B(fake_B.float()),valid)
            fake_A = G_BA(syn)
            loss_GAN_BA = criterion_GAN(D_A(fake_A.float()),valid)

            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

            recov_real = G_BA(fake_B)
            loss_cycle_A = criterion_cycle(recov_real,rea)
            recov_syn = G_AB(fake_A)
            loss_cycle_B = criterion_cycle(recov_syn,syn)
            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

            lambda_cyc = 10.0
            lambda_id = 5.0
            
            loss_G = loss_GAN + lambda_cyc * loss_cycle + lambda_id * loss_identity

            loss_G.backward()
            optimizer_G.step()

            set_requires_grad([D_A,D_B],True)
            optimizer_D_A.zero_grad()

            loss_real = criterion_GAN(D_A(rea), valid)
            fake_A_ = fake_A_Buffer.push_and_pop(fake_A)

            loss_fake = criterion_GAN(D_A(fake_A_.detach()),fake)

            loss_D_A = (loss_real + loss_fake) / 2

            loss_D_A.backward()
            optimizer_D_A.step()

            optimizer_D_B.zero_grad()

            loss_real = criterion_GAN(D_B(syn), valid)
            fake_B_ = fake_B_Buffer.push_and_pop(fake_B)

            loss_fake = criterion_GAN(D_A(fake_B_.detach()),fake)

            loss_D_B = (loss_real + loss_fake) / 2

            loss_D_B.backward()
            optimizer_D_B.step()

            loss_D = (loss_D_A + loss_D_B) / 2

            batches_done = epoch * len_dataloader+ iter
            batches_left = config['n_epochs'] * len_dataloader - batches_done

            
            if batches_done % config['sample_interval'] == 0:
                sample_images(dataloader_A,dataloader_B,batches_done)

            if batches_done % 200 == 0:
                print("in iter {}, D loss: {}, G loss: {}, adv loss: {}, cycle loss: {}, identity loss: {}".format(batches_done,
                            loss_D.item(),loss_G.item(),loss_GAN.item(),loss_cycle.item(),loss_identity.item()))

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        if config['checkpoint_interval'] != -1 and epoch % config['checkpoint_interval'] == 0:
            # Save model checkpoints
            torch.save(G_AB.state_dict(), "saved_models/%s/G_AB_%d.pth" % (config['dataset_name'], epoch))
            torch.save(G_BA.state_dict(), "saved_models/%s/G_BA_%d.pth" % (config['dataset_name'], epoch))
            torch.save(D_A.state_dict(), "saved_models/%s/D_A_%d.pth" % (config['dataset_name'], epoch))
            torch.save(D_B.state_dict(), "saved_models/%s/D_B_%d.pth" % (config['dataset_name'], epoch))

if __name__=='__main__':
    train()
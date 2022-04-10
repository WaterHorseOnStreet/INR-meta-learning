import argparse
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

from cycleGANModel import *

from utils import ReplayBuffer, LambdaLR 
from GTA5_dataset import GTA5

import torch.nn as nn
import torchvision.transforms.functional as F
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epoch", type=int, default=10, help="epoch to start training")

opt = parser.parse_args()

criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

cuda = torch.cuda.is_available()

input_shape = (3,128,256)

n_residual_blocks = 9

# our images in two domains have different size, pay attention
G_AB = GeneratorResNet(input_shape,n_residual_blocks)
G_BA = GeneratorResNet(input_shape,n_residual_blocks)

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

G_AB.apply(weights_init_normal)
G_BA.apply(weights_init_normal)
D_A.apply(weights_init_normal)
D_B.apply(weights_init_normal)

lr = 0.0001
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

transforms_ = [
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
]

def sample_images(val_dataloader,batches_done):
    """Saves a generated sample from the test set"""
    imgs = next(iter(val_dataloader))
    G_AB.eval()
    G_BA.eval()
    real_A = Variable(imgs["A"].type(Tensor))
    fake_B = G_AB(real_A)
    real_B = Variable(imgs["B"].type(Tensor))
    fake_A = G_BA(real_B)
    # Arange images along x-axis
    real_A = make_grid(real_A, nrow=5, normalize=True)
    real_B = make_grid(real_B, nrow=5, normalize=True)
    fake_A = make_grid(fake_A, nrow=5, normalize=True)
    fake_B = make_grid(fake_B, nrow=5, normalize=True)
    # Arange images along y-axis
    image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
    save_image(image_grid, "images/%s/%s.png" % (opt.dataset_name, batches_done), normalize=False)


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
        img = torch.nn.functional.interpolate(img, size=(128,256))
        imgs.append(img.squeeze())
        img = torch.nn.functional.interpolate(img, size=(128,256))
        imgs.append(img.squeeze())

    images = torch.stack(imgs,dim=0)

    return images,labels

def GTA5_collate(data):
    """
       data: is a list of tuples with (example, label, length)
             where 'example' is a tensor of arbitrary shape
             and label/length are scalars
    """
    images, labels = zip(*data)

    imgs = []
    for img in images:
        img = torch.tensor(img)
        img = F.crop(img,0,0,384,1240)
        img = img.unsqueeze(0)
        img = torch.nn.functional.interpolate(img, size=(128,256))
        imgs.append(img.squeeze())

    images = torch.stack(imgs,dim=0)

    return images,labels

here = os.getcwd()
kitti_path = os.path.join(here,'./data/')

kitti_dataset = torchvision.datasets.Kitti(root=kitti_path,download=False)

data_path = os.path.join(here,'./data/GTA5/')

GTA5_dataset = GTA5(data_path)

batch_size = 1
# mixed_synth = 0.4
real_dataloader = DataLoader(kitti_dataset,batch_size=batch_size,
                        shuffle=True, num_workers=0,collate_fn=kitti_collate)

synth_dataloader = DataLoader(GTA5_dataset,batch_size=batch_size,
                        shuffle=True, num_workers=0,collate_fn=GTA5_collate)

def train():
    for epoch in range(10):
        for iter, (real,synth) in enumerate(zip(real_dataloader, synth_dataloader)):
            rea = Variable(real[0].type(Tensor))
            syn = Variable(synth[0].type(Tensor))

            valid = Variable(Tensor(np.ones((rea.size(0),*D_A.output_shape))), requires_grad=False)
            fake = Variable(Tensor(np.ones((rea.size(0),*D_A.output_shape))), requires_grad=False)

            G_AB.train()
            G_BA.train()

            optimizer_G.zero_grad()

            loss_id_A = criterion_identity(G_BA(rea), rea)
            loss_id_B = criterion_identity(G_AB(syn), syn)

            loss_identity = (loss_id_A + loss_id_B) / 2

            fake_B = G_AB(rea)
            loss_GAN_AB = criterion_GAN(D_B(fake_B),valid)
            fake_A = G_BA(syn)
            loss_GAN_BA = criterion_GAN(D_A(fake_A),valid)

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

            batches_done = epoch * len(real_dataloader) + i
            batches_left = opt.n_epochs * len(real_dataloader) - batches_done

            print("in iter {}, D loss: {}, G loss: {}, adv loss: {}, cycle loss: {}, identity loss: {}".format(iter,
                        loss_D.item(),loss_G.item(),loss_GAN.item(),loss_cycle.item(),loss_identity.item()))
            
            if batches_done % opt.sample_interval == 0:
                sample_images(batches_done)

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(G_AB.state_dict(), "saved_models/%s/G_AB_%d.pth" % (opt.dataset_name, epoch))
            torch.save(G_BA.state_dict(), "saved_models/%s/G_BA_%d.pth" % (opt.dataset_name, epoch))
            torch.save(D_A.state_dict(), "saved_models/%s/D_A_%d.pth" % (opt.dataset_name, epoch))
            torch.save(D_B.state_dict(), "saved_models/%s/D_B_%d.pth" % (opt.dataset_name, epoch))
if __name__=="__main__":
    train()
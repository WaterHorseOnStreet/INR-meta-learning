import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os

from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize, CenterCrop
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms.functional as F

import time
from GTA5_dataset import GTA5

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
        img = F.crop(img,0,0,370,1240)
        imgs.append(img)

    images = torch.stack(imgs,dim=0)

    return images,labels

here = os.getcwd()
kitti_path = os.path.join(here,'./data/')

kitti_dataset = torchvision.datasets.Kitti(root=kitti_path,download=False)

data_path = os.path.join(here,'./data/GTA5/')

GTA5_dataset = GTA5(data_path)

batch_size = 32
mixed_synth = 0.4
real_dataloader = DataLoader(kitti_dataset,batch_size=int((1-mixed_synth)*batch_size),
                        shuffle=True, num_workers=0,collate_fn=kitti_collate)

synth_dataloader = DataLoader(GTA5_dataset,batch_size=int(mixed_synth*batch_size),
                        shuffle=True, num_workers=0)

for iter, (real,synth) in enumerate(zip(real_dataloader, synth_dataloader)):
    print(real[1][0][0].keys())
    print(synth[0].shape)

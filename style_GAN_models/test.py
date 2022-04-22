from cProfile import label
import imp
# from turtle import forward
# from tkinter import Image
from PIL import Image
from matplotlib.pyplot import xcorr
from modules import MergeModule, CondInstanceNorm,CINResnetBlock,ResnetBlock_unbalance
import numpy as np
import torch
from networks import define_stochastic_G,define_InitialDeconv,MappingNetwork,StyleAdder,SyntheticNet,Discriminator
import os
from torch.autograd import Variable
import torchvision
from Kitti import Kitti
from BDD import BDDDataset
import torch.nn as nn
import torchvision.transforms as T

# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
latent_dim = 100
input_dim = 3
output_dim = 10
ConstNorm1 = CondInstanceNorm(input_dim,latent_dim)
input_feat = torch.rand([32,3,256,256])
latent_feat = torch.rand([2,100,224,224])
# # # print(ConstNorm1(input_feat,latent_feat).shape)
# # CRes = CINResnetBlock(input_dim, latent_dim)
# # unbanl_resnet = ResnetBlock_unbalance(input_dim, latent_dim)

from style_GAN_models import MixVisionTransformer
MVit = MixVisionTransformer(img_size=256)

class MyGenerator(nn.Module):
    def __init__(self,img_size):
        super(MyGenerator, self).__init__()

        self.img_size = img_size

        channel_table = []
        self.latent_dim = 512
        fmap_base = 8192
        for i in range(1,8):
            channel_table.append(min(int(fmap_base / (2.0 ** (i * 1))), latent_dim))

        self.channel_table = channel_table
        self.MixViT = MixVisionTransformer(img_size=self.img_size)
        self.Synthetic = SyntheticNet(in_dim=self.latent_dim,out_channel=2*self.latent_dim,channel_table=self.channel_table )
        self.input_tensor = torch.rand([1,512,4,4])

    def forward(self,x):
        result = self.MixViT(x.float())
        result = torch.cat(result,dim=1)
        result = self.Synthetic(self.input_tensor,result)

        return result.float()

batch_size = 32
latent_dim = 512
label_dim = 512
# # label_feat = torch.rand([batch_size,label_dim])
# # latent_feat = torch.rand([batch_size,latent_dim])
# # MNet = MappingNetwork(latent_dim,300,100,label_dim=label_dim)
# # result = MNet(latent_feat,label_feat)
# # print(result.shape)
channel = 18

# x = torch.rand([batch_size,100,latent_dim,latent_dim])
# SD = StyleAdder(latent_dim,2*latent_dim)
# result = SD(latent_feat)
# print(result.shape)
# result = result[:,0].view([-1, 2, x.shape[1]] + [1] * (len(x.shape) - 2))
# print(result.shape)
# print(result[:,0].shape)
# x = x*result[:,0] + result[:,1]

# transform = T.ToPILImage()
# image = Image.open('./1.png')
# image = image.resize((256,256))
# image = torch.tensor(np.array(image).astype(np.uint8).transpose(2,0,1)[:3,:,:]).unsqueeze(0)
# print(image.shape)
# img1 = transform(image[0,:,:,:])
# img1.save('test1.png')

# print(type(image))
# result = MVit(image.float())
# result = torch.cat(result,dim=1)

# channel_table = []
# fmap_base = 8192
# for i in range(1,8):
#     channel_table.append(min(int(fmap_base / (2.0 ** (i * 1))), latent_dim))
# print(channel_table)
# latent_feat = torch.rand([batch_size,channel,latent_dim])
# input_tensor = torch.rand([batch_size,latent_dim,4,4]) # input of synthetic network is a constant tensor
# SN = SyntheticNet(in_dim=latent_dim,out_channel=2*latent_dim,channel_table=channel_table)
# # print(SN)
# result = SN(input_tensor,result)
# print(result.shape)
# img2 = transform(result[0,:,:,:])
# img2.save('test2.png')





# channel_table = channel_table[::-1]
# print(channel_table)
# in_channels = 3
# DD = Discriminator(in_channels,result.shape[-1],latent_dim,channel_table=channel_table)
# result_D = DD(result)
# print(result_D.shape)



# here = os.getcwd()
# kitti_path = os.path.join(here,'../../../dataset/kitti/')
# kitti = Kitti(root=kitti_path)
# print(next(iter(kitti)))

# cityscapes_path = os.path.join(here,'../../../dataset/cityscapes')
# Cityscapes_dataset = torchvision.datasets.Cityscapes(root=cityscapes_path, split='train', mode='fine',target_type='semantic')
# sample,img = Cityscapes_dataset[40]
# img = np.array(img)
# sample = np.array(sample)
# print(np.unique(np.array(img)))
# id = 26
# mask = img == id
# R = np.where(img == id,sample[:,:,0],0)
# print(R.shape)
# G = np.where(img == id,sample[:,:,1],0)
# print(G.shape)
# B = np.where(img == id,sample[:,:,2],0)
# A = np.ones_like(R)*255
# pos = np.where(mask)
# x_min = np.min(pos[1])
# x_max = np.max(pos[1])

# y_min = np.min(pos[0])
# y_max = np.max(pos[0])
# print((x_min,y_min,x_max,y_max))
# image = Image.fromarray(np.stack((R,G,B,A)).transpose(1,2,0))
# image = image.crop((x_min,y_min,x_max,y_max))
# image.save('1.png')
# print(img.shape)

# image = Image.fromarray(sample)
# image.save('2.png')
# print(img.shape)


# sample = np.array(sample)
# semantic = np.array(img[0])
# color = np.array(img[1])
# print(np.unique(semantic))
# masks = semantic == 33
# print(np.sum(masks))
# mask = np.unique(masks)
# print(mask)
# for i in range(masks.shape[0]):
#     for j in range(masks.shape[1]):
#         if color[i,j,0] != 64 or color[i,j,1] != 0 or color[i,j,2] != 128:
#             sample[i,j,:] = [0,0,0]

# print(np.sum(sample))
# print(np.unique(sample))

# img1 = sample.astype(np.uint8)
# image = Image.fromarray(img1)
# image.save('1.png')
# print(img1.shape)





# print(CRes(input_feat,latent_feat).shape)

# sign = '3'
# netDeConv=define_InitialDeconv()
# random_number = np.zeros((1,4,1,1))
# if sign == '0':
#     random_number = np.zeros((1,4,1,1))
#     random_number[0,0,0,0] = 1
# elif sign == '1':
#     random_number = np.zeros((1,4,1,1))
#     random_number[0,1,0,0] = 1
# elif sign == '2':
#     random_number = np.zeros((1,4,1,1))
#     random_number[0,2,0,0] = 1
# elif sign == '3':
#     random_number = np.zeros((1,4,1,1))
#     random_number[0,3,0,0] = 1
# elif sign == '4':
#     random_number = np.random.rand(1,4,1,1)
#     random_number = random_number/(random_number[0,0,0,0]+random_number[0,1,0,0]+random_number[0,2,0,0]+random_number[0,3,0,0])
# else:
#     print("Setinput: Error occur when getting the 0, 1, 0to1")

# # print(random_number)
# # print(netDeConv)
# add_item = netDeConv(Variable(torch.FloatTensor([random_number]).view(1,4,1,1)))
# # print(add_item)
# stoch_G = define_stochastic_G(latent_dim, input_dim, output_dim)
# print(stoch_G(input_feat,add_item).shape)


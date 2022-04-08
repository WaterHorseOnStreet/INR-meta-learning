import imp
from modules import MergeModule, CondInstanceNorm,CINResnetBlock,ResnetBlock_unbalance
import numpy as np
import torch
from networks import define_stochastic_G,define_InitialDeconv,MappingNetwork,StyleAdder,SyntheticNet,Discriminator
import os
from torch.autograd import Variable
import torchvision
from Kitti import Kitti
from BDD import BDDDataset

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# latent_dim = 100
# input_dim = 3
# output_dim = 10
# ConstNorm1 = CondInstanceNorm(input_dim,latent_dim)
# input_feat = torch.rand([2,3,224,224])
# latent_feat = torch.rand([2,100,224,224])
# # print(ConstNorm1(input_feat,latent_feat).shape)
# CRes = CINResnetBlock(input_dim, latent_dim)
# unbanl_resnet = ResnetBlock_unbalance(input_dim, latent_dim)

batch_size = 32
latent_dim = 512
label_dim = 512
# label_feat = torch.rand([batch_size,label_dim])
# latent_feat = torch.rand([batch_size,latent_dim])
# MNet = MappingNetwork(latent_dim,300,100,label_dim=label_dim)
# result = MNet(latent_feat,label_feat)
# print(result.shape)
channel = 18

# x = torch.rand([batch_size,100,latent_dim,latent_dim])

# SD = StyleAdder(latent_dim,2*latent_dim)
# result = SD(latent_feat)
# print(result.shape)
# result = result[:,0].view([-1, 2, x.shape[1]] + [1] * (len(x.shape) - 2))
# print(result.shape)
# print(result[:,0].shape)
# x = x*result[:,0] + result[:,1]

here = os.getcwd()

kitti_path = os.path.join(here,'../../../dataset/kitti/')

kitti = Kitti(root=kitti_path)
print(next(iter(kitti)))

cityscapes_path = os.path.join(here,'../../../dataset/cityscapes')
Cityscapes_dataset = torchvision.datasets.Cityscapes(root=cityscapes_path, split='train', mode='fine',target_type='semantic')
print(next(iter(Cityscapes_dataset)))

# channel_table = []
# fmap_base = 8192
# for i in range(1,10):
#     channel_table.append(min(int(fmap_base / (2.0 ** (i * 1))), latent_dim))
# print(channel_table)
# latent_feat = torch.rand([batch_size,channel,latent_dim])
# input_tensor = torch.rand([batch_size,latent_dim,4,4])
# SN = SyntheticNet(in_dim=latent_dim,out_channel=2*latent_dim,channel_table=channel_table)
# print(SN)
# result = SN(input_tensor,latent_feat)
# print(result.shape)

# channel_table = channel_table[::-1]
# print(channel_table)
# in_channels = 3
# DD = Discriminator(in_channels,result.shape[-1],latent_dim,channel_table=channel_table)
# result_D = DD(result)
# print(result_D.shape)



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


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
import torch.nn as nn

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
class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(self,
                 img_size=224,
                 patch_size=7,
                 stride=4,
                 in_chans=3,
                 embed_dim=768):
        super().__init__()
        img_size = (img_size,img_size)
        patch_size = (patch_size,patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[
            1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.norm(x)

        return x, H, W

class Attention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f'dim {dim} should be divided by ' \
                                     f'num_heads {num_heads}.'

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(
                dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads,
                              C // self.num_heads).permute(0, 2, 1,
                                                           3).contiguous()
        

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).contiguous().reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1).contiguous()
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads,
                                     C // self.num_heads).permute(
                                         2, 0, 3, 1, 4).contiguous()
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads,
                                    C // self.num_heads).permute(
                                        2, 0, 3, 1, 4).contiguous()

        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1).contiguous()) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        

        x = (attn @ v).transpose(1, 2).contiguous().reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class DWConv(nn.Module):

    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()

        return x

class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better
        # than dropout here
        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


NN = OverlapPatchEmbed()
sample = torch.rand([32,3,224,224])
result = NN(sample)
print(result[0].shape)
Atten = Block(dim=result[0].shape[-1],num_heads=8)
print(Atten)
ATT_result = Atten(result[0],result[1],result[2])
print(ATT_result.shape)

# batch_size = 32
# latent_dim = 512
# label_dim = 512
# # label_feat = torch.rand([batch_size,label_dim])
# # latent_feat = torch.rand([batch_size,latent_dim])
# # MNet = MappingNetwork(latent_dim,300,100,label_dim=label_dim)
# # result = MNet(latent_feat,label_feat)
# # print(result.shape)
# channel = 18

# x = torch.rand([batch_size,100,latent_dim,latent_dim])

# SD = StyleAdder(latent_dim,2*latent_dim)
# result = SD(latent_feat)
# print(result.shape)
# result = result[:,0].view([-1, 2, x.shape[1]] + [1] * (len(x.shape) - 2))
# print(result.shape)
# print(result[:,0].shape)
# x = x*result[:,0] + result[:,1]

# here = os.getcwd()

# kitti_path = os.path.join(here,'../../../dataset/kitti/')

# kitti = Kitti(root=kitti_path)
# print(next(iter(kitti)))

# cityscapes_path = os.path.join(here,'../../../dataset/cityscapes')
# Cityscapes_dataset = torchvision.datasets.Cityscapes(root=cityscapes_path, split='train', mode='fine',target_type='semantic')
# print(next(iter(Cityscapes_dataset)))

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


from math import gamma
from random import sample
from tabnanny import verbose
# from turtle import forward
import torch.nn.functional as F
import datetime
from collections import OrderedDict, Mapping
import torch
import matplotlib.pyplot as plt
from torch import dtype, nn
from torchmeta.modules import (MetaModule, MetaSequential)
from collections import OrderedDict
import numpy as np
from Siren_meta import BatchLinear, Siren, get_mgrid, SpatialT
import torchvision
from torchvision import transforms
import os
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import h5py
import math
from copy import deepcopy

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
def dict_to_gpu(ob):
    if isinstance(ob, Mapping):
        return {k: dict_to_gpu(v) for k, v in ob.items()}
    else:
        return ob.cuda()

def init_weights_normal(m):
    if hasattr(m, 'weight'):
        nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')

    if hasattr(m, 'bias') and m.bias is not None:
        with torch.no_grad():
            m.bias.data = torch.randn_like(m.bias)*1e-2

def hyper_weight_init(m, in_features_main_net):
    if hasattr(m, 'weight'):
        nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        m.weight.data = m.weight.data / 1.e2

    if hasattr(m, 'bias') and m.bias is not None:
        with torch.no_grad():
            m.bias.zero_()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def hyper_bias_init(m):
    if hasattr(m, 'weight'):
        nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        m.weight.data = m.weight.data / 1.e2

    if hasattr(m, 'bias') and m.bias is not None:
        with torch.no_grad():
            m.bias.zero_()

class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.sigmoid(input)

class FCLayer(MetaModule):
    def __init__(self, in_features, out_features, nonlinearity='relu', dropout=0.0):
        super().__init__()
        self.net = [BatchLinear(in_features, out_features)]
        if nonlinearity == 'relu':
            self.net.append(nn.ReLU(inplace=True))
        elif nonlinearity == 'leaky_relu':
            self.net.append(nn.LeakyReLU(0.2, inplace=True))
        elif nonlinearity == 'silu':
            self.net.append(Swish())

        self.dropout = dropout
        self.dropout_layer = torch.nn.Dropout(dropout, False)

        self.net = MetaSequential(*self.net)

    def forward(self, input, params=None):
        output = self.net(input, params=self.get_subdict(params, 'net'))
        if self.dropout != 0.0:
            output = self.dropout_layer(output)

        return output

class FCBlock(MetaModule):
    def __init__(self,
                 hidden_ch,
                 num_hidden_layers,
                 in_features,
                 out_features,
                 final_bias=True,
                 outermost_linear=False,
                 nonlinearity='relu',
                 dropout=0.0):
        super().__init__()

        self.net = []
        self.net.append(FCLayer(in_features=in_features, out_features=hidden_ch, nonlinearity=nonlinearity, dropout=dropout))

        for i in range(num_hidden_layers):
            self.net.append(FCLayer(in_features=hidden_ch, out_features=hidden_ch, nonlinearity=nonlinearity, dropout=dropout))

        if outermost_linear:
            self.net.append(BatchLinear(in_features=hidden_ch, out_features=out_features, bias=final_bias))
        else:
            self.net.append(FCLayer(in_features=hidden_ch, out_features=out_features, nonlinearity=nonlinearity))

        self.net = MetaSequential(*self.net)
        self.net.apply(init_weights_normal)

    def forward(self, input, params=None):
        return self.net(input, params=self.get_subdict(params, 'net'))



class GraphFeatureMap_Manifold(nn.Module):    
    def __init__(self, in_features, out_features, nonlinear="LeakySoftPlus",bias=True):        
        super(GraphFeatureMap_Manifold, self).__init__()        
        assert nonlinear in ['LeakyReLU','LeakySoftPlus']        
        self.nonlinear = nonlinear        
        self.in_features = in_features        
        self.out_features = out_features        
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))        
        self.register_parameter('weight',self.weight)        
        if bias:            
            self.bias = nn.Parameter(torch.FloatTensor(out_features))            
            self.register_parameter('bias',self.bias)        
        else:            
            self.register_parameter('bias', None)        
            self.reset_parameters()
    def reset_parameters(self):        
        stdv = 1. / math.sqrt(self.weight.size(1))        
        self.weight.data.uniform_(-stdv, stdv)        
        if self.bias is not None:            
            self.bias.data.uniform_(-stdv, stdv)
    def forward(self, input):        
        output = torch.mm(input, self.weight)        
        if self.bias is not None:            
            output = output + self.bias        
        else:            
            output = output        
        if self.nonlinear == "LeakyReLU":            
            output = F.leaky_relu(output,negative_slope=1)        
        if self.nonlinear == "LeakySoftPlus":            
            output = F.softplus(output)+output;        
        if self.bias is not None:            
            output = output - self.bias        
        else:            
            output = output      
        #pseudo_inverse = torch.mm( torch.t(self.weight), torch.inverse( torch.mm(self.weight,torch.t(self.weight)) )  )  
        output = torch.mm(output,torch.linalg.pinv(self.weight))   
        return output

class Hyper_Net_Embedd(nn.Module):
    def __init__(self, embedd_size, hyper_in_features, hyper_hidden_layers, hyper_hidden_features, hypo_module, GM, linear=False):
        '''

        Args:
            hyper_in_features: In features of hypernetwork
            hyper_hidden_layers: Number of hidden layers in hypernetwork
            hyper_hidden_features: Number of hidden units in hypernetwork
            hypo_module: MetaModule. The module whose parameters are predicted.
        '''
        super().__init__()

        hypo_parameters = hypo_module.meta_named_parameters()

        self.names = []
        self.nets = nn.ModuleList()
        self.param_shapes = []
        self.embedd_size = embedd_size
        self.embedd = []
        self.GM = GM

        # with h5py.File('./model_saves/final_latents.h5', 'r') as f:
        #     keys = list(f.keys())
        #     data = f[keys[0]]
        #     data = torch.tensor(data)
        #     print(data.shape)
        #     self.embedd.weight = torch.nn.Parameter(data)

        for name, param in hypo_parameters:
            print(name)
            self.names.append(name)
            self.param_shapes.append(param.size())

            if linear:
                hn = BatchLinear(in_features=hyper_in_features,
                                         out_features=int(torch.prod(torch.tensor(param.size()))),
                                         bias=True)
                if 'weight' in name:
                    hn.apply(lambda m: hyper_weight_init(m, param.size()[-1]))
                elif 'bias' in name:
                    hn.apply(lambda m: hyper_bias_init(m))
            else:
                hn = FCBlock(in_features=hyper_in_features, out_features=int(torch.prod(torch.tensor(param.size()))),
                                     num_hidden_layers=hyper_hidden_layers, hidden_ch=hyper_hidden_features,
                                     outermost_linear=True, nonlinearity='relu')
                if 'weight' in name:
                    hn.net[-1].apply(lambda m: hyper_weight_init(m, param.size()[-1]))
                elif 'bias' in name:
                    hn.net[-1].apply(lambda m: hyper_bias_init(m))
            self.nets.append(hn)

        self.meshgrid = get_mgrid(sidelen=28)
        self.init_embedd(GM)
    def forward(self, z):
        '''
        Args:
            z: Embedding. Input to hypernetwork. Could be output of "Autodecoder" (see above)

        Returns:
            params: OrderedDict. Can be directly passed as the "params" parameter of a MetaModule.
        '''
        params = OrderedDict()
        z = torch.from_numpy(self.embedd[z]).cuda()
        for name, net, param_shape in zip(self.names, self.nets, self.param_shapes):
            batch_param_shape = (-1,) + param_shape
            params[name] = net(z).reshape(batch_param_shape)
        return z, params

    def set_embedd(self,GM,idx):
        params = []
        for param in GM.parameters():
            params.append(param)

        self.embedd[idx] = params[0].reshape(1,64).detach().cpu().numpy()

    def set_GM(self,GM,idx):
        #print(self.embedd[:10])
        for name,param in GM.named_parameters():
            if 'weight' in name:
                param.data = torch.nn.parameter.Parameter(torch.from_numpy(self.embedd[idx]).cuda().reshape(2,32),requires_grad=True)

    def init_embedd(self,GM):
        params = []
        for param in GM.parameters():
            params.append(param)

        for i in range(self.embedd_size):
            self.embedd.append(params[0].reshape(1,64).detach().cpu().numpy())


    def get_embedd(self,index):
        return self.embedd[index]

    def embedd2inr(self,embedd):
        params = OrderedDict()
        z = embedd
        for name, net, param_shape in zip(self.names, self.nets, self.param_shapes):
            batch_param_shape = (-1,) + param_shape
            params[name] = net(z).reshape(batch_param_shape)
        return params

class CIFAR10():
    def __init__(self, data_path):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.dataset = torchvision.datasets.CIFAR10(root=data_path, train=True,
                                                download=False, transform=transform)
        
        self.meshgrid = get_mgrid(sidelen=32)
    
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, item):
        img, label = self.dataset[item]
        img_flat = img.permute(1,2,0).view(-1, 3)
        return {'context':{'x':self.meshgrid, 'y':img_flat}, 
                'query':{'x':self.meshgrid, 'y':img_flat},
                'label':label}


class MNIST():
    def __init__(self, data_path):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5), (0.5))])

        dataset = torchvision.datasets.MNIST(root=data_path, train=True,
                                                download=False, transform=transform)
        self.dataset = dataset
        self.meshgrid = get_mgrid(sidelen=28)
        idx = dataset.targets==3
        idx2 = dataset.targets==8
        idx3 = dataset.targets==4
        idx += idx2
        idx += idx3
        self.dataset.data = dataset.data[idx]
        self.dataset.targets = dataset.targets[idx]
    
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, item):
        img, label = self.dataset[item]
        img_flat = img.view(-1, 1)
        return {'context':{'x':self.meshgrid, 'y':img_flat}, 
                'label':label,
                'index':item}


def l2_loss(prediction, gt):
    return ((prediction - gt)**2).mean() + 1e-6 


def lin2img(tensor):
    batch_size, num_samples, channels = tensor.shape
    sidelen = np.sqrt(num_samples).astype(int)
    return tensor.view(batch_size, sidelen, sidelen, channels).squeeze(-1)

    
def plot_sample_image(img_batch, ax):
    img = lin2img(img_batch)[0].detach().cpu().numpy()
    img += 1
    img /= 2.
    img = np.clip(img, 0., 1.)
    #print(img)
    ax.set_axis_off()
    ax.imshow(img,cmap='Greys_r')

def cdist(embedd_dict):
    cross_dist = 0.0
    count = 0
    for key, value in embedd_dict.items():
        embedd = torch.stack(value,dim=0).cuda()
        cross_dist += dist(embedd, embedd)
        count += 1

    return cross_dist/count

def dist(input1,input2):
    dist = 0.0
    for i in range(input1.shape[0]-1):
        in1 = input1[i]
        for j in range(i+1,input1.shape[0]):
            in2 = input1[j]
            dist += torch.norm(in1 - in2)

    return dist/(input1.shape[0]*(input1.shape[0]-1))

def MNIST_test():
    batch_size = 32
    here = os.path.join(os.path.dirname(__file__)) 
    data_path = os.path.join(here, '../../dataset/')
    dataset = MNIST(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4,shuffle=True)

    hyper_in_features = 64
    hyper_hidden_layers = 1
    hyper_hidden_features = 1024

    in_features=2
    hidden_features=512
    hidden_layers=0
    out_features=1

    img_siren = Siren(in_features=in_features, hidden_features=hidden_features, 
                        hidden_layers=hidden_layers, out_features=out_features, outermost_linear=True)

    GM = GraphFeatureMap_Manifold(in_features=2,out_features=hyper_in_features //2, bias=False).cuda()
    HyperNetEmbedd = Hyper_Net_Embedd(len(dataset),hyper_in_features,hyper_hidden_layers,hyper_hidden_features,img_siren,GM)
    # print(HyperNetEmbedd.get_embedd(5))
    HyperNetEmbedd.cuda()

    optim = torch.optim.Adam(lr=0.0001, params=HyperNetEmbedd.parameters())
    train_scheduler = torch.optim.lr_scheduler.StepLR(optim,10,gamma=0.5)

    optim2 = torch.optim.Adam(lr=0.0001, params=GM.parameters())
    train_scheduler2 = torch.optim.lr_scheduler.StepLR(optim2,10,gamma=0.5)
    steps_til_summary = 100
    log_dir = os.path.join(here, './output')
    check_point_dir = os.path.join(here, './checkpoint8')
    writer = SummaryWriter(log_dir=log_dir)
    iteration = 0

    #torch.autograd.set_detect_anomaly(True)
    for epoch in range(1,50):

        for step, sample in enumerate(dataloader):

            sample = dict_to_gpu(sample)
            feats = sample['index']
            labels = sample['label']

            loss = 0.0

            loss_reconstruction = 0.0

            if not feats.numel():
                print('continue')
                continue

            # for param in HyperNetEmbedd.parameters():
            #     param.requires_grad = False
            for idx in range(len(feats)):
                loss_GM = 0.0
                feat = feats[idx]
                HyperNetEmbedd.set_GM(GM,feat)
                # for param in GM.parameters():
                #     print(param)
                embedding, model_output = HyperNetEmbedd(feat.unsqueeze(0))
                param = model_output
                input = GM(sample['context']['x'][idx])
                output = img_siren(input,params=param)
                loss_GM = l2_loss(output,sample['context']['y'][idx]) 
                optim2.zero_grad()
                loss_GM.backward()
                optim2.step() 
                HyperNetEmbedd.set_embedd(GM,feat)

            # for param in HyperNetEmbedd.parameters():
            #     param.requires_grad = True


            # for param in GM.parameters():
            #     param.requires_grad = False

            for idx in range(len(feats)):
                loss_GM = 0.0
                feat = feats[idx]

                embedding, model_output = HyperNetEmbedd(feat.unsqueeze(0))
                param = model_output
                input = GM(sample['context']['x'][idx])
                output = img_siren(input,params=param)
                loss_reconstruction += l2_loss(output,sample['context']['y'][idx]) 

            loss = loss_reconstruction

            if not step % steps_til_summary:
                print('in epoch {}, step {}, the loss is {}'.format(epoch, step, loss/(3*batch_size)))  
                # print('the dist loss is {}'.format(loss_dist))
                # print('the reconstruction loss is {}'.format(loss_reconstruction))
                

            # index_start = index_end

            optim.zero_grad()
            loss.backward()
            optim.step()     

            # for param in GM.parameters():
            #     param.requires_grad = True

            iteration += 1

            writer.add_scalar('Loss/train', loss, iteration)
            

        # meshgrid = get_mgrid(sidelen=28)
        # meshgrid = meshgrid.unsqueeze(0)
        # meshgrid = meshgrid.cuda()
        # with torch.no_grad():
        #     feats = np.arange(0,5,1)
        #     #print(feats)
        #     feats = torch.LongTensor(feats).unsqueeze(0)
        #     feats = feats.cuda()

        #     for idx in range(feats.shape[1]):
        #         fig, axes = plt.subplots(1,1)
        #         feat = feats[0][idx]
        #         embedding, model_output = HyperNetEmbedd(feat.unsqueeze(0))
        #         param = model_output
        #         output = img_siren(sample['context']['x'][idx],params=param)
        #         output = img_siren(meshgrid,params=param)
        #         plot_sample_image(output, ax=axes)
        #         axes.set_title(str(idx), fontsize=25)
        #         path = os.path.join(here, './{}_{}.png'.format(epoch,idx))
        #         plt.savefig(path,format='png')
        #         plt.cla()
        #     plt.close('all')
        if(train_scheduler.get_last_lr()[0] > 1e-6):
            train_scheduler.step()
        
        train_scheduler.step()
        print('the learning rate is {}'.format(train_scheduler.get_last_lr()))

    state_dict = HyperNetEmbedd.state_dict()
    for k, v in state_dict.items():
        state_dict[k] = v.cpu()

    torch.save({'model_state_dict':state_dict},check_point_dir)

def main():

    # if torch.cuda.is_available():
    #     device = torch.device('cuda:{}'.format(cuda_device[0]))

    hyper_in_features = 128
    hyper_hidden_layers = 4
    hyper_hidden_features = 500

    in_features=2
    hidden_features=128
    hidden_layers=1
    out_features=1

    img_siren = Siren(in_features=in_features, hidden_features=hidden_features, 
                        hidden_layers=hidden_layers, out_features=out_features, outermost_linear=True).cuda()

    hyper_network = Hyper_Net_Embedd(hyper_in_features,hyper_hidden_layers,hyper_hidden_features,img_siren).cuda()

    

    use_gaussian = False
    use_embedd = False
    here = os.path.join(os.path.dirname(__file__)) 
    data_path = os.path.join(here, '../../dataset/cifar10/')

    dataset = CIFAR10(data_path)
    dataloader = DataLoader(dataset, batch_size=32, num_workers=0,shuffle=False)

    HyperNetEmbedd = Hyper_Net_Embedd(len(dataset),hyper_in_features,hyper_hidden_layers,hyper_hidden_features,img_siren).cuda()

    optim = torch.optim.Adam(lr=1e-4, params=HyperNetEmbedd.parameters())
    train_scheduler = torch.optim.lr_scheduler.StepLR(optim,5,gamma=0.1)
    steps_til_summary = 100
    log_dir = os.path.join(here, './output')
    writer = SummaryWriter(log_dir=log_dir)
    iteration = 0

    if use_embedd:
        for epoch in range(1,5):
            index_start = 0
            index_end = 0
            for step, sample in enumerate(dataloader):
                index_end = index_start + len(sample['label'])
                #print('from {} to {}'.format(index_start,index_end))
                index_end = len(dataset) if index_end > len(dataset) - 1 else index_end
                sample = dict_to_gpu(sample)
                feats = np.arange(index_start,index_end,1)
                #print(feats)
                #print(feats)
                feats = torch.LongTensor(feats).unsqueeze(0)
                feats = feats.cuda()
                loss = 0.0
                #print(step)
                if not feats.numel():
                    print('continue')
                    continue
                for idx in range(feats.shape[1]):
                    feat = feats[0][idx]
                    model_output = HyperNetEmbedd(feat.unsqueeze(0))
                    param = model_output
                    output = img_siren(sample['context']['x'][idx],params=param)
                    loss += l2_loss(output,sample['context']['y'][idx]) 

                if not step % steps_til_summary:
                    print('in epoch {}, step {}, the loss is {}'.format(epoch, step, loss))  

                index_start = index_end
                optim.zero_grad()
                loss.backward()
                optim.step()      

                iteration += 1

                writer.add_scalar('Loss/train', loss, iteration)

            meshgrid = get_mgrid(sidelen=32)
            meshgrid = meshgrid.unsqueeze(0)
            meshgrid = meshgrid.cuda()
            with torch.no_grad():
                feats = np.arange(0,5,1)
                #print(feats)
                feats = torch.LongTensor(feats).unsqueeze(0)
                feats = feats.cuda()

                for idx in range(feats.shape[1]):
                    fig, axes = plt.subplots(1,1)
                    feat = feats[0][idx]
                    model_output = HyperNetEmbedd(feat.unsqueeze(0))
                    param = model_output
                    output = img_siren(sample['context']['x'][idx],params=param)
                    output = img_siren(meshgrid,params=param)
                    plot_sample_image(output, ax=axes)
                    axes.set_title(str(idx), fontsize=25)
                    path = os.path.join(here, './{}_{}.png'.format(epoch,idx))
                    plt.savefig(path,format='png')
                    plt.cla()
                plt.close('all')

            train_scheduler.step()

        torch.save({'model_state_dict':hyper_network.state_dict()})
    if use_gaussian:
        # features random sampled from a normal distribution
        mean = np.arange(10,20)
        std = np.ones(10)
        # features = torch.normal(mean, std, (10,len(dataloader),hyper_in_features)) 

        log_dir = os.path.join(here, './output')
        writer = SummaryWriter(log_dir=log_dir)

        for epoch in range(1, 25):
            
            for step, sample in enumerate(dataloader):
                feats = []
                for label in sample['label']:
                    feats.append(torch.normal(mean[label], std[label], (1,hyper_in_features)))

                feats = torch.stack(feats, dim=0)
                sample = dict_to_gpu(sample)
                loss = 0.0
                feats = feats.cuda()
                for idx, feat in enumerate(feats):
                    model_output = hyper_network(feat)
                    param = model_output
                    output = img_siren(sample['context']['x'][idx],params=param)
                    loss += l2_loss(output,sample['context']['y'][idx])

                if not step % steps_til_summary:
                    print('in epoch {}, step {}, the loss is {}'.format(epoch, step, loss))

                optim.zero_grad()
                loss.backward()
                optim.step()

                iteration += 1

                writer.add_scalar('Loss/train', loss, iteration)


            meshgrid = get_mgrid(sidelen=32)
            meshgrid = meshgrid.unsqueeze(0)
            meshgrid = meshgrid.cuda()
            with torch.no_grad():
                val = []
                for i in range(10):
                    val.append(torch.normal(mean[i], std[i], (1,hyper_in_features)))
                val = torch.stack(val, dim=0)
                val = val.cuda()

                for idx, feat in enumerate(val):
                    fig, axes = plt.subplots(1,1)
                    model_output = hyper_network(feat)
                    param = model_output
                    output = img_siren(meshgrid,params=param)
                    plot_sample_image(output, ax=axes)
                    axes.set_title(str(idx), fontsize=25)
                    path = os.path.join(here, './{}_{}.png'.format(epoch,idx))
                    plt.savefig(path,format='png')
                    plt.cla()
                plt.close('all')

            train_scheduler.step()

        torch.save({'model_state_dict':hyper_network.state_dict()})

if __name__ == '__main__':
    # device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    MNIST_test()



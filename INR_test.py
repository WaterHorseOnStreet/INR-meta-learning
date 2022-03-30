from math import gamma
from random import sample
# from turtle import forward
import torch.nn.functional as F
import datetime
import torch
import matplotlib.pyplot as plt
from torch import dtype, nn
from torchmeta.modules import (MetaModule, MetaSequential)
from collections import OrderedDict
import numpy as np
from Siren_meta import BatchLinear, Siren, get_mgrid, dict_to_gpu
import torchvision
from torchvision import transforms
import os
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

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

class Hyper_Net_Embedd(nn.Module):
    def __init__(self, embedd_size, hyper_in_features, hyper_hidden_layers, hyper_hidden_features, hypo_module, linear=False):
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
        self.embedd = nn.Embedding(embedd_size,hyper_in_features)
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

    def forward(self, z):
        '''
        Args:
            z: Embedding. Input to hypernetwork. Could be output of "Autodecoder" (see above)

        Returns:
            params: OrderedDict. Can be directly passed as the "params" parameter of a MetaModule.
        '''
        params = OrderedDict()
        z = self.embedd(z)
        for name, net, param_shape in zip(self.names, self.nets, self.param_shapes):
            batch_param_shape = (-1,) + param_shape
            params[name] = net(z).reshape(batch_param_shape)
        return params


    def get_embedd(self,index):
        return self.embedd(index)

class HyperNetwork(nn.Module):
    def __init__(self, hyper_in_features, hyper_hidden_layers, hyper_hidden_features, hypo_module, linear=False):
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

    def forward(self, z):
        '''
        Args:
            z: Embedding. Input to hypernetwork. Could be output of "Autodecoder" (see above)

        Returns:
            params: OrderedDict. Can be directly passed as the "params" parameter of a MetaModule.
        '''
        params = OrderedDict()
        for name, net, param_shape in zip(self.names, self.nets, self.param_shapes):
            batch_param_shape = (-1,) + param_shape
            params[name] = net(z).reshape(batch_param_shape)
        return params


class LowRankHyperNetwork(nn.Module):
    def __init__(self, hyper_in_features, hyper_hidden_layers, hyper_hidden_features, hypo_module, linear=False,
                 rank=10):
        '''

        Args:
            hyper_in_features: In features of hypernetwork
            hyper_hidden_layers: Number of hidden layers in hypernetwork
            hyper_hidden_features: Number of hidden units in hypernetwork
            hypo_module: MetaModule. The module whose parameters are predicted.
        '''
        super().__init__()

        self.hypo_parameters = dict(hypo_module.meta_named_parameters())
        self.representation_dim = 0

        self.rank = rank
        self.names = []
        self.nets = nn.ModuleList()
        self.param_shapes = []
        for name, param in self.hypo_parameters.items():
            self.names.append(name)
            self.param_shapes.append(param.size())

            out_features = int(torch.prod(torch.tensor(param.size()))) if 'bias' in name else param.shape[0]*rank + param.shape[1]*rank
            self.representation_dim += out_features

            hn = FCBlock(in_features=hyper_in_features, out_features=out_features,
                                 num_hidden_layers=hyper_hidden_layers, hidden_ch=hyper_hidden_features,
                                 outermost_linear=True)
            self.nets.append(hn)

    def forward(self, z):
        '''
        Args:
            z: Embedding. Input to hypernetwork. Could be output of "Autodecoder" (see above)

        Returns:
            params: OrderedDict. Can be directly passed as the "params" parameter of a MetaModule.
        '''
        params = OrderedDict()
        representation = []
        for name, net, param_shape in zip(self.names, self.nets, self.param_shapes):
            low_rank_params = net(z)
            representation.append(low_rank_params.detach())

            if 'bias' in name:
                batch_param_shape = (-1,) + param_shape
                params[name] = low_rank_params.reshape(batch_param_shape)
            else:
                a = low_rank_params[:, :self.rank*param_shape[0]].view(-1, param_shape[0], self.rank)
                b = low_rank_params[:, self.rank*param_shape[0]:].view(-1, self.rank, param_shape[1])
                low_rank_w = a.matmul(b)
                params[name] = self.hypo_parameters[name] * torch.sigmoid(low_rank_w)

        representations = representation
        representation = torch.cat(representation, dim=-1).cuda()
        return {'params':params, 'representation':representation, 'representations': representations}

    def gen_params(self, representation):
        params = OrderedDict()
        start = 0

        for name, net, param_shape in zip(self.names, self.nets, self.param_shapes):
            if 'bias' in name:
                nelem = np.prod(param_shape)
            else:
                nelem = param_shape[0] * self.rank + param_shape[1] * self.rank

            low_rank_params = representation[:, start:start+nelem]

            if 'bias' in name:
                batch_param_shape = (-1,) + param_shape
                params[name] = low_rank_params.reshape(batch_param_shape)
            else:
                a = low_rank_params[:, :self.rank*param_shape[0]].view(-1, param_shape[0], self.rank)
                b = low_rank_params[:, self.rank*param_shape[0]:].view(-1, self.rank, param_shape[1])
                low_rank_w = a.matmul(b)
                params[name] = self.hypo_parameters[name] * torch.sigmoid(low_rank_w)

            start = start + nelem

        return {'params':params, 'representation':representation}

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
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.dataset = torchvision.datasets.MNIST(root=data_path, train=True,
                                                download=False, transform=transform)
        
        self.meshgrid = get_mgrid(sidelen=28)
    
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, item):
        img, label = self.dataset[item]
        img_flat = img.view(-1, 1)
        return {'context':{'x':self.meshgrid, 'y':img_flat}, 
                'query':{'x':self.meshgrid, 'y':img_flat},
                'label':label}


def l2_loss(prediction, gt):
    return ((prediction - gt)**2).mean()


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
    ax.imshow(img)

def MNIST_test():

    here = os.path.join(os.path.dirname(__file__)) 
    data_path = os.path.join(here, '../../dataset/')
    dataset = MNIST(data_path)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0,shuffle=False)
    sample = next(iter(dataloader))
    print(sample)


def main():

    # if torch.cuda.is_available():
    #     device = torch.device('cuda:{}'.format(cuda_device[0]))

    hyper_in_features = 100
    hyper_hidden_layers = 3
    hyper_hidden_features = 300

    in_features=2
    hidden_features=128
    hidden_layers=3
    out_features=3

    img_siren = Siren(in_features=in_features, hidden_features=hidden_features, 
                        hidden_layers=hidden_layers, out_features=out_features, outermost_linear=True).cuda()

    hyper_network = HyperNetwork(hyper_in_features,hyper_hidden_layers,hyper_hidden_features,img_siren).cuda()

    

    use_gaussian = False
    use_embedd = False
    here = os.path.join(os.path.dirname(__file__)) 
    data_path = os.path.join(here, '../../dataset/cifar10/')

    dataset = CIFAR10(data_path)
    dataloader = DataLoader(dataset, batch_size=32, num_workers=0,shuffle=False)

    HyperNetEmbedd = Hyper_Net_Embedd(len(dataset),hyper_in_features,hyper_hidden_layers,hyper_hidden_features,img_siren).cuda()
    embedd = nn.Embedding(len(dataset),hyper_in_features).cuda()

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
    MNIST_test()
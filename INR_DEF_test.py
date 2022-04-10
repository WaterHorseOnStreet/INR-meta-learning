from fileinput import close
from http.client import ImproperConnectionState
import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
from INR_DEF import Hyper_Net_Embedd, MNIST, plot_sample_image
from Siren_meta import BatchLinear, Siren, get_mgrid, dict_to_gpu
from torch.utils.data import DataLoader, Dataset
import os
from sklearn.datasets import fetch_mldata
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd


def dist(input1,center):
    avg_dist = 0.0
    dist_arr = []
    for i in range(input1.shape[0]):
        dist = np.linalg.norm(input1[i] - center)
        avg_dist += dist
        dist_arr.append(dist)
    avg_dist /= input1.shape[0]
    return avg_dist, dist_arr

if __name__ == "__main__":
    print("Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset.")

    here = os.path.join(os.path.dirname(__file__)) 
    data_path = os.path.join(here, '../../dataset/')
    dataset = MNIST(data_path)
    dataloader = DataLoader(dataset, batch_size=32, num_workers=0,shuffle=False)

    hyper_in_features = 128
    hyper_hidden_layers = 3
    hyper_hidden_features = 1024

    in_features=2
    hidden_features=512
    hidden_layers=0
    out_features=1


    img_siren = Siren(in_features=in_features, hidden_features=hidden_features, 
                        hidden_layers=hidden_layers, out_features=out_features, outermost_linear=True)
    img_siren2 = Siren(in_features=in_features, hidden_features=hidden_features, 
                        hidden_layers=hidden_layers, out_features=out_features, outermost_linear=True)

    HyperNetEmbedd = Hyper_Net_Embedd(len(dataset),hyper_in_features,hyper_hidden_layers,hyper_hidden_features,img_siren,img_siren2).cuda()

    checkpoint = torch.load('./checkpointTwoPart.pth')
    HyperNetEmbedd.load_state_dict(checkpoint['model_state_dict'])


    # X = []
    # labels = []
    # for i in range(len(dataset)):
    #     i_t = torch.tensor(i)
    #     i_t_c = i_t.cuda()
    #     embedd = HyperNetEmbedd.get_embedd(index=i_t_c)
    #     embedd = embedd.detach().cpu().numpy()
    #     X.append(embedd)
    #     index = dataset[i_t]
    #     labels.append(index['label'])

    # X = np.asarray(X)
    # print(X.shape)

    # labels = [int(i) for i in labels]    

    # pca = PCA(n_components=90)
    # X_pca = pca.fit_transform(X) # randomly sample data to run quickly

    # n_select = 10000 # reduce dimensionality with t-sne
    # tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=2000, learning_rate=100)
    # tsne_results = tsne.fit_transform(X_pca)# visualize
    # df_tsne = pd.DataFrame(tsne_results, columns=['comp1', 'comp2'])

    # S1 = df_tsne['comp1']
    # S2 = df_tsne['comp2']

    # fig, axes = plt.subplots(1,1)

    # axes.scatter(S1, S2,c=labels,cmap=plt.cm.cool)

    # path = os.path.join(here, './1.png')
    # plt.savefig(path,format='png')
    # plt.cla()

    # indices_3 = []
    # indices_4 = []
    # indices_8 = []
    # for i, v in enumerate(labels):
    #     if v == 3:
    #         indices_3.append(i)
    #     elif v == 4:
    #         indices_4.append(i)
    #     elif v == 8:
    #         indices_8.append(i)
    #     else:
    #         continue

    # print(indices_3[:10])
    # print(indices_4[:10])
    # print(indices_8[:10])

    # embedd_3 = np.squeeze(X[indices_3,:])
    # embedd_4 = np.squeeze(X[indices_4,:])
    # embedd_8 = np.squeeze(X[indices_8,:])
    # # print(embedd_3[:2])
    # # print(embedd_4[:2])
    # # print(embedd_8[:2])


    # center_3 = np.mean(embedd_3,axis=0)
    # center_4 = np.mean(embedd_4,axis=0)
    # center_8 = np.mean(embedd_8,axis=0)
    # print(center_3)
    # print(center_4)
    # print(center_8)

    # print('var of 3 is {}'.format(np.var(embedd_3)))
    # print('var of 4 is {}'.format(np.var(embedd_4)))
    # print('var of 8 is {}'.format(np.var(embedd_8)))

    # avg_dist_3, dist_arr_3 = dist(embedd_3,center_3)
    # avg_dist_4, dist_arr_4 = dist(embedd_4,center_4)
    # avg_dist_8, dist_arr_8 = dist(embedd_8,center_8)

    # print('avg radius of 3 is {}'.format(avg_dist_3))
    # print('avg radius of 4 is {}'.format(avg_dist_4))
    # print('avg radius of 8 is {}'.format(avg_dist_8))

    # bins = np.arange(0,14)
    # print('distance histogram of 3 is {}'.format(np.histogram(dist_arr_3,bins= bins)))

    # print('distance histogram of 4 is {}'.format(np.histogram(dist_arr_4,bins= bins)))

    # print('distance histogram of 8 is {}'.format(np.histogram(dist_arr_8,bins= bins)))

    # # print('distance histogram of all is {}'.format(np.histogram(dist_arr_8+dist_arr_3+dist_arr_4,bins= bins)))

    # print('distance 3 <----> 4 is {}'.format(np.linalg.norm(center_3 - center_4)))
    # print('distance 3 <----> 8 is {}'.format(np.linalg.norm(center_3 - center_8)))
    # print('distance 4 <----> 8 is {}'.format(np.linalg.norm(center_4 - center_8)))

    # inter = np.random.randint(0,embedd_3.shape[0],2)

    # meshgrid = get_mgrid(sidelen=28)
    # meshgrid = meshgrid.unsqueeze(0)
    # meshgrid = meshgrid.cuda()
    # num = 20
    # with torch.no_grad():
    #     embedd_3 = torch.tensor(embedd_3)
    #     embedd_4 = torch.tensor(embedd_4)
    #     embedd_8 = torch.tensor(embedd_8)

    #     a = embedd_3[inter[0]]
    #     b = embedd_3[inter[1]]
    #     a = a.unsqueeze(0).cuda()
    #     b = b.unsqueeze(0).cuda()
    #     s = torch.tensor(np.linspace(0,1,num)).cuda()
    #     for idx in range(num):
    #         fig, axes = plt.subplots(1,1)
    #         model_output,model_output2 = HyperNetEmbedd.embedd2inr(a+s[idx]*(b-a))
    #         param = model_output
    #         param2 = model_output2
    #         output = img_siren(meshgrid,params=param)
    #         # output2 = img_siren2(meshgrid,params=param2)
    #         # output += output2
    #         plot_sample_image(output, ax=axes)
    #         axes.set_title(str(idx), fontsize=25)
    #         path = os.path.join(here, './inference_twostage/{}.png'.format(idx))
    #         plt.savefig(path,format='png')
    #         plt.cla()
    #     plt.close('all')

    meshgrid = get_mgrid(sidelen=28)
    meshgrid = meshgrid.unsqueeze(0)
    meshgrid = meshgrid.cuda()

    with torch.no_grad():
        feats = np.random.randint(0,len(dataset),20)
        #print(feats)
        feats = torch.LongTensor(feats).unsqueeze(0)
        feats = feats.cuda()

        for idx in range(feats.shape[1]):
            fig, axes = plt.subplots(1,1)
            feat = feats[0][idx]
            embedding, model_output,model_output2 = HyperNetEmbedd(feat.unsqueeze(0))
            param = model_output
            param2 = model_output2
            output = img_siren(meshgrid,params=param)
            output1 = img_siren2(meshgrid,params=param2)
            # output += output1
            plot_sample_image(output1, ax=axes)
            axes.set_title(str(idx), fontsize=25)
            path = os.path.join(here, './second_channel/{}.png'.format(idx))
            plt.savefig(path,format='png')
            plt.cla()
        plt.close('all')
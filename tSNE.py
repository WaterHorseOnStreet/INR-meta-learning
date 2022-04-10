from fileinput import close
from http.client import ImproperConnectionState
import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
from INR_test import Hyper_Net_Embedd, MNIST, plot_sample_image
from Siren_meta import BatchLinear, Siren, get_mgrid, dict_to_gpu
from torch.utils.data import DataLoader, Dataset
import os

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import pandas as pd


def Hbeta_torch(D, beta=1.0):
    P = torch.exp(-D.clone() * beta)

    sumP = torch.sum(P)

    H = torch.log(sumP) + beta * torch.sum(D * P) / sumP
    P = P / sumP

    return H, P

def x2p_torch(X, tol=1e-5, perplexity=30.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape

    sum_X = torch.sum(X*X, 1)
    D = torch.add(torch.add(-2 * torch.mm(X, X.t()), sum_X).t(), sum_X)

    P = torch.zeros(n, n)
    beta = torch.ones(n, 1)
    logU = torch.log(torch.tensor([perplexity]))
    n_list = [i for i in range(n)]

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        # there may be something wrong with this setting None
        betamin = None
        betamax = None
        Di = D[i, n_list[0:i]+n_list[i+1:n]]

        (H, thisP) = Hbeta_torch(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while torch.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].clone()
                if betamax is None:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].clone()
                if betamin is None:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta_torch(Di, beta[i])

            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, n_list[0:i]+n_list[i+1:n]] = thisP

    # Return final P-matrix
    return P

def pca_torch(X, no_dims=50):
    print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    X = X - torch.mean(X, 0)

    (l, M) = torch.eig(torch.mm(X.t(), X), True)
    # split M real
    # this part may be some difference for complex eigenvalue
    # but complex eignevalue is meanless here, so they are replaced by their real part
    i = 0
    while i < d:
        if l[i, 1] != 0:
            M[:, i+1] = M[:, i]
            i += 2
        else:
            i += 1

    Y = torch.mm(X, M[:, 0:no_dims])
    return Y

def tsne(X, no_dims=2, initial_dims=50, perplexity=30.0):
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should not have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    X = pca_torch(X, initial_dims)
    (n, d) = X.shape
    max_iter = 1000
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = torch.randn(n, no_dims)
    dY = torch.zeros(n, no_dims)
    iY = torch.zeros(n, no_dims)
    gains = torch.ones(n, no_dims)

    # Compute P-values
    P = x2p_torch(X, 1e-5, perplexity)
    P = P + P.t()
    P = P / torch.sum(P)
    P = P * 4.    # early exaggeration
    print("get P shape", P.shape)
    P = torch.max(P, torch.tensor([1e-21]))

    # Run iterations
    for iter in range(max_iter):

        # Compute pairwise affinities
        sum_Y = torch.sum(Y*Y, 1)
        num = -2. * torch.mm(Y, Y.t())
        num = 1. / (1. + torch.add(torch.add(num, sum_Y).t(), sum_Y))
        num[range(n), range(n)] = 0.
        Q = num / torch.sum(num)
        Q = torch.max(Q)
        if Q < 1e-10:
            Q = 1e-10

        # Compute gradient
        PQ = P - Q
        for i in range(n):
            dY[i, :] = torch.sum((PQ[:, i] * num[:, i]).repeat(no_dims, 1).t() * (Y[i, :] - Y), 0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum

        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)).double() + (gains * 0.8) * ((dY > 0.) == (iY > 0.)).double()
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - torch.mean(Y, 0)

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = torch.sum(P * torch.log(P / Q))
            print("Iteration %d: error is %f" % (iter + 1, C))

        # Stop lying about P-values
        if iter == 100:
            P = P / 4.

    # Return solution
    return Y


def dist(input1,center):
    avg_dist = 0.0
    for i in range(input1.shape[0]):
        dist = np.linalg.norm(input1[i] - center)
        avg_dist += dist
    avg_dist /= input1.shape[0]
    return avg_dist

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

    from google_drive_downloader import GoogleDriveDownloader as gdd
    gdd.download_file_from_google_drive(file_id='1ZXoDNzxUGsMCmQKTixcKHf0rHYYWTR1c',
                                    dest_path='./checkpoint4.pth',
                                    unzip=False)

    img_siren = Siren(in_features=in_features, hidden_features=hidden_features, 
                        hidden_layers=hidden_layers, out_features=out_features, outermost_linear=True)

    HyperNetEmbedd = Hyper_Net_Embedd(len(dataset),hyper_in_features,hyper_hidden_layers,hyper_hidden_features,img_siren)

<<<<<<< HEAD
    checkpoint = torch.load('./checkpoint1.pth')
=======
    checkpoint = torch.load('./checkpoint5.pth')
>>>>>>> 984e0bcb471cd152c6ee63f67b7119b465c40a11
    HyperNetEmbedd.load_state_dict(checkpoint['model_state_dict'])


    X = []
    labels = []
<<<<<<< HEAD
    choice = np.random.randint(0,17000,1000)
    for i in choice:
=======
    for i in range(len(dataset)):
>>>>>>> 984e0bcb471cd152c6ee63f67b7119b465c40a11
        i_t = torch.tensor(i)
        i_t_c = i_t
        embedd = HyperNetEmbedd.get_embedd(index=i_t_c)
        embedd = embedd.detach().cpu().numpy()
        X.append(embedd)
        index = dataset[i_t]
        labels.append(index['label'])

    X = np.asarray(X)
    print(X.shape)

    labels = [int(i) for i in labels]    

    indices_3 = []
    indices_4 = []
    indices_8 = []
    for i, v in enumerate(labels):
        if v == 3:
            indices_3.append(i)
        elif v == 4:
            indices_4.append(i)
        elif v == 8:
            indices_8.append(i)
        else:
            continue

    print(indices_3[:10])
    print(indices_4[:10])
    print(indices_8[:10])

    embedd_3 = np.squeeze(X[indices_3,:])
    embedd_4 = np.squeeze(X[indices_4,:])
    embedd_8 = np.squeeze(X[indices_8,:])
    # print(embedd_3[:2])
    # print(embedd_4[:2])
    # print(embedd_8[:2])


    center_3 = np.mean(embedd_3,axis=0)
    center_4 = np.mean(embedd_4,axis=0)
    center_8 = np.mean(embedd_8,axis=0)
    # print(center_3)
    # print(center_4)
    # print(center_8)

    # print('var of 3 is {}'.format(np.var(embedd_3)))
    # print('var of 4 is {}'.format(np.var(embedd_4)))
    # print('var of 8 is {}'.format(np.var(embedd_8)))

    # print('avg radius of 3 is {}'.format(dist(embedd_3,center_3)))
    # print('avg radius of 4 is {}'.format(dist(embedd_4,center_4)))
    # print('avg radius of 8 is {}'.format(dist(embedd_8,center_8)))

    # print('distance 3 <----> 4 is {}'.format(np.linalg.norm(center_3 - center_4)))
    # print('distance 3 <----> 8 is {}'.format(np.linalg.norm(center_3 - center_8)))
    # print('distance 4 <----> 8 is {}'.format(np.linalg.norm(center_4 - center_8)))

    inter = np.random.randint(0,embedd_3.shape[0],2)

    meshgrid = get_mgrid(sidelen=28)
    meshgrid = meshgrid.unsqueeze(0)
    meshgrid = meshgrid
    num = 20
    with torch.no_grad():
        embedd_3 = torch.tensor(embedd_3)
        embedd_4 = torch.tensor(embedd_4)
        embedd_8 = torch.tensor(embedd_8)

        a = embedd_3[inter[0]]
        b = embedd_3[inter[1]]
        a = a.unsqueeze(0)
        b = b.unsqueeze(0)
        s = np.linspace(0,1,num)
        for idx in range(num):
            fig, axes = plt.subplots(1,1)
            model_output = HyperNetEmbedd.embedd2inr(a+s[idx]*(b-a))
            param = model_output
            output = img_siren(meshgrid,params=param)
            plot_sample_image(output, ax=axes)
            axes.set_title(str(idx), fontsize=25)
            path = os.path.join(here, './inference/{}.png'.format(idx))
            plt.savefig(path,format='png')
            plt.cla()
        plt.close('all')


    use_torch = False
    use_ski = False

    if use_torch:
        X = torch.from_numpy(X)
        # confirm that x file get same number point than label file
        # otherwise may cause error in scatter
        assert(len(X[:, 0])==len(X[:,1]))
        assert(len(X)==len(labels))

        with torch.no_grad():
            Y = tsne(X, 2, 50, 10.0)

        Y = Y.cpu().numpy()

        # You may write result in two files
        print("Save Y values in file")
        S1 = Y[:,0]
        S2 = Y[:,1]

        # S1 = np.loadtxt("y1.txt", comments="#", delimiter="\n", unpack=False)
        # S2 = np.loadtxt("y2.txt", comments="#", delimiter="\n", unpack=False)

        fig, axes = plt.subplots(1,1)

        axes.scatter(S1, S2,c=labels,cmap=plt.cm.cool)

        path = os.path.join(here, './1.png')
        plt.savefig(path,format='png')
        plt.cla()
    if use_ski:
        pca = PCA(n_components=50)
        X_pca = pca.fit_transform(X) # randomly sample data to run quickly

        n_select = 10000 # reduce dimensionality with t-sne
        tsne = TSNE(n_components=2, verbose=1, perplexity=80, n_iter=2000, learning_rate=200)
        tsne_results = tsne.fit_transform(X_pca)# visualize
        df_tsne = pd.DataFrame(tsne_results, columns=['comp1', 'comp2'])

        S1 = df_tsne['comp1']
        S2 = df_tsne['comp2']

        fig, axes = plt.subplots(1,1)

        axes.scatter(S1, S2,c=labels,cmap=plt.cm.cool)

        path = os.path.join(here, './1.png')
        plt.savefig(path,format='png')
        plt.cla()
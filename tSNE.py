from fileinput import close
from http.client import ImproperConnectionState
import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
from INR_test import Hyper_Net_Embedd, MNIST
from Siren_meta import BatchLinear, Siren, get_mgrid, dict_to_gpu
from torch.utils.data import DataLoader, Dataset
import os
from sklearn.datasets import fetch_mldata
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
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


if __name__ == "__main__":
    print("Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset.")

    here = os.path.join(os.path.dirname(__file__)) 
    data_path = os.path.join(here, '../../dataset/')
    dataset = MNIST(data_path)
    dataloader = DataLoader(dataset, batch_size=32, num_workers=0,shuffle=False)

    hyper_in_features = 100
    hyper_hidden_layers = 4
    hyper_hidden_features = 300

    in_features=2
    hidden_features=128
    hidden_layers=1
    out_features=1

    img_siren = Siren(in_features=in_features, hidden_features=hidden_features, 
                        hidden_layers=hidden_layers, out_features=out_features, outermost_linear=True).cuda()

    HyperNetEmbedd = Hyper_Net_Embedd(len(dataset),hyper_in_features,hyper_hidden_layers,hyper_hidden_features,img_siren).cuda()

    checkpoint = torch.load('./checkpoint2.pth')
    HyperNetEmbedd.load_state_dict(checkpoint['model_state_dict'])

    X = []
    labels = []
    for i in range(1000):
        i_t = torch.tensor(i)
        i_t_c = i_t.cuda()
        embedd = HyperNetEmbedd.get_embedd(index=i_t_c)
        embedd = embedd.detach().cpu().numpy()
        X.append(embedd)
        index = dataset[i_t]
        labels.append(index['label'])

    X = np.asarray(X)

    labels = [int(i) for i in labels]

    use_torch = False
    use_ski = True

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
        pca = PCA(n_components=90)
        X_pca = pca.fit_transform(X) # randomly sample data to run quickly

        n_select = 10000 # reduce dimensionality with t-sne
        tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=2000, learning_rate=100)
        tsne_results = tsne.fit_transform(X_pca)# visualize
        df_tsne = pd.DataFrame(tsne_results, columns=['comp1', 'comp2'])

        S1 = df_tsne['comp1']
        S2 = df_tsne['comp2']

        fig, axes = plt.subplots(1,1)

        axes.scatter(S1, S2,c=labels,cmap=plt.cm.cool)

        path = os.path.join(here, './1.png')
        plt.savefig(path,format='png')
        plt.cla()
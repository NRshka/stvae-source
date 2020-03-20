import scipy
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.neighbors import NearestNeighbors
from numpy.random import randint, choice
from numpy import stack, zeros, unique, arange, log, vectorize, logical_or
from numpy import mean as np_mean
from math import log2
from warnings import warn
from torch import Tensor
import pdb

def count2d(arr, max_ind):
    freq = zeros((arr.shape[0], max_ind+1))
    for idx, line in enumerate(arr):
        for item in line:
            freq[idx][int(item)] += 1 / (arr.shape[1] + 1e-13)

    return freq



def get_knn_purity(latents, labels, n_neighs=30):
    latents = latents.cpu().detach().numpy() if isinstance(latents, Tensor) else latents
    labels = labels.cpu().detach().numpy() if isinstance(labels, Tensor) else labels

    nbrs = NearestNeighbors(n_neighbors=n_neighs + 1).fit(latents)
    indices = nbrs.kneighbors(latents, return_distance=False)[:, 1:]
    neigh_labels = vectorize(lambda x: labels[x])(indices)

    scores = ((neigh_labels - labels.reshape(-1, 1)) == 0).mean(axis=1)
    res = [np_mean(scores[labels.reshape(scores.shape) == i]) for i in unique(labels)]

    return np_mean(res)


def get_batch_entropy(data, max_ind, classifier=None, batch_size=50):
    #TODO assert classfier.n_neighbor == batch_size
    if classifier is None:
        classifier = KNN(batch_size)

    X, Y = data[0], data[1]
    classifier.fit(X, Y)

    indeces = randint(X.shape[0], size=(batch_size,))
    points = X[indeces]
    neigh = classifier.kneighbors(points, batch_size, False)#no distances
    #neigh's shape is (num_points x batch_size)
    freqs = count2d(Y[neigh], max_ind)

    entropy_score = 0.
    for line in freqs:
        entropy_score += sum([-i*log2(i) for i in line if i != 0.])

    return entropy_score / neigh.shape[0]


def entropy_batch_mixing(latents, batches, n_neighs=50, n_pools=50, n_samples=100):
    unique_batch_ind = unique(batches)
    count_batches = unique_batch_ind.shape[0]
    if count_batches < 2:
        warn('Not enough unique batches')
        return -1.
    elif count_batches == 2:
        return entropy_batch_mixing_(latents, batches, n_neighs, n_pools, n_samples)
    else:
        batches = batches.reshape((-1,))
        average_ebm = 0.
        counter = 0
        ebm_dict = {}
        for i in unique_batch_ind:
            ebm_dict[i] = {}
            for j in unique_batch_ind:
                if i == j:
                    continue

                ebm_dict[i][j] = entropy_batch_mixing_(latents[logical_or(batches == i, batches == j)],
                                             batches[logical_or(batches == i, batches == j)])
                average_ebm += ebm_dict[i][j]
                counter += 1

        return ebm_dict, average_ebm / counter


def entropy_batch_mixing_(latents, batches, n_neighs=50, n_pools=50, n_samples=100):
    def cross_entropy(data):
        n_batches = len(unique(data))
        assert n_batches == 2, ValueError("Entropy can be calculated with only 2 batches")

        freq = np_mean(data == unique(data)[0])
        if freq == 0 or freq == 1:
            return 0

        return -freq * log(freq) - (1 - freq) * log(1 - freq)

    n_neighs = min(n_neighs, latents.shape[0] - 1)
    knn = NearestNeighbors(n_neighbors=n_neighs + 1, n_jobs=8)
    knn.fit(latents)
    kmatrix = knn.kneighbors_graph(latents) - scipy.sparse.identity(latents.shape[0])

    score = 0
    #pdb.set_trace()
    for t in range(n_pools):
        indices = choice(arange(latents.shape[0]), size=n_samples)
        while unique(batches[kmatrix[indices].nonzero()[1][kmatrix[indices].nonzero()[0] == 1]]).shape[0] < 2:
            indices = choice(arange(latents.shape[0]), size=n_samples)
        score += np_mean([
            cross_entropy(
                batches[kmatrix[indices].nonzero()[1][kmatrix[indices].nonzero()[0] == 1]]
            )
        for i in range(n_samples)])

    return score / n_pools

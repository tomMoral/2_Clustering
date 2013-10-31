import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from load import load


class Kmeans:
    def __init__(self, K, spread=False):
        self.name = 'Kmeans'
        self.K = K
        self.spread=spread
    def init(self, X):
        n = X.shape[0]
        K = self.K
        if not self.spread:
            #Select random center in the X data:
            indices = np.random.choice(range(n), K, replace = False)
            centroid = X[indices]
        else:
            #Kmeans++, initialisation of centroid with spread points
            centroid = []
            indices = []
            i0 = np.random.randint(n)
            indices.append(i0)
            centroid.append(X[i0])
            for i in range(1,K):
                dist = np.array([[i,min([norm(x-c) for c in centroid])] 
                                 for i, x in enumerate(X) 
                                     if i not in indices])
                n_p = dist[:,1].sum()
                proba = dist[:,1]/n_p
                bins = np.add.accumulate(proba)
                i0 = np.digitize(np.random.random(1), bins)[0]

                indices.append(dist[i0][0])
                centroid.append(X[i0])
            
        return np.array(centroid)

    def fit(self, X):
        n = X.shape[0]
        K = self.K

        centroid = self.init(X)

        dist = [[norm(x-c) for c in centroid] for x in X]
        clusters_prev = np.zeros(n)
        clusters = np.argmin(dist, 1)
        iterations = 0

        while (clusters_prev != clusters).any() and iterations < 100 :
            iterations += 1
            for k in range(self.K):
                centroid[k] = np.mean(X[np.where(clusters == k)], axis=0)
            clusters_prev = clusters
            dist = [[norm(x-c) for c in centroid] for x in X]
            clusters = np.argmin(dist, 1)
       
        self.centroid = centroid
        disto = np.square(dist).sum()

        return (disto, centroid)

    def plot(self, filename, centroid, step):

        X = load(filename)
        dist = [[norm(x-c) for c in centroid] for x in X]
        clusters = np.argmin(dist, 1)

        cm = ListedColormap(['r', 'b', 'g', 'c', 'y', 'm', 'k'])
        fig = plt.figure()
        plt.scatter(X[:,0], X[:,1], marker = 'o', c=clusters, cmap=cm)
        plt.scatter(centroid[:,0], centroid[:,1], marker='o', linewidths=10)
        fig.show()
        plt.savefig('../figures/'+self.name +'_' + step + '.eps')


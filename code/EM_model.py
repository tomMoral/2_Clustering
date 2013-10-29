import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from numpy import linalg

from load import load

from gaussian import gaussian_density
from Kmeans import Kmeans

class EM:
    def __init__(self, K, indep=False):
        self.name = 'EM'
        self.K = K
        self.indep = indep
        self.initializer = Kmeans(K)

    def init(self, X):
       (n,d) = X.shape

       indices = np.random.choice(range(n), self.K, replace = False)
       _, centroid = self.initializer.fit(X)
       clusters = [gaussian_density(centroid[k], np.eye(d)) 
                   for k in range(self.K)]
       return clusters

    def covariance(self, X, centroid, tau):
        tau = tau.reshape((self.K,-1,1))
        sigma = [(X-centroid[k]).T.dot(tau[k]*(X-centroid[k])) / tau[k].sum() for k in range(self.K)]
        if self.indep:
            for k in range(self.K):
                sigma[k][0,1] = sigma[k][1,0] = 0
                s_k = sigma[k].sum()
                sigma[k][0,0] = sigma[k][1,1] = s_k
        return sigma

    def fit(self, filename):

        X = load(filename)
        clusters = self.init(X)
        pi = [1.0/self.K for i in range(self.K)]
        e = 1;

        while e > 0.1:

            #E-step
            #generate for the data the probability to be in each gaussian
            tau = np.array([pi[k]*clusters[k].probability(X) 
                            for k in range(self.K)])
            Z = tau.mean(axis=0)
            tau /= Z

            #M-step
            #fit the gaussian to the actual clustering
            centroids = [np.average(X, axis=0, weights=tau[k]) 
                         for k in range(self.K)]
            sigma = self.covariance(X, centroids, tau)
            pi = [tau[k].sum() / tau.sum() for k in range(self.K)]
            for k in range(self.K):
                e = clusters[k].set_moy(centroids[k])
                e += clusters[k].set_var(sigma[k])

        self.centroids = np.array(centroids)
        self.sigma = np.array(sigma)
        self.pi = np.array(pi)

        fig = plt.figure()
        cm = ListedColormap(['r', 'b', 'g', 'c', 'y', 'm', 'k'])
        plt.scatter(X[:,0], X[:,1], marker = 'o', 
                    c=tau.argmax(axis=0).flatten(), cmap=cm)
        plt.scatter(self.centroids[:,0], self.centroids[:,1], 
                    marker='o', linewidths=10)
        fig.show()
        plt.savefig('../figures/'+self.name +'_train.eps')

        return tau

    def test(self, filename):
        X = load(filename)
        d = X.shape[1]

        centroid = self.centroids
        sigma = self.sigma
        pi = self.pi

        clusters = [gaussian_density(centroid[k], np.eye(d)) 
                    for k in range(self.K)]

        tau = np.array([pi[k]*clusters[k].probability(X) 
                        for k in range(self.K)])
        Z = tau.mean(axis=0)
        tau /= Z
        clusters = tau.argmax(axis=0).flatten()

        cm = ListedColormap(['r', 'b', 'g', 'c', 'y', 'm', 'k'])
        fig = plt.figure()
        plt.scatter(X[:,0], X[:,1], marker = 'o', c=clusters, cmap=cm)
        plt.scatter(centroid[:,0], centroid[:,1], marker='o', linewidths=10)
        plt.show()
        plt.savefig('../figures/'+self.name +'_test.eps')



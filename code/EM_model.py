import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg
from pylab import get_cmap

from load import load

from gaussian import gaussian_density
from Kmeans import Kmeans

class EM:
    def __init__(self, K, isotropic=False):
        self.name = 'EM'
        self.K = K
        self.isotropic = isotropic
        self.initializer = Kmeans(K)

    def init(self, X):
       (n,d) = X.shape

       indices = np.random.choice(range(n), self.K, replace = False)
       _, centroid = self.initializer.fit(X)
       clusters = [gaussian_density(centroid[k], np.eye(d)) 
                   for k in range(self.K)]
       return clusters

    def covariance(self, X, centroid, tau):
        d = X.shape[1]
        if self.isotropic:
            sigma = [np.average(((X-centroid[k])**2), weights=tau[k],
                     axis=0).sum()/2*np.eye(d) 
                     for k in range(self.K)]
        else:
            tau = tau.reshape((self.K,-1,1))
            sigma = [(X-centroid[k]).T.dot(tau[k]*(X-centroid[k])) / tau[k].sum() 
                     for k in range(self.K)]
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
            tau /= tau.sum(axis=0)

            #M-step
            #fit the gaussian to the actual clustering
            centroids = [np.average(X, axis=0, weights=tau[k]) 
                         for k in range(self.K)]
            sigma = self.covariance(X, centroids, tau)
            pi = [tau[k].sum() / tau.sum() for k in range(self.K)]
            e = 0
            for k in range(self.K):
                e += clusters[k].set_moy(centroids[k])
                e += clusters[k].set_var(sigma[k])
            e /= self.K

        self.centroids = np.array(centroids)
        self.sigma = np.array(sigma)
        self.pi = np.array(pi)
        labels = tau.argmax(axis=0).flatten()

        llh = (tau*np.log(tau)).sum()
        print 'Train log-likelyhood for {} {}  : {}'.format(self.name,
               ('Iso' if self.isotropic else 'Gen'), llh)



        fig = plt.figure()
        cm = get_cmap('jet', self.K)
        plt.scatter(X[:,0], X[:,1], marker = 'o', 
                    c=labels, cmap=cm)
        plt.scatter(self.centroids[:,0], self.centroids[:,1], 
                    marker='o', linewidths=8)
        ax  = fig.axes[0]
        for k in range(self.K):
            ax.add_artist(clusters[k].plot_conf(cm(k)))
        fig.show()
        plt.savefig('../figures/'+self.name + '_' +
                    ('iso' if self.isotropic else 'gen') 
                    +'_train.eps')

        return tau

    def test(self, filename):
        X = load(filename)
        d = X.shape[1]

        centroid = self.centroids
        sigma = self.sigma
        pi = self.pi

        clusters = [gaussian_density(centroid[k], sigma[k]) 
                    for k in range(self.K)]

        tau = np.array([pi[k]*clusters[k].probability(X) 
                        for k in range(self.K)])
        tau /= tau.sum(axis=0)
        labels = tau.argmax(axis=0).flatten()

        llh = (tau*np.log(tau)).sum()
        print 'Test log-likelyhood for {} {}  : {}'.format(self.name,
               ('Iso' if self.isotropic else 'Gen'), llh)

        fig = plt.figure()
        cm = get_cmap('jet', self.K)
        plt.scatter(X[:,0], X[:,1], marker = 'o', c=labels, cmap=cm)
        plt.scatter(centroid[:,0], centroid[:,1], marker='o', linewidths=8)
        ax  = fig.axes[0]
        for k in range(self.K):
            ax.add_artist(clusters[k].plot_conf(cm(k)))
        fig.show()
        plt.savefig('../figures/'+self.name + '_' +
                    ('iso' if self.isotropic else 'gen')
                    +'_test.eps')


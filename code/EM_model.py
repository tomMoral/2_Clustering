import numpy as np

from load import load

from gaussian import gaussian_density

class EM:
    def __init__(self, K):
        self.name = 'EM'
        self.K = K

    def init(self, X):
       n = X.shape[0]
       indices = np.random.choice(range(n), self.K, replace = False)
       centroid = X[indices]
       clusters = [gaussian(centroid[k], np.eye(n)) for k in range(self.K)]
       return clusters

    def fit(self, filename):
        #TODO: fit the EM model to the data
        X = load(filename)
        self.clusters = self.init(X)
        self.pi = [1.0/self.K for i in range(self.K)]

        #E-step
        #generate for the data the probability to be in each gaussian
        tau = np.array([pi[k]*clusters[k].probability(X) 
                             for k in range(self.K)])
        Z = tau.mean(axis=0)
        tau /= Z

        #M-step
        #fit the gaussian to the actual clustering
        centroids = [x*tau[k]/tau[k].sum() for k in range(self.K)]
        sigma = [tau[k]*(x-centroids[k]).T*(s-centroids[k])/tau[k].sum()
                 for k in range(self.K)]


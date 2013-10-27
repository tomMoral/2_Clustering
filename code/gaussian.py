import numpy as np
from numpy.linalg import pinv, det

class gaussian_density:
    def __init__(self, mu, sigma):
        self.mu = mu.reshape((1,-1))
        self.sigma = sigma

    def probability(self, X):
        prob = 1/(np.sqrt(det(self.sigma)))*np.exp(
                 -(X-self.mu).dot(pinv(self.sigma)).dot((X-self.mu).T)/2)
        return prob.diagonal()

    def set_var(self, sigma):
        self.sigma = sigma

    def set_moy(self, mu):
        self.mu = mu



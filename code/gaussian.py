import numpy as np
from numpy import linalg
from numpy.linalg import pinv, det
import matplotlib as mpl

class gaussian_density:
    def __init__(self, mu, sigma):
        self.mu = mu.reshape((1,-1))
        self.sigma = sigma

    def probability(self, X):
        prob = 1/(np.sqrt(det(self.sigma)))*np.exp(
                 -(X-self.mu).dot(pinv(self.sigma)).dot((X-self.mu).T)/2)
        return prob.diagonal()

    def set_var(self, sigma):
        e = abs(sigma - self.sigma)
        self.sigma = sigma
        return np.sqrt(e.mean())/2

    def set_moy(self, mu):
        e = (mu - self.mu)
        self.mu = mu
        return e.mean()/2

    def plot_conf(self, c):
        v, w = linalg.eigh(self.sigma)
        angle = np.arctan2(w[0][1], w[0][0])
        angle = 180 * angle / np.pi
        v = 1.83 * 2 * np.sqrt(v)
        ell = mpl.patches.Ellipse(self.mu.flatten(), v[0], v[1], 180 +
                                  angle, color=c, alpha=0.5)
        return ell


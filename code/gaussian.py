import numpy as np
from numpy import linalg
from numpy.linalg import pinv, det
import matplotlib as mpl

class gaussian_density:
    '''Gaussian distribution object

    Parameters
    ----------
    mu: array like object
        Center of the distribution of dimension d.

    sigma: array like object
        Covariance matrix of the distribution.
        It should be a square matrix with dimension
        d equal to the one of mu

    Usage
    -----
    You can use set_var(sigma) and set_moy(mu) to 
    change sigma and mu parameters

    You can use probability(x) to compute the probability
    to get x from this distribution

    You can use plot_conf(c) to get an ellipse representation
    of your confidence interval.
    If d =2 then you will get the confidence 90% ellipse.
    Else, you will get the 90% confidence ellipse for the 2
    strongest eigenvalues
    
    '''
    def __init__(self, mu, sigma):

        self.mu = np.array(mu).reshape((1,-1))
        self.sigma = np.array(sigma)

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
        if v.shape[0] > 2:
            order = v.argsort()
            v = v[order]
            w = w[:,order]
        angle = np.arctan2(w[0][1], w[0][0])
        angle = 180 * angle / np.pi
        v = 1.83 * 2 * np.sqrt(v)
        ell = mpl.patches.Ellipse(self.mu.flatten(), v[0], v[1], 180 +
                                  angle, color=c, alpha=0.5)
        return ell


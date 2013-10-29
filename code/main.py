import numpy as np
import os
from sys import stdout as out

from EM_model import EM
from Kmeans import Kmeans
from load import load

base_dir = '../data'
filenames_trn = ['EMGaussian.data']
filenames_tst = ['EMGaussian.test']
f_trn = os.path.join(base_dir,filenames_trn[0])
f_tst = os.path.join(base_dir,filenames_tst[0])


import argparse

parser = argparse.ArgumentParser(description='Perform a clustering.')
parser.add_argument('-k', metavar='K', dest='K',
                    type=int,  default=4,
                    help='Set the number of cluster clusters.')
parser.add_argument('-m', metavar='model',
                    type=str, default='kmeans',
                    help='Set the clustering model used (kmeans or EM)')
parser.add_argument('--spread', action='store_true')
parser.add_argument('--indep', action='store_true')
args = parser.parse_args()

if args.m == 'EM':
    model = EM(args.K, indep=args.indep)
    for i in range(10):
        model.fit(f_trn)
        model.test(f_tst)
else: 
    model = Kmeans(args.K, args.spread)
    distos = []
    centers = []
    X = load(f_trn)
    for i in range(50):
        out.write('\b\b\b\b\b\b{:6.2%}'.format(i/(50.)))
        out.flush()
        d,c = model.fit(X)
        distos.append(d)
        centers.append(c)
    distos = np.array(distos)
    print "distortion avg : ", distos.mean()
    print "Best model : " , distos.min()

    i0 = distos.argmin()

    model.plot(f_trn, centers[i0], 'train')
    model.plot(f_tst, centers[i0], 'test')


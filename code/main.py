import numpy as np
import os

from EM_model import EM
from Kmeans import Kmeans

base_dir = '../data'
filenames_trn = ['EMGaussian.data']
filenames_tst = ['EMGaussian.test']

import argparse

parser = argparse.ArgumentParser(description='Perform a clustering.')
parser.add_argument('-k', metavar='K', dest='K',
                    type=int,  default=4,
                    help='Set the number of cluster clusters.')
parser.add_argument('-m', metavar='model',
                    type=str, default='kmeans',
                    help='Set the clustering model used (kmeans or EM)')
parser.add_argument('--spread', action='store_true')
args = parser.parse_args()

if args.m == 'EM':
    model = EM(args.K)
else: 
    model = Kmeans(args.K, args.spread)

for f in filenames_trn:
    model.fit(os.path.join(base_dir,f))

for i, f in enumerate(filenames_tst):
    model.test(os.path.join(base_dir,f))

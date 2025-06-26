## Script for generating the N test patterns to be used in optimizing the networks.
## Authors: Dr. Madeleine Bartlett
'''
'''

import argparse 
import numpy as np 
from scipy.stats import beta
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--n_tests', required = True)


## define the __main__ loop that will run this from terminal
if __name__ == '__main__':
    args = parser.parse_args()
    ## create domain
    domain = np.arange(0,4,0.01)
    ## set number of test bundles to create 
    n_tests = int(args.n_tests)
    dists = []
    print('Generating patterns')
    for seed in np.random.randint(5000, size=n_tests):
        np.random.seed(seed)
        a = np.random.uniform(1,10,1)
        b = np.random.uniform(1,10,1)

        Ps = beta.pdf(domain, a,b, scale=domain[-1])
        dists.append(Ps)

print('Saving as pickle as test_bundles.pkl')
with open("test_bundles", "wb") as fp: 
    pickle.dump(dists, fp)

print('All done. Great work!')
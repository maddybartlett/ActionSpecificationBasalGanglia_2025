## Script to run an optuna study optimizing the Shallow Decoder Attractor Network. Author: Madeleine Bartlett
'''
This script will run the optuna study by calling the sampler to fetch the parameter values we want to try, 
using those to construct the network and to define the training parameters, then calling the trial run script
to run the training and testing.

This script will then fetch the performance metrics (training, validation and test loss values) and use them to
generate the next set of parameter values. 

We will optimize using TPE Sampler - Tree-structured Parzen Estimator - to sample our parameters.
'''

import optuna

import os, sys
# sys.path.append(os.path.join("..//"))

from samplers import * 
from trial import bgTrial, daTrial

from datetime import datetime
import joblib
import numpy as np
import argparse
import pickle

SAMPLERS = {'hopfield': sample_hop, 
            'accumulator': sample_ia, 
            'wta': sample_wta,
            'dnf': sample_dnf,
            'shallowAttractor': sample_sa,
            'shallowAttractorDecode': sample_sad,
            'deepAttractor': sample_da,}

SAVE_NAMES = {'hopfield': 'HOP', 
            'accumulator': 'IA', 
            'wta': 'WTA',
            'dnf': 'DNF',
            'shallowAttractor': 'SA',
            'shallowAttractorDecode': 'SAD',
             'deepAttractor': 'DA',}

## load test bundles
with open(".//test_bundles", "rb") as fp:
    test_bundles = pickle.load(fp)

def objective(trial, params):
    sample_net = SAMPLERS[params['network']]
    ## sample our parameters
    sampled_params = sample_net(trial)
    ## update the network parameters
    params.update(sampled_params)
    params.update( {'optuna_trial_number': trial.number,
                    'optimize': True} )
    ## generate some random seeds
    seed = np.random.randint(100, size=1)

    ## initialise the trial to run the attractor
    if params['network'] == 'deepAttractor':
        run_trial = daTrial()
    else:
        run_trial = bgTrial()

    try:
        data = run_trial.run(seed=int(seed),
                                    **params)   
    except FloatingPointError:
        data={'rmse_peak':1000,
              'mean_delta_entropy':1000,}

    print('DONE!')

    return data['rmse_peak'], data['mean_delta_entropy']

parser = argparse.ArgumentParser()
parser.add_argument('--network', required = True)

## define the __main__ loop that will run this from terminal
if __name__ == '__main__':
    args = parser.parse_args()
    ## generate a unique string for naming the folder where data will be saved
    dat_time = datetime.now().strftime('%d_%m_%Y_%H_%M_%S') 
    data_dir = SAVE_NAMES[args.network]+'_HPO_{}'.format(dat_time)
    ## add data directory to path to data folder
    data_dir_full = os.path.join('..//data',data_dir)
    ## check if directory exists and if not, make the directory
    if not os.path.exists( data_dir_full ):
        os.mkdir( data_dir_full )

    ## set some of the pytry params
    params={}
    params.update( {'network'             : args.network,
                    'tests'               : test_bundles,
                    'data_dir'            : data_dir_full,
                    'verbose'             : False,
                    'data_format'         : 'npz', } )

    ## set the sampler
    sampler = optuna.samplers.TPESampler()
    ## create the optuna study
    study = optuna.create_study( directions = ["minimize", "minimize"], sampler = sampler )
    ## save the optuna data
    joblib.dump(study, os.path.join(data_dir_full, 'optuna.pkl') )
    ## run the optimization study
    study.optimize(lambda trial:objective(trial, params), n_trials = 100, catch=(FloatingPointError) )
    ## save the optuna data
    joblib.dump(study, os.path.join(data_dir_full,'optuna.pkl') )
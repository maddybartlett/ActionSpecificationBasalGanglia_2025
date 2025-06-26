## Script to run the repeat experiments testing performance over multiple seeds. 
# Authors: Dr. Madeleine Bartlett
'''

'''
import nengo
import pandas as pd
import numpy as np
import os, sys
sys.path.append(os.path.join("..//"))

from trial import bgTrial, daTrial
from datetime import datetime
import argparse

SAVE_NAMES = {'hopfield': 'HOP', 
            'accumulator': 'IA', 
            'wta': 'WTA',
            'dnf': 'DNF',
            'shallowAttractor': 'SA',
            'shallowAttractorDecode': 'SAD',
            'deepAttractor': 'DA'}

def run_exps(network, n_seeds, params):

    assert network in SAVE_NAMES.keys(), f"Network name must be one of: {SAVE_NAMES.keys()}"

    ## Get the optimized parameters for the network
    ## HOPFIELD ##
    if network == 'hopfield':
        params.update({
            'network': network,
            'inp_weight' : 0.03012592296036941,
            'temp' : 5.143638191096048,
            'ens_neurons' : 5000,
            'present_time': 0.001,
        })
    ## INDEPENDENT ACCUMULATOR ##
    elif network == 'accumulator':
        params.update({
            'network': network,
            'inp_weight': 0.02786689086159566,
            'thresh': 0.39058938860367004,
            'intercept_width': 0.9936724585602154,
            'ens_neurons': 5000,
            'net_neurons': 500,
            'present_time': 1.0,
        })
    ## WINNER-TAKE-ALL ##
    elif network == 'wta':
        params.update({
            'network': network,
            'inp_weight': 0.5078060616037147, 
            'thresh': 0.535314219440191, 
            'inhib_scale': 3.7186217698242467, 
            'intercept_width': 0.11743057936021345, 
            'ens_neurons': 1000, 
            'net_neurons': 200,
            'present_time': 1.0,
        })
    ## DYNAMIC NEURAL FIELD ##
    elif network == 'dnf':
        params.update({
            'network': network,
            'inp_weight': 1.8747462341208676e-06,
            'dnf_h': -1.7970921845582923, 
            'dnf_global_inhib': 8.689358012947885,
            'dnf_tau': 0.12645440732422591,
            'kernel_excit': 10.265535449219659,
            'kernel_inhib': 0.40606955705785097,
            'ens_neurons': 1000,
            'present_time': 1.0,
        })
    ## SHALLOW ATTRACTOR ##
    elif network == 'shallowAttractor':
        params.update({
            'network': network,
            'inp_weight': 0.00017724588760742127, 
            'ens_neurons': 2000,
            'present_time': 1.0,
        })
    ## SHALLOW ATTRACTOR WITH PLACE CELLS ##
    elif network == 'shallowAttractorDecode':
        params.update({
            'network': network,
            'inp_weight': 0.0005777630429817888,
            'present_time': 1.0,
        })
    ## DEEP ATTRACTOR ##
    elif network == 'deepAttractor':
        params.update({
            'network': network,
            'inp_weight': 0.6929820220799751,
            'learn_rate': 1.1376171238316905e-07,
            'n_layers': 4,
            'ens_neurons': 5000,
            'ens_neurons': 5000,
            'present_time': 1.0,
        })

    ## data lists
    out_std_lst = []
    out_peak_lst = []
    test_std_lst = []
    test_peak_lst = []
    out_entropy_lst = []
    test_entropy_lst = []
    sims_before_lst = []
    sims_after_lst = []

    ## run experiment
    if network == 'deepAttractor':
        trial = daTrial()
    else:
        trial = bgTrial()
    seeds = np.random.randint(0,500,n_seeds)
    for seed in seeds:
        out_data = trial.run(seed=int(seed), data_format='npz', **params)

        out_std_lst.append(out_data['out_std'])
        out_peak_lst.append(out_data['out_peak'])
        test_std_lst.append(out_data['test_std'])
        test_peak_lst.append(out_data['test_peak'])
        out_entropy_lst.append(out_data['out_entropy'])
        test_entropy_lst.append(out_data['test_entropy'])
        sims_before_lst.append(out_data['sims_before_list'])
        sims_after_lst.append(out_data['sims_after_list'])

    ## save data
    print('Saving data')
    data_lists = [out_std_lst, out_peak_lst, test_std_lst, test_peak_lst, out_entropy_lst, test_entropy_lst, sims_before_lst]
    csv_names = ['out_std', 'out_peak', 'test_std', 'test_peak', 'out_entropy', 'test_entropy', 'sims_before']
    for i,lst in enumerate(data_lists):
        data = {}
        columns=[]
        for j in range(0,n_seeds):
            data.update({
                f'exp_{j+1}': lst[j]
            })
            columns.append(f'exp_{j+1}')
        
        df = pd.DataFrame(data)
        df = df.explode(columns)
        
        print(df.memory_usage(deep=True).sum() / 1024**2)
        df.to_csv(os.path.join(params['data_dir'], f'{csv_names[i]}.csv'))

    ## now the sims_after data because it's a different shape
    data = {}
    columns=[]

    for j in range(0,n_seeds):
        for i in range(len(sims_after_lst[j][0])):
            data.update({
                    f'exp_{j+1}_timestep{i}': np.hstack(sims_after_lst[j][0][i])
                })
            columns.append(f'exp_{j+1}_timestep{i}')
        df = pd.DataFrame(data)
        df = df.explode(columns)

        print(df.memory_usage(deep=True).sum() / 1024**2)
        df.to_csv(os.path.join(params['data_dir'], 'sims_after.csv'))

    ## finish message
    print('All done. Great work!')
    

parser = argparse.ArgumentParser()
parser.add_argument('--network', required = True)
parser.add_argument('--n_seeds', default = 20)

## define the __main__ loop that will run this from terminal
if __name__ == '__main__':
    args = parser.parse_args()
    dat_time = datetime.now().strftime('%d_%m_%Y_%H_%M_%S') 
    data_dir = SAVE_NAMES[args.network]+'_EXP_{}'.format(dat_time)

    data_dir_full = os.path.join('..//data',data_dir)

    params = {
        'ens_dims': 512,
        'dec_neurons': 400,
        'neuron_type': nengo.LIFRate(),
        'ssp_dims': 512,
        'domain_dist': 0.01,
        'act_width': 4,
        'sim_time': 1.0,
        'dt': 0.001,
        'space_type': 'continuous',
        'n_tests': 60, 
        'network': args.network,
        'data_dir': data_dir_full,
        'verbose': False,
    }

    run_exps(args.network, args.n_seeds, params)
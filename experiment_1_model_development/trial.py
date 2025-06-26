## Authors: Dr. Madeleine Bartlett

import pytry
import sspspace
import numpy as np
import pandas as pd
import random
from utils import *
from nets import *
import os
import nengo
import nengo_dl
import tensorflow as tf
from build_da import DeepAttractor as DA
import dill as pickle

class bgTrial(pytry.Trial):
    def params(self):
        ## Network params
        self.param('Number of neurons in ensemble representing SSP', ens_neurons=500),
        self.param('Number of dimensions represented by ensemble representing SSP', ens_dims=512),
        self.param('Number of neurons in ensemble decoding SSP', dec_neurons=400),
        self.param('Type of neuron', neuron_type=nengo.LIFRate()),
        self.param('Transform on connection from input to ensemble', inp_weight=1),
        self.param('Temperature for softmax', temp=1),
        self.param('Value of u where rate of change = 0', dnf_h=0),
        self.param('Value for global inhibition', dnf_global_inhib=0),
        self.param('Value for tau - time constant', dnf_tau=0.001),
        self.param('Value of local excitation', kernel_excit=0),
        self.param('Value of global inhibition', kernel_inhib=0),
        self.param('Width of local excitation', exc_width=5),
        self.param('Width of global inhibition', inh_width=10),
        self.param('Scale for inhibition in WTA', inhib_scale=1.0),
        self.param('Threshold for Accumulator and WTA', thresh=0.8),
        self.param('How widely distributed the intercepts are', intercept_width=0.15)
        self.param('Neurons per ensemble in WTA or IA networks', net_neurons=100),

        ## Representation params
        self.param('Dimensionality of SSP representation', ssp_dims=512),
        self.param('Dimensionality of domain', domain_dims=1),
        self.param('Length scale for SSP representation', length_scale=0.5),
        self.param('Max value in domain OR number of discrete patterns', act_width=4),
        self.param('Distance between samples from domain space', domain_dist=0.01),
    
        ## Experiment params
        self.param('Number of random seeds', n_seeds=1),
        self.param('Run time of simulation (seconds)', sim_time=0.5),
        self.param('dt for simulation (seconds)', dt=0.001),
        self.param('Domain space type (continuous vs. discrete)', space_type='continuous'),
        self.param('Network being used', network='hopfield'),
        self.param('Number of training points for Attractors', n_training_pts=5000),
        self.param('Test patterns', tests=[])
        self.param('Presentation time', present_time=None)
    
        ## optimization param
        self.param('Hyperparameter optimization', optimize=False)
        self.param('Optuna trial number', optuna_trial_number = None)

    def evaluate(self, param):
        assert len(param.tests) > 0, f'A list of test patterns must be provided.'
        present_time = param.present_time

        if present_time == None:
            present_time = param.sim_time
        ## generate random seeds
        np.random.seed(param.seed)
        network = param.network

        ## create ssp encoder
        ssp_encoder = sspspace.RandomSSPSpace(domain_dim=param.domain_dims, ssp_dim=param.ssp_dims, 
                                              rng=np.random.RandomState(), length_scale=param.length_scale)

        ## create the domain space
        domain = np.arange(0, param.act_width, param.domain_dist).reshape((-1,1))
        ## encode the whole domain space
        domain_phis = ssp_encoder.encode(domain)

        ## create dictionary of network params
        net_params={
            'mode': param.space_type,
            'seed': param.seed,
            'neuron_type': param.neuron_type,
            'ens_neurons': param.ens_neurons, ## will be overwritten for the shallow decode attractor 
            'ens_dims': param.ens_dims, ## will be overwritten for the shallow decode attractor
            'inp_weight': param.inp_weight,
        }

        if param.space_type == 'discrete':            
            ## values to be encoded as SSPs and used as the patterns in memory
            ssp_points = (np.arange(domain.shape[0]))
            ## encode the discrete set of actions
            action_ssps = ssp_encoder.encode(ssp_points.reshape(-1,1))
            actions = np.asarray(action_ssps)

            ## recreate domain and domain_phis for decoding
            domain = np.arange(0, param.act_width, 0.001).reshape((-1,1))
            ## encode the whole domain space
            domain_phis = ssp_encoder.encode(domain)

            if 'ttractor' in network:
                ## Training data
                n_training_pts = param.n_training_pts #50000
                samples, targets = get_training_data_discrete(n_training_pts, domain, param.act_width)
                targets = np.asarray(targets) 
                samples = np.asarray(samples)
                
                net_params.update({
                    'samples': samples, 
                    'targets': targets,
                })

            if network == 'hopfield':
                net_params.update({'m_patterns': actions,
                                  'temp': param.temp})

        elif param.space_type == 'continuous':
            ## generate place cell-like encoders
            ## sample evenly from across the space
            low = 0
            high = param.act_width
            width = high - low
            places_ = np.arange(low, high, width/param.dec_neurons)
            encoders = np.asarray(ssp_encoder.encode(places_.reshape(-1,1))).squeeze()
            ## update net params with encoders for decoder ensemble
            net_params.update({'encoders': encoders})

            if 'ttractor' not in network:
                ## update net params with decoder ensemble params
                net_params.update({'dec_neurons': param.dec_neurons,
                                   'dec_dims': param.ssp_dims,
                                  })
            
            ## network-specific params
            if network == 'hopfield':
                # softmax temp (hopfield only)
                net_params.update({'temp': param.temp})

            if 'ttractor' in network:
                ## Training data
                samples, targets = get_training_data_cont(param.n_training_pts, ssp_encoder, param.act_width, domain, domain_phis, dist_type='beta')
                targets = np.asarray(targets) 
                samples = np.asarray(samples)
                
                net_params.update({
                    'samples': samples, 
                    'targets': targets,
                })

        ## create network using the set parameters
        if param.network == 'hopfield':
            net = Hopfield(**net_params)
            present_time = 0.001
        elif param.network == 'wta':
            net_params.update({
                'thresh': param.thresh,
                'inhib_scale': param.inhib_scale,
                'intercept_width': param.intercept_width,
                'WTA_neurons': param.net_neurons,
            })
            net = WTA(**net_params)
        elif param.network == 'accumulator':
            net_params.update({
                'thresh': param.thresh, 
                'intercept_width': param.intercept_width,
                'IA_neurons': param.net_neurons,
            })
            net = Accumulator(**net_params)
        elif param.network == 'dnf':
            net_params.update({
                'dnf_h': param.dnf_h, 
                'dnf_global_inhib': param.dnf_global_inhib,
                'dnf_tau': param.dnf_tau,
                'kernel_excit': param.kernel_excit,
                'kernel_inhib': param.kernel_inhib,
                'exc_width': param.exc_width,
                'inh_width': param.inh_width,
            })
            net = DNF(**net_params)
        elif param.network == 'shallowAttractor':    
            net_params.update({'encoders': None,})     
            net = ShallowAttractor(**net_params)
        elif param.network == 'shallowAttractorDecode':
            net_params.update({
                'encoders': encoders,
                'ens_neurons': param.dec_neurons,
                'ens_dims': param.ssp_dims,
            })
            net = ShallowAttractor(**net_params)

        ## RUN EXPERIMENT ##
        all_patterns = []
        for i in range(len(param.tests)):
            ## for the hopfield network we need to fill in the gaps with 0's 
            stim_pattern = gen_beta_cont_dist(param.tests[i], domain_phis)
            all_patterns.append(stim_pattern)
            if param.network == 'hopfield':
                for step in range(int(param.sim_time/param.dt)-1):
                    all_patterns.append(np.zeros(stim_pattern.shape))

        T = param.sim_time * len(param.tests)
                
        ## Build and run network
        results = net.run_model(stim_pattern=all_patterns, T=T, present_time=present_time)
    
        out_std, test_std, out_peak, test_peak, out_entropy, test_entropy, sims_before_list, sims_after_list = get_data(domain, 
                                                                                                                        domain_phis, 
                                                                                                                        results['output'], 
                                                                                                                        results['input'], 
                                                                                                                        len(param.tests))

        if param.optimize:
            ## Performance metrics
            ## Accuracy - RMSE between the test and target peak locations on the final timestep
            final_peaks = np.asarray(out_peak)
            test_peaks = np.asarray(test_peak)
            rmse_peak = np.sqrt(((test_peaks - final_peaks)**2).mean())
        
            ## Entropy - Difference between the entropy of the 
            ## input distribution and the entropy of the output distribution 
            ## on the final timestep
            final_ent = np.asarray(out_entropy)
            test_ent = np.asarray(test_entropy)

            ## subtract the initial from the final entropy. 
            ## Good delta entropy scores will be negative indicating a drop in entropy
            delta_entropy = final_ent - test_ent
            mean_delta_entropy = (delta_entropy).mean()

            out_data = {
                'out_std': out_std, 
                'test_std': test_std, 
                'out_peak': out_peak, 
                'test_peak': test_peak, 
                'out_entropy': out_entropy, 
                'test_entropy': test_entropy, 
                'sims_before_list': sims_before_list, 
                'sims_after_list': sims_after_list,
                'rmse_peak': rmse_peak,
                'mean_delta_entropy': mean_delta_entropy,
                    }
        elif not param.optimize:
            out_data = {
                'out_std': out_std, 
                'test_std': test_std, 
                'out_peak': out_peak, 
                'test_peak': test_peak, 
                'out_entropy': out_entropy, 
                'test_entropy': test_entropy, 
                'sims_before_list': sims_before_list, 
                'sims_after_list': sims_after_list
            }
            
        return out_data
    


class daTrial(pytry.Trial):
    def params(self):
        ## Network params
        self.param('Number of neurons in ensemble representing SSP', ens_neurons=500),
        self.param('Number of dimensions represented by ensemble representing SSP', ens_dims=512),
        self.param('Number of neurons in ensemble decoding SSP', dec_neurons=400),
        self.param('Type of neuron', neuron_type=nengo.LIFRate()),
        self.param('Transform on connection from input to ensemble', inp_weight=1),
        self.param('Temperature for softmax', temp=1),
        self.param('Value of u where rate of change = 0', dnf_h=0),
        self.param('Value for global inhibition', dnf_global_inhib=0),
        self.param('Value for tau - time constant', dnf_tau=0.001),
        self.param('Value of local excitation', kernel_excit=0),
        self.param('Value of local inhibition', kernel_inhib=0),
        self.param('Width of local excitation', exc_width=0),
        self.param('Width of global inhibition', inh_width=0),
        self.param('Scale for inhibition in WTA', inhib_scale=1.0),
        self.param('Threshold for Accumulator and WTA', thresh=0.8),
        self.param('How widely distributed the intercepts are', intercept_width=0.15)
        self.param('Neurons per ensemble in WTA or IA networks', net_neurons=100),
        self.param('Number of layers in deep attractor', n_layers=3),
        self.param('Learning rate for deep attractor', learn_rate=0.1),
        self.param('Minibatch size for training batches', minibatch_size=250),

        ## Representation params
        self.param('Dimensionality of SSP representation', ssp_dims=512),
        self.param('Dimensionality of domain', domain_dims=1),
        self.param('Length scale for SSP representation', length_scale=0.5),
        self.param('Max value in domain OR number of discrete patterns', act_width=4),
        self.param('Distance between samples from domain space', domain_dist=0.01),
    
        ## Experiment params
        self.param('Number of random seeds', n_seeds=1),
        self.param('Run time of simulation (seconds)', sim_time=1.0),
        self.param('dt for simulation (seconds)', dt=0.001),
        self.param('Domain space type (continuous vs. discrete)', space_type='continuous'),
        self.param('Network being used', network='deepAttractor'),
        self.param('Number of training points for Attractors', n_training_pts=5000),
        self.param('Test patterns', tests=[])
        self.param('Presentation time', present_time=None)
    
        ## optimization param
        self.param('Hyperparameter optimization', optimize=False)
        self.param('Optuna trial number', optuna_trial_number = None)

    def evaluate(self, param):
        assert len(param.tests) > 0, f'A list of test patterns must be provided.'

        ## presentation time for da doesn't get used so set it to None 
        present_time = None
        ## generate random seeds
        network = param.network
        T = param.sim_time * len(param.tests)

        ## create ssp encoder
        ssp_encoder = sspspace.RandomSSPSpace(domain_dim=param.domain_dims, ssp_dim=param.ssp_dims, 
                                              rng=np.random.RandomState(), length_scale=param.length_scale)

        ## create the domain space
        domain = np.arange(0, param.act_width, param.domain_dist).reshape((-1,1))
        ## encode the whole domain space
        domain_phis = ssp_encoder.encode(domain)

        ## create dictionary of network params
        net_params={
            'n_layers': param.n_layers,
            'neuron_type': param.neuron_type,
            'ens_neurons': param.ens_neurons,
            'ens_dims': param.ens_dims,
            'inp_weight': param.inp_weight,
            'seed': param.seed,
        }

        ## create training data
        train_patterns, train_labels = get_training_data_cont(param.n_training_pts, ssp_encoder, param.act_width, domain, domain_phis, dist_type='beta')

        ## reshape to include time
        train_patterns_ = np.array(train_patterns)[:, None, :]
        train_labels_ = np.array(train_labels)[:, None, :]

        ## build the network
        net = nengo.Network(label="Deep Attractor")
        with net:
            da = DA(**net_params)

        ## train the network
        ## create list of ensembles and connections
        net_objs = da.__dict__['params_list']

        ## create list of the ensemble probes
        net_probes = [getattr(da, a) for a in da.__dict__.keys() if 'probe_ens' in a]

        ## now that we have the network defined:
        ## create data dictionaries
        inputs = {da.inpt: train_patterns_}
        targets = {da.probe_out_nofilt: train_labels_}

        # print(inputs[da.inpt].shape)
        # print(targets[da.probe_out_nofilt].shape)
        targets[da.probe_out_nofilt] = targets[da.probe_out_nofilt].reshape(param.n_training_pts,1,param.ssp_dims)

        ## train the network
        with nengo_dl.Simulator(net, minibatch_size=param.minibatch_size, seed=param.seed) as sim:
            ## set the early stopping 
            es = tf.keras.callbacks.EarlyStopping(monitor="out_no_filter_probe_loss", patience=3, 
                                                    min_delta=0.002, verbose=1)
            ## create a dictionary of losses to train the ensembles and connections
            losses = {da.probe_out_nofilt: tf.keras.losses.MeanSquaredError(), 
                        da.probe_out_nofilt: tf.keras.losses.CosineSimilarity(),}
            for probe in net_probes:
                losses.update({probe: nengo_dl.losses.Regularize(order=2)})

            loss_weights = {da.probe_out_nofilt: [0.2,0.8]}
            for probe in net_probes:
                loss_weights.update({probe: 1e-3})

            ## create the compiler
            sim.compile(
                optimizer=tf.optimizers.RMSprop(param.learn_rate), loss=losses,
                loss_weights=loss_weights,
            )

            ## train the model
            history = sim.fit(inputs, targets, epochs=25, validation_split=0.1, callbacks=[es])

            ## store the trained parameters so that the model can be reproduced 
            trained_params = sim.get_nengo_params(net_objs)
            
            saved_params = {}
            for i in range(len(net_objs)):
                saved_params.update({net_objs[i].label: trained_params[i]})

            with open(f'da_params_{param.seed}.pkl', 'wb') as filename:
                pickle.dump(saved_params, filename)
            

        ## load the trained parameters
        with (open(f'da_params_{param.seed}.pkl', "rb")) as f:
            trained_params = pickle.load(f)

        ## build the network
        net_trained = nengo.Network(label="Deep Attractor")
        with net_trained:
            da = DA(**net_params,
                    trained_params=trained_params)
                
        ## create the testing patterns for testing the network after the training is done
        all_patterns = []
        ## for each pattern, add in the time dimensions
        for i in range(len(param.tests)):
            stim_pattern = gen_beta_cont_dist(param.tests[i], domain_phis).reshape(1,512)
            ## add in time dimension and make long enough for run time
            stim_pattern = np.array(stim_pattern)[:, None, :]
            stim_pattern = np.repeat(stim_pattern, int(param.sim_time/param.dt), axis=1)

            all_patterns.append(stim_pattern)

        all_patterns = np.vstack(np.asarray(all_patterns).squeeze())
        all_patterns = all_patterns.reshape(1, all_patterns.shape[0], all_patterns.shape[1])

        print(f'Run time: {T}')

        with nengo_dl.Simulator(net_trained, minibatch_size=1, seed=param.seed) as sim_trained:
            sim_trained.run(T, data={da.inpt: all_patterns})

            results = {'output':sim_trained.data[da.probe_out],
                        'input':sim_trained.data[da.probe_stim], 
                        'domain': domain,
                        'domain_phis': domain_phis}
            
            out_std, test_std, out_peak, test_peak, out_entropy, test_entropy, sims_before_list, sims_after_list = get_data(domain, 
                                                                                                                            domain_phis, 
                                                                                                                            results['output'], 
                                                                                                                            results['input'], 
                                                                                                                            len(param.tests),
                                                                                                                            deepatt=True)

        if param.optimize:
            ## Performance metrics
            ## Accuracy - RMSE between the test and target peak locations on the final timestep
            final_peaks = np.asarray(out_peak)
            test_peaks = np.asarray(test_peak)
            rmse_peak = np.sqrt(((test_peaks - final_peaks)**2).mean())
        
            ## Entropy - Difference between the entropy of the 
            ## input distribution and the entropy of the output distribution 
            ## on the final timestep
            final_ent = np.asarray(out_entropy)
            test_ent = np.asarray(test_entropy)

            ## subtract the initial from the final entropy. 
            ## Good delta entropy scores will be negative indicating a drop in entropy
            delta_entropy = final_ent - test_ent
            mean_delta_entropy = delta_entropy.mean()

            out_data = {
                'out_std': out_std, 
                'test_std': test_std, 
                'out_peak': out_peak, 
                'test_peak': test_peak, 
                'out_entropy': out_entropy, 
                'test_entropy': test_entropy, 
                'sims_before_list': sims_before_list, 
                'sims_after_list': sims_after_list,
                'rmse_peak': rmse_peak,
                'mean_delta_entropy': mean_delta_entropy,
                    }
        if not param.optimize:
            out_data = {
                'out_std': out_std, 
                'test_std': test_std, 
                'out_peak': out_peak, 
                'test_peak': test_peak, 
                'out_entropy': out_entropy, 
                'test_entropy': test_entropy, 
                'sims_before_list': sims_before_list, 
                'sims_after_list': sims_after_list,
            }
            
        return out_data
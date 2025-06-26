## Script for building a Deep Attractor Network.
## Authors: Dr. Madeleine Bartlett
'''
Code Structure:

Trial run script:
- run the network through training, validation and test, measure performance

Optuna sampler:
- select network params -- number of layers, learning rate

**Network builder**:
- based on chosen n_layers, build network 

'''

import nengo
from nengo.network import Network
import numpy as np

class DeepAttractor(Network):
    def __init__(self, n_layers=2,
                 neuron_type=nengo.LIFRate(),
                 ens_neurons=5000, ens_dims=1,
                 encoders=nengo.Default,
                 dec_neurons=None, dec_dims=None,
                 dec_encoders=None,
                 inp_weight=1,
                 trained_params=None,
                 seed=None):
        '''
        Class for generating a recurrently-connected N layer network 

        Must contain at least 2 hidden layers
        '''
        self.n_layers = n_layers
        self.neuron_type = neuron_type
        self.ens_neurons = ens_neurons
        self.ens_dims = ens_dims
        self.encoders = encoders
        self.dec_neurons = dec_neurons
        self.dec_dims = dec_dims
        self.dec_encoders = dec_encoders
        self.inp_weight = inp_weight
        self.trained_params = trained_params
        self.seed = seed
        

        assert n_layers >= 2, f"Number of layers must be at least 2, got: {n_layers}"
        
        self.params_list = []

        ## create list of ensemble names and connection names
        ens_names = []
        for i in range(self.n_layers):
            ens_names.append(f'ens_{i}')

        conn_names = []
        for i in range(self.n_layers-1):
            conn_names.append(f'conn_{i}')

        probe_names = ['probe_' + x for x in ens_names]

        ## Create network
        self.net = nengo.Network(seed=self.seed)
        with self.net:
            ## input node, initialise with input=0
            self.inpt = nengo.Node(lambda t: np.zeros(self.ens_dims))

            ## create the ensemble layers
            for i in range(len(ens_names)):
                if trained_params==None:
                    ens_names[i] = nengo.Ensemble(n_neurons=self.ens_neurons,
                                        dimensions=self.ens_dims,
                                        encoders=self.encoders,
                                        intercepts = nengo.dists.CosineSimilarity(self.ens_dims+2),
                                        radius=1,
                                        neuron_type=self.neuron_type,
                                        label=ens_names[i])
                else:
                    ens_names[i] = nengo.Ensemble(n_neurons=self.ens_neurons,
                                        dimensions=self.ens_dims,
                                        encoders=trained_params[ens_names[i]]['encoders'],
                                        bias=trained_params[ens_names[i]]['bias'],
                                        gain=trained_params[ens_names[i]]['gain'],
                                        # intercepts = nengo.dists.CosineSimilarity(self.ens_dims+2),
                                        radius=1,
                                        neuron_type=self.neuron_type,
                                        label=ens_names[i])

                self.params_list.append(ens_names[i])
                
            ## create connections
            if self.dec_neurons == None:
                ## input to first ensemble
                if trained_params == None:
                    conn_fst = nengo.Connection(self.inpt, ens_names[0], synapse=None, transform=self.inp_weight, label='conn_fst')
                else:
                    conn_fst = nengo.Connection(self.inpt, ens_names[0], synapse=None, transform=trained_params['conn_fst']['transform'], label='conn_fst')
            else:
                ## TODO: design what to do when it comes to saving this population and reloading it. 
                if trained_params:
                    raise NotImplementedError('No option for rebuilding the decoder population')
                ## create place-cell ensemble
                dec = nengo.Ensemble(n_neurons=self.dec_neurons, 
                                     dimensions=self.dec_dims,
                                     radius=1,
                                     encoders=self.dec_encoders,
                                     intercepts=nengo.dists.CosineSimilarity(self.dec_dims+2),
                                     label='place_ensemble',)
                ## connect input to place-cells
                conn_place = nengo.Connection(self.inpt, dec, synapse=None, label='conn_place')
                ## connect place-cells to first ensemble
                conn_fst = nengo.Connection(dec.neurons,  ens_names[0], synapse=None, transform=self.inp_weight, label='conn_fst')

                self.params_list.append(dec)
                self.params_list.append(conn_place)
            
            ## the recurrent connection from the last to the first ensemble
            if trained_params == None:
                conn_lst = nengo.Connection(ens_names[-1], ens_names[0], synapse=0.001, transform=1-self.inp_weight, label='conn_lst')
            else:
                conn_lst = nengo.Connection(ens_names[-1], ens_names[0], synapse=0.001, 
                                            solver=nengo.solvers.NoSolver(trained_params['conn_lst']['solver'].values),
                                            function=trained_params['conn_lst']['function'],
                                            # transform=1-self.inp_weight, 
                                            label='conn_lst')

            self.params_list.append(conn_fst)
            self.params_list.append(conn_lst)

            ## connections between ensembles
            for i in range(len(conn_names)):
                if trained_params == None:
                    conn_names[i] = nengo.Connection(ens_names[i], ens_names[i+1], synapse=None, label=conn_names[i])
                else:
                    conn_names[i] = nengo.Connection(ens_names[i], ens_names[i+1], 
                                                     solver=nengo.solvers.NoSolver(trained_params[conn_names[i]]['solver'].values),
                                                     function=trained_params[conn_names[i]]['function'],
                                                     synapse=None, label=conn_names[i])
                self.params_list.append(conn_names[i])

            ## create probes that will be used to train the network
            self.probe_in = nengo.Probe(self.inpt, label='input probe')
            for i in range(len(probe_names)):
                setattr(self, probe_names[i], nengo.Probe(ens_names[i], label=probe_names[i]))

            if self.dec_neurons==None:
                self.probe_out = nengo.Probe(ens_names[-1], synapse=0.01, label='out probe')
                self.probe_out_nofilt = nengo.Probe(ens_names[-1], label='out no filter probe')
                self.probe_stim = nengo.Probe(self.inpt, label='probe_input')
            else:
                out = nengo.Ensemble(n_neurons=self.ens_neurons,
                                          dimensions=self.dec_dims,
                                          radius=1,
                                          encoders=self.encoders,
                                          label='encoding_pop',)
                conn_out = nengo.Connection(ens_names[0], out, transform=self.dec_encoders.T, synapse=None)

                self.params_list.append(out, conn_out)

                self.probe_out = nengo.Probe(out, synapse=0.01, label='out probe')
                self.probe_out_nofilt = nengo.Probe(out, label='out no filter probe')
                self.probe_stim = nengo.Probe(self.inpt, label='probe_input')


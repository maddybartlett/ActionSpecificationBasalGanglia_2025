## Script containing the different networks tested as the dynamics of the basal ganglia. Author Dr. Madeleine Bartlett
'''
'''

import nengo
import nengo_spa as spa
import nengo_dft
from nengo.config import Config
from nengo.ensemble import Ensemble

import os, sys
sys.path.append(os.path.join("..//"))
from utils import softmax_nonlinearity

import numpy as np

## Requires softmax
class Hopfield(object):
    def __init__(self, mode='continuous', seed=None, 
                 neuron_type=nengo.LIFRate(),
                 ens_neurons=5000, ens_dims=1, 
                 dec_neurons=400, dec_dims=1,
                 encoders=nengo.Default, 
                 m_patterns=None, temp=1,
                 inp_weight=1, 
                 ):
        self.seed = seed
        self.ens_neurons = ens_neurons
        self.ens_dims = ens_dims
        self.dec_neurons = dec_neurons
        self.dec_dims = dec_dims
        self.neuron_type = neuron_type
        self.encoders = encoders
        self.m_patterns = m_patterns
        if m_patterns == None:
            pass
        elif m_patterns.all() != None:
            self.n_patterns=len(m_patterns)
        self.temp = temp
        self.inp_weight = inp_weight
        
        self.mode = mode
        self.stim_pattern = np.zeros(shape=(self.ens_dims,))
        
    def run_model(self, stim_pattern, T=0.5, present_time=None):
        self.stim_pattern = stim_pattern
        if present_time == None:
            present_time = T
            
        config = Config(Ensemble)
        config[Ensemble].neuron_type = self.neuron_type
    
        model = nengo.Network(seed=self.seed)
        with model:
            with config:
                ## input node
                stim = nengo.Node(nengo.processes.PresentInput(stim_pattern, presentation_time=present_time))
                
                ## hopfield layer with N neurons
                ens = nengo.Ensemble(n_neurons=self.ens_neurons, 
                                    dimensions=self.ens_dims, 
                                    radius=1)
                
                ## connect the stimulation to the hopfield layer
                nengo.Connection(stim, ens, synapse=None, transform=self.inp_weight)
                
                if self.mode=='discrete':
                    ## hidden layer with M neurons
                    softmax_node = nengo.Node(softmax_nonlinearity, size_in=self.n_patterns)
                    ## create bidirectional connections - equivalent of a recurrent network
                    ## for this version, we don't need the trained weight matrix. We use the transform to find the encoders and decoders we want
                    nengo.Connection(ens, softmax_node, transform=self.temp*self.m_patterns, synapse=None) ## update immediately
                    nengo.Connection(softmax_node, ens, transform=self.m_patterns.T, synapse=0) ## update after 1 timestep
                elif self.mode=='continuous':
                    ## hidden layer whose encoders create SSP `place` cells, decoding SSP positions on the Hypersphere
                    dec = nengo.Ensemble(n_neurons = self.dec_neurons, 
                                    dimensions = self.dec_dims, 
                                    radius = 1, 
                                    encoders = self.encoders,
                                    intercepts = nengo.dists.CosineSimilarity(self.ens_dims+2))
                    
                    softmax_node = nengo.Node(softmax_nonlinearity, size_in=self.dec_neurons)
                    ## create bidirectional connections - equivalent of a recurrent network
                    ## for this version, we don't need the trained weight matrix. We use the transform to find the encoders and decoders we want
                    nengo.Connection(ens, dec, synapse=None) ## update immediately
                    nengo.Connection(dec.neurons, softmax_node, transform=self.temp, synapse=None)
                    nengo.Connection(softmax_node, ens, transform=self.encoders.T, synapse=0) ## update after 1 timestep
                
                ## probe the network
                p_stim = nengo.Probe(stim, synapse=None)
                ## the output signal tends to be pretty stable so we'll apply a relatively short filter
                p_out = nengo.Probe(ens, synapse=0.01)

        ## Run network 
        sim = nengo.Simulator(model, seed=self.seed)
        with sim:
            sim.run(T)
            
        results = {'output':sim.data[p_out],
                   'input':sim.data[p_stim]}
        return results
    
class Accumulator(object):
    def __init__(self, mode='continuous', seed=None, 
                 neuron_type=nengo.LIFRate(), 
                 ens_neurons=1000, ens_dims=1, 
                 dec_neurons=400, dec_dims=1,
                 encoders=nengo.Default, 
                 IA_neurons=1000, m_patterns=None,
                 inp_weight=1, thresh=0.8, intercept_width=0.15,
                ):
        
        self.m_patterns = m_patterns
        self.seed = seed
        self.neuron_type = neuron_type 
        self.ens_neurons = ens_neurons
        self.ens_dims = ens_dims
        self.dec_neurons = dec_neurons
        self.dec_dims = dec_dims
        self.encoders = encoders
        if m_patterns == None:
            pass
        elif m_patterns.all() != None:
            self.n_patterns=len(m_patterns)
        self.IA_neurons = IA_neurons
        self.inp_weight = inp_weight
        self.thresh = thresh
        self.intercept_width = intercept_width
        
        self.mode = mode
        self.stim_pattern = np.zeros(shape=(self.ens_dims,))
        
    def run_model(self, stim_pattern, T=0.5, present_time=None):
        self.stim_pattern = stim_pattern
        if present_time == None:
            present_time = T
            
        config = Config(Ensemble)
        config[Ensemble].neuron_type = self.neuron_type
        
        model = nengo.Network(seed=self.seed)
        with model:
            with config:
                ## input node
                stim = nengo.Node(nengo.processes.PresentInput(self.stim_pattern, presentation_time=present_time))
                
                ## layers for breaking bundle into saliences
                ens = nengo.Ensemble(n_neurons=self.ens_neurons, dimensions=self.ens_dims, radius=1)
                
                ## connect stimulus to ensemble
                nengo.Connection(stim, ens, synapse=None, transform=self.inp_weight)
                
                ## create output ensemble
                out = nengo.Ensemble(n_neurons=self.ens_neurons, dimensions=self.ens_dims, radius=1)
                
                if self.mode == 'discrete':
                    ## independent accumulator
                    acc = spa.networks.selection.IA(n_neurons=self.IA_neurons, 
                                                    n_ensembles=self.n_patterns, 
                                                    accum_threshold=self.thresh,
                                                    intercept_width=self.intercept_width)
                    
                    nengo.Connection(ens, acc.input, transform=self.m_patterns, synapse=None)
                    
                    ## connect the accumulator to the out ensemble such that the output is a bundle again
                    nengo.Connection(acc.output, out, transform=self.m_patterns.T, synapse=0.01)
                    
                elif self.mode == 'continuous':
                    ## hidden layer of place cells
                    dec = nengo.Ensemble(n_neurons = self.dec_neurons, 
                                            dimensions = self.dec_dims, 
                                            radius = 1, 
                                            encoders = self.encoders,
                                            intercepts = nengo.dists.CosineSimilarity(512+2))
                    
                    ## create bidirectional connections - equivalent of a recurrent network
                    ## for this version, we don't need the trained weight matrix. We use the transform to find the encoders and decoders we want
                    nengo.Connection(ens, dec, synapse=None) ## update immediately
                    
                    ## independent accumulator
                    acc = spa.networks.selection.IA(n_neurons=self.IA_neurons, n_ensembles=self.dec_neurons, accum_threshold=self.thresh)
                
                    nengo.Connection(dec.neurons, acc.input, synapse=None)
                    
                    ## connect the accumulator to the out ensemble such that the output is a bundle again
                    nengo.Connection(acc.output, out, transform=self.encoders.T, synapse=0.01)
                
                ## probe accumulator and output 
                p_stim = nengo.Probe(stim, synapse=None)
                ## the output signal tends to be pretty stable so we'll apply a relatively short filter
                p_out = nengo.Probe(out, synapse=0.01)
            
        ## run the network 
        sim = nengo.Simulator(model, seed=self.seed)
        with sim:
            sim.run(T)
            
        results = {'output':sim.data[p_out],
                   'input':sim.data[p_stim]}
        return results
    
class WTA(object):
    def __init__(self, mode='continuous', seed=None, 
                 neuron_type=nengo.LIFRate(), 
                 ens_neurons=1000, ens_dims=1, 
                 dec_neurons=400, dec_dims=1,
                 encoders=nengo.Default, 
                 WTA_neurons=1000, m_patterns=None,
                 inp_weight=1, thresh=0.001, 
                 inhib_scale=2.0, intercept_width=0.15,
                 ):
        
        self.m_patterns = m_patterns
        self.seed = seed
        self.neuron_type = neuron_type 
        self.ens_neurons = ens_neurons
        self.ens_dims = ens_dims
        self.dec_neurons = dec_neurons
        self.dec_dims = dec_dims
        self.encoders = encoders
        if m_patterns == None:
            pass
        elif m_patterns.all() != None:
            self.n_patterns=len(m_patterns)
        self.WTA_neurons = WTA_neurons
        self.inp_weight = inp_weight
        self.thresh = thresh
        self.inhib_scale = inhib_scale
        self.intercept_width = intercept_width
        
        self.mode = mode
        self.stim_pattern = np.zeros(shape=(self.ens_dims,))
        
        
    def run_model(self, stim_pattern, T=0.5, present_time=None):
        self.stim_pattern = stim_pattern
        if present_time == None:
            present_time = T
            
        config = Config(Ensemble)
        config[Ensemble].neuron_type = self.neuron_type
        
        model = nengo.Network(seed=self.seed)
        with model:
            with config:
                ## input node
                stim = nengo.Node(nengo.processes.PresentInput(self.stim_pattern, presentation_time=present_time))
                
                ## layers for breaking bundle into saliences
                ens = nengo.Ensemble(n_neurons=self.ens_neurons, dimensions=self.ens_dims, radius=1)
                
                ## connect stimulus to ensemble
                nengo.Connection(stim, ens, synapse=None, transform=self.inp_weight)
                
                ## create output ensemble
                out = nengo.Ensemble(n_neurons=self.ens_neurons, dimensions=self.ens_dims, radius=1)
                
                if self.mode == 'discrete':                
                    ## winner -take-all
                    wta = spa.networks.selection.WTA(n_neurons=self.WTA_neurons, 
                                                     n_ensembles=self.n_patterns, 
                                                     threshold=self.thresh, 
                                                     inhibit_scale=self.inhib_scale,
                                                     intercept_width=self.intercept_width,)
                    
                    nengo.Connection(ens, wta.input, transform=self.m_patterns, synapse=None)
                    
                    ## connect the accumulator to the out ensemble such that the output is a bundle again
                    nengo.Connection(wta.output, out, transform=self.m_patterns.T, synapse=0.01)
                    
                elif self.mode == 'continuous':
                    ## hidden layer of place cells
                    dec = nengo.Ensemble(n_neurons = self.dec_neurons, 
                                            dimensions = self.dec_dims, 
                                            radius = 1, 
                                            encoders = self.encoders,
                                            intercepts = nengo.dists.CosineSimilarity(512+2))
                    
                    ## create bidirectional connections - equivalent of a recurrent network
                    ## for this version, we don't need the trained weight matrix. We use the transform to find the encoders and decoders we want
                    nengo.Connection(ens, dec, synapse=None) ## update immediately
                    
                    ## independent accumulator
                    wta = spa.networks.selection.WTA(n_neurons=self.WTA_neurons, 
                                                     n_ensembles=self.dec_neurons, 
                                                     threshold=self.thresh, 
                                                     inhibit_scale=self.inhib_scale,
                                                     intercept_width=self.intercept_width,)
                
                    nengo.Connection(dec.neurons, wta.input, synapse=None)
                    
                    ## connect the accumulator to the out ensemble such that the output is a bundle again
                    nengo.Connection(wta.output, out, transform=self.encoders.T, synapse=0.01)
   
                ## probe accumulator and output 
                p_stim = nengo.Probe(stim, synapse=None)
                ## the output signal tends to be pretty stable so we'll apply a relatively short filter
                p_out = nengo.Probe(out, synapse=0.01)
            
        ## run the network 
        sim = nengo.Simulator(model, seed=self.seed)
        with sim:
            sim.run(T)
            
        results = {'output':sim.data[p_out],
                   'input':sim.data[p_stim]}
        return results
    
class ShallowAttractor(object):
    def __init__(self, samples, targets,
                 mode=None, seed=None,
                 neuron_type=nengo.LIFRate(),
                 ens_neurons=5000, ens_dims=1, 
                 dec_neurons=None, dec_dims=None,
                 encoders=nengo.Default, 
                 inp_weight=1,
                 ):
        self.samples = samples
        self.targets = targets
        self.ens_neurons = ens_neurons
        self.ens_dims = ens_dims
        self.neuron_type = neuron_type
        self.encoders = nengo.Default
        self.inp_weight = inp_weight
        self.seed = seed

        self.stim_pattern = np.zeros(shape=(self.ens_dims,))
            
        
    def run_model(self, stim_pattern, T=0.5, present_time=None):
        self.stim_pattern = stim_pattern
        if present_time == None:
            present_time = T
            
        config = Config(Ensemble)
        config[Ensemble].neuron_type = self.neuron_type
    
        model = nengo.Network(seed=self.seed)
        with model:
            with config:
                ## input node
                stim = nengo.Node(nengo.processes.PresentInput(self.stim_pattern, presentation_time=present_time))
                
                ## neuron layer for representing bundle
                ens = nengo.Ensemble(n_neurons=self.ens_neurons, 
                                    dimensions=self.ens_dims, 
                                    radius=1,
                                    encoders=self.encoders,
                                    intercepts = nengo.dists.CosineSimilarity(self.ens_dims+2) )
                
                ## connect the stimulation to the ensemble layer
                nengo.Connection(stim, ens, synapse=None, transform=self.inp_weight)
                
                ## requires training on examples of noisy bundles and their valid SSP targets, but does not require direct access to distribution
                conn = nengo.Connection(ens, ens, 
                                        eval_points=self.samples.squeeze(), 
                                        function=self.targets.squeeze(), 
                                        synapse=0.1, transform=1-self.inp_weight)
                
                ## probe the network
                p_stim = nengo.Probe(stim, synapse=None)
                ## the output signal tends to be pretty stable so we'll apply a relatively short filter
                p_out = nengo.Probe(ens, synapse=0.01)

        ## Run network 
        sim = nengo.Simulator(model, seed=self.seed)
        with sim:
            sim.run(T)
            
        results = {'output':sim.data[p_out],
                   'input':sim.data[p_stim]}
        return results

class DNF(object):
    def __init__(self, mode='continuous', seed=None, 
                 neuron_type=nengo.LIFRate(), 
                 ens_neurons=1000, ens_dims=1,
                 dec_neurons=400, dec_dims=None,
                 dnf_h=-20, dnf_global_inhib=10, dnf_tau=0.001,
                 kernel_excit=10, kernel_inhib=0,
                 exc_width=5, inh_width=10,
                 encoders=nengo.Default, inp_weight=None,
                 optimize=False,
                 ):
        
        self.seed = seed
        self.neuron_type = neuron_type 
        self.ens_neurons = ens_neurons
        self.ens_dims = ens_dims
        self.dnf_neurons = dec_neurons
        self.dnf_h = dnf_h
        self.dnf_global_inhib = dnf_global_inhib
        self.dnf_tau = dnf_tau
        self.encoders = encoders
        self.inp_weight = inp_weight
        self.kernel_excit = kernel_excit
        self.kernel_inhib = kernel_inhib
        self.exc_width = exc_width
        self.inh_width = inh_width
        
        self.mode = mode
        self.optimize=optimize
        
        
    def run_model(self, stim_pattern, T=0.5, present_time=None):
        if present_time == None:
            present_time = T
            
        config = Config(Ensemble)
        config[Ensemble].neuron_type = self.neuron_type
        
        model = nengo.Network(seed=self.seed)
        with model:
            with config:
                ## input node
                if self.optimize:
                    stim = nengo.Node(lambda t: stim_pattern if t<=present_time else np.zeros(self.ens_dims))
                else:    
                    stim = nengo.Node(nengo.processes.PresentInput(stim_pattern, presentation_time=present_time))
                
                ## layers for breaking bundle into saliences
                dnf = nengo_dft.DFT(shape=[self.dnf_neurons], h=self.dnf_h, 
                                    global_inh=self.dnf_global_inhib, tau=self.dnf_tau)
                dnf.add_kernel(exc=self.kernel_excit, inh=self.kernel_inhib, 
                               exc_width=self.exc_width, inh_width=self.inh_width)
                
                ## connect stimulus to dft
                nengo.Connection(stim, dnf.s, transform=self.encoders, synapse=None)

                ## create output ensemble
                out = nengo.Node(None, size_in=self.ens_dims)
                #out = nengo.Ensemble(n_neurons=self.ens_neurons, dimensions=self.ens_dims, radius=1)
                
                nengo.Connection(dnf.g.neurons, out, transform=self.encoders.T, synapse=None)

                ## probe accumulator and output 
                p_stim = nengo.Probe(stim, synapse=None)
                ## the output signal tends to be pretty stable so we'll apply a relatively short filter
                p_out = nengo.Probe(out, synapse=0.01)
            
        ## run the network 
        sim = nengo.Simulator(model, seed=self.seed)
        with sim:
            sim.run(T)
            
        results = {'output':sim.data[p_out],
                   'input':sim.data[p_stim]}
        return results

        
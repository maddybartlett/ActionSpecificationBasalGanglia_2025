## Author: Dr Madeleine Bartlett
import numpy as np
import nengo
import nengo_dft
from nengo.config import Config
from nengo.ensemble import Ensemble
from nengo.network import Network

class DNF(Network):
    """A dynamic neural fields network, performing action selection on a bundle of salience-weighted action SSPs.

    Parameters
    ----------
    mode : string, optional (Default: 'continuous')
        The mode for the network to operate in, either continuous or discrete to choose between.
    seed: int, optional (Default: None)
        seed for controlling randomness in network initialisation
    neuron_type : nengo obj, optional (Default: nengo.LIFRate())
        The type of neurons used in the output ensemble. (The DNF network always uses Sigmoid neurons).
    ens_neurons : int, optional (Default: 1000)
        The number of neurons in the output ensemble.
    ens_dims : int, optional (Default: 1)
        Dimensionality of the output ensemble. Should match the dimensionality of the action space.
    dec_neurons : int, optional (Default: 400)
        The number of neurons in the dnf. Equates to the number of place cells used to decode the SSP bundle
        into the salience distribution.
    dnf_h: int, optional (Default: -20)
    dnf_global_inhib: int, optional (Default: 10)
    dnf_tau: float, optional (Default: 0.001)
    kernel_excit: int, optional (Default: 10)
    kernel_inhib: int, optional (Default: 10)
    encoders: array-like, optional (Default: nengo.Default)
        A N X M size array of the encoders for the dnf neurons. 
    """
    def __init__(self, seed=None, 
                 neuron_type=nengo.LIFRate(), 
                 ens_dims=1,
                 dec_neurons=400, 
                 dnf_h=-20, dnf_global_inhib=10, dnf_tau=0.001,
                 kernel_excit=10, kernel_inhib=0,
                 exc_width=5, inh_width=10,
                 encoders=nengo.Default,
                 radius=1.0,
                 ):
        
        self.seed = seed
        self.neuron_type = neuron_type 
        self.ens_dims = ens_dims
        self.dnf_neurons = dec_neurons
        self.dnf_h = dnf_h
        self.dnf_global_inhib = dnf_global_inhib
        self.dnf_tau = dnf_tau
        self.encoders = encoders
        self.kernel_excit = kernel_excit
        self.kernel_inhib = kernel_inhib
        self.exc_width = exc_width
        self.inh_width = inh_width
        self.radius = radius
            
        config = Config(Ensemble)
        config[Ensemble].neuron_type = self.neuron_type

        # self.out_bundle = np.zeros(self.ens_dims)
        
        self.model = nengo.Network(seed=self.seed)
        with self.model:
            # self.state = np.zeros(self.ens_dims) 
            with config:

                ## input node
                self.input = nengo.Node(label="input", size_in=self.ens_dims)
                
                ## layers for breaking bundle into saliences
                self.dnf = nengo_dft.DFT(shape=[self.dnf_neurons], h=self.dnf_h, 
                                    global_inh=self.dnf_global_inhib, tau=self.dnf_tau, 
                                    radius=self.radius)
                self.dnf.add_kernel(exc=self.kernel_excit, inh=self.kernel_inhib, 
                               exc_width=self.exc_width, inh_width=self.inh_width)
                
                ## connect stimulus to dft
                nengo.Connection(self.input, self.dnf.s, transform=self.encoders, synapse=None)

                ## create output node for interacting with outside networks 
                self.output = nengo.Node(label="output", size_in=self.ens_dims)
                ## connect neuron activities to output and re-encode as SSP bundle 
                nengo.Connection(self.dnf.g.neurons, self.output, transform=self.encoders.T, synapse=0.01)
  

class DNF_BG(Network):
    """An action selection model using the DNF to perform competition resolution. 
    Takes, as input, a bundle of salience-weighted action SSPs.

    Parameters
    ----------
    neuron_type : nengo obj, optional (Default: nengo.LIFRate())
        The type of neurons used in the output ensemble. (The DNF network always uses Sigmoid neurons).
    dnf_params :  dictionary, optional (DNF net has its own defaults)
        A dictionary of the values for the parameters of the DNF network
    dopamine : float, optional (Default: 0.2)
        Amount of dopamine signal influencing the strength of the striatal connections. 
        TODO: find a way to change this on the fly so that we can link it with the tderror signal
    ssp_dim : int, optional (Default: 1)
        Dimensionality of the SSP input. Should be the same as the ens_dims parameter of the DNF
    """

    def __init__(self, seed=None,
                 neuron_type=nengo.LIFRate(),
                 dnf_params=None,
                 dopamine=0.2,
                 encoders=nengo.Default,
                 d1_weight=1.0,
                 d2_weight=1.0,
                ):
        self.seed = seed
        self.dopamine = dopamine
        self.ssp_dim = dnf_params['ens_dims']
        self.encoders = encoders

        self.gaba = None #0.008
        self.ampa = None #0.002

        config = Config(Ensemble)
        config[Ensemble].neuron_type = neuron_type

        # self.model = nengo.Network(seed=self.seed)
        # with self.model:
        with config:

            ## create an input node
            self.input = nengo.Node(label="input", size_in=self.ssp_dim)

            ## create a population of D1 neurons, these will be made to fire faster in the presence of dopamine
            striatum_d1 = DNF(**dnf_params,
                     encoders=self.encoders)
            ## create a population of D2 neurons, these will fire less frequently in the presence of dopamine
            striatum_d2 = DNF(**dnf_params,
                     encoders=self.encoders)
            
            ## now we need the indirect pathway. 
            gpe = nengo.Ensemble(n_neurons=1000,
                            dimensions=self.ssp_dim,
                            radius=1.0,
                            neuron_type=nengo.LIFRate()
                            )

            stn = nengo.Ensemble(n_neurons=1000,
                            dimensions=self.ssp_dim,
                            radius=1.0,
                            neuron_type=nengo.LIFRate()
                            )

            ## next we'll set up the direct pathway, creating the GPi and connecting the D1 neurons directly to it
            gpi = nengo.Ensemble(n_neurons=1000,
                                dimensions=self.ssp_dim,
                                radius=1.0,
                                neuron_type=nengo.LIFRate()
                                )

            ## finally, we need an output node to collect the result
            self.output = nengo.Node(size_in=self.ssp_dim)

            ## connect input to striatum and stn
            nengo.Connection(self.input, striatum_d1.input, transform=d1_weight+self.dopamine, synapse=None)
            nengo.Connection(self.input, striatum_d2.input, transform=d2_weight-self.dopamine, synapse=None)
            nengo.Connection(self.input, stn, synapse=None)
            ## indirect pathway connections
            nengo.Connection(striatum_d2.output, gpe, transform=-1.0, synapse=self.gaba)
            nengo.Connection(stn, gpe, transform=1.0, synapse=self.ampa)
            nengo.Connection(gpe, gpi, transform=-1.0, synapse=self.gaba)
            ## hyperdirect pathway connections
            nengo.Connection(stn, gpi, transform=1.0, synapse=self.ampa)
            ## direct pathway connection
            nengo.Connection(striatum_d1.output, gpi, transform=-1.0, synapse=self.gaba)
            ## output connection
            nengo.Connection(gpi, self.output, transform=-3.0, synapse=None)


class DNF_BG_DYNDA(Network):
    """An action selection model using the DNF to perform competition resolution. 
    Takes, as input, a bundle of salience-weighted action SSPs and a float value
    representing the amount of dopamine.

    Parameters
    ----------
    neuron_type : nengo obj, optional (Default: nengo.LIFRate())
        The type of neurons used in the output ensemble. (The DNF network always uses Sigmoid neurons).
    dnf_params :  dictionary, optional (DNF net has its own defaults)
        A dictionary of the values for the parameters of the DNF network
    ssp_dim : int, optional (Default: 1)
        Dimensionality of the SSP input. Should be the same as the ens_dims parameter of the DNF
    """

    def __init__(self, seed=None,
                 neuron_type=nengo.LIFRate(),
                 num_neurons=1000,
                 dnf_params=None,
                 encoders=nengo.Default,
                 d1_weight=1.0,
                 d2_weight=1.0,
                ):
        self.seed = seed
        self.neuron_type = neuron_type
        self.num_neurons = num_neurons
        self.ssp_dim = dnf_params['ens_dims']
        self.encoders = encoders

        self.gaba = 0.008
        self.ampa = 0.002

        config = Config(Ensemble)
        config[Ensemble].neuron_type = neuron_type

        # self.model = nengo.Network(seed=self.seed)
        # with self.model:
        with config:

            ## create an input node
            self.input = nengo.Node(label="input", size_in=self.ssp_dim)

            ## create an dopamine node
            self.dopamine = nengo.Node(label="dopamine", size_in=1)

            ## ensemble where we'll multiply the input by the weight +/- dopamine
            inp_times_dop = nengo.Ensemble(n_neurons=self.num_neurons, dimensions=self.ssp_dim+1, neuron_type=nengo.Direct(), radius=1.0)

            ## create a population of D1 neurons, these will be made to fire faster in the presence of dopamine
            self.striatum_d1 = DNF(**dnf_params,
                     encoders=self.encoders)
            ## create a population of D2 neurons, these will fire less frequently in the presence of dopamine
            self.striatum_d2 = DNF(**dnf_params,
                     encoders=self.encoders)
            
            ## now we need the indirect pathway. 
            self.gpe = nengo.Ensemble(n_neurons=self.num_neurons,
                            dimensions=self.ssp_dim,
                            radius=1.0,
                            )

            self.stn = nengo.Ensemble(n_neurons=self.num_neurons,
                            dimensions=self.ssp_dim,
                            radius=1.0,
                            )

            ## next we'll set up the direct pathway, creating the GPi and connecting the D1 neurons directly to it
            self.gpi = nengo.Ensemble(n_neurons=self.num_neurons,
                                dimensions=self.ssp_dim,
                                radius=1.0,
                                )

            ## finally, we need an output node to collect the result
            self.output = nengo.Node(size_in=self.ssp_dim)

            def product_d1(x):
                return x[:self.ssp_dim] * (d1_weight + x[-1])
            
            def product_d2(x):
                return x[:self.ssp_dim] * (d2_weight - x[-1])

            ## transform the input
            nengo.Connection(self.input, inp_times_dop[:self.ssp_dim], synapse = None)
            nengo.Connection(self.dopamine, inp_times_dop[-1], synapse = None)
            ## connect input to striatum and stn
            nengo.Connection(inp_times_dop, self.striatum_d1.input, function=product_d1, synapse=None)
            nengo.Connection(inp_times_dop, self.striatum_d2.input, function=product_d2, synapse=None)
            nengo.Connection(self.input, self.stn, synapse=None)
            ## indirect pathway connections
            nengo.Connection(self.striatum_d2.output, self.gpe, transform=-1.0, synapse=self.gaba)
            nengo.Connection(self.stn, self.gpe, transform=1.0, synapse=self.ampa)
            nengo.Connection(self.gpe, self.gpi, transform=-1.0, synapse=self.gaba)
            nengo.Connection(self.gpe, self.stn, transform=-1.0, synapse=self.gaba)
            ## hyperdirect pathway connections
            nengo.Connection(self.stn, self.gpi, transform=1.0, synapse=self.ampa)
            ## direct pathway connection
            nengo.Connection(self.striatum_d1.output, self.gpi, transform=-1.0, synapse=self.gaba)
            ## output connection
            nengo.Connection(self.gpi, self.output, transform=-3.0, synapse=None)


## Script for setting the Optuna samplers. Author Dr. Madeleine Bartlett

import optuna
from typing import Any, Dict

def sample_hop(trial: optuna.Trial) -> Dict[str, Any]:
    inp_weight = trial.suggest_float("inp_weight", 1e-9, 2, log=True)
    temp = trial.suggest_float("temp", 0.1, 10, log=False)
    ens_neurons = trial.suggest_categorical("ens_neurons", [1000,2000,5000])

    return{
        'inp_weight': inp_weight,
        'temp': temp,
        'ens_neurons': ens_neurons,
    }

def sample_ia(trial: optuna.Trial) -> Dict[str, Any]:
    inp_weight = trial.suggest_float("inp_weight", 1e-9, 2, log=True)
    thresh = trial.suggest_float("thresh", 0.1, 1.5, log=False)
    intercept_width = trial.suggest_float("intercept_width", 1e-3, 1.0, log=False)
    ens_neurons = trial.suggest_categorical("ens_neurons", [1000,2000,5000])
    net_neurons = trial.suggest_categorical("net_neurons", [50,100,200,500])

    return{
        'inp_weight': inp_weight,
        'thresh': thresh,
        'intercept_width': intercept_width,
        'ens_neurons': ens_neurons,
        'net_neurons': net_neurons,
    }

def sample_wta(trial: optuna.Trial) -> Dict[str, Any]:
    inp_weight = trial.suggest_float("inp_weight", 1e-9, 2, log=True)
    thresh = trial.suggest_float("thresh", 0.1, 1.5, log=False)
    inhib_scale = trial.suggest_float("inhib_scale", 0.1, 5.0, log=False)
    intercept_width = trial.suggest_float("intercept_width", 1e-3, 1.0, log=False)
    ens_neurons = trial.suggest_categorical("ens_neurons", [1000,2000,5000])
    net_neurons = trial.suggest_categorical("net_neurons", [50,100,200,500])

    return{
        'inp_weight': inp_weight,
        'thresh': thresh,
        'inhib_scale': inhib_scale,
        'intercept_width': intercept_width,
        'ens_neurons': ens_neurons,
        'net_neurons': net_neurons,
    }

def sample_dnf(trial: optuna.Trial) -> Dict[str, Any]:
    inp_weight = trial.suggest_float("inp_weight", 1e-9, 2, log=True)
    ens_neurons = trial.suggest_categorical("ens_neurons", [1000,2000,5000])
    dnf_h = trial.suggest_float("dnf_h", -30, 30, log=False)
    dnf_global_inhib = trial.suggest_float("dnf_global_inhib", 0, 20, log=False)
    dnf_tau = trial.suggest_float("dnf_tau", 1e-5, 0.5, log=True)
    kernel_excit = trial.suggest_float("kernel_excit", 0, 20, log=False)
    kernel_inhib = trial.suggest_float("kernel_inhib", 0, 20, log=False)
    exc_width = trial.suggest_float("kernel_inhib", 0, 20, log=False)
    inh_width = trial.suggest_float("kernel_inhib", 0, 20, log=False)
    
    return{
        'inp_weight': inp_weight,
        'ens_neurons': ens_neurons,
        'dnf_h': dnf_h, 
        'dnf_global_inhib': dnf_global_inhib,
        'dnf_tau': dnf_tau,
        'kernel_excit': kernel_excit,
        'kernel_inhib': kernel_inhib,
        'exc_width': exc_width,
        'inh_width': inh_width,
    }

def sample_sa(trial: optuna.Trial) -> Dict[str, Any]:
    inp_weight = trial.suggest_float("inp_weight", 1e-9, 2, log=True)
    ens_neurons = trial.suggest_categorical("ens_neurons", [1000,2000,5000])

    return{
        'inp_weight': inp_weight,
        'ens_neurons': ens_neurons,
    }

def sample_da(trial: optuna.Trial) -> Dict[str, Any]:
    n_layers = trial.suggest_int("n_layers", 2, 10)
    learn_rate = trial.suggest_float("learn_rate", 1e-9, 0.5, log=True)
    inp_weight = trial.suggest_float("inp_weight", 1e-9, 1.5, log=True)
    ens_neurons = trial.suggest_categorical("ens_neurons", [1000,2000,5000])

    return{
        'n_layers': n_layers,
        'learn_rate': learn_rate,
        'inp_weight': inp_weight,
        'ens_neurons': ens_neurons,
    }

def sample_sad(trial: optuna.Trial) -> Dict[str, Any]:
    inp_weight = trial.suggest_float("inp_weight", 1e-9, 2, log=True)

    return{
        'inp_weight': inp_weight,
    }
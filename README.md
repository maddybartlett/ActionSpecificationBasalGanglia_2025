# Action Specification Basal Ganglia 2025

Code repository to accompany publication titled "A Computational Model of Action Specification in the Basal Ganglia" by Anonymised for Review.

This project was supported by collaborative research funding from Anonymised for Review.

## Requirements:

- Python == 3.7.16
- nengo == 3.2.0
- nengo_dl == 3.6.0
- nengo_dft == 0.0.1 (from https://github.com/tcstewar/nengo-dft/tree/main)
- nengo_spa == 1.3.0
- nengo_extras == 0.5.0
- optuna == 4.0.0
- sspspace == 0.1 (from https://github.com/ctn-waterloo/sspspace)
- tensorflow == 2.10.1

## Hyperparameter Optimisation

Hyperparameter optimisation experiments utilised the scripts in *experiment_1_model_development*. 

These experiments were run on a remote CPU using SLURM. Shell scripts (.sh) were used to run the experiments. 

### Analysis

.pkl files are provided containing the data from the optuna optimisation experiments. Jupyter Notebook *explore_hpo_data.ipynb* has optional cells for loading the .pkl data. Users must ignore or skip cells 4&5 where data from .npz files generated during optimisation are loaded, converted to dataframes and saved as .pkl files.  

## Simulation Experiments to Demonstrate Network Properties

The dnf network was chosen as the basal ganglia network dynamics and incorporated into the Nengo implementation of the Gurney, Prescott & Redgrave (2001) model (see Stewart, Choo & Eliasmith, 2010). 

![Basal Ganglia model with DNF in striatal layer](https://github.com/maddybartlett/ActionSpecificationBasalGanglia_2025/blob/main/figs/bg_dnf_schematic.pdf)

The scripts for these simulations are in *experiment_2_model_properties*. 

## Citation

Cite this work as:

```bibtex
Anonymised for Review - work not yet published.
```

## Script for generating the pareto front plot shown in figure 5.
## Authors: Dr. Madeleine Bartlett
'''
'''

import os, sys
sys.path.append(os.path.join("..//"))

import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import Rectangle

import scipy.stats as st
from scipy.stats import bootstrap


## Load the data
dir_Hop = r'..//data//HOP_HPO_31_10_2024_17_42_24'
dir_IA = r'..//data//IA_HPO_31_10_2024_18_31_22'
dir_WTA = r'..//data//WTA_HPO_31_10_2024_18_31_22'
dir_DNF = r'..//data//WTA_HPO_31_10_2024_18_31_22'
dir_SA = r'..//data//SA_HPO_31_10_2024_22_31_55'
dir_SAD = r'..//data//SAD_HPO_31_10_2024_23_27_07'
dir_DA = r'..//data//DA_HPO_04_11_2024_17_02_31'
dir_Hop = r'..//data//HOP_HPO_11_11_2024_13_39_52'
dir_IA = r'..//data//IA_HPO_11_11_2024_14_26_44'
dir_WTA = r'..//data//WTA_HPO_11_11_2024_22_14_49'
dir_DNF = r'..//data//DNF_HPO_15_11_2024_14_05_47'
dir_SA = r'..//data//SA_HPO_11_11_2024_18_15_39'
dir_SAD = r'..//data//SAD_HPO_11_11_2024_21_51_15'
dir_DA = r'..//data//DA_HPO_12_11_2024_14_43_46'


def get_data(folder):
    ''' Collect the npz files and convert them to a pandas data frame '''
    allData=[]
    i=0
    for filename in os.listdir(folder):
        if '.npz' in filename:
            filepath = os.path.join(folder, filename)
            arr=np.load(filepath, allow_pickle=True)
            
            vals=[]
            if i==0:
                header = arr.files
                df = pd.DataFrame(header)
            
            for item in arr.files:
                vals.append(arr[item])
            
            allData.append(vals)
            i+=1
    return allData, header

def my_root_mean(xs, axis=-1):
  return np.sqrt(np.mean(xs, axis))

def get_square_errors(df):
  ## create a column containing the 50 square error terms for each optuna trial
  df['square_error'] = (df['test_peak'] - df['out_peak'])**2
  return df
  
def get_delta_entropy(df):
  ## create a column containing the 50 delta entropy terms for each optuna trial
  df['delta_entropy'] = df['out_entropy'] - df['test_entropy']
  return df
    
def get_cis(df):
  df = get_square_errors(df)
  df = get_delta_entropy(df)

  rmse_cis_list = []
  entropy_cis_list = []
  ## for each optuna trial
  for i in range(len(df)):
    ## collect the 50 square errors and delta entropy terms
    ys = df['square_error'][i].squeeze()
    xs = df['delta_entropy'][i].squeeze()

    ## bootstrap across the 50 terms
    ## use my_root_mean for the square error terms
    rmse_retval = bootstrap((ys,), # The data has to be stored as a sequence.
                      statistic=my_root_mean, # Working with the mean function, but can be any callable.
                      vectorized=True, # working n-d data (2d in this case)
                      axis=0, # axis=0 because each row is a sample.
                      n_resamples=1000,
                      ) 
    
    entropy_retval = bootstrap((xs,), # The data has to be stored as a sequence.
                      statistic=np.mean, # Working with the mean function, but can be any callable.
                      vectorized=True, # working n-d data (2d in this case)
                      axis=0, # axis=0 because each row is a sample.
                      n_resamples=1000,
                      ) 

    rmse_cis = np.vstack(
        (df['rmse_peak'][i] - rmse_retval.confidence_interval.low,
        rmse_retval.confidence_interval.high - df['rmse_peak'][i])
    ).T
    
    entropy_cis = np.vstack(
        (df['mean_delta_entropy'][i] - entropy_retval.confidence_interval.low,
        entropy_retval.confidence_interval.high - df['mean_delta_entropy'][i])
    ).T
    
    ## collect cis' into lists
    rmse_cis_list.append(rmse_cis)
    entropy_cis_list.append(entropy_cis)
    
  df['rmse_cis'] = rmse_cis_list
  df['entropy_cis'] = entropy_cis_list
  
  return df

def get_score(df):
    df['score'] = df['mean_delta_entropy'] + df['rmse_peak']
    df['score'] = df['score'].astype(float)
    return df
    

def plot_pareto(df, network, marker):
    for i in range(len(pareto_front[network])):
        x = df[df['optuna_trial_number']==pareto_front[network][i]]['mean_delta_entropy']
        y = df[df['optuna_trial_number']==pareto_front[network][i]]['rmse_peak']
        if i == 0:            
            plt.errorbar(x, y, 
                         xerr=np.asarray(df[df['optuna_trial_number']==pareto_front[network][i]]['entropy_cis'])[0].T, 
                         yerr=np.asarray(df[df['optuna_trial_number']==pareto_front[network][i]]['rmse_cis'])[0].T,  
                         fmt=marker, capsize=5, markeredgecolor='black', markeredgewidth=0.5,
                         color=COLOURS[network], label=network, ms=15)
            
        else:
            plt.errorbar(x, y, 
                         xerr=np.asarray(df[df['optuna_trial_number']==pareto_front[network][i]]['entropy_cis'])[0].T, 
                         yerr=np.asarray(df[df['optuna_trial_number']==pareto_front[network][i]]['rmse_cis'])[0].T, 
                         fmt=marker, capsize=5, markeredgecolor='black', markeredgewidth=0.5,
                         color=COLOURS[network], ms=15)
    plt.xlabel('Mean Delta Entropy', fontsize='xx-large')
    plt.xticks(fontsize='x-large')
    plt.ylabel('RMSE', fontsize='xx-large')
    plt.yticks(fontsize='x-large')


def plot_best(df, network, marker):
    top_10 = np.argsort(df['score'].values, 0)[:10]

    for i in top_10:

        x = df.iloc[i]['mean_delta_entropy']
        y = df.iloc[i]['rmse_peak']
        if i == top_10[0]:
            plt.errorbar(x,y, 
                         xerr=np.asarray(df.iloc[i]['entropy_cis'])[0].reshape(-1,1), 
                         yerr=np.asarray(df.iloc[i]['rmse_cis'])[0].reshape(-1,1),
                         fmt=marker, ms=10, capsize=10, markeredgecolor='black', markeredgewidth=0.5, color=COLOURS[network], label=network)
        else:
            plt.errorbar(x,y, 
                         xerr=np.asarray(df.iloc[i]['entropy_cis'])[0].reshape(-1,1), 
                         yerr=np.asarray(df.iloc[i]['rmse_cis'])[0].reshape(-1,1),
                         fmt=marker, ms=15, capsize=10, markeredgecolor='black', markeredgewidth=0.5, color=COLOURS[network])
        
    x = df.iloc[top_10[0]]['mean_delta_entropy']
    y = df.iloc[top_10[0]]['rmse_peak']
    plt.errorbar(x,y, 
                 xerr=np.asarray(df.iloc[top_10[0]]['entropy_cis'])[0].reshape(-1,1), 
                 yerr=np.asarray(df.iloc[top_10[0]]['rmse_cis'])[0].reshape(-1,1),
                 fmt=marker, ms=15, capsize=10, markeredgecolor='black', markeredgewidth=4, color=COLOURS[network])

    plt.xlabel('Mean Delta Entropy', fontsize='xx-large')
    plt.xticks(fontsize='x-large')
    plt.ylabel('RMSE', fontsize='xx-large')
    plt.yticks(fontsize='x-large')
    # plt.legend(bbox_to_anchor=(1.0,1.0), fontsize='x-large')
    plt.legend(bbox_to_anchor=(1.0,1.2), ncol=4, fontsize='x-large')
    # plt.legend(fontsize='x-large')

### CONVERT NPZ DATA TO PANDAS DATAFRAMES ###
data_hop, header_hop = get_data(dir_Hop)   
df_hop = pd.DataFrame(data_hop, columns=header_hop)

data_acc, header_acc = get_data(dir_IA)   
df_acc = pd.DataFrame(data_acc, columns=header_acc)

data_wta, header_wta = get_data(dir_WTA)   
df_wta = pd.DataFrame(data_wta, columns=header_wta)

data_dnf, header_dnf = get_data(dir_DNF)   
df_dnf = pd.DataFrame(data_dnf, columns=header_dnf)

data_sa, header_sa = get_data(dir_SA)   
df_sa = pd.DataFrame(data_sa, columns=header_sa)

data_sad, header_sad = get_data(dir_SAD)   
df_sad = pd.DataFrame(data_sad, columns=header_sad)

data_da, header_da = get_data(dir_DA)   
df_da = pd.DataFrame(data_da, columns=header_da)

### ADD COLUMN WITH CONFIDENCE INTERVALS FOR OPTIMISATION METRICS ###
df_hop = get_cis(df_hop)
df_acc = get_cis(df_acc)
df_wta = get_cis(df_wta)
df_dnf = get_cis(df_dnf)
df_sa = get_cis(df_sa)
df_sad = get_cis(df_sad)
df_da = get_cis(df_da)


### ADD COLUMN WITH OVERALL SCORE (RMSE + DELTA ENTROPY) ###
df_hop = get_score(df_hop)
df_acc = get_score(df_acc)
df_wta = get_score(df_wta)
df_dnf = get_score(df_dnf)
df_sa = get_score(df_sa)
df_sad = get_score(df_sad)
df_da = get_score(df_da)


### CREATE PLOT FOR FIGURE 5 ###
plt.figure(figsize=(12,6))
ax1 = plt.subplot(121)

plot_pareto(df_hop, 'Hopfield', 's')
plot_pareto(df_acc, 'Accumulator', '*')
plot_pareto(df_wta, 'WTA', '^')
plot_pareto(df_dnf, 'DNF', 'd')
plot_pareto(df_sa, 'Shallow Attractor',  'x')
plot_pareto(df_sad, 'Shallow Attractor Place', 'X')
plot_pareto(df_da, 'Deep Attractor', 'o')
ax1.add_patch(Rectangle((-0.7, -0.1), 0.85, 0.6, facecolor="white", edgecolor="#882d4b", linewidth=2.0))

plt.subplot(122)
plot_best(df_hop, 'Hopfield', 's')
plot_best(df_acc, 'Accumulator', '*')
plot_best(df_wta, 'WTA', '^')
plot_best(df_dnf, 'DNF', 'd')
plot_best(df_sa, 'Shallow Attractor', 'x')
plot_best(df_sad, 'Shallow Attractor Place', 'X')
plot_best(df_da, 'Deep Attractor', 'o')

# plt.tight_layout()
plt.savefig('../figs/optimisation_plots.pdf', bbox_inches='tight')
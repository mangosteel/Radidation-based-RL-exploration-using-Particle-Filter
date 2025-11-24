import math
import random
import json #for writing env parameters


import numpy as np
import matplotlib.pyplot as plt

import os
import torch
from torch.autograd import Variable

def prRed(prt): print("\033[91m {}\033[00m" .format(prt))
def prGreen(prt): print("\033[92m {}\033[00m" .format(prt))
def prYellow(prt): print("\033[93m {}\033[00m" .format(prt))
def prLightPurple(prt): print("\033[94m {}\033[00m" .format(prt))
def prPurple(prt): print("\033[95m {}\033[00m" .format(prt))
def prCyan(prt): print("\033[96m {}\033[00m" .format(prt))
def prLightGray(prt): print("\033[97m {}\033[00m" .format(prt))
def prBlack(prt): print("\033[98m {}\033[00m" .format(prt))


def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.savefig("gradient_flow.png")

def get_output_folder(parent_dir, env_name):
    """Return save folder.
    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.
    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.
    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    os.makedirs(parent_dir, exist_ok=True)
    return parent_dir

def export_parameter_to_text(net_param, env):
    with open('{}/env_n_net_parameters'.format(net_param.model_path)+"_{}.txt".format(net_param.mode), 'w') as f:
        f.write('Environment Paramters\n{\n')
        f.write('  "seed": '+str(env.seed_num)+',\n' )
        f.write('  "gas_d": '+str(env.gas.d)+' m^2/s (diffusivity),\n' )
        f.write('  "gas_t": '+str(env.gas.t)+' s (gas life time),\n' )
        f.write('  "gas_q": '+str(env.gas.q)+' mg/s (gas strength),\n' )
        f.write('  "wind_mean_phi": '+str(env.t_wind.mean_phi)+' degree,\n' )
        f.write('  "wind_mean_speed": '+str(env.t_wind.mean_speed)+' m/s,\n')
        f.write('  "court_lx": '+str(env.court_lx)+' m,\n')
        f.write('  "court_lx": '+str(env.court_ly)+' m,\n')
        f.write('}\n')

        f.write('Agent Paramters\n{\n')
        f.write('  "max_steps": '+str(env.max_step)+',\n')
        f.write('  "agent_v": '+str(env.agent_v)+' m/s,\n' )
        f.write('  "delta_t": '+str(env.delta_t)+' s,\n' )
        f.write('  "pf_num": '+str(env.pf_num)+',\n' )
        f.write('  "conc_max": '+str(env.dose_max)+' mg/m^3,\n' )
        f.write('  "env_sig": '+str(env.env_sig)+' mg/m^3,\n')
        f.write('  "sensor_sig_m":  '+str(env.sensor_sig_m)+',\n')
        #if env.eps in locals():
        f.write('  "eps:"  '+str(env.eps)+' m (success criteria distance), \n')
        #if env.conv_eps in locals():
        f.write('  "conv_eps:"  '+str(env.conv_eps)+' m (success criteria particle converge STD), \n')
        f.write('}\n')

        f.write('Learning Parameters\n')
        json.dump(net_param.__dict__, f, indent=2)

import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle
import pandas as pd
import torch
import os
import numpy as np
from scipy.spatial.distance import pdist, squareform
from mpl_toolkits.mplot3d import Axes3D  
from scipy.linalg import eigh

import sys

sys.path.append(r"dunl-compneuro\src")
sys.path.append("")

import model

def init_params():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--res-path-sphinx",
        type = str,
        help = "path to solve data",
        default=[f"sparseness/results/supervised_roi{i}_Sphinx" for i in range(20)]
    )      
    parser.add_argument(
        "--res-path-hw1",
        type = str,
        help = "path to solve data",
        default=[f"sparseness/results/supervised_roi{i}_HW1" for i in range(12)] 
    )     
    parser.add_argument(
        "--res-path-hw4",
        type = str,
        help = "path to solve data",
        default=[f"sparseness/results/supervised_roi{i}_HW4" for i in range(53)]
    )              
    parser.add_argument(
        "--path",
        type=str,
        help="path",
        default="sparseness/Data/animals_data_processed.pkl",
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        help="sampling rate",
        default=10,
    )
    parser.add_argument(
        "--figsize",
        type=tuple,
        help="figsize",
        default=(8, 2),
    )
    parser.add_argument(
        "--event-filter",
        type=str,
        help="can take value inh or exh",
        default="inh",
    )

    args = parser.parse_args()
    params = vars(args)

    return params


def process_array(arr, kernel):
    """if (np.signbit(arr[0])):
        print(-arr)
        return -arr"""
    if (kernel[5]<0):
        return -arr
    else:
        return arr



def diffusion_map_embedding(X, df):
    D = squareform(pdist(X, metric='euclidean'))
    epsilon = np.median(D)**2
    W = np.exp(-D**2 / epsilon)

    P = W / W.sum(axis=1, keepdims=True)

    evals, evecs = eigh(P)  

    dm_components = evecs[:, -2::-1] 

    df['dm1'], df['dm2'], df['dm3'] = dm_components[:,0], dm_components[:,1], dm_components[:,2]    


def plot_phase_cond_traj_2d(df):
    """groups = {
        'Inhalation': df[(df.event_resp == 'inh')],
        'Exhalation':   df[df.event_resp == 'exh']
    }"""
    
    groups= {
        'Paired Pulses': df[df.delta_t1 < 20],
        'Isolated Pulses': df[df.delta_t1 >= 40]
    }

    fig, ax = plt.subplots(figsize=(6,6))

    colors = {'Paired Pulses': 'C0', 'Isolated Pulses': 'C1'}
    #colors = {'Inhalation':'C0', 'Exhalation':'C2'}

    for label, subset in groups.items():
        ax.scatter(subset.dm1, subset.dm2,
                color=colors[label], s=40, alpha=0.7, label=label)

    ax.set_xlabel('Diffusion Map 1')
    ax.set_ylabel('Diffusion Map 2')
    ax.set_title('2D Diffusion Map Scatter by Sniff Phase')
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.show()


def plot_phase_cond_traj_3d(df):

    groups = {
        'Early Inhale': df[(df.event_resp == 'inh')], # & (df.median_phase < np.pi/2)],
        #'Late Inhale':  df[(df.event_resp == 'inh') & (df.median_phase >= np.pi/2)],
        'Exhalation':   df[df.event_resp == 'exh']
    }
    """
    groups= {
        'Paired Pulses': df[df.delta_t1 < 20],
        'Isolated Pulses': df[df.delta_t1 >= 40]
    }"""

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    colors = {'Early Inhale': 'C0', 'Exhalation': 'C2'}
    #colors = {'Early Inhale': 'C0', 'Late Inhale': 'C1', 'Exhalation': 'C2'}
    #colors = {'Paired Pulses': 'C0', 'Isolated Pulses': 'C1'}

    for label, subset in groups.items():
        ax.scatter(subset.dm1, subset.dm2, subset.dm3,
                color=colors[label], s=30, alpha=0.7, label=label)

    ax.set_xlabel('Diffusion Map 1')
    ax.set_ylabel('Diffusion Map 2')
    ax.set_zlabel('Diffusion Map 3')
    ax.legend()
    plt.title('3D Diffusion Map Scatter by Sniff Phase')
    plt.tight_layout()
    plt.show()


def main():
    print("Predict.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device is", device)

    # init parameters -------------------------------------------------------#
    print("init parameters.")
    params_init = init_params()

    # load whiffs-----------------------------------------------------------#

    with open(params_init["path"], "rb") as f:
        data = pickle.load(f)
        
    animals = ['Sphinx', 'HW1'] #, 'HW4']

    all_results = []
    calcium_rois_by_animal={}

    for animal in animals:
        valve = data["valve_dict"][animal] / 100
        phase = data["phase_peaks_dict"][animal]
        calcium_signal = data["calcium_dict"][animal].mean(axis=1)   
        calcium_signal_rois = data["calcium_dict"][animal].values.T.tolist()    
        calcium_rois_by_animal[animal] = calcium_signal_rois   

        valve_ts = np.arange(0, len(valve) / 1000, 0.001)  # 1 kHz sampling
        ca_ts = np.arange(0, len(calcium_signal) / 10, 0.1)       # 10 Hz sampling
        
        whiff_onsets = np.where(np.diff(valve) > 0)[0]

        # express valve in ca ref frame
        onset_resp = []
        event_resp = []
        median_phase= []
        for i in range(len(whiff_onsets)):
            start_idx = whiff_onsets[i]

            inh_points = np.sum((0 <= phase[start_idx+1:start_idx+51]) & (phase[start_idx+1:start_idx+51] < np.pi))

            if inh_points <= 30:
                current_event= "exh"
            else:
                current_event="inh"

            # index of the value  of resp_ts that is the closest to stim_ts[start_idx]
            index = np.absolute(ca_ts-valve_ts[start_idx]).argmin()
            
            inst_phase = np.median(phase[start_idx+1:start_idx+51])
            
            median_phase.append(inst_phase)
            onset_resp.append(index)
            event_resp.append(current_event)

        valve_down = np.zeros(len(calcium_signal))
        valve_down[onset_resp] = 1 
        
        for i, onset in enumerate(onset_resp):
                if event_resp[i] == "inh":
                    if i == 0:
                        delta_t1 = None
                        delta_t2 = None  
                    elif i == 1:
                        delta_t1 = onset_resp[i] - onset_resp[i - 1] if event_resp[i - 1] == "inh" else None
                        delta_t2 = None
                    else:
                        delta_t1 = onset_resp[i] - onset_resp[i - 1] if event_resp[i - 1] == "inh" else None
                        delta_t2 = onset_resp[i - 1] - onset_resp[i - 2] if event_resp[i - 1] == "inh" and event_resp[i - 2] == "inh" else None
                else:
                    delta_t1 = None
                    delta_t2 = None
                    
                all_results.append({
                    "onset_resp": onset,
                    "event_resp": event_resp[i],
                    "median_phase": median_phase[i],
                    "delta_t1": delta_t1,
                    "delta_t2": delta_t2,
                    "animal": animal, 
                })

    df = pd.DataFrame(all_results)
    df = df.iloc[:-2]
    
    # load codes ------------------------------------------------------#

    animal_paths = {
        #'Sphinx': params_init["res_path_sphinx"],
        'HW1': params_init["res_path_hw1"],
        #'HW4': params_init["res_path_hw4"],
    }
    
    
    for animal, res_paths in animal_paths.items():
        df_animal = df[df["animal"] == animal].copy()
        
        for idx, res_path in enumerate(res_paths):
            model_path = os.path.join(res_path, "model", "model_final.pt")
            postprocess_path = os.path.join(res_path, "postprocess")

            net = torch.load(model_path, map_location=device, weights_only=False)
            net.to(device)
            net.eval()

            kernel = np.squeeze(net.get_param("H").clone().detach().cpu().numpy())
            xhat = torch.load(os.path.join(postprocess_path, "xhat.pt"))
            codehat = xhat[0, 0, 0, :].clone().detach().cpu().numpy()
            codehat = process_array(codehat, kernel)

            # Store selected codes for this animal only
            code_selected = []
            for onset in df_animal["onset_resp"].to_numpy():
                start_idx = int(onset)
                end_idx = start_idx + 2

                if start_idx < len(codehat):
                    window_vals = codehat[start_idx:min(end_idx, len(codehat))]
                    selected = 0
                    for val in window_vals:
                        if val != 0:
                            selected = val
                            break
                    code_selected.append(selected)
                else:
                    code_selected.append(0)

            # Assign the code values back into the main df, for this animal only
            df.loc[df["animal"] == animal, f"codes_{animal}_{idx}"] = code_selected
    
    columns = ['onset_resp', 'delta_t2', 'animal']
    df = df[df['animal']=='HW1'].drop(columns=columns)
    df = df.reset_index(drop=True) 
    df['pulse_idx'] = df.index   
    
    code_cols = [c for c in df.columns if c.startswith('codes_')]
    X = df[code_cols].values  # shape (N_pulses, 12)
        
    diffusion_map_embedding(X, df)
    
    plot_phase_cond_traj_2d(df)
    
    plot_phase_cond_traj_3d(df)

    
    
if __name__ == "__main__":
    main()

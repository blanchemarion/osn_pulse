import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import argparse
from scipy.signal import find_peaks
import torch
import os

import sys

sys.path.append("dunl-compneuro/src")
sys.path.append("")

import model

def init_params():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--res-path",
        type=str,
        help="res path",
        default= "sparseness/results/olfactorycalciumkernellength20num1_2025_02_20_11_30_36"
    )
    parser.add_argument(
        "--path",
        type=str,
        help="data path",
        default="sparseness/Data/animals_data_processed.pkl",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        help="size of the window to look backward",
        default="1500",
    )
    parser.add_argument(
        "--event-filter",
        type=str,
        help="can take values inh or exh",
        default="inh"
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        help="number of bins",
        default=4
    )
    args = parser.parse_args()
    params = vars(args)

    return params


def align_pulse_calcium(valve, calcium, phase, params):

    valve_ts = np.arange(0, len(valve) / 1000, 0.001)  # 1 kHz sampling
    ca_ts = np.arange(0, len(calcium) / 10, 0.1)       # 10 Hz sampling
    
    whiff_onsets = np.where(np.diff(valve) > 0)[0]

    # express valve in ca ref frame
    onset_resp = []
    event_resp = []
    preceding_pulses = []
    for i in range(len(whiff_onsets)):
        start_idx = whiff_onsets[i]

        inh_points = np.sum((0 <= phase[start_idx+1:start_idx+51]) & (phase[start_idx+1:start_idx+51] < np.pi))

        if inh_points <= 25:
            current_event= "exh"
        else:
            current_event="inh"

        if params["event_filter"] != current_event:
            number_pulses = 0
            continue

        idx = max(0, whiff_onsets[i] - params["window_size"])
        preceding_peaks = whiff_onsets[(whiff_onsets >= idx) & (whiff_onsets < whiff_onsets[i])]

        filtered_peaks = []
        for p_idx in preceding_peaks:
            inh_points = np.sum((0 <= phase[p_idx+1:p_idx+51]) & (phase[p_idx+1:p_idx+51] < np.pi))

            if inh_points <= 25:
                preceding_event= "exh"
            else:
                preceding_event="inh"  

            if preceding_event == current_event:
                filtered_peaks.append(p_idx)

        number_pulses = len(filtered_peaks)
        
        # index of the value  of resp_ts that is the closest to stim_ts[start_idx]
        index = np.absolute(ca_ts-valve_ts[start_idx]).argmin()

        onset_resp.append(index)
        event_resp.append(current_event)
        preceding_pulses.append(number_pulses)

    valve_down = np.zeros(len(calcium))
    valve_down[onset_resp] = 1 

    return valve_down, onset_resp, event_resp, preceding_pulses


def plot_codes_vs_preceding_pulses(df):
    plt.figure(figsize=(8, 6))
    animals = df["animal"].unique()
    
    for animal in animals:
        # Filter the dataframe for the current animal
        df_animal = df[df["animal"] == animal]
        
        # Group by the discrete 'preceding_pulses' values and compute statistics on 'codes'
        grouped = df_animal.groupby("preceding_pulses")["codes"]
        means = grouped.mean()
        stds = grouped.std()
        counts = grouped.count()
        std_errors = stds / np.sqrt(counts)
        
        # x-values are the unique number of preceding pulses
        x_vals = means.index.values
        
        # Plot with error bars for this animal
        plt.errorbar(x_vals, means, yerr=std_errors, fmt='o-', capsize=5, label=f"{animal}")
    
    plt.xlabel("Number of preceding pulses")
    plt.xticks(x_vals)
    plt.ylabel("Code")
    plt.title("Code as a function of preceding pulses (by Animal)")
    plt.legend(title="Animal")
    plt.grid(True)
    plt.show()



def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device is", device)

    # init parameters -------------------------------------------------------#
    print("init parameters.")
    params_init = init_params()

    # take parameters from the result path
    params = pickle.load(
        open(os.path.join(params_init["res_path"], "params.pickle"), "rb")
    )
    for key in params_init.keys():
        params[key] = params_init[key]

    # create folders -------------------------------------------------------#
    model_path = os.path.join(
        params["res_path"],
        "model",
        "model_final.pt",
    )

    out_path = os.path.join(
        params["res_path"],
        "figures",
    )
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    postprocess_path = os.path.join(
        params["res_path"],
        "postprocess",
    )
    
    if params["data_path"] == "":
        data_folder = params["data_folder"]
        filename_list = os.listdir(data_folder)
        data_path_list  = [
            f"{data_folder}/{x}" for x in filename_list if "trainready.pt" in x
        ]
    else:
        data_path_list = params["data_path"]

    # load valve -------------------------------------------------------------------#

    with open(params["path"], "rb") as f:
        data = pickle.load(f)
        
        
    animals = ['HW1', 'HW4', 'Sphinx']

    all_results = []
    offset = 0

    for animal in animals:
        valve = data["valve_dict"][animal] / 100
        phase = data["phase_peaks_dict"][animal]
        calcium_signal = data["calcium_dict"][animal].mean(axis=1)

        valve_down, onset_resp, event_resp, preceding_pulses = align_pulse_calcium(valve, calcium_signal, phase, params)
        
        # Adjust onsets by offset
        onset_resp = [onset + offset for onset in onset_resp
                      if onset < len(calcium_signal) - params["kernel_length"]
        ]
        
        for i, onset in enumerate(onset_resp):
            all_results.append({
                "onset_resp": onset,
                "event_resp": event_resp[i],
                "preceding_pulses": preceding_pulses[i],
                "animal": animal
            })

        # Update offset for the next animal
        offset += len(calcium_signal)

    df = pd.DataFrame(all_results)
        
    # load codes ------------------------------------------------------#
    
    net = torch.load(model_path, map_location=device, weights_only=False)
    net.to(device)
    net.eval()

    for data_path in data_path_list:
        datafile_name = data_path.split("/")[-1].split(".pt")[0]

        xhat = torch.load(
            os.path.join(postprocess_path, "xhat_{}.pt".format(datafile_name))
        )

        y = torch.load(
            os.path.join(postprocess_path, "y_{}.pt".format(datafile_name))
        )    
        
    codes = xhat[0, 0, :].cpu().numpy()[0][df["onset_resp"].values]
        
    df["codes"] = codes

    plot_codes_vs_preceding_pulses(df)

if __name__ == "__main__":
    main()
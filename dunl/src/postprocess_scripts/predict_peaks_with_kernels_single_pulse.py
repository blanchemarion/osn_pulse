import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt

import sys

sys.path.append("dunl-compneuro\src")
sys.path.append("")

import model
from functions import load_dataset

def init_params():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--res-path",
        type=str,
        help="res path",
        default="Results\olfactorycalciumkernellength20num2_2025_01_06_12_43_29",
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
        "--index-beg",
        type=int,
        help="index beg",
        default=1000,
    )
    parser.add_argument(
        "--index-end",
        type=int,
        help="index end",
        default=2000,
    )

    args = parser.parse_args()
    params = vars(args)

    return params

def predict_response(kernels):
    _, _, valve_dict, _, calcium_dict, _, _, phase_peaks_dict, _, _, _, _ = load_dataset("Processed/all_animals_data.pkl")

    valve_data = valve_dict['HW1']
    calcium_data = calcium_dict['HW1']
    phase_peaks_data = phase_peaks_dict['HW1']

    valve_ts = np.arange(0, len(valve_data) / 1000, 0.001)  # 1 kHz sampling
    ca_ts = np.arange(0, len(calcium_data) / 10, 0.1)       # 10 Hz sampling
    
    max_ca_duration = ca_ts[-1] # duration of calcium data
    
    max_valve_idx = np.searchsorted(valve_ts, max_ca_duration) # index in valve_ts that corresponds to max ca duration

    valve_data = valve_data[:max_valve_idx]/100
    phase_peaks_data = phase_peaks_data[:max_valve_idx]
    calcium_data = calcium_data.mean(axis=1).to_numpy()

    new_valve_ts = np.arange(0, len(valve_data) / 1000, 0.001)
    # Detect events indices
    whiff_onsets = np.where(np.diff(valve_data) > 0)[0]

    # Find phase and ca values around events
    phases = []
    stim_indices = []
    for i in range(len(whiff_onsets)):
        start_idx = whiff_onsets[i]
        phase_value = np.mean(phase_peaks_data[start_idx:start_idx + 75])  # Mean phase in the 75 ms following pulse onset (75 samples at 1kHz)
        phases.append(phase_value)

        # index of the value  of resp_ts that is the closest to stim_ts[start_idx]
        index = np.absolute(ca_ts-new_valve_ts[start_idx]).argmin()
        stim_indices.append(index)

    # Initialize the reconstructed signal
    reconstructed_signal = np.zeros_like(calcium_data)

    # Loop through all stimulus events
    for stim_idx, stim_phase in zip(stim_indices, phases):
        if 0 <= stim_phase < np.pi:
            kernel = kernels[0]  # Select kernel 0
        elif np.pi <= stim_phase < 2 * np.pi:
            kernel = kernels[1]  # Select kernel 1
        else:
            raise ValueError(f"Stimulus phase out of bounds: {stim_phase}")
        
        start_idx = max(0, stim_idx)
        end_idx = min(len(calcium_data), stim_idx + len(kernel))
        
        # Add the kernel contribution to the reconstructed signal
        reconstructed_signal[start_idx:end_idx] += kernel[:end_idx - start_idx]
    
    return calcium_data, reconstructed_signal, stim_indices


def plot_predicted_response(kernels, index_beg, index_end, params, out_path):
    axes_fontsize = 15
    legend_fontsize = 12
    tick_fontsize = 12
    title_fontsize = 18
    fontfamily = "sans-serif"

    mpl.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "text.usetex": True,
            "axes.labelsize": axes_fontsize,
            "axes.titlesize": title_fontsize,
            "legend.fontsize": legend_fontsize,
            "xtick.labelsize": tick_fontsize,
            "ytick.labelsize": tick_fontsize,
            "text.latex.preamble": r"\usepackage{bm}",
            "axes.unicode_minus": False,
            "font.family": fontfamily,
        }
    )

    original_signal, reconstructed_signal, stim_indices = predict_response(kernels)

    original_signal_zero_mean = original_signal - np.mean(original_signal)
    reconstructed_signal_zero_mean = reconstructed_signal - np.mean(reconstructed_signal)

    ca_response_norm = (original_signal_zero_mean - np.min(original_signal_zero_mean)) / (np.max(original_signal_zero_mean) - np.min(original_signal_zero_mean))
    reconstructed_signal_norm = 0.1+(reconstructed_signal_zero_mean - np.min(reconstructed_signal_zero_mean)) / (np.max(reconstructed_signal_zero_mean) - np.min(reconstructed_signal_zero_mean))

    plt.figure(figsize=(15, 5))
    time = np.arange(len(ca_response_norm))

    plt.plot(
        time[index_beg:index_end],
        ca_response_norm[index_beg:index_end],
        label=r"\textbf{Actual Calcium Response (Normalized)}",
        color="blue",
        alpha=0.7,
    )
    plt.plot(
        time[index_beg:index_end],
        reconstructed_signal_norm[index_beg:index_end],
        label=r"\textbf{Reconstrcuted Response (Normalized)}",
        color="red",
        linestyle="--",
        alpha=0.7,
    )
    """for stim_idx in stim_indices:
        if index_beg <= stim_idx < index_end:  # Only plot lines within the range
            plt.axvline(x=stim_idx, color="green", linestyle=":", alpha=0.8)"""

    plt.title(
        r"\textbf{Normalized Actual vs. Predicted Calcium Response}",
        fontsize=title_fontsize,
    )
    plt.xlabel(r"\textbf{Time (ms)}", fontsize=axes_fontsize)
    plt.ylabel(r"\textbf{Normalized Signal}", fontsize=axes_fontsize)
    plt.legend(loc="upper right", frameon=True, fontsize=legend_fontsize)

    # Save the figure
    output_path = os.path.join(out_path, "pred_vs_actual.svg")
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
    plt.close()

    print(f"Plotting completed. Plot saved at {output_path}")



def main():
    print("Predict.")

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

    # load model ------------------------------------------------------#
    net = torch.load(model_path, map_location=device)
    net.to(device)
    net.eval()

    kernels = np.squeeze(net.get_param("H").clone().detach().cpu().numpy())

    plot_predicted_response(kernels, params["index_beg"], params["index_end"], params, out_path)



if __name__ == "__main__":
    main()


import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import math

import sys

sys.path.append("dunl-compneuro\src")
sys.path.append("")

import model

def init_params():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--res-path",
        type=str,
        help="res path",
        default="Results\olfactorycalciumkernellength20num4_2025_01_03_17_11_10",
    )
    parser.add_argument(
        "--ca-path",
        type=str,
        help="ca path",
        default="Data\processed_ca_resp_to_paired_pulses_data.npy",
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

    args = parser.parse_args()
    params = vars(args)

    return params


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

    postprocess_path = os.path.join(
        params["res_path"],
        "postprocess",
    )
    
    if params["data_path"] == "":
        data_folder = params["data_folder"]
        filename_list = os.listdir(data_folder)
        data_path_list = [
            f"{data_folder}/{x}" for x in filename_list if "trainready.pt" in x
        ]
    else:
        data_path_list = params["data_path"]

    print("There {} dataset in the folder.".format(len(data_path_list)))

    # load kernels ------------------------------------------------------#
    net = torch.load(model_path, map_location=device)
    net.to(device)
    net.eval()

    kernels = np.squeeze(net.get_param("H").clone().detach().cpu().numpy())

    # load codes ---------------------------------------------------------#
    for data_path in data_path_list:
        datafile_name = data_path.split("/")[-1].split(".pt")[0]

        x = torch.load(os.path.join(postprocess_path, "x_{}.pt".format(datafile_name)))
        xhat = torch.load(
            os.path.join(postprocess_path, "xhat_{}.pt".format(datafile_name))
        )
        label = torch.load(
            os.path.join(postprocess_path, "label_{}.pt".format(datafile_name))
        )

        num_trials = xhat.shape[0]

        code = list()

        for i in range(num_trials):
            xihat = xhat[i]
            code.append(torch.sum(xihat, dim=-1))

        code = torch.stack(code, dim=0)
        print(code)

    # load
    ca_path = params["ca_path"]
    data = torch.load(ca_path)

    reconstructed_data, trials_type= reconstruct_calcium(data, kernels)
    #plot_trial_with_events(3, data, reconstructed_data, code)
    plot_multiple_trials_with_events(list(range(65,80)), data, reconstructed_data, trials_type, code)



def reconstruct_calcium(data, kernels):
    reconstructed_data = {}
    trials_type = []
    for trial_name, trial_info in data.items():
        if not trial_name.startswith('trial'):
            continue 
        
        trial_type = trial_info['type']
        event_onsets = []
        
        if trial_type == 0:
            event_onsets += trial_info['event0_onsets'] + trial_info['event2_onsets']
            kernel = kernels[0]
        elif trial_type == 1:
            event_onsets += trial_info['event0_onsets'] + trial_info['event3_onsets']
            kernel = kernels[1]
        elif trial_type == 2:
            event_onsets += trial_info['event1_onsets'] + trial_info['event2_onsets']
            kernel = kernels[2]
        elif trial_type == 3:
            event_onsets += trial_info['event1_onsets'] + trial_info['event3_onsets']
            kernel = kernels[3]
        
        reconstructed_signal = np.zeros(len(data['y'][int(trial_name[5:])].flatten()))
        
        # Apply kernels at the specified event onsets
        for onset in event_onsets:
            onset = int(onset) 
            reconstructed_signal[onset:onset + 20] += kernel
        
        reconstructed_data[trial_name] = reconstructed_signal
        trials_type.append(trial_type)
    
    return reconstructed_data, trials_type


def plot_trial_with_events(trial_num, data, reconstructed_signal, code):
    
    trial_name = f'trial{trial_num}'

    trial_data = data[trial_name]
    events = [f'event{i}_onsets' for i in range(4)]  # List of event keys
    kernel_codes = code[:, 0, :]  # Extract codes for all trials

    # Get the trial index from the trial name
    trial_idx = int(trial_name[5:])  # Extract the number after 'trial'

    original_signal = data['y'][trial_num].flatten()  # Flatten the original signal for plotting
    reconstructed_signal = reconstructed_signal[trial_name]

    original_signal_zero_mean = original_signal - np.mean(original_signal)
    reconstructed_signal_zero_mean = reconstructed_signal - np.mean(reconstructed_signal)
    original_signal_norm = (original_signal_zero_mean - np.min(original_signal_zero_mean)) / (np.max(original_signal_zero_mean) - np.min(original_signal_zero_mean))
    reconstructed_signal_norm = 0.1+(reconstructed_signal_zero_mean - np.min(reconstructed_signal_zero_mean)) / (np.max(reconstructed_signal_zero_mean) - np.min(reconstructed_signal_zero_mean))

    # Plot the original and reconstructed signals
    plt.figure(figsize=(10, 6))
    plt.plot(original_signal_norm, label='original signal (norm)', alpha=0.7)
    plt.plot(reconstructed_signal_norm, label='reconstructed signal (norm)', alpha=0.7)
    
    # Overlay event points with corresponding codes
    for event_idx, event_key in enumerate(events):
        event_onsets = trial_data[event_key]
        code_value = kernel_codes[trial_idx, event_idx]  # Get code for this event type
        
        for onset in event_onsets:
            plt.scatter(onset, code_value, color='red', label=f'Event {event_idx} (Code: {code_value})')
            plt.text(onset, code_value, f'{code_value:.4f}', fontsize=8, ha='right')

    plt.xlabel('Time')
    plt.ylabel('Signal amplitude / Event code')
    plt.title(f'Original vs Reconstructed Signal with Events for {trial_name}')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_multiple_trials_with_events(trial_nums, data, reconstructed_data, trials_type, code):

    num_trials = len(trial_nums)
    num_cols = min(5, num_trials) 
    num_rows = math.ceil(num_trials / num_cols)
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 3 * num_rows), sharex=True, sharey=True)
    axes = axes.flatten() if num_trials > 1 else [axes]
    
    for i, trial_num in enumerate(trial_nums):
        trial_name = f'trial{trial_num}'
        trial_data = data[trial_name]
        events = [f'event{i}_onsets' for i in range(4)]  # List of event keys
        kernel_codes = code[:, 0, :]  # Extract codes for all trials

        trial_idx = int(trial_name[5:])
        original_signal = data['y'][trial_num].flatten()
        reconstructed_signal = reconstructed_data[trial_name]

        original_signal_zero_mean = original_signal - np.mean(original_signal)
        reconstructed_signal_zero_mean = reconstructed_signal - np.mean(reconstructed_signal)
        original_signal_norm = (original_signal_zero_mean - np.min(original_signal_zero_mean)) / (np.max(original_signal_zero_mean) - np.min(original_signal_zero_mean))
        reconstructed_signal_norm = 0.1 + (reconstructed_signal_zero_mean - np.min(reconstructed_signal_zero_mean)) / (np.max(reconstructed_signal_zero_mean) - np.min(reconstructed_signal_zero_mean))

        # Plot on the corresponding subplot
        ax = axes[i] if num_trials > 1 else axes
        ax.plot(original_signal_norm, label='original signal (norm)', alpha=0.7)
        ax.plot(reconstructed_signal_norm, label='reconstructed signal (norm)', alpha=0.7)
        
        # Overlay event points with corresponding codes
        for event_idx, event_key in enumerate(events):
            event_onsets = trial_data[event_key]
            code_value = kernel_codes[trial_idx, event_idx]
            
            for onset in event_onsets:
                ax.scatter(onset, code_value, color='red')
                ax.text(onset, code_value, f'{code_value:.4f}', fontsize=6, ha='right')

        ax.set_ylabel('Signal amplitude / Event code', fontsize=6)
        ax.set_title(f'Trial {trial_num} and Type {trials_type[i]} ', fontsize=8)
        ax.grid(True)
        if i == num_trials - 1:
            ax.set_xlabel('Time', fontsize=6)
        ax.legend(fontsize=6)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()


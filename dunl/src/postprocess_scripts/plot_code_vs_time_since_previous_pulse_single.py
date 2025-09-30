import torch
import os
import pickle
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import sys

sys.path.append("dunl-compneuro\src")

import model, utils


def init_params():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--res-path",
        type=str,
        help="res path",
        default="Results\olfactorycalciumkernellength20num2_2025_01_06_12_43_29",
    )

    parser.add_argument(
        "--ca-path",
        type=str,
        help="ca path",
        default="Data\processed_ca_resp_to_pulse_data.npy",
    )
    parser.add_argument(
        "--path",
        type=str,
        help="path",
        default="Data/ca_resp_to_pulse_data.pkl",
    )
    parser.add_argument(
        "--figsize",
        type=tuple,
        help="figsize",
        default=(10, 8),
    )

    args = parser.parse_args()
    params = vars(args)

    return params

def main():
    print("WARNING! This script assumes that each code is 1-sparse.")

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

    if params["data_path"] == "":
        data_folder = params["data_folder"]
        filename_list = os.listdir(data_folder)
        data_path_list = [
            f"{data_folder}/{x}" for x in filename_list if "trainready.pt" in x
        ]
    else:
        data_path_list = params["data_path"]

    print("There {} dataset in the folder.".format(len(data_path_list)))

    # create folders -------------------------------------------------------#

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

    print(postprocess_path)

    # load codes -------------------------------------------------------#

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

    code =  code[code != 0].view(-1).numpy() #join all codes

    # load trials ------------------------------------------------------#

    ca_path = params["ca_path"]
    data = torch.load(ca_path)
    phase = np.array(data["phase"])

    # load whiffs ------------------------------------------------------#

    print(params_init["path"])
    with open(params_init["path"],"rb") as f:
        final_dict= pickle.load(f)

    all_whiffs=[]
    for animal in final_dict.keys():
        whiff_onsets = final_dict[animal]['whiff_onset']
        all_whiffs.append(whiff_onsets)
    whiffs_indices = np.concatenate(all_whiffs)

    df = pd.DataFrame({
        'code': code,
        'phase': phase,
        'whiffs_indices': whiffs_indices
    })
    print(df.head())
    filtered_df = df[(df['phase'] >= 0) & (df['phase'] < np.pi)]

    delta_whiffs_list = []
    for i in filtered_df.index:
        if i >0: 
            current_whiff = whiffs_indices[i] 
            previous_whiff = whiffs_indices[i-1]
            delta_whiffs = (current_whiff-previous_whiff)/10 # time in s between the 2 whiffs
        else:
            delta_whiffs=np.nan # for the 1st whiff
        delta_whiffs_list.append(delta_whiffs)
    
    filtered_df["delta_whiffs"] = delta_whiffs_list
    filtered_df=filtered_df[(filtered_df['delta_whiffs'] >= 0) & (filtered_df['delta_whiffs'] <= 10)]
    plot_hist_per_time_bin(filtered_df, params, out_path)
    plot_code_vs_time_boxplot(filtered_df["delta_whiffs"], filtered_df["code"], params, out_path)



def plot_hist_per_time_bin(filtered_df, params, out_path, num_bins = 4):

    bins = np.linspace(0, filtered_df["delta_whiffs"].max(), num_bins)
    bin_labels = [f"{bins[i]:.1f} - {bins[i + 1]:.1f}" for i in range(len(bins) - 1)]

    fig, axes = plt.subplots(len(bin_labels), 1, figsize=params["figsize"], sharex=True)
    fig.suptitle("Frequency of Codes Across Time Interval Bins", fontsize=16)

    for i, ax in enumerate(axes):
        mask = (filtered_df['delta_whiffs'] >= bins[i]) & (filtered_df['delta_whiffs'] < bins[i + 1])
        codes_in_bin = filtered_df.loc[mask, 'code']
        
        ax.hist(codes_in_bin, bins=30, color='black', alpha=0.7)
        ax.set_title(f"Time Interval: {bin_labels[i]} s")
        ax.set_ylabel("Frequency")
        ax.grid(True)

    axes[-1].set_xlabel("Code")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(out_path, "hist_codes_vs_time.svg"), bbox_inches="tight", pad_inches=0.02)
    plt.close()
    print(f"Plotting completed. Plot saved at {out_path}")




def plot_code_vs_time_boxplot(time, code, params, out_path, num_bins=4):
    bins = np.linspace(0, time.max(), num_bins)
    time_bin = pd.cut(time, bins)

    plt.figure(figsize=params["figsize"])
    sns.boxplot(x=time_bin, y=code, color="grey")
    plt.xticks(rotation=45)
    plt.title("Code Vs Time since Previous Pulse")
    plt.xlabel("Time Interval")
    plt.ylabel("Code")
    plt.grid(True)
    plt.savefig(os.path.join(out_path, "codes_vs_time_boxplot.svg"), bbox_inches="tight", pad_inches=0.02)
    plt.close()
    print(f"Plotting completed. Plot saved at {out_path}")


if __name__ == "__main__":
    main()
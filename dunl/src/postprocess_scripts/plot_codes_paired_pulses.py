import torch
import os
import pickle
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import sys

sys.path.append("dunl-compneuro\src")

import utils


def init_params():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--res-path",
        type=str,
        help="res path",
        default= "Results\olfactorycalciumkernellength20num4.2_2025_01_13_13_33_57"
        # "Results\olfactorycalciumkernellength20num4.2_2025_01_07_15_45_50",
        # "Results\olfactorycalciumkernellength20num1_2025_01_07_12_21_00",
    )

    parser.add_argument(
        "--color-list",
        type=list,
        help="color list",
        default=[
            "black",
            "orange",
            "blue",
            "red",
        ], 
    )
    parser.add_argument(
        "--figsize",
        type=tuple,
        help="figsize",
        default=(12, 3),
    )

    args = parser.parse_args()
    params = vars(args)

    return params


def main():
    print("Predict.")
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
        "codes_vs_reward",
    )
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    postprocess_path = os.path.join(
        params["res_path"],
        "postprocess",
    )

    print(postprocess_path)

    # load data -------------------------------------------------------#

    for data_path in data_path_list:
        datafile_name = data_path.split("/")[-1].split(".pt")[0]

        x = torch.load(os.path.join(postprocess_path, "x_{}.pt".format(datafile_name)))
        xhat = torch.load(
            os.path.join(postprocess_path, "xhat_{}.pt".format(datafile_name))
        )
        label = torch.load(
            os.path.join(postprocess_path, "label_{}.pt".format(datafile_name))
        )

        codes_stacked = process_trials(x, xhat)
        save_histograms(codes_stacked, params, out_path, datafile_name)


def process_trials(x, xhat):
    num_trials = xhat.shape[0]
    codes = [[] for _ in range(4)]

    for i in range(num_trials):
        for event_idx in range(4): 
            event_flag = torch.sum(torch.abs(x[i][event_idx]), dim=-1).item()
            if event_flag:
                codes[event_idx].append(torch.sum(xhat[i][0][event_idx], dim=-1))
    return [torch.stack(c, dim=0) if c else torch.zeros(1) for c in codes]


def save_histograms(codes_stacked, params, out_path, datafile_name):
    import math
    
    colors = params["color_list"]
    event_names = ["Event 0", "Event 1", "Event 2", "Event 3"]

    mpl.rcParams.update({
        "pgf.texsystem": "pdflatex",
        "text.usetex": True,
        "axes.labelsize": 15,
        "axes.titlesize": 20,
        "legend.fontsize": 10,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "text.latex.preamble": r"\usepackage{bm}",
        "axes.unicode_minus": False,
        "font.family": "sans-serif",
    })

    valid_codes = [(i, code) for i, code in enumerate(codes_stacked) if code.numel() > 1]
    num_valid_events = len(valid_codes)
    
    if num_valid_events == 0:
        print("No valid codes to plot.")
        return

    num_cols = 1
    num_rows = math.ceil(num_valid_events / num_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(params["figsize"][0], params["figsize"][1] * num_rows))
    axes = axes.flatten()  

    all_values = torch.cat([code for _, code in valid_codes]).numpy()
    x_min, x_max = all_values.min(), all_values.max()

    for idx, (i, code) in enumerate(valid_codes):
        ax = axes[idx]
        ax.hist(code.numpy(), bins=30, alpha=0.7, color=colors[i], edgecolor="black")
        ax.set_title(f"Histogram of Code Features ({event_names[i]})")
        ax.set_xlabel(f"Code {i} Values")
        ax.set_ylabel("Frequency")
        ax.set_xlim(x_min, x_max) 
        ax.grid(True)

    for j in range(idx + 1, len(axes)):
        axes[j].axis('off')

    fig.tight_layout()
    out_path_name = os.path.join(out_path, f"histograms_{datafile_name}.svg")
    plt.savefig(out_path_name, bbox_inches="tight", pad_inches=0.02)
    plt.close()

    print(f"Plotting of codes is done. Plots are saved at {out_path_name}")


if __name__ == "__main__":
    main()
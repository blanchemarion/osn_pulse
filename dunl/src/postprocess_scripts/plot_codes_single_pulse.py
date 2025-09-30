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
        default="sparsenessresults\calcium_unsupervised_numwindow1_neuron0_kernellength20_1kernels_1000unroll_2025_02_05_19_07_13"
        #"Results\olfactorycalciumkernellength20num2_2025_01_16_12_52_32",
    )

    parser.add_argument(
        "--color-list",
        type=list,
        help="color list",
        default=[
            "black",
            "orange",
        ], 
    )
    parser.add_argument(
        "--figsize",
        type=tuple,
        help="figsize",
        default=(12, 6),
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

        num_trials = xhat.shape[0]
        num_neurons = xhat.shape[1]

        code_0 = list()
        code_1 = list()

        for i in range(num_trials):
            xi = x[i]
            xihat = xhat[i]

            event_flag_1 = torch.sum(torch.abs(xi[1]), dim=-1).item()
            print(xi)

            if event_flag_1:
                code_1.append(torch.sum(xihat, dim=-1))
            else:
                code_0.append(torch.sum(xihat, dim=-1))

        code_0 = torch.stack(code_0, dim=0)
        code_1 = torch.stack(code_1, dim=0)

        out_path_name_0 = os.path.join(
                out_path,
                "histogram_code_0_{}.svg".format(datafile_name),
            )
        
        out_path_name_1 = os.path.join(
                out_path,
                "histogram_code_1_{}.svg".format(datafile_name),
            )
        

        plot_code_histograms(
            code_0[:, 0], 
            code_1[:, 0], 
            params, 
            os.path.join(out_path, f"histograms_{datafile_name}.svg")
        )
        
        print(f"Plotting of codes is done. Plots are saved at {out_path}")


def plot_code_histograms(
    code_0,
    code_1,
    params,
    out_path_name
):

    code_0_np = code_0.squeeze(1).numpy()  # Shape becomes [N, F]
    code_1_np = code_1.squeeze(1).numpy()  # Shape becomes [M, F]

    mpl.rcParams.update(
        {
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
        }
    )

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=params["figsize"], sharex=True)

    # Plot histogram for Code 0
    axes[0].hist(
        code_0_np[:, 0],
        bins=30,
        alpha=0.7,
        color=params["color_list"][0],
        edgecolor="black",
    )
    axes[0].set_title("Histogram of Code Features (Event 0)")
    axes[0].set_ylabel("Frequency")
    axes[0].grid(True)

    # Plot histogram for Code 1
    axes[1].hist(
        code_1_np[:, 1],
        bins=30,
        alpha=0.7,
        color=params["color_list"][1],
        edgecolor="black",
    )
    axes[1].set_title("Histogram of Code Features (Event 1)")
    axes[1].set_xlabel("Code Values")
    axes[1].set_ylabel("Frequency")
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(out_path_name, bbox_inches="tight", pad_inches=0.02)
    plt.close()


if __name__ == "__main__":
    main()

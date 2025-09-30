import torch
import os
import pickle
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

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
        "--figsize",
        type=tuple,
        help="figsize",
        default=(12, 6),
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
            xi = x[i]
            xihat = xhat[i]

            code.append(torch.sum(xihat, dim=-1))

        code = torch.stack(code, dim=0)

    # load trials ------------------------------------------------------#
    ca_path = params["ca_path"]
    data = torch.load(ca_path)

    plot_calcium_vs_code(data, code, params, out_path)

def plot_calcium_vs_code(data, code, params, out_path):
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

    code =  code[code != 0].view(-1).numpy()
    phase = np.array(data["phase"])
    
    plt.figure(figsize=params["figsize"])
    heatmap, xedges, yedges = np.histogram2d(phase, code, bins=(25, 25)) 
    plt.imshow(
        heatmap.T, 
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
        origin='lower', 
        aspect='auto', 
        cmap='viridis'
    )
    plt.colorbar(label="Density")    
    plt.title("Heatmap: Code Density as a Function of Phase")
    plt.xlabel("Phase")
    plt.ylabel("Code")
    plt.grid(True)
    out_path_heatmap= os.path.join(out_path, "codes_vs_phase_heatmap.svg")
    plt.savefig(out_path_heatmap, bbox_inches="tight", pad_inches=0.02)
    plt.close()
    print(f"Plotting completed. Plot saved at {out_path_heatmap}")

if __name__ == "__main__":
    main()
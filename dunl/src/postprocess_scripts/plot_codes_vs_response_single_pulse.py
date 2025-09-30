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
        "codes_vs_reward",
    )
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    postprocess_path = os.path.join(
        params["res_path"],
        "postprocess",
    )

    print(postprocess_path)

    out_path = os.path.join(
        params["res_path"],
        "figures",
    )
    if not os.path.exists(out_path):
        os.makedirs(out_path)

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

    # Extract phase and calcium data
    phase = data["phase"]
    calcium_responses = data["y"]

    # Initialize lists to store code values and calcium max values for each phase range
    codes_phase1 = []
    calcium_max_phase1 = []
    codes_phase2 = []
    calcium_max_phase2 = []

    num_trials = len(phase)
    for i in range(num_trials):
        calcium_max = np.max(calcium_responses[i])  # Get max calcium response for each trial

        if 0 <= phase[i] < np.pi:  # Phase range [0, π)
            codes_phase1.append(code[i][0][0].item())
            calcium_max_phase1.append(calcium_max)
        elif np.pi <= phase[i] < 2 * np.pi:  # Phase range [π, 2π)
            codes_phase2.append(code[i][0][1].item())
            calcium_max_phase2.append(calcium_max)

    # Create scatter plot for phase range [0, π)
    plt.figure(figsize=params["figsize"])
    plt.scatter(codes_phase1, calcium_max_phase1, color='blue', alpha=0.7, label="Inhalation Phase")
    plt.title("Maximum Calcium Response vs Code Value (Inhalation)")
    plt.xlabel("Code Value")
    plt.ylabel("Maximum Calcium Response")
    plt.grid(True)
    plt.legend()
    output_path_phase1 = os.path.join(out_path, "codes_vs_response_phase1.svg")
    plt.savefig(output_path_phase1, bbox_inches="tight", pad_inches=0.02)
    plt.close()

    print(f"Plotting completed for Inhalation Phase. Plot saved at {output_path_phase1}")

    # Create scatter plot for phase range [π, 2π)
    plt.figure(figsize=params["figsize"])
    plt.scatter(codes_phase2, calcium_max_phase2, color='orange', alpha=0.7, label="Exhalation Phase")
    plt.title("Maximum Calcium Response vs Code Value (Exhalation))")
    plt.xlabel("Code Value")
    plt.ylabel("Maximum Calcium Response")
    plt.grid(True)
    plt.legend()
    output_path_phase2 = os.path.join(out_path, "codes_vs_response_phase2.svg")
    plt.savefig(output_path_phase2, bbox_inches="tight", pad_inches=0.02)
    plt.close()

    print(f"Plotting completed for Exhalation Phase. Plot saved at {output_path_phase2}")

    # Create scatter plot for both phase range
    plt.figure(figsize=params["figsize"])
    plt.scatter(codes_phase1, calcium_max_phase1, color='blue', alpha=0.7, label="Inhalation Phase")
    plt.scatter(codes_phase2, calcium_max_phase2, color='orange', alpha=0.7, label="Exhalation Phase")
    plt.title("Maximum Calcium Response vs Code Value (All breath cycle))")
    plt.xlabel("Code Value")
    plt.ylabel("Maximum Calcium Response")
    plt.grid(True)
    plt.legend()
    output_path = os.path.join(out_path, "codes_vs_response_all_phase.svg")
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
    plt.close()

    print(f"Plotting completed for All Phase. Plot saved at {output_path}")

if __name__ == "__main__":
    main()
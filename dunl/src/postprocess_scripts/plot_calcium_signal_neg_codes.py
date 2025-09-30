import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import argparse
import torch 
import os


def init_params():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--res-path",
        type=str,
        help="res path",
        default="Results\olfactorycalciumkernellength20num1_2025_01_23_11_02_14",
    )
    parser.add_argument(
        "--path",
        type=str,
        help="data path",
        default="sparseness/Data/animals_data_processed.pkl",
    )
    parser.add_argument(
        "--seg-path",
        type=str,
        help="segmented data path",
        default="sparseness/Data/ca_resp_to_pulse_data.pkl", 
    )
    parser.add_argument(
        "--baseline-path",
        type = str,
        help="calcium, baseline and phase",
        default= "sparseness\Data\processed_ca_resp_to_pulse_data.npy"
    )
    parser.add_argument(
        "--figsize",
        type=tuple,
        help="figsize",
        default=(12, 2),
    )
    args = parser.parse_args()
    params = vars(args)

    return params


def plot_calcium_with_whiffs(calcium, whiff_df, index_beg, index_end, params, out_path):

    calcium_segment = calcium[index_beg:index_end]
    time = np.arange(index_beg, index_end)

    whiffs_in_range = whiff_df[(whiff_df['whiff_onsets'] >= index_beg) & 
                               (whiff_df['whiff_onsets'] < index_end)]

    plt.figure(figsize=(12, 6))
    plt.plot(time, calcium_segment, label="Calcium Signal", color="blue")

    for _, row in whiffs_in_range.iterrows():
        whiff_onset = row['whiff_onsets']
        code = row['codes']
        phase=row['phase']
        #baseline= row['baseline']
        convolved_stim= row['convolved_stim']

        color = "green" if 0 <= phase < np.pi else "red"

        plt.axvline(x=whiff_onset, color=color, linestyle="--", alpha=0.7, label=f"{'Inhalation' if color == 'green' else 'Exhalation'}")

        plt.scatter(whiff_onset, code, color="green")
        plt.text(whiff_onset, code, f"{code}", color="green", fontsize=8, ha="left", va="bottom")

        plt.scatter(whiff_onset, convolved_stim/100, color="purple", marker="x", zorder=3, label="Convolved Stim")
        plt.text(whiff_onset, convolved_stim/100, f"{convolved_stim:.2f}", color="purple", fontsize=8, ha="left", va="bottom")

    plt.xlabel("Time (ms)")
    plt.ylabel("Calcium Fluorescence")
    plt.title(f"Calcium Signal with Whiff Onsets ({index_beg} to {index_end})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, f"calcium_pulse_code_{index_beg}_{index_end}_conv.svg"), bbox_inches="tight", pad_inches=0.02)
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

    data_path_list = params["data_path"]

    print("There {} dataset in the folder.".format(len(data_path_list)))

    # create folders -------------------------------------------------------#

    out_path = os.path.join(
        params["res_path"],
        "figures"    
    )
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    postprocess_path = os.path.join(
        params["res_path"],
        "postprocess",
    )

    print(postprocess_path)

    # load data codes-------------------------------------------------------#

    for data_path in data_path_list:
        datafile_name = data_path.split("/")[-1].split(".pt")[0]

        xhat = torch.load(
            os.path.join(postprocess_path, "xhat_{}.pt".format(datafile_name))
        )
        num_trials = xhat.shape[0]

        code= list()
        for i in range(num_trials):
            xihat = xhat[i]
            code.append(torch.sum(xihat, dim=-1))

        code = torch.stack(code, dim=0)
        code_np = code[:,0].squeeze(1).numpy()

    # load onsets -------------------------------------------------------------------#

    with open(params["seg_path"], "rb") as f:
        data = pickle.load(f)

    whiff_onsets = data['HW1']["whiff_onset"]
    phase = data['HW1']['phase']
    conv_stim = data['HW1']['downsampled_convolved']

    # load calcium -------------------------------------------------------------------#

    with open(params["path"], "rb") as f:
        data_calcium = pickle.load(f)

    calcium = data_calcium['calcium_dict']["HW1"].mean(axis=1).to_numpy()

    # load baseline ----------------------------------------------------------------#

    """data = torch.load(params["baseline_path"])
    baseline = data["a"].squeeze()"""

    # -----------------------------------------------------------------------------#

    dict = {'whiff_onsets': whiff_onsets,
            'convolved_stim': conv_stim,
            'codes':code_np, 
            #'baseline': baseline,
            'phase': phase}
    
    whiff_df = pd.DataFrame(dict)

    #plot_calcium_code(result_df, 192)
    plot_calcium_with_whiffs(calcium, whiff_df, 360, 390, params, out_path)

if __name__ == "__main__":
    main()

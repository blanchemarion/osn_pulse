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
        default= "sparseness/results/olfactorycalciumkernellength20num1_2025_02_20_11_30_36"
        #"Results\olfactorycalciumkernellength20num1_2025_01_23_11_02_14"
        # "Results\olfactorycalciumkernellength20num1_2025_01_23_11_02_14",
    )
    parser.add_argument(
        "--figsize",
        type=tuple,
        help="figsize",
        default=(12, 2),
    )
    parser.add_argument(
        "--path",
        type=str,
        help="data path",
        default="sparseness\Data/animals_data_processed.pkl",
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
        "--window-size",
        type=int,
        help="size of the window to look backward",
        default="300",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        help="number of bins",
        default="3",
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


def compute_number_preceding_pulses(valve, phase, params):

    whiff_onsets = np.where(np.diff(valve) > 0)[0]

    results = []
    for _, peak_idx in enumerate(whiff_onsets):        
        phase_pulse = np.mean(phase[peak_idx + 1:peak_idx + 51])
        if 0 <= phase_pulse < np.pi:
            current_event = "inh"
        else:
            current_event = "exh"

        if params["event_filter"] != current_event:
            results.append({"pulse_index": peak_idx, "preceding_train": 0, "phase": f"not_{params['event_filter']}"})
            continue

        start_idx = max(0, peak_idx - params["window_size"])
        preceding_peaks = whiff_onsets[(whiff_onsets >= start_idx) & (whiff_onsets < peak_idx)]

        filtered_peaks = []
        for p_idx in preceding_peaks:
            preceding_phase = np.mean(phase[p_idx + 1:p_idx + 51])
            if 0 <= preceding_phase < np.pi:
                preceding_event = "inh"
            else:
                preceding_event = "exh"

            if preceding_event == current_event:
                filtered_peaks.append(p_idx)

        number_pulses = len(filtered_peaks) + 1
        results.append({"pulse_index": peak_idx, "preceding_train": number_pulses, "phase": current_event})

    result_df = pd.DataFrame(results)

    return result_df

def plot_binned_effect_vs_max_ca(df, params, out_path):
    df = df[df['phase'] == params['event_filter']]

    bins = np.linspace(0, df["preceding_train"].max(), params["n_bins"] + 1)
    binned = pd.cut(df["preceding_train"], bins)
    means = df.groupby(binned)["codes"].mean()
    stds = df.groupby(binned)["codes"].std() 
    counts = df.groupby(binned)["codes"].count()
    std_errors = stds / np.sqrt(counts) 

    bin_centers = range(1,df["preceding_train"].max()+1)

    plt.figure(figsize=(8, 6))
    plt.errorbar(
        bin_centers, 
        means, 
        yerr=std_errors, 
        fmt="o", 
        linestyle="-", 
        capsize=5, 
        label="Mean ± SE"
    )
    plt.ylabel("Mean code")
    plt.xlabel(f"Number of pulses in the [-{params['window_size']}ms, -100ms] window preceding each pulse")
    plt.title(f"Mean code vs. preceding pulse train ({params['event_filter']})")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(out_path, f"codes_vs_train_{params['window_size']}_{params['event_filter']}.svg"), bbox_inches="tight", pad_inches=0.02)
    plt.show()


def plot_code_vs_convolved_stim(codes, conv_stim, phase):

    phase_whiffs = ["inh" if 0 <= p < np.pi else "exh" for p in phase]
    colors = ["green" if label == "inh" else "red" for label in phase_whiffs]

    plt.figure(figsize=(10, 6))
    plt.scatter(conv_stim, codes, c=colors, alpha=0.7, edgecolor="k", linewidths=0.5, label="Inhalation/Exhalation") 
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='Inhalation'),
               plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Exhalation')]
    plt.legend(handles=handles, loc='best')

    plt.xlabel("Convolved Stimulus")
    plt.ylabel("Code")
    plt.title("Code as a Function of Convolved Stimulus")
    plt.grid(alpha=0.3)
    plt.show()


def plot_effect_vs_max_ca(df, params, out_path):
    df = df[df['phase'] == params['event_filter']]
    plt.figure(figsize=(8, 6))

    sns.scatterplot(
        x="codes",
        y="preceding_train",
        hue="phase",  
        palette={"inh": "blue", "exh": "orange"},  
        data=df,
        s=50,  
        alpha=0.8  
    )

    sns.regplot(
        x="codes",
        y="preceding_train",
        data=df,
        scatter=False,  
        line_kws={"color": "red"}, 
        ci=None
    )

    plt.xlabel("Code")
    plt.ylabel("Preceding pulse train")
    plt.title(f"Effect of Preceding Pulse Train vs. code ({params['event_filter']})")
    plt.grid(True)
    plt.legend(title="Phase", loc="best")  
    plt.savefig(os.path.join(out_path, f"scatter_codes_vs_train_{params['window_size']}_{params['event_filter']}.svg"), bbox_inches="tight", pad_inches=0.02)
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

    # load valve -------------------------------------------------------------------#

    with open(params["path"], "rb") as f:
        data = pickle.load(f)

    valve = data["valve_dict"]['HW1']
    phase = data["phase_peaks_dict"]["HW1"]

    # load convolved stimulus -------------------------------------------------------#
    with open(params["seg_path"], "rb") as f:
        data = pickle.load(f)

    phase_stim = data['HW1']['phase']
    conv_stim = data['HW1']['downsampled_convolved']

    # -----------------------------------------------------------------------------#

    df = compute_number_preceding_pulses(valve, phase, params)
    df["codes"] = code_np # np.abs(code_np-baseline)
    plot_binned_effect_vs_max_ca(df, params, out_path)
    plot_effect_vs_max_ca(df, params, out_path)
    plot_code_vs_convolved_stim(code_np, conv_stim, phase_stim)

if __name__ == "__main__":
    main()

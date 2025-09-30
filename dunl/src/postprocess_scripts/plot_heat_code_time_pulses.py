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
        default= "Results\olfactorycalciumkernellength20num1_2025_01_23_11_02_14"
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
        default="sparseness/Data/animals_data_processed.pkl",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        help="number of bins",
        default="20",
    )
    parser.add_argument(
        "--total-window",
        type=int,
        help="total size of the window to look backwards",
        default="2000",
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


def compute_number_preceding_pulses_phase(valve, phase, params, window):

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

        start_idx = max(0, peak_idx - window)
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


def compute_number_preceding_pulses(valve, phase, params, window):

    whiff_onsets = np.where(np.diff(valve) > 0)[0]

    results = []
    for _, peak_idx in enumerate(whiff_onsets):        

        start_idx = max(0, peak_idx - window)
        preceding_peaks = whiff_onsets[(whiff_onsets >= start_idx) & (whiff_onsets < peak_idx)]

        number_pulses = len(preceding_peaks) + 1
        results.append({"pulse_index": peak_idx, "preceding_train": number_pulses})

    result_df = pd.DataFrame(results)

    return result_df


def plot_heat_code_time_pulses(valve, phase, code, params):

    bin_edges = np.linspace(100, params['total_window'], params['n_bins'])  
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  

    heatmap_data = []

    for window in bin_edges[:-1]:
        df = compute_number_preceding_pulses(valve, phase, params, int(window))
        df["codes"] = code 

        grouped = df.groupby("preceding_train")["codes"].mean()

        row = [grouped.get(i, np.nan) for i in range(1, 6)] 
        heatmap_data.append(row)

    heatmap_data = np.array(heatmap_data).T  
    plt.figure(figsize=(10, 5))  # Adjust the aspect ratio
    sns.heatmap(
        heatmap_data,
        xticklabels=np.round(bin_centers / 1000, 2),  # Convert ms to seconds
        yticklabels=range(1, 6),
        cmap="viridis",
        cbar_kws={"label": "Code"}
    )

    # Update x-axis and y-axis labels
    plt.xticks(rotation=45, fontsize=10)  # Rotate and enlarge x-axis labels
    plt.yticks(fontsize=10)  # Enlarge y-axis labels
    plt.xlabel("Window Size (s)", fontsize=11)  # Larger x-axis label
    plt.ylabel("Number of Pulses", fontsize=11)  # Larger y-axis label

    plt.tight_layout()
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

    # ----------------------------------------------------------------------------#

    plot_heat_code_time_pulses(valve, phase, code_np, params)


if __name__ == "__main__":
    main()

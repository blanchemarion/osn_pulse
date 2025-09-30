"""
Quantify how recent pulse history relates to calcium response amplitude

"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import argparse
from scipy.signal import find_peaks

def init_params():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--data-path",
        type=str,
        help="data path",
        default="sparseness/Data/animals_data_processed.pkl",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        help="size of the window to look backward",
        default="2000",
    )
    parser.add_argument(
        "--event-filter",
        type=str,
        help="can take values inh or exh",
        default="inh"
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        help="number of bins",
        default=5
    )
    args = parser.parse_args()
    params = vars(args)

    return params

def compute_number_preceding_pulses(stimulus, phase, params):

    #peaks, _ = find_peaks(stimulus, distance=51)
    peaks = np.where(np.diff(stimulus) > 0)[0]

    results = []
    for _, peak_idx in enumerate(peaks):      

        inh_points = np.sum((0 <= phase[peak_idx+1:peak_idx+501]) & (phase[peak_idx+1:peak_idx+501] < np.pi))

        if inh_points <= 1:
            current_event= "exh"
        else:
            current_event="inh"  

        """phase_pulse = np.mean(phase[peak_idx + 1:peak_idx + 51])
        if 0 <= phase_pulse < np.pi:
            current_event = "inh"
        else:
            current_event = "exh"""

        if params["event_filter"] != current_event:
            results.append({"pulse_index": peak_idx, "preceding_train": 0, "phase": "inh"})
            continue

        start_idx = max(0, peak_idx - params["window_size"])
        preceding_peaks = peaks[(peaks >= start_idx) & (peaks < peak_idx)]

        filtered_peaks = []
        for p_idx in preceding_peaks:

            """preceding_phase = np.mean(phase[p_idx + 1:p_idx + 51])
            if 0 <= preceding_phase < np.pi:
                preceding_event = "inh"
            else:
                preceding_event = "exh"""

            inh_points = np.sum((0 <= phase[p_idx+1:p_idx+501]) & (phase[p_idx+1:p_idx+501] < np.pi))

            if inh_points <= 1:
                preceding_event= "exh"
            else:
                preceding_event="inh"  

            if preceding_event == current_event:
                filtered_peaks.append(p_idx)

        number_pulses = len(filtered_peaks) + 1
        results.append({"pulse_index": peak_idx, "preceding_train": number_pulses, "phase": current_event})

    result_df = pd.DataFrame(results)
    return result_df


def compute_effect_pulse_train(convolved_stimulus, params, alpha=0.005):

    peaks, _ = find_peaks(convolved_stimulus, distance=51)
    amplitudes = convolved_stimulus[peaks]

    results = []
    for i, peak_idx in enumerate(peaks):

        start_idx = max(0, peak_idx - params["window_size"])
        preceding_peaks = peaks[(peaks >= start_idx) & (peaks < peak_idx)]
        normalization_factor = 1+np.sum(convolved_stimulus[preceding_peaks] * np.exp(-(peak_idx - preceding_peaks)*alpha))
        effect = amplitudes[i] * normalization_factor
        results.append({"pulse_index": peak_idx, "preceding_train": effect})

    result_df = pd.DataFrame(results)
    return result_df

def extract_max_ca_interp(effect_pulses, ca_interp):
    max_values = []

    for pulse_idx in effect_pulses["pulse_index"]:
        start_idx = pulse_idx
        end_idx = min(start_idx + 600, len(ca_interp))

        max_value = np.max(ca_interp[start_idx:end_idx])
        max_values.append(max_value)

    effect_pulses["max_ca_interp_after_pulse"] = max_values
    return effect_pulses

def plot_binned_effect_vs_max_ca(effect_pulses, params):

    bins = np.linspace(0, effect_pulses["preceding_train"].max(), params['n_bins'] + 1)
    binned = pd.cut(effect_pulses["preceding_train"], bins)
    means = effect_pulses.groupby(binned)["max_ca_interp_after_pulse"].mean()
    stds = effect_pulses.groupby(binned)["max_ca_interp_after_pulse"].std() 
    counts = effect_pulses.groupby(binned)["max_ca_interp_after_pulse"].count()
    std_errors = stds / np.sqrt(counts) 

    #bin_centers = range(1,effect_pulses["preceding_train"].max()+1)
    bin_centers = bins[:-1] + np.diff(bins) / 2

    plt.figure(figsize=(8, 6))
    #plt.plot(bin_centers, means, marker="o", linestyle="-")
    plt.errorbar(bin_centers, means, yerr=std_errors, fmt='o', linestyle="-", capsize=5, label= "Mean ± SE")
    plt.ylabel("Max calcium response after the pulse")
    plt.xlabel(f"Number of pulses in the {params['window_size']} ms window preceding each pulse")
    plt.title("Max calcium response vs. preceding pulse train")
    plt.legend()
    plt.grid(True)
    plt.show()



def plot_effect_vs_max_ca(effect_pulses):

    effect_pulses = effect_pulses[effect_pulses['phase'] == "inh"]

    plt.figure(figsize=(8, 6))

    sns.scatterplot(
        x="max_ca_interp_after_pulse",
        y="preceding_train",
        hue="phase",  
        palette={"inh": "blue", "exh": "orange"},  
        data=effect_pulses,
        s=50,  
        alpha=0.8  
    )

    sns.regplot(
        x="max_ca_interp_after_pulse",
        y="preceding_train",
        data=effect_pulses,
        scatter=False,  
        line_kws={"color": "red"}, 
        ci=None
    )

    plt.xlabel("Maximum calcium after the pulse")
    plt.ylabel("Preceding pulse train")
    plt.title("Effect of Preceding Pulse Train vs. Max Calcium Response")
    plt.legend()
    plt.grid(True)
    plt.show()




def main():
    
    params = init_params()

    with open(params["data_path"], "rb") as f:
        data = pickle.load(f)

    valve_dict = data["valve_dict"]
    convolved_stim_dict = data["convolved_stim_dict"]
    ca_interp_dict = data["ca_interp_dict"]
    phase_dict = data["phase_peaks_dict"]

    effect_pulses = compute_effect_pulse_train(convolved_stim_dict['HW1'], params)
    number_pulses = compute_number_preceding_pulses(convolved_stim_dict['HW1'], phase_dict['HW1'], params)
    number_pulses = compute_number_preceding_pulses(valve_dict['HW1'], phase_dict['HW1'], params)
    number_pulses = extract_max_ca_interp(number_pulses, ca_interp_dict['HW1'])
    plot_binned_effect_vs_max_ca(number_pulses, params)
    plot_effect_vs_max_ca(number_pulses)

if __name__ == "__main__":
    main()
"""
On the avg over all the ROIs
Performs ridge regression deconvolution on a short calcium window around each pulse to estimate the impulse response kernel.
Aggregates all results into pkl file linear_regression_all.pkl
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from scipy.linalg import toeplitz
from scipy.signal import savgol_filter
import argparse
import pickle
import pandas as pd
import matplotlib.colors as mcolors
from scipy.stats import pearsonr

def init_params():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--data-path",
        type=str,
        help="data path",
        default="sparseness/Data/animals_data_processed.pkl",
    )
    parser.add_argument(
        "--out-path",
        type = str,
        help = "path to solve data",
        default= "sparseness/Data/linear_regression_all.pkl"
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


def align_pulse_calcium(valve, calcium, phase, convolved):

    valve_ts = np.arange(0, len(valve) / 1000, 0.001)  # 1 kHz sampling
    ca_ts = np.arange(0, len(calcium) / 10, 0.1)       # 10 Hz sampling
    
    whiff_onsets = np.where(np.diff(valve) > 0)[0]

    # express valve in ca ref frame
    onset_resp = []
    event_resp = []
    onset_convolved=[]
    for i in range(len(whiff_onsets)):
        start_idx = whiff_onsets[i]
        
        # number of points in inh in the 50 points following a pulse
        inh_points = np.sum((0 <= phase[start_idx+1:start_idx+51]) & (phase[start_idx+1:start_idx+51] < np.pi))
        if inh_points <= 30 :
            current_event= "exh"
        else:
            current_event="inh"
        """if inh_points >= 49 :
            current_event= "inh"
        elif 49 > inh_points >= 25:
            current_event = "between"
        else:
            current_event="exh"""

        window_start = max(0, start_idx - 50 // 2)
        window_end = min(len(convolved), start_idx + 50 // 2)
        mean_value = np.mean(convolved[window_start:window_end])

        # index of the value  of resp_ts that is the closest to stim_ts[start_idx]
        index = np.absolute(ca_ts-valve_ts[start_idx]).argmin()

        onset_resp.append(index)
        event_resp.append(current_event)
        onset_convolved.append(mean_value)

    valve_down = np.zeros(len(calcium))
    valve_down[onset_resp] = 1 

    return valve_down, onset_resp, event_resp, onset_convolved


def deconvolve_response(valve, calcium_signal, kernel_length=10, alpha=0.5, smooth=False):

    baseline = np.median(calcium_signal)
    corrected_signal = calcium_signal - baseline

    if smooth:
        corrected_signal = savgol_filter(corrected_signal, window_length=2, polyorder=1)

    num_samples = len(valve)
    first_column = np.concatenate([valve, np.zeros(kernel_length - 1)])
    first_row = np.zeros(kernel_length)
    X = toeplitz(first_column, first_row)[:num_samples]

    y = corrected_signal
    model = Ridge(alpha=alpha)
    model.fit(X, y)
    kernel = model.coef_

    mean_kernel = np.mean(kernel[2:6]) 
    sym_auc_kernel = np.sum(np.abs(kernel))

    return kernel, mean_kernel, sym_auc_kernel


def kernel_metrics(kernel):

    metrics = {
        "symmetric_auc": np.sum(np.abs(kernel)),
        #"mean": np.mean(kernel[2:6]),
    }
    return metrics


def compute_number_preceding_pulses(df, window):
    results = []
    
    for i, onset in enumerate(df["onset_resp"]):        
        current_event = df["event_resp"].iloc[i]

        start_idx = max(0, onset - window)
        preceding_events = df[(df["onset_resp"] >= start_idx) & (df["onset_resp"] < onset)]

        filtered_events = preceding_events[preceding_events["event_resp"] == current_event]

        number_events = len(filtered_events) + 1 
        results.append(number_events)

    return results

def compute_number_preceding_pulses_in_bin(df, lower_bound, upper_bound):
    results = []
    
    for i, onset in enumerate(df["onset_resp"]):        
        current_event = df["event_resp"].iloc[i]
        
        start_idx = max(0, onset - upper_bound)
        end_idx = max(0, onset - lower_bound)

        preceding_events = df[(df["onset_resp"] >= start_idx) & (df["onset_resp"] < end_idx)]
        filtered_events = preceding_events[preceding_events["event_resp"] == current_event]

        number_events = len(filtered_events) + 1  
        results.append(number_events)

    return results


def main():

    params = init_params()

    animals = ['HW1', 'HW4', 'Sphinx']
    all_results = []

    with open(params["data_path"], "rb") as f:
        data = pickle.load(f)

    for animal in animals:
        valve = data["valve_dict"][animal] / 100
        phase = data["phase_peaks_dict"][animal]
        
        calcium_signal = data["calcium_dict"][animal].mean(axis=1)
        calcium_signal = (calcium_signal - calcium_signal.min()) / (calcium_signal.max() - calcium_signal.min())
        
        convolved = data["convolved_stim_dict"][animal]

        valve_down, onset_resp, event_resp, onset_convolved = align_pulse_calcium(valve, calcium_signal, phase, convolved)

        """kernel,mean_kernel= deconvolve_response(valve_down[3171:3191],calcium_signal[3171:3191])
        time_vector = np.arange(0,2,1/10)
        plot_kernels(kernel, mean_kernel, time_vector)"""

        for i, onset in enumerate(onset_resp):
            valve_segment = valve_down[onset-1:onset+9]
            calcium_segment = calcium_signal[onset-1:onset+9]

            kernel, mean_kernel, sym_auc_kernel = deconvolve_response(valve_segment, calcium_segment)
            
            delta_t = onset_resp[i] - onset_resp[i - 1] if i > 0 else None
            delta_plus_t = onset_resp[i+1] - onset_resp[i] if i < (len(onset_resp)-1) else None

            if event_resp[i] == "inh":
                if i == 0:
                    delta_t1 = None
                    delta_t2 = None  
                elif i == 1:
                    delta_t1 = onset_resp[i] - onset_resp[i - 1] if event_resp[i - 1] == "inh" else None
                    delta_t2 = None
                else:
                    delta_t1 = onset_resp[i] - onset_resp[i - 1] if event_resp[i - 1] == "inh" else None
                    delta_t2 = onset_resp[i - 1] - onset_resp[i - 2] if event_resp[i - 1] == "inh" and event_resp[i - 2] == "inh" else None
            else:
                delta_t1 = None
                delta_t2 = None
                
            all_results.append({
                "onset_resp": onset,
                "onset_convolved": onset_convolved[i],
                "event_resp": event_resp[i],
                "max_resp": max(kernel), #max(calcium_segment.to_numpy()),
                "mean_kernel": mean_kernel,
                "sym_auc_kernel": sym_auc_kernel,
                "kernel": kernel,
                "original_segment": calcium_segment.to_numpy(),
                "mean_original": np.mean(calcium_segment.to_numpy()),
                "delta_t": delta_t,
                "delta_plus_t": delta_plus_t,
                "delta_t1": delta_t1,
                "delta_t2": delta_t2,
                "animal": animal  
            })


    df = pd.DataFrame(all_results)

    with open(params['out_path'], "wb") as f:
        pickle.dump(df, f)


    
if __name__ == "__main__":
    main()

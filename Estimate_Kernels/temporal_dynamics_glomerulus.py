"""
Visualize calcium response timing to odor pulses across ROIs/animals.

Looking at inhalation events with preceding inh-to-inh interval in [0.6 s, <2.0 s] (set via `delta_t` in frames at 10 Hz)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import pickle
import pandas as pd
from sklearn.cluster import KMeans


def init_params():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--data-path",
        type=str,
        help="data path",
        default="sparseness/Data/animals_data_processed.pkl",
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

        inh_points = np.sum((0 <= phase[start_idx+1:start_idx+51]) & (phase[start_idx+1:start_idx+51] < np.pi))

        if inh_points <= 49 :
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
        max_value = np.max(convolved[window_start:window_end])

        # index of the value  of resp_ts that is the closest to stim_ts[start_idx]
        index = np.absolute(ca_ts-valve_ts[start_idx]).argmin()

        onset_resp.append(index)
        event_resp.append(current_event)
        onset_convolved.append(max_value)

    valve_down = np.zeros(len(calcium))
    valve_down[onset_resp] = 1 

    return valve_down, onset_resp, event_resp, onset_convolved


def compute_time_to_peak_dict_animal(data, animal, delta_t):

    all_results = []

    valve = data["valve_dict"][animal] / 100
    phase = data["phase_peaks_dict"][animal]
    convolved = data["convolved_stim_dict"][animal]
    
    calcium_df = data["calcium_dict"][animal]
    
    # Use the first ROI (first column) as the reference for alignment.
    ref_roi = calcium_df.columns[0]
    calcium_signal_ref = calcium_df[ref_roi].to_numpy()
    valve_down, onset_resp, event_resp, onset_convolved = align_pulse_calcium(
        valve, calcium_signal_ref, phase, convolved
    )
    
    for i, onset in enumerate(onset_resp):
        if event_resp[i] == "inh":
            if i == 0:
                delta_t1 = None
                delta_t2 = None  
            elif i == 1:
                delta_t1 = onset_resp[i] - onset_resp[i - 1] if event_resp[i - 1] == "inh" else None
                delta_t2 = None
            else:
                delta_t1 = onset_resp[i] - onset_resp[i - 1] if event_resp[i - 1] == "inh" else None
                delta_t2 = (onset_resp[i - 1] - onset_resp[i - 2] 
                            if event_resp[i - 1] == "inh" and event_resp[i - 2] == "inh" 
                            else None)
        else:
            delta_t1 = None
            delta_t2 = None

        # Only select events where delta_t1 is defined and is >= 20.
        if delta_t1 is not None and delta_t1 >=6 and delta_t1 < delta_t:
        #if delta_t1 is not None and delta_t1 >= delta_t:
        #if delta_t1 is not None and delta_t1 < delta_t:
            row_dict = {
                "onset_resp": onset,
                "onset_convolved": onset_convolved[i],
                "event_resp": event_resp[i],
                "delta_t1": delta_t1,
                "delta_t2": delta_t2,
                "animal": animal  
            }
            
            for roi in calcium_df.columns:
                calcium_signal = calcium_df[roi].to_numpy()
                #calcium_signal = (calcium_signal - calcium_signal.min()) / (calcium_signal.max() - calcium_signal.min())
                #start = max(0, onset - 5)
                start = onset 
                end = onset + 10
                segment = np.array(calcium_signal[start:end])
                row_dict[f"array_resp_roi{roi}"] = segment

            all_results.append(row_dict)
        
    
    df = pd.DataFrame(all_results)
    time_to_peak_dict_all_responses = compute_time_to_peak_all_responses(df)
    time_to_peak_dict_avg_roi = compute_time_to_peak(df)
    return df, time_to_peak_dict_all_responses, time_to_peak_dict_avg_roi


def compare_glom_timescales(df, event_index):

    if event_index in df.index:
        event_row = df.loc[event_index]
        
        roi_columns = [col for col in df.columns if col.startswith("array_resp_roi")]
        
        plt.figure(figsize=(10, 6))
        
        for roi_col in roi_columns:
            signal = event_row[roi_col]
            plt.plot(signal, label=roi_col)
        
        plt.xlabel("Time (frames)")
        plt.ylabel("Calcium Signal")
        plt.title(f"Calcium Signal Response Comparison for Event Index {event_index}")
        plt.legend()
        plt.show()
    else:
        print(f"Event index {event_index} not found in the DataFrame.")


def plot_heatmap_avg_resp_each_roi(df, animal, delta_t):
    roi_columns = [col for col in df.columns if col.startswith("array_resp_roi")]
    
    segment_length = df[roi_columns[0]].iloc[0].shape[0]
    
    roi_info = [] 
    for roi in roi_columns:
        responses = np.array(list(df[roi]))
        mean_response = responses.mean(axis=0)
        global_mean = mean_response.mean()  
        roi_info.append((roi, mean_response, global_mean))
    
    #roi_info_sorted = sorted(roi_info, key=lambda x: x[2])
    
    #heatmap_data = np.array([info[1] for info in roi_info_sorted])
    heatmap_data = np.array([info[1] for info in roi_info])
    
    plt.figure(figsize=(10, 6))
    plt.imshow(heatmap_data, aspect='auto', cmap='viridis',
               extent=[0, segment_length, len(roi_columns), 0])
    plt.colorbar(label='Calcium Signal')
    plt.xlabel("Time (frames)")
    plt.ylabel("Glomerulus (ROI, increasing global mean)")
    plt.title(f"Heatmap of Average Calcium Response\nOrdered by Global Mean Value for {animal} (paired pulses, delta t > 6s and delta t < {delta_t/10}s)")
    plt.show()


def compute_time_to_peak(df):

    roi_columns = [col for col in df.columns if col.startswith("array_resp_roi")]
    
    roi_time_to_peak = {}
    
    for roi in roi_columns:
        responses = np.array(list(df[roi]))
        mean_response = responses.mean(axis=0)
        
        peak_idx = np.argmax(mean_response)
        
        roi_time_to_peak[roi] = peak_idx
    
    return roi_time_to_peak


def compute_time_to_peak_all_responses(df):

    roi_columns = [col for col in df.columns if col.startswith("array_resp_roi")]
    
    roi_time_to_peaks = {}
    
    for roi in roi_columns:
        responses = np.array(list(df[roi]))
        time_to_peaks = [np.argmax(r) for r in responses]
        
        roi_time_to_peaks[roi] = time_to_peaks
    
    return roi_time_to_peaks


def plot_time_to_peak(time_to_peak_dict) :
    time_to_peak_values = list(time_to_peak_dict.values())

    plt.figure(figsize=(6, 4))
    plt.hist(time_to_peak_values, bins=10, alpha=0.7)
    plt.xlabel("Time to Peak (frames)")
    plt.ylabel("Count")
    plt.title("Distribution of Time to Peak Across ROIs")
    plt.show()


def plot_time_to_peak_each_roi_all_animals(roi_time_to_peaks_by_animal, delta_t) :
    num_animals = len(roi_time_to_peaks_by_animal)
    fig, axs = plt.subplots(1, num_animals, figsize=(6*num_animals, 4), sharey=True)

    if num_animals == 1:
        axs = [axs]

    bins = np.linspace(0, 10, 11)
    
    for ax, (animal, roi_time_to_peaks) in zip(axs, roi_time_to_peaks_by_animal.items()):
        time_to_peak_values = list(roi_time_to_peaks.values())
        ax.hist(time_to_peak_values, bins=bins, alpha=0.7)
        ax.set_xlim(0, 11)
        ax.set_xlabel("Time to Peak (frames)")
        ax.set_title(animal)
    
    axs[0].set_ylabel("Count")
    fig.suptitle(f"Distribution of Time to Peak across all ROIs for each Animal\n(paired pulses, delta t >= 0.6s and delta t < {delta_t/10}s)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    

def plot_time_to_peak_all_resp_all_animals(roi_time_to_peaks_by_animal, delta_t):
    num_animals = len(roi_time_to_peaks_by_animal)
    fig, axs = plt.subplots(1, num_animals, figsize=(6*num_animals, 4), sharey=True)
    
    if num_animals == 1:
        axs = [axs]
    bins = np.linspace(0, 10, 11)
    for ax, (animal, roi_time_to_peaks) in zip(axs, roi_time_to_peaks_by_animal.items()):
        all_time_to_peaks = []
        for roi, ttp_list in roi_time_to_peaks.items():
            all_time_to_peaks.extend(ttp_list)
        ax.hist(all_time_to_peaks, bins=bins, alpha=0.7)
        ax.set_xlim(0, 11)
        ax.set_xlabel("Time to Peak (frames)")
        ax.set_title(animal)
    
    axs[0].set_ylabel("Count")
    fig.suptitle(f"Distribution of Time to Peak across all ROIs for each Animal\n(paired pulses, delta t >= 0.6s and delta t < {delta_t/10}s)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()




def main():
    params = init_params()
    animals = ['HW1', 'Sphinx','HW4']
    delta_t = 20

    with open(params["data_path"], "rb") as f:
        data = pickle.load(f)

    time_to_peaks_all_resp_by_animal = {}
    time_to_peaks_avg_roi_by_animal = {}

    for i, animal in enumerate(animals):
        df, time_to_peak_dict, time_to_peak_dict_avg_roi = compute_time_to_peak_dict_animal(data, animal, delta_t)
        time_to_peaks_all_resp_by_animal[animal] = time_to_peak_dict
        time_to_peaks_avg_roi_by_animal[animal] = time_to_peak_dict_avg_roi
        plot_heatmap_avg_resp_each_roi(df, animal, delta_t)

    
    """plot_time_to_peak_all_resp_all_animals(time_to_peaks_all_resp_by_animal, delta_t)
    plot_time_to_peak_each_roi_all_animals(time_to_peaks_avg_roi_by_animal, delta_t)"""

    #compare_glom_timescales(df, 38)

    

if __name__ == "__main__":
    main()

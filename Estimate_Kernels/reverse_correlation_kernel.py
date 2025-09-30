"""Kernel Computation with Reverse Correlation between Response and Pulse
Study the relationship between the stimulus and the response at different time lags to analyze how stimulus events relate to the calcium response
Reverse Correlation: Average integral of the product of stimulus s(t − τ) and response r(t) as a function of time-shift τ 
Similar to estimating an approximate gradient of the response function of a neuron.

Kernel computed as follows: 
 - The response is shifted relative to the pulse signal for a range of time lags 
 - The breathing phase is divided into bins to account for phase-dependent activity.
 - For each time lag:
   - Calculate the response to the stimulus at the given lag (response * stimulus)
   - Accumulate the results into the corresponding phase bins
 - Normalize for each phase and lag by the total number of pulse events """

import numpy as np 
import matplotlib.pyplot as plt
import pickle
import argparse
from scipy.stats import sem

def init_params():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--data-path",
        type=str,
        help="data path",
        default="sparseness/Data/animals_data_processed.pkl",
    )
    parser.add_argument(
        "--fs-breath",
        type=int,
        help="sampling frequency of the breath signal",
        default=1000,
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        help="number of phase bins",
        default=2,
    )
    parser.add_argument(
        "--only-paired",
        type=bool,
        help="include only pulses separated by less than 500ms",
        default=False,
    )
    parser.add_argument(
        "--exclude-paired",
        type=bool,
        help="exclude pulses separated by less than 500ms",
        default=False,
    )
    parser.add_argument(
        "--exclude-boundaries",
        type=bool,
        help="exclude pulses that occur too closely to the boundaries between phase bins",
        default=False,
    )
    parser.add_argument(
        "--downsample",
        type=bool,
        help="downsample valve and phase signals",
        default=False,
    )
    parser.add_argument(
        "--convolved",
        type = bool,
        help = "true to use the convolution of valve and resp as the stimulus",
        default=False,
    )
    parser.add_argument(
        "--all-animals",
        type = bool,
        help = "Aggregate data from all the animals",
        default=False,
    )
    args = parser.parse_args()
    params = vars(args)

    return params

def reverse_correlation_2d(pulse, ca, max_lag, phase_indices, n_phase_bins):
    max_lag = min(max_lag, len(pulse) - 1)
    
    rc_2d = np.zeros((max_lag, n_phase_bins))
    
    for lag in range(max_lag):
        products = ca[lag+1:] * pulse[:-(lag+1)] # Multiply shifted calcium response with pulse signal
        np.add.at(rc_2d[lag], phase_indices[lag+1:], products) # Accumulate products into phase bins
        print(f'done for {lag}')
    # Normalize by total number of pulses
    pulse_sum = np.sum(pulse)
    if pulse_sum > 0:
        rc_2d /= pulse_sum

    return rc_2d

def concat_signal_only_paired_pulses(concat_valve, concat_calcium, concat_phase):
    # Look for paired pulses (<500 ms within pulses)
    whiff_onsets = np.where(np.diff(concat_valve) > 0)[0]

    stim_segments = []
    resp_segments = []
    phase_segments=[]
    for i in range(1, len(whiff_onsets)):
        diff = whiff_onsets[i] - whiff_onsets[i - 1]
        if diff < 500:
            start = max(0, whiff_onsets[i - 1] - 200) 
            end = min(len(concat_valve), whiff_onsets[i] + 2000)  
            
            stim_segment = concat_valve[start:end]
            resp_segment = concat_calcium[start:end]
            phase_segment = concat_phase[start:end]
            
            stim_segments.append(stim_segment)
            resp_segments.append(resp_segment)
            phase_segments.append(phase_segment)
            
            stim_segments.append(np.zeros(1000))
            resp_segments.append(np.zeros(1000))
            phase_segments.append(np.full(1000, 4))

    valve_final = np.concatenate(stim_segments)
    calcium_final = np.concatenate(resp_segments)
    phase_final = np.concatenate(phase_segments)

    return valve_final, calcium_final, phase_final

def concat_signal_exclude_paired_pulses(concat_valve, concat_calcium, concat_phase):
    whiff_onsets = np.where(np.diff(concat_valve) > 0)[0]

    mask = np.ones(len(concat_valve), dtype=bool)

    for i in range(1, len(whiff_onsets)):
        diff = whiff_onsets[i] - whiff_onsets[i - 1]
        if diff < 500:
            start = max(0, whiff_onsets[i - 1] - 200)
            end = min(len(concat_valve), whiff_onsets[i] + 2000)
            mask[start:end] = False  

    valve_final = concat_valve[mask]
    calcium_final = concat_calcium[mask]
    phase_final = concat_phase[mask]

    return valve_final, calcium_final, phase_final

def concat_signal_exclude_boundaries(concat_valve, concat_calcium, concat_phase, phase_bins):
    # Exclude points that are too close to the intervals' boundaries
    margin = 0.5
    mask = np.zeros_like(concat_phase, dtype=bool)
    for i in range(len(phase_bins) - 1):
        lower_bound = phase_bins[i] + margin
        upper_bound = phase_bins[i + 1] - margin
        bin_mask = (concat_phase >= lower_bound) & (concat_phase <= upper_bound)
        mask |= bin_mask

    concat_phase = concat_phase[mask]
    concat_valve = concat_valve[mask]
    concat_calcium = concat_calcium[mask]
    
    return concat_valve, concat_calcium, concat_phase

def downsample_valve_phase(concat_valve, concat_phase, concat_calcium):
    # Aggregate stimulus and phase data over specific time windows preceding each response ts
    stim_ts = np.arange(0, len(concat_valve)/1000, 0.001) 
    resp_ts = np.arange(0, len(concat_calcium)/10, 0.1) 

    window_size_stim = 0.1 # seconds
    window_size_phase = 0.01
    stim_agg = np.zeros_like(resp_ts)
    phase_agg = np.zeros_like(resp_ts)

    for i, t in enumerate(resp_ts):
        start_time_stim = t - window_size_stim
        valid_indices_stim = np.where((stim_ts >= start_time_stim) & (stim_ts < t))[0]
        if len(valid_indices_stim) > 0:
            stim_agg[i] = np.sum(concat_valve[valid_indices_stim])
        else:
            stim_agg[i] = 0  # Assign 0 if no valid indices

        start_time_phase = t - window_size_phase
        valid_indices_phase = np.where((stim_ts >= start_time_phase) & (stim_ts < t))[0]
        if len(valid_indices_phase) > 0:
            phase_agg[i] = np.mean(concat_phase[valid_indices_phase])
        else:
            phase_agg[i] = 0  # Assign 0 if no valid indices

    return stim_agg, phase_agg

def plot_2d_kernel(rc_2d, max_lag):
    # Plot the 2D kernel
    plt.figure(figsize=(10, 6))
    img = plt.imshow(rc_2d.T, aspect='auto', origin='lower', extent=[0, max_lag, 0, 2*np.pi], cmap='viridis')
    plt.colorbar(img, label='Correlation')  
    plt.title(f'2D Kernel estimated with Reverse Correlation: Lag vs Breathing Phase')
    plt.xlabel('Lag (ms)')
    plt.ylabel('Breathing Phase (radians)')
    plt.tight_layout()
    plt.show()

def plot_kernel_bin(rc_2d):

    # Plot the kernel for each phase
    plt.figure(figsize=(8, 6))
    time = np.arange(len(rc_2d.T[0]))
    for i in range(len(rc_2d[0])):
        #plt.plot(time, rc_2d.T[i], label=f'Phase Bin {i}', color='b')
        plt.plot(time, rc_2d.T[i], color='b')
    plt.xlabel('Lag (τ)')
    plt.ylabel('Kernel Amplitude')
    #plt.legend(loc='best')
    #plt.title(f'Comparison of Kernels Across Phases')
    plt.title(f'Reverse Correlation Kernel - HW1')
    plt.tight_layout()
    plt.show()
    


def plot_kernels_each_animal(rc_list):
    plt.figure(figsize=(8, 6))
    time = np.arange(rc_list[0].shape[0])

    animals = ['HW1', 'HW4', 'Sphinx']
    base_colors = ['blue', 'red', 'green'] 
    alphas = [0.9, 0.4] 

    for j, rc_2d in enumerate(rc_list):
        for i in range(rc_2d.shape[1]):
            if i == 0:
                label = f"{animals[j]} - Inhalation"
            elif i == 1 :
                label = f"{animals[j]} - Exhalation"
            plt.plot(
                time, 
                rc_2d[:, i], 
                color=base_colors[j % len(base_colors)], 
                alpha=alphas[i % len(alphas)],
                linewidth=2,
                label=label
            )

    plt.xlabel('Lag (τ)')
    plt.ylabel('Kernel Amplitude')
    plt.title('Reverse Correlation Kernels')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


def plot_actual_vs_reconstructed(rc_2d, concat_calcium, concat_valve, valve_original, concat_phase, phase_bins, max_lag, start_idx, end_idx):
    
    rc_2d = np.array(rc_2d)
    print(rc_2d.shape)
    
    # Compare the reconstructed signal to the actual signal
    sub_valve_original = valve_original[start_idx:end_idx]
    sub_valve = concat_valve[start_idx:end_idx]
    sub_phase = concat_phase[start_idx:end_idx]
    sub_phase_indices = np.digitize(sub_phase, phase_bins) - 1
    
    print(sub_phase)

    predicted_response = np.zeros(len(sub_valve))
    for t in range(max_lag, len(sub_valve)):
        for lag in range(max_lag):
            predicted_response[t] += rc_2d[lag, sub_phase_indices[t-lag]] * sub_valve[t-lag]

    sub_resp = concat_calcium[start_idx:end_idx]

    sub_resp_zero_mean = sub_resp - np.mean(sub_resp)
    predicted_response_zero_mean = predicted_response - np.mean(predicted_response)
    sub_resp_norm = (sub_resp_zero_mean - np.min(sub_resp_zero_mean)) / (np.max(sub_resp_zero_mean) - np.min(sub_resp_zero_mean))
    predicted_response_norm = (predicted_response_zero_mean - np.min(predicted_response_zero_mean)) / (np.max(predicted_response_zero_mean) - np.min(predicted_response_zero_mean))

    #sub_valve_norm = (sub_valve - np.min(sub_valve)) / (np.max(sub_valve) - np.min(sub_valve))
    sub_valve_original_norm = (sub_valve_original - np.min(sub_valve_original)) / (np.max(sub_valve_original) - np.min(sub_valve_original))

    t = np.arange(start_idx, start_idx+len(predicted_response_norm)) 
    plt.figure(figsize=(12, 4))
    plt.plot(t, sub_resp_norm, label='Actual Calcium Response', color='black', alpha=0.7)
    plt.plot(t, predicted_response_norm, label='Predicted Response', color='blue', linestyle='--', alpha=0.7)
    #plt.plot(t, sub_valve_norm, label='Stimulus Convolved', color='black', alpha=0.4)
    plt.plot(t, sub_valve_original_norm, label='Valve Opening', color='grey', alpha=0.4)
    plt.title(f'Actual vs. Predicted Calcium Response (Reverse Correlation)')
    plt.xlabel('Time (ms)')
    plt.ylabel('Signal')
    plt.legend()
    plt.show()
    
def plot_actual_vs_reconstructed_1d(rc_2d, concat_calcium, concat_valve, valve_original, max_lag, start_idx, end_idx):
    rc_2d = np.array(rc_2d)
    
    sub_valve_original = valve_original[start_idx:end_idx]
    sub_valve = concat_valve[start_idx:end_idx]
    predicted_response = np.zeros(len(sub_valve))
    for t in range(max_lag, len(sub_valve)):
        for lag in range(max_lag):
            predicted_response[t] += rc_2d[lag, 0] * sub_valve[t - lag]

    sub_resp = concat_calcium[start_idx:end_idx]

    sub_resp_zero_mean = sub_resp - np.mean(sub_resp)
    predicted_response_zero_mean = predicted_response - np.mean(predicted_response)
    
    sub_resp_norm = (sub_resp_zero_mean - np.min(sub_resp_zero_mean)) / (np.max(sub_resp_zero_mean) - np.min(sub_resp_zero_mean))
    predicted_response_norm = (predicted_response_zero_mean - np.min(predicted_response_zero_mean)) / (np.max(predicted_response_zero_mean) - np.min(predicted_response_zero_mean))
    sub_valve_original_norm = (sub_valve_original - np.min(sub_valve_original)) / (np.max(sub_valve_original) - np.min(sub_valve_original))
    
    t_axis = np.arange(start_idx, start_idx + len(predicted_response_norm))
    plt.figure(figsize=(12, 4))
    plt.plot(t_axis, sub_resp_norm, label='Actual Calcium Response', color='black', alpha=1, linewidth=0.7)
    plt.plot(t_axis, predicted_response_norm, label='Predicted Response', color='blue', alpha=1, linewidth=0.7)
    plt.plot(t_axis, sub_valve_original_norm, label='Valve Opening', color='red', alpha=1, linewidth=0.7)
    plt.title('Actual vs. Predicted Calcium Response (Reverse Correlation)')
    plt.xlabel('Time (ms)')
    plt.ylabel('Signal')
    plt.legend()
    plt.show()

def main():
    
    params = init_params()

    with open(params["data_path"], "rb") as f:
        data = pickle.load(f)

    valve_dict= data["valve_dict"]
    calcium_dict= data["calcium_dict"]
    ca_interp_dict = data["ca_interp_dict"]
    peaks_phase_dict = data["phase_peaks_dict"]
    convolved_stim_dict = data['convolved_stim_dict']
    t_valve = data["t_valve"]
    t_calcium = data["t_calcium"]

    n_phase_bins = params["n_bins"]
    phase_bins = np.linspace(0, 2*np.pi, n_phase_bins + 1)

    calcium_all = []
    valve_all =[]
    phase_all = []
    valve_original_all=[]
        
    for animal in valve_dict.keys():
        valve_data = valve_dict[animal]
        calcium_data = calcium_dict[animal]
        ca_interp_data = ca_interp_dict[animal]
        phase_peaks_data = peaks_phase_dict[animal]
        convolved_stim_data = convolved_stim_dict[animal]

        valve_original=valve_data
        
        if params['convolved']:
            valve_data = convolved_stim_data # use the conv between valve and resp as the stimulus

        if not params['downsample']:
            calcium_data = ca_interp_data
        else:
            max_ca_duration = t_calcium[animal][-1] # duration of calcium data
            max_valve_idx = np.searchsorted(t_valve[animal], max_ca_duration) # index in valve_ts that corresponds to max ca duration
            valve_data = valve_data[:max_valve_idx]
            phase_peaks_data = phase_peaks_data[:max_valve_idx]

        calcium_all.append(calcium_data.mean(axis=1).to_numpy())
        valve_all.append(valve_data)
        phase_all.append(phase_peaks_data)
        valve_original_all.append(valve_original)

    if params['all_animals']:
        concat_calcium = np.concatenate(calcium_all)
        concat_valve = np.concatenate(valve_all)
        concat_phase = np.concatenate(phase_all)
        concat_valve_original =np.concatenate(valve_original_all)
    else :
        concat_calcium = calcium_all[0]
        concat_valve= valve_all[0]
        concat_phase = phase_all[0]
        concat_valve_original = valve_original_all[0]

    if params["only_paired"]:
        concat_valve, concat_calcium, concat_phase = concat_signal_only_paired_pulses(concat_valve, concat_calcium, concat_phase)
    if params["exclude_paired"]:
        concat_valve, concat_calcium, concat_phase = concat_signal_exclude_paired_pulses(concat_valve, concat_calcium, concat_phase)
    if params['exclude_boundaries']:
        concat_valve, concat_calcium, concat_phase = concat_signal_exclude_boundaries(concat_valve, concat_calcium, concat_phase, phase_bins)
    if params['downsample']:
        concat_valve, concat_phase = downsample_valve_phase(concat_valve, concat_phase, concat_calcium)
    
    if not params['downsample']:
        max_lag = 1000 # 1.5s at 1kHz 
    else:
        max_lag = 15 # 1.5s at 10Hz
        
                
    """phase_indices=[]
    rc_2d = []
    for i in range(len(calcium_all)):
        phase = np.digitize(phase_all[i], phase_bins) - 1
        kernel = reverse_correlation_2d(valve_all[i], calcium_all[i], max_lag, phase, n_phase_bins)
        phase_indices.append(phase)
        rc_2d.append(kernel)
        
    plot_kernels_each_animal(rc_2d)"""
    
    phase_bins = np.digitize(concat_phase, phase_bins) - 1
    rc_2d = reverse_correlation_2d(concat_valve, concat_calcium, max_lag, phase_bins, n_phase_bins)
            
    #plot_2d_kernel(rc_2d, max_lag)
    plot_kernel_bin(rc_2d)
    plot_actual_vs_reconstructed_1d(rc_2d, concat_calcium, concat_valve, concat_valve_original, max_lag, 100000, 125000)
    #plot_actual_vs_reconstructed(rc_2d, concat_calcium, concat_valve, concat_valve_original, concat_phase, phase_bins, max_lag, 100000, 125000)

if __name__ == "__main__":
    main()


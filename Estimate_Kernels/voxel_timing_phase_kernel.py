"""
Binning by phase
Estimate stimulus response kernels via voxel timing reverse correlation.
Aligns calcium response timestamps to valve indices, making it useful when stimulus and response are sampled at different rates.

Based on method from "Using slow frame rate imaging to extract fast receptive fields" (https://www.nature.com/articles/s41467-019-12974-0)
"""

import numpy as np
import matplotlib.pyplot as plt 
import pickle 
from scipy.linalg import toeplitz


def predict_S_time(num_kernel_el, num_resp, num_stim):
    t_S = 1.0e-8*num_resp*(60+num_kernel_el)
    t_ST = 1.4e-8*num_resp*(20+num_kernel_el);
    t_D = (1.8e-8*num_stim + 3e-9*num_resp)*num_kernel_el;

    ts = [t_S,t_ST,t_D]
    return ts


"""Calculates a S matrix that captures the stimulus values in a time window around each response timestamp
Each row of S corresponds to one response time (resp_ts)
Each column of S represents a point in time, spanning from numBack steps before to numForward steps after the corresponding sIdx"""
def generateS_sparse(stim,sIdxs,numForward,numBack):
    #S = np.zeros((len(sIdxs), numForward+numBack+1))
    S = np.zeros((len(sIdxs), numForward+numBack))
    for ii in range(len(sIdxs)):
        #S[ii,:] = stim[sIdxs[ii]-numBack : sIdxs[ii]+numForward+1]
        S[ii,:] = stim[sIdxs[ii]-numBack : sIdxs[ii]+numForward]
    return S

def generateS_sparseTranspose(stim,sIdxs,numForward,numBack):
    S = np.zeros((numForward+numBack,len(sIdxs)))
    for ii in range(len(sIdxs)):
        #S[:,ii] = stim[sIdxs[ii]-numBack : sIdxs[ii]+numForward+1]
        S[:,ii] = stim[sIdxs[ii]-numBack : sIdxs[ii]+numForward]
    return S.T

def generateS_dense(stim,sIdxs,numForward,numBack):
    row = np.zeros(numForward+numBack)
    row[0] = stim[0]
    toep = toeplitz(stim,row)
    S_flipped = toep[sIdxs+numForward,:]
    return S_flipped

def apply_moving_average(kernel, window_size=75):
    return np.convolve(kernel, np.ones(window_size)/window_size, mode='same')


def calculate_voxel_timing_filter(stim_ts, stim, breath, resp_ts, resp, num_stim_past, num_stim_future, num_bins, method='ols', min_len=0):

    if method not in ['xcorr', 'ols', 'asd']:
        raise ValueError("Invalid method. Use 'xcorr', 'ols', or 'asd'.")
    
    stim = np.asarray(stim).flatten()
    stim_ts = np.asarray(stim_ts).flatten()

    breath = np.asarray(breath).flatten()

    resp = np.asarray(resp).flatten()
    resp_ts = np.asarray(resp_ts).flatten()

    if len(stim) != len(stim_ts):
        raise ValueError("stim and stim_ts must have the same length.")
    if len(resp) != len(resp_ts):
        raise ValueError("resp and resp_ts must have the same length.")
    if np.any(np.diff(stim_ts) <= 0):
        raise ValueError("stim_ts must be monotonically increasing.")
    if num_stim_past < 0 or num_stim_future < 0:
        raise ValueError("num_stim_past and num_stim_future must be non-negative.")

    # Ensure stimulus timestamps are regular
    stim_dt = np.mean(np.diff(stim_ts))
    stim_dt_std = np.std(np.diff(stim_ts))
    if stim_dt_std / stim_dt > 0.1:
        raise ValueError("Stimulus timestamps must be regularly spaced.")
    
    # select the part of the response that correspod to activation events
    """peaks ,_ = find_peaks(resp, height = 0.025,distance=1000)
    threshold = 50
    all_indices = set()
    for peak in peaks:
        start = max(0, peak - threshold)  
        end = min(len(resp) - 1, peak + threshold)  
        all_indices.update(range(start, end + 1))  
    all_peaks= np.array(sorted(all_indices)) 

    resp_ts_peaks = resp_ts[all_peaks]

    time_offset = 0 # Stimulus starts at time 0
    stim_dt = 0.001 # Stimulus sampling interval"""

    # maps response timestamps to stim timestamps using linear fit
    indices = np.arange(1, len(stim_ts) + 1)
    X = np.vstack([indices, np.ones(len(stim_ts))]).T  # Design matrix
    b = np.linalg.lstsq(X, stim_ts, rcond=None)[0]     # Least squares fit
    stim_dt = b[0]  # Slope (sampling interval)
    time_offset = b[1]  # Intercept (offset)
    #stim_index = np.round((resp_ts_peaks - time_offset) / stim_dt).astype(int)
    stim_index = np.round((resp_ts - time_offset) / stim_dt).astype(int)
    valid_index = (stim_index - num_stim_past > 0) & (stim_index + num_stim_future <= len(stim))
    stim_index = stim_index[valid_index]
    r = resp[valid_index]
    #stim_index_resp = stim_ts[stim_index]

    # NOTE
    # resp_ts_peaks: indices of the response peaks in the reference frame of the response
    # stim_index: indices of the response peaks in the reference frame of the stimulus


    phase_bins = np.linspace(0, 2*np.pi, num_bins + 1)  # Bin edges
    #bin_indices = np.digitize(breath, phase_bins) - 1       # Assign bin indices (0 to num_bins-1)
    bin_indices = np.digitize(breath, phase_bins) - 1 

    kernels = []
    kernels_timestamps = []
    for bin_index in range(num_bins):
        valid_stim_indices = bin_indices == bin_index
        stim_index_bin = np.where(valid_stim_indices)[0]

        # stim_ts_bin_valid: indices of the valid response samples, mapped to stimulus time (both in the appropriate phase bin and that are valid)
        stim_ts_bin_valid= np.intersect1d(stim_index, stim_index_bin) 
                
        ### NOTE 
        # Problem: the resp indices chosen should be selected on the basis of the phase at which the stimulus is integrated, not the resp like here

        # S: each row corresponds to a neural response sample and contains stimulus values for a specified time window around this response
        fastest_method = np.argmin(
            predict_S_time(num_kernel_el=(num_stim_past + num_stim_future + 1), num_resp=len(resp),num_stim=len(stim))) + 1 
        if fastest_method == 1:
            S = generateS_sparse(stim, stim_ts_bin_valid, num_stim_future, num_stim_past)
        elif fastest_method == 2:
            S = generateS_sparseTranspose(stim, stim_ts_bin_valid, num_stim_future, num_stim_past)
        elif fastest_method == 3:
            S = generateS_dense(stim, stim_ts_bin_valid, num_stim_future, num_stim_past)

        valid_index = np.isin(stim_index, stim_index_bin)
        resp_bin = r[valid_index]

        if method == 'xcorr':
            f_hat = np.dot(resp_bin.T,S) / len(resp_bin)
        elif method == 'ols':
            f_hat = np.linalg.lstsq(S, resp_bin, rcond=None)[0]    
        elif method == 'asd':
            raise NotImplementedError("ASD method requires additional implementation.")
        
        #smoooth the filters
        f_hat= apply_moving_average(f_hat)
        f_hat = np.flipud(f_hat.ravel())

        f_hat_ts = stim_dt*np.arange(-(num_stim_past), num_stim_future)
        kernels.append(f_hat)
        kernels_timestamps.append(f_hat_ts) 

    """if fastest_method < 3: #flip to match standard convention (but already implemented for 3)
        f_hat = np.flipud(f_hat.ravel())"""

    # reshape kernels as an array where each row rpz a lag and each column a phase
    kernels_array = np.column_stack(kernels)

    # reconstruct the signal
    stim_subset = stim[0:100000]
    breath_subset = breath[0:100000]
    predicted_response = np.zeros(len(stim_subset))
    """for t in range(num_stim_past, len(stim_subset)):
        for lag in range(num_stim_past):
            phase_bin_index = np.digitize(breath_subset[t-lag], phase_bins) - 1
            predicted_response[t] += kernels_array[lag, phase_bin_index] * stim_subset[t-lag]"""

    return kernels, kernels_array, kernels_timestamps, predicted_response


def plot_kernel_phase(kernels, f_hat_ts, key, n_phase_bins, title_suffix=""):
    ncols = 4
    nrows = (n_phase_bins + ncols - 1) // ncols  # Calculate rows based on the number of bins
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, nrows * 2), sharex=True, sharey=True)
    axes = axes.flatten()

    # Plot each kernel in its own subplot
    for i in range(len(kernels)):
        axes[i].plot(f_hat_ts[i], kernels[i])
        axes[i].set_xlabel('Lag (τ)')
        axes[i].set_ylabel('Kernel Amplitude')
        axes[i].legend([f'Phase Bin {i + 1}'], loc='best')

    for i in range(len(kernels), len(axes)):
        axes[i].axis('off')

    fig.suptitle(f'Kernels for Each Phase {title_suffix}', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95]) 
    fig.savefig(f'Phase_Kernel_{key}_voxel.jpg')
    plt.show()

    """plt.figure(figsize=(10, 6))
    for i, kernel in enumerate(kernels):
        plt.plot(f_hat_ts[i], kernel, label=f'Phase Bin {i + 1}')
    plt.xlabel('Lag (τ)')
    plt.ylabel('Kernel Amplitude')
    plt.legend(loc='best')
    plt.title(f'Comparison of Kernels Across Phases ({title_suffix})')
    plt.grid(True)
    plt.tight_layout()
    plt.show()"""
    
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
    plt.title(f'Reverse Correlation Kernel - Sphinx')
    plt.tight_layout()
    plt.show()
    

def plot_kernels_each_animal(kernels_array_dict):
    plt.figure(figsize=(8, 6))
    
    # Get one of the kernel arrays to define the time axis.
    first_key = list(kernels_array_dict.keys())[0]
    time = np.arange(kernels_array_dict[first_key].shape[0])
    
    base_colors = ['blue', 'red', 'green'] 
    alphas = [0.9, 0.4]
    
    for j, (animal, rc_2d) in enumerate(kernels_array_dict.items()):
        for i in range(rc_2d.shape[1]):
            if i == 0:
                label = f"{animal} - Inhalation"
            elif i == 1:
                label = f"{animal} - Exhalation"
            else:
                label = f"{animal} - Bin {i}"
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


def main():

    with open("sparseness/Data/animals_data_processed.pkl", "rb") as f:
        data = pickle.load(f)
        
    animals = ['HW1', 'HW4', 'Sphinx']

    valve_dict= data["valve_dict"]
    calcium_dict= data["calcium_dict"]
    ca_interp_dict = data["ca_interp_dict"]
    phase_peaks_dict = data["phase_peaks_dict"]
    convolved_stim_dict = data['convolved_stim_dict']
    t_valve = data["t_valve"]
    t_calcium = data["t_calcium"]
    
    num_stim_past = 1500 # Include 2000 past stimulus samples (in the ref frame of the stimulus)
    num_stim_future = 0  # Assume no future stimulus effect
    num_bins=2

    kernels = []
    kernels_array = []
    kernels_timestamps = []
    predicted_responses = []

    for i, key in enumerate(calcium_dict.keys()):

        stim = valve_dict[key]/100
        resp = calcium_dict[key].mean(axis=1).to_numpy()
        phase = phase_peaks_dict[key]
        #phase = np.random.uniform(-np.pi, np.pi, size=len(stim)) # random phase to eahc point


        stim_ts = np.arange(0, len(stim)/1000, 0.001) # sampled every 0.001 seconds (1000 Hz)
        resp_ts = np.arange(0, len(resp)/10, 0.1) # sampled every 0.1s (10 Hz)

        # Estimate the filter
        #f_hat, f_hat_ts, S = calculate_voxel_timing_filter(stim_ts, stim, resp_ts, resp, num_stim_past, num_stim_future, method='xcorr')
        kernel, kernel_array, kernel_timestamp, predicted_response= calculate_voxel_timing_filter(stim_ts, stim, phase, resp_ts, resp, num_stim_past, num_stim_future, num_bins, method='xcorr')

        kernels.append(kernel)
        kernels_array.append(kernel_array)
        kernels_timestamps.append(kernel_timestamp)
        predicted_responses.append(predicted_response)

    kernels_dict = dict(zip(animals, kernels))
    kernels_array_dict = dict(zip(animals, kernels_array))
    kernels_timestamps_dict = dict(zip(animals, kernels_timestamps))
    predicted_resp_dict = dict(zip(animals, predicted_responses))

    plot_kernels_each_animal(kernels_array_dict)

    """with open("voxel_phase_kernels_all.pkl","wb") as f:
        pickle.dump(kernels_dict,f)

    with open("voxel_phase_kernels_arrays_all.pkl","wb") as f:
        pickle.dump(kernels_array_dict,f)

    with open("voxel_phase_kernels_ts_all.pkl","wb") as f:
        pickle.dump(kernels_timestamps_dict,f)

    with open("voxel_phase_predicted_all.pkl","wb") as f:
        pickle.dump(predicted_resp_dict,f)"""
        
    

    
    for i, key in enumerate(kernels_array_dict.keys()):

        """plt.figure(figsize=(8, 3))
        img = plt.imshow(kernels_array_dict[key].T, aspect='auto', origin='lower', extent=[0, num_stim_past, -np.pi, np.pi], cmap='viridis')
        plt.colorbar(img, label='Correlation')  
        plt.title(f'2D Kernel estimated with Reverse Correlation: Lag vs Breathing Phase for {key}')
        plt.xlabel('Lag (ms)')
        plt.ylabel('Breathing Phase (radians)')
        plt.tight_layout()
        plt.show()"""
        
        #plot_kernel_bin(kernels_array_dict[key])
        #plot_kernel_phase(kernels_dict[key], kernels_timestamps_dict[key], key, num_bins, title_suffix=f"voxel_phase_phase_peaks_{key}")

        """stim = valve_dict[key]/100
        resp = calcium_dict[key].mean(axis=1).to_numpy()
        stim_ts = np.arange(0, len(stim)/1000, 0.001) # sampled every 0.001 seconds (1000 Hz)
        resp_ts = np.arange(0, len(resp)/10, 0.1) # sampled every 0.1s (10 Hz)

        plt.figure(figsize=(10, 3))
        plt.plot(resp_ts[0:1000], resp[0:1000], label='Actual Calcium Response', color='blue', alpha=0.7)
        plt.plot(stim_ts[0:100000], predicted_resp_dict[key], label='Predicted Response', color='red', linestyle='--', alpha=0.7)
        plt.title(f'Actual vs. Predicted Calcium Response for {key} using phase binned Voxel timing and rev corr')
        plt.xlabel('Time (ms)')
        plt.ylabel('Signal')
        plt.legend()
        plt.savefig(f'Predicted_Response_{key}_Phase_Voxel_Timing.jpg')
        plt.show()"""


if __name__ == "__main__":
    main()
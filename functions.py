import numpy as np
from scipy.signal import butter, filtfilt
import pickle
from scipy.signal import hilbert, find_peaks
import matplotlib.pyplot as plt


def butter_bandpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_bandpass_filter(data, cutoff, fs, order=5):
    b, a = butter_bandpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def get_sniff_phase(breath, fc, fs, min_dist):
    # High-pass filter
    b_hp, a_hp = butter(3, fc / (fs / 2), btype='high')
    breath_filtered = filtfilt(b_hp, a_hp, breath)

    # Low-pass smoothing filter
    b_lp, a_lp = butter(3, fc / (fs / 2), btype='low')
    breath_smoothed = filtfilt(b_lp, a_lp, breath_filtered)

    breath_norm = (breath_smoothed - np.mean(breath_smoothed)) / np.std(breath_smoothed)

    min_samples = int((min_dist / 1000) * fs)  # Convert minDist from ms to samples
    peaks, _ = find_peaks(breath_norm, distance=min_samples)

    sniff_phase = np.zeros_like(breath)

    # Assign 0 to 2π phase for each sniff cycle
    for i in range(len(peaks) - 1):
        start, end = peaks[i], peaks[i + 1]
        T = end - start 
        sniff_phase[start:end] = np.linspace(0, 2*np.pi, T, endpoint=False)

    return sniff_phase

def load_dataset(path="sparseness/Data/animals_data_processed.pkl"):

    with open(path, "rb") as f:
        datasets = pickle.load(f)

    fs_breath = 1000 
    fs_calcium = 10

    animals = []
    breath = []
    breath_filt = []
    valve_data = []
    pulse_data = []
    calcium_ds = []
    ca_interp_ds = []

    t_breath = []
    t_pulse = []
    t_calcium = []

    for key, _ in datasets.items():

        animals.append(key)
        breath.append(datasets[key]['breath']['breath'].flatten())
        breath_filt.append(datasets[key]['breath']['breath_filt'].flatten())
        valve_data.append(datasets[key]['breath']['valve'].flatten())
        pulse_data.append(datasets[key]['breath']['pulse'].flatten())
        calcium_ds.append(datasets[key]['calcium_imaging']['ca'])
        ca_interp_ds.append(datasets[key]['calcium_imaging']['ca_nn_interp'])

        t_breath.append(np.arange(len(datasets[key]['breath']['breath'].flatten())) / fs_breath)
        t_pulse.append(np.arange(len(datasets[key]['breath']['pulse'].flatten())) / fs_breath)
        t_calcium.append(np.arange(len(datasets[key]['calcium_imaging']['ca']))/ fs_calcium)

    breath_dict = dict(zip(animals, breath))
    breath_filt_dict = dict(zip(animals, breath_filt))
    valve_dict = dict(zip(animals, valve_data))
    pulse_dict = dict(zip(animals, pulse_data))
    calcium_dict = dict(zip(animals, calcium_ds))
    ca_interp_dict = dict(zip(animals, ca_interp_ds))


    analytic_signal = []
    phase_hilbert = []
    phase_peaks=[]

    for i, key in enumerate(breath_filt_dict.keys()):
        analytic_signal.append(hilbert(breath_filt_dict[key]))
        phase_hilbert.append(np.angle(analytic_signal[i]))
        phase_peaks.append(get_sniff_phase(breath_filt_dict[key], 3, 1000, 120))


    phase_hilbert_dict = dict(zip(animals, phase_hilbert))
    phase_peaks_dict = dict(zip(animals, phase_peaks))

    #return breath_dict, breath_filt_dict, valve_dict, pulse_dict, calcium_dict, ca_interp_dict, instant_phase_dict, t_breath, t_pulse, t_calcium, animals
    return breath_dict, breath_filt_dict, valve_dict, pulse_dict, calcium_dict, ca_interp_dict, phase_hilbert_dict, phase_peaks_dict, t_breath, t_pulse, t_calcium, animals


def plot_predicted_vs_actual(actual, predicted, index_beg, index_end, key, method):

    ca_response_norm = (actual - np.min(actual)) / (np.max(actual) - np.min(actual))
    predicted_response_norm = (predicted - np.min(predicted)) / (np.max(predicted) - np.min(predicted))

    plt.figure(figsize=(20, 6))
    time = np.arange(len(actual))
    plt.plot(time[index_beg:index_end], ca_response_norm[index_beg:index_end], label='Actual Calcium Response (Normalized)', color='blue', alpha=0.7)
    plt.plot(time[index_beg:index_end], predicted_response_norm[index_beg:index_end], label='Predicted Response (Normalized)', color='red', linestyle='--', alpha=0.7)
    plt.title(f'Normalized Actual vs. Predicted Calcium Response for {key}')
    plt.xlabel('Time (ms)')
    plt.ylabel('Normalized Signal')
    plt.legend()
    plt.savefig(f'pred_vs_actual_{key}_{method}.jpg')
    plt.show()


def plot_2d_kernel(array_2d_dict, max_lag, method):
    fig, axs = plt.subplots(nrows=3, figsize=(8, 10), sharex=True)
    for i, key in enumerate(array_2d_dict.keys()):
        img = axs[i].imshow(array_2d_dict[key].T, aspect='auto', origin='lower', extent=[0, max_lag, -np.pi, np.pi], cmap='viridis')
        fig.colorbar(img, ax=axs[i], label='Correlation')  
        axs[i].set_title(f'2D Kernel estimated with Reverse Correlation: Lag vs Breathing Phase for {key}')
        axs[i].set_xlabel('Lag (ms)')
        axs[i].set_ylabel('Breathing Phase (radians)')
    fig.tight_layout()
    fig.savefig(f'2D_Kernels_all_animals_{method}.jpg')
    plt.show()


def plot_kernel_phase(array_2d, n_phase_bins,key, method):
    ncols = 4 
    nrows = (n_phase_bins + ncols - 1) // ncols  

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, nrows * 2), sharex=True, sharey=True)
    axes = axes.flatten()
    for i in range(n_phase_bins):
        axes[i].plot(array_2d.T[i])
        axes[i].set_title(f'Valve Kernel for bin {i}')
        axes[i].set_xlabel('Lag (τ)')
        axes[i].set_ylabel('Kernel Amplitude')
        axes[i].legend([f'Bin {i}'])

    for i in range(n_phase_bins, len(axes)):
        axes[i].axis('off')

    fig.suptitle(f'Kernels for each phase for {key}')
    fig.tight_layout()
    fig.savefig(f'Phase_Kernel_{key}_{method}.jpg')
    plt.show()


def z_score_normalize(data):
    return (data - np.mean(data)) / np.std(data)

def normalize_and_concat(calcium_dict):
    normalized_data = []
    
    for animal, data in calcium_dict.items():
        avg_roi = data.mean(axis=1).to_numpy()
        normalized_data.append(z_score_normalize(avg_roi))

    concatenated_array = np.concatenate(normalized_data)
    return concatenated_array
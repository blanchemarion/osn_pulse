"""Kernel Computation with Pulse-Centered Averaging
Kernel computed as follows: 
 - Assign unique identifiers to odorant pulses
 - Extend them by a fixed time window to capture delayed calcium responses
 - For each pulse, relevant features such as time, phase, valve strength, and adjusted calcium amplitude are extracted and aggregated within the pulse window
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.stats import sem
import math

def max_interval(series, start_index, end_index):
    return series.iloc[start_index:end_index].max()


def compute_kernels(calcium_pulse, n_bins):
    min_phase, max_phase = calcium_pulse['phase'].min(), calcium_pulse['phase'].max()
    bin_edges = np.linspace(min_phase, max_phase, n_bins + 1)     
    bin_indices = np.digitize(calcium_pulse['phase'], bin_edges) - 1

    sniff_kernel = np.zeros(n_bins)  # Mean response per bin
    sniff_kernel_sem = np.zeros(n_bins)  # Standard error per bin

    for j in range(n_bins):
        indices_in_bin = np.where(bin_indices == j)[0]
        if len(indices_in_bin) > 0:
            values = calcium_pulse['amp_ca'][indices_in_bin]
            sniff_kernel[j] = np.nanmean(values)  # Mean calcium amplitude
            sniff_kernel_sem[j] = sem(values, nan_policy='omit')  # SEM (ignores NaN values)
        else:
            sniff_kernel[j] = np.nan
            sniff_kernel_sem[j] = np.nan  # Assign NaN if no data in bin

    return sniff_kernel, sniff_kernel_sem


def plot_kernel(calcium_pulse_dict, n_bins=30):
    fig, axs = plt.subplots(ncols=len(calcium_pulse_dict), figsize=(12, 3), sharex=True)

    for i, (key, calcium_pulse) in enumerate(calcium_pulse_dict.items()):
        kernels, kernel_sem = compute_kernels(calcium_pulse, n_bins)

        min_phase, max_phase = calcium_pulse['phase'].min(), calcium_pulse['phase'].max()
        phase_edges = np.linspace(min_phase, max_phase, n_bins + 1)
        phase_midpoints = (phase_edges[:-1] + phase_edges[1:]) / 2

        # Plot the mean kernel
        axs[i].plot(phase_midpoints, kernels, color='b', linewidth=2, label='Mean Kernel')

        # Shaded error region (mean ± SEM)
        axs[i].fill_between(phase_midpoints, kernels - kernel_sem, kernels + kernel_sem, 
                            color='b', alpha=0.3, label='SEM')

        axs[i].set_title(f'Sniff Kernel - {key}', fontsize=9)
        axs[i].set_xlabel('Sniff Phase (radians)', fontsize=9)
        axs[i].set_ylabel('Average Calcium Amplitude', fontsize=9)

    fig.tight_layout()
    plt.show()
    


def plot_kernel_each_glom(glom_dict, n_bins=30, animal_key=""):
    n_gloms = len(glom_dict)

    n_cols = math.ceil(math.sqrt(n_gloms))
    n_rows = math.ceil(n_gloms / n_cols)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), sharex=True)
    axs = axs.flatten()  

    for i, (glom_idx, calcium_pulse) in enumerate(glom_dict.items()):
        kernels, kernel_sem = compute_kernels(calcium_pulse, n_bins)

        min_phase, max_phase = calcium_pulse['phase'].min(), calcium_pulse['phase'].max()
        phase_edges = np.linspace(min_phase, max_phase, n_bins + 1)
        phase_midpoints = (phase_edges[:-1] + phase_edges[1:]) / 2

        axs[i].plot(phase_midpoints, kernels, color='b', linewidth=2)
        axs[i].fill_between(phase_midpoints, kernels - kernel_sem, kernels + kernel_sem, 
                            color='b', alpha=0.3)

        axs[i].set_title(f'Glomerulus {glom_idx}', fontsize=9)
        axs[i].set_xlabel('Sniff Phase (radians)', fontsize=9)
        axs[i].set_ylabel('Avg Calcium Amp', fontsize=9)

    for j in range(i + 1, len(axs)):
        axs[j].axis('off')

    fig.suptitle(f"Phase Kernels - Animal {animal_key}", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def assign_pulse_ids(ca_pulse, window=900):
    valve_on = ca_pulse['valve'] != 0

    pulse_starts = np.where(valve_on & ~valve_on.shift(fill_value=False))[0]
    pulse_ends = np.where(~valve_on & valve_on.shift(fill_value=False))[0]

    ca_pulse['pulse_id'] = np.nan
    for pulse_idx, (start, end) in enumerate(zip(pulse_starts, pulse_ends), start=1):
        window_end = min(end + window, len(ca_pulse) - 1)
        ca_pulse.loc[start:window_end, 'pulse_id'] = pulse_idx

    return ca_pulse.dropna(subset=['pulse_id'])

def process_pulse_data(ca_pulse):
    return (
        ca_pulse.groupby('pulse_id')
        .agg(
            time=('time', 'first'),
            phase=('phase', lambda x: x.head(50).mean()),  # Mean of first 50 samples during pulse
            valve=('valve', 'first'),
            amp_ca=('avg_roi_ca', lambda x: max_interval(x, 200, 600))# - abs(max_interval(x, 50, 200)))
        )
        .reset_index(drop=True)
    )

def main():
    # Load the dataset
    with open("sparseness/Data/animals_data_processed.pkl", "rb") as f:
        data = pickle.load(f)

    calcium_pulse_dict = {}
    for key, ca_interp in data["ca_interp_dict"].items():
        ca_pulse = pd.DataFrame({
            'time': data["t_breath"][key],
            'phase': data["phase_peaks_dict"][key],
            'valve': data["valve_dict"][key],
            'avg_roi_ca': ca_interp.mean(axis=1)
        })

        ca_pulse = assign_pulse_ids(ca_pulse)
        calcium_pulse_dict[key] = process_pulse_data(ca_pulse)

    #plot_kernel(calcium_pulse_dict)
    
    for animal_key, ca_interp in data["ca_interp_dict"].items():
        ca_interp_array = np.array(ca_interp)  
        _, n_gloms = ca_interp_array.shape

        glom_dict = {}

        for glom_idx in range(n_gloms):
            ca_pulse = pd.DataFrame({
                'time': data["t_breath"][animal_key],
                'phase': data["phase_peaks_dict"][animal_key],
                'valve': data["valve_dict"][animal_key],
                'avg_roi_ca': ca_interp_array[:, glom_idx]  
            })

            ca_pulse = assign_pulse_ids(ca_pulse)
            processed = process_pulse_data(ca_pulse)
            glom_dict[glom_idx] = processed

        plot_kernel_each_glom(glom_dict, animal_key=animal_key)

if __name__ == "__main__":
    main()


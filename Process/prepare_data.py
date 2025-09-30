######## Script to load the .mat data and organnize the data as a global dict with all the animals; the breath, valve and calcium data for each

import pandas as pd
import numpy as np
import h5py
from scipy.io import loadmat
import pickle
from scipy.interpolate import interp1d
from scipy.signal import resample
import argparse
import matplotlib.pyplot as plt

import sys
sys.path.append("sparseness")
from functions import butter_bandpass_filter, hilbert, get_sniff_phase

# interested in df_data, breath, valve, pulse/odor 

def init_params():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--data-files",
        type=dict,
        help="data files",
        default={
        'HW1': {
            'ca_path': 'sparseness/Data/HW1_50ms/adaptation_data_HW1_50ms_struct.mat',
            'breath_path': 'sparseness/Data/HW1_50ms/HW1_50ms_0.mat'
        },
        'HW4': {
            'ca_path': 'sparseness/Data/HW4_50ms_real/adaptation_data_HW4_50ms_real_struct-001.mat',
            'breath_path': 'sparseness/Data/HW4_50ms_real/HW4_50ms_real_0.mat'
        },
        'Sphinx': {
            'ca_path': 'sparseness/Data/Sphinx/adaptation_data_Sphinx_50ms_random_struct.mat',
            'breath_path': 'sparseness/Data/Sphinx/Sphinx_50ms_random.mat'
        }
        }
    )
    parser.add_argument(
        "--fs-breath",
        type=int,
        help="sampling frequency of the breath signal",
        default=1000,
    )
    parser.add_argument(
        "--fs-calcium",
        type=int,
        help="sampling frequency of the calcium signal",
        default=10,
    )
    parser.add_argument(
        "--highcut",
        type=int,
        help="highcut frequency",
        default=10,
    )
    parser.add_argument(
        "--filter-order",
        type=int,
        help="filter order",
        default=3,
    )
    parser.add_argument(
        "--interp-kind",
        type=str,
        help="interpolation kind",
        default='linear',
    )
    parser.add_argument(
        "--convolve",
        type=bool,
        help="convolve the valve and breath",
        default=True,
    )
    parser.add_argument(
        "--delay",
        type=int,
        help="delay from odor onset to the peak in the PID",
        default=10,
    )

    args = parser.parse_args()
    params = vars(args)

    return params

def convolve_stimuli(breath_filt, valve):

    convolved_signal = breath_filt * valve

    return convolved_signal

def gaussian_weight(phase, mean=np.pi/2, std_dev=np.pi/6):
    return np.exp(-0.5 * ((phase - mean) / std_dev) ** 2)

def process_data(data_files, params):

    datasets = {}
    for dataset, paths in data_files.items():
        breath_data = loadmat(paths['breath_path'])
        breath = breath_data['breath'].flatten()
        valve = breath_data['valve'].flatten()
        pulse = breath_data['pulse'].flatten()

        breath_filt = butter_bandpass_filter(breath, params["highcut"], params["fs_breath"], params["filter_order"]) - np.mean(breath)

        # Shift the valve signal by the delay from odor onset to the peak in the PID 
        valve_delayed = np.zeros_like(valve)
        valve_delayed[params['delay']:] = valve[:-params['delay']]

        with h5py.File(paths['ca_path'], 'r') as mat_file:
            imaging_struct = mat_file['imaging_struct']
            df_data = pd.DataFrame(imaging_struct['df_data'][:])

            calcium_time = np.arange(df_data.shape[0]) / params["fs_calcium"]
            interp_funcs = [
                interp1d(calcium_time, df_data.iloc[:, i], kind=params["interp_kind"], bounds_error=False, fill_value="extrapolate")
                for i in range(df_data.shape[1])
            ]
            df_data_nn_interp = pd.DataFrame(
                np.array([func(np.arange(breath.size) / params["fs_breath"]) for func in interp_funcs]).T
            )

        analytic_signal = hilbert(breath_filt)
        phase_hilbert = np.angle(analytic_signal)
        phase_peaks = get_sniff_phase(breath_filt, 3, params["fs_breath"], 120)

        if params['convolve']:

            weights = np.zeros_like(phase_peaks) 
            #mask = (phase_peaks>=np.pi) & (phase_peaks <= phase_peaks.max())
            #mask = (phase_peaks>phase_peaks.min()) & (phase_peaks < np.pi)

            mu = np.pi / 2  # Center of the peak
            sigma = np.pi / 4  # Spread for Gaussian
            weights = np.exp(-0.5 * ((phase_peaks - mu) / sigma)**2)
            weights *= 0.1 
            """weights[mask] = 1
            weights[~mask] = 0"""
            convolved_stimulus = convolve_stimuli(weights, valve_delayed)
        else:
            convolved_stimulus = None
        
        datasets[dataset] = {
            'breath': breath,
            'breath_filt': breath_filt,
            'valve': valve_delayed,
            'pulse': pulse,
            'convolved_stimulus': convolved_stimulus,
            'calcium': df_data,
            'ca_nn_interp': df_data_nn_interp,
            'phase_hilbert': phase_hilbert,
            'phase_peaks': phase_peaks,
            't_breath': np.arange(len(breath)) / params["fs_breath"],
            't_valve': np.arange(len(valve_delayed)) / params["fs_breath"],
            't_calcium': np.arange(len(df_data)) / params["fs_calcium"]
        }

        print(f"{dataset} data processed")

    return datasets


def save_datasets(datasets, output_path="sparseness/Data/animals_data_processed.pkl"):
    with open(output_path, "wb") as f:
        pickle.dump(datasets, f)
    print(f"All data saved to {output_path}")


def main():
    params = init_params()
    datasets = process_data(params['data_files'], params)
    
    organized_data = {
        'breath_dict': {key: data['breath'] for key, data in datasets.items()},
        'breath_filt_dict': {key: data['breath_filt'] for key, data in datasets.items()},
        'valve_dict': {key: data['valve'] for key, data in datasets.items()},
        'pulse_dict': {key: data['pulse'] for key, data in datasets.items()},
        'convolved_stim_dict': {key: data['convolved_stimulus'] for key, data in datasets.items()},
        'calcium_dict': {key: data['calcium'] for key, data in datasets.items()},
        'ca_interp_dict': {key: data['ca_nn_interp'] for key, data in datasets.items()},
        'phase_hilbert_dict': {key: data['phase_hilbert'] for key, data in datasets.items()},
        'phase_peaks_dict': {key: data['phase_peaks'] for key, data in datasets.items()},
        't_breath': {key: data['t_breath'] for key, data in datasets.items()},
        't_valve': {key: data['t_valve'] for key, data in datasets.items()},
        't_calcium': {key: data['t_calcium'] for key, data in datasets.items()},
        'animals': list(datasets.keys())
    }

    save_datasets(organized_data)

if __name__ == "__main__":
    main()
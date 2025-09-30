"""Phase-Dependent Lagged Stimulus-Response Kernel Estimation
Evaluates the contribution of past stimuli at varying lags and breath phases to neural activity using a weighted average of response-to-stimulus ratios
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from functions import load_dataset, plot_2d_kernel, plot_kernel_phase, plot_predicted_vs_actual
from joblib import Parallel, delayed
import pickle

##### Functions

# Function to compute filter for a single (tau, phi) pair with regularizatio
def compute_filter_with_regularization(odor_pulse, ca, discrete_phase, chunk_size, tau, phi_idx):
    mask_current_phi = discrete_phase == phi_idx
    mask_past_phi = np.roll(mask_current_phi, tau)

    f_tau_phi_val = 0
    count = 0
    n_chunks = len(ca) // chunk_size + 1

    for i in range(n_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, len(ca))

        valid_indices = (odor_pulse[start:end] *
                         mask_current_phi[start:end] *
                         mask_past_phi[start:end]).astype(bool)

        if np.any(valid_indices):
            response_subset = ca[start:end][valid_indices]
            stimulus_subset = odor_pulse[start:end][valid_indices]
            f_tau_phi_val += np.sum(response_subset * stimulus_subset)
            count += np.sum(stimulus_subset**2)
    
    # Add regularization term
    count += lambda_reg
    return f_tau_phi_val / count if count > 0 else 0


# Parallelized computation of the regularized filter matrix
def compute_filter_matrix_with_regularization(odor_pulse, ca, discrete_phase, chunk_size):
    f_tau_phi = np.zeros((len(time_lags), n_bins))
    results = Parallel(n_jobs=-1)(
        delayed(compute_filter_with_regularization)(odor_pulse, ca, discrete_phase, chunk_size, tau, phi_idx)
        for tau_idx, tau in enumerate(time_lags)
        for phi_idx in range(n_bins)
    )

    for tau_idx, tau in enumerate(time_lags):
        for phi_idx in range(n_bins):
            f_tau_phi[tau_idx, phi_idx] = results[tau_idx * n_bins + phi_idx]
    return f_tau_phi


#####

breath_dict, breath_filt_dict, valve_dict, pulse_dict, calcium_dict, ca_interp_dict, hilbert_phase_dict, peaks_phase_dict, t_breath, _, _, animals = load_dataset(path="Processed/all_animals_data.pkl")

max_lag = 2000
time_lags = np.arange(0, max_lag)  
n_bins = 6                      
bin_edges = np.linspace(0, 2*np.pi, n_bins + 1) 
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
chunk_size = 5000  
lambda_reg = 1e-4    

f_tau_phis = []
predicted_responses=[]
for i, key in enumerate(ca_interp_dict.keys()):
    # Estimate the kernels
    ca = ca_interp_dict[key].mean(axis=1).to_numpy()
    odor_pulse = valve_dict[key]/100
    breath_phase = peaks_phase_dict[key]
    discrete_phase = np.digitize(breath_phase, bin_edges) - 1 

    f_tau_phi = compute_filter_matrix_with_regularization(odor_pulse, ca, discrete_phase, chunk_size)

    # Reconstruct the signal
    """predicted_response = np.zeros(len(odor_pulse))
    for t in range(max_lag, len(odor_pulse)):
        for lag in range(max_lag):
            predicted_response[t] += f_tau_phi[lag, discrete_phase[t-lag]] * odor_pulse[t-lag]
    
    predicted_responses.append(predicted_response)"""
    f_tau_phis.append(f_tau_phi)

    print(f'done for {key}')

f_tau_phi_dict = dict(zip(animals, f_tau_phis))
#predicted_resp_dict = dict(zip(animals, predicted_responses))

with open("phase_dep_kernel_dict.pkl","wb") as f:
    pickle.dump(f_tau_phi_dict,f)

##### Plots 

# Plot the 2D kernel
plot_2d_kernel(f_tau_phi_dict, max_lag, 'phase_dep')
# Observations: 

# Plot the kernel for each phase
for _, key in enumerate(f_tau_phi_dict.keys()):
    plot_kernel_phase(f_tau_phi_dict[key], n_bins, key, 'phase_dep')
# Observations: 
# - some phase bins exhbit distinct peaks -> the filter seem to capture some phase dependant sensitivities to the stimulus pulses 
# - multiple peaks for some phase bins: some stimuli may participate in later ca responses
# - phase bins with no peaks (flat kernels): effectively captures the phases when the neurons are insesnitve to pulses
# Problems: 
# - High noise for each kernel 


# Compare the reconstructed signal to the actual signal
"""for i, key in enumerate(predicted_resp_dict.keys()):
    plot_predicted_vs_actual(ca_interp_dict[key].mean(axis=1).to_numpy(), predicted_resp_dict[key], 0, 50000, key, 'phase_dep')"""
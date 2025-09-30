import numpy as np
from functions import load_dataset, plot_kernel_phase
import matplotlib.pyplot as plt 
import pickle 
from scipy.special import eval_laguerre
from scipy.linalg import toeplitz



def predict_S_time(num_kernel_el, num_resp, num_stim):
    t_S = 1.0e-8*num_resp*(60+num_kernel_el)
    t_ST = 1.4e-8*num_resp*(20+num_kernel_el);
    t_D = (1.8e-8*num_stim + 3e-9*num_resp)*num_kernel_el;

    ts = [t_S,t_ST,t_D]
    return ts

def generateS_sparse(stim,sIdxs,numForward,numBack):
    S = np.zeros((len(sIdxs), numForward+numBack))
    for ii in range(len(sIdxs)):
        S[ii,:] = stim[sIdxs[ii]-numBack : sIdxs[ii]+numForward]
    return S

def generateS_sparseTranspose(stim,sIdxs,numForward,numBack):
    S = np.zeros((numForward+numBack,len(sIdxs)))
    for ii in range(len(sIdxs)):
        S[:,ii] = stim[sIdxs[ii]-numBack : sIdxs[ii]+numForward]
    return S.T

def generateS_dense(stim,sIdxs,numForward,numBack):
    row = np.zeros(numForward+numBack)
    row[0] = stim[0]
    toep = toeplitz(stim,row)
    S_flipped = toep[sIdxs+numForward,:]
    return S_flipped

def laguerre_polynomial(n, x):
    L0 = np.ones_like(x)
    L1 = 1 - x
    if n == 0:
        return L0
    elif n == 1:
        return L1
    else:
        for k in range(2, n + 1):
            Lk = ((2 * k - 1 - x) * L1 - (k - 1) * L0) / k
            L0, L1 = L1, Lk
        return Lk

def apply_moving_average(kernel, window_size=75):
    return np.convolve(kernel, np.ones(window_size)/window_size, mode='same')

def calculate_voxel_timing_filter(stim_ts, stim, resp_ts, resp, num_stim_past, num_stim_future, method='ols', min_len=0):
    if method not in ['xcorr', 'ols', 'asd']:
        raise ValueError("Invalid method. Use 'xcorr', 'ols', or 'asd'.")
    
    stim = np.asarray(stim).flatten()
    stim_ts = np.asarray(stim_ts).flatten()
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

    indices = np.arange(1, len(stim_ts) + 1)
    X = np.vstack([indices, np.ones(len(stim_ts))]).T  # Design matrix
    b = np.linalg.lstsq(X, stim_ts, rcond=None)[0]     # Least squares fit
    stim_dt = b[0]  # Slope (sampling interval)
    time_offset = b[1]  # Intercept (offset)

    # Map response timestamps to stimulus indices
    stim_index = np.round((resp_ts - time_offset) / stim_dt).astype(int)
    # Filter for valid indices
    valid_index = (stim_index - num_stim_past > 0) & (stim_index + num_stim_future <= len(stim))
    stim_index = stim_index[valid_index]

    fastest_method = np.argmin(
        predict_S_time(num_kernel_el=(num_stim_past + num_stim_future + 1), num_resp=len(resp),num_stim=len(stim))) + 1 
    if fastest_method == 1:
        S = generateS_sparse(stim, stim_index, num_stim_future, num_stim_past)
    elif fastest_method == 2:
        S = generateS_sparseTranspose(stim, stim_index, num_stim_future, num_stim_past)
    elif fastest_method == 3:
        S = generateS_dense(stim, stim_index, num_stim_future, num_stim_past)

    r = resp[valid_index]

    # Add regularization term based on Laguerre polynomials
    lambda_reg = 1e-5
    laguerre_order = num_stim_past + num_stim_future  # Or adjust based on your needs
    laguerre_basis = np.array([laguerre_polynomial(n, np.arange(len(S))) for n in range(laguerre_order)]).T
    reg_term = lambda_reg * np.dot(laguerre_basis.T, laguerre_basis)

    if method == 'xcorr':
        f_hat = np.dot(r.T,S) / len(r)
    elif method == 'ols':
        #f_hat = np.linalg.lstsq(S, r, rcond=None)[0]    
        f_hat = np.linalg.lstsq(S.T @ S + reg_term, S.T @ r, rcond=None)[0]
    elif method == 'asd':
        raise NotImplementedError("ASD method requires additional implementation.")
    
    #smooth the filter
    f_hat= apply_moving_average(f_hat)

    """if fastest_method < 3: #flip to match standard convention (but already implemented for 3)
        f_hat = np.flipud(f_hat.ravel())"""

    f_hat_ts = stim_dt*np.arange(-(num_stim_past), num_stim_future)

    return f_hat, f_hat_ts, S




if __name__ == "__main__":

    """
    # Test 
    stim = np.array([0,1,1,1,1,1,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,0])
    resp = np.array([0,0,0,0,4,5,4,0,0,0,0,2,5,1,1,0,0,0,0,0])
    phase = np.sin((2 * np.pi / 5) * np.arange(30)) * np.pi

    stim_ts = np.arange(0, 30, 1)
    resp_ts = np.arange(0, 30, 3/2)"""

    breath_dict, breath_filt_dict, valve_dict, pulse_dict, calcium_dict, ca_interp_dict, instant_phase_dict, t_breath, _, _, animals = load_dataset()

    """stim = valve_dict['HW1']/100
    resp = calcium_dict['HW1'].mean(axis=1).to_numpy()

    stim_ts = np.arange(0, len(stim)/1000, 0.001) # sampled every 0.001 seconds (1000 Hz)
    resp_ts = np.arange(0, len(resp)/10, 0.1) # sampled every 0.1s (10 Hz)

    num_stim_past = 2000 # Include past stimulus samples
    num_stim_future = 0  # Assume no future stimulus effect

    # Estimate the filter
    f_hat, f_hat_ts, S = calculate_voxel_timing_filter(stim_ts, stim, resp_ts, resp, num_stim_past, num_stim_future, method='xcorr')

    plt.figure(figsize=(10, 3))
    plt.plot(f_hat_ts, f_hat)
    plt.legend()
    plt.show()

    predicted_response = np.zeros(len(stim))
    for t in range(num_stim_past, len(stim)):
        for lag in range(num_stim_past):
            predicted_response[t] += f_hat[lag] * stim[t-lag]
    
    with open("predicted_hw1_voxel.pkl","wb") as f:
        pickle.dump(predicted_response,f)

    with open("kernel_hw1_voxel.pkl","wb") as f:
        pickle.dump(f_hat,f)
    with open("kernel_timestamps_hw1_voxel.pkl","wb") as f:
        pickle.dump(f_hat_ts,f)"""

    num_stim_past = 2000 # Include past stimulus samples
    num_stim_future = 0  # Assume no future stimulus effect

    f_hats = []
    f_hats_ts = []
    predicted_responses = []

    for i, key in enumerate(calcium_dict.keys()):
        stim = valve_dict[key]/100
        resp = calcium_dict[key].mean(axis=1).to_numpy()

        stim_ts = np.arange(0, len(stim)/1000, 0.001) # sampled every 0.001 seconds (1000 Hz)
        resp_ts = np.arange(0, len(resp)/10, 0.1) # sampled every 0.1s (10 Hz)

        # Estimate the filter
        f_hat, f_hat_ts, S = calculate_voxel_timing_filter(stim_ts, stim, resp_ts, resp, num_stim_past, num_stim_future, method='ols')

        """stim_subset = stim[0:100000]
        predicted_response = np.zeros(len(stim_subset))
        for t in range(num_stim_past, len(stim_subset)):
            for lag in range(num_stim_past):
                predicted_response[t] += f_hat[lag] * stim_subset[t-lag]"""

        f_hats.append(f_hat)
        f_hats_ts.append(f_hats_ts)
        #predicted_responses.append(predicted_response)
        print(f'done for {key}')

    f_hats_dict = dict(zip(animals, f_hats))
    f_hats_ts_dict = dict(zip(animals, f_hats_ts))
    #predicted_resp_dict = dict(zip(animals, predicted_responses))

    with open("voxel_kernel_dict.pkl","wb") as f:
        pickle.dump(f_hats_dict,f)

    with open("voxel_kernel_timestamps_dict.pkl","wb") as f:
        pickle.dump(f_hats_ts_dict,f)

    """with open("predicted_voxel_dict.pkl","wb") as f:
        pickle.dump(predicted_resp_dict,f)"""

    for i, key in enumerate(f_hats_dict.keys()):
        plt.figure(figsize=(10, 3))
        t=np.arange(-(len(f_hats_dict[key])), 0)
        plt.plot(t, f_hats_dict[key])
        plt.legend()
        plt.title(f'Kernel for {key} using Voxel Timing and Reverse Correlation')
        plt.savefig(f'Kernel_{key}_Voxel_Timing.jpg')
        plt.show()

    """for i, key in enumerate(predicted_resp_dict.keys()):
        resp = calcium_dict[key].mean(axis=1).to_numpy()
        stim = valve_dict[key]

        stim_ts = np.arange(0, len(stim)/1000, 0.001) # sampled every 0.001 seconds (1000 Hz)
        resp_ts = np.arange(0, len(resp)/10, 0.1) # sampled every 0.1s (10 Hz)

        plt.figure(figsize=(10, 3))
        plt.plot(resp_ts[0:1000], resp[0:1000], label='Actual Calcium Response', color='blue', alpha=0.7)
        plt.plot(stim_ts[0:100000], predicted_resp_dict[key], label='Predicted Response', color='red', linestyle='--', alpha=0.7)
        plt.title(f'Normalized Actual vs. Predicted Calcium Response for {key} using Voxel timing and rev corr')
        plt.xlabel('Time (ms)')
        plt.ylabel('Signal')
        plt.legend()
        ing.jpg')
        plt.show()"""


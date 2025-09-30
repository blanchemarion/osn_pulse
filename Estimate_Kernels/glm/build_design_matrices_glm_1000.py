
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import pickle
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import scipy.stats
from sklearn.preprocessing import SplineTransformer, QuantileTransformer


def init_params():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--data-path",
        type=str,
        help="data path",
        default="sparseness/Data/animals_data_processed.pkl",
    )
    parser.add_argument(
        "--animal",
        type = str,
        help = "animal",
        default= "HW1"
    )
    args = parser.parse_args()
    params = vars(args)

    return params

def create_cosine_bumps(x, centers, widths):
  '''Create raised cosine bumps

  Input parameters::
  x: x positions to evaluate the cosine bumps on, ndarray of shape (n_samples, )
  centers: contains center positions of bumps, ndarray of shape (number of bumps, )
  widths: the width of each bump, should be same shape as centers

  Returns::
  bases: basis functions, ndarray of shape (n_samples, number of bumps)
  '''
  # Sanity check
  assert centers.shape == widths.shape, 'Centers and widths should have same number of elements'  
  x_reshape = x.reshape(-1,)

  # Create empty array for basis functions
  bases = np.full((x.shape[0], centers.shape[0]), np.nan)
  
  # Loop over center positions
  for idx, cent in enumerate(centers):
    bases[:, idx] = (np.cos(2 * np.pi * (x_reshape - cent) / widths[idx]) * 0.5 + 0.5) * \
                    (np.absolute(x_reshape - cent) < widths[idx] / 2)
  
  return bases

def mean_pulse_response(ca, onsets, pre, post):
    ca = np.asarray(ca.T)
    n_rois, T = ca.shape

    snippets = []
    
    #mean_iso = np.zeros((n_rois, pre + post), dtype=float)
    
    for roi in range(n_rois):
        roi_snips = []
        for onset in onsets:
            start, end = onset - pre, onset + post
            if start >= 0 and end <= T:
                roi_snips.append(ca[roi, start:end])
        if not roi_snips:
            raise ValueError(f"Pas assez de pulses isolées pour ROI {roi}!")
        roi_mean = np.mean(np.stack(roi_snips, axis=0), axis=0)
        snippets.append(roi_mean)
    
    """plt.figure(figsize=(6,4))
    plt.plot(np.arange(-pre, post) / 10, mean_iso)
    plt.show()"""
    return np.stack(snippets, axis=0)

def deconvolve_trace(ca, onsets, mean_iso, pre, post, eps=1e-6):
    """
    Renvoie:
        ca_deconv           : signal ΔF/F - template (longueur T)
        pulse_amplitudes    : amplitude (moyenne 0–1 s) de chaque pulse
    """     
    
    ca = np.asarray(ca.T)
        
    n_rois, T = ca.shape

    ca_deconv = ca.copy()
    
    for roi in range(n_rois):
        template = mean_iso[roi]
        for onset in onsets.astype(int):
            start, end = onset - pre, onset + post
            if 0 <= start < end <= T:
                ca_deconv[roi, start:end] -= template
                  
    mins = ca_deconv.min(axis=1)             # shape (n_rois,)
    offsets = np.where(mins < eps, -mins + eps, 0.0)
    ca_deconv += offsets[:, None]            # broadcast offset per ROI
    
        
    """time = np.arange(T) / 1000  # in seconds

    plt.figure(figsize=(12, 6))
    for roi in range(min(5, n_rois)):
        plt.subplot(5, 1, roi + 1)
        plt.plot(time, ca[roi], 'k', lw=0.5, label='raw')
        plt.plot(time, ca_deconv[roi], 'r', lw=0.75, label='deconv+shift')
        if roi == 0:
            plt.legend(loc='upper right')
        plt.ylabel(f'ROI {roi}')
    plt.xlabel('Time (s)')
    plt.suptitle('Raw vs. Deconvolved+Shifted Traces')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()"""
    
    """plt.figure(figsize=(10, 4))
    plt.plot(ca.T.mean(axis=1), label='ca', lw=1)
    plt.plot(ca_deconv.T.mean(axis=1), label='deconv', lw=1)
    plt.legend()
    plt.xlabel("Sample index")
    plt.ylabel("Signal")
    plt.tight_layout()
    plt.show()"""

    return ca.T
    #return ca_deconv.T       
    #return ca_deconv, amp


def make_raised_cosines(dt, span, n_bases):
    """
    dt      : bin width (s)           e.g. 0.1  (10 Hz imaging)
    span    : total window (s)        e.g. 2.0  or 0.5
    n_bases : number of bumps         e.g. 5
    Returns B with shape (T, n_bases) where T = span/dt
    """
    T = int(np.round(span / dt))
    t = np.arange(T) * dt

    centers = np.linspace(0, span, n_bases)
    width   = centers[1] - centers[0] + 1e-9          # avoid /0

    B = np.zeros((T, n_bases))
    for k, c in enumerate(centers):
        arg = (t - c) * np.pi / width                 # 0 at center
        mask = np.abs(arg) < np.pi
        B[mask, k] = 0.5 * (1 + np.cos(arg[mask]))    # raised cosine
    return B


def main():
    
    params = init_params()
        
    with open(params["data_path"], "rb") as f:
        data = pickle.load(f)
            
    animal= "HW1"
    
    calcium = data["ca_interp_dict"][animal]
    breath_filt = data["breath_filt_dict"][animal]
    phase = data["phase_peaks_dict"][animal] - np.pi
    valve = data["valve_dict"][animal]/100
    conv_stim = data["convolved_stim_dict"][animal]
    
    fs = 1000   
    pulse_len = 50         
    window_s  = 1.0     
    window_f = int(window_s * fs)

    onsets = np.flatnonzero(np.diff(np.r_[0, valve.astype(bool)]) == 1) # detetct pulse onsets
        
    ## Building the predicting variable
    pre = 0
    post = 1000
    mean_iso = mean_pulse_response(calcium, onsets, pre, post)
    ca_deconv=deconvolve_trace(calcium, onsets, mean_iso, pre, post)
        
    y = {'y': ca_deconv}
        
    ## Building the task variables

    prev_pulse_flag = np.zeros_like(valve, dtype=int) # 1 for the duration of the current pulse if there was a pulse in a preceding 1s window
    prev_pulse_strength = np.zeros_like(valve, dtype=float)

    for k in range(1, len(onsets)):
        if onsets[k] - onsets[k-1] <= window_f:  
            start = onsets[k]
            end   = min(start + pulse_len, len(valve))
            prev_pulse_flag[start:end] = 1
            prev_strength = conv_stim[ onsets[k-1] : onsets[k-1] + pulse_len ]
            slice_len = min(pulse_len, len(prev_strength), end-start)
            prev_pulse_strength[start : start+slice_len] = prev_strength[:slice_len]
    
    task_var = {
        'valve': valve - 0.5,
        'conv_stim': (conv_stim - np.mean(conv_stim)) / np.std(conv_stim, ddof=0),
        'prev_pulse_flag': prev_pulse_flag - 0.5,
        'prev_pulse_strength': (prev_pulse_strength - np.mean(prev_pulse_strength)) / np.std(prev_pulse_strength, ddof=0)
    }
    
    # Uncomment to plot the vectors of current pulse and previous pulse   
    """"t = np.arange(len(conv_stim)) / fs
    plt.figure(figsize=(9,3))
    plt.plot(t, conv_stim, drawstyle='steps-post', label='conv stim')
    plt.plot(t, prev_pulse_strength, drawstyle='steps-post',
            label='prev conv stim', color='tomato')
    plt.xlabel('Time (s)'); plt.legend(); plt.show()
    

    t = np.arange(len(valve)) / fs
    plt.figure(figsize=(9,3))
    plt.plot(t, valve, drawstyle='steps-post', label='valve')
    plt.plot(t, prev_pulse_flag*1.1, drawstyle='steps-post',
            label='prev‑pulse flag', color='tomato')
    plt.xlabel('Time (s)'); plt.ylim(-0.2, 1.3); plt.legend(); plt.show()"""
    
    
    ## Building the instantaneous variables
    
    # instantaneous sniff frequency
    sgn = np.signbit(breath_filt)  
    onset_mask = (sgn[1:] & ~sgn[:-1])
    onsets_sniff     = np.where(onset_mask)[0] + 1 

    times  = onsets_sniff / fs  
    isi    = np.diff(times)  # inter‑sniff intervals
    freq_cycle = 1.0 / isi 
    t_cycle    = (times[:-1] + times[1:]) / 2   # mid‑cycle timestamp

    freq_cycle = np.interp(t_cycle, t_cycle[~np.isnan(freq_cycle)], freq_cycle[~np.isnan(freq_cycle)])
    
    frame_t = np.arange(len(breath_filt)) / fs
    freq_step = np.interp(frame_t, t_cycle, freq_cycle, left=np.nan, right=np.nan)

    # smoother variant: low‑pass the step
    sniff_freq_vec = gaussian_filter1d(freq_step, sigma=fs*0.1)   # 100‑ms SD
    sniff_freq_vec[np.isnan(sniff_freq_vec)] = 5
    sniff_freq_vec = (sniff_freq_vec - np.mean(sniff_freq_vec)) / np.std(sniff_freq_vec, ddof=0)
    
    # sanity check plot
    """plt.figure(figsize=(9,3))
    plt.plot(frame_t, freq_step, drawstyle='steps-post', label='freq step')
    plt.plot(frame_t, sniff_freq_vec, drawstyle='steps-post', label='low pass freq step', color='tomato')
    plt.plot(frame_t, breath_filt, drawstyle='steps-post', label='breath filt')
    plt.xlabel('Time (s)'); plt.legend(); plt.show()"""
    
    #instantaneous valve frequency 
    win_s = 1.0                               # 1‑s causal window
    win_f = int(win_s * fs)
    pulse_train = np.zeros_like(valve, dtype=int)
    pulse_train[onsets] = 1

    kernel = np.ones(win_f)                   # flat window
    pulse_freq_vec = np.convolve(pulse_train, kernel, mode='full')[:len(valve)]
    pulse_freq_vec /= win_s                # pulses per second
    pulse_freq_vec = (pulse_freq_vec - np.mean(pulse_freq_vec)) / np.std(pulse_freq_vec, ddof=0)
    
    # Sanity check plot
    """t = np.arange(len(pulse_freq_vec)) / fs
    plt.figure(figsize=(9,3))
    plt.plot(t, pulse_freq_vec, drawstyle='steps-post', label='pulse freq')
    plt.plot(t, valve, drawstyle='steps-post', label='pulse train')
    plt.xlabel('Time (s)'); plt.legend(); plt.show()"""
    
    inst_var = {
        'sniff_freq_vec': sniff_freq_vec,
        'pulse_freq_vec': pulse_freq_vec,
    }
    
    # B splines expansion
    
    inst_bins = np.arange(0,1,0.02).reshape(-1,1)  # Note: the number of datapoints here is arbitrary; it just has to be of enough resolution for visualization
    spline = SplineTransformer(degree = 3, n_knots = 5, knots = 'uniform') # initialized sklearn SplineTransformer
    inst_bspl = spline.fit_transform(inst_bins) # create b-slines
    
    inst_var_names = list(inst_var.keys())
    
    # Initialize for expanded features and names
    expanded_features_inst = []
    expanded_feature_names_inst = [] 

    # Specify b-spline setting
    degree = 3
    n_konts = 5
    n_bsplines = degree + n_konts - 1

    # Loop over variables in movement variables and create b-spline expansion and expanded feature names
    for i, var in enumerate(inst_var_names):
        # quantile transform velocity
        this_var_quant = QuantileTransformer(n_quantiles=1000).fit_transform(inst_var[var].reshape(-1,1))
        # transform velocity quantiles into b-splines
        these_splines = SplineTransformer(degree=degree, n_knots=n_konts, knots='uniform').fit_transform(this_var_quant)
        # append features and feature names
        expanded_features_inst.append(these_splines)
        expanded_feature_names_inst.extend([f'{var}_bump{i}' for i in range(n_bsplines)])

    # Concatenate expanded features for all movement variables
    expanded_features_inst = np.hstack(expanded_features_inst)

    print('Shape of expanded inst features =', expanded_features_inst.shape, '\nNumber of expanded inst features =', len(expanded_feature_names_inst))
    
    """start_idx = expanded_feature_names_inst.index('sniff_freq_vec_bump0')
    end_idx = expanded_feature_names_inst.index('sniff_freq_vec_bump{}'.format(n_bsplines - 1))

    fig, axes = plt.subplots(2, 1, figsize = (8, 3))
    sniff_freq_quant = QuantileTransformer(n_quantiles=1000).fit_transform(inst_var['sniff_freq_vec'].reshape(-1,1))
    axes[0].plot(sniff_freq_quant[:100000],'k')
    axes[0].set(title = 'Sniff frequency (quantile transformed)')
    axes[1].plot(expanded_features_inst[:100000, start_idx:end_idx+1])
    axes[1].set(title = 'B-spline expanded sniff frequency')
    plt.tight_layout()
    plt.show()"""
    
    """start_idx = expanded_feature_names_inst.index('pulse_freq_vec_bump0')
    end_idx = expanded_feature_names_inst.index('pulse_freq_vec_bump{}'.format(n_bsplines - 1))

    fig, axes = plt.subplots(2, 1, figsize = (8, 3))
    sniff_freq_quant = QuantileTransformer(n_quantiles=1000).fit_transform(inst_var['pulse_freq_vec'].reshape(-1,1))
    axes[0].plot(sniff_freq_quant[:100000],'k')
    axes[0].set(title = 'Pulse frequency (quantile transformed)')
    axes[1].plot(expanded_features_inst[:100000, start_idx:end_idx+1])
    axes[1].set(title = 'B-spline expanded pulse frequency')
    plt.tight_layout()
    plt.show()"""
     
    
    ##Building the trial phase variables 
    
    # creating the trials (from a pulse to the next)
    T = len(valve)

    trial_index = np.zeros(T, dtype=int)
    for k in range(len(onsets) - 1):
        trial_index[onsets[k] : onsets[k+1]] = k
    trial_index[onsets[-1] :] = len(onsets) - 1    # tail after last pulse
        
    y['trial_index']=trial_index
    
    gap_run = np.zeros(T, dtype=int)
    elapsed = 0
    current_trial = 0

    for t in range(T):
        gap_run[t] = elapsed
        if t + 1 in onsets:          # next sample starts new trial
            elapsed = 0
            current_trial += 1
        else:
            elapsed += 1

    prev_gap_run = np.zeros(T, dtype=int)    # starts at gap, then ramps
    for k in range(1, len(onsets)):        
        start_cur = onsets[k]
        end_cur = onsets[k+1] if k < len(onsets)-1 else T

        gap_k = onsets[k] - onsets[k-1]      

        # running version: gap + 0,1,2,… within this trial
        length = end_cur - start_cur
        prev_gap_run[start_cur:end_cur] = gap_k + np.arange(length)
        

    """t = np.arange(len(valve)) / fs
    plt.figure(figsize=(9,3))
    plt.plot(t, gap_run, label='time since trial start')
    plt.plot(t, prev_gap_run, label='time since previous trial start')
    plt.vlines(onsets/fs, 0, pts_since_start.max(), color='k', alpha=.4, label='pulse onsets')
    plt.xlabel('Time (s)'); plt.legend(); plt.tight_layout(); plt.show()"""
    
    
    ## Temporal basis expansion for task variables in time since trial has started/ time since preceding trial has strated
    
    gap_run_s       = gap_run       / fs           # same length as gap_run
    prev_gap_run_s  = prev_gap_run  / fs
    
    n_tm_bases_gap_run = 20 
    n_tm_bases_prev_gap_run = 25

    tm_centers_gap_run      = np.linspace(0, gap_run_s.max(), n_tm_bases_gap_run)
    tm_centers_prev_gap_run = np.linspace(0, prev_gap_run_s.max(), n_tm_bases_prev_gap_run)
    
    spacing_gap  = np.diff(tm_centers_gap_run).mean()
    spacing_prev = np.diff(tm_centers_prev_gap_run).mean()

    tm_width_gap_run      = 4 * spacing_gap
    tm_width_prev_gap_run = 4 * spacing_prev

    timepoints_gap_run  = np.linspace(0, gap_run_s.max(), 300)
    tm_bases_gap_run    = create_cosine_bumps(timepoints_gap_run, tm_centers_gap_run, np.full_like(tm_centers_gap_run, tm_width_gap_run))

    timepoints_prev_gap = np.linspace(0, prev_gap_run_s.max(), 300)
    tm_bases_prev_gap   = create_cosine_bumps(timepoints_prev_gap, tm_centers_prev_gap_run, np.full_like(tm_centers_prev_gap_run, tm_width_prev_gap_run))

    trial_tm_bases_gap_run = create_cosine_bumps(gap_run_s, tm_centers_gap_run, np.full_like(tm_centers_gap_run, tm_width_gap_run))
    trial_tm_names_gap_run = [f'gap_run_bump{i}' for i in range(len(tm_centers_gap_run))]
    
    trial_tm_bases_prev_gap_run = create_cosine_bumps(prev_gap_run_s, tm_centers_prev_gap_run, np.full_like(tm_centers_prev_gap_run, tm_width_prev_gap_run))
    trial_tm_names_prev_gap_run = [f'prev_gap_run_bump{i}' for i in range(len(tm_centers_prev_gap_run))]


    """fig, axes = plt.subplots(2, 1, figsize = (8, 3))
    axes[0].plot(gap_run, 'k')
    axes[0].set(title='Trial duration')
    axes[1].plot(cho_tm_bases_gap_run)
    axes[1].set(title='Expanded time')
    plt.title('gap run')
    plt.show()
    
    fig, axes = plt.subplots(2, 1, figsize = (8, 3))
    axes[0].plot(prev_gap_run, 'k')
    axes[0].set(title='Trial duration')
    axes[1].plot(cho_tm_bases_prev_gap_run)
    axes[1].set(title='Expanded time')
    plt.title('prev gap run')
    plt.show()"""
        
    ## Apply temporal bases to task variables
    var_names = list(task_var.keys())

    # Initialize features and names with expanded timepoints
    expanded_features_gap_run_tm = np.full((trial_tm_bases_gap_run.shape[0], n_tm_bases_gap_run * (len(var_names) + 1)), np.nan)
    expanded_features_gap_run_tm[:, :trial_tm_bases_gap_run.shape[1]] = trial_tm_bases_gap_run.copy()
    expanded_feature_names_gap_run_tm = [f'trialPhase_{base_name}' for base_name in trial_tm_names_gap_run] 
    
    expanded_features_prev_gap_run_tm = np.full((trial_tm_bases_prev_gap_run.shape[0], n_tm_bases_prev_gap_run * (len(var_names) + 1)), np.nan)
    expanded_features_prev_gap_run_tm[:, :trial_tm_bases_prev_gap_run.shape[1]] = trial_tm_bases_prev_gap_run.copy()
    expanded_feature_names_prev_gap_run_tm = [f'trialPhase_{base_name}' for base_name in trial_tm_names_prev_gap_run] 


    # Multiply individual variables with expanded position predictors
    for i, name in enumerate(var_names):  
        expanded_features_gap_run_tm[:, (i + 1) * n_tm_bases_gap_run:(i + 2) * n_tm_bases_gap_run] = task_var[name][:, None] * trial_tm_bases_gap_run
        expanded_feature_names_gap_run_tm += [f'{name}_{base_name}' for base_name in trial_tm_names_gap_run]
        
        expanded_features_prev_gap_run_tm[:, (i + 1) * n_tm_bases_prev_gap_run:(i + 2) * n_tm_bases_prev_gap_run] = task_var[name][:, None] * trial_tm_bases_prev_gap_run
        expanded_feature_names_prev_gap_run_tm += [f'{name}_{base_name}' for base_name in trial_tm_names_prev_gap_run]
        
    print('Shape of temporally gap run expanded features =', expanded_features_gap_run_tm.shape, '\nNumber of temporally gap run expanded features =', len(expanded_feature_names_gap_run_tm))
    print('Shape of temporally prev gap run expanded features =', expanded_features_prev_gap_run_tm.shape, '\nNumber of temporally gap run expanded features =', len(expanded_feature_names_prev_gap_run_tm))


    ## Temporal basis expansion for task variables in phase

    ## Create position basis functions
    start_phase = -np.pi
    end_phase = np.pi
    n_phase_bases = 4

    phase_centers = np.linspace(start_phase, end_phase, n_phase_bases)

    width_to_spacing_ratio = 4
    pos_width = width_to_spacing_ratio * scipy.stats.mode(np.diff(phase_centers))[0]

    phase_bases = create_cosine_bumps(phase, phase_centers, pos_width * np.ones_like(phase_centers))
    phase_names = [f'phase_bump{i}' for i in range(len(phase_centers))] # create a list of names for each expanded feature

    """fig, axes = plt.subplots(2, 1, figsize = (8, 3))
    axes[0].plot(phase, 'k')
    axes[0].set(title='sniff phase')
    axes[1].plot(phase_bases)
    axes[1].set(title='Expanded phase')
    plt.show()"""
    
    ## Apply forward position bases to task variables
    var_names = list(task_var.keys())

    # Initialize features and names with expanded position bases
    expanded_features_phase = np.full((phase_bases.shape[0], n_phase_bases * (len(var_names) + 1)), np.nan)
    expanded_features_phase[:, :phase_bases.shape[1]] = phase_bases.copy()
    expanded_feature_names_phase = [f'trialPhase_{base_name}' for base_name in phase_names] # attach 'trialPhase' before 'fPos_bump#'

    # Multiply individual variables with expanded position predictors
    for i, name in enumerate(var_names):  
        expanded_features_phase[:, (i + 1) * n_phase_bases:(i + 2) * n_phase_bases] = task_var[name][:, None] * phase_bases
        expanded_feature_names_phase += [f'{name}_{base_name}' for base_name in phase_names]
        
    print('Shape of position expanded features =', expanded_features_phase.shape, '\nNumber of position expanded features =', len(expanded_feature_names_phase))
    
    
    ## Group all features
    
    all_features = np.concatenate((expanded_features_phase, expanded_features_gap_run_tm, expanded_features_prev_gap_run_tm, expanded_features_inst), axis=1)
    all_feature_names = expanded_feature_names_phase.copy()
    all_feature_names.extend(expanded_feature_names_gap_run_tm)
    all_feature_names.extend(expanded_feature_names_prev_gap_run_tm)
    all_feature_names.extend(expanded_feature_names_inst)
    
    print('Shape of all features combined =', all_features.shape, '\nNumber of all expanded features =', len(all_feature_names))
    
    """plt.plot(group_ind)
    plt.yticks(np.arange(0,len(group_size),5))
    plt.xlabel('Feature number')
    plt.ylabel('Group index')
    plt.show()"""
    
    with open("sparseness/Data/all_features_glm.pkl", "wb") as f:
        pickle.dump(all_features, f)

    with open("sparseness/Data/all_feature_names_glm.pkl", "wb") as f:
        pickle.dump(all_feature_names, f)
        
    with open("sparseness/Data/ca_deconv.pkl", "wb") as f:
        pickle.dump(y, f)
    
if __name__ == "__main__":
    main()

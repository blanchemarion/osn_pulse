"""
Downsampled (10 Hz) feature engineering with the SAME organization as the 1000hz script:
- Phase basis × task variables
- Trial-time bases (time since trial start / since previous trial start) × task variables
- Instantaneous variables (sniff & pulse frequency) expanded with B-splines

"""
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import argparse
import scipy.stats
from scipy.signal import butter, sosfiltfilt
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import SplineTransformer, QuantileTransformer
import re


# ---------------------------
# Utilities from your 10 Hz pipeline
# ---------------------------

def base_name(nm: str) -> str:
    # strip trailing _bump\d+, -bump\d+, _lag\d+ patterns, and any leftover digits
    b = re.sub(r'(_|-)?bump\d+$', '', nm)
    b = re.sub(r'_lag\d+$', '', b)
    b = re.sub(r'\d+$', '', b)
    return b

def build_group_map(names):
    groups = {}
    for i, nm in enumerate(names):
        g = base_name(nm)
        groups.setdefault(g, []).append(i)
    group_names = list(groups.keys())
    group_sizes = [len(groups[g]) for g in group_names]
    # flat index vector of length n_features
    group_ind = np.empty(sum(group_sizes), dtype=int)
    ptr = 0
    for gi, g in enumerate(group_names):
        sz = len(groups[g])
        group_ind[ptr:ptr+sz] = gi
        ptr += sz
    return group_sizes, group_names, group_ind, groups

def lag_no_cross(vec, L, trial_index):
    """Shift vec by L; zero out (a) the first L frames and (b) locations where
    trial_index[t] != trial_index[t-L] to avoid crossing trial boundaries."""
    z = np.roll(vec, L)
    z[:L] = 0
    cross = np.zeros_like(vec, dtype=bool)
    cross[L:] = (trial_index[L:] != trial_index[:-L])
    z[cross] = 0
    return z

def make_lags_matrix(X_df, trial_index, cols, lags=(1, 2)):
    mats, names = [], []
    for c in cols:
        v = X_df[c].values.astype(float)
        mats.append(v)
        names.append(c)
        for L in lags:
            mats.append(lag_no_cross(v, L, trial_index))
            names.append(f"{c}_lag{L}")
    return np.column_stack(mats).astype(np.float32), names


def calcium_time_grid(n_ca, fs_ca):
    edges = np.arange(n_ca + 1, dtype=float) / fs_ca      # [s]
    centers = (edges[:-1] + edges[1:]) / 2.0
    return edges, centers

def rising_edges(x):
    x = (x > 0).astype(np.uint8)
    return np.flatnonzero((x[1:] > x[:-1])) + 1

def bin_fraction(binary_1khz, fs, edges_s):
    dt = 1.0 / fs
    csum = np.concatenate(([0.0], np.cumsum(binary_1khz.astype(float) * dt)))
    idx = np.clip(np.floor(edges_s * fs).astype(int), 0, len(binary_1khz))
    area = csum[idx[1:]] - csum[idx[:-1]]
    width = np.diff(edges_s)
    return area / width

def bin_time_average(x, fs, edges_s):
    x = np.asarray(x, dtype=float)
    idx = np.clip(np.floor(edges_s * fs).astype(int), 0, len(x))
    csum = np.concatenate(([0.0], np.cumsum(x)))
    sums = csum[idx[1:]] - csum[idx[:-1]]
    counts = (idx[1:] - idx[:-1]).clip(min=1)
    return sums / counts

def bin_sum(x, fs, edges_s):
    x = np.asarray(x, dtype=float)
    idx = np.clip(np.floor(edges_s * fs).astype(int), 0, len(x))
    csum = np.concatenate(([0.0], np.cumsum(x)))
    return csum[idx[1:]] - csum[idx[:-1]]

def circular_mean_in_bins(angles, fs, edges_s, weights=None):
    if weights is None:
        weights = np.ones_like(angles, dtype=float)
    s = np.sin(angles) * weights
    c = np.cos(angles) * weights
    idx = np.clip(np.floor(edges_s * fs).astype(int), 0, len(angles))
    s_cum = np.concatenate(([0.0], np.cumsum(s)))
    c_cum = np.concatenate(([0.0], np.cumsum(c)))
    w_cum = np.concatenate(([0.0], np.cumsum(weights)))
    s_sum = s_cum[idx[1:]] - s_cum[idx[:-1]]
    c_sum = c_cum[idx[1:]] - c_cum[idx[:-1]]
    w_sum = (w_cum[idx[1:]] - w_cum[idx[:-1]]).clip(min=1e-12)
    s_mean = s_sum / w_sum
    c_mean = c_sum / w_sum
    ang = np.arctan2(s_sum, c_sum)
    R = np.sqrt(s_sum**2 + c_sum**2) / w_sum
    return ang, s_mean, c_mean, R

def breath_raw_to_ca(breath_1k, fs_fast, ca_edges_s, cutoff_hz=4.0):
    sos = butter(5, cutoff_hz, btype='lowpass', fs=fs_fast, output='sos')
    breath_lp = sosfiltfilt(sos, breath_1k)
    idx = np.clip(np.floor(ca_edges_s * fs_fast).astype(int), 0, len(breath_lp))
    cum = np.concatenate(([0.0], np.cumsum(breath_lp)))
    sums = cum[idx[1:]] - cum[idx[:-1]]
    counts = (idx[1:] - idx[:-1]).clip(min=1)
    return sums / counts

def inh_exh_fraction_per_bin(phase_rad_1k, fs_fast, ca_edges_s):
    inh_mask = (phase_rad_1k >= 0) & (phase_rad_1k < np.pi)
    exh_mask = ~inh_mask
    inh_frac = bin_fraction(inh_mask.astype(np.uint8), fs_fast, ca_edges_s)
    exh_frac = bin_fraction(exh_mask.astype(np.uint8), fs_fast, ca_edges_s)
    return inh_frac, exh_frac

def double_exp_kernel(fs, tau_r=0.2, tau_d=1.0, T=10.0):
    t = np.arange(0, int(T*fs)) / fs
    k = (np.exp(-t / tau_d) - np.exp(-t / tau_r))
    k[k < 0] = 0.0
    area = k.sum()
    if area > 0:
        k = k / area
    return k

def convolved_valve_to_ca(valve_1k, fs_valve, ca_edges_s, tau_r=0.2, tau_d=1.0):
    k = double_exp_kernel(fs_valve, tau_r=tau_r, tau_d=tau_d)
    conv_1k = np.convolve(valve_1k.astype(float), k, mode='full')[:len(valve_1k)]
    idx = np.clip(np.floor(ca_edges_s * fs_valve).astype(int), 0, len(conv_1k))
    conv_cum = np.concatenate(([0.0], np.cumsum(conv_1k)))
    conv_sum = conv_cum[idx[1:]] - conv_cum[idx[:-1]]
    counts = (idx[1:] - idx[:-1]).clip(min=1)
    return conv_sum / counts

# Raised-cosine bases 
def create_cosine_bumps(x, centers, widths):
    assert centers.shape == widths.shape, 'Centers and widths should have same number of elements'
    x_reshape = x.reshape(-1,)
    bases = np.full((x.shape[0], centers.shape[0]), np.nan)
    for idx, cent in enumerate(centers):
        bases[:, idx] = (np.cos(2*np.pi*(x_reshape - cent)/widths[idx]) * 0.5 + 0.5) * \
                        (np.abs(x_reshape - cent) < widths[idx]/2)
    return bases

####################
# Valve features @10 Hz 
class ValveFeatures:
    def __init__(self, any_frac, mult_mean, mult_inh, mult_exh, counts, onset_phase_sin, onset_phase_cos):
        self.any_frac = any_frac
        self.mult_mean = mult_mean
        self.mult_inh = mult_inh
        self.mult_exh = mult_exh
        self.counts = counts
        self.onset_phase_sin = onset_phase_sin
        self.onset_phase_cos = onset_phase_cos

def downsample_valve_features(valve_nary_1k, fs_valve, phase_rad_1k, fs_phase, ca_edges_s):
    assert fs_valve == fs_phase, "Assuming valve and phase share the same time base (1 kHz)."
    any_frac = bin_fraction(valve_nary_1k > 0, fs_valve, ca_edges_s)
    mult_mean = bin_time_average(valve_nary_1k, fs_valve, ca_edges_s)
    inh_mask = (phase_rad_1k >= 0) & (phase_rad_1k < np.pi)
    exh_mask = ~inh_mask
    mult_inh = bin_time_average(valve_nary_1k * inh_mask.astype(int), fs_valve, ca_edges_s)
    mult_exh = bin_time_average(valve_nary_1k * exh_mask.astype(int), fs_valve, ca_edges_s)
    dx = np.diff(valve_nary_1k.astype(int), prepend=0)
    dx_pos = np.maximum(dx, 0)
    counts = bin_sum(dx_pos, fs_valve, ca_edges_s)
    s_sum = bin_sum(np.sin(phase_rad_1k) * dx_pos, fs_valve, ca_edges_s)
    c_sum = bin_sum(np.cos(phase_rad_1k) * dx_pos, fs_valve, ca_edges_s)
    with np.errstate(invalid='ignore', divide='ignore'):
        onset_phase_sin = np.divide(s_sum, counts, out=np.zeros_like(s_sum), where=counts > 0)
        onset_phase_cos = np.divide(c_sum, counts, out=np.zeros_like(c_sum), where=counts > 0)
    return ValveFeatures(any_frac, mult_mean, mult_inh, mult_exh, counts, onset_phase_sin, onset_phase_cos)

##################3
# Phase features @10 Hz (sin, cos, R)
class PhaseFeatures:
    def __init__(self, sin_phase, cos_phase, R):
        self.sin_phase = sin_phase
        self.cos_phase = cos_phase
        self.R = R

def downsample_phase_features(phase_rad_1k, fs, ca_edges_s):
    _, s_mean, c_mean, R = circular_mean_in_bins(phase_rad_1k, fs, ca_edges_s)
    return PhaseFeatures(s_mean, c_mean, R)

# ---------------------------
# Build 10 Hz “base” signals (X dataframe)
# ---------------------------
def build_10hz_features(valve_nary_1k, breath_1k, phase_1k, calcium_10hz, fs_fast=1000, fs_ca=10):
    n_ca = len(calcium_10hz)
    ca_edges, ca_centers = calcium_time_grid(n_ca, fs_ca)
    phase_feats = downsample_phase_features(phase_1k, fs_fast, ca_edges)
    valve_feats = downsample_valve_features(valve_nary_1k, fs_fast, phase_1k, fs_fast, ca_edges)
    valve_conv = convolved_valve_to_ca(valve_nary_1k, fs_fast, ca_edges, tau_r=0.2, tau_d=1.0)
    breath_ca = breath_raw_to_ca(breath_1k, fs_fast, ca_edges, cutoff_hz=4.0)
    inh_frac, exh_frac = inh_exh_fraction_per_bin(phase_1k, fs_fast, ca_edges)

    X = pd.DataFrame({
        "valve_any_frac": valve_feats.any_frac,
        "valve_mult_mean": valve_feats.mult_mean,
        "valve_mult_inh": valve_feats.mult_inh,
        "valve_mult_exh": valve_feats.mult_exh,
        "valve_counts": valve_feats.counts,
        "onset_phase_sin": valve_feats.onset_phase_sin,
        "onset_phase_cos": valve_feats.onset_phase_cos,
        "sin_phase": phase_feats.sin_phase,
        "cos_phase": phase_feats.cos_phase,
        "R": phase_feats.R,
        "breath_ca": breath_ca,
        "inh_frac": inh_frac,
        "exh_frac": exh_frac,
        "valve_convolved": valve_conv
    }, index=np.arange(n_ca))
    return X, ca_edges, ca_centers

# make trial indices on the 10 Hz grid from 1 kHz onsets
def onsets_1k_to_ca_bins(valve_nary_1k, fs_fast, ca_edges_s):
    # Binary "trial" onsets = transitions 0->1
    on_1k = rising_edges(valve_nary_1k > 0)
    on_t = on_1k / fs_fast
    on_bins = np.digitize(on_t, ca_edges_s) - 1
    # keep only onsets that land inside [0, n_ca-1]
    nbins = len(ca_edges_s) - 1
    on_bins = on_bins[(on_bins >= 0) & (on_bins < nbins)]
    return on_bins

def make_trial_vectors_on_ca(on_bins, n_ca):
    trial_index = np.zeros(n_ca, dtype=int)
    if on_bins.size == 0:
        return trial_index, np.zeros_like(trial_index), np.zeros_like(trial_index)
    for k in range(len(on_bins) - 1):
        trial_index[on_bins[k]:on_bins[k+1]] = k
    trial_index[on_bins[-1]:] = len(on_bins) - 1

    # gap_run: frames since current trial start
    gap_run = np.zeros(n_ca, dtype=int)
    current = 0
    for t in range(n_ca):
        gap_run[t] = t - on_bins[current]
        if (current + 1 < len(on_bins)) and (t + 1 == on_bins[current + 1]):
            current += 1

    # prev_gap_run: gap length before this trial + elapsed within this trial
    prev_gap_run = np.zeros(n_ca, dtype=int)
    for k in range(1, len(on_bins)):
        start = on_bins[k]
        end = on_bins[k+1] if k < len(on_bins)-1 else n_ca
        gap_k = on_bins[k] - on_bins[k-1]
        length = end - start
        prev_gap_run[start:end] = gap_k + np.arange(length)
    return trial_index, gap_run, prev_gap_run

# Raised-cosine bases for phase & trial-time 
def make_phase_bases_from_X(X, n_phase_bases=4):
    # Dominant phase angle from downsampled sin/cos
    phase = np.arctan2(X["sin_phase"].values, X["cos_phase"].values)  # [-π, π]
    start_phase, end_phase = -np.pi, np.pi
    centers = np.linspace(start_phase, end_phase, n_phase_bases)
    spacing = np.diff(centers).mean()
    widths = np.full_like(centers, 4 * spacing)
    phase_bases = create_cosine_bumps(phase, centers, widths)
    phase_names = [f'phase_bump{i}' for i in range(len(centers))]
    return phase_bases, phase_names

def make_trial_time_bases(gap_run, prev_gap_run, fs_ca, n_tm_bases_gap_run=20, n_tm_bases_prev_gap_run=25):
    gap_run_s = gap_run / fs_ca
    prev_gap_run_s = prev_gap_run / fs_ca
    
    print(gap_run_s)
    print(prev_gap_run_s)

    # centers & widths (uniform)
    centers_gap = np.linspace(0, gap_run_s.max(), n_tm_bases_gap_run)
    centers_prev = np.linspace(0, prev_gap_run_s.max(), n_tm_bases_prev_gap_run)
    spacing_gap = np.diff(centers_gap).mean() if n_tm_bases_gap_run > 1 else max(1e-3, gap_run_s.max())
    spacing_prev = np.diff(centers_prev).mean() if n_tm_bases_prev_gap_run > 1 else max(1e-3, prev_gap_run_s.max())
    width_gap = 4 * spacing_gap
    width_prev = 4 * spacing_prev

    bases_gap = create_cosine_bumps(gap_run_s, centers_gap, np.full_like(centers_gap, width_gap))
    bases_prev = create_cosine_bumps(prev_gap_run_s, centers_prev, np.full_like(centers_prev, width_prev))

    names_gap = [f'gap_run_bump{i}' for i in range(len(centers_gap))]
    names_prev = [f'prev_gap_run_bump{i}' for i in range(len(centers_prev))]
    return bases_gap, names_gap, bases_prev, names_prev

# Instantaneous vars @10 Hz 
def sniff_freq_10hz(breath_filt_1k, fs_fast, ca_edges_s, sigma_s=0.1, default_hz=5.0):
    sgn = np.signbit(breath_filt_1k)
    onset_mask = (sgn[1:] & ~sgn[:-1])
    onsets_sniff = np.where(onset_mask)[0] + 1
    times = onsets_sniff / fs_fast
    isi = np.diff(times)
    freq_cycle = np.zeros_like(isi)
    with np.errstate(divide='ignore', invalid='ignore'):
        freq_cycle = 1.0 / isi
    t_cycle = (times[:-1] + times[1:]) / 2.0
    # interpolate to 1 kHz grid, then smooth, then bin-average
    frame_t = np.arange(len(breath_filt_1k)) / fs_fast
    freq_step = np.interp(frame_t, t_cycle, freq_cycle, left=np.nan, right=np.nan)
    freq_step = np.where(np.isnan(freq_step), default_hz, freq_step)
    freq_sm = gaussian_filter1d(freq_step, sigma=int(round(sigma_s * fs_fast)))
    return bin_time_average(freq_sm, fs_fast, ca_edges_s)

def pulse_freq_10hz(valve_nary_1k, fs_fast, ca_edges_s, win_s=1.0):
    on_1k = rising_edges(valve_nary_1k > 0)  # 0->1 pulses
    pulse_train = np.zeros_like(valve_nary_1k, dtype=int)
    pulse_train[on_1k] = 1
    win_f = int(win_s * fs_fast)
    kernel = np.ones(win_f)
    freq_1k = np.convolve(pulse_train, kernel, mode='full')[:len(valve_nary_1k)] / win_s
    return bin_time_average(freq_1k, fs_fast, ca_edges_s)

# Main: build GLM-ready matrix 
def init_params():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-path", type=str, default="sparseness/Data/animals_data_processed.pkl")
    p.add_argument("--animal", type=str, default="HW1")
    p.add_argument("--out-prefix", type=str, default="sparseness/Data/")
    return vars(p.parse_args())

def main():
    params = init_params()
    with open(params["data_path"], "rb") as f:
        data = pickle.load(f)

    animal = params["animal"]
    fs_fast, fs_ca = 1000, 10

    # ------------- Load signals -------------
    valve_nary_1k = (data["valve_dict"][animal] / 100).astype(int)   # 0/1/2/...
    breath_1k = np.asarray(data["breath_filt_dict"][animal], dtype=float)
    phase_1k = np.asarray(data["phase_peaks_dict"][animal], dtype=float)  # [0, 2π)
    ca = np.asarray(data["calcium_dict"][animal])                     # (T_ca, nROI) or (T_ca,)
    calcium_10hz = ca[:, 0] if ca.ndim == 2 else ca
    T_ca = len(calcium_10hz)

    # ------------- Base 10 Hz features -------------
    X, ca_edges, ca_centers = build_10hz_features(
        valve_nary_1k, breath_1k, phase_1k, calcium_10hz, fs_fast=fs_fast, fs_ca=fs_ca
    )
    
    print(X.columns)

    # ------------- Trial structure on 10 Hz grid -------------
    onset_bins = onsets_1k_to_ca_bins(valve_nary_1k, fs_fast, ca_edges)
    trial_index, gap_run, prev_gap_run = make_trial_vectors_on_ca(onset_bins, T_ca)
    
    # ---- Simple first-order lags (no interactions) ----
    lag_cols = ["valve_any_frac", "valve_convolved", "breath_ca", "sin_phase", "cos_phase"]
    lag_features, lag_names = make_lags_matrix(X, trial_index, lag_cols, lags=(1, 2, 3, 4))

    # ------------- Task variables at 10 Hz (match names/centering) -------------
    # 'valve' -> use saturating occupancy in [0,1] then center by 0.5
    valve_centered = X["valve_any_frac"].values - 0.5
    # 'conv_stim' -> use valve_convolved (z-score)
    conv = X["valve_convolved"].values
    conv_z = (conv - conv.mean()) / (conv.std(ddof=0) + 1e-12)
    # Previous pulse flag/strength on CURRENT pulse bin if previous onset within 1s
    prev_flag = np.zeros(T_ca, dtype=int)
    prev_strength = np.zeros(T_ca, dtype=float)
    window_bins = int(1.0 * fs_ca)
    for k in range(1, len(onset_bins)):
        if onset_bins[k] - onset_bins[k-1] <= window_bins:
            b = onset_bins[k]
            prev_flag[b] = 1
            prev_strength[b] = conv_z[onset_bins[k-1]]
    prev_flag_centered = prev_flag - 0.5
    prev_strength_z = (prev_strength - prev_strength.mean()) / (prev_strength.std(ddof=0) + 1e-12)

    task_var = {
        "valve": valve_centered,
        "conv_stim": conv_z,
        "prev_pulse_flag": prev_flag_centered,
        "prev_pulse_strength": prev_strength_z,
    }
    task_names = list(task_var.keys())

    # ------------- Instantaneous variables at 10 Hz -------------
    sniff_freq_ca = sniff_freq_10hz(breath_1k, fs_fast, ca_edges, sigma_s=0.1, default_hz=5.0)
    sniff_freq_ca = (sniff_freq_ca - np.mean(sniff_freq_ca)) / (np.std(sniff_freq_ca, ddof=0) + 1e-12)

    pulse_freq_ca = pulse_freq_10hz(valve_nary_1k, fs_fast, ca_edges, win_s=1.0)
    pulse_freq_ca = (pulse_freq_ca - np.mean(pulse_freq_ca)) / (np.std(pulse_freq_ca, ddof=0) + 1e-12)

    inst_var = {
        "sniff_freq_vec": sniff_freq_ca,
        "pulse_freq_vec": pulse_freq_ca
    }

    # ------------- B-spline expansion of instantaneous vars (same as above script) -------------
    degree = 3
    n_knots = 5
    n_bsplines = degree + n_knots - 1
    expanded_features_inst = []
    expanded_feature_names_inst = []
    for varname, vec in inst_var.items():
        vec_q = QuantileTransformer(n_quantiles=1000, output_distribution='uniform').fit_transform(vec.reshape(-1, 1))
        spl = SplineTransformer(degree=degree, n_knots=n_knots, knots='uniform')
        bumps = spl.fit_transform(vec_q)
        expanded_features_inst.append(bumps)
        expanded_feature_names_inst.extend([f'{varname}_bump{i}' for i in range(n_bsplines)])
    expanded_features_inst = np.hstack(expanded_features_inst)

    # ------------- Phase basis (from 10 Hz sin/cos) -------------
    phase_bases, phase_names = make_phase_bases_from_X(X, n_phase_bases=4)
    # Compose [trialPhase_*] first (like your script)
    expanded_features_phase = np.full((T_ca, phase_bases.shape[1] * (len(task_names) + 1)), np.nan)
    expanded_features_phase[:, :phase_bases.shape[1]] = phase_bases.copy()
    expanded_feature_names_phase = [f"trialPhase_{nm}" for nm in phase_names]
    # task × phase
    for i, name in enumerate(task_names):
        cols = slice((i+1)*phase_bases.shape[1], (i+2)*phase_bases.shape[1])
        expanded_features_phase[:, cols] = task_var[name][:, None] * phase_bases
        expanded_feature_names_phase += [f"{name}_{nm}" for nm in phase_names]

    # ------------- Trial-time bases (gap_run / prev_gap_run) -------------
    bases_gap, names_gap, bases_prev, names_prev = make_trial_time_bases(
        gap_run, prev_gap_run, fs_ca,
        n_tm_bases_gap_run=10, n_tm_bases_prev_gap_run=10
    )

    # time-since-trial-start
    expanded_features_gap_run_tm = np.full((T_ca, bases_gap.shape[1] * (len(task_names) + 1)), np.nan)
    expanded_features_gap_run_tm[:, :bases_gap.shape[1]] = bases_gap.copy()
    expanded_feature_names_gap_run_tm = [f"trialPhase_{nm}" for nm in names_gap]
    for i, name in enumerate(task_names):
        cols = slice((i+1)*bases_gap.shape[1], (i+2)*bases_gap.shape[1])
        expanded_features_gap_run_tm[:, cols] = task_var[name][:, None] * bases_gap
        expanded_feature_names_gap_run_tm += [f"{name}_{nm}" for nm in names_gap]

    # time-since-previous-trial-start + gap
    expanded_features_prev_gap_run_tm = np.full((T_ca, bases_prev.shape[1] * (len(task_names) + 1)), np.nan)
    expanded_features_prev_gap_run_tm[:, :bases_prev.shape[1]] = bases_prev.copy()
    expanded_feature_names_prev_gap_run_tm = [f"trialPhase_{nm}" for nm in names_prev]
    for i, name in enumerate(task_names):
        cols = slice((i+1)*bases_prev.shape[1], (i+2)*bases_prev.shape[1])
        expanded_features_prev_gap_run_tm[:, cols] = task_var[name][:, None] * bases_prev
        expanded_feature_names_prev_gap_run_tm += [f"{name}_{nm}" for nm in names_prev]

    # ------------- Group all features (same ordering as your 1 kHz script) -------------
    
    
    all_features = np.concatenate((
        expanded_features_phase,
        expanded_features_gap_run_tm,
        expanded_features_prev_gap_run_tm,
        expanded_features_inst,
        lag_features,                    
    ), axis=1)

    # names
    all_feature_names = []
    all_feature_names.extend(expanded_feature_names_phase)
    all_feature_names.extend(expanded_feature_names_gap_run_tm)
    all_feature_names.extend(expanded_feature_names_prev_gap_run_tm)
    all_feature_names.extend(expanded_feature_names_inst)
    all_feature_names.extend(lag_names)    
    
    
    
    group_size, group_name, group_ind, group_dict = build_group_map(all_feature_names)

    
    # ------------- Outputs -------------
    out_prefix = Path(params["out_prefix"])
    out_prefix.mkdir(parents=True, exist_ok=True)

    with open(out_prefix / "all_features_glm_10hz.pkl", "wb") as f:
        pickle.dump(all_features, f)

    with open(out_prefix / "all_feature_names_glm_10hz.pkl", "wb") as f:
        pickle.dump(all_feature_names, f)

    y = {
        "y": ca if ca.ndim == 2 else ca[:, None],   # ensure 2D: (T_ca, nROI)
        "trial_index": trial_index
    }
    with open(out_prefix / "y_ca_10hz.pkl", "wb") as f:
        pickle.dump(y, f)
        
    with open(out_prefix / "groups_glm_10hz.pkl", "wb") as f:
        pickle.dump({
            "group_size": group_size,
            "group_name": group_name,
            "group_ind": group_ind,
            "groups": group_dict
        }, f)

    print("all_features shape:", all_features.shape)
    print("#features:", len(all_feature_names))
    print("first 10 feature names:", all_feature_names[:10])

if __name__ == "__main__":
    main()

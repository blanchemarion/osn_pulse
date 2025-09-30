"""
Analyze potential coupling between odor valve pulse rate and sniffing frequency

"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle
from scipy.stats import wilcoxon, mode
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf
import seaborn as sns
from scipy.stats import pearsonr


def init_params():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--data-path",
        type=str,
        help="data path",
        default="sparseness/Data/animals_data_processed.pkl",
    )
    parser.add_argument(
        "--fs-ca",
        type=int,
        help="sampling freq",
        default=10,
    )
    args = parser.parse_args()
    params = vars(args)

    return params


def make_raised_cosines(dt, span, n_bases):
    """
    dt      : bin width (s)           e.g. 0.001  (1000 Hz imaging)
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

    with open(params['data_path'], "rb") as f:
        data = pickle.load(f)
    
    animal= "Sphinx"
    
    calcium = data["calcium_dict"][animal]
    breath_filt = data["breath_filt_dict"][animal]
    phase = data["phase_peaks_dict"][animal]
    valve = data["valve_dict"][animal]

    fs = 1000
    
    sgn = np.signbit(breath_filt)  
    onset_mask = (sgn[1:] & ~sgn[:-1])
    onsets     = np.where(onset_mask)[0] + 1 

    min_isi = 0.08        # seconds
    keep   = np.r_[True, np.diff(onsets) > min_isi*fs]
    onsets = onsets[keep]
    
    print(f"{len(onsets)} inhalations detected")

    times  = onsets / fs                   # seconds
    isi    = np.diff(times)                # inter‑sniff intervals
    freq_cycle = 1.0 / isi                 # Hz
    t_cycle    = (times[:-1] + times[1:]) / 2   # mid‑cycle timestamp

    # fill NaNs by linear interpolation
    freq_cycle = np.interp(t_cycle, t_cycle[~np.isnan(freq_cycle)],
                                    freq_cycle[~np.isnan(freq_cycle)])
    
    frame_t = np.arange(len(breath_filt)) / fs
    # nearest‑neighbour step function (each frame gets its cycle’s freq)
    freq_step = np.interp(frame_t, t_cycle, freq_cycle, left=np.nan, right=np.nan)

    # smoother variant: low‑pass the step
    from scipy.ndimage import gaussian_filter1d
    sniff_freq_vec = gaussian_filter1d(freq_step, sigma=fs*0.1)   # 100‑ms SD

    plt.plot(frame_t, breath_filt / np.max(np.abs(breath_filt)), lw=0.5, label='breath (norm)')
    plt.plot(frame_t, sniff_freq_vec, lw=1.2, label='inst. freq (Hz)')
    plt.vlines(onsets/fs, -1, 1, colors='k', alpha=.1)      # inhale marks
    plt.xlim(20, 40); 
    plt.legend(); plt.xlabel('Time (s)')
    plt.show()

    valve_bool   = valve.astype(bool)
    # rising edges: 0→1 transition
    pulse_onsets = np.flatnonzero(np.diff(np.r_[False, valve_bool]) == 1)

    """pulse_freq_vec = np.full_like(valve, np.nan, dtype=float)
    for i in range(1, len(pulse_onsets)):
        start = pulse_onsets[i-1]
        end   = pulse_onsets[i]
        isi   = (end - start) / fs
        pulse_freq_vec[start:end] = 1.0 / isi
    pulse_freq_vec[end:] = pulse_freq_vec[end-1]"""
    
    win_s = 3.0                            # window length in seconds
    win_f = int(win_s * fs)                # frames

    # stick train: 1 at each onset, 0 elsewhere
    pulse_train = np.zeros_like(valve, dtype=int)
    pulse_train[pulse_onsets] = 1

    pulse_freq_vec = (np.convolve(pulse_train, np.ones(win_f), mode='same')
                    / win_s)             # units: pulses · s⁻¹


    t = np.arange(len(sniff_freq_vec)) / fs
    plt.figure(figsize=(10,3))
    plt.plot(t, sniff_freq_vec,  label='sniff Hz',  lw=1.2)
    plt.plot(t, pulse_freq_vec,  label='valve Hz',  lw=1.2)
    #plt.xlim(100,200); 
    plt.legend(); plt.ylabel('Hz'); plt.xlabel('Time (s)')
    plt.tight_layout()
    plt.show()
    
    # -------------------------------------------------------------
    # Parameters
    # -------------------------------------------------------------
    max_lag_s  = 2.0
    lag_step_s = 0.2

    max_lag_f  = int(max_lag_s  * fs)           # frames
    lag_step_f = int(lag_step_s * fs)

    # -------------------------------------------------------------
    # 1)  Choose a common “valid” slice for both series
    #     We’ll keep frames that have data for *all* lags.
    # -------------------------------------------------------------
    start = max_lag_f
    end   = len(pulse_freq_vec) - max_lag_f     # exclusive

    y = sniff_freq_vec[start:end]

    # -------------------------------------------------------------
    # 2)  Build the lagged predictors
    #     For each lag ℓ, take pulse_freq_vec shifted *earlier* by ℓ frames
    # -------------------------------------------------------------
    lags_f = np.arange(0, max_lag_f+1, lag_step_f)          # [0, 200, 400, …]
    X_cols = [
        pulse_freq_vec[start - ℓ : end - ℓ]                 # all same length
        for ℓ in lags_f
    ]
    X = np.column_stack(X_cols)

    # -------------------------------------------------------------
    # 3)  Drop rows with NaNs (if any)
    # -------------------------------------------------------------
    mask = ~np.isnan(X).any(1) & ~np.isnan(y)
    X, y = X[mask], y[mask]

    # -------------------------------------------------------------
    # 4)  Fit and plot
    # -------------------------------------------------------------
    model = sm.OLS(y, sm.add_constant(X)).fit()

    lags_s = lags_f / fs
    plt.errorbar(lags_s,
                model.params[1:],                # β for each lag
                yerr=model.bse[1:], fmt='o-')
    plt.xlabel('Pulse → sniff lag (s)')
    plt.ylabel('β  (Hz per Hz pulse rate)')
    plt.axhline(0, color='k', lw=.5)
    plt.title('Lagged linear influence of pulse rate on sniff rate')
    plt.tight_layout()
    plt.show()


    
if __name__ == "__main__":
    main()

import torch
import os
import pickle
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize, LinearSegmentedColormap
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler


import sys

sys.path.append("dunl-compneuro/src")
sys.path.append("")

import model

def init_params():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--res-path",
        type=str,
        help="res path",
        default= "sparseness/results/olfactorycalciumkernellength20num1_2025_02_20_11_30_36"
        #"sparseness/results/supervised_roi13_Sphinx"
        #"sparseness/results/HW1_calcium_supervised_across_rois_numwindow1_roi0_kernellength20_1kernels_1000unroll_2025_02_26_18_42_27"
        #"sparseness/results/olfactorycalciumkernellength20num1_2025_02_20_11_30_36"
        #"sparseness/results/olfactorycalciumkernellength20num1_2025_02_18_15_32_24"
        #"sparseness/results/olfactorycalciumkernellength20num1_2025_02_14_15_47_49" #HW1
        #"sparseness/results/olfactorycalciumkernellength20num1_2025_02_18_14_43_15" #HW4
    )
    parser.add_argument(
        "--path",
        type=str,
        help="data path",
        default="sparseness/Data/animals_data_processed.pkl",
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        help="sampling rate",
        default=10,
    )
    parser.add_argument(
        "--figsize",
        type=tuple,
        help="figsize",
        default=(12, 4),
    )
    parser.add_argument(
        "--event-filter",
        type=str,
        help="can take value inh or exh",
        default="inh",
    )
    parser.add_argument(
        "--kernel-length",
        type=int,
        help="kernel length",
        default=20,
    )
    args = parser.parse_args()
    params = vars(args)

    return params



def align_pulse_calcium(valve, calcium, phase, convolved_stim,
                        count_threshold=3, ipi_threshold=0.75,
                        window_sec=2, sigma=2.0,
                        cumulative_strength_threshold=None):
    """
    Align valve pulses onto the calcium time base and classify pulses as 'sparse'
    or 'non-sparse' based on:
      1. A low number of pulses in a sliding window of length window_sec.
      2. A long inter-pulse interval (IPI) relative to ipi_threshold.
      3. A high "cumulative stimulus strength," which is the sum of:
         - The current pulse's strength,
         - Each preceding pulse's strength in the window, weighted by a Gaussian
           that decays with the time difference.

    Also outputs convolved_stim_down, where for each pulse the corresponding
    calcium index is set to the median of convolved_stim over [start_idx : start_idx+50].
    """
    valve_ts = np.arange(0, len(valve) / 1000, 0.001)  # valve at 1 kHz
    ca_ts = np.arange(0, len(calcium) / 10, 0.1)       # calcium at 10 Hz
    
    whiff_onsets = np.where(np.diff(valve) > 0)[0]
        
    onset_resp = []
    event_resp = []
    sparse_classification = []
    onset_convolved=[]
    median_phase= []
    # Default threshold for cumulative strength if none is provided
    if cumulative_strength_threshold is None:
        cumulative_strength_threshold_val = 0.5 * np.max(convolved_stim)
    else:
        cumulative_strength_threshold_val = cumulative_strength_threshold

    for i in range(len(whiff_onsets)):
        start_idx = whiff_onsets[i]
        
        # Classify event based on phase
        inh_points = np.sum((0 <= phase[start_idx+1:start_idx+51]) & (phase[start_idx+1:start_idx+51] < np.pi))
        
        if inh_points <= 30:
            current_event = "exh"
        else:
            current_event = "inh"
        
        # Align to calcium time base
        index = np.abs(ca_ts - valve_ts[start_idx]).argmin()
        
        # Count pulses in the preceding window
        current_time = valve_ts[start_idx]
        window_start_time = current_time - window_sec
        count_in_window = np.sum(
            (valve_ts[whiff_onsets] >= window_start_time) & 
            (valve_ts[whiff_onsets] < current_time)
        )
        
        # Compute IPI
        if i == 0:
            ipi = np.nan
        else:
            ipi = current_time - valve_ts[whiff_onsets[i - 1]]
        
        # Strength of the current pulse
        current_strength = np.mean(convolved_stim[start_idx:start_idx+51])
        
        # Gaussian-weighted cumulative strength in the preceding window
        cumulative_strength = current_strength
        
        for j in range(i):
            prev_idx = whiff_onsets[j]
            prev_time = valve_ts[prev_idx]
            dt = current_time - prev_time
            if dt <= window_sec:
                weight = np.exp(-0.5 * (dt / sigma)**2)
                prev_strength = np.mean(convolved_stim[prev_idx:prev_idx+51])
                cumulative_strength += prev_strength * weight
        
        if (
            (count_in_window < count_threshold) and 
            (i == 0 or ipi >= ipi_threshold) and 
            (cumulative_strength >= cumulative_strength_threshold_val)
        ):
            classification = "sparse"
        else:
            classification = "non-sparse"

        window_start = max(0, start_idx - 50 // 2)
        window_end = min(len(convolved_stim), start_idx + 50 // 2)
        mean_value = np.mean(convolved_stim[window_start:window_end])
        inst_phase = np.median(phase[start_idx+1:start_idx+51])
            
        median_phase.append(inst_phase)
        onset_resp.append(index)
        event_resp.append(current_event)
        sparse_classification.append(classification)
        onset_convolved.append(mean_value)

        end_idx = min(start_idx + 51, len(convolved_stim))
        
    
    valve_down = np.zeros(len(calcium))
    valve_down[onset_resp] = 1
        
    return valve_down, onset_resp, event_resp, sparse_classification, onset_convolved, median_phase




def plot_heatmap(df, animal, out_path):
    
    df = df.dropna(subset=["delta_t1", "delta_t2"])

    bins_t1 = np.array([0, 1, 5, 10, 20, 30, 50, 80])
    bins_t2 = np.array([0, 1, 5, 10, 20, 30, 50, 80])

    df["delta_t1_bin"] = pd.cut(df["delta_t1"], bins=bins_t1, include_lowest=True)
    df["delta_t2_bin"] = pd.cut(df["delta_t2"], bins=bins_t2, include_lowest=True)

    heatmap_data = df.pivot_table(index="delta_t2_bin", columns="delta_t1_bin", values="codes", aggfunc="mean", observed=False)

    count_data = df.pivot_table(index="delta_t2_bin", columns="delta_t1_bin", values="codes", aggfunc="count", observed=False)
    annot_text = heatmap_data.map(lambda x: f"{x:.4f}" if pd.notna(x) else "") + "\n(n=" + count_data.map(lambda x: f"{int(x)}" if pd.notna(x) else "") + ")"

    plt.figure(figsize=(9, 9))
    ax = sns.heatmap(
        heatmap_data,
        annot=annot_text.values,
        fmt="",
        cmap="viridis",
        cbar_kws={'label': 'codes'},
        annot_kws={"fontsize": 8}
    )

    x_labels = [f"[{int(bin.left)/10};{int(bin.right)/10}]" for bin in heatmap_data.columns]
    ax.set_xticklabels(x_labels, rotation=0, ha='right')

    y_labels = [f"[{int(bin.left)/10};{int(bin.right)/10}]" for bin in heatmap_data.index]
    ax.set_yticklabels(y_labels, rotation=0, ha='right')

    plt.tick_params(labelsize=8)
    plt.xlabel("Delta t1: Time between current and previous pulse (s)", fontsize=10)
    plt.ylabel("Delta t2: Time between previous and previous previous pulse (s)", fontsize=10)
    plt.title(f'{animal}')
    plt.savefig(os.path.join(out_path, f"heatmap.png"), bbox_inches="tight", pad_inches=0.02)
    plt.show()


def plot_heatmap_t2(df, animal, out_path):
    df = df.dropna(subset=["delta_t1", "delta_t2"])
    
    bins_t1 = np.array([0, 1, 2, 5, 10, 15, 20, 25, 30, 50, 80])
    bins_t2 = np.array([15, df["delta_t2"].max()])

    df["delta_t1_bin"] = pd.cut(df["delta_t1"], bins=bins_t1, include_lowest=True)
    df["delta_t2_bin"] = pd.cut(df["delta_t2"], bins=bins_t2, include_lowest=True)

    heatmap_data = df.pivot_table(index="delta_t2_bin", columns="delta_t1_bin", values="codes", aggfunc="mean", observed=False)

    count_data = df.pivot_table(index="delta_t2_bin", columns="delta_t1_bin", values="codes", aggfunc="count", observed=False)
    annot_text = heatmap_data.map(lambda x: f"{x:.4f}" if pd.notna(x) else "") + "\n(n=" + count_data.map(lambda x: f"{int(x)}" if pd.notna(x) else "") + ")"

    plt.figure(figsize=(12, 3))
    ax=sns.heatmap(
        heatmap_data,
        annot=annot_text.values,
        fmt="",
        cmap="viridis",
        cbar_kws={'label': 'codes'},
        annot_kws={"fontsize": 8}
    )

    ax.set_aspect(0.8)
    x_labels = [f"[{int(bin.left)/10};{int(bin.right)/10}]" for bin in heatmap_data.columns]
    ax.set_xticklabels(x_labels, rotation=0, ha='right')

    ax.set_yticklabels([])

    plt.tick_params(labelsize=8)
    plt.xlabel("Delta t1: Time between current and previous pulse (s)", fontsize=10)
    plt.ylabel(" ", fontsize=10)
    plt.title(f'{animal}')
    plt.savefig(os.path.join(out_path, f"heatmap_t2.png"), bbox_inches="tight", pad_inches=0.02)
    plt.show()



def plot_linegraph_t2_by_animal(df, out_path):
    #df = df.dropna(subset=["delta_t1", "delta_t2"])
    
    #df = df.dropna(subset=["delta_t1"])
    df = df.dropna(subset=["delta_t"])

    #bins_t1 = np.array([0, 1, 2, 5, 10, 15, 20, 25, 30, 50, 80])
    #bins_t1 = np.array([0, 5, 15, 25, 35, 45, 80])
    #bins_t1 = np.array([1, 2, 5, 15, 25, 35, 45, 80])
    bins_t1 = np.array([1, 15, 35, 146])
    bin_centers = bins_t1[:-1] + np.diff(bins_t1) / 2
    
    colors = ['black', 'grey', 'lightgrey']
    
    col = 'codes'
    df[f'{col}_z'] = (df[col] - df[col].mean()) / df[col].std(ddof=0)

    plt.figure(figsize=(7, 4))
    
    for i, animal in enumerate(df["animal"].unique()):
        df_animal = df[df["animal"] == animal].copy()
        #df_animal["delta_t1_bin"] = pd.cut(df_animal["delta_t1"], bins=bins_t1, include_lowest=True)
        df_animal["delta_t1_bin"] = pd.cut(df_animal["delta_t"], bins=bins_t1, include_lowest=True)
        
        grouped = df_animal.groupby("delta_t1_bin")["codes_z"].agg(["mean", "std", "count"]).reset_index()
        grouped["std_error"] = grouped["std"] / np.sqrt(grouped["count"])
        
        plt.errorbar(bin_centers, grouped["mean"], yerr=grouped["std_error"], 
                     fmt='o-', capsize=3, label=f"{animal}", markersize=3, linewidth=1, color=colors[i % len(colors)])
    
    x_labels = [f"[{bins_t1[i]/10};{bins_t1[i+1]/10}[" for i in range(len(bins_t1)-1)]
    plt.xticks(bin_centers, x_labels, rotation=15, fontsize=6)
    plt.yticks(fontsize=6)
    
    plt.xlabel("δt₁: Time between current and previous pulse (s)", fontsize=7)
    plt.ylabel("Codes", fontsize=7)
    plt.legend(fontsize=6)
    
    out_file = os.path.join(out_path, "linegraph_t2_by_animal.png")
    plt.savefig(out_file, bbox_inches="tight", pad_inches=0.02)
    plt.show()


def plot_linegraph_t2_per_segment(df, animal):
    df = df.dropna(subset=["delta_t1"])

    df_animal = df[df['animal'] == animal].sort_values(by="onset_resp")
    
    df_segments = np.array_split(df_animal, 3)
    
    bins_t1 = np.array([0, 10, 25, 30, 50, 80])
    bin_centers = bins_t1[:-1] + np.diff(bins_t1) / 2

    if animal == 'HW1':
        cmap = plt.cm.Blues
    elif animal == 'HW4':
        cmap = plt.cm.Greens
    elif animal == 'Sphinx':
        cmap = plt.cm.Reds
    else:
        cmap = plt.cm.gray

    colors = [cmap(x) for x in np.linspace(0.4, 0.8, 3)]

    plt.figure(figsize=(12, 8))
    
    for i, seg in enumerate(df_segments):
        seg = seg.copy()
        seg["delta_t1_bin"] = pd.cut(seg["delta_t1"], bins=bins_t1, include_lowest=True)
        grouped = seg.groupby("delta_t1_bin")["codes"].agg(["mean", "std", "count"]).reset_index()
        
        group_centers = grouped["delta_t1_bin"].apply(lambda interval: interval.left + (interval.right - interval.left) / 2)
        
        se = grouped["std"] / np.sqrt(grouped["count"])
        
        plt.errorbar(group_centers, grouped["mean"], yerr=se, fmt='o-', capsize=5, 
                     label=f'Segment {i+1}', color=colors[i])
    
    x_labels = [f"[{interval.left/10:.1f};{interval.right/10:.1f}]" 
                for interval in pd.cut(df["delta_t1"], bins=bins_t1, include_lowest=True).cat.categories]
    
    plt.xticks(bin_centers, labels=x_labels, rotation=35, ha='right', fontsize=8)
    plt.xlabel("Delta t1: Time between current and previous pulse (s)", fontsize=10)
    plt.ylabel("Codes", fontsize=10)
    plt.title(f"{animal} (Supervised DUNL)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    plt.show()



def plot_hist_codes(df, out_path):
    
    codes_inh = df[df["event_resp"] == "inh"]["codes"]
    codes_exh = df[df["event_resp"] == "exh"]["codes"]

    bins = 20

    counts_inh, bin_edges = np.histogram(codes_inh, bins=bins)
    counts_exh, _ = np.histogram(codes_exh, bins=bin_edges)  
    ymax = max(counts_inh.max(), counts_exh.max())  

    fig, axes = plt.subplots(2, 1, figsize=(8, 16), sharex=True, sharey=True)

    axes[0].hist(codes_inh, bins=bin_edges, alpha=0.7, color="green", edgecolor="black", linewidth=1.2)
    axes[0].set_title("Codes - Inhalation")
    axes[0].set_ylabel("Frequency")
    axes[0].grid(True)
    axes[0].set_ylim(0, ymax)  

    axes[1].hist(codes_exh, bins=bin_edges, alpha=0.7, color="red", edgecolor="black", linewidth=1.2)
    axes[1].set_title("Codes - Exhalation")
    axes[1].set_xlabel("Code Values")
    axes[1].set_ylabel("Frequency")
    axes[1].grid(True)
    axes[1].set_ylim(0, ymax)  

    plt.tight_layout()
    plt.savefig(os.path.join(out_path, f"hist_codes.png"), bbox_inches="tight", pad_inches=0.02)
    plt.show()


def plot_hist_codes_cond(df, out_path):
        
    df_filt = df[(df["delta_t2"] >= 10) & (df["delta_t1"] < 25) & (df["delta_t1"] >= 20)]
    codes_inh = df_filt[df_filt["event_resp"] == "inh"]["codes"]
    
    bins = 10

    _, bin_edges = np.histogram(codes_inh, bins=bins)

    plt.figure(figsize=(6, 6))

    plt.hist(codes_inh, bins=bin_edges, alpha=0.7, color="blue", edgecolor="black", linewidth=1.2)
    plt.title("Codes - Inhalation - Isolated Pulses [2s,2.5s[")
    plt.ylabel("Frequency")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(out_path, f"hist_codes_isolated.png"), bbox_inches="tight", pad_inches=0.02)
    plt.show()
    

def plot_hist_codes_panel(df, out_path):
    #bins_t1 = np.array([0, 1, 2, 5, 10, 15, 20, 25, 30, 50, 80])
    bins_t1 = np.array([0, 20, 100])
    
    n_bins = len(bins_t1) - 1  # Number of subplots
    n_cols = 3  # Number of columns in the panel
    n_rows = int(np.ceil(n_bins / n_cols))  # Determine number of rows needed

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3), sharex=True, sharey=True)
    axes = axes.flatten()  # Flatten in case of a single row

    bins = 10  # Number of bins for histogram

    for i in range(n_bins):
        t1_min, t1_max = bins_t1[i], bins_t1[i + 1]
        #df_filt = df[(df["delta_t2"] >= 10) & (df["delta_t1"] < t1_max) & (df["delta_t1"] >= t1_min)]
        df_filt = df[(df["delta_t1"] < t1_max) & (df["delta_t1"] >= t1_min)]
        codes_inh = df_filt[df_filt["event_resp"] == "inh"]["codes"]

        _, bin_edges = np.histogram(codes_inh, bins=bins)

        ax = axes[i]
        ax.hist(codes_inh, bins=bin_edges, alpha=0.7, color="blue", edgecolor="black", linewidth=1.2)
        ax.set_title(f"Δt in [{t1_min/10}, {t1_max/10})", fontsize = 8)
        ax.set_ylabel("Freq", fontsize = 8)

    for i in range(n_bins, len(axes)):
        fig.delaxes(axes[i])

    axes[-1].set_xlabel("Codes", fontsize = 8) 
    fig.suptitle("Supervised DUNL", fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(out_path, "hist_codes_grid.png"), bbox_inches="tight", pad_inches=0.02)
    plt.show()
    
def plot_code_isolated_paired(df):
    isolated = df[df["delta_t1"] >= 20]["codes"]  # 2s = 200 * 10ms

    paired = df[df["delta_t1"] < 20]["codes"]

    means = [np.mean(isolated), np.mean(paired)]
    std_errors = [np.std(isolated) / np.sqrt(len(isolated)), np.std(paired) / np.sqrt(len(paired))]

    plt.figure(figsize=(8, 6))
    plt.errorbar(["Isolated Pulses (Δt>=2s)", "Paired Pulses (Δt<2s)"], means, yerr=std_errors, fmt='o-', capsize=5, markersize=8, label="Mean Code")
    
    plt.ylabel("Mean Code")
    plt.title("Effect of Pulse Timing on Code")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.show()



def plot_evolution_paired_isolated(df):
    plt.figure(figsize=(10, 6))
    
    color_map = {
        'HW1': {"isolated": "#66b3ff", "paired": "#0059b3"},  # light blue, dark blue
        'HW4': {"isolated": "#66ff66", "paired": "#008000"},    # light green, dark green
        'Sphinx': {"isolated": "#ff6666", "paired": "#b30000"}    # light red, dark red
    }
    
    for animal in df['animal'].unique():
        df_animal = df[df["animal"] == animal].sort_values(by="onset_resp")
        df_animal_seg = np.array_split(df_animal, 3)
        df_ani_1, df_ani_2, df_ani_3 = df_animal_seg    
        
        means_isolated = []
        means_paired = []
        errors_isolated = []
        errors_paired = []
        
        for seg in [df_ani_1, df_ani_2, df_ani_3]:
            isolated = seg[seg["delta_t1"] >= 25]["codes"]
            paired = seg[seg["delta_t1"] < 25]["codes"]
            
            mean_isolated = np.mean(isolated) if len(isolated) > 0 else np.nan
            mean_paired = np.mean(paired) if len(paired) > 0 else np.nan
            
            se_isolated = np.std(isolated) / np.sqrt(len(isolated)) if len(isolated) > 0 else np.nan
            se_paired = np.std(paired) / np.sqrt(len(paired)) if len(paired) > 0 else np.nan
            
            means_isolated.append(mean_isolated)
            means_paired.append(mean_paired)
            errors_isolated.append(se_isolated)
            errors_paired.append(se_paired)
        
        segments = [1, 2, 3]
        
        colors = color_map.get(animal, {"isolated": "gray", "paired": "black"})
        isolated_color = colors["isolated"]
        paired_color = colors["paired"]
        
        plt.errorbar(segments, means_isolated, yerr=errors_isolated, marker='o', linestyle='-', 
                     label=f'{animal} Isolated', color=isolated_color)
        plt.errorbar(segments, means_paired, yerr=errors_paired, marker='s', linestyle='--', 
                     label=f'{animal} Paired', color=paired_color)
    
    plt.xlabel("Segment ID")
    plt.xticks(segments)
    plt.ylabel("Code")
    plt.title("Evolution of Codes: Isolated vs. Paired Pulses")
    plt.legend()
    plt.show()


def plot_sparse_boxplot(df, classification_col='sparse_class', value_col='codes'):
    
    df = df[df["event_resp"]=="inh"]

    plt.figure(figsize=(8, 6))
    # Ensure the order is consistent: sparse first, non-sparse second.
    order = ['sparse', 'non-sparse']
    print(f"there are {df[df['sparse_class']=='sparse'].shape} sparse pulses")
    print(f"there are {df[df['sparse_class']=='non-sparse'].shape} non-sparse pulses")
    sns.boxplot(data=df, x=classification_col, y=value_col, order=order)
    plt.xlabel("Pulse Classification")
    plt.ylabel(value_col)
    plt.title(f"Distribution of {value_col} for Sparse vs. Non-Sparse Pulses")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()


def plot_scatter_3d(df):

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    order = ['inh', 'exh']
    colors = ['blue', 'orange']
    
    for label, color in zip(order, colors):
        sub_df = df[df['event_resp'] == label]
        ax.scatter(sub_df["conv_stim"], sub_df["codes"], sub_df["IPI"], 
                   color=color, label=label, alpha=0.7)
    
    ax.set_xlabel("Convolved Stim")
    ax.set_ylabel("Code")
    ax.set_zlabel("IPI")
    ax.set_title("3D Scatter: Code vs. Convolved Stim vs. IPI")
    ax.legend(title="Event Resp")
    plt.show()


def plot_scatter_2d(df, animal):
    
    df = df[df['animal']==animal]

    fig, ax = plt.subplots(figsize=(8, 6))
    
    sc = ax.scatter(df["IPI"], df["conv_stim"], c=df["codes"], cmap="viridis", alpha=0.7)
    
    ax.set_ylabel("Conv Stim")
    ax.set_xlabel("IPI")
    ax.set_title(f"IPI vs. Conv Stim for {animal}")
    plt.grid(True, linestyle="--", alpha=0.5)
    
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Code")
    
    plt.show()



def plot_scatter_2d_logx(df, animal, log_x=True):
    
    df_animal = df[df['animal'] == animal].copy()
    
    if log_x:
        df_animal = df_animal[df_animal['IPI'] > 0]
    
    fig, ax = plt.subplots(figsize=(8, 6))

    sc = ax.scatter(
        df_animal["IPI"], 
        df_animal["codes"], 
        alpha=0.7
    )
    
    if log_x:
        ax.set_xscale("log")
        ax.set_xlabel("IPI (log scale)")
    else:
        ax.set_xlabel("IPI")
    
    ax.set_ylabel("Codes")
    ax.set_title(f"IPI vs. Codes for {animal}")
    
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()


def compute_adaptation_gain(df):

    results = []
    for idx, row in df.iterrows():
        original = np.array(row['original_segment'])
        dt = row['delta_t1']
        
        if (len(original) == 0) or pd.isna(dt):
            # Skip invalid or missing data
            continue
        
        actual_peak = np.max(original)
        predicted_peak = row["codes"]
        
        if predicted_peak == 0:
            gain = np.nan
        else:
            gain = actual_peak / predicted_peak
        
        results.append({'delta_t1': dt, 'gain': gain})
    
    return pd.DataFrame(results)


def analyze_adaptation_gain(df, animal):

    # Compute adaptation gain for each whiff
    gain_df = compute_adaptation_gain(df)
    gain_df.dropna(inplace=True)
    
    if gain_df.empty:
        print("No valid data to analyze.")
        return
    
    # Define groups based on delta_t1 intervals
    conditions = [
        ((gain_df["delta_t1"] > 0) & (gain_df["delta_t1"] <= 5)),
        ((gain_df["delta_t1"] >= 10) & (gain_df["delta_t1"] <= 15)),
        ((gain_df["delta_t1"] >= 25) & (gain_df["delta_t1"] <= 30))
    ]
    choices = ['(0,5]', '[10,15]', '[25,30]']
    gain_df['group'] = np.select(conditions, choices, default='Other')
    
    # Keep only the events in the three groups of interest
    gain_df = gain_df[gain_df['group'] != 'Other']
    
    # Define colors for each group
    color_map = {'(0,5]': 'blue',
                 '[10,15]': 'green',
                 '[25,30]': 'yellow'}
    
    plt.figure(figsize=(6, 6))
    
    # Plot scatter points for each group with its designated color
    for grp, group_df in gain_df.groupby('group'):
        x = group_df['delta_t1']
        y = group_df['gain']
        color = color_map.get(grp, 'black')
        plt.scatter(x, y, color=color, alpha=0.8, label=grp)
    
    # Global regression on all data points
    x_all = gain_df['delta_t1']
    y_all = gain_df['gain']
    coeffs = np.polyfit(x_all, y_all, 1)
    poly1d_fn = np.poly1d(coeffs)
    x_line = np.linspace(x_all.min(), x_all.max(), 100)
    y_line = poly1d_fn(x_line)
    
    # Compute R² for the global regression
    y_pred = poly1d_fn(x_all)
    SS_res = np.sum((y_all - y_pred) ** 2)
    SS_tot = np.sum((y_all - np.mean(y_all)) ** 2)
    R2 = 1 - SS_res / SS_tot if SS_tot != 0 else np.nan
    
    plt.plot(x_line, y_line, color='black', linestyle='--', linewidth=2, 
             label=f'Global Regression (R²={R2:.2f})')
    
    plt.xlabel("δt₁ (time since previous pulse)")
    plt.ylabel("Gain (actual_peak / predicted_peak)")
    plt.title(f"Gain vs. δt₁ ({animal})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    

def plot_density_filled_by_category(df, animal):
    
    df = df[df['onset_convolved']<6]
    df = df[df['IPI']<100]
    df = df[df['codes']>0]
    
    median_codes = df['codes'].mean()

    df_above = df[df['codes'] > median_codes]
    df_below = df[df['codes'] <= median_codes]
    
    df_above = df_above[df_above['onset_conv_prev']<6]
    df_below = df_below[df_below['onset_conv_prev']<6]

    plt.figure(figsize=(6, 6))

    sns.scatterplot(data=df_above, x='onset_conv_prev', y='IPI', color='green',
                    label='Codes > Median', s=20, alpha=0.7)
    sns.scatterplot(data=df_below, x='onset_conv_prev', y='IPI', color='red',
                    label='Codes <= Median', s=20, alpha=0.7)

    sns.kdeplot(data=df_above, x='onset_convolved', y='IPI', fill=True,thresh=0.1, levels=10,cmap="Greens",alpha=0.4)
    sns.kdeplot(data=df_below, x='onset_convolved', y='IPI', fill=True,thresh=0.1, levels=10,cmap="Reds",alpha=0.4)

    plt.xlabel("Previous Stimulus strength")
    plt.ylabel("Time since preceding whiff")
    plt.title(animal)
    plt.legend()
    plt.show()
    


def compare_logistic_regressions(df):

    df = df.copy()

    df = df[(df['onset_convolved'] < 6) & (df['IPI'] < 100) & (df['codes'] > 0)]
    df = df.dropna(subset=['onset_conv_prev', 'IPI', 'onset_convolved', 'codes'])

    mean_code = df['codes'].mean()
    df['deviation'] = (df['codes'] > mean_code).astype(int)
    y = df['deviation']

    feature_sets = {
        'Stimulus History (prev_strength + IPI)': ['onset_conv_prev', 'IPI'],
        'Current Stimulus Only (current_strength)': ['onset_convolved'],
        'Combined (prev_strength + IPI + current_strength)': ['onset_conv_prev', 'IPI', 'onset_convolved'],
    }

    plt.figure(figsize=(7, 6))

    for i, (label, features) in enumerate(feature_sets.items()):
        X = df[features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = LogisticRegression()
        model.fit(X_scaled, y)

        y_prob = model.predict_proba(X_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_prob)
        auc = roc_auc_score(y, y_prob)

        plt.plot(fpr, tpr, label=f"{label} (AUC = {auc:.2f})")

        intercept = model.intercept_[0]
        coefs = model.coef_[0]
        eqn_terms = [f"{intercept:.2f}"]
        for feat_name, coef in zip(features, coefs):
            eqn_terms.append(f"{coef:+.2f}×{feat_name}")
        eqn = "logit = " + " ".join(eqn_terms)

        idx = int(len(fpr) * 0.6)
        x_annot = fpr[idx]
        y_annot = tpr[idx]
        plt.text(x_annot + 0.03, y_annot, eqn, fontsize=7, alpha=0.85)

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve: Predicting Response Deviation")
    plt.legend()
    plt.tight_layout()
    plt.show()




def compare_conditional_log_regressions(df, targets, tolerance=0.2):

    df = df.copy()
    results = []

    mean_code = df['codes'].mean()
    df['deviation'] = (df['codes'] > mean_code).astype(int)

    df['onset_conv_prev_ipi'] = df['onset_conv_prev'] * df['IPI']

    plt.figure(figsize=(7, 6))

    for i, target in enumerate(targets):
        subset = df[
            (df['onset_convolved'] >= target - tolerance) &
            (df['onset_convolved'] <= target + tolerance)
        ].dropna(subset=['onset_conv_prev', 'IPI', 'onset_conv_prev_ipi', 'deviation'])

        if len(subset) < 20:
            print(f"Skipping onset_convolved ≈ {target} — not enough data.")
            continue

        X = subset[['onset_conv_prev', 'IPI', 'onset_conv_prev_ipi']]
        y = subset['deviation']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = LogisticRegression()
        model.fit(X_scaled, y)

        y_prob = model.predict_proba(X_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_prob)
        auc = roc_auc_score(y, y_prob)

        label = f"{target:.2f} ± {tolerance:.2f} (AUC = {auc:.2f})"
        plt.plot(fpr, tpr, label=label)

        coefs = model.coef_[0]
        intercept = model.intercept_[0]
        eqn = (
            f"logit = {intercept:.2f} + "
            f"{coefs[0]:.2f}×prev + "
            f"{coefs[1]:.2f}×IPI + "
            f"{coefs[2]:.2f}×(prev×IPI)"
        )

        try:
            x_annot = fpr[min(len(fpr)-1, 8+i)]
            y_annot = tpr[min(len(tpr)-1, 8+i)]
        except IndexError:
            x_annot, y_annot = 0.6, 0.4 + 0.1*i

        plt.text(x_annot + 0.02, y_annot, eqn, fontsize=7, alpha=0.8)

        results.append({
            'target': target,
            'auc': auc,
            'model': model,
            'coefficients': pd.Series(coefs, index=['onset_conv_prev', 'IPI', 'interaction']),
            'intercept': intercept
        })

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve: Predicting Response Deviation")
    plt.legend(title="Fixed Stimulus Strength")
    plt.tight_layout()
    plt.show()

    return results


def compute_pre_pulse_strength(row, df, window):

    t_start = row['onset_resp'] - window
    t_end = row['onset_resp']
    window_df = df[(df['onset_resp'] >= t_start) & (df['onset_resp'] < t_end)]
    if not window_df.empty:
        return window_df.sort_values('onset_resp')['onset_convolved'].iloc[-1]
    else:
        return 0

def plot_codes_deviation_by_animal_with_gradient(df):

    strong_df = df[(df['onset_convolved'] >= 2) & (df['onset_convolved'] <= 10)].copy()
    
    median_codes = df['codes'].median()
    strong_df['code_deviation'] = strong_df['codes'] - median_codes
    
    df_sorted = df.sort_values('onset_resp')
    strong_df = strong_df.sort_values('onset_resp')
    
    strong_df['pre_pulse_strength'] = strong_df.apply(
        lambda row: compute_pre_pulse_strength(row, df_sorted, window=3), axis=1
    )
    
    strong_df = strong_df[strong_df['pre_pulse_strength'] > 0.05]
    
    rho, p_value = spearmanr(strong_df['pre_pulse_strength'], strong_df['code_deviation'])
    
    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    
    norm = Normalize(vmin=2, vmax=10)
    
    animal_color_bounds = {
        "HW4": ["#ffcccc", "#cc0000"],    # light red to dark red
        "Sphinx": ["#cce5ff", "#003366"],   # light blue to dark blue
        "HW1": ["#ccffcc", "#006600"],      # light green to dark green
    }
    
    default_cmap = LinearSegmentedColormap.from_list("default", ["#d3d3d3", "#000000"])
    
    animal_list = strong_df['animal'].unique()
    
    for animal in animal_list:
        animal_df = strong_df[strong_df['animal'] == animal]

        if animal in animal_color_bounds:
            cmap = LinearSegmentedColormap.from_list(animal, animal_color_bounds[animal])
            base_color = animal_color_bounds[animal][1]
        else:
            cmap = default_cmap
            base_color = "black"
            
        animal_df = animal_df[(animal_df['pre_pulse_strength'] <8) & (animal_df['pre_pulse_strength'] >=1)]
        
        face_colors = [cmap(norm(val)) for val in animal_df['onset_convolved'].values]
        
        ax.scatter(animal_df['pre_pulse_strength'], animal_df['code_deviation'],
                   c=face_colors, s=60, alpha=0.8, label=animal)
        
        if len(animal_df) > 1:
            x = animal_df['pre_pulse_strength'].values
            y = animal_df['code_deviation'].values
            slope, intercept = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, linestyle=":", color=base_color, linewidth=2)
    
    ax.set_xlabel("Last Stimulus Strength in 3ms Before Pulse")
    ax.set_ylabel("Deviation from Median Codes")
    
    ax.text(0.05, 0.95, f"Spearman rho = {rho:.2f}\np = {p_value:.3f}",
            transform=ax.transAxes, verticalalignment='top')
    
    legend_elements = []
    for animal in animal_list:

        if animal in animal_color_bounds:
            legend_color = animal_color_bounds[animal][1]  
        else:
            legend_color = "black"
        
        legend_elements.append(
            Line2D([0], [0],
                marker='o', color='w',
                markerfacecolor=legend_color,
                markersize=8, label=animal)
        )
    ax.legend(handles=legend_elements, title="Animal")
        
    plt.show()


def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device is", device)

    # init parameters -------------------------------------------------------#
    print("init parameters.")
    params_init = init_params()

    # take parameters from the result path
    params = pickle.load(
        open(os.path.join(params_init["res_path"], "params.pickle"), "rb")
    )
    for key in params_init.keys():
        params[key] = params_init[key]

    # create folders -------------------------------------------------------#
    model_path = os.path.join(
        params["res_path"],
        "model",
        "model_final.pt",
    )

    out_path = os.path.join(
        params["res_path"],
        "figures",
    )
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    postprocess_path = os.path.join(
        params["res_path"],
        "postprocess",
    )

    # load valve -------------------------------------------------------------------#

    with open(params["path"], "rb") as f:
        data = pickle.load(f)
        
        
    animals = ['HW1', 'HW4', 'Sphinx']
    #animals = ['HW1']
    
    all_results = []
    offset = 0

    for animal in animals:
        valve = data["valve_dict"][animal] / 100
        phase = data["phase_peaks_dict"][animal]
        convolved_stim = data["convolved_stim_dict"][animal]
        calcium_signal = data["calcium_dict"][animal].mean(axis=1)

        #valve_down, onset_resp, event_resp = align_pulse_calcium(valve, calcium_signal, phase)
        valve_down, onset_resp, event_resp, sparse_classification, onset_convolved, median_phase = align_pulse_calcium(valve, calcium_signal, phase, convolved_stim)

        # Adjust onsets by offset
        onset_resp = [onset + offset for onset in onset_resp
                      if onset < len(calcium_signal) - params["kernel_length"]
        ]
        
        for i, onset in enumerate(onset_resp):
            calcium_segment = calcium_signal[onset-offset:onset-offset+20]
            
            delta_t = onset_resp[i] - onset_resp[i - 1] if i > 0 else None
            
            if event_resp[i] == "inh":
                delta_t1 = onset_resp[i] - onset_resp[i - 1] if i > 0 and event_resp[i - 1] == "inh" else None
                delta_t2 = onset_resp[i - 1] - onset_resp[i - 2] if i > 1 and event_resp[i - 1] == "inh" and event_resp[i - 2] == "inh" else None
            else:
                delta_t1, delta_t2 = None, None

            all_results.append({
                "onset_resp": onset,
                "event_resp": event_resp[i],
                "median_phase": median_phase[i],
                #"sparse_class": sparse_classification[i],
                "onset_convolved":onset_convolved[i],
                #"original_segment": calcium_segment.to_numpy(),
                "delta_t": delta_t,
                "delta_t1": delta_t1,
                "delta_t2": delta_t2,
                "animal": animal
            })

        # Update offset for the next animal
        offset += len(calcium_signal)

    df = pd.DataFrame(all_results)
        
    # load codes ------------------------------------------------------#
    
    net = torch.load(model_path, map_location=device, weights_only=False)
    net.to(device)
    net.eval()

    """for data_path in data_path_list:
        datafile_name = data_path.split("/")[-1].split(".pt")[0]

        xhat = torch.load(
            os.path.join(postprocess_path, "xhat_{}.pt".format(datafile_name))
        )

        y = torch.load(
            os.path.join(postprocess_path, "y_{}.pt".format(datafile_name))
        )"""    

    xhat = torch.load(
        os.path.join(postprocess_path, "xhat.pt")
    )

    y = torch.load(
        os.path.join(postprocess_path, "y.pt")
    )
    
    codehat = xhat[0, 0, 0, :].clone().detach().cpu().numpy()

    code_selected = []
    for onset in df["onset_resp"].to_numpy():
        start_idx = int(onset) 
        end_idx = start_idx + 2  

        if start_idx < len(codehat):
            window_vals = codehat[start_idx:min(end_idx, len(codehat))]
            selected = 0
            for val in window_vals:
                if val != 0:
                    selected = val
                    break
            code_selected.append(selected)
        else:
            code_selected.append(0)

    df["codes"] = code_selected
        
    #codes = xhat[0, 0, :].cpu().numpy()[0][df["onset_resp"].values]
    #df["codes"] = codes
    
    df['IPI'] = df['onset_resp'].diff()
    df["onset_conv_prev"] = df["onset_convolved"].shift(1)

    animal = "Sphinx"
    df_hw1= df[df['animal']==animal]
    
    print(df_hw1.head())

    #plot_codes_deviation_by_animal_with_gradient(df)
    
    #results = compare_conditional_log_regressions(df, targets=[0.5, 4], tolerance=0.2)
    #compare_logistic_regressions(df)

    #plot_density_filled_by_category(df_hw1, animal)    
    
    #plot_scatter_2d_logx(df, animal)
    
    #plot_scatter_2d(df, animal)
    #plot_scatter_3d(df)
    
    #plot_evolution_paired_isolated(df)
    
    #plot_code_isolated_paired(df)
    #plot_hist_codes_panel(df, out_path)
    
    #plot_hist_codes(df, out_path)
    #plot_hist_codes_panel(df, out_path)
    #plot_hist_codes_cond(df, out_path)

    #plot_sparse_boxplot(df_hw1)
    plot_linegraph_t2_by_animal(df, out_path)

    #plot_linegraph_t2_per_segment(df, animal)
    #plot_heatmap(df, animal, out_path)
    #plot_heatmap_t2(df, animal, out_path)

if __name__ == "__main__":
    main()


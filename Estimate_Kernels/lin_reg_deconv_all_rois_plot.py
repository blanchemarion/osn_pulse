"""
Functions to plot the result df created in lin_reg_deconv_all_rois_compute.py
"""


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import pickle
import pandas as pd
import matplotlib.colors as mcolors
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score

def init_params():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--data-path",
        type=str,
        help="data path",
        default="sparseness/Data/animals_data_processed.pkl",
    )
    parser.add_argument(
        "--out-path",
        type = str,
        help = "path to solve data",
        default= "sparseness/Data/linear_regression_all.pkl"
    )
    parser.add_argument(
        "--event-filter",
        type=str,
        help="can take value inh or exh",
        default="inh",
    )
    args = parser.parse_args()
    params = vars(args)

    return params



def kernel_metrics(kernel):

    metrics = {
        "symmetric_auc": np.sum(np.abs(kernel)),
        #"mean": np.mean(kernel[2:6]),
    }
    return metrics


def compute_number_preceding_pulses(df, window):
    results = []
    
    for i, onset in enumerate(df["onset_resp"]):        
        current_event = df["event_resp"].iloc[i]

        start_idx = max(0, onset - window)
        preceding_events = df[(df["onset_resp"] >= start_idx) & (df["onset_resp"] < onset)]

        filtered_events = preceding_events[preceding_events["event_resp"] == current_event]

        number_events = len(filtered_events) + 1 
        results.append(number_events)

    return results

def compute_number_preceding_pulses_in_bin(df, lower_bound, upper_bound):
    results = []
    
    for i, onset in enumerate(df["onset_resp"]):        
        current_event = df["event_resp"].iloc[i]
        
        start_idx = max(0, onset - upper_bound)
        end_idx = max(0, onset - lower_bound)

        preceding_events = df[(df["onset_resp"] >= start_idx) & (df["onset_resp"] < end_idx)]
        filtered_events = preceding_events[preceding_events["event_resp"] == current_event]

        number_events = len(filtered_events) + 1  
        results.append(number_events)

    return results


def reconstruct_calcium_signal(df, signal_length):

    reconstructed_signal = np.zeros(signal_length)

    for _, row in df.iterrows():
        onset = row["onset_resp"]
        kernel = row["kernel"]

        kernel_length = len(kernel)
        if onset + kernel_length < signal_length:
            reconstructed_signal[onset:onset+kernel_length] += kernel

    return reconstructed_signal

def plot_reconstructed_signal(original_signal, reconstructed_signal, df, start_index, end_index):
    df_range = df[(df["onset_resp"] >= start_index) & (df["onset_resp"] <= end_index)].copy()

    df_range["metrics"] = df_range["kernel"].apply(kernel_metrics)

    sample_metrics = df_range.iloc[0]["metrics"]
    metric_names = list(sample_metrics.keys())

    colors = sns.color_palette("tab10", len(metric_names))
    metric_colors = {name: color for name, color in zip(metric_names, colors)}

    fig, ax1 = plt.subplots(figsize=(12, 4))

    ax1.plot(range(start_index, end_index), original_signal[start_index:end_index], 
             label="Original Calcium Signal", color="black", alpha=1, linewidth = 0.7)
    ax1.plot(range(start_index, end_index), reconstructed_signal[start_index:end_index], 
             label="Reconstructed Signal", color="blue", alpha=1, linewidth = 0.7)
    
    ax1.set_xlabel("Time (frames)")
    ax1.set_ylabel("Calcium Response", color="black")
    ax1.tick_params(axis='y', labelcolor="black")

    for _, row in df_range.iterrows():
        color = "green" if row["event_resp"] == "inh" else "red"
        ax1.axvline(x=row["onset_resp"], color=color, linestyle="--", alpha=0.7, linewidth = 0.6)

    ax2 = ax1.twinx()
    ax2.set_ylim(-0.8, 1.3)
    ax2.set_ylabel("Symmetric AUC Values")

    for name, color in metric_colors.items():
        values = [row['metrics'][name] for _, row in df_range.iterrows()]
        onsets = df_range["onset_resp"]
        ax2.scatter(onsets, values, color='g', label=name, alpha=0.8, s=5)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    
    #ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right", bbox_to_anchor=(1, 1))

    plt.title("Original vs Reconstructed Calcium Signal (Linear Regression Deconvolution)")
    plt.tight_layout()
    plt.show()


def plot_heat_ca_time_pulses(df, title="All Animals"):

    df = df[df['event_resp']=='inh']

    bin_edges = np.array([0, 1, 10, 20, 30, 50])
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  

    heatmap_data = np.zeros((4, len(bin_centers), len(df['animal'].unique())))  
    count_data = np.zeros((4, len(bin_centers)))

    animals = df['animal'].unique()

    for animal_idx, animal in enumerate(animals):
        df_animal = df[df['animal']==animal]

        for col_idx, (lower_bound, upper_bound) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
            df_animal["preceding_train"] = compute_number_preceding_pulses_in_bin(df_animal, lower_bound, upper_bound)

            grouped = df_animal.groupby("preceding_train")["sym_auc_kernel"].agg(["mean", "count"]).reindex(range(1, 5))

            heatmap_data[:, col_idx, animal_idx] = grouped["mean"].fillna(np.nan)
            count_data[:, col_idx] += grouped["count"].fillna(0)

    mean_heatmap = np.nanmean(heatmap_data, axis=2)

    annotations = np.empty((4, len(bin_centers)), dtype=object)  

    for row_idx in range(4):
        for col_idx in range(len(bin_centers)):
            mean_val = mean_heatmap[row_idx, col_idx]
            total_count= int(count_data[row_idx, col_idx])
            if not np.isnan(mean_val):
                annotations[row_idx, col_idx] = f"{mean_val:.4f}\n(n={total_count})"
            else:
                annotations[row_idx, col_idx]=""

    plt.figure(figsize=(10, 5))
    sns.heatmap(
        mean_heatmap,
        annot=annotations,
        fmt="",
        xticklabels=[f"[{lower/10}, {upper/10}]" for lower, upper in zip(bin_edges[:-1], bin_edges[1:])],
        yticklabels=range(0, 4),
        cmap="viridis",
        cbar_kws={"label": "Mean Symmetric AUC Kernel"}
    )

    plt.xticks(rotation=45, fontsize=8)  
    plt.yticks(fontsize=8) 
    plt.xlabel("Time Window (s)", fontsize=10) 
    plt.ylabel("# of Preceding Pulses", fontsize=10) 
    plt.title(f'{title}')

    plt.tight_layout()
    plt.show()



def plot_heatmap(df, animal="All Animals"):
    df = df.dropna(subset=["delta_t1", "delta_t2"])

    bins_t1 = np.array([0, 1, 5, 10, 20, 30, 50, 80])
    bins_t2 = np.array([0, 1, 5, 10, 20, 30, 50, 80])
    
    """bins_t1 = np.array([0, 1, 7, 15, df["delta_t1"].max()])
    bins_t2 = np.array([0, 1, 7, 15, df["delta_t2"].max()])"""

    df["delta_t1_bin"] = pd.cut(df["delta_t1"], bins=bins_t1, include_lowest=True)
    df["delta_t2_bin"] = pd.cut(df["delta_t2"], bins=bins_t2, include_lowest=True)

    heatmap_data = df.pivot_table(index="delta_t2_bin", columns="delta_t1_bin", values="sym_auc_kernel", aggfunc="mean", observed=False)

    count_data = df.pivot_table(index="delta_t2_bin", columns="delta_t1_bin", values="sym_auc_kernel", aggfunc="count", observed=False)
    annot_text = heatmap_data.map(lambda x: f"{x:.4f}" if pd.notna(x) else "") + "\n(n=" + count_data.map(lambda x: f"{int(x)}" if pd.notna(x) else "") + ")"

    plt.figure(figsize=(9, 9))
    ax = sns.heatmap(
        heatmap_data,
        annot=annot_text.values,
        fmt="",
        cmap="viridis",
        cbar_kws={'label': 'Mean symmetric auc'},
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
    plt.show()


def plot_heatmap_t2(df, animal="All Animals"):
    df = df.dropna(subset=["delta_t1", "delta_t2"])
    
    bins_t1 = np.array([0, 1, 2, 5, 10, 15, 20, 25, 30, 50, 80])
    bins_t2 = np.array([15, df["delta_t2"].max()])

    df["delta_t1_bin"] = pd.cut(df["delta_t1"], bins=bins_t1, include_lowest=True)
    df["delta_t2_bin"] = pd.cut(df["delta_t2"], bins=bins_t2, include_lowest=True)

    heatmap_data = df.pivot_table(index="delta_t2_bin", columns="delta_t1_bin", values="sym_auc_kernel", aggfunc="mean", observed=False)

    count_data = df.pivot_table(index="delta_t2_bin", columns="delta_t1_bin", values="sym_auc_kernel", aggfunc="count", observed=False)
    annot_text = heatmap_data.map(lambda x: f"{x:.4f}" if pd.notna(x) else "") + "\n(n=" + count_data.map(lambda x: f"{int(x)}" if pd.notna(x) else "") + ")"

    plt.figure(figsize=(12, 3))
    ax=sns.heatmap(
        heatmap_data,
        annot=annot_text.values,
        fmt="",
        cmap="viridis",
        cbar_kws={'label': 'Mean symmetric auc'},
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
    plt.show()


def plot_linegraph_t2(df, animal="All Animals"):
    df = df.dropna(subset=["delta_t1", "delta_t2"])

    bins_t1 = np.array([0, 5, 15, 25, 35, 45, 80])
    #bins_t1 = np.array([0, 1, 2, 5, 10, 15, 20, 25, 30, 50, 80])
    #bins_t2 = np.array([15, df["delta_t2"].max()])

    df["delta_t1_bin"] = pd.cut(df["delta_t1"], bins=bins_t1, include_lowest=True)
    #df["delta_t2_bin"] = pd.cut(df["delta_t2"], bins=bins_t2, include_lowest=True)

    grouped = df.groupby("delta_t1_bin")["sym_auc_kernel"].agg(["mean", "std", "count"]).reset_index()

    bin_centers = bins_t1[:-1] + np.diff(bins_t1) / 2

    plt.figure(figsize=(7, 4))
    plt.errorbar(bin_centers, grouped["mean"], yerr=grouped["std"] / np.sqrt(grouped["count"]), 
                 fmt='o-', capsize=3, label="Mean ± SE", color="royalblue", markersize=4, linewidth=1)

    x_labels = [f"[{interval.left/10:.1f};{interval.right/10:.1f}]" for interval in grouped["delta_t1_bin"]]
    plt.xticks(bin_centers, labels=x_labels, rotation=45, ha='right', fontsize=6)
    plt.yticks(fontsize=6)  

    plt.xlabel("δt₁: Time between current and previous pulse (s)", fontsize=8)
    plt.ylabel("Mean symmetric AUC", fontsize=8)
    plt.title(f"{animal} (Linear Reg Deconvolution)", fontsize=8)
    plt.legend(fontsize=6)
    plt.tight_layout()
    plt.show()

def plot_linegraph_t2_by_animal(df):
    #df = df.dropna(subset=["delta_t1"])
    df = df.dropna(subset=["delta_t"])
    
    col = 'mean_original'
    df[f'{col}_z'] = (df[col] - df[col].mean()) / df[col].std(ddof=0)
    
    #bins_t1 = np.array([1, 2, 5, 15, 25, 35, 45, 80])
    bins_t1 = np.array([1, 15, 35, 146])
    bin_centers = bins_t1[:-1] + np.diff(bins_t1) / 2
    
    colors = ['black', 'grey', 'lightgrey']

    plt.figure(figsize=(7, 4))

    for i, animal in enumerate(df["animal"].unique()):
        df_animal = df[df["animal"] == animal].copy()
        #df_animal["delta_t1_bin"] = pd.cut(df_animal["delta_t1"], bins=bins_t1, include_lowest=True)
        df_animal["delta_t1_bin"] = pd.cut(df_animal["delta_t"], bins=bins_t1, include_lowest=True)

        #grouped = df_animal.groupby("delta_t1_bin")["sym_auc_kernel"].agg(["mean", "std", "count"]).reset_index()
        #grouped = df_animal.groupby("delta_t1_bin")["max_resp"].agg(["mean", "std", "count"]).reset_index()
        grouped = df_animal.groupby("delta_t1_bin")["mean_original_z"].agg(["mean", "std", "count"]).reset_index()
        grouped["std_error"] = grouped["std"] / np.sqrt(grouped["count"])

        plt.errorbar(bin_centers, grouped["mean"], yerr=grouped["std_error"],
                     fmt='o-', capsize=3, label=f"{animal}", markersize=3, linewidth=1, color=colors[i % len(colors)])

    x_labels = [f"[{bins_t1[i]/10};{bins_t1[i+1]/10}[" for i in range(len(bins_t1)-1)]
    plt.xticks(bin_centers, x_labels, rotation=15, fontsize=6)
    plt.yticks(fontsize=6) 

    plt.xlabel("δt₁: Time between current and previous pulse (s)", fontsize=7)
    #plt.ylabel("Mean symmetric AUC", fontsize=7)
    plt.ylabel("Mean calcium signal", fontsize=7)
    plt.legend(fontsize=6)
    plt.show()


def plot_hist_codes_panel(df):
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
        auc_inh = df_filt[df_filt["event_resp"] == "inh"]["sym_auc_kernel"]

        _, bin_edges = np.histogram(auc_inh, bins=bins)

        ax = axes[i]
        ax.hist(auc_inh, bins=bin_edges, alpha=0.7, color="blue", edgecolor="black", linewidth=1.2)
        ax.set_title(f"Δt in [{t1_min/10}, {t1_max/10})", fontsize = 8)
        ax.set_ylabel("Freq", fontsize = 8)

    for i in range(n_bins, len(axes)):
        fig.delaxes(axes[i])

    axes[-1].set_xlabel("Sym AUC", fontsize = 8) 
    
    fig.suptitle("Linear Regression Decomposition", fontsize=12)

    plt.tight_layout()
    plt.show()


def plot_hist_line(df):
    plt.figure(figsize=(10, 6))
            
    data = df[df["event_resp"] == "inh"]["sym_auc_kernel"]
    
    sns.kdeplot(data, bw_adjust=0.5)
    
    plt.xlabel("Sym AUC")
    plt.ylabel("Density")
    plt.title("Repartiton of responses (HW1 - Lin Reg)")
    plt.tight_layout()
    plt.savefig("hist_line_lin_reg.png", dpi=300, bbox_inches="tight")
    plt.show()



def plot_boxplot(df):
    
    event_inh = df[df["event_resp"] == "inh"]["sym_auc_kernel"]
    event_bet = df[df["event_resp"] == "between"]["sym_auc_kernel"]
    event_exh = df[df["event_resp"] == "exh"]["sym_auc_kernel"]

    plt.figure(figsize=(10, 7))
    plt.boxplot([event_inh, event_bet, event_exh], tick_labels=["Full Inhalation", "Mix of Inh and Exh","Full Exhalation"])
    plt.ylabel("Symmetric AUC")
    plt.title("Symmetric AUC Distribution Across Sniff Cycle Phases")
    plt.show()

    
def plot_kernel_trends(df, animal="All Animals"):
    inh_kernels = np.vstack(df[df["event_resp"] == "inh"]["kernel"].dropna().values) 
    exh_kernels = np.vstack(df[df["event_resp"] == "exh"]["kernel"].dropna().values)  

    time_points = np.arange(inh_kernels.shape[1])  

    inh_mean = np.mean(inh_kernels, axis=0)
    inh_std = np.std(inh_kernels, axis=0)
    
    exh_mean = np.mean(exh_kernels, axis=0)
    exh_std = np.std(exh_kernels, axis=0)

    plt.figure(figsize=(10, 5))

    # inh
    plt.subplot(1, 2, 1)
    plt.plot(time_points, inh_mean, color="green", label="Mean Kernel (inh)", linewidth=2)
    plt.fill_between(time_points, inh_mean - inh_std, inh_mean + inh_std, color="green", alpha=0.2)
    plt.title(f"Kernels {animal} - Inhalation")
    plt.xlabel("Time")
    plt.ylabel("Kernel Response")
    plt.legend()

    #exh
    plt.subplot(1, 2, 2)
    plt.plot(time_points, exh_mean, color="red", label="Mean Kernel (exh)", linewidth=2)
    plt.fill_between(time_points, exh_mean - exh_std, exh_mean + exh_std, color="red", alpha=0.2)
    plt.title(f"Kernels {animal} - Exhalation")
    plt.xlabel("Time")
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_kernel(df, animal="All Animals"):
    kernels = np.vstack(df["kernel"].dropna().values) 

    time_points = np.arange(kernels.shape[1])  

    mean = np.mean(kernels, axis=0)
    std = np.std(kernels, axis=0)

    plt.figure(figsize=(5, 5))

    plt.plot(time_points, mean, color="black", label="Mean Kernel", linewidth=1)
    plt.fill_between(time_points, mean - std, mean + std, color="black", alpha=0.2)
    plt.title(f"Kernels {animal}")
    plt.xlabel("Time")
    plt.ylabel("Kernel Response")
    plt.legend()


    plt.tight_layout()
    plt.show()
    
def plot_auc_isolated_paired(df):
    isolated = df[df["delta_t1"] >= 20]["sym_auc_kernel"]  # 2s = 200 * 10ms
    paired = df[df["delta_t1"] < 20]["sym_auc_kernel"]

    means = [np.mean(isolated), np.mean(paired)]
    std_errors = [np.std(isolated) / np.sqrt(len(isolated)), np.std(paired) / np.sqrt(len(paired))]

    plt.figure(figsize=(8, 6))
    plt.errorbar(["Isolated Pulses (Δt>=2s)", "Paired Pulses (Δt<2s)"], means, yerr=std_errors, fmt='o-', capsize=5, markersize=8, label="Mean Sym AUC")
    
    plt.ylabel("Mean Symmetric AUC")
    plt.title("Effect of Pulse Timing on Symmetric AUC")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.show()
    

def plot_actual_vs_predicted_deviations(df):

    df['actual_peak'] = df['original_segment'].apply(lambda seg: np.max(seg))
    
    df['predicted_peak'] = df['kernel'].apply(lambda seg: np.max(seg))
    
    # deviation (actual - predicted)
    df['deviation'] = df['actual_peak'] - df['predicted_peak']
    
    df_inh = df[df['event_resp'] == 'inh']
    df_exh = df[df['event_resp'] == 'exh']
    
    #Inhalation events
    plt.figure(figsize=(7, 6))
    plt.scatter(df_inh['predicted_peak'], df_inh['deviation'],
                color='green', edgecolor='k', s=80, label='Inhalation')
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.xlabel('Predicted Response (from projected stimulus)')
    plt.ylabel('Deviation (Actual - Predicted)')
    plt.title('Deviation from Linear Prediction (Inhalation Events)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Exhalation events
    plt.figure(figsize=(7, 6))
    plt.scatter(df_exh['predicted_peak'], df_exh['deviation'],
                color='red', edgecolor='k', s=80, label='Exhalation')
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.xlabel('Predicted Response (from projected stimulus)')
    plt.ylabel('Deviation (Actual - Predicted)')
    plt.title('Deviation from Linear Prediction (Exhalation Events)')
    plt.legend()
    plt.grid(True)
    plt.show()



def plot_parametric_excursions(df, animal, event_type=None):
    if event_type is not None:
        event_types = [event_type]
    else:
        event_types = df['event_resp'].unique()

    for et in event_types:
        df_event = df[df['event_resp'] == et]
        
        unique_dt = np.sort(df_event['delta_t1'].dropna().unique())
        N = len(unique_dt)
        cmap = plt.get_cmap('viridis', N)
        color_map = {dt: cmap(i) for i, dt in enumerate(unique_dt)}
        
        fig, ax = plt.subplots(figsize=(8, 6))
        for idx, row in df_event.iterrows():
            predicted = np.array(row['kernel'])
            original = np.array(row['original_segment'])
            dt = row['delta_t1']
            color = 'black' if (dt is None or np.isnan(dt)) else color_map[dt]
            ax.plot(predicted, original, '-', color=color, alpha=0.7)
        
        ax.set_xlabel('Predicted Signal')
        ax.set_ylabel('Original Signal')
        ax.set_title(f'Parametric Excursions for {et} Events ({animal})')
        ax.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax.axvline(0, color='gray', linestyle='--', linewidth=1)
        ax.grid(True)

        boundaries = np.concatenate((unique_dt, [unique_dt[-1] + 1]))
        norm = mcolors.BoundaryNorm(boundaries, ncolors=N)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([]) 
        cbar = fig.colorbar(sm, ax=ax, ticks=unique_dt, label='delta_t1 (time points)')
        
        plt.tight_layout()
        plt.show()
        
        

def plot_parametric_excursions_all(df, animal, event_type=None):
    if event_type is not None:
        event_types = [event_type]
    else:
        event_types = df['event_resp'].unique()

    color_map = {'exh': 'red', 'inh': 'green'}

    for et in event_types:
        df_event = df[df['event_resp'] == et]
        plt.figure(figsize=(8, 6))
        for idx, row in df_event.iterrows():
            predicted = np.array(row['kernel'])
            original = np.array(row['original_segment'])
            color = color_map.get(et, 'black')
            plt.plot(predicted, original, '-', color=color, alpha=0.7)
        plt.xlabel('Predicted Signal')
        plt.ylabel('Original Signal')
        plt.title(f'Parametric Excursions for {et} Events ({animal})')
        plt.axhline(0, color='gray', linestyle='--', linewidth=1)
        plt.axvline(0, color='gray', linestyle='--', linewidth=1)
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def compute_adaptation_gain(df):

    results = []
    for idx, row in df.iterrows():
        original = np.array(row['original_segment'])
        predicted = np.array(row['kernel'])
        dt = row['delta_t1']
        
        if (len(original) == 0) or (len(predicted) == 0) or pd.isna(dt):
            continue
        
        actual_peak = np.max(original)
        predicted_peak = np.max(predicted)
        
        if predicted_peak == 0:
            gain = np.nan
        else:
            gain = actual_peak / predicted_peak
        
        results.append({'delta_t1': dt, 'gain': gain})
    
    return pd.DataFrame(results)

def analyze_adaptation_gain(df, animal):
    """
    quantify how gain (actual_peak / predicted_peak) depends on delta_t1.
    
    3 groups:
      - Group 1: 0 < delta_t1 <= 2 (colored red)
      - Group 2: 10 <= delta_t1 <= 11 (colored green)
      - Group 3: 20 <= delta_t1 <= 22 (colored blue)
    """
    gain_df = compute_adaptation_gain(df)
    gain_df.dropna(inplace=True)
    
    if gain_df.empty:
        print("No valid data to analyze.")
        return
    
    conditions = [
        ((gain_df["delta_t1"] > 0) & (gain_df["delta_t1"] <= 2)),
        ((gain_df["delta_t1"] >= 10) & (gain_df["delta_t1"] <= 12)),
        ((gain_df["delta_t1"] >= 20) & (gain_df["delta_t1"] <= 22))
    ]
    choices = ['(0,2]', '[10,12]', '[20,22]']
    gain_df['group'] = np.select(conditions, choices, default='Other')
    
    gain_df = gain_df[gain_df['group'] != 'Other']
    
    color_map = {'(0,2]': 'blue',
                 '[10,12]': 'green',
                 '[20,22]': 'yellow'}
    
    plt.figure(figsize=(6, 6))
    
    for grp, group_df in gain_df.groupby('group'):
        x = group_df['delta_t1']
        y = group_df['gain']
        color = color_map.get(grp, 'black')
        plt.scatter(x, y, color=color, alpha=0.8, label=grp)
    
    x_all = gain_df['delta_t1']
    y_all = gain_df['gain']
    coeffs = np.polyfit(x_all, y_all, 1)
    poly1d_fn = np.poly1d(coeffs)
    x_line = np.linspace(x_all.min(), x_all.max(), 100)
    y_line = poly1d_fn(x_line)
    
    y_pred = poly1d_fn(x_all)
    SS_res = np.sum((y_all - y_pred) ** 2)
    SS_tot = np.sum((y_all - np.mean(y_all)) ** 2)
    R2 = 1 - SS_res / SS_tot if SS_tot != 0 else np.nan
    
    plt.plot(x_line, y_line, color='black', linestyle='--', linewidth=2, 
             label=f'Global Regression (R²={R2:.2f})')
    
    plt.xlabel("δt₁ (time since previous pulse)")
    plt.ylabel("Gain (actual_peak / predicted_peak)")
    plt.title(f"Gain vs. δt₁ ({animal})")
    plt.legend()
    plt.tight_layout()
    plt.show()

def compute_loop_area(predicted, original):
    """
    area enclosed by the loop defined by (predicted, original)
    """
    predicted = np.asarray(predicted)
    original = np.asarray(original)
    x = np.concatenate([predicted, predicted[0:1]])
    y = np.concatenate([original, original[0:1]])
    area = 0.5 * np.abs(np.dot(x[:-1], y[1:]) - np.dot(x[1:], y[:-1]))
    return area

def compute_loop_geometry_metric(df):

    results = []
    for idx, row in df.iterrows():
        predicted = np.array(row['kernel'])
        original = np.array(row['original_segment'])
        dt = row['delta_t1']
        
        if len(predicted) == 0 or len(original) == 0 or pd.isna(dt):
            continue
        
        area = compute_loop_area(predicted, original)
        results.append({'delta_t1': dt, 'loop_area': area})
    
    return pd.DataFrame(results)

def analyze_adaptation_loop_geometry(df, animal):

    loop_df = compute_loop_geometry_metric(df)
    loop_df.dropna(inplace=True)
    
    if loop_df.empty:
        print("No valid data to analyze.")
        return
    
    conditions = [
        ((loop_df["delta_t1"] > 0) & (loop_df["delta_t1"] <= 2)),
        ((loop_df["delta_t1"] >= 10) & (loop_df["delta_t1"] <= 12)),
        ((loop_df["delta_t1"] >= 20) & (loop_df["delta_t1"] <= 22))
    ]
    choices = ['(0,2]', '[10,12]', '[20,22]']
    loop_df['group'] = np.select(conditions, choices, default='Other')
    
    loop_df = loop_df[loop_df['group'] != 'Other']
    
    color_map = {'(0,2]': 'blue',
                 '[10,12]': 'green',
                 '[20,22]': 'yellow'}
    
    plt.figure(figsize=(6, 6))
    
    for grp, group_df in loop_df.groupby('group'):
        x = group_df['delta_t1']
        y = group_df['loop_area']
        color = color_map.get(grp, 'black')
        plt.scatter(x, y, color=color, alpha=0.8, label=grp)
    
    x_all = loop_df['delta_t1']
    y_all = loop_df['loop_area']
    coeffs = np.polyfit(x_all, y_all, 1)
    poly1d_fn = np.poly1d(coeffs)
    x_line = np.linspace(x_all.min(), x_all.max(), 100)
    y_line = poly1d_fn(x_line)
    
    y_pred = poly1d_fn(x_all)
    SS_res = np.sum((y_all - y_pred)**2)
    SS_tot = np.sum((y_all - np.mean(y_all))**2)
    R2 = 1 - SS_res / SS_tot if SS_tot != 0 else np.nan
    
    plt.plot(x_line, y_line, color='black', linestyle='--', linewidth=2, label=f'Global Regression (R²={R2:.2f})')
    plt.xlabel("δt₁ (time since previous pulse)")
    plt.ylabel("Loop Area (geometric measure)")
    plt.title(f"Loop Area vs. δt₁ ({animal})")
    plt.legend()
    plt.tight_layout()
    plt.show()



def measure_prediction_accuracy_by_bin(df, animal, short_threshold=2, middle_theshold = 10, long_threshold=20):

    short_pred, short_actual = [], []
    middle_pred, middle_actual = [], []
    long_pred, long_actual = [], []

    for _, row in df.iterrows():
        dt = row['delta_t1']
        pred = np.array(row['kernel'])
        act = np.array(row['original_segment'])
        
        if len(pred) == 0 or len(act) == 0 or pd.isna(dt):
            continue
        
        if (dt > 0) & (dt <= short_threshold):
            short_pred.extend(pred)
            short_actual.extend(act)
        elif (dt >= middle_theshold) & (dt <= middle_theshold + 2):
            middle_pred.extend(pred)
            middle_actual.extend(act)
        elif (dt >= long_threshold) & (dt <= long_threshold + 2):
            long_pred.extend(pred)
            long_actual.extend(act)
            
    short_pred = np.array(short_pred)
    short_actual = np.array(short_actual)
    middle_pred = np.array(middle_pred)
    middle_actual = np.array(middle_actual)
    long_pred = np.array(long_pred)
    long_actual = np.array(long_actual)
    
    coeffs_short = np.polyfit(short_pred, short_actual, 1)
    poly_fn_short = np.poly1d(coeffs_short)
    x_short_line = np.linspace(short_pred.min(), short_pred.max(), 100)
    y_short_line = poly_fn_short(x_short_line)
    r_short, _ = pearsonr(short_pred, short_actual)
    r2_short = r_short**2
    
    coeffs_middle = np.polyfit(middle_pred, middle_actual, 1)
    poly_fn_middle = np.poly1d(coeffs_middle)
    x_middle_line = np.linspace(middle_pred.min(), middle_pred.max(), 100)
    y_middle_line = poly_fn_middle(x_middle_line)
    r_middle, _ = pearsonr(middle_pred, middle_actual)
    r2_middle = r_middle**2
    
    coeffs_long = np.polyfit(long_pred, long_actual, 1)
    poly_fn_long = np.poly1d(coeffs_long)
    x_long_line = np.linspace(long_pred.min(), long_pred.max(), 100)
    y_long_line = poly_fn_long(x_long_line)
    r_long, _ = pearsonr(long_pred, long_actual)
    r2_long = r_long**2
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.scatter(short_pred, short_actual, alpha=0.8, color = 'blue', label="Data")
    plt.plot(x_short_line, y_short_line, color='red', linestyle='--',
             linewidth=1, label=f"Fit (R² = {r2_short:.2f})")
    plt.xlabel("Predicted (Short bin)")
    plt.ylabel("Actual (Short bin)")
    plt.title(f"Short bin: δt₁ ∈ ]0,{short_threshold}]")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.scatter(middle_pred, middle_actual, alpha=0.8, color = 'green', label="Data")
    plt.plot(x_middle_line, y_middle_line, color='red', linestyle='--',
             linewidth=1, label=f"Fit (R² = {r2_middle:.2f})")
    plt.xlabel("Predicted (Middle bin)")
    plt.ylabel("Actual (Middle bin)")
    plt.title(f"Middle bin: δt₁ ∈ [{middle_theshold},{middle_theshold+2}]")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.scatter(long_pred, long_actual, alpha=0.8, color='yellow', label="Data")
    plt.plot(x_long_line, y_long_line, color='red', linestyle='--',
             linewidth=1, label=f"Fit (R² = {r2_long:.2f})")
    plt.xlabel("Predicted (Long bin)")
    plt.ylabel("Actual (Long bin)")
    plt.title(f"Long bin: δt₁ ∈ [{long_threshold},{long_threshold+2}]")
    plt.legend()
    plt.grid(True)
    
    plt.suptitle(animal)
    plt.tight_layout()
    plt.show()


def plot_max_kernel_vs_onset_conv(df, animal):
    df = df[df['onset_convolved'] < 6]
    
    plt.figure(figsize=(6, 6))
    plt.scatter(df["onset_convolved"].to_numpy(), df["max_resp"].to_numpy(), 
                alpha=0.8, color='grey', s=10)
    
    plt.yscale('log')
    
    plt.ylabel('Response Amplitude (log scale)')
    plt.xlabel('Stimulus Strength')
    plt.title(f'Response Amplitude VS Stimulus Strength ({animal})')
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.show()
    

def plot_avg_traces_for_intervals(df, data, animal, intervals, pre_s=1.0, post_s=3.0, fs_ca=10.0):

    ca = data["calcium_dict"][animal]
    if ca.ndim > 1:
        ca = ca.mean(axis=1)
    pre  = int(pre_s  * fs_ca)
    post = int(post_s * fs_ca)
    t = (np.arange(-pre, post) / fs_ca)

    plt.figure(figsize=(6,4))
    for dt_min, dt_max in intervals:
        dt_min_samps = dt_min * fs_ca
        dt_max_samps = dt_max * fs_ca

        sel = (
            (df["animal"] == animal)
            & (df["delta_t"] >= dt_min_samps)
            & (df["delta_t"] <  dt_max_samps)
        )
        onsets = df.loc[sel, "onset_resp"].values
        snippets = []
        for onset in onsets:
            start, end = onset - pre, onset + post
            if start < 0 or end > len(ca):
                continue
            snippets.append(ca[start:end])
        if not snippets:
            continue
        snippets = np.stack(snippets, 0)
        mean_trace = snippets.mean(0)
        sem_trace  = snippets.std(0) / np.sqrt(snippets.shape[0])

        label = f"{dt_min:.2f}–{dt_max:.2f}s (n={snippets.shape[0]})"
        plt.plot(t, mean_trace, label=label)
        plt.fill_between(t,
                         mean_trace - sem_trace,
                         mean_trace + sem_trace,
                         alpha=0.3)

    plt.axvline(0, color='k', linestyle='--', lw=0.5)
    plt.xlabel("time (seconds)")
    plt.ylabel("calcium")
    plt.title(f"{animal} - avg ca")
    plt.legend(fontsize=6)
    plt.tight_layout()
    plt.show()


def plot_avg_by_interpulse(df, data, animal, intervals, pre_s=1.0, post_s=5.0, fs_ca=10.0):

    ca = data["calcium_dict"][animal]
    if ca.ndim > 1:
        ca = ca.mean(axis=1)

    post = int(post_s * fs_ca)
    pre  = int(pre_s  * fs_ca)
    t    = np.arange(-pre, post) / fs_ca

    plt.figure(figsize=(6,4))
    colors = plt.cm.tab10(np.linspace(0, 1, len(intervals)))

    for (dt_min, dt_max), color in zip(intervals, colors):
        min_samps = int(dt_min * fs_ca)
        max_samps = int(dt_max * fs_ca)

        sel = (
            (df["animal"] == animal)
            & (df["delta_t"] >= 10)
            & (df["delta_plus_t"] >= min_samps)
            & (df["delta_plus_t"] <  max_samps)
        )
        onsets = df.loc[sel, "onset_resp"].values

        snippets = []
        for onset in onsets:
            start, end = onset - pre, onset + post
            # skip if window runs off either edge
            if start < 0 or end > len(ca):
                continue
            snippets.append(ca[start:end])

        arr     = np.stack(snippets, axis=0)
        mean_tr = arr.mean(axis=0)
        sem_tr  = arr.std(axis=0, ddof=0) / np.sqrt(arr.shape[0])

        label = f"{dt_min:.2f}–{dt_max:.2f}s (n={arr.shape[0]})"
        plt.plot(t, mean_tr, color=color, linewidth=1.5, label=label)
        plt.fill_between(
            t,
            mean_tr - sem_tr,
            mean_tr + sem_tr,
            color=color,
            alpha=0.3
        )
        """dt_center = (dt_min + dt_max) / 2.0
        plt.axvline(dt_center, color=color, linestyle='--', linewidth=1)"""
        
    plt.axvline(0, color='k', linestyle='--', lw=0.5)
    plt.xlabel("time since pulse i (s)")
    plt.ylabel("calcium")
    plt.title(animal)
    plt.legend(fontsize=6)
    plt.tight_layout()
    plt.show()



def plot_raw_traces_by_interpulse(
    df,
    data,
    animal,
    intervals,
    pre_s=1.0,
    post_s=3.0,
    fs_ca=10.0
):

    # load & flatten trace
    ca = data["calcium_dict"][animal]
    if ca.ndim > 1:
        ca = ca.mean(axis=1)

    pre  = int(pre_s  * fs_ca)
    post = int(post_s * fs_ca)
    t    = np.arange(-pre, post) / fs_ca

    fig, ax_raw = plt.subplots(figsize=(6,4))
    #ax_mean     = ax_raw.twinx()

    colors = plt.cm.tab10(np.linspace(0,1,len(intervals)))
    #colors = plt.cm.tab10(np.linspace(0,1,6))

    for (dt_min, dt_max), color in zip(intervals, colors):
        min_samps = int(dt_min * fs_ca)
        max_samps = int(dt_max * fs_ca)

        sel = (
            (df["animal"] == animal)
            & (df["delta_t"] >= 10)
            & (df["delta_plus_t"] >= min_samps)
            & (df["delta_plus_t"] <  max_samps)
            #& (df["event_resp"] == "inh")
        )
        onsets = df.loc[sel, "onset_resp"].values
        if len(onsets) == 0:
            continue

        snippets = []
        for onset in onsets:
            start, end = onset - pre, onset + post
            if start < 0 or end > len(ca):
                continue
            snippet = ca[start:end]
            snippets.append(snippet)
            ax_raw.plot(t, snippet, color=color, alpha=0.2)

        arr        = np.stack(snippets, axis=0)
        mean_tr    = arr.mean(axis=0)

        #ax_mean.plot(t, mean_tr, color=colors[0], linewidth=2, label=f"{dt_min:.2f}–{dt_max:.2f}s (n={arr.shape[0]})")
        ax_raw.plot(t, mean_tr, color=color, linewidth=2, label=f"{dt_min:.2f}–{dt_max:.2f}s (n={arr.shape[0]})")
        
        dt_center = (dt_min + dt_max) / 2.0
        plt.axvline(dt_center, color=color, linestyle='--', linewidth=1)


    ax_raw.axvline(0, color='k', linestyle='--', lw=0.5)
    ax_raw.set_xlabel("time since pulse i (s)")
    ax_raw.set_ylabel("raw calcium")
    #ax_mean.set_ylabel("mean calcium")

    #lines, labels = ax_mean.get_legend_handles_labels()
    #ax_raw.legend(lines, labels, fontsize=6, loc="upper right")
    plt.legend(fontsize=6)

    plt.title(animal)
    fig.tight_layout()
    plt.show()
    
def plot_deconvolved_traces(df, data, animal, intervals, pre_s=1.0, post_s=5.0, fs_ca=10.0):

    ca = data["calcium_dict"][animal]
    if ca.ndim > 1:
        ca = ca.mean(axis=1)

    post = int(post_s * fs_ca)
    pre  = int(pre_s  * fs_ca)
    t    = np.arange(-pre, post) / fs_ca

    plt.figure(figsize=(6,4))
    colors = plt.cm.tab10(np.linspace(0, 1, len(intervals)))
    
    min_iso = int(3.0*fs_ca)
    max_iso = int(14.6*fs_ca)
    
    sel = (
        (df["animal"]==animal)
        & (df["delta_t"] >= 10)
        & (df["delta_plus_t"] >= min_iso)
        & (df["delta_plus_t"] < max_iso)
    )
    snippets =[]
    for onset in df.loc[sel, "onset_resp"].values:
        start = onset - pre 
        end = onset + post 
        if start < 0 or end > len(ca):
            continue
        snippets.append(ca[start:end])

    mean_iso = np.stack(snippets, axis=0).mean(axis=0)
    
    print(f"mean_iso: {mean_iso}")
    
    for (dt_min, dt_max), color in zip(intervals, colors):
        min_samps = int(dt_min * fs_ca)
        max_samps = int(dt_max * fs_ca)

        sel = (
            (df["animal"] == animal)
            & (df["delta_t"] >= 10)
            & (df["delta_plus_t"] >= min_samps)
            & (df["delta_plus_t"] <  max_samps)
        )
        onsets = df.loc[sel, "onset_resp"].values

        snippets = []
        for onset in onsets:
            start, end = onset - pre, onset + post
            # skip if window runs off either edge
            if start < 0 or end > len(ca):
                continue
            snippets.append(ca[start:end])

        arr     = np.stack(snippets, axis=0)
        mean_tr = arr.mean(axis=0) - mean_iso
        sem_tr  = arr.std(axis=0, ddof=0) / np.sqrt(arr.shape[0])

        label = f"{dt_min:.2f}–{dt_max:.2f}s (n={arr.shape[0]})"
        plt.plot(t, mean_tr, color=color, linewidth=1.5, label=label)
        plt.fill_between(
            t,
            mean_tr - sem_tr,
            mean_tr + sem_tr,
            color=color,
            alpha=0.3
        )
        """dt_center = (dt_min + dt_max) / 2.0
        plt.axvline(dt_center, color=color, linestyle='--', linewidth=1)"""

    plt.axvline(0, color='k', linestyle='--', lw=0.5)
    plt.xlabel("time since pulse i (s)")
    plt.ylabel("calcium")
    plt.title(animal)
    plt.legend(fontsize=6)
    plt.tight_layout()
    plt.show()



def main():
    
    params = init_params()

    with open(params['out_path'], "rb") as f:
        df = pickle.load(f)
        
    print(df.head())

    with open(params["data_path"], "rb") as f:
        data = pickle.load(f)
            
    animal = "HW4"
    df_hw1 = df[df["animal"]==animal]
    #plot_max_kernel_vs_onset_conv(df_hw1, animal)
    
    #df_hw1 = df_hw1[(df_hw1["delta_t1"] > 20) & (df_hw1["delta_t1"] <= 30)]
    #df_hw1 = df_hw1[((df_hw1["delta_t1"] <= 2) & (df_hw1["delta_t1"] > 0)) | ((df_hw1["delta_t1"] <= 11) & (df_hw1["delta_t1"] >= 10)) | ((df_hw1["delta_t1"] >= 20) & (df_hw1["delta_t1"] <= 22))]
    df_hw1 = df_hw1[((df_hw1["delta_t1"] <= 2) & (df_hw1["delta_t1"] > 0)) | ((df_hw1["delta_t1"] >= 20) & (df_hw1["delta_t1"] <= 21))]
    #plot_parametric_excursions(df_hw1, animal, event_type="inh")
    #measure_prediction_accuracy_by_bin(df_hw1, animal, short_threshold=2, long_threshold=20)
    #analyze_adaptation_loop_geometry(df_hw1, animal)
    #analyze_adaptation_gain(df_hw1, animal)
    #plot_parametric_excursions_all(df_hw1, animal, event_type=None)
    
    #plot_actual_vs_predicted_deviations(df_hw1)
    #plot_kernel_trends(df_hw1)
    #plot_kernel(df_hw1)
    """plot_hist_line(df_hw1)
    plot_auc_isolated_paired(df)
    plot_hist_codes_panel(df)"""
    
    #plot_linegraph_t2_by_animal(df)
    intervals = [(0.2,0.5), (0.5,0.8), (0.8,1.1), (1.1, 1.4), (1.4, 1.7), (1.7, 2.0)]
                 #, (3.0, 14.6)]
    #0.2-0.5s, 0.5-0.8s, 0.8-1.1s
    
    plot_raw_traces_by_interpulse(df, data, "HW1", intervals)
    plot_deconvolved_traces(df, data, "HW1", intervals)
    plot_avg_by_interpulse(df, data, "HW4", intervals)
    plot_deconvolved_traces(df, data, "Sphinx", intervals)

    """intervals = [(3.0, 14.6)]
    plot_raw_traces_by_interpulse(df, data, "HW4", intervals)
    plot_avg_by_interpulse(df, data, "HW4", intervals)"""

    print(df[(df["delta_plus_t"] >=20) & (df["animal"]=="HW1")].shape)
    print(df[(df["delta_plus_t"] >=20) & (df["animal"]=="HW4")].shape)
    print(df[(df["delta_plus_t"] >=20) & (df["animal"]=="Sphinx")].shape)
    
    #plot_avg_traces_for_intervals(df, data, "HW4", intervals, pre_s=2.5, post_s=1.5, fs_ca=10.0)


    #1, 15, 35, 146
    #plot_linegraph_t2(df_hw1, animal)
    #plot_heatmap(df)
    #plot_heatmap_t2(df)

    """plot_heat_ca_time_pulses(df)
    plot_boxplot(df)"""

    """df_hw1 = df[df['animal']=='HW1']
    calcium_signal = data["calcium_dict"]["HW1"].mean(axis=1)
    signal_length = len(calcium_signal)
    reconstructed_signal = reconstruct_calcium_signal(df_hw1, signal_length)
    
    mse = mean_squared_error(calcium_signal, reconstructed_signal)
    corr_coef, p_value = pearsonr(calcium_signal, reconstructed_signal)
    r2 = r2_score(calcium_signal, reconstructed_signal)

    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print(f"Pearson Correlation Coefficient: {corr_coef:.6f}")
    print(f"Coefficient of Determination (R²): {r2:.6f}")
    
    plot_reconstructed_signal(calcium_signal, reconstructed_signal, df_hw1, 1000, 1250)"""


    
if __name__ == "__main__":
    main()

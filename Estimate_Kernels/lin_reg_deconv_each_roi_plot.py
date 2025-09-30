import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import pickle
import pandas as pd
import matplotlib.cm as cm


def init_params():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--out-path",
        type = str,
        help = "path to solve data",
        default= "sparseness/Data/linear_regression_each.pkl"
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
        "mean": np.mean(kernel[2:6]),
    }
    return metrics


def compute_number_preceding_pulses_in_bin(df, lower_bound, upper_bound):
    results = []
    
    for i, onset in enumerate(df["onset_resp"]):        
        current_event = df["event_resp"].iloc[i]
        
        start_idx = max(0, onset - upper_bound)
        end_idx = max(0, onset - lower_bound)

        #print(f'start index: {start_idx}')
        #print(f'end index: {end_idx}')

        preceding_events = df[(df["onset_resp"] >= start_idx) & (df["onset_resp"] < end_idx)]
        filtered_events = preceding_events[preceding_events["event_resp"] == current_event]

        number_events = len(filtered_events) + 1  
        #print(f'number events: {number_events}')
        results.append(number_events)

    return results


def plot_linegraph_t2(df, animal):

    df = df[df["animal"] == animal].dropna(subset=["delta_t1", "delta_t2"])

    bins_t1 = np.array([0, 1, 2, 5, 10, 15, 20, 25, 30, 50, 80])
    bins_t2 = np.array([15, df["delta_t2"].max()])

    df["delta_t1_bin"] = pd.cut(df["delta_t1"], bins=bins_t1, include_lowest=True)
    df["delta_t2_bin"] = pd.cut(df["delta_t2"], bins=bins_t2, include_lowest=True)

    rois = sorted(df["ROI"].unique())
    num_rois = len(rois)

    plt.rcParams.update({
        'font.size': 7,       # General font size
        'axes.titlesize': 7,  # Title font size
        'axes.labelsize': 6,  # Label font size
        'xtick.labelsize': 5, # X-axis tick labels
        'ytick.labelsize': 5, # Y-axis tick labels
        'legend.fontsize': 5  # Legend font size
    })

    cols = 3  # Number of columns in the subplot grid
    rows = (num_rois // cols) + (num_rois % cols > 0)  # Adjust rows dynamically

    fig, axes = plt.subplots(rows, cols, figsize=(10, rows * 2))
    axes = axes.flatten() if num_rois > 1 else [axes]

    for i, roi in enumerate(rois):
        df_roi = df[df["ROI"] == roi]
        grouped = df_roi.groupby("delta_t1_bin")["sym_auc_kernel"].agg(["mean", "std", "count"]).reset_index()
        bin_centers = bins_t1[:-1] + np.diff(bins_t1) / 2
        
        ax = axes[i]
        ax.errorbar(bin_centers, grouped["mean"], yerr=grouped["std"] / np.sqrt(grouped["count"]), 
                    fmt='o-', capsize=3, label=f"ROI {roi}", color="royalblue")
        
        x_labels = [f"[{int(interval.left)/10};{int(interval.right)/10}]" for interval in grouped["delta_t1_bin"]]
        ax.set_xticks(bin_centers)
        ax.set_xticklabels(x_labels, rotation=45, ha='right')

        ax.set_xlabel("Delta t1: Time between current and previous pulse (s)")
        ax.set_ylabel("Mean symmetric AUC")
        ax.set_title(f"ROI {roi}")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def plot_linegraph_t2_superimp(df, animal):
    df = df[df["animal"] == animal].dropna(subset=["delta_t1", "delta_t2"])

    bins_t1 = np.array([0, 1, 2, 5, 10, 15, 20, 25, 30, 50, 80])
    bins_t2 = np.array([15, df["delta_t2"].max()])

    df["delta_t1_bin"] = pd.cut(df["delta_t1"], bins=bins_t1, include_lowest=True)
    df["delta_t2_bin"] = pd.cut(df["delta_t2"], bins=bins_t2, include_lowest=True)

    rois = sorted(df["ROI"].unique())
    num_rois = len(rois)

    colors = cm.get_cmap("tab20", num_rois)

    plt.rcParams.update({
        'font.size': 10,       # General font size
        'axes.titlesize': 10,  # Title font size
        'axes.labelsize': 9,  # Label font size
        'xtick.labelsize': 8, # X-axis tick labels
        'ytick.labelsize': 8, # Y-axis tick labels
        'legend.fontsize': 8  # Legend font size
    })

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, roi in enumerate(rois):
        df_roi = df[df["ROI"] == roi]
        grouped = df_roi.groupby("delta_t1_bin")["sym_auc_kernel"].agg(["mean", "std", "count"]).reset_index()
        bin_centers = bins_t1[:-1] + np.diff(bins_t1) / 2
        
        ax.errorbar(bin_centers, grouped["mean"], yerr=grouped["std"] / np.sqrt(grouped["count"]), 
                    fmt='o-', capsize=3, label=f"ROI {roi}", color =colors(i))
        
    x_labels = [f"[{int(interval.left)/10};{int(interval.right)/10}]" for interval in grouped["delta_t1_bin"]]
    ax.set_xticks(bin_centers)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')

    ax.set_xlabel("Delta t1: Time between current and previous pulse (s)")
    ax.set_ylabel("Mean symmetric AUC")
    ax.set_title(f"All ROIs Superimposed")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_kernel_trends(df, animal):
    df_animal = df[df['animal'] == animal]
    
    rois = sorted(df_animal['ROI'].unique())

    plt.rcParams.update({
        'font.size': 9,       # General font size
        'axes.titlesize': 10,  # Title font size
        'axes.labelsize': 8,  # Label font size
        'xtick.labelsize': 7, # X-axis tick labels
        'ytick.labelsize': 7, # Y-axis tick labels
        'legend.fontsize': 7  # Legend font size
    })

    plt.figure(figsize=(8, 4))  

    inh_kernels = []
    exh_kernels = []

    for roi in rois:
        df_roi = df_animal[df_animal['ROI'] == roi]

        roi_inh_kernels = df_roi[df_roi["event_resp"] == "inh"]["kernel"].dropna().values
        roi_exh_kernels = df_roi[df_roi["event_resp"] == "exh"]["kernel"].dropna().values
        
        if len(roi_inh_kernels) > 0:
            inh_kernels.append(np.vstack(roi_inh_kernels))
        if len(roi_exh_kernels) > 0:
            exh_kernels.append(np.vstack(roi_exh_kernels))

    if inh_kernels:
        time_points = np.arange(inh_kernels[0].shape[1])
    elif exh_kernels:
        time_points = np.arange(exh_kernels[0].shape[1])
    else:
        print("No kernels found for the selected animal.")
        return

    # Inhalation plot
    plt.subplot(1, 2, 1)
    for roi_kernel in inh_kernels:
        mean_inh = np.mean(roi_kernel, axis=0)
        std_inh = np.std(roi_kernel, axis=0)
        plt.plot(time_points, mean_inh, alpha=0.7)
        plt.fill_between(time_points, mean_inh - std_inh, mean_inh + std_inh, alpha=0.2)
    
    plt.title(f"{animal} - Inhalation Kernels")
    plt.xlabel("Time (frames)")
    plt.ylabel("Kernel Response")
    plt.grid(True)

    # Exhalation plot
    plt.subplot(1, 2, 2)
    for roi_kernel in exh_kernels:
        mean_exh = np.mean(roi_kernel, axis=0)
        std_exh = np.std(roi_kernel, axis=0)
        plt.plot(time_points, mean_exh, alpha=0.7)
        plt.fill_between(time_points, mean_exh - std_exh, mean_exh + std_exh, alpha=0.2)
    
    plt.title(f"{animal} - Exhalation Kernels")
    plt.xlabel("Time (frames)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()



def plot_response_to_isolated(df):
    df_inh = df[df['event_resp'] == 'inh']

    mask = df_inh['onset_resp'].diff().fillna(20) >= 20
    df_iso = df_inh[mask]

    print("Number of isolated inhalation pulses:", len(df_iso)/12)

    # Group by ROI, compute mean and std, then sort by increasing mean
    group_stats = (
        df_iso.groupby("ROI")["sym_auc_kernel"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("mean")
    )

    plt.figure(figsize=(8, 6))
    plt.errorbar(
        group_stats["mean"],      # x-values (mean AUC)
        range(len(group_stats)),  # y-values (sorted positions)
        xerr=group_stats["std"],   # error bars (std dev)
        fmt="o",
        capsize=5
    )
    plt.xlabel("AUC Value")
    plt.ylabel("ROI")
    plt.yticks(range(len(group_stats)), group_stats["ROI"])
    plt.title("Mean Response for Isolated Inhalation Pulses (HW4)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def plot_auc_isolated_paired_by_roi(df):
    df_inh = df[df['event_resp'] == 'inh']

    # Split into isolated vs. paired
    df_iso = df_inh[df_inh["delta_t1"] >= 20]
    df_paired = df_inh[df_inh["delta_t1"] < 20]

    print("Number of isolated inhalation pulses:", len(df_iso)/12)
    print("Number of paired inhalation pulses:", len(df_paired)/12)

    grouped_iso = (
        df_iso.groupby("ROI")["sym_auc_kernel"]
        .agg(["mean", "count", "std"])
        .reset_index()
    )
    
    grouped_paired = (
        df_paired.groupby("ROI")["sym_auc_kernel"]
        .agg(["mean", "count", "std"])
        .reset_index()
    )

    merged_stats = pd.merge(
        grouped_iso, grouped_paired, on="ROI", suffixes=("_iso", "_paired")
    )

    merged_stats["sem_iso"] = merged_stats["std_iso"] / np.sqrt(merged_stats["count_iso"])
    merged_stats["sem_paired"] = (
        merged_stats["std_paired"] / np.sqrt(merged_stats["count_paired"])
    )

    merged_stats = merged_stats.sort_values("mean_iso")

    rois = merged_stats["ROI"]
    x = np.arange(len(rois))
    width = 0.4

    plt.figure(figsize=(10, 6))
    plt.bar(
        x - width/2,
        merged_stats["mean_iso"],
        yerr=merged_stats["sem_iso"],
        width=width,
        capsize=5,
        label="Isolated (Δt≥2s)"
    )
    plt.bar(
        x + width/2,
        merged_stats["mean_paired"],
        yerr=merged_stats["sem_paired"],
        width=width,
        capsize=5,
        label="Paired (Δt<2s)"
    )

    plt.xticks(x, rois, rotation=45)
    plt.ylabel("Symmetric AUC")
    plt.title("Comparison of Sym AUC for Isolated vs Paired Pulses by ROI (HW4)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_hist_codes_panel(df, roi):
    df = df[df["ROI"]==roi]
    bins_t1 = np.array([0, 20, 100])
    
    n_bins = len(bins_t1) - 1  
    n_cols = 2  
    n_rows = int(np.ceil(n_bins / n_cols)) 

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3), sharex=True, sharey=True)
    axes = axes.flatten() 

    bins = 10  

    for i in range(n_bins):
        t1_min, t1_max = bins_t1[i], bins_t1[i + 1]
        df_filt = df[(df["delta_t1"] < t1_max) & (df["delta_t1"] >= t1_min)]
        auc_inh = df_filt[df_filt["event_resp"] == "inh"]["sym_auc_kernel"]

        _, bin_edges = np.histogram(auc_inh, bins=bins)

        ax = axes[i]
        ax.hist(auc_inh, bins=bin_edges, alpha=0.7, color="blue", edgecolor="black", linewidth=1.2)
        ax.set_title(f"Δt in [{t1_min/10}, {t1_max/10})", fontsize=8)
        ax.set_ylabel("Freq", fontsize=8)
        
        ax.set_xlim(0, 0.5)

    for i in range(n_bins, len(axes)):
        fig.delaxes(axes[i])

    axes[-1].set_xlabel("Sym AUC", fontsize=8)
    
    fig.suptitle(f"Linear Regression Decomposition for ROI {roi}", fontsize=12)
    
    #plt.savefig(f"lin_reg_auc_hist_roi{roi}.png", dpi=300, bbox_inches="tight")
    plt.tight_layout()
    #plt.show()



def plot_hist_all_rois(df):
    plt.figure(figsize=(10, 6))
    
    rois = sorted(df["ROI"].unique())
    
    colors = sns.color_palette("husl", 53)
    
    for idx, roi in enumerate(rois):
        data = df[(df["ROI"] == roi) & (df["event_resp"] == "inh")]["sym_auc_kernel"]
        
        sns.kdeplot(data, bw_adjust=0.5, label=f"ROI {roi}", color=colors[idx])
    
    plt.xlabel("Sym AUC")
    plt.ylabel("Density")
    plt.title("Repartiton of responses across ROIs (HW4 - Lin Reg)")
    plt.xlim(0, 0.5)
    plt.legend(title="ROI")
    plt.tight_layout()
    #plt.savefig("hist_all_rois_lin_reg.png", dpi=300, bbox_inches="tight")
    plt.show()


def main():

    params = init_params()

    with open(params['out_path'], "rb") as f:
        df = pickle.load(f)
        
    df_hw1 = df[df['animal']=='HW4']
    print(df_hw1)
    
    animal = "HW4"
    plot_response_to_isolated(df_hw1)
    plot_auc_isolated_paired_by_roi(df_hw1)

    plot_hist_all_rois(df_hw1)
    """for i in range(12):
        plot_hist_codes_panel(df_hw1, i)
    plot_linegraph_t2(df, animal)
    plot_linegraph_t2_superimp(df, animal)
    plot_kernel_trends(df, animal)"""


    
if __name__ == "__main__":
    main()

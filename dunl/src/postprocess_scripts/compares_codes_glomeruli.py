import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import pickle
import pandas as pd
from matplotlib.ticker import MaxNLocator
import torch
import os
from scipy.stats import linregress

import sys

sys.path.append(r"dunl-compneuro\src")
sys.path.append("")

import model

def init_params():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--res-path",
        type = str,
        help = "path to solve data",
        default=[f"sparseness/results/supervised_roi{i}_HW1" for i in range(12)]
        #[f"sparseness/results/supervised_roi{i}_HW1" for i in range(12)]
        #[f"sparseness/results/supervised_roi{i}_Sphinx" for i in range(19)]
        #[f"sparseness/results/supervised_roi{i}_HW4" for i in range(53)]
        
    )               
    parser.add_argument(
        "--path",
        type=str,
        help="path",
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
        default=(8, 2),
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



def compute_number_preceding_pulses_in_bin(df, lower_bound, upper_bound):
    results = []
    
    for i, onset in enumerate(df["onset_resp"]):        
        current_event = df["event_resp"].iloc[i]
        
        start_time = onset - upper_bound
        end_time = onset - lower_bound

        preceding_events = df[(df["onset_resp"] >= start_time) & (df["onset_resp"] < end_time)]
        filtered_events = preceding_events[preceding_events["event_resp"] == current_event]

        if not filtered_events.empty:
            last_pulse = filtered_events.sort_values("onset_resp").iloc[-1]
            if last_pulse["delta_t1"] <= 20:
                results.append(np.nan)
                continue

        number_events = len(filtered_events) 
        results.append(number_events)

    return results


def plot_preceding_train_vs_codes(df, animal):

    df = df[df['event_resp'] == 'inh']

    roi_columns = [col for col in df.columns if col.startswith("codes_")]
    
    lower_bound = 0
    upper_bound = 30

    df["preceding_train"] = compute_number_preceding_pulses_in_bin(df, lower_bound, upper_bound)

    plt.figure(figsize=(10, 6))

    colors = plt.cm.viridis(np.linspace(0, 1, len(roi_columns)))

    for i, roi in enumerate(roi_columns):
        grouped = df.groupby("preceding_train")[roi].agg(["mean", "count", "std"]).reindex(range(0, 3))
        print(grouped)
        #plt.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'], fmt='o-', capsize=5, label=roi)
        plt.plot(grouped.index, grouped['mean'], 'o-', color=colors[i])
        
        valid_points = grouped['mean'].dropna()
        if not valid_points.empty:
            last_index = valid_points.index[-1]
            last_value = valid_points.iloc[-1]
            plt.text(last_index + 0.1, last_value, roi, color=colors[i], fontsize=10, va='center')

    plt.title(f'Avg codes vs. Number of preceding pulses ({animal})')
    plt.xlabel(f'Number of pulses in a preceding window of {upper_bound/10}s')
    plt.ylabel('Avg code')

    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show()


def plot_response_to_isolated(df, animal):
    df_inh = df[df["event_resp"] == "inh"]
    
    # "Isolated" inhalation pulses: consecutive onset_resp separated by >= 20
    mask = df_inh["onset_resp"].diff().fillna(20) >= 20
    df_iso = df_inh[mask]
    
    print("Number of isolated inhalation pulses:", len(df_iso))
    
    code_columns = [col for col in df.columns if col.startswith("codes_")]
    
    mean_codes = df_iso[code_columns].mean()
    std_codes = df_iso[code_columns].std()
    
    sorted_indices = mean_codes.sort_values().index
    sorted_means = mean_codes.loc[sorted_indices]
    sorted_stds  = std_codes.loc[sorted_indices]
    
    plt.figure(figsize=(8, 6))
    plt.errorbar(
        x=sorted_means,
        y=range(len(sorted_means)),
        xerr=sorted_stds,
        fmt="o",
        capsize=5,
    )
    
    plt.xlabel("Code Value")
    plt.ylabel("Glomerulus")
    plt.yticks(range(len(sorted_means)), sorted_indices)
    plt.title(f"Mean Glomerulus Code Response for Isolated Inhalation Pulses ({animal})")
    plt.tight_layout()
    plt.show()

    

def process_array(arr, kernel):
    """if (np.signbit(arr[0])):
        print(-arr)
        return -arr"""
    if (kernel[5]<0):
        return -arr
    else:
        return arr



def plot_auc_isolated_paired_by_roi(df, animal):
    df_inh = df[df["event_resp"] == "inh"]

    df_iso = df_inh[df_inh["delta_t1"] >= 20]
    df_paired = df_inh[df_inh["delta_t1"] < 20]

    code_columns = [col for col in df.columns if col.startswith("codes_")]

    mean_codes_iso = df_iso[code_columns].mean()
    std_codes_iso  = df_iso[code_columns].std()
    
    mean_codes_paired = df_paired[code_columns].mean()
    std_codes_paired  = df_paired[code_columns].std()

    max_vals_for_sort = []
    for col in code_columns:
        max_val = max(mean_codes_iso[col], mean_codes_paired[col])
        max_vals_for_sort.append(max_val)
    
    sort_series = pd.Series(max_vals_for_sort, index=code_columns).sort_values()
    sorted_code_columns = sort_series.index

    iso_means_sorted   = mean_codes_iso[sorted_code_columns]
    iso_stds_sorted    = std_codes_iso[sorted_code_columns]
    paired_means_sorted = mean_codes_paired[sorted_code_columns]
    paired_stds_sorted  = std_codes_paired[sorted_code_columns]

    x = np.arange(len(sorted_code_columns))
    width = 0.4
    
    cmap = plt.get_cmap('viridis', 4)
    paired_color = cmap(0)  # first color in the map
    isolated_color  = cmap(2)  # second color in the map

    plt.figure(figsize=(10, 6))
    plt.bar(
        x - width/2,
        iso_means_sorted,
        yerr=iso_stds_sorted,
        width=width,
        capsize=5,
        label="Isolated (Δt≥2s)",  
        color=isolated_color,                     
        ecolor='black',         
        alpha=0.8,     
        error_kw=dict(alpha=0.4)
    )
    plt.bar(
        x + width/2,
        paired_means_sorted,
        yerr=paired_stds_sorted,
        width=width,
        capsize=5,
        label="Paired (Δt<2s)",
        color=paired_color,
        ecolor='black',         
        alpha=0.8,     
        error_kw=dict(alpha=0.4)
    )

    plt.xticks(x, sorted_code_columns, rotation=45)
    plt.ylabel("Code")
    plt.title(animal)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_iso_paired_ratio(df, animal):
    
    df_inh = df[df["event_resp"] == "inh"]
    df_iso = df_inh[df_inh["delta_t1"] >= 20]
    df_paired = df_inh[df_inh["delta_t1"] < 20]

    code_columns = [col for col in df.columns if col.startswith("codes_")]

    mean_codes_iso = df_iso[code_columns].mean()
    mean_codes_paired = df_paired[code_columns].mean()

    # Sort ROIs by paired response
    sorted_cols = mean_codes_paired.sort_values().index

    iso = mean_codes_iso[sorted_cols]
    paired = mean_codes_paired[sorted_cols]

    # Compute log2 ratio safely
    epsilon = 0
    log_ratio = np.log2((iso + epsilon) / (paired + epsilon))

    x = np.arange(len(sorted_cols))

    plt.figure(figsize=(8, 4))

    for i in x:
        # Use .iloc for future-proofing
        plt.plot([i, i], [iso.iloc[i], paired.iloc[i]], color='gray', linewidth=1)
        y_pos = max(iso.iloc[i], paired.iloc[i]) + 0.02
        ratio_value = log_ratio.iloc[i]

        fontweight = 'bold' if ratio_value > 0 else 'normal'

        plt.text(
            i,
            y_pos,
            f"{ratio_value:.2f}",
            fontsize=8,
            ha='center',
            color='black',
            alpha=0.8,
            fontweight=fontweight
        )

    plt.scatter(x, iso, label='Isolated (Δt≥2s)', color='green', s=40)
    plt.scatter(x, paired, label='Paired (Δt<2s)', color='purple', s=40)

    plt.xticks(x, sorted_cols, rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    plt.ylabel("Code")
    plt.title(animal)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_hist_all_rois(df, animal):
    plt.figure(figsize=(10, 6))
    
    # rois = sorted(df["ROI"].unique())
    code_columns = [col for col in df.columns if col.startswith("codes_")]
    
    colors = sns.color_palette("husl", 53)
    
    for idx, roi in enumerate(code_columns):
        data = df[df["event_resp"] == "inh"][f"codes_{idx}"]
        
        sns.kdeplot(data, bw_adjust=0.5, label=f"ROI {roi}", color=colors[idx])
    
    plt.xlabel("Codes")
    plt.ylabel("Density")
    plt.title(f"Repartiton of responses across ROIs ({animal} - DUNL)")
    plt.legend(title="ROI")
    plt.tight_layout()
    plt.savefig("hist_all_rois_DUNL.png", dpi=300, bbox_inches="tight")
    plt.show()
    


def plot_linegraph_isolated_paired(df, animal):
    df = df.copy().dropna(subset=['delta_t1'])
    
    df['pulse_type'] = df['delta_t1'].apply(lambda x: 'isolated' if x >= 20 else 'paired')
    
    code_cols = [col for col in df.columns if col.startswith("codes_")]
    
    x_positions = [0, 1]
    pulse_labels = ['Isolated (Δt1 ≥ 20)', 'Paired (Δt1 < 20)']
    
    plt.figure(figsize=(8, 6))
    
    cmap = plt.get_cmap("viridis")
    n_rois = len(code_cols)
    
    for i, col in enumerate(code_cols):
        color = cmap(i / (n_rois - 1)) if n_rois > 1 else cmap(0.5)
        
        grouped = df.groupby("pulse_type")[col].agg(['mean', 'std', 'count'])
        grouped = grouped.reindex(['isolated', 'paired'])
        means = grouped['mean']
        stds = grouped['std']
        counts = grouped['count']
        sem = stds / np.sqrt(counts)
        
        plt.errorbar(x_positions, means, yerr=sem, marker='o', linestyle='-', capsize=5, label=col, color=color)
    
    plt.xticks(x_positions, pulse_labels)
    plt.xlabel("Pulse Type")
    plt.ylabel("Codes")
    plt.title(f"Codes for Isolated vs. Paired Pulses (by ROI) for {animal}")
    #plt.legend(title="ROI")
    plt.show()


def melt_codes_df(df):
    code_cols = [col for col in df.columns if col.startswith("codes_")]
    df_melted = df.melt(
        id_vars=['onset_resp', 'event_resp', 'median_phase', 'delta_t1', 'delta_t2', 'animal'],
        value_vars=code_cols,
        var_name='glomerulus',
        value_name='code_response'
    )
    return df_melted


def compute_binned_mean_matrix(df_melted, num_bins=30):
    df_melted['phase_bin'] = pd.cut(df_melted['median_phase'], bins=num_bins)
    df_melted['phase_bin_center'] = df_melted['phase_bin'].apply(lambda x: x.mid)

    stats = (
        df_melted.groupby(['glomerulus', 'phase_bin_center'])['code_response']
        .mean()
        .reset_index()
    )

    heatmap_data = stats.pivot(index='glomerulus', columns='phase_bin_center', values='code_response')

    glom_order = heatmap_data.mean(axis=1).sort_values().index
    heatmap_data = heatmap_data.loc[glom_order]

    return heatmap_data


def plot_heatmap(heatmap_data, animal):
    plt.figure(figsize=(8, 5))
    sns.heatmap(heatmap_data, cmap="viridis", cbar_kws={'label': 'Mean Response Amplitude'})
    plt.xlabel("Sniff Phase")
    plt.ylabel("Glomerular Subpopulation")
    plt.title(animal)
    plt.xticks([1, 15, 29], ["0", "π", "2π"])
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.show()

def compute_normalized_tuning(df_melted, num_bins=30):
    df_melted['phase_bin'] = pd.cut(df_melted['median_phase'], bins=num_bins)
    df_melted['phase_bin_center'] = df_melted['phase_bin'].apply(lambda x: x.mid)

    stats = (
        df_melted.groupby(['glomerulus', 'phase_bin_center'])['code_response']
        .mean()
        .reset_index()
    )

    tuning_matrix = stats.pivot(index='glomerulus', columns='phase_bin_center', values='code_response')
    tuning_norm = tuning_matrix.div(tuning_matrix.max(axis=1), axis=0)
    return tuning_norm


def get_activation_windows(tuning_norm, threshold=0.5):
    phase_bins = np.array(tuning_norm.columns)
    activation_lines = []

    for glom, row in tuning_norm.iterrows():
        active_mask = row > threshold
        if active_mask.any():
            active_phases = phase_bins[active_mask]
            start = active_phases.min()
            end = active_phases.max()
            width = end - start
            activation_lines.append((glom, start, end, width))

    activation_lines = sorted(activation_lines, key=lambda x: x[3]) 
    return activation_lines


def plot_activation_windows(activation_lines, animal):
    plt.figure(figsize=(8, 5))

    for i, (glom, start, end, width) in enumerate(activation_lines):
        plt.hlines(y=i, xmin=start, xmax=end, color='teal', linewidth=2)
        plt.text(end + 0.1, i, glom, va='center', fontsize=7, alpha=0.6)

    plt.xlabel("Sniff Phase")
    plt.ylabel("Glomerular Subpopulations")
    plt.title(animal)
    plt.yticks([])
    plt.xticks([0, np.pi, 2 * np.pi], ["0", "π", "2π"])
    plt.xlim(0, 2 * np.pi)
    plt.grid(axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()


def compute_mean_amplitude(df_melted):
    return (
        df_melted.groupby('glomerulus')['code_response']
        .mean()
        .rename("mean_amplitude")
    )

def get_activation_metrics(tuning_norm, threshold):
    phase_bins = np.array(tuning_norm.columns)
    num_bins = len(phase_bins)
    phase_width = 2 * np.pi
    bin_width = phase_width / num_bins

    activation_info = []

    for glom, row in tuning_norm.iterrows():
        active_mask = row > threshold
        if active_mask.any():
            active_phases = phase_bins[active_mask]
            width = active_phases.max() - active_phases.min()
            activation_info.append((glom, width))

    activation_df = pd.DataFrame(activation_info, columns=["glomerulus", "activation_width"])
    return activation_df.set_index("glomerulus")


def plot_width_vs_amplitude(activation_df, amplitude_df, animal):
    merged = activation_df.join(amplitude_df)

    # Linear regression
    slope, intercept, r_value, p_value, std_err = linregress(
        merged['activation_width'], merged['mean_amplitude']
    )

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=merged,
        x="activation_width",
        y="mean_amplitude",
        s=60,
        color='teal'
    )
    x_vals = np.linspace(0, np.pi, 100)
    plt.plot(x_vals, intercept + slope * x_vals, color='gray', linestyle='--')
    
    plt.xlabel("Activation Width (rad)")
    plt.ylabel("Mean Code")
    plt.title(animal)
    plt.text(0.05, 0.95, f"$R^2$ = {r_value**2:.3f}\n$p$ = {p_value:.3g}", transform=plt.gca().transAxes,
            fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round", fc="white", ec="gray"))
    plt.tight_layout()
    plt.show()
  
    
    
def compute_response_variability(df_melted, method='cv'):
    """
    Compute trial-to-trial variability (variance or coefficient of variation) for each glomerulus.
    """
    grouped = df_melted.groupby('glomerulus')['code_response']
    
    if method == 'var':
        variability = grouped.var()
    elif method == 'cv':
        variability = grouped.std() / grouped.mean()
    else:
        raise ValueError("method must be 'var' or 'cv'")
    
    variability = variability.rename("response_variability")
    return variability

def merge_with_metadata(df_melted, variability_df):
    # Median phase across trials for each glomerulus
    phase_mean = df_melted.groupby('glomerulus')['median_phase'].mean()
    
    # Animal info (one per glomerulus)
    animal_info = df_melted.groupby('glomerulus')['animal'].first()
    
    merged = pd.concat([variability_df, phase_mean.rename("mean_phase"), animal_info.rename("animal")], axis=1)
    return merged


def plot_variability_by_glomerulus(variability_df, animal):
    variability_df_sorted = variability_df.sort_values('response_variability')
    plt.figure(figsize=(6, 3))
    plt.bar(variability_df_sorted.index, variability_df_sorted['response_variability'], width=0.6)
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(fontsize=6)
    plt.ylabel("Response Variability", fontsize=7)
    plt.title(f"Trial-to-Trial Response Variability by Glomerulus for {animal}", fontsize=7)
    plt.tight_layout()
    plt.show()
    
    
def plot_glomerular_response_heatmap(df, animal, bins=None, normalize=None, cmap='viridis'):

    df = df.copy()

    df = df.dropna(subset=['delta_t1', 'code_response'])

    if bins is None:
        bins = np.array([1, 2, 5, 10, 20, 30, 50, 80])
    df['delta_t1_bin'] = pd.cut(df['delta_t1'], bins=bins)

    if normalize == 'zscore':
        df['code_norm'] = df.groupby('glomerulus')['code_response'].transform(
            lambda x: (x - x.mean()) / x.std(ddof=0)
        )
    elif normalize == 'minmax':
        df['code_norm'] = df.groupby('glomerulus')['code_response'].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6)
        )
    else:
        df['code_norm'] = df['code_response']

    heatmap_data = df.pivot_table(
        index='glomerulus',
        columns='delta_t1_bin',
        values='code_norm',
        aggfunc='mean'
    )

    mean_raw_codes = df.groupby('glomerulus')['code_response'].mean()

    heatmap_data = heatmap_data.loc[mean_raw_codes.sort_values().index]

    plt.figure(figsize=(10, 6))
    sns.heatmap(
        heatmap_data,
        cmap=cmap,
        cbar_kws={"label": "Code"},
        linewidths=0.5,
        linecolor='lightgray'
    )

    plt.xlabel("Δt₁ (Time since preceding pulse)")
    plt.ylabel("ROIs")
    plt.title(animal)
    plt.tight_layout()
    plt.show()


    
def main():
    print("Predict.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device is", device)

    # init parameters -------------------------------------------------------#
    print("init parameters.")
    params_init = init_params()

    # load whiffs-----------------------------------------------------------#

    with open(params_init["path"], "rb") as f:
        data = pickle.load(f)
        
    animals = ['HW1']

    all_results = []

    for animal in animals:
        valve = data["valve_dict"][animal] / 100
        phase = data["phase_peaks_dict"][animal]
        calcium_signal = data["calcium_dict"][animal].mean(axis=1)      

        valve_ts = np.arange(0, len(valve) / 1000, 0.001)  # 1 kHz sampling
        ca_ts = np.arange(0, len(calcium_signal) / 10, 0.1)       # 10 Hz sampling
        
        whiff_onsets = np.where(np.diff(valve) > 0)[0]

        # express valve in ca ref frame
        onset_resp = []
        event_resp = []
        median_phase= []
        for i in range(len(whiff_onsets)):
            start_idx = whiff_onsets[i]

            inh_points = np.sum((0 <= phase[start_idx+1:start_idx+51]) & (phase[start_idx+1:start_idx+51] < np.pi))

            if inh_points <= 30:
                current_event= "exh"
            else:
                current_event="inh"
            """if inh_points >= 49 :
                current_event= "inh"
            elif 49 > inh_points >= 25:
                current_event = "between"
            else:
                current_event="exh"""
            # index of the value  of resp_ts that is the closest to stim_ts[start_idx]
            index = np.absolute(ca_ts-valve_ts[start_idx]).argmin()
            
            inst_phase = np.median(phase[start_idx+1:start_idx+51])
            
            median_phase.append(inst_phase)
            onset_resp.append(index)
            event_resp.append(current_event)

        valve_down = np.zeros(len(calcium_signal))
        valve_down[onset_resp] = 1 
        
        for i, onset in enumerate(onset_resp):
                if event_resp[i] == "inh":
                    if i == 0:
                        delta_t1 = None
                        delta_t2 = None  
                    elif i == 1:
                        delta_t1 = onset_resp[i] - onset_resp[i - 1] if event_resp[i - 1] == "inh" else None
                        delta_t2 = None
                    else:
                        delta_t1 = onset_resp[i] - onset_resp[i - 1] if event_resp[i - 1] == "inh" else None
                        delta_t2 = onset_resp[i - 1] - onset_resp[i - 2] if event_resp[i - 1] == "inh" and event_resp[i - 2] == "inh" else None
                else:
                    delta_t1 = None
                    delta_t2 = None
                    
                all_results.append({
                    "onset_resp": onset,
                    "event_resp": event_resp[i],
                    "median_phase": median_phase[i],
                    "delta_t1": delta_t1,
                    "delta_t2": delta_t2,
                    "animal": animal, 
                })

    df = pd.DataFrame(all_results)
    df = df.iloc[:-2]
    
    # load codes ------------------------------------------------------#

    for idx, res_path in enumerate(params_init["res_path"]):
                
        # create folders -------------------------------------------------------#
        model_path = os.path.join(
            res_path,
            "model",
            "model_final.pt",
        )

        out_path = os.path.join(
            res_path,
            "figures",
        )
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        postprocess_path = os.path.join(
            res_path,
            "postprocess",
        )

        net = torch.load(model_path, map_location=device, weights_only=False)
        net.to(device)
        net.eval()

        kernel = np.squeeze(net.get_param("H").clone().detach().cpu().numpy())
        xhat = torch.load(
            os.path.join(postprocess_path, "xhat.pt")
        )
        codehat = xhat[0, 0, 0, :].clone().detach().cpu().numpy()
        codehat = process_array(codehat, kernel)   
        #code_selected = codehat[df["onset_resp"]]
        
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
        
        df[f"codes_{idx}"] = code_selected
            
    animal = 'HW1'
    df_melted = melt_codes_df(df)
    
    print(df_melted.head(10))
    
    #plot_glomerular_response_heatmap(df_melted, animal)

    """heatmap_data = compute_binned_mean_matrix(df_melted)
    plot_heatmap(heatmap_data, animal)
    
    tuning_norm = compute_normalized_tuning(df_melted, num_bins=50)
    activation_lines = get_activation_windows(tuning_norm, threshold=0.75)
    plot_activation_windows(activation_lines, animal)
    
    amplitude_df = compute_mean_amplitude(df_melted)
    activation_df = get_activation_metrics(tuning_norm, threshold=0.75)
    plot_width_vs_amplitude(activation_df, amplitude_df, animal)"""
    
    """variability_df = compute_response_variability(df_melted, method='var') 
    variability_df = merge_with_metadata(df_melted, variability_df)

    plot_variability_by_glomerulus(variability_df, animal)"""
    
    #plot_iso_paired_ratio(df, animal)

    """plot_auc_isolated_paired_by_roi(df, animal)
    plot_preceding_train_vs_codes(df, animal)
    plot_hist_all_rois(df, animal)
    plot_response_to_isolated(df, animal)"""
    
    plot_linegraph_isolated_paired(df, animal)
    

    
if __name__ == "__main__":
    main()

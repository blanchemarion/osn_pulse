import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import pickle
import pandas as pd
import torch
import os
from sklearn.feature_selection import mutual_info_classif

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
        #[f"sparseness/results/supervised_roi{i}_Sphinx" for i in range(20)]
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

    

def process_array(arr, kernel):
    """if (np.signbit(arr[0])):
        print(-arr)
        return -arr"""
    if (kernel[5]<0):
        return -arr
    else:
        return arr



def plot_codes_vs_delta_t1(df, animal, roi_list, figsize = (7,4)):
    
    df_a = df[(df['animal'] == animal) & df['delta_t1'].notna()].copy()
    if df_a.empty:
        raise ValueError(f"No data for animal '{animal}' with non-null delta_t1")
    
    bins_t1 = np.array([1, 2, 5, 15, 25, 35, 45, 80])
    df_a['delta_t1_bin'] = pd.cut(df_a['delta_t1'], bins=bins_t1, include_lowest=True)
    bin_centers = bins_t1[:-1] + np.diff(bins_t1)/2

    plt.figure(figsize=figsize)
    for roi in roi_list:
        col = f"codes_{roi}"
        if col not in df_a:
            raise KeyError(f"Column '{col}' not found in DataFrame")

        stats = (
            df_a
            .groupby('delta_t1_bin')[col]
            .agg(mean='mean', std='std', count='count')
            .reset_index()
        )
        stats['sem'] = stats['std'] / np.sqrt(stats['count'])

        plt.errorbar(bin_centers, stats['mean'], yerr=stats['sem'], fmt='o-', capsize=3, label=f"ROI {roi}", markersize=4, linewidth=1)

    x_labels = [f"[{bins_t1[i]:.0f};{bins_t1[i+1]:.0f}[" for i in range(len(bins_t1)-1)]
    plt.xticks(bin_centers, x_labels, rotation=45, fontsize=6)
    plt.yticks(fontsize=6)
    plt.xlabel("δt₁: Time between current and previous pulse (samples)", fontsize=7)
    plt.ylabel("Code value", fontsize=7)
    plt.title(f"{animal}: codes vs. Δt₁", fontsize=8)
    plt.legend(fontsize=6, ncol=2, loc='best')
    plt.tight_layout()

    plt.show()

def trimmed_mean(arr: np.ndarray, proportion: float = 0.1) -> float:
    """
    Compute the two‐sided trimmed mean of `arr`, cutting the lowest
    and highest `proportion` of values.
    """
    a = np.sort(arr)
    n = len(a)
    k = int(np.floor(proportion * n))
    # if array is too small to trim, just return the regular mean
    if 2*k >= n:
        return a.mean()
    return a[k : n-k].mean()


def plot_three_cluster_averages(df, animal, cluster1, cluster2, cluster3, figsize = (7,4)):
    
    df_a = df[(df['animal']==animal) & df['delta_t1'].notna()].copy()
    if df_a.empty:
        raise ValueError(f"No data for animal '{animal}' with valid delta_t1")
    
    all_rois = set(cluster1 + cluster2 + cluster3)
    for i in all_rois:
        col = f"codes_{i}"
        m = df_a[col].max()
        if m>0:
            df_a[col] = df_a[col] / m

    clusters = {
        'Cluster 1': cluster1,
        'Cluster 2': cluster2,
        'Cluster 3': cluster3
    }
    for name, rois in clusters.items():
        cols = [f"codes_{i}" for i in rois]
        missing = set(cols) - set(df_a.columns)
        if missing:
            raise KeyError(f"Columns not found for {name}: {missing}")
        df_a[name] = df_a[cols].median(axis=1)
        #df_a[name] = df_a[cols].apply(lambda row: trimmed_mean(row.values, proportion=0.1),axis=1)

    bins_t1 = np.array([1, 2, 5, 15, 25, 35, 45, 80])
    df_a['delta_t1_bin'] = pd.cut(df_a['delta_t1'], bins=bins_t1, include_lowest=True)
    bin_centers = bins_t1[:-1] + np.diff(bins_t1)/2
    
    
    """plt.figure(figsize=figsize)
    for name in clusters:
        stats = (df_a.groupby('delta_t1_bin')[name].agg(median='median', q1=lambda x: x.quantile(0.25), q3=lambda x: x.quantile(0.75), count='count').reset_index())
        stats['lower_err'] = stats['median'] - stats['q1']
        stats['upper_err'] = stats['q3'] - stats['median']
        plt.errorbar(bin_centers, stats['median'], yerr=[stats['lower_err'], stats['upper_err']], fmt='o-', capsize=4, label=name, markersize=5, linewidth=1)"""
    
    plt.figure(figsize=figsize)
    for name in clusters:
        stats = (df_a.groupby('delta_t1_bin')[name].agg(mean='mean', std='std', count='count').reset_index())
        stats['sem'] = stats['std'] / np.sqrt(stats['count'])
        plt.errorbar(bin_centers, stats['mean'], yerr=stats['sem'], fmt='o-', capsize=4, label=name, markersize=5, linewidth=1)

    x_labels = [f"[{bins_t1[i]:.0f};{bins_t1[i+1]:.0f}[" for i in range(len(bins_t1)-1)]
    plt.xticks(bin_centers, x_labels, rotation=45, fontsize=6)
    plt.yticks(fontsize=6)
    plt.xlabel("δt₁ (samples)", fontsize=7)
    plt.ylabel("Median code value", fontsize=7)
    plt.title(f"{animal}: Cluster‐median codes vs. Δt₁", fontsize=8)
    plt.legend(fontsize=6, loc='best', ncol=1)
    plt.tight_layout()

    plt.show()
    
def compute_mi_classif(df, cols, label_col, mapping, random_state = 0):
    
    X = df[cols].to_numpy()
    y = df[label_col].map(mapping).to_numpy()
    mi_nats = mutual_info_classif(
        X, y,
        discrete_features=False,
        random_state=random_state
    )
    return mi_nats / np.log(2)


def permutation_test_mi(df, code_cols, label_col, mapping, n_perms = 1000, random_state=0):
    mi_obs = compute_mi_classif(df, code_cols, label_col, mapping, random_state)
    # null distribution
    rng = np.random.RandomState(random_state)
    null_mis = np.zeros((n_perms, len(code_cols)))
    y_orig = df[label_col].map(mapping).to_numpy()
    X = df[code_cols].to_numpy()
    for i in range(n_perms):
        y_perm = rng.permutation(y_orig)
        mi_perm = mutual_info_classif(
            X, y_perm,
            discrete_features=False,
            random_state=rng
        ) / np.log(2)
        null_mis[i] = mi_perm
    # p-values
    p_vals = ((np.abs(null_mis) >= np.abs(mi_obs)).sum(axis=0) + 1) / (n_perms + 1)
    return pd.DataFrame({
        'MI_bits': mi_obs,
        'p_value': p_vals
    }, index=code_cols)


def cluster_level_mi(df, cluster_cols, label_col = 'period', mapping = {'start': 0, 'end': 1}, random_state = 0):
    mi = compute_mi_classif(df, cluster_cols, label_col, mapping, random_state)
    return pd.Series(mi, index=cluster_cols)


def sliding_window_mi(df, code_cols, label_col, mapping, window_size = 50, step = 10, random_state =0):
    results = []
    N = len(df)
    rng = np.random.RandomState(random_state)
    y_orig = df[label_col].map(mapping).to_numpy()
    X_full = df[code_cols].to_numpy()
    for start in range(0, N - window_size + 1, step):
        end = start + window_size
        X_win = X_full[start:end]
        y_win = y_orig[start:end]
        mi_win = mutual_info_classif(
            X_win, y_win,
            discrete_features=False,
            random_state=rng
        ) / np.log(2)
        results.append(
            dict(window_center=start + window_size // 2, **dict(zip(code_cols, mi_win)))
        )
    return pd.DataFrame(results)

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
        
    print(df.head())
        
    animal = 'HW1'
    
    code_cols = [col for col in df.columns if col.startswith("codes_")]
    K = 150
    
    df_valid = df[df['event_resp']=='inh'].reset_index(drop=True)
    df_valid['period'] = 'middle'
    df_valid.loc[:K-1, 'period'] = 'start'
    df_valid.loc[-K:, 'period'] = 'end'
    df_pe = df_valid[df_valid['period'].isin(['start','end'])]

    # Permutation test on glomeruli:
    perm_df = permutation_test_mi(df_pe, code_cols, 'period', {'start':0,'end':1}, n_perms=500)
    print(perm_df)

    # Cluster-level MI:
    #mi_clusters = cluster_level_mi(df_pe, ['Cluster 1','Cluster 2','Cluster 3'], 'period')

    # Time-resolved MI:
    sw_df = sliding_window_mi(df_valid, code_cols, 'period', {'start':0,'end':1}, window_size=50, step=10)
    sw_df.plot(x='window_center', y=code_cols, figsize=(8,4)); plt.ylabel('MI [bits]')
    plt.show()
    
    roi_list_hw1_c1 = [6, 7, 8, 10]
    roi_list_hw1_c2 = [3, 4, 9, 11]
    roi_list_hw1_c3 = [0, 1, 2, 5 ]
    
    roi_list_sphinx_c1 = [6, 7, 10, 14, 15, 17, 18, 19]
    roi_list_sphinx_c2 = [0, 2, 5, 8, 9, 12, 13, 16]
    roi_list_sphinx_c3 = [1, 3, 4, 11]
    
    #plot_three_cluster_averages(df, animal, roi_list_hw1_c1, roi_list_hw1_c2, roi_list_hw1_c3, figsize = (7,4))
    #plot_three_cluster_averages(df, animal, roi_list_sphinx_c1, roi_list_sphinx_c2, roi_list_sphinx_c3, figsize = (7,4))
    #plot_codes_vs_delta_t1(df, animal, roi_list_sphinx_c3, figsize = (7,4))    

    
if __name__ == "__main__":
    main()
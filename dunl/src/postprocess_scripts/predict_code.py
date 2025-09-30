import numpy as np
import pickle
import pandas as pd
import torch
import os
import sys
import statsmodels.api as sm
import matplotlib.pyplot as plt


sys.path.append(r"dunl-compneuro\src")
sys.path.append("")

import model

import numpy as np
import pandas as pd
import torch
import pickle
import os

def load_codes(res_paths, device):
    roi_codes = []
    for res_path in res_paths:
        model_path = os.path.join(res_path, "model", "model_final.pt")
        postprocess_path = os.path.join(res_path, "postprocess")

        net = torch.load(model_path, map_location=device, weights_only=False)
        net.to(device)
        net.eval()

        kernel = np.squeeze(net.get_param("H").detach().cpu().numpy())
        xhat = torch.load(os.path.join(postprocess_path, "xhat.pt"))
        codehat = xhat[0, 0, 0, :].detach().cpu().numpy()
        codehat = -codehat if kernel[5] < 0 else codehat

        roi_codes.append(codehat)
    return np.stack(roi_codes, axis=0)


def load_main_data(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def generate_feature_matrices(animal, data, roi_codes, sampling_rate=10):
    valve = data["valve_dict"][animal] / 100
    phase = data["phase_peaks_dict"][animal]
    calcium_signal = data["calcium_dict"][animal]

    valve_ts = np.arange(len(valve)) / 1000  # Valve timestamps at 1 kHz
    ca_ts = np.arange(calcium_signal.shape[0]) / sampling_rate  # Calcium timestamps

    whiff_onsets = np.where(np.diff(valve) > 0)[0]
    
    print(f'There are {len(whiff_onsets)} trials and {calcium_signal.shape} ROIs for {animal}')

    records = []
    
    last_inh_time = None
    inh_threshold = 0
    
    for i in range(len(whiff_onsets)):
        
        t0 = whiff_onsets[i]
        onset_time = valve_ts[t0]
        ca_idx = np.argmin(np.abs(ca_ts - onset_time))
        
        window_phase = phase[t0+1:t0+51]
        inh_points = np.sum((0 <= window_phase) & (window_phase < np.pi))
        current_event = "inh" if inh_points > inh_threshold else "exh"
        
        if current_event == "inh":
            if last_inh_time is not None:
                delta_inh = onset_time - last_inh_time
            else:
                delta_inh = np.nan
            last_inh_time = onset_time
        else:
            delta_inh = np.nan

        if i >= 2:
            t1, t2 = whiff_onsets[i-1], whiff_onsets[i-2]
            delta_t1 = valve_ts[t0] - valve_ts[t1]
            delta_t2 = valve_ts[t1] - valve_ts[t2]
            phase_current = np.median(phase[t0:t0+50])
            phase_prev = np.median(phase[t1:t1+50])
            phase_prev_prev = np.median(phase[t2:t2+50])

            for roi_idx, roi_code in enumerate(roi_codes):
                y_val = roi_code[ca_idx] if ca_idx < len(roi_code) else np.nan

                records.append({
                    "animal": animal,
                    "roi": roi_idx,
                    "delta_t1": delta_t1,
                    "delta_inh": delta_inh,
                    "delta_t1_sq": pow(delta_t1,2),
                    "delta_t2_sq": pow(delta_t2,2),
                    "delta_t2": delta_t2,
                    "phase_current": phase_current,
                    "phase_prev": phase_prev,
                    "phase_prev_prev": phase_prev_prev,
                    "t1_x_phase_current": delta_t1 * phase_current,
                    "t1_x_phase_prev": delta_t1 * phase_prev,
                    "t2_x_phase_prev_prev": delta_t2 * phase_prev_prev,
                    
                    "y": y_val
                })

    df_features = pd.DataFrame.from_records(records).dropna()
    return df_features


def fit_glm(X,y):
    X_const = sm.add_constant(X)  
    glm_model = sm.GLM(y, X_const, family=sm.families.Gaussian())
    result = glm_model.fit()

    print(result.summary())
    
    return result

def compute_regressor_contributions_by_roi(full_df, feature_cols, n_bootstrap=100):
    """
    For each ROI and each predictor, estimate its deviance contribution by bootstrapping:
      partial = (D_red - D_full) / D_null
    Returns:
      roi_list, contrib_mean, contrib_se
    """
    roi_list = sorted(full_df['roi'].unique())
    boot_vals = {f: {roi: [] for roi in roi_list} for f in feature_cols}

    for roi in roi_list:
        df_roi = full_df[full_df['roi'] == roi]
        X_full = df_roi[feature_cols].values
        y_full = df_roi['y'].values
        D_null = sm.GLM(y_full, sm.add_constant(np.ones_like(y_full)),
                        family=sm.families.Gaussian()).fit().null_deviance

        for _ in range(n_bootstrap):
            idxs = np.random.choice(len(df_roi), len(df_roi), replace=True)
            Xb = X_full[idxs]
            yb = y_full[idxs]
            mf = sm.GLM(yb, sm.add_constant(Xb), family=sm.families.Gaussian()).fit()
            D_full = mf.deviance
            for j, f in enumerate(feature_cols):
                Xr = np.delete(Xb, j, axis=1)
                mr = sm.GLM(yb, sm.add_constant(Xr), family=sm.families.Gaussian()).fit()
                D_red = mr.deviance
                boot_vals[f][roi].append((D_red - D_full) / D_null if D_null != 0 else np.nan)

    contrib_mean = {f: [np.nanmean(boot_vals[f][r]) for r in roi_list] for f in feature_cols}
    contrib_se = {f: [np.nanstd(boot_vals[f][r], ddof=1) for r in roi_list] for f in feature_cols}
    return roi_list, contrib_mean, contrib_se


def plot_regressor_contributions(roi_list, contrib_mean, contrib_se, animal):
    """
    Multi-panel errorbar plot: one subplot per feature showing each ROI's contribution.
    """
    features = list(contrib_mean.keys())
    n = len(features)
    n_cols = int(np.ceil(np.sqrt(n)))
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4 * n_cols, 3 * n_rows), sharex=True)
    axes = axes.flatten()

    for i, f in enumerate(features):
        ax = axes[i]
        ax.errorbar(roi_list, contrib_mean[f], yerr=contrib_se[f], fmt='o', capsize=4)
        ax.set_title(f)
        ax.set_ylabel('Frac. deviance')
        ax.label_outer()

    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    fig.suptitle(f'{animal}: Regressor Contributions per ROI', fontsize=16)
    axes[0].set_xlabel('ROI index')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()




def fit_glm_rois(full_df, X, y):
    X_const = sm.add_constant(X)  
    roi_results = {}
    for roi in full_df['roi'].unique():
        idx = full_df['roi'] == roi
        X_roi = X_const[idx]
        y_roi = y[idx]

        model_roi = sm.GLM(y_roi, X_roi, family=sm.families.Gaussian()).fit()
        roi_results[roi] = model_roi
        
    return roi_results


def evaluate_model_fit(result):
    deviance_explained = 1 - result.deviance / result.null_deviance
    print("Fraction of deviance explained:", deviance_explained)
    print(result.pvalues)


def coeff_plot(result, feature_cols):
    coefs = result.params[1:]  # Skip intercept
    errors = result.bse[1:]

    plt.bar(feature_cols, coefs, yerr=errors)
    plt.ylabel("Coefficient value")
    plt.title("Feature Influence")
    plt.xticks(rotation=45)
    plt.show()
    

def compare_rois(roi_results):
    deviance_explained_rois = {roi: 1 - m.deviance / m.null_deviance for roi, m in roi_results.items()}
    plt.bar(deviance_explained_rois.keys(), deviance_explained_rois.values())
    plt.ylabel("Fraction of Deviance Explained")
    plt.xlabel("ROI")
    plt.title("Explained Deviance Across ROIs")
    plt.show()
    


def compute_feature_deviance_by_roi(full_df, feature_cols):

    roi_list = sorted(full_df['roi'].unique())
    deviance_dict = {f: [] for f in feature_cols}
    se_dict = {f: [] for f in feature_cols}
    for roi in roi_list:
        df_roi = full_df[full_df['roi'] == roi]
        for f in feature_cols:
            X_feat = df_roi[[f]].values
            y_feat = df_roi['y'].values
            Xc = sm.add_constant(X_feat)
            m = sm.GLM(y_feat, Xc, family=sm.families.Gaussian()).fit()
            # deviance explained
            deviance_dict[f].append(1 - m.deviance / m.null_deviance)
            # standard error of coefficient (index 1)
            se_dict[f].append(m.bse[1])
    return roi_list, deviance_dict, se_dict


    
"""def plot_feature_deviance(roi_list, deviance_dict, animal):
    plt.figure(figsize=(10, 6))
    for feature, vals in deviance_dict.items():
        plt.plot(roi_list, vals, marker='o', label=feature)
    plt.xlabel('ROI index')
    plt.ylabel('Fraction of Deviance Explained')
    plt.title(f'{animal}: Feature-wise Deviance Explained by ROI')
    plt.legend()
    plt.tight_layout()
    plt.show()"""


def plot_feature_deviance(roi_list, deviance_dict, se_dict, animal):

    features = list(deviance_dict.keys())
    n_feats = len(features)

    n_cols = int(np.ceil(np.sqrt(n_feats)))
    n_rows = int(np.ceil(n_feats / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), sharex=True)
    axes = np.array(axes).reshape(-1)

    for idx, feature in enumerate(features):
        ax = axes[idx]
        dev_vals = deviance_dict[feature]
        err_vals = se_dict[feature]
        ax.errorbar(roi_list, dev_vals, yerr=err_vals, fmt='o', capsize=4)
        ax.set_ylabel('Deviance Explained')
        ax.set_title(feature)
        ax.label_outer()

    # Turn off unused axes
    for j in range(n_feats, len(axes)):
        axes[j].axis('off')

    fig.suptitle(f'{animal}: Feature-wise Deviance Explained by ROI', fontsize=16)
    axes[-1].set_xlabel('ROI index')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    

def main_test():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    params_init = {
        "res_path_sphinx": [f"sparseness/results/supervised_roi{i}_Sphinx" for i in range(20)],
        "res_path_hw1": [f"sparseness/results/supervised_roi{i}_HW1" for i in range(12)],
        "path": "sparseness/Data/animals_data_processed.pkl"
    }

    animal_choice = "Sphinx" 

    data = load_main_data(params_init["path"])

    paths = params_init[f"res_path_{animal_choice.lower()}"]
    roi_codes = load_codes(paths, device)
    df_features = generate_feature_matrices(animal_choice, data, roi_codes)

    """feature_cols = ["delta_t1", "delta_inh", "delta_t2", "delta_t1_sq", "delta_t2_sq", "phase_current", "phase_prev", "phase_prev_prev", 
                    "t1_x_phase_current", "t1_x_phase_prev", "t2_x_phase_prev_prev"]"""
    feature_cols = ["delta_t1", "delta_inh", "delta_t2", "phase_current", "phase_prev", "phase_prev_prev"]
    X = df_features[feature_cols].values
    y = df_features["y"].values

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    result = fit_glm(X, y)
    roi_results = fit_glm_rois(df_features, X, y)

    evaluate_model_fit(result)
    #coeff_plot(result, feature_cols)
    #compare_rois(roi_results)
    
    roi_list, deviance_dict, se_dict=compute_feature_deviance_by_roi(df_features, feature_cols)
    plot_feature_deviance(roi_list, deviance_dict, se_dict, animal_choice)

    return X, y

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    animal_choice = "HW1" 
    res_paths = {
        'Sphinx': [f"sparseness/results/supervised_roi{i}_Sphinx" for i in range(20)],
        'HW1':    [f"sparseness/results/supervised_roi{i}_HW1"    for i in range(12)]
    }[animal_choice]
    data_path = "sparseness/Data/animals_data_processed.pkl"
    feature_cols = ["delta_t1", "delta_inh", "delta_t2", "phase_current", "phase_prev", "phase_prev_prev"]

    data = load_main_data(data_path)
    roi_codes = load_codes(res_paths, device)
    full_df = generate_feature_matrices(animal_choice, data, roi_codes)

    X = full_df[feature_cols].values
    y = full_df['y'].values
    print("Fitting full GLM...")
    full_model = fit_glm(X, y)

    roi_list, dev_dict, contrib_se = compute_regressor_contributions_by_roi(full_df, feature_cols)
    plot_regressor_contributions(roi_list, dev_dict, contrib_se, animal_choice)
    


    return full_model, roi_list, dev_dict


if __name__ == "__main__":
    main()


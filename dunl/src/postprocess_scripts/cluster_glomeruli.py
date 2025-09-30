import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle
import pandas as pd
import torch
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon
from sklearn.metrics import silhouette_score
import seaborn as sns
from scipy.stats import kurtosis
from scipy.stats import skew
from scipy.stats import f_oneway

import sys

sys.path.append(r"dunl-compneuro\src")
sys.path.append("")

import model

def init_params():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--res-path-sphinx",
        type = str,
        help = "path to solve data",
        default=[f"sparseness/results/supervised_roi{i}_Sphinx" for i in range(20)]
    )      
    parser.add_argument(
        "--res-path-hw1",
        type = str,
        help = "path to solve data",
        default=[f"sparseness/results/supervised_roi{i}_HW1" for i in range(12)] 
    )     
    parser.add_argument(
        "--res-path-hw4",
        type = str,
        help = "path to solve data",
        default=[f"sparseness/results/supervised_roi{i}_HW4" for i in range(53)]
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

def melt_codes_df(df):
    code_cols = [col for col in df.columns if col.startswith("codes_")]
    
    df_melted = df.melt(
        id_vars=['onset_resp', 'event_resp', 'median_phase', 'delta_t1', 'delta_t2', 'animal'],
        value_vars=code_cols,
        var_name='code_column',      
        value_name='code_response'
    )

    #df_melted['glomerulus'] = df_melted['animal'] + "_" + df_melted['code_column']
    df_melted['glomerulus'] = df_melted['code_column']
    
    return df_melted


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


def fit_sinusoidal_model(tuning_norm):
    phase_bins = np.array(tuning_norm.columns)
    sin_features = np.sin(phase_bins)
    cos_features = np.cos(phase_bins)

    coef_list = []

    for glom, row in tuning_norm.iterrows():
        y = row.values
        if np.any(np.isnan(y)):  # <-- skip glomeruli with NaN
            continue

        X = np.column_stack([sin_features, cos_features])
        model = LinearRegression()
        model.fit(X, y)
        a, b = model.coef_
        c = model.intercept_
        coef_list.append((glom, a, b, c))

    return pd.DataFrame(coef_list, columns=['glomerulus', 'a', 'b', 'c']).set_index('glomerulus')


def compute_response_variability(df_melted, method='cv'):
    grouped = df_melted.groupby('glomerulus')['code_response']
    if method == 'var':
        variability = grouped.var()
    elif method == 'cv':
        variability = grouped.std() / grouped.mean()
    else:
        raise ValueError("method must be 'var' or 'cv'")
    return variability.rename("response_variability")


def compute_kurtosis(calcium_rois_by_animal): 
    records = []

    for animal, roi_traces in calcium_rois_by_animal.items():
        for i, trace in enumerate(roi_traces):
            glom_label = f"codes_{animal}_{i}"
            kurt = kurtosis(trace, fisher=False)
            records.append({'glomerulus': glom_label, 'kurtosis': kurt})

    return pd.DataFrame(records).set_index("glomerulus")

def compute_skewness(calcium_rois_by_animal):
    records = []

    for animal, roi_traces in calcium_rois_by_animal.items():
        for i, trace in enumerate(roi_traces):
            glom_label = f"codes_{animal}_{i}"
            skew_val = skew(trace)
            records.append({'glomerulus': glom_label, 'skewness': skew_val})

    return pd.DataFrame(records).set_index("glomerulus")


def compute_adaptation_ratio(df):
    df_inh = df[df["event_resp"] == "inh"]
    df_iso = df_inh[df_inh["delta_t1"] >= 20]
    df_paired = df_inh[df_inh["delta_t1"] < 20]

    code_columns = [col for col in df.columns if col.startswith("codes_")]

    iso_mean = df_iso[code_columns].mean()
    paired_mean = df_paired[code_columns].mean()
    
    epsilon=0
    ratio = (iso_mean + epsilon) / (paired_mean + epsilon)
    log_ratio = np.log2(ratio)

    return pd.DataFrame({'adaptation_log_ratio': log_ratio}).rename_axis('glomerulus')

def compute_dt1_response_profile(df_melted, bins=np.array([1, 2, 5, 10, 20, 30, 50, 80])):
    df = df_melted.dropna(subset=['delta_t1', 'code_response']).copy()
    df['delta_t1_bin'] = pd.cut(df['delta_t1'], bins=bins)
    profile = df.pivot_table(index='glomerulus', columns='delta_t1_bin', values='code_response', aggfunc='mean')

    profile.columns = profile.columns.astype(str)
    
    #return df.pivot_table(index='glomerulus', columns='delta_t1_bin', values='code_response', aggfunc='mean')

    return profile



def build_feature_matrix(tuning_norm, df_melted, df_raw, calcium_rois_by_animal):
    mean_amplitude_df = compute_mean_amplitude(df_melted)
    sinusoidal_df = fit_sinusoidal_model(tuning_norm)
    width_df = get_activation_metrics(tuning_norm, 0.75)
    variability_df = compute_response_variability(df_melted, method='var')  
    kurtosis_df = compute_kurtosis(calcium_rois_by_animal)
    skewness_df = compute_skewness(calcium_rois_by_animal)
    adaptation_df = compute_adaptation_ratio(df_raw)
    dt1_profile_df = compute_dt1_response_profile(df_melted)  
       
    feature_df = pd.concat([sinusoidal_df, width_df, mean_amplitude_df, variability_df, kurtosis_df, skewness_df, dt1_profile_df, adaptation_df], axis=1)
    
    tuning_flat = pd.DataFrame(tuning_norm.values, index=tuning_norm.index)
    tuning_flat.columns = [f"bin_{i}" for i in range(tuning_flat.shape[1])]
    feature_df = pd.concat([feature_df, tuning_flat], axis=1)
        
    #feature_df["animal"] = df_melted.groupby("glomerulus")["animal"].first()
    feature_df['animal'] = feature_df.index.to_series().str.extract(r'codes_(\w+)_\d+')

    return feature_df


def cluster_and_plot(feature_df, method='tsne', n_clusters=3, cluster_colors=None):
    X = feature_df.drop(columns=['animal']).values
    animal_ids = feature_df['animal']
    glomeruli = feature_df.index

    if method == 'pca':
        reducer = PCA(n_components=2)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=15, random_state=42)
    else:
        raise ValueError("Choose method='pca' or 'tsne'")

    X_reduced = reducer.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X)

    animal_palette = {'Sphinx': 'lightgrey', 'HW1': 'black'} #, 'HW4': 'darkgrey'}
    animal_colors = animal_ids.map(animal_palette)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=animal_colors, s=60, alpha=0.8, label=None)

    for i, glom in enumerate(glomeruli):
        plt.text(X_reduced[i, 0] + 0.02, X_reduced[i, 1], glom.split('_')[-1], fontsize=7, alpha=0.5)

    for cluster in np.unique(cluster_labels):
        points = X_reduced[cluster_labels == cluster]
        if len(points) >= 3:
            hull = ConvexHull(points)
            hull_pts = points[hull.vertices]
            poly_color = cluster_colors[cluster] if cluster_colors else 'black'
            polygon = Polygon(hull_pts, edgecolor=poly_color, facecolor='none', linestyle='--', linewidth=1.5)
            plt.gca().add_patch(polygon)

    for animal, color in animal_palette.items():
        plt.scatter([], [], c=color, label=animal, s=60)

    plt.title(f"Glomerular Clustering Across Animals ({method.upper()} + K-Means)")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(title="Animal")
    plt.tight_layout()
    plt.show()

    return cluster_labels, feature_df



def evaluate_silhouette_scores(feature_df, max_k=10, method='tsne'):

    print("Evaluating silhouette scores...")
    X = feature_df.drop(columns=['animal']).values

    if method == 'pca':
        reducer = PCA(n_components=2)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=15, random_state=42)
    else:
        raise ValueError("Choose method='pca' or 'tsne'")
    
    X_reduced = reducer.fit_transform(X)

    scores = []
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X_reduced, labels)
        scores.append((k, score))
        print(f"Clusters: {k}, Silhouette Score: {score:.3f}")

    # Plot
    ks, s_vals = zip(*scores)
    plt.figure(figsize=(6, 4))
    plt.plot(ks, s_vals, marker='o')
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.tight_layout()
    plt.show()
    
    return scores



def plot_average_tuning_curves(tuning_norm, cluster_labels):
    cluster_labels = np.array(cluster_labels)
    tuning_df = pd.DataFrame(tuning_norm.values, index=tuning_norm.index, columns=tuning_norm.columns)
    tuning_df['cluster'] = cluster_labels

    clusters = np.unique(cluster_labels)
    cluster_colors = sns.color_palette("tab10", len(clusters)) 

    plt.figure(figsize=(10, 4))

    for i, cluster in enumerate(clusters):
        cluster_tuning = tuning_df[tuning_df['cluster'] == cluster].drop(columns='cluster')
        mean_curve = cluster_tuning.mean(axis=0)
        sem_curve = cluster_tuning.sem(axis=0)

        plt.plot(tuning_norm.columns, mean_curve, label=f"Cluster {cluster}", color=cluster_colors[i])
        plt.fill_between(tuning_norm.columns, mean_curve - sem_curve, mean_curve + sem_curve, alpha=0.2, color=cluster_colors[i])

    plt.xlabel("Sniff Phase (radians)")
    plt.ylabel("Normalized Code Response")
    plt.title("Average Tuning Curves per Cluster")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return cluster_colors


def boxplot_features(feature_df, features_to_plot, cols=4):

    features_to_plot = [feat for feat in features_to_plot if feat in feature_df.columns]

    n_feats = len(features_to_plot)
    rows = int(np.ceil(n_feats / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5), constrained_layout=True)
    axes = axes.flatten()

    for i, feat in enumerate(features_to_plot):
        sns.boxplot(data=feature_df, x="cluster", y=feat, hue="cluster", ax=axes[i], palette="Set3", legend=False)

        axes[i].set_title(feat, fontsize=8)
        axes[i].tick_params(axis='x', labelsize=8)
        axes[i].tick_params(axis='y', labelsize=8)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=3.0, w_pad=1.5)
    plt.show()


from sklearn.metrics.pairwise import cosine_similarity

def compute_rsms_by_cluster(df_melted, feature_df, value_col="code_response", condition_col="delta_t1", bins=np.array([1, 2, 5, 10, 20, 30, 50, 80])):
    df = df_melted.dropna(subset=[condition_col, value_col]).copy()
    df['bin'] = pd.cut(df[condition_col], bins=bins)
    
    glomeruli_with_clusters = feature_df.index.intersection(df['glomerulus'].unique())
    cluster_dict = feature_df.loc[glomeruli_with_clusters]['cluster'].to_dict()
    
    df = df[df['glomerulus'].isin(cluster_dict.keys())]
    df['cluster'] = df['glomerulus'].map(cluster_dict)

    rsms = {}

    for cluster_id in sorted(df['cluster'].unique()):
        df_cluster = df[df['cluster'] == cluster_id]

        pivot = df_cluster.pivot_table(index='glomerulus', columns='bin', values=value_col, aggfunc='mean')

        if len(pivot) == 0:
            continue

        sim_matrix = cosine_similarity(pivot.fillna(0))
        sim_df = pd.DataFrame(sim_matrix, index=pivot.index, columns=pivot.index)
        rsms[cluster_id] = sim_df

    return rsms

def plot_rsms(rsms):
    n = len(rsms)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))

    if n == 1:
        axes = [axes]

    for i, (cluster_id, sim_df) in enumerate(rsms.items()):
        sns.heatmap(sim_df, ax=axes[i], cmap="viridis", square=True, cbar=False)
        axes[i].set_title(f"Cluster {cluster_id} RSM")
        axes[i].set_xticks([])
        axes[i].set_yticks([])

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
        
    animals = ['Sphinx', 'HW1'] #, 'HW4']

    all_results = []
    calcium_rois_by_animal={}

    for animal in animals:
        valve = data["valve_dict"][animal] / 100
        phase = data["phase_peaks_dict"][animal]
        calcium_signal = data["calcium_dict"][animal].mean(axis=1)   
        calcium_signal_rois = data["calcium_dict"][animal].values.T.tolist()    
        calcium_rois_by_animal[animal] = calcium_signal_rois   

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

    animal_paths = {
        'Sphinx': params_init["res_path_sphinx"],
        'HW1': params_init["res_path_hw1"],
        #'HW4': params_init["res_path_hw4"],
    }

    for animal, res_paths in animal_paths.items():
        df_animal = df[df["animal"] == animal].copy()
        
        for idx, res_path in enumerate(res_paths):
            model_path = os.path.join(res_path, "model", "model_final.pt")
            postprocess_path = os.path.join(res_path, "postprocess")

            net = torch.load(model_path, map_location=device, weights_only=False)
            net.to(device)
            net.eval()

            kernel = np.squeeze(net.get_param("H").clone().detach().cpu().numpy())
            xhat = torch.load(os.path.join(postprocess_path, "xhat.pt"))
            codehat = xhat[0, 0, 0, :].clone().detach().cpu().numpy()
            codehat = process_array(codehat, kernel)

            # Store selected codes for this animal only
            code_selected = []
            for onset in df_animal["onset_resp"].to_numpy():
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

            # Assign the code values back into the main df, for this animal only
            df.loc[df["animal"] == animal, f"codes_{animal}_{idx}"] = code_selected
                
    df_melted = melt_codes_df(df)
    
    tuning_norm = compute_normalized_tuning(df_melted, num_bins=22)
        
    feature_df = build_feature_matrix(tuning_norm, df_melted, df, calcium_rois_by_animal)

    features_to_plot = ["animal", "kurtosis", "skewness", "adaptation_log_ratio", "c", "a", "(1, 2]", "activation_width"]
    feature_df = feature_df[features_to_plot]

    silhouette_scores = evaluate_silhouette_scores(feature_df, max_k=10, method='tsne')
    best_k = max(silhouette_scores, key=lambda x: x[1])[0]
    
    cluster_labels, feature_df = cluster_and_plot(feature_df, method='tsne', n_clusters=3)
    feature_df["cluster"] = cluster_labels
        

    rsms = compute_rsms_by_cluster(df_melted, feature_df)
    plot_rsms(rsms)
            
    boxplot_features(feature_df, features_to_plot)

    """for feat in features_to_plot:
        groups = [group[feat].dropna().values for _, group in feature_df.groupby("cluster")]
        stat, p = f_oneway(*groups)
        print(f"{feat}: ANOVA p = {p:.3e}")"""

    
if __name__ == "__main__":
    main()

import torch
import os
import pickle
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import sys

sys.path.append(r"dunl-compneuro\src")
sys.path.append("")

import model

os.environ['PATH'] += os.pathsep + r"C:\Users\Blanche\AppData\Local\Programs\MiKTeX\miktex\bin\x64\\"


def init_params():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--res-path",
        type=str,
        help="res path",
        default= "sparseness/results/olfactorycalciumkernellength20num1_2025_02_20_11_30_36"
        #"sparseness/results/supervised_roi0_Sphinx"
        #"sparseness/results/olfactorycalciumkernellength20num1_2025_03_04_15_15_41"
        #"sparseness/results/HW1_calcium_supervised_across_rois_numwindow1_roi0_kernellength20_1kernels_1000unroll_2025_03_04_11_51_10"
        #"sparseness/results/HW1_calcium_supervised_across_rois_numwindow1_roi0_kernellength20_1kernels_1000unroll_2025_03_03_22_24_44"
        #"sparseness/results/olfactorycalciumkernellength20num1_2025_02_20_11_30_36" 
        #"sparseness/results/olfactorycalciumkernellength20num1_2025_02_18_15_32_24"
        #"sparsenessresults\calcium_unsupervised_numwindow1_neuron0_kernellength20_1kernels_1000unroll_2025_02_08_12_38_01",
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
        default=(8, 2),
    )

    args = parser.parse_args()
    params = vars(args)

    return params



def plot_y_yhat_xhat(df, y, yhat, xhat, index_beg, index_end, selected_index=None):
    df_range = df[(df["onset_resp"] >= index_beg) & (df["onset_resp"] <= index_end)].copy()

    i = 0  
    yi = y[i, 0, :].clone().detach().cpu().numpy()
    yihat = yhat[i, 0, :].clone().detach().cpu().numpy()[0]
    codehat = xhat[i, 0, :].clone().detach().cpu().numpy()[0]

    axes_fontsize = 10
    legend_fontsize = 8
    tick_fontsize = 10
    title_fontsize = 10
    fontfamily = "sans-serif"

    mpl.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "text.usetex": True,
            "axes.labelsize": axes_fontsize,
            "axes.titlesize": title_fontsize,
            "legend.fontsize": legend_fontsize,
            "xtick.labelsize": tick_fontsize,
            "ytick.labelsize": tick_fontsize,
            "text.latex.preamble": r"\usepackage{bm}",
            "axes.unicode_minus": False,
            "font.family": fontfamily,
        }
    )

    fig, ax = plt.subplots(1, 1, figsize=(12, 4), sharex=True, sharey=True)

    ax.tick_params(axis="x", direction="in")
    ax.tick_params(axis="y", direction="in")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    time = range(index_beg, index_end)
    ax.plot(time, yi[index_beg:index_end], color="black", label="raw", lw=0.7)
    ax.plot(time, yihat[index_beg:index_end], color="blue", label="rec", lw=0.7)
    ax.plot(time, codehat[index_beg:index_end], ".", color="green", alpha=0.7, lw=0.7, label="code")
            
    for _, row in df_range.iterrows():
        color = "green" if row["event_resp"] == "inh" else "red"
        ax.axvline(x=row["onset_resp"], color=color, linestyle="--", alpha=0.7, linewidth=0.6)
    
    if selected_index is not None:
        ax.axvline(x=selected_index, color="magenta", linestyle="-", alpha=1, linewidth=1.5, label="selected pulse")

    plt.xlabel("Time")
    plt.legend()
    plt.title("Original vs Reconstructed Calcium Signal (Supervised DUNL)")
    fig.tight_layout(pad=0.8, w_pad=0.7, h_pad=0.5)
    plt.show()
    plt.close()

def plot_selected_pulse(df, y, yhat, xhat, window=50):
    
    code_tol = 1e-3      
    phase_target = np.pi / 2
    phase_tol = 0.2      

    candidates = df[(df["codes"].abs() < code_tol) & (np.abs(df["median_phase"] - phase_target) < phase_tol)]
    
    if candidates.empty:
        print("No pulse meets the strict criteria; selecting pulse with minimum absolute code value.")
        selected_pulse = df.iloc[(df["codes"].abs()).argmin()]
    else:
        selected_pulse = candidates.iloc[3]
    
    selected_index = int(selected_pulse["onset_resp"])
    index_beg = max(0, selected_index - window)
    index_end = selected_index + window

    print("Selected pulse at onset index:", selected_index)
    print(selected_pulse)

    plot_y_yhat_xhat(df, y, yhat, xhat, index_beg, index_end, selected_index=selected_index)

    
def plot_boxplot(df):
    
    event_inh = df[df["event_resp"] == "inh"]["codes"]
    event_bet = df[df["event_resp"] == "between"]["codes"]
    event_exh = df[df["event_resp"] == "exh"]["codes"]

    plt.figure(figsize=(8, 6))
    plt.boxplot([event_inh, event_bet, event_exh], tick_labels=["Full Inhalation", "Mix of Inh and Exh","Full Exhalation"])
    plt.ylabel("Code")
    plt.title("Code Distribution Across Sniff Cycle Phases")
    plt.show()
    
    
def plot_sym_auc(df):
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


def scatter_code_phase(df):
    plt.figure(figsize=(8, 6))
    plt.scatter(df['median_phase'], df['codes'], color='blue', alpha=1)
    plt.xlabel('Median Phase')
    plt.ylabel('Codes')
    plt.show()    
    


def boxplot_strip(df):
    num_bins = 11
    df['phase_bin'] = pd.cut(df['median_phase'], bins=num_bins)

    df['phase_bin_center'] = df['phase_bin'].apply(lambda x: x.mid)

    df = df.sort_values('phase_bin_center')

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='phase_bin_center', y='codes', color='white', fliersize=0, linewidth=1.2)
    sns.stripplot(data=df, x='phase_bin_center', y='codes', color='blue', alpha=1, jitter=0.25, size=3)

    plt.xlabel('Median Pulse Phase')
    plt.ylabel('Code')
    plt.xticks([0, 5, 10], ["0", "π", "2π"])
    plt.tight_layout()
    plt.show()


def reg_sniff_response(df):
    df['sin_phase'] = np.sin(df['median_phase'])
    df['cos_phase'] = np.cos(df['median_phase'])

    X = df[['sin_phase', 'cos_phase']].values
    y = df['codes'].values

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)

    print(f"R² (variance explained by sin/cos model): {r2:.3f}")

    phi = np.linspace(0, 2 * np.pi, 300)
    phi_sin = np.sin(phi)
    phi_cos = np.cos(phi)
    phi_features = np.vstack([phi_sin, phi_cos]).T
    phi_pred = model.predict(phi_features)
    
    plt.figure(figsize=(8, 5))
    plt.scatter(df['median_phase'], y, color='blue', label="Actual Data")
    plt.plot(phi, phi_pred, color='red', label="Sinusoidal Fit", linewidth=2)
    plt.xticks([0, np.pi, 2 * np.pi], ["0", "π", "2π"])
    plt.xlabel("Median Sniff Phase (radians)")
    plt.ylabel("Code Amplitude")
    plt.text(0.05, 0.95, f"$R^2$ = {r2:.3f}", transform=plt.gca().transAxes,
            fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round", fc="white", ec="gray"))
    plt.legend()
    plt.tight_layout()
    plt.show()


def combined_sniff_phase_plot(df):
    # Sin/cos regression features
    df['sin_phase'] = np.sin(df['median_phase'])
    df['cos_phase'] = np.cos(df['median_phase'])
    X = df[['sin_phase', 'cos_phase']].values
    y = df['codes'].values

    model = LinearRegression()
    model.fit(X, y)
    r2 = r2_score(y, model.predict(X))

    # Prepare smooth curve
    phi = np.linspace(0, 2 * np.pi, 300)
    phi_features = np.vstack([np.sin(phi), np.cos(phi)]).T
    phi_pred = model.predict(phi_features)

    # Plot raw points + sinusoidal fit
    plt.figure(figsize=(10, 6))
    plt.scatter(df['median_phase'], df['codes'], color='blue', s=10, alpha=0.7, label='Data')
    plt.plot(phi, phi_pred, color='red', linewidth=2.5, label='Sinusoidal Fit')

    plt.xlabel("Median Sniff Phase (radians)")
    plt.ylabel("Code Amplitude")
    plt.xticks([0, np.pi, 2 * np.pi], ["0", "π", "2π"])
    plt.ylim(bottom=min(df['codes'].min(), -0.1))
    plt.text(0.05, 0.95, f"$R^2$ = {r2:.3f}", transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round", fc="white", ec="gray"))
    plt.legend()
    plt.title("Code vs. Sniff Phase with Sinusoidal Fit")
    plt.tight_layout()
    plt.show()

    

def main():
    print("Predict.")

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
    
    # load whiffs-----------------------------------------------------------#
        
    with open(params["path"], "rb") as f:
        data = pickle.load(f)
        
    animals = ['HW1']
    #animals = ['HW1', 'HW4', 'Sphinx']

    all_results = []
    offset = 0

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

            """if inh_points <= 25:
                current_event= "exh"
            else:
                current_event="inh"""
            if inh_points >= 49 :
                current_event= "inh"
            elif 49 > inh_points >= 30:
                current_event = "between"
            else:
                current_event="exh"
            # index of the value  of resp_ts that is the closest to stim_ts[start_idx]
            index = np.absolute(ca_ts-valve_ts[start_idx]).argmin()
            
            inst_phase = np.median(phase[start_idx+1:start_idx+51])

            median_phase.append(inst_phase)
            onset_resp.append(index)
            event_resp.append(current_event)

        valve_down = np.zeros(len(calcium_signal))
        valve_down[onset_resp] = 1 
        
        # Adjust onsets by offset
        onset_resp = [onset + offset for onset in onset_resp
                      if onset < len(calcium_signal) - params["kernel_length"]
        ]
        
        for i, onset in enumerate(onset_resp):

            all_results.append({
                "onset_resp": onset,
                "event_resp": event_resp[i],
                "median_phase": median_phase[i],
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
        yhat = torch.load(
            os.path.join(postprocess_path, "yhat_{}.pt".format(datafile_name))
        )
        y = torch.load(
            os.path.join(postprocess_path, "y_{}.pt".format(datafile_name))
        )"""

    xhat = torch.load(
        os.path.join(postprocess_path, "xhat.pt")
    )
    yhat = torch.load(
        os.path.join(postprocess_path, "yhat.pt")
    )
    y = torch.load(
        os.path.join(postprocess_path, "y.pt")
    )
    
    codehat = xhat[0, 0, 0, :].clone().detach().cpu().numpy()
    #code_selected = codehat[df["onset_resp"].to_numpy()]

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
    
    print(df.head())

    yi = y[0, 0, :].clone().detach().cpu().numpy()
    yihat = yhat[0, 0, :].clone().detach().cpu().numpy()[0]

    mse = mean_squared_error(yi, yihat)
    corr_coef, p_value = pearsonr(yi, yihat)
    r2 = r2_score(yi, yihat)

    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print(f"Pearson Correlation Coefficient: {corr_coef:.6f}")
    print(f"Coefficient of Determination (R²): {r2:.6f}")
    
    #combined_sniff_phase_plot(df)
    
    #reg_sniff_response(df)
    #boxplot_strip(df)
    #scatter_code_phase(df)
    
    #plot_boxplot(df)
    #plot_y_yhat_xhat(df, y, yhat, xhat, 1000, 1250)
    #plot_selected_pulse(df, y, yhat, xhat, window=50)


if __name__ == "__main__":
    main()


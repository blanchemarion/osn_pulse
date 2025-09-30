import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import pandas as pd
import pickle
import scipy.stats
from sklearn.model_selection import train_test_split, GroupShuffleSplit
#import tensorflow as tf
import glm_class as glm
#from tensorflow.python.client import device_lib
import torch
import dill
import statsmodels.api as sm
import optuna
from sklearn.preprocessing import StandardScaler
import numpy.linalg as npl
import os

def init_params():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--features-path",
        type=str,
        help="features path",
        default="sparseness/Data/all_features_glm_10hz.pkl",
        #"sparseness/Data/all_features_glm.pkl",
    )
    parser.add_argument(
        "--feature-names-path",
        type=str,
        help="feature names path",
        default="sparseness/Data/all_feature_names_glm_10hz.pkl",
        #"sparseness/Data/all_feature_names_glm.pkl",
    )
    parser.add_argument(
        "--ca-deconv-path",
        type=str,
        help="ca deconv path",
        default="sparseness/Data/y_ca_10hz.pkl",
        #"sparseness/Data/ca_deconv.pkl",
    )
    args = parser.parse_args()
    params = vars(args)

    return params


def parse_group_from_feature_names(feature_names):
    ''' 
    Parse feature_names into groups using hand-crafted rules

    Input parameters:: 
    feature_names: List of feature names. In this example, expanded features must contain bumpX in the name

    Returns:: 
    group_size: list of number of features in each group
    group_name: name of each group
    group_ind: group index of each feature in feature_names, ndarray of size (len(feature_names),)
    '''
    
    # Find expanded features and their number of sub-features:
    group_size = list()
    group_name = list()
    group_ind = list()
    for name in feature_names:
        if 'bump' not in name:
            # Non-bump expanded feature:
            group_size.append(1)
            group_name.append(name)

        elif 'bump0' in name:
            # First bump of a bump-expanded feature:
            group_size.append(1)
            group_name.append(name[:-6])

        else: 
            # Subsequent time shifts and bumps
            group_size[-1] += 1  

    # Create group index for each feature
    for i_group, this_size in enumerate(group_size):
        group_ind += [i_group]*this_size
    
    return group_size, group_name, np.array(group_ind)


def statsmodels_baseline(X_train_t, Y_train_t, X_test_t, Y_test_t, neuron=0):
    # 1) move to CPU and NumPy
    X_train_np = X_train_t.cpu().numpy()
    Y_train_np = Y_train_t.cpu().numpy()
    X_test_np  = X_test_t.cpu().numpy()
    Y_test_np  = Y_test_t.cpu().numpy()

    # 2) pick one neuron and (optionally) round to ints
    y0_train = np.round(Y_train_np[:, neuron]).astype(int)
    y0_test  = np.round(Y_test_np[:, neuron]).astype(int)

    # 3) add intercept column to both
    X0_train = sm.add_constant(X_train_np)  # shape (n_train, n_feat+1)
    X0_test  = sm.add_constant(X_test_np)   # shape (n_test,  n_feat+1)

    # 4) fit
    mod = sm.GLM(y0_train, X0_train, family=sm.families.Poisson())
    res = mod.fit()
    print(res.summary())

    # 5) predict
    pred0 = res.predict(X0_test)  # length = n_test

    # 6) compute fraction deviance explained
    frac, dmod, dnull = glm.deviance(
        pred0[:, None], 
        y0_test[:, None], 
        loss_type='poisson'
    )
    print(f"Neuron {neuron}: frac deviance explained = {frac[0]:.3f}")
    return frac[0]



def main():
    
    def objective(trial):
        # ---- model family fixed to your best performer ----
        reg = trial.suggest_categorical("regularization", ["elastic_net", "group_lasso"])
        if reg == "elastic_net":
            # center near 0.3; allow moderate sparsity either side
            l1 = trial.suggest_float("l1_ratio", 0.15, 0.5)
        else:
            l1 = 0.0

        # Optimizer (Adam); keep tight around 1e-3
        lr  = trial.suggest_float("learning_rate", 5e-4, 2e-3, log=True)
        # momentum not used by Adam; keep for completeness but unused
        mom = trial.suggest_float("momentum", 0.85, 0.95)

        # 1SE rule often 0.3–0.6
        sefrac = trial.suggest_float("se_fraction", 0.3, 0.6)

        # Optional tiny smoothing (most runs 0); occasionally try a small value
        smooth_strength = trial.suggest_categorical("smooth_strength",
            [0.0, trial.suggest_float("smooth_strength_small", 1e-6, 3e-4, log=True)]
        )

        # ---- lambda grid strategies (pick one) ----
        grid_style = trial.suggest_categorical("lambda_grid_style",
                                            ["narrow", "knee_dense", "wide"])
        if grid_style == "narrow":
            # tightly around your good band
            lam_hi = trial.suggest_float("lambda_high", 3e-3, 3e-2, log=True)
            lam_lo = trial.suggest_float("lambda_low",  3e-6, 3e-4, log=True)
            n_lams = trial.suggest_int("n_lambdas", 10, 16)
            lambdas = np.logspace(np.log10(lam_hi), np.log10(lam_lo), n_lams)

        elif grid_style == "knee_dense":
            # denser sampling near the knee (~1e-3–1e-2), plus a bit on each side
            knee_lo  = trial.suggest_float("knee_lo",  5e-4, 3e-3, log=True)
            knee_hi  = trial.suggest_float("knee_hi",  3e-3, 2e-2, log=True)
            flank_lo = trial.suggest_float("flank_lo", 1e-5, 3e-4, log=True)
            flank_hi = trial.suggest_float("flank_hi", 3e-2, 1e-1, log=True)
            n_knee   = trial.suggest_int("n_knee", 12, 24)
            n_flank  = trial.suggest_int("n_flank", 4, 8)
            lambdas = np.unique(np.r_[
                np.logspace(np.log10(flank_hi), np.log10(knee_hi), n_flank),
                np.logspace(np.log10(knee_hi),  np.log10(knee_lo), n_knee),
                np.logspace(np.log10(knee_lo),  np.log10(flank_lo), n_flank)
            ])

        else:  # "wide"
            # a safety net exploration; still biased to <1e-1
            lam_hi = trial.suggest_float("lambda_high_wide", 1e-2, 1e-1, log=True)
            lam_lo = trial.suggest_float("lambda_low_wide",  1e-6, 1e-4, log=True)
            n_lams = trial.suggest_int("n_lambdas_wide", 16, 28)
            lambdas = np.logspace(np.log10(lam_hi), np.log10(lam_lo), n_lams)

        # Occasionally include the unpenalized fit (use sparingly for Poisson)
        include_zero = trial.suggest_categorical("include_lambda_zero", [False, True])
        if include_zero:
            lambdas = np.unique(np.r_[lambdas, 0.0])

        lambdas = np.sort(lambdas)[::-1]  # descending

        # ---- model ----
        model = glm.GLM_CV(
            n_folds=5,
            auto_split=True, split_by_group=True,
            activation="exp", loss_type="poisson",
            regularization=reg,
            lambda_series=lambdas,
            l1_ratio=l1,
            smooth_strength=smooth_strength,
            optimizer="adam",
            learning_rate=lr,
            momentum=mom,
            min_iter_per_lambda=100,
            max_iter_per_lambda=10**4,
            num_iter_check=100,
            convergence_tol=1e-6,
        )

        model.fit(X_train, Y_train,
                group_idx=trial_id_train,
                feature_group_size=group_size,
                verbose=False)

        model.select_model(se_fraction=sefrac, make_fig=False)
        return float(model.selected_frac_dev_expl_cv.mean())

        
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("Current device:", torch.cuda.current_device())
        print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    params = init_params()

    with open(params['features_path'], "rb") as f:
        all_features = pickle.load(f)
        
    with open(params['feature_names_path'], "rb") as f:
        all_feature_names = pickle.load(f)
        
    with open(params['ca_deconv_path'], "rb") as f:
        y = pickle.load(f)
    
    print(all_feature_names)
        
    ## Data pre-processing
        
    # Clean up design matrix and z-score along sample dimension
    all_features = all_features[:1210000]
    all_features[np.isnan(all_features)] = 0
    X = scipy.stats.zscore(all_features, axis = 0)
    Y_raw = y['y'][:1210000].astype(np.float32)
    
    # Get indices for splitting according to trial_id 
    n_samples = X.shape[0]
    group_id = y['trial_index'][:1210000]
    gss = GroupShuffleSplit(n_splits = 1, train_size = 0.85, random_state = 42)
    train_idx, test_idx = next(gss.split(X, Y_raw, group_id))
    
    # after you create train_idx, test_idx
    scaler = StandardScaler(with_mean=True, with_std=True)

    X_train_np = all_features[train_idx, :].astype(np.float32)
    X_test_np  = all_features[test_idx, :].astype(np.float32)

    X_train_np[np.isnan(X_train_np)] = 0
    X_test_np[np.isnan(X_test_np)] = 0

    X_train_np = scaler.fit_transform(X_train_np)
    X_test_np  = scaler.transform(X_test_np)
    
    def project_out(B, A):
        # remove columns of A from B: B <- B - A @ (A^+ @ B)
        if A.size == 0:
            return B
        AtA = A.T @ A
        pinv = npl.pinv(AtA) @ A.T
        return B - A @ (pinv @ B)

    # build names and groups (load them from the file saved by data script)
    """with open("sparseness/Data/groups_glm_10hz.pkl", "rb") as f:
        G = pickle.load(f)
    group_name = G["group_name"]; groups = G["groups"]"""

    groups_path = "sparseness/Data/groups_glm_10hz.pkl"
    if os.path.exists(groups_path):
        with open(groups_path, "rb") as f:
            G = pickle.load(f)
        group_size  = G["group_size"]    # list[int], one entry per *group*
        group_name  = G["group_name"]    # list[str], group labels (optional/useful for reports)
        group_ind   = G["group_ind"]     # flat indices (rarely needed)
        groups = G["groups"]        # dict{name -> list[col_indices]} (handy for analysis)

        # sanity check: sum of group sizes must equal #features
        assert sum(group_size) == X_train_np.shape[1], \
            f"group_size sums to {sum(group_size)} but X has {X_train_np.shape[1]} columns"
    else:
        # fallback: if you didn’t save the groups file, use your parser
        group_size, group_name, group_ind = parse_group_from_feature_names(all_feature_names)
            
    fs = 10 
        
    #group_size, group_name, group_ind = parse_group_from_feature_names(all_feature_names)
    print('Number of groups =', len(group_size))
    print(group_size)
    print(group_name)
    
    

    # indices for respiration-ish features
    resp_names = [n for n in group_name if any(k in n for k in
                ["breath_ca","sin_phase","cos_phase","inh_frac","exh_frac","onset_phase"])]
    resp_cols = np.unique(np.concatenate([groups[n] for n in resp_names])) if resp_names else np.array([], int)

    A_train = X_train_np[:, resp_cols] if resp_cols.size else np.zeros((X_train_np.shape[0],0), X_train_np.dtype)

    # choose task groups to orthogonalize (examples)
    task_groups = [n for n in group_name if any(k in n for k in
                    ["conv_stim", "trialPhase_", "gap_run_bump", "prev_gap_run_bump"])]

    task_cols = np.unique(np.concatenate([groups[n] for n in task_groups])) if task_groups else np.array([], int)

    B_train = X_train_np[:, task_cols]
    B_train_proj = project_out(B_train, A_train)

    # replace in train
    X_train_np[:, task_cols] = B_train_proj

    # apply same transform to test:
    A_test = X_test_np[:, resp_cols] if resp_cols.size else np.zeros((X_test_np.shape[0],0), X_test_np.dtype)

    # compute projection operator using train quantities
    AtA = A_train.T @ A_train
    pinv = npl.pinv(AtA) @ A_train.T
    print(pinv)
    X_test_np[:, task_cols] = X_test_np[:, task_cols] - A_test @ (pinv @ X_test_np[:, task_cols])
    

    """Y = 10. * y['y'][:1210000]    
    scale = 4 # tune so mean counts/bin is ~0.1–1.0
    Y = np.maximum(np.rint(np.clip(Y * scale, 0, None)), 0).astype(np.float32)"""
    

    scale = 20.0  # tune so np.mean(Y_scaled) ~ 0.1–0.5
    Y_scaled = np.clip(Y_raw * scale, 0, None).astype(np.float32)  # no rounding

    Y_train = torch.tensor(Y_scaled[train_idx,:], dtype=torch.float32).to(device)
    Y_test  = torch.tensor(Y_scaled[test_idx,:],  dtype=torch.float32).to(device)

    X_train = torch.tensor(X_train_np, dtype=torch.float32).to(device)
    X_test  = torch.tensor(X_test_np,  dtype=torch.float32).to(device)
    """Y_train = torch.tensor(Y[train_idx,:], dtype=torch.float32).to(device) 
    Y_test = torch.tensor(Y[test_idx,:], dtype=torch.float32).to(device)"""

    # Split data into train and test set
    """X_train = torch.tensor(X[train_idx,:], dtype=torch.float32).to(device) 
    X_test = torch.tensor(X[test_idx,:], dtype=torch.float32).to(device)"""
    trial_id_train = group_id[train_idx] # extract trial_id for training data, which is used in CV splits later during fitting
    
        
    ## Model initialization and fitting
    """study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=25)
    print("Best hyper-params:", study.best_params, flush= True)"""
        
    # Initialize GLM_CV (here we're only specifying key input arguments; others are left as default values; see documentation for details)
    
    
    """model_cv = glm.GLM_CV(n_folds = 5, auto_split = True, split_by_group = True,
                      activation = 'linear', loss_type = 'gaussian', 
                      regularization = 'elastic_net', lambda_series = np.logspace(-4, -1, 40), #10.0 ** np.linspace(3, -6, 10),
                      l1_ratio = 0.1, smooth_strength= 0.,
                      optimizer = 'adam', learning_rate = 1e-3)"""
                          
    """model_cv = glm.GLM_CV(n_folds = 5, auto_split = True, split_by_group = True,
                      activation = 'exp', loss_type = 'poisson', 
                      regularization = 'group_lasso', lambda_series = np.logspace(-2, -1, 10),
                      l1_ratio = 0.1, smooth_strength= 0.,
                      optimizer = 'adam', learning_rate = 1e-3)"""
    
    lambdas =np.logspace(-4, -0.5, 10)  # ~6e-3..1.25e-1
    model_cv = glm.GLM_CV(
        n_folds=2, auto_split=True, split_by_group=True,
        activation='exp', loss_type='poisson',
        regularization='group_lasso',
        lambda_series=lambdas,
        l1_ratio=0.1, smooth_strength=0.,
        optimizer='adam', learning_rate=1e-3)

    

    # np.hstack([10.0 ** np.linspace(-8, -2, 25), [0.0]])
    # 10.0 ** np.linspace(3, -6, 10)
    # np.logspace(-4, -1, 40)
    
    # Fit the GLM_CV on training data
    model_cv.fit(X_train, Y_train, group_idx = trial_id_train, feature_group_size = group_size, verbose = True) 
    print("fitting done", flush=True)
     

    # Select models based on CV performance
    model_cv.select_model(se_fraction = 0.3, make_fig = True)
    print("model selected", flush=True)
    
    def recalibrate_intercept_poisson(model, X, Y):
        # X: torch or np, Y: torch or np, shapes (T,F),(T,R)
        Xnp = X.detach().cpu().numpy() if torch.is_tensor(X) else np.asarray(X)
        Ynp = Y.detach().cpu().numpy() if torch.is_tensor(Y) else np.asarray(Y)

        W = model.selected_w            # (F,R)
        b = model.selected_w0.copy()     # (R,)

        eta_no_b = Xnp @ W              # (T,R)
        mu_bar   = Ynp.mean(axis=0).clip(min=1e-12)      # (R,)
        exp_eta_bar = np.exp(eta_no_b).mean(axis=0).clip(min=1e-12)

        b_new = np.log(mu_bar) - np.log(exp_eta_bar)
        model.selected_b = b_new.astype(b.dtype, copy=False)

    # after model.select_model(...)
    recalibrate_intercept_poisson(model_cv, X_train, Y_train)


    # Evaluate model performance on test data
    frac_dev_expl, dev_model, dev_null, dev_expl = model_cv.evaluate(X_test, Y_test, make_fig = True)
    print("evaluation done", flush=True)

    # Make prediction on test data
    y_pred = model_cv.predict(X_test)
    print("prediction done", flush=True)
    print(y_pred.shape, flush=True)

    #with open("/n/home07/blanchemarion/sparseness/results/glm_model_cv.pkl", "wb") as f:
    #with open("/n/home07/blanchemarion/sparseness/results/model_cv.dill", "wb") as f:
    with open("sparseness/results/model_cv.dill", "wb") as f:
        dill.dump(model_cv, f)
        #pickle.dump(model_cv, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("model saved to glm_model_cv.pkl")
    
    
    # Compare data and prediction for selected neuron
    #i_neuron = 1
    
    for i_neuron in range(y_pred.shape[1]):
        start = 0
        n_timepoints = 1000

        y_true_plot = Y_test[start:start+n_timepoints, i_neuron].cpu().numpy() / 10
        y_pred_plot = y_pred[start:start+n_timepoints, i_neuron] / 10

        fig, ax1 = plt.subplots(figsize=(16, 3))

        # Plot y_true on the left y-axis
        ax1.plot(np.arange(n_timepoints)/fs, y_true_plot, 'k', lw=0.5, label='data')
        ax1.set_xlabel('Time (sec)')
        ax1.set_ylabel('Deconvolved activity (data)', color='k')
        ax1.tick_params(axis='y', labelcolor='k')

        # Create second y-axis for prediction
        ax2 = ax1.twinx()
        ax2.plot(np.arange(n_timepoints)/fs, y_pred_plot, 'r', lw=0.75, label='prediction')
        ax2.set_ylabel('Deconvolved activity (prediction)', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        fig.tight_layout()
        plt.show()
    #plt.savefig('/n/home07/blanchemarion/sparseness/results/prediction_glm.png')  


if __name__ == "__main__":
    main()

"""
Ex use:
python sparseness\Estimate_Kernels\glm\analyse_glm_fit.py --model sparseness/results/model_cv.dill --features sparseness/Data/all_features_glm_10hz.pkl --feature-names sparseness/Data/all_feature_names_glm_10hz.pkl --targets sparseness/Data/y_ca_10hz.pkl --outdir sparseness/results/analysis_glm --roi 3 --importance both --group-ablation --ablation-mode zero --ablate-top 30 --plots
"""

from __future__ import annotations
import argparse
import dill
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# GLM utilities
import glm_class as glm


# ---------------------------
#helpers
# ---------------------------

def numpy_to_native(obj):
    if isinstance(obj, dict):
        return {k: numpy_to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [numpy_to_native(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def load_model(model_path: str):
    with open(model_path, "rb") as f:
        model = dill.load(f)
    return model


def load_features(features_path: str, feature_names_path: str | None) -> Tuple[np.ndarray, List[str]]:
    with open(features_path, "rb") as f:
        X = dill.load(f) if features_path.endswith((".dill", ".pkl.dill")) else pd.read_pickle(f)
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
    if feature_names_path is not None:
        with open(feature_names_path, "rb") as f:
            names = pd.read_pickle(f)
            if isinstance(names, np.ndarray):
                names = names.tolist()
    else:
        names = [f"x{i}" for i in range(X.shape[1])]
    return X, names


def load_targets(targets_path: str) -> Tuple[np.ndarray, np.ndarray | None]:
    """Returns (Y, trial_index or None)."""
    with open(targets_path, "rb") as f:
        yobj = pd.read_pickle(f)
    if isinstance(yobj, dict) and "y" in yobj:
        Y = np.asarray(yobj["y"], dtype=np.float32)
        trial_index = np.asarray(yobj.get("trial_index", None)) if "trial_index" in yobj else None
    else:
        Y = np.asarray(yobj, dtype=np.float32)
        trial_index = None
    if Y.ndim == 1:
        Y = Y[:, None]
    return Y, trial_index


# ---------------------------
# Metrics
# ---------------------------

def compute_metrics(model, X: np.ndarray, Y: np.ndarray) -> Dict[str, object]:
    """Make predictions and compute deviance-based metrics per ROI."""
    Y_pred = model.predict(X)
    frac, d_model, d_null = glm.deviance(Y_pred, Y, loss_type=model.loss_type)
    return {
        "frac_dev_expl_per_roi": frac,  # (R,)
        "dev_model_per_roi": d_model,
        "dev_null_per_roi": d_null,
    }


# ---------------------------
# Feature grouping
# ---------------------------

def canonical_feature_groups(names: List[str]) -> Dict[str, List[int]]:
    """Heuristic grouping by base-name (before suffixes like _bumpNN, _lagK)."""
    groups: Dict[str, List[int]] = {}
    for i, nm in enumerate(names):
        base = nm
        for tag in ["_bump", "_lag", "-bump", "-lag"]:
            if tag in base:
                base = base.split(tag)[0]
        while len(base) and base[-1].isdigit():
            base = base[:-1]
        groups.setdefault(base, []).append(i)
    return groups


# ---------------------------
# Coefficient-based importance (ROI-specific)
# ---------------------------

def coefficient_importance(model, X: np.ndarray, names: List[str], roi: int) -> pd.DataFrame:
    """Effect-size style importance for a single ROI.

    Gaussian/linear: importance_j = | w_j * std(X_j) |
    Poisson/exp   : importance_j = | exp(w_j * std(X_j)) - 1 |
    """
    W = model.selected_w  # (F, R)
    F, R = W.shape
    assert F == X.shape[1], "Feature count mismatch between X and model.selected_w"
    assert 0 <= roi < R, f"ROI index out of bounds (0..{R-1})"

    sd = X.std(axis=0, ddof=0)
    sd[sd == 0] = 1.0
    w = W[:, roi]

    if model.activation == "exp" and model.loss_type == "poisson":
        imp = np.abs(np.exp(w * sd) - 1.0)
    else:
        imp = np.abs(w * sd)

    df = pd.DataFrame({
        "feature": names,
        "importance": imp,
        "weight": w,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return df


# ---------------------------
# Ablation-based importance (ROI-specific)
# ---------------------------

def ablation_importance(
    model,
    X: np.ndarray,
    Y: np.ndarray,
    names: List[str],
    roi: int,
    groups: Dict[str, List[int]] | None = None,
    mode: str = "zero",   # "zero" or "shuffle"
    top_k_features: int | None = None,
) -> pd.DataFrame:
    """Measure drop in *ROI's* fraction deviance explained after ablating columns."""
    base_pred = model.predict(X)
    base_frac, _, _ = glm.deviance(base_pred, Y, loss_type=model.loss_type)
    base_roi = float(base_frac[roi])

    rows = []

    if groups is None:
        feat_indices = list(range(X.shape[1]))
        if top_k_features is not None:
            coef_df = coefficient_importance(model, X, names, roi)
            keep = set(coef_df["feature"].head(top_k_features).tolist())
            feat_indices = [i for i, nm in enumerate(names) if nm in keep]

        for j in feat_indices:
            X_abl = X.copy()
            if mode == "zero":
                X_abl[:, j] = 0.0
            else:
                X_abl[:, j] = np.random.permutation(X_abl[:, j])
            pred = model.predict(X_abl)
            frac, _, _ = glm.deviance(pred, Y, loss_type=model.loss_type)
            drop = base_roi - float(frac[roi])
            rows.append({
                "target": names[j],
                "type": "feature",
                "size": 1,
                "delta_frac_roi": drop,
            })
    else:
        for gname, idxs in groups.items():
            X_abl = X.copy()
            if mode == "zero":
                X_abl[:, idxs] = 0.0
            else:
                for j in idxs:
                    X_abl[:, j] = np.random.permutation(X_abl[:, j])
            pred = model.predict(X_abl)
            frac, _, _ = glm.deviance(pred, Y, loss_type=model.loss_type)
            drop = base_roi - float(frac[roi])
            rows.append({
                "target": gname,
                "type": "group",
                "size": int(len(idxs)),
                "delta_frac_roi": drop,
            })

    df = pd.DataFrame(rows).sort_values("delta_frac_roi", ascending=False).reset_index(drop=True)
    return df


# ---------------------------
# Plot helpers (ROI-centric)
# ---------------------------

def plot_top_importance(df: pd.DataFrame, title: str, out_png: Path, k: int = 25):
    top = df.head(k)
    plt.figure(figsize=(8, max(3, 0.25*k)))
    plt.barh(top["feature"][::-1], top["importance"][::-1])
    plt.xlabel("importance (ROI)")
    plt.ylabel("feature")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_top_ablation(df: pd.DataFrame, title: str, out_png: Path, k: int = 25):
    top = df.head(k)
    labels = top["target"][::-1]
    vals = top["delta_frac_roi"][::-1]
    plt.figure(figsize=(8, max(3, 0.25*k)))
    plt.barh(labels, vals)
    plt.xlabel("Δ frac dev exp (ROI) — drop when ablated")
    plt.ylabel("target")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def scatter_pred_vs_true(Y_true: np.ndarray, Y_pred: np.ndarray, out_png: Path, roi: int = 0, npts: int = 5000):
    y = Y_true[:, roi]
    p = Y_pred[:, roi]
    if len(y) > npts:
        idx = np.random.choice(len(y), npts, replace=False)
        y = y[idx]; p = p[idx]
    plt.figure(figsize=(4.5,4.5))
    plt.plot(y, p, ".", alpha=0.3)
    lim = [min(y.min(), p.min()), max(y.max(), p.max())]
    plt.plot(lim, lim, "k--", lw=1)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(f"True vs Pred (ROI {roi})")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


# ---------------------------
# CLI
# ---------------------------

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--model", type=str, required=True, help="Path to model .dill file")
    ap.add_argument("--features", type=str, required=True, help="Path to features .pkl")
    ap.add_argument("--feature-names", type=str, default=None, help="Path to feature names .pkl (optional)")
    ap.add_argument("--targets", type=str, required=True, help="Path to y .pkl (dict with 'y' key)")
    ap.add_argument("--outdir", type=str, default="sparseness/results/analysis_glm", help="Output directory")
    ap.add_argument("--roi", type=int, default=0, help="Index of ROI to analyze (0-based)")
    ap.add_argument("--importance", type=str, choices=["coeff","ablation","both"], default="both")
    ap.add_argument("--ablation-mode", type=str, choices=["zero","shuffle"], default="zero")
    ap.add_argument("--ablate-top", type=int, default=None, help="If set, only ablate top-K features by coefficient importance")
    ap.add_argument("--group-ablation", action="store_true", help="Group ablation by base feature name (e.g., *_bump*)")
    ap.add_argument("--plots", action="store_true", help="Save diagnostic plots")
    return ap.parse_args()


# ---------------------------
# Main
# ---------------------------

def main():
    args = parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # --- Load
    model = load_model(args.model)
    X, names = load_features(args.features, args.feature_names)
    Y, _ = load_targets(args.targets)

    # --- Sanity
    assert X.shape[0] == Y.shape[0], "X and Y must have the same number of samples"
    R = model.selected_w.shape[1]
    assert X.shape[1] == model.selected_w.shape[0], (
        f"Feature dimension mismatch: X has {X.shape[1]}, model expects {model.selected_w.shape[0]}"
    )
    assert 0 <= args.roi < R, f"--roi must be in [0, {R-1}]"

    # --- Metrics (all ROIs), then save ROI slice
    metrics_all = compute_metrics(model, X, Y)
    frac_all = metrics_all["frac_dev_expl_per_roi"]
    dmod_all = metrics_all["dev_model_per_roi"]
    dnull_all = metrics_all["dev_null_per_roi"]

    metrics_roi = {
        "roi": int(args.roi),
        "frac_dev_expl": float(frac_all[args.roi]),
        "dev_model": float(dmod_all[args.roi]),
        "dev_null": float(dnull_all[args.roi]),
    }
    with open(outdir / f"metrics_roi{args.roi}.json", "w") as f:
        json.dump(numpy_to_native(metrics_roi), f, indent=2)

    # Also save per-ROI table for convenience
    pd.DataFrame({
        "roi": np.arange(len(frac_all)),
        "frac_dev_expl": frac_all,
        "dev_model": dmod_all,
        "dev_null": dnull_all,
    }).to_csv(outdir / "metrics_per_roi.csv", index=False)

    # --- Coefficient-based importance (ROI)
    if args.importance in ("coeff", "both"):
        coef_df = coefficient_importance(model, X, names, args.roi)
        coef_df.to_csv(outdir / f"feature_importance_coeff_roi{args.roi}.csv", index=False)
        if args.plots:
            plot_top_importance(coef_df, f"Coefficient-based importance (ROI {args.roi})", outdir / f"importance_coeff_top_roi{args.roi}.png")

    # --- Ablation-based importance (ROI)
    if args.importance in ("ablation", "both"):
        groups = canonical_feature_groups(names) if args.group_ablation else None
        ab_df = ablation_importance(
            model, X, Y, names,
            roi=args.roi,
            groups=groups,
            mode=args.ablation_mode,
            top_k_features=args.ablate_top,
        )
        ab_csv = outdir / (f"importance_ablation_groups_roi{args.roi}.csv" if args.group_ablation else f"importance_ablation_features_roi{args.roi}.csv")
        ab_df.to_csv(ab_csv, index=False)
        if args.plots:
            title = f"Group ablation (ROI {args.roi})" if args.group_ablation else f"Feature ablation (ROI {args.roi})"
            out_png = outdir / (f"ablation_groups_top_roi{args.roi}.png" if args.group_ablation else f"ablation_features_top_roi{args.roi}.png")
            plot_top_ablation(ab_df, title, out_png)

    # --- Example scatter for the ROI
    if args.plots:
        Y_pred = model.predict(X)
        scatter_pred_vs_true(Y, Y_pred, outdir / f"scatter_true_vs_pred_roi{args.roi}.png", roi=args.roi)

    print(f"Analysis complete for ROI {args.roi}. Outputs in:", outdir)


if __name__ == "__main__":
    main()

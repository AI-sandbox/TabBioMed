from __future__ import annotations
from tabbiomed.analysis.paths import RESULTS, OUTPUT as _OUTPUT_ROOT
import os
import re
import warnings
from pathlib import Path
import numpy as np

DEFAULT_RESULTS = RESULTS
DEFAULT_OUT = _OUTPUT_ROOT / "calibration"
EPS = 1e-06
N_BINS = 15


def _to_np(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _load_pred(path: Path) -> np.ndarray:
    import torch

    obj = torch.load(path, weights_only=False, map_location="cpu")
    proba = _to_np(obj["y_proba"] if isinstance(obj, dict) else obj)
    return proba


def _positive(proba: np.ndarray) -> np.ndarray:
    proba = np.asarray(proba, dtype=float)
    if proba.ndim == 1:
        return proba
    return proba[:, 1] if proba.shape[1] >= 2 else proba[:, 0]


def load_pair(run_dir: Path, variant: str = "best"):
    files = set(os.listdir(run_dir))
    for var in [variant, "default"] if variant != "default" else ["default"]:
        if f"predictions_{var}.pt" in files and "test_labels_fixed.npy" in files:
            y = np.load(run_dir / "test_labels_fixed.npy", allow_pickle=True).ravel()
            proba = _load_pred(run_dir / f"predictions_{var}.pt")
            return (y, proba, var, "fixed", np.zeros(len(y), dtype=int))
        folds = sorted(
            (
                int(m.group(1))
                for f in files
                if (m := re.fullmatch(f"predictions_fold_(\\d+)_{var}\\.pt", f))
            )
        )
        if folds and folds == list(range(5)):
            ys, ps, fid = ([], [], [])
            for k in folds:
                lab = run_dir / f"test_labels_fold_{k}.npy"
                if not lab.exists():
                    return None
                y = np.load(lab, allow_pickle=True).ravel()
                ys.append(y)
                ps.append(_load_pred(run_dir / f"predictions_fold_{k}_{var}.pt"))
                fid.append(np.full(len(y), k))
            return (
                np.concatenate(ys),
                np.concatenate(ps),
                var,
                f"cv{len(folds)}",
                np.concatenate(fid),
            )
    return None


def ece(y, p, n_bins=N_BINS, strategy="uniform"):
    if strategy == "uniform":
        edges = np.linspace(0, 1, n_bins + 1)
    else:
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
        if len(edges) == 1:
            return float(abs(np.mean(y) - np.mean(p)))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    tot = 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        if m.any():
            tot += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(tot)


def _irls_logistic(X, y, offset=None, n_iter=100, tol=1e-10):
    offset = np.zeros(len(y)) if offset is None else offset
    beta = np.zeros(X.shape[1])
    for _ in range(n_iter):
        eta = np.clip(X @ beta + offset, -35, 35)
        mu = 1 / (1 + np.exp(-eta))
        w = np.maximum(mu * (1 - mu), 1e-12)
        step = np.linalg.solve(
            X.T @ (X * w[:, None]) + 1e-10 * np.eye(X.shape[1]), X.T @ (y - mu)
        )
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    return beta


def logistic_recal(y, p):
    if np.unique(y).size < 2:
        return (np.nan, np.nan)
    pc = np.clip(p, EPS, 1 - EPS)
    lp = np.log(pc / (1 - pc))
    ones = np.ones((len(y), 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            slope = float(_irls_logistic(np.hstack([ones, lp[:, None]]), y)[1])
        except Exception:
            slope = np.nan
        try:
            intercept = float(_irls_logistic(ones, y, offset=lp)[0])
        except Exception:
            intercept = np.nan
    return (slope, intercept)


def lowess_curve(y, p, frac=2 / 3, grid=None):
    order = np.argsort(p)
    xs, ys = (p[order], y[order].astype(float))
    n = len(xs)
    k = max(int(np.ceil(frac * n)), 2)
    if grid is None:
        grid = np.unique(np.quantile(xs, np.linspace(0, 1, 201)))
    fit = np.empty(len(grid))
    for i, g in enumerate(grid):
        c = np.searchsorted(xs, g)
        lo, hi = (max(c - k, 0), min(c + k, n))
        d = np.abs(xs[lo:hi] - g)
        sel = np.argpartition(d, min(k, len(d)) - 1)[:k]
        xx, yy, dd = (xs[lo:hi][sel], ys[lo:hi][sel], d[sel])
        h = max(dd.max(), 1e-12)
        w = (1 - (dd / h) ** 3) ** 3
        sw = w.sum()
        if sw <= 0:
            fit[i] = yy.mean()
            continue
        xm = (w * xx).sum() / sw
        ym = (w * yy).sum() / sw
        sxx = (w * (xx - xm) ** 2).sum()
        b = (w * (xx - xm) * (yy - ym)).sum() / sxx if sxx > 1e-15 else 0.0
        fit[i] = ym + b * (g - xm)
    return (grid, np.clip(fit, 0, 1))


def lowess_errors(y, p):
    grid, fit = lowess_curve(y, p)
    sm_ = np.interp(p, grid, fit)
    err = np.abs(sm_ - p)
    return (float(np.mean(err)), float(np.quantile(err, 0.9)), float(np.max(err)))


def compute_metrics(y, p) -> dict:
    from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

    y = y.astype(float)
    pc = np.clip(p, EPS, 1 - EPS)
    both_classes = np.unique(y).size == 2
    slope, intercept = logistic_recal(y, p)
    ici, e90, emax = lowess_errors(y, p)
    return {
        "n_test": int(len(y)),
        "n_pos": int(y.sum()),
        "prevalence": float(y.mean()),
        "mean_pred": float(p.mean()),
        "bias": float(p.mean() - y.mean()),
        "roc_auc_pooled": float(roc_auc_score(y, p)) if both_classes else np.nan,
        "pr_auc_pooled": float(average_precision_score(y, p)) if y.any() else np.nan,
        "log_loss_pooled": float(log_loss(y, pc, labels=[0, 1])),
        "brier": float(np.mean((p - y) ** 2)),
        "brier_skill": float(1 - np.mean((p - y) ** 2) / (y.mean() * (1 - y.mean())))
        if both_classes
        else np.nan,
        "ece_uniform": ece(y, p, strategy="uniform"),
        "ece_quantile": ece(y, p, strategy="quantile"),
        "ici": ici,
        "e90": e90,
        "emax": emax,
        "cal_slope": slope,
        "cal_intercept": intercept,
        "frac_extreme": float(np.mean((p < 0.01) | (p > 0.99))),
        "p_range": float(p.max() - p.min()),
    }


def compute_metrics_multiclass(y, P) -> dict:
    from sklearn.metrics import log_loss, roc_auc_score

    y = np.asarray(y).astype(int)
    K = P.shape[1]
    conf = P.max(1)
    correct = (P.argmax(1) == y).astype(float)
    slope, intercept = logistic_recal(correct, conf)
    ici, e90, emax = lowess_errors(correct, conf)
    onehot = np.eye(K)[y]
    present = [k for k in range(K) if 0 < onehot[:, k].sum() < len(y)]
    try:
        auc = float(
            roc_auc_score(
                y, P, multi_class="ovr", average="macro", labels=list(range(K))
            )
        )
    except Exception:
        auc = np.nan
    return {
        "n_test": int(len(y)),
        "n_classes": int(K),
        "accuracy_pooled": float(correct.mean()),
        "mean_conf": float(conf.mean()),
        "bias": float(conf.mean() - correct.mean()),
        "roc_auc_pooled": auc,
        "log_loss_pooled": float(
            log_loss(y, np.clip(P, EPS, 1), labels=list(range(K)))
        ),
        "brier": float(np.mean(np.sum((P - onehot) ** 2, axis=1))),
        "ece_uniform": ece(correct, conf, strategy="uniform"),
        "ece_quantile": ece(correct, conf, strategy="quantile"),
        "ici": ici,
        "e90": e90,
        "emax": emax,
        "cal_slope": slope,
        "cal_intercept": intercept,
        "classwise_ece": float(np.mean([ece(onehot[:, k], P[:, k]) for k in present]))
        if present
        else np.nan,
        "frac_extreme": float(np.mean(conf > 0.99)),
    }

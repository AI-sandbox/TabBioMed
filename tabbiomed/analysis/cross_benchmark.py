from __future__ import annotations
from tabbiomed.analysis.paths import CONFIG
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

DEFAULT_TABARENA_RANKS = CONFIG / "tabarena_verified_20261108.csv"
DEFAULT_ALIASES = CONFIG / "tabarena_model_aliases.json"
TABARENA_RANK_COLS = {
    "overall": "avg_rank_overall",
    "class": "avg_rank_class",
    "regr": "avg_rank_regr",
    "binary": "avg_rank_binary",
    "multi": "avg_rank_multi",
    "small": "avg_rank_small",
    "medium": "avg_rank_medium",
}


def load_aliases(path: Path = DEFAULT_ALIASES) -> dict[str, dict[str, Any]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for k, v in raw.items():
        if isinstance(v, str):
            out[str(k)] = {"tabarena_model": v, "protocol": None}
        elif isinstance(v, dict):
            out[str(k)] = dict(v)
        else:
            raise TypeError(f"Bad alias value for {k!r}: {type(v)}")
    return out


def load_tabarena_ranks(
    path: Path = DEFAULT_TABARENA_RANKS, *, source: str = "glance_best"
) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path, comment="#", dtype=str)
    df["protocol"] = df["protocol"].astype(str)
    if "source" in df.columns and source:
        df = df[df["source"].astype(str) == source].copy()
    for col in TABARENA_RANK_COLS.values():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "elo" in df.columns:
        df["elo"] = pd.to_numeric(df["elo"], errors="coerce")
    return df.reset_index(drop=True)


def tabbiomed_mean_ranks(
    rows: pd.DataFrame, *, models: list[str] | None = None
) -> pd.DataFrame:
    df = rows.copy()
    if models is not None:
        df = df[df["model"].isin(models)].copy()
    rows_out: list[dict[str, Any]] = []
    for model, g in df.groupby("model", sort=True):
        ranks = pd.to_numeric(g["rank"], errors="coerce").dropna()
        norms = (
            pd.to_numeric(g["norm_metric"], errors="coerce").dropna()
            if "norm_metric" in g.columns
            else pd.Series(dtype=float)
        )
        family = (
            str(g["model_family"].iloc[0])
            if "model_family" in g.columns and len(g)
            else ""
        )
        rows_out.append(
            {
                "model": str(model),
                "model_family": family,
                "mean_rank": float(ranks.mean()) if len(ranks) else float("nan"),
                "median_rank": float(ranks.median()) if len(ranks) else float("nan"),
                "n_datasets": int(g["dataset"].nunique()),
                "mean_norm": float(norms.mean()) if len(norms) else float("nan"),
            }
        )
    out = pd.DataFrame(rows_out)
    if out.empty:
        return out
    out["ordinal_rank"] = out["mean_rank"].rank(method="average", ascending=True)
    return out.sort_values(["mean_rank", "model"]).reset_index(drop=True)


def align_shared_models(
    *,
    aliases: dict[str, dict[str, Any]],
    tabarena: pd.DataFrame,
    tabbiomed: pd.DataFrame,
    tabarena_metric: str = "overall",
    drop_notes: bool = False,
) -> pd.DataFrame:
    col = TABARENA_RANK_COLS.get(tabarena_metric)
    if col is None:
        raise ValueError(
            f"Unknown tabarena_metric={tabarena_metric!r}; choose from {sorted(TABARENA_RANK_COLS)}"
        )
    if col not in tabarena.columns:
        raise ValueError(f"TabArena table missing column {col!r}")
    tb = tabbiomed.set_index("model")
    rows: list[dict[str, Any]] = []
    for tbm_name, spec in aliases.items():
        if drop_notes and spec.get("note"):
            continue
        if tbm_name not in tb.index:
            continue
        ta_name = str(spec["tabarena_model"])
        protocol = spec.get("protocol")
        cand = tabarena[tabarena["model"] == ta_name]
        if protocol is not None and "protocol" in cand.columns:
            cand = cand[cand["protocol"].astype(str) == str(protocol)]
        if cand.empty:
            continue
        if "source" in cand.columns and (cand["source"] == "glance_best").any():
            cand = cand[cand["source"] == "glance_best"]
        r = cand.iloc[0]
        ta_rank = pd.to_numeric(r.get(col), errors="coerce")
        if not np.isfinite(float(ta_rank)):
            continue
        tb_row = tb.loc[tbm_name]
        if isinstance(tb_row, pd.DataFrame):
            tb_row = tb_row.iloc[0]
        mean_rank = float(tb_row["mean_rank"])
        if not np.isfinite(mean_rank):
            continue
        rows.append(
            {
                "model": tbm_name,
                "tabarena_model": ta_name,
                "tabarena_protocol": str(r.get("protocol", protocol or "")),
                "model_family": str(tb_row.get("model_family", "")),
                "note": str(spec.get("note") or ""),
                "tabarena_avg_rank": float(ta_rank),
                "tabbiomed_mean_rank": mean_rank,
                "tabbiomed_n_datasets": int(tb_row.get("n_datasets", 0)),
                "tabbiomed_mean_norm": float(tb_row.get("mean_norm", float("nan"))),
            }
        )
    aligned = pd.DataFrame(rows)
    if aligned.empty:
        return aligned
    aligned["tabarena_rank"] = aligned["tabarena_avg_rank"].rank(
        method="average", ascending=True
    )
    aligned["tabbiomed_rank"] = aligned["tabbiomed_mean_rank"].rank(
        method="average", ascending=True
    )
    aligned["delta_rank"] = aligned["tabbiomed_rank"] - aligned["tabarena_rank"]
    aligned["abs_delta_rank"] = aligned["delta_rank"].abs()
    return aligned.sort_values(["tabbiomed_rank", "model"]).reset_index(drop=True)


def rank_correlations(aligned: pd.DataFrame) -> dict[str, Any]:
    if aligned.empty or len(aligned) < 2:
        return {
            "n_models": int(len(aligned)),
            "kendall_tau": float("nan"),
            "kendall_p": float("nan"),
            "spearman_rho": float("nan"),
            "spearman_p": float("nan"),
        }
    x = aligned["tabarena_avg_rank"].to_numpy(dtype=float)
    y = aligned["tabbiomed_mean_rank"].to_numpy(dtype=float)
    kd = kendalltau(x, y)
    sp = spearmanr(x, y)
    return {
        "n_models": int(len(aligned)),
        "kendall_tau": float(kd.correlation)
        if kd.correlation is not None
        else float("nan"),
        "kendall_p": float(kd.pvalue) if kd.pvalue is not None else float("nan"),
        "spearman_rho": float(sp.correlation)
        if sp.correlation is not None
        else float("nan"),
        "spearman_p": float(sp.pvalue) if sp.pvalue is not None else float("nan"),
    }


def top_k_overlap(aligned: pd.DataFrame, k: int) -> dict[str, Any]:
    if aligned.empty or k <= 0:
        return {
            "k": int(k),
            "overlap": 0,
            "overlap_frac": float("nan"),
            "tabarena_top": [],
            "tabbiomed_top": [],
            "shared_top": [],
        }
    k_eff = min(int(k), len(aligned))
    ta = set(aligned.nsmallest(k_eff, "tabarena_rank")["model"].astype(str).tolist())
    tb = set(aligned.nsmallest(k_eff, "tabbiomed_rank")["model"].astype(str).tolist())
    shared = sorted(ta & tb)
    return {
        "k": k_eff,
        "overlap": len(shared),
        "overlap_frac": float(len(shared) / k_eff) if k_eff else float("nan"),
        "tabarena_top": sorted(ta),
        "tabbiomed_top": sorted(tb),
        "shared_top": shared,
    }

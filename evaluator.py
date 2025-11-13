"""
evaluator.py
============

Purpose
-------
Utilities to load run logs, aggregate metrics, and create compact plots:
- Regret curves (median + IQR) over iterations
- Boxplots of final regret / normalized regret / best value per method
- y_best curves (median + IQR)
- Rank & Winrate heatmaps (optional faceting by noise)

Input format
------------
One CSV per run, written by runner.py, with columns like:
    problem, method, seed, budget, iter, x, y, y_noisy, best_so_far, regret, time_s
Optional:
    noise_pct, noise_seed, best_so_far_obs, regret_obs, sigma_noise

Notes
-----
- Aggregations sind robust ggü. fehlenden Spalten (z.B. noise_pct).
- Bei Bedarf werden monotone "best_so_far"-Traces pro Seed erzwungen (cummin).
- 2D-Plots, publikations-tauglich, keine 3D-Grafiken.
"""
from __future__ import annotations
from typing import Sequence, Tuple, Optional, Union

try:
    from tb_style import apply as _apply_style
    _apply_style('thesis_v1')
except Exception:
    pass
from scipy.optimize import minimize
from numpy.linalg import lstsq
import json
import re
from typing import List
from typing import Optional, Dict
import math

# Problem-Factory (liefert f*, Bounds, Dim usw.)
from problems import get_problem

# ----- Imports ---------------------------------------------------------------

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# --- Safe saver to avoid gigantic images when "tight" explodes ---


def _safe_savefig(path, dpi=150, bbox_inches="tight"):
    import matplotlib as _mpl
    fig = plt.gcf()
    try:
        fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
    except ValueError as e:
        # Fallback: ohne "tight" und mit niedrigerem DPI erneut speichern
        if "Image size of" in str(e):
            fig.savefig(path, dpi=min(dpi, 120), bbox_inches=None)
        else:
            raise


# Bessere Layout-Automatik + sauberes Speichern
mpl.rcParams["figure.constrained_layout.use"] = True
mpl.rcParams["savefig.bbox"] = "tight"

# --- Table formatting options -------------------------------------------------
TABLE_DECIMALS: int = 3      # Standard: 3 Nachkommastellen
ROUND_BEST_AT_T: bool = True  # Wenn False, keine Rundung/Formatierung


# ----- I/O & Small Utils -----------------------------------------------------


def load_logs(results_dir: Union[str, Path]) -> pd.DataFrame:
    """
    Load experiment logs from a results folder:
    1) Prefer a unified 'all_runs.csv' at the root.
    2) Else concatenate all 'log_*.csv' under '<results_dir>/logs'.
    3) Else fall back to legacy '<results_dir>/log_*.csv'.
    """
    results_dir = Path(results_dir)
    all_runs_csv = results_dir / "all_runs.csv"
    if all_runs_csv.exists():
        return pd.read_csv(all_runs_csv)

    logs_dir = results_dir / "logs"
    logs = sorted(logs_dir.glob("log_*.csv")) if logs_dir.exists() else []
    if not logs:
        logs = sorted(results_dir.glob("log_*.csv"))  # legacy fallback

    if not logs:
        raise FileNotFoundError(
            f"No logs found under: {results_dir} (checked all_runs.csv, logs/log_*.csv, log_*.csv)")

    return pd.concat([pd.read_csv(p) for p in logs], ignore_index=True)


def _ensure_outdir(outdir: Union[str, Path]) -> Path:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _gb_apply(gb, func):
    """GroupBy.apply wrapper that avoids operating on grouping columns (pandas>=2.2)."""
    try:
        return gb.apply(func, include_groups=False)
    except TypeError:  # pandas < 2.2
        return gb.apply(func)


# ----- Problem helpers (canonical name, f*, q95) -----------------------------

def _canonical_from_logged(pname: str) -> str:
    """Map logged names like 'branin', 'rosenbrock5' to canonical factory names."""
    s = pname.strip().lower()
    if s.startswith("branin"):
        return "Branin"
    if s.startswith("sphere"):
        return "Sphere"
    if s.startswith("rosenbrock"):
        return "Rosenbrock"
    if s.startswith("rastrigin"):
        return "Rastrigin"
    if s.startswith("goldstein"):
        return "GoldsteinPrice"
    if s.startswith("hartmann6"):
        return "Hartmann6"
    return pname


def _fstar_for_problem(pname: str) -> float:
    canon = _canonical_from_logged(pname)
    return get_problem(canon).f_star


def _estimate_q95(problem_name: str, n: int = 20000, seed: int = 0) -> float:
    """Monte-Carlo-Schätzung des 95%-Quantils für f(x) unter Uniform-Sampling."""
    rng = np.random.default_rng(seed)
    canon = _canonical_from_logged(problem_name)
    prob = get_problem(canon)
    lo, hi = prob.bounds_lo, prob.bounds_hi
    X = lo + rng.random((n, prob.dim)) * (hi - lo)
    vals = np.array([prob.f_true(x) for x in X], dtype=float)
    return float(np.quantile(vals, 0.95))


# ----- Trace building (monotone, aligned) -----------------------------------

def _per_seed_monotone_traces(
    df: pd.DataFrame,
    value_col: str,                 # "regret" or "best_so_far"
    problem_col: str = "problem",
    method_col: str = "method",
    seed_col: str = "seed",
    iter_col: str = "iter",
) -> pd.DataFrame:
    """
    Build per-seed, per-method, per-problem **monotone, forward-filled** traces.

    - If value_col == 'regret', recompute as cummin(best_so_far) - f*
    - If value_col == 'best_so_far', enforce cummin(best_so_far)
    - Forward-fill to common T per (problem, method)
    """
    rows = []
    for (p, m, s), grp in df.groupby([problem_col, method_col, seed_col]):
        g = grp.sort_values(iter_col).copy()
        # Monotone "best" sicherstellen
        if "best_so_far" in g.columns and g["best_so_far"].notna().any():
            best = np.minimum.accumulate(g["best_so_far"].to_numpy(float))
        else:
            best = np.minimum.accumulate(g["y"].to_numpy(float))

        if value_col == "regret":
            fstar = _fstar_for_problem(p)
            vals = best - float(fstar)
        elif value_col == "best_so_far":
            vals = best
        else:
            raise ValueError(f"Unsupported value_col: {value_col}")

        it = g[iter_col].to_numpy(int)
        rows.append(pd.DataFrame(
            {"problem": p, "method": m, "seed": s, "iter": it, value_col: vals}
        ))

    traces = pd.concat(rows, ignore_index=True)

    # Auf gemeinsames T je (problem, method) auffüllen
    out_rows = []
    for (p, m), grp in traces.groupby(["problem", "method"]):
        T = int(grp["iter"].max())
        full_iters = np.arange(0, T + 1, dtype=int)
        for s, gs in grp.groupby("seed"):
            tmp = pd.DataFrame({"iter": full_iters})
            merged = tmp.merge(gs[["iter", value_col]],
                               on="iter", how="left").sort_values("iter")
            merged[value_col] = merged[value_col].ffill()
            # evtl. führende NaNs auf ersten gültigen Wert setzen
            first_val = merged[value_col].dropna(
            ).iloc[0] if merged[value_col].notna().any() else np.nan
            merged[value_col] = merged[value_col].fillna(first_val)
            merged["problem"] = p
            merged["method"] = m
            merged["seed"] = s
            out_rows.append(
                merged[["problem", "method", "seed", "iter", value_col]])

    return pd.concat(out_rows, ignore_index=True)


# ----- Aggregation (curves, normalized, AUC, final) -------------------------

def aggregate_regret(df: pd.DataFrame, use_observed: bool = False) -> pd.DataFrame:
    """
    Aggregate regret curves (median and IQR) über Seeds.

    Returns columns:
      ['problem','method','budget','noise_pct','iter','median','q25','q75','n']
    """
    if "noise_pct" not in df.columns:
        df = df.copy()
        df["noise_pct"] = 0.0

    val_col = "regret_obs" if use_observed else "regret"

    gcols = ["problem", "method", "budget", "noise_pct", "iter"]

    def _summ(d: pd.DataFrame) -> pd.Series:
        v = d[val_col].to_numpy(float)
        return pd.Series({
            "median": float(np.median(v)),
            "q25": float(np.percentile(v, 25)),
            "q75": float(np.percentile(v, 75)),
            "n": int(v.size),
        })

    out = (
        df.groupby(gcols, as_index=False)
        .apply(_summ, include_groups=False)
        .reset_index(drop=True)
    )
    return out


def add_normalized_regret(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'regret_norm' = (best_so_far - f*) / (q95 - f*), in [0,1] geklemmt.
    """
    df = df.copy()
    fstar_map, q95_map = {}, {}
    for pname in df["problem"].unique():
        canon = _canonical_from_logged(pname)
        prob = get_problem(canon)
        fstar_map[pname] = prob.f_star
        try:
            q95_map[pname] = _estimate_q95(canon)
        except Exception:
            q95_map[pname] = np.nan

    def _norm(row):
        fstar = fstar_map.get(row["problem"], np.nan)
        q95 = q95_map.get(row["problem"], np.nan)
        if not np.isfinite(fstar) or not np.isfinite(q95) or q95 == fstar:
            return np.nan
        val = (row["best_so_far"] - fstar) / (q95 - fstar)
        return float(np.clip(val, 0.0, 1.0))

    df["regret_norm"] = df.apply(_norm, axis=1)
    return df


def aggregate_regret_norm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate normalized regret (0..1) über Seeds (nach monotone best Traces).
    Columns: problem, method, iter, [noise_pct], median, q25, q75
    """
    if "noise_pct" in df.columns:
        outs = []
        for nlev, dfn in df.groupby("noise_pct"):
            mono_best = _per_seed_monotone_traces(dfn, value_col="best_so_far")
            df_sc = dfn[["problem", "method", "seed",
                         "iter", "regret_norm"]].copy()
            merged = mono_best.merge(
                df_sc, on=["problem", "method", "seed", "iter"], how="left")
            need = merged.dropna(subset=["regret_norm"]).copy()
            if need.empty:
                continue
            agg_n = (
                need.groupby(["problem", "method", "iter"])["regret_norm"]
                .agg(median="median",
                     q25=lambda s: s.quantile(0.25),
                     q75=lambda s: s.quantile(0.75))
                .reset_index()
            )
            agg_n["noise_pct"] = nlev
            outs.append(agg_n)
        if outs:
            return pd.concat(outs, ignore_index=True)
        return pd.DataFrame(columns=["problem", "method", "iter", "median", "q25", "q75", "noise_pct"])
    else:
        mono_best = _per_seed_monotone_traces(df, value_col="best_so_far")
        df_sc = df[["problem", "method", "seed", "iter", "regret_norm"]].copy()
        merged = mono_best.merge(
            df_sc, on=["problem", "method", "seed", "iter"], how="left")
        need = merged.dropna(subset=["regret_norm"]).copy()
        if need.empty:
            return pd.DataFrame(columns=["problem", "method", "iter", "median", "q25", "q75"])
        agg = (
            need.groupby(["problem", "method", "iter"])["regret_norm"]
                .agg(median="median",
                     q25=lambda s: s.quantile(0.25),
                     q75=lambda s: s.quantile(0.75))
                .reset_index()
        )
        return agg


def _per_run_monotone_regret(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-run monotone regret traces. Columns: problem, method, seed, budget, iter, regret
    """
    rows = []
    for (p, m, s, b), grp in df.groupby(["problem", "method", "seed", "budget"]):
        g = grp.sort_values("iter").copy()
        best = np.minimum.accumulate(
            (g["best_so_far"] if "best_so_far" in g.columns else g["y"]).to_numpy(float))
        try:
            fstar = _fstar_for_problem(p)
        except Exception:
            fstar = np.nan
        reg = best - fstar if np.isfinite(fstar) else np.nan
        it = g["iter"].to_numpy(int)
        rows.append(pd.DataFrame(
            {"problem": p, "method": m, "seed": s, "budget": b, "iter": it, "regret": reg}))
    return pd.concat(rows, ignore_index=True)


def compute_auc_regret(df: pd.DataFrame, normalize_by_T: bool = True) -> pd.DataFrame:
    """
    Compute AUC(regret) per run (trapezoidal rule on monotone regret trace).
    Returns: ['problem','method','seed','budget','auc_regret']
    """
    traces = _per_run_monotone_regret(df).dropna(subset=["regret"])
    out_rows = []
    for (p, m, s, b), g in traces.groupby(["problem", "method", "seed", "budget"]):
        g = g.sort_values("iter")
        it = g["iter"].to_numpy(float)
        y = g["regret"].to_numpy(float)
        if len(it) >= 2:
            auc = np.trapz(y, it)
            if normalize_by_T:
                T = float(g["iter"].max() + 1)
                if T > 0:
                    auc = auc / T
        else:
            auc = float(y[-1]) if len(y) else np.nan
        out_rows.append({"problem": p, "method": m, "seed": s,
                        "budget": b, "auc_regret": auc})
    return pd.DataFrame(out_rows)


def summarize_auc_regret(auc_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate AUC(regret) across seeds per (problem, method, budget)."""
    def ci95(s: pd.Series) -> float:
        n = int(s.count())
        if n <= 1:
            return np.nan
        return 1.96 * float(s.std(ddof=1)) / np.sqrt(n)

    agg = (
        auc_df.groupby(["problem", "method", "budget"])["auc_regret"]
              .agg(mean="mean", std=lambda s: s.std(ddof=1), ci95=ci95, n="count")
              .reset_index()
    )
    return agg


def final_regret_per_run(df: pd.DataFrame) -> pd.DataFrame:
    """One row per run with the final regret."""
    dff = df.sort_values("iter").groupby(
        ["problem", "method", "seed", "budget"], as_index=False).tail(1)
    dff = dff.dropna(subset=["regret"])
    return dff[["problem", "method", "seed", "budget", "regret"]].rename(columns={"regret": "final_regret"})


# ----- Ranking & Group summaries --------------------------------------------

def metric_per_run(
    df: pd.DataFrame,
    metric: str = "final_regret",       # {"final_regret","auc_regret"}
    normalize_by_T: bool = True
) -> pd.DataFrame:
    """
    Return one row per run (problem, budget, seed, method) with a scalar metric,
    carrying over run-level meta columns if present (e.g., noise_pct).
    """
    keys = ["problem", "method", "seed", "budget"]

    # optional meta columns (if present)
    extras = [c for c in ("noise_pct", "noise_seed",
                          "shift_seed", "init_factor") if c in df.columns]
    meta = (df.sort_values("iter").groupby(keys, as_index=False).tail(1))[keys + extras].drop_duplicates() \
        if extras else None

    if metric == "final_regret":
        dff = df.sort_values("iter").groupby(keys, as_index=False).tail(1)
        out = dff[keys + ["regret"]
                  ].dropna(subset=["regret"]).rename(columns={"regret": "metric"})
        if meta is not None:
            out = out.merge(meta, on=keys, how="left")
        return out

    if metric == "auc_regret":
        auc = compute_auc_regret(df, normalize_by_T=normalize_by_T).rename(
            columns={"auc_regret": "metric"})
        if meta is not None:
            auc = auc.merge(meta, on=keys, how="left")
        return auc

    raise ValueError("metric must be 'final_regret' or 'auc_regret'.")


def compute_average_rank(df: pd.DataFrame, metric: str = "final_regret") -> pd.DataFrame:
    """
    Compute average rank per method über Gruppen (problem, budget, seed).
    """
    if metric == "final_regret":
        base = final_regret_per_run(df).rename(
            columns={"final_regret": "metric"})
    elif metric == "auc_regret":
        base = compute_auc_regret(df, normalize_by_T=True).rename(
            columns={"auc_regret": "metric"})
    else:
        raise ValueError("metric must be 'final_regret' or 'auc_regret'.")

    ranks = _gb_apply(
        base.groupby(["problem", "budget", "seed"], as_index=False),
        lambda g: g.assign(rank=g["metric"].rank(method="average"))
    ).reset_index(drop=True)

    summary = (
        ranks.groupby("method")["rank"]
             .agg(avg_rank="mean", std_rank=lambda s: s.std(ddof=1), n="count")
             .reset_index()
             .sort_values("avg_rank")
    )
    return summary


def rank_by_group(
    df: pd.DataFrame,
    metric: str = "auc_regret",
    group_cols: Sequence[str] = ("problem", "budget"),
    seed_agg: str = "median",
    rank_method: str = "average",
) -> pd.DataFrame:
    """
    Average rank per parameter group über Methoden.
    Steps:
      1) metric_per_run → one row per (group, method, seed)
      2) aggregate across seeds per (group, method) via seed_agg
      3) rank within each group (lower is better)
    """
    per = metric_per_run(df, metric=metric).dropna(subset=["metric"]).copy()

    group_cols = [c for c in group_cols if c in per.columns]
    if not group_cols:
        raise ValueError(
            "rank_by_group: no valid group columns found in data.")

    agg = (
        per.groupby(group_cols + ["method"], as_index=False)
           .agg(metric_agg=("metric", seed_agg))
    )
    agg["rank"] = agg.groupby(group_cols)["metric_agg"].rank(
        method=rank_method, ascending=True)
    return agg[[*group_cols, "method", "rank", "metric_agg"]]


def winrate_by_group(
    df: pd.DataFrame,
    metric: str = "auc_regret",
    group_cols: Sequence[str] = ("problem", "budget"),
    rank_method: str = "average",
) -> pd.DataFrame:
    """
    Win rate per group = Anteil Seeds mit Rang #1 (Ties erlaubt).
    """
    per = metric_per_run(df, metric=metric).dropna(subset=["metric"]).copy()

    group_cols = [c for c in group_cols if c in per.columns]
    if not group_cols:
        raise ValueError(
            "winrate_by_group: no valid group columns found in data.")

    seed_keys = group_cols + ["seed"]
    per["rank"] = per.groupby(seed_keys)["metric"].rank(
        method=rank_method, ascending=True)

    min_rank = per.groupby(seed_keys)["rank"].transform("min")
    winners = per[per["rank"] == min_rank]

    wins = winners.groupby(
        group_cols + ["method"]).size().reset_index(name="wins")
    n_seeds = per.groupby(group_cols)[
        "seed"].nunique().reset_index(name="n_seeds")

    methods_all = sorted(per["method"].unique().tolist())
    groups_df = per[group_cols].drop_duplicates().reset_index(drop=True)
    cart = (groups_df.assign(_k=1)
            .merge(pd.DataFrame({"method": methods_all, "_k": 1}), on="_k")
            .drop(columns="_k"))

    out = (cart
           .merge(wins, on=[*group_cols, "method"], how="left")
           .merge(n_seeds, on=list(group_cols), how="left"))
    out["wins"] = out["wins"].fillna(0)
    out["win_rate"] = out["wins"] / out["n_seeds"].replace(0, np.nan)
    return out


# ----- Plotting helpers ------------------------------------------------------

def _ordered_methods(methods_in_data: List[str], method_order: Optional[List[str]]) -> List[str]:
    if method_order is None:
        return sorted(methods_in_data)
    present = [m for m in method_order if m in methods_in_data]
    rest = sorted([m for m in methods_in_data if m not in present])
    return present + rest


def _final_rows_per_run(df: pd.DataFrame) -> pd.DataFrame:
    """Return the last logged row per (problem, method, budget, seed[, noise_pct])."""
    dd = df.copy()
    if "noise_pct" not in dd.columns:
        dd["noise_pct"] = 0.0
    keys = ["problem", "method", "budget", "seed", "noise_pct"]
    dd = dd.sort_values("iter")
    final = dd.groupby(keys, as_index=False, sort=False).tail(1)
    return final


def _fmt_noise(n: Optional[float]) -> tuple[str, str]:
    if n is None:
        return "", ""
    pct = 100.0 * float(n)
    if abs(pct - round(pct)) < 1e-6:
        return f"{int(round(pct))}%", f"_N{int(round(pct))}"
    return f"{pct:.1f}%", f"_N{str(round(pct,1)).replace('.', 'p')}"


def _add_suffix_to_filename(filename: str, suffix: str) -> str:
    stem, dot, ext = filename.partition(".")
    return f"{stem}{suffix}.{ext}" if dot else f"{filename}{suffix}"


def _group_label(row: pd.Series, cols: Sequence[str]) -> str:
    """Human-friendly label for heatmap columns."""
    parts = []
    for c in cols:
        v = row[c]
        if c == "budget":
            try:
                parts.append(f"T{int(v)}")
            except Exception:
                parts.append(f"T{v}")
        elif c == "noise_pct":
            try:
                parts.append(f"N{int(round(100*float(v)))}")
            except Exception:
                parts.append(f"N{v}")
        elif c == "problem":
            parts.append(str(v))
        else:
            parts.append(f"{c}={v}")
    return " | ".join(parts) if parts else "(all)"


def _annotate_median_labels_next_to_lines(ax, bp, data_len: int) -> None:
    """
    Schreibe die Median-Zahl in der **gleichen Farbe** wie die Median-Linie
    **rechts neben** die Linie. Erweitert die x-Achse minimal nach rechts.
    """
    # etwas Platz rechts neben der Median-Linie
    left, right = ax.get_xlim()
    span = right - left if right > left else max(1.0, data_len)
    xpad = 0.03 * span
    ax.set_xlim(0.5, max(right, data_len + 1.2))  # rechts Luft

    med_lines = bp.get("medians", [])
    if not med_lines:
        return

    for line in med_lines:
        # Koordinaten der Median-Linie
        x0, x1 = line.get_xdata()
        y0, y1 = line.get_ydata()
        y_med = 0.5 * (y0 + y1)
        x_right = max(x0, x1)

        # Farbe der Linie ermitteln
        color = line.get_color()
        if isinstance(color, (list, tuple)) and len(color) > 0:
            color = color[0]

        # Zahl formatieren (Hole Median-Wert aus y_med)
        try:
            txt = f"{float(y_med):.3g}"
        except Exception:
            txt = f"{y_med}"

        ax.text(x_right + xpad, y_med, txt, ha="left", va="center",
                fontsize=9, color=color)


# ----- Plots: Curves ---------------------------------------------------------


# --- axis formatting helpers ---
def _apply_integer_iteration_axis(ax, T_max: int | None, *, force_zero: bool = True):
    """Force integer ticks on x, set 0..T, and show integer labels."""
    try:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    except Exception:
        pass
    if T_max is not None:
        try:
            T_max = int(T_max)
            if T_max >= 0:
                step = 1 if T_max <= 30 else 2 if T_max <= 60 else max(
                    5, T_max // 10)
                import numpy as _np
                ax.set_xlim(0, T_max)
                ax.set_xticks(_np.arange(0, T_max + 1, step))
        except Exception:
            pass
    if force_zero:
        try:
            lo, hi = ax.get_xlim()
            if lo > 0:
                ax.set_xlim(0, hi)
        except Exception:
            pass


def plot_regret_curves(
    agg: pd.DataFrame,
    outdir: Union[str, Path],
    ylabel: str = "Regret (median, IQR)",
    suffix: str = "",
    method_order: Optional[List[str]] = None,
    n_seeds: Optional[int] = None,
) -> None:
    out = _ensure_outdir(Path(outdir) / "plots" /
                         ("regret_curves" + (suffix if suffix else "")))
    if "noise_pct" not in agg.columns:
        agg = agg.copy()
        agg["noise_pct"] = 0.0
    has_budget = ("budget" in agg.columns) and agg["budget"].notna().any()

    def _ticks(ax, T, n_labels=10):
        T = int(T)
        step = max(1, (T + n_labels - 1) // n_labels)
        ticks = list(range(step, T + 1, step))
        if not ticks or ticks[-1] != T:
            ticks.append(T)
        ax.set_xlim(1, T)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])

    for p in sorted(agg["problem"].unique()):
        aggp = agg[agg["problem"] == p]
        if has_budget:
            for T in sorted({int(b) for b in aggp["budget"].dropna().unique()}):
                aggb = aggp[aggp["budget"] == T]
                for n in sorted(aggb["noise_pct"].unique()):
                    sub = aggb[aggb["noise_pct"] == n]
                    if sub.empty:
                        continue
                    present = sorted(sub["method"].unique())
                    methods = [m for m in (
                        method_order or present) if m in present]
                    fig, ax = plt.subplots(figsize=(8, 4.8))
                    for m in methods:
                        sm = sub[sub["method"] == m].sort_values("iter")
                        x = sm["iter"].to_numpy() + 1
                        ax.plot(x, sm["median"], label=m)
                        ax.fill_between(x, sm["q25"], sm["q75"], alpha=0.15)
                    inferred = None
                    if "n" in sub.columns and sub["n"].notna().any():
                        try:
                            inferred = int(sub["n"].max())
                        except Exception:
                            inferred = None
                    seeds_for_title = n_seeds if n_seeds is not None else inferred
                    title = f"{p}: Regret curves (median ± IQR"
                    if seeds_for_title is not None:
                        title += f", N={seeds_for_title} seeds"
                    title += f"), noise={int(round(100*float(n)))}%"
                    ax.set_title(title)
                    ax.set_xlabel("Iteration")
                    ax.set_ylabel(ylabel)
                    ax.legend(loc="upper right")
                    try:
                        _apply_integer_iteration_axis(ax, int(T))
                    except Exception:
                        pass
                    _ticks(ax, int(T), n_labels=10)
                    fname = f"regret_curves_{p}_T{int(T)}_N{int(round(100*float(n)))}{suffix}.png"
                    plt.savefig(out / fname, dpi=150, bbox_inches="tight")
                    plt.close(fig)
        else:
            for n in sorted(aggp["noise_pct"].unique()):
                sub = aggp[aggp["noise_pct"] == n]
                if sub.empty:
                    continue
                present = sorted(sub["method"].unique())
                methods = [m for m in (
                    method_order or present) if m in present]
                fig, ax = plt.subplots(figsize=(8, 4.8))
                for m in methods:
                    sm = sub[sub["method"] == m].sort_values("iter")
                    x = sm["iter"].to_numpy() + 1
                    ax.plot(x, sm["median"], label=m)
                    ax.fill_between(x, sm["q25"], sm["q75"], alpha=0.15)
                inferred = None
                if "n" in sub.columns and sub["n"].notna().any():
                    try:
                        inferred = int(sub["n"].max())
                    except Exception:
                        inferred = None
                seeds_for_title = n_seeds if n_seeds is not None else inferred
                T_inf = int(sub["iter"].max()) + \
                    1 if "iter" in sub.columns else None
                title = f"{p}: Regret curves (median ± IQR"
                if seeds_for_title is not None:
                    title += f", N={seeds_for_title} seeds"
                title += f"), noise={int(round(100*float(n)))}%"
                ax.set_title(title)
                ax.set_xlabel("Iteration")
                ax.set_ylabel(ylabel)
                ax.legend(loc="upper right")
                try:
                    _apply_integer_iteration_axis(ax, int(T_inf))
                except Exception:
                    pass
                _ticks(ax, int(T_inf), n_labels=10)
                fname = f"regret_curves_{p}_T{T_inf}_N{int(round(100*float(n)))}{suffix}.png"
                plt.savefig(out / fname, dpi=150, bbox_inches="tight")
                plt.close(fig)


def plot_regret_norm_curves(
    agg: pd.DataFrame,
    outdir: Union[str, Path],
    ylabel: str = "Normalized regret (median, IQR)",
    method_order: Optional[List[str]] = None,
    n_seeds: Optional[int] = None,
) -> None:
    out = _ensure_outdir(Path(outdir) / "plots" / "regret_norm_curves")
    if "noise_pct" not in agg.columns:
        agg = agg.copy()
        agg["noise_pct"] = 0.0
    has_budget = ("budget" in agg.columns) and agg["budget"].notna().any()

    def _ticks(ax, T, n_labels=10):
        T = int(T)
        step = max(1, (T + n_labels - 1) // n_labels)
        ticks = list(range(step, T + 1, step))
        if not ticks or ticks[-1] != T:
            ticks.append(T)
        ax.set_xlim(1, T)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])

    for p in sorted(agg["problem"].unique()):
        aggp = agg[agg["problem"] == p]
        if has_budget:
            for T in sorted({int(b) for b in aggp["budget"].dropna().unique()}):
                aggb = aggp[aggp["budget"] == T]
                for n in sorted(aggb["noise_pct"].unique()):
                    sub = aggb[aggb["noise_pct"] == n]
                    if sub.empty:
                        continue
                    present = sorted(sub["method"].unique())
                    methods = [m for m in (
                        method_order or present) if m in present]
                    fig, ax = plt.subplots(figsize=(8, 4.8))
                    for m in methods:
                        sm = sub[sub["method"] == m].sort_values("iter")
                        x = sm["iter"].to_numpy() + 1
                        ax.plot(x, sm["median"], label=m)
                        ax.fill_between(x, sm["q25"], sm["q75"], alpha=0.15)
                    inferred = None
                    if "n" in sub.columns and sub["n"].notna().any():
                        try:
                            inferred = int(sub["n"].max())
                        except Exception:
                            inferred = None
                    seeds_for_title = n_seeds if n_seeds is not None else inferred
                    title = f"{p}: Normalized regret curves (median ± IQR"
                    if seeds_for_title is not None:
                        title += f", N={seeds_for_title} seeds"
                    title += f"), noise={int(round(100*float(n)))}%"
                    ax.set_title(title)
                    ax.set_xlabel("Iteration")
                    ax.set_ylabel(ylabel)
                    ax.legend(loc="upper right")
                    try:
                        _apply_integer_iteration_axis(ax, int(T))
                    except Exception:
                        pass
                    _ticks(ax, int(T), n_labels=10)
                    fname = f"regret_norm_curves_{p}_T{int(T)}_N{int(round(100*float(n)))}.png"
                    plt.savefig(out / fname, dpi=150, bbox_inches="tight")
                    plt.close(fig)
        else:
            for n in sorted(aggp["noise_pct"].unique()):
                sub = aggp[aggp["noise_pct"] == n]
                if sub.empty:
                    continue
                present = sorted(sub["method"].unique())
                methods = [m for m in (
                    method_order or present) if m in present]
                fig, ax = plt.subplots(figsize=(8, 4.8))
                for m in methods:
                    sm = sub[sub["method"] == m].sort_values("iter")
                    x = sm["iter"].to_numpy() + 1
                    ax.plot(x, sm["median"], label=m)
                    ax.fill_between(x, sm["q25"], sm["q75"], alpha=0.15)
                inferred = None
                if "n" in sub.columns and sub["n"].notna().any():
                    try:
                        inferred = int(sub["n"].max())
                    except Exception:
                        inferred = None
                seeds_for_title = n_seeds if n_seeds is not None else inferred
                T_inf = int(sub["iter"].max()) + \
                    1 if "iter" in sub.columns else None
                title = f"{p}: Normalized regret curves (median ± IQR"
                if seeds_for_title is not None:
                    title += f", N={seeds_for_title} seeds"
                title += f"), noise={int(round(100*float(n)))}%"
                ax.set_title(title)
                ax.set_xlabel("Iteration")
                ax.set_ylabel(ylabel)
                ax.legend(loc="upper right")
                try:
                    _apply_integer_iteration_axis(ax, int(T_inf))
                except Exception:
                    pass
                _ticks(ax, int(T_inf), n_labels=10)
                fname = f"regret_norm_curves_{p}_T{T_inf}_N{int(round(100*float(n)))}.png"
                plt.savefig(out / fname, dpi=150, bbox_inches="tight")
                plt.close(fig)


def plot_ybest_curves(
    df: pd.DataFrame,
    outdir: Union[str, Path],
    n_seeds: Optional[int] = None,
    method_order: Optional[List[str]] = None,
) -> None:
    out = _ensure_outdir(Path(outdir) / "plots" / "best_value")
    if "noise_pct" not in df.columns:
        df = df.copy()
        df["noise_pct"] = 0.0
    has_budget = ("budget" in df.columns) and df["budget"].notna().any()

    def _ticks(ax, T, n_labels=10):
        T = int(T)
        step = max(1, (T + n_labels - 1) // n_labels)
        ticks = list(range(step, T + 1, step))
        if not ticks or ticks[-1] != T:
            ticks.append(T)
        ax.set_xlim(1, T)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])

    for p in sorted(df["problem"].unique()):
        subp = df[df["problem"] == p]
        if has_budget:
            for T in sorted({int(b) for b in subp["budget"].dropna().unique()}):
                subT = subp[subp["budget"] == T]
                for n in sorted(subT["noise_pct"].unique()):
                    sub = subT[subT["noise_pct"] == n]
                    present = sorted(sub["method"].unique())
                    methods = [m for m in (
                        method_order or present) if m in present]
                    fig, ax = plt.subplots(figsize=(8, 4.8))
                    for m in methods:
                        sm = sub[sub["method"] == m].sort_values("iter")
                        x = sm["iter"].to_numpy() + 1
                        ax.plot(x, sm["best_so_far"], label=m)
                    inferred = None
                    if "n" in sub.columns and sub["n"].notna().any():
                        try:
                            inferred = int(sub["n"].max())
                        except Exception:
                            inferred = None
                    seeds_for_title = n_seeds if n_seeds is not None else inferred
                    title = f"{p}: Best-so-far (median ± IQR"
                    if seeds_for_title is not None:
                        title += f", N={seeds_for_title} seeds"
                    title += f"), noise={int(round(100*float(n)))}%"
                    ax.set_title(title)
                    ax.set_xlabel("Iteration")
                    ax.set_ylabel("Best observed value")
                    ax.legend(loc="upper right")
                    try:
                        _apply_integer_iteration_axis(ax, int(T))
                    except Exception:
                        pass
                    _ticks(ax, int(T), n_labels=10)
                    plt.savefig(
                        out / f"ybest_curves_{p}_T{int(T)}_N{int(round(100*float(n)))}.png", dpi=150, bbox_inches="tight")
                    plt.close(fig)
        else:
            for n in sorted(subp["noise_pct"].unique()):
                sub = subp[subp["noise_pct"] == n]
                present = sorted(sub["method"].unique())
                methods = [m for m in (
                    method_order or present) if m in present]
                fig, ax = plt.subplots(figsize=(8, 4.8))
                for m in methods:
                    sm = sub[sub["method"] == m].sort_values("iter")
                    x = sm["iter"].to_numpy() + 1
                    ax.plot(x, sm["best_so_far"], label=m)
                inferred = None
                if "n" in sub.columns and sub["n"].notna().any():
                    try:
                        inferred = int(sub["n"].max())
                    except Exception:
                        inferred = None
                seeds_for_title = n_seeds if n_seeds is not None else inferred
                T_inf = int(sub["iter"].max()) + \
                    1 if "iter" in sub.columns else None
                title = f"{p}: Best-so-far (median ± IQR"
                if seeds_for_title is not None:
                    title += f", N={seeds_for_title} seeds"
                title += f"), noise={int(round(100*float(n)))}%"
                ax.set_title(title)
                ax.set_xlabel("Iteration")
                ax.set_ylabel("Best observed value")
                ax.legend(loc="upper right")
                try:
                    _apply_integer_iteration_axis(ax, int(T_inf))
                except Exception:
                    pass
                _ticks(ax, int(T_inf), n_labels=10)
                plt.savefig(
                    out / f"ybest_curves_{p}_T{T_inf}_N{int(round(100*float(n)))}.png", dpi=150, bbox_inches="tight")
                plt.close(fig)


def plot_box_regret_at_T(
    df: pd.DataFrame,
    outdir: Union[str, Path],
    method_order: Optional[List[str]] = None,
    highlight_seed: Optional[int] = 0,
    show_all_seeds: bool = True,
    jitter: float = 0.12,
    metric: str = "regret",           # "regret" or "regret_obs"
    lock_ylim_across_noise: bool = True,
    ypad: float = 0.05,
    y_min_override: Optional[float] = None,
    y_max_override: Optional[float] = None,
    # backward compatibility:
    only_seed: Optional[int] = None,
) -> None:
    """Boxplot of final regret per method (optional observed). One fig per (problem,budget,noise)."""
    if only_seed is not None and highlight_seed is None:
        highlight_seed = int(only_seed)

    value_col = "regret_obs" if metric == "regret_obs" else "regret"
    ylabel = "Final observed regret @ T" if metric == "regret_obs" else "Final regret @ T"

    out = _ensure_outdir(Path(outdir) / "plots" /
                         ("regret" + ("_obs" if metric == "regret_obs" else "")))
    base = _final_rows_per_run(df)
    if base.empty:
        return
    if "noise_pct" not in base.columns:
        base = base.copy()
        base["noise_pct"] = 0.0

    for p in sorted(base["problem"].unique()):
        dfp = base[base["problem"] == p]
        for T in sorted(dfp["budget"].unique()):
            dfpt = dfp[dfp["budget"] == T]

            # y-Achse über Noise für (p,T) fixieren?
            if lock_ylim_across_noise:
                y_min = 0.0 if y_min_override is None else float(
                    y_min_override)
                if y_max_override is None:
                    y_max = dfpt[value_col].max()
                    y_max = (1.0 if not np.isfinite(y_max)
                             else y_max) * (1.0 + ypad)
                else:
                    y_max = float(y_max_override)

            for n in sorted(dfpt["noise_pct"].unique()):
                sub = dfpt[dfpt["noise_pct"] == n]
                if sub.empty:
                    continue

                present = sorted(sub["method"].unique())
                methods = [m for m in (
                    method_order or present) if m in present]
                data = [sub.loc[sub["method"] == m, value_col].to_numpy()
                        for m in methods]
                n_seeds = int(sub["seed"].nunique())

                fig, ax = plt.subplots(
                    figsize=(8, 4.8), )
                bp = ax.boxplot(data, labels=methods, showfliers=False)

                # --- Datenpunkte ---
                if show_all_seeds:
                    for i, m in enumerate(methods, start=1):
                        vals = sub.loc[sub["method"]
                                       == m, value_col].to_numpy()
                        if vals.size == 0:
                            continue
                        xs = i + (np.linspace(-jitter, jitter, vals.size)
                                  if vals.size > 1 else np.array([0.0]))
                        ax.scatter(xs, vals, s=20, c="0.6",
                                   zorder=3, label=None)

                if highlight_seed is not None:
                    hs = sub[sub["seed"] == int(highlight_seed)]
                    for i, m in enumerate(methods, start=1):
                        vals = hs.loc[hs["method"] == m, value_col].to_numpy()
                        if vals.size > 0:
                            ax.scatter(np.repeat(i, vals.size), vals,
                                       s=36, c="black", zorder=4, label=None)

                # --- Legende (All seeds / Highlight) ---
                handles = []
                if show_all_seeds:
                    handles.append(plt.Line2D([0], [0], marker='o', linestyle='None', markersize=6,
                                              color='0.6', label='All seeds'))
                if highlight_seed is not None:
                    handles.append(plt.Line2D([0], [0], marker='o', linestyle='None', markersize=6,
                                              color='black', label=f'Seed {highlight_seed}'))
                if handles:
                    ax.legend(handles=handles, loc="upper right")

                if lock_ylim_across_noise:
                    ax.set_ylim(y_min, y_max)

                # --- Beschriftungen ---
                ax.set_ylabel(ylabel)
                ax.set_title(
                    f"{p}: Regret at final budget (T={int(T)}, seeds={n_seeds}), noise={int(round(100*n))}%")

                # --- Median-Zahl in gleicher Farbe neben die Median-Linie ---
                _annotate_median_labels_next_to_lines(
                    ax, bp, data_len=len(methods))

                fname = f"regret_box_{p}_T{int(T)}_N{int(round(100*n))}{'_obs' if metric=='regret_obs' else ''}.png"
                _safe_savefig(out / fname, dpi=150, bbox_inches="tight")
                plt.close()


def render_bestvalue_boxplots_per_method_grouped_by_noise(
    results_dir: Union[str, Path],
    decimals: int = 3,
    metric: str = "best_so_far",    # oder "best_so_far_obs"
    show_all_seeds: bool = True,
    highlight_seed: Optional[int] = None,  # z.B. 0 oder None
) -> int:
    """
    Für JE Problem und JE Budget T und JE Methode:
      - Boxplot über NOISE-Level (X-Achse = Noise-Level), Seeds als Punkte,
        gleiche Optik wie bestehende Boxplots (Farbe/Box/Median-Zahl).
      - Tabelle: Zeilen = Noise-Level; Spalten = alle Seeds + 'Median'.
    Dateien:
      plots/bestvalue_box_method_<Problem>_<Methode>_T<T>_Nall.png
      plots/bestvalue_box_method_<Problem>_<Methode>_T<T>_Nall_table.png
    """
    import os
    import glob
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    logs_dir = os.path.join(results_dir, "logs")
    files = sorted(glob.glob(os.path.join(logs_dir, "*.csv")))
    if not files:
        print("[per-method] keine Logs gefunden:", logs_dir)
        return 0

    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)

    # Zielspalte robust wählen (best_value Ansicht)
    target_col = None
    for cand in [metric, "best_so_far", "best_so_far_obs"]:
        if cand in df.columns:
            target_col = cand
            break
    if target_col is None:
        print("[per-method] keine geeignete Zielspalte gefunden (best_so_far / _obs).")
        return 0

    # Finale Zeile pro (problem, method, seed, noise_pct, budget)
    if "iter" in df.columns:
        fin = df.sort_values("iter").groupby(
            ["problem", "method", "seed", "noise_pct", "budget"], dropna=False
        ).tail(1)
    else:
        fin = df.groupby(["problem", "method", "seed",
                         "noise_pct", "budget"], dropna=False).tail(1)

    # Rundung Noise für stabile Labels
    fin = fin[["problem", "method", "seed",
               "noise_pct", "budget", target_col]].copy()
    if fin["noise_pct"].dtype.kind in "fc":
        fin["noise_pct"] = fin["noise_pct"].round(3)

    out_plots = Path(results_dir) / "plots"
    out_plots.mkdir(parents=True, exist_ok=True)

    total = 0
    # Konsistente Reihenfolge
    problems = sorted(fin["problem"].unique().tolist(), key=lambda s: str(s))
    for pb in problems:
        dfp = fin[fin["problem"] == pb]
        budgets = sorted(dfp["budget"].unique().tolist())
        for T in budgets:
            dft = dfp[dfp["budget"] == T]
            methods = sorted(
                dft["method"].unique().tolist(), key=lambda s: str(s))
            noises = sorted(dft["noise_pct"].unique().tolist())

            for m in methods:
                dfm = dft[dft["method"] == m]
                if dfm.empty:
                    continue

                # Daten je Noise: Liste von Seed-Vektoren
                data = []
                for nz in noises:
                    vals = dfm[dfm["noise_pct"] ==
                               nz][target_col].astype(float).values
                    data.append(vals)

                # ---- Plot (Boxen = Noise-Level) ----
                fig, ax = plt.subplots(
                    figsize=(8, 4.8), )
                # Stimmen die Box-Parameter mit deinen anderen Boxplots überein:
                bp = ax.boxplot(
                    data,
                    labels=[f"N={nz:g}" for nz in noises],
                    showfliers=False,
                )

                # Seeds als Punkte (gleiche Optik wie bisher)
                if show_all_seeds:
                    jitter = 0.12
                    for i, nz in enumerate(noises, start=1):
                        vals = dfm[dfm["noise_pct"] ==
                                   nz][target_col].astype(float).values
                        if vals.size == 0:
                            continue
                        xs = i + (np.linspace(-jitter, jitter, vals.size)
                                  if vals.size > 1 else np.array([0.0]))
                        ax.scatter(xs, vals, s=20, c="0.6",
                                   zorder=3, label=None)

                # Optional: Highlight-Seed
                if highlight_seed is not None:
                    hs = dfm[dfm["seed"] == int(highlight_seed)]
                    for i, nz in enumerate(noises, start=1):
                        vals = hs[hs["noise_pct"] ==
                                  nz][target_col].astype(float).values
                        if vals.size > 0:
                            ax.scatter(np.repeat(i, vals.size), vals,
                                       s=36, c="black", zorder=4, label=None)

                # Y-Label / Titel wie bei Best-Value Boxplots
                ylabel = "Best value (final iteration)" if target_col.startswith(
                    "best") else target_col
                ax.set_ylabel(ylabel)
                n_seeds = int(dfm["seed"].nunique())
                ax.set_title(f"{pb} — {m} @ T={int(T)} (seeds={n_seeds})")

                # Median-Zahl direkt an die Median-Linie (gleiche Farbe)
                try:
                    _annotate_median_labels_next_to_lines(
                        ax, bp, data_len=len(noises))
                except Exception:
                    pass

                # Keine Legende notwendig (Wunsch)
                # ax.legend().remove()  # explizit keine Legende

                fname = f"bestvalue_box_method_{pb}_{m}_T{int(T)}_Nall.png".replace(
                    " ", "")
                fig.savefig(out_plots / fname, dpi=150)
                plt.close(fig)

                # ---- Tabelle: Zeilen = Noise-Level; Spalten = Seeds + Median ----
                # Pivot: index=noise_pct, columns=seed, values=target_col
                piv = dfm.pivot_table(
                    index="noise_pct", columns="seed", values=target_col, aggfunc="median")
                piv = piv.sort_index()
                # Median-Spalte je Noise
                piv["Median"] = dfm.groupby("noise_pct")[
                    target_col].median().reindex(piv.index).values
                # Runden
                piv = piv.astype(float).round(decimals)

                # Render als PNG
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(
                    figsize=(max(6.0, 0.9 * (len(piv.columns) + 2)), 0.5 * (len(piv.index) + 2)))
                ax.axis("off")
                the_table = ax.table(
                    cellText=piv.reset_index().values,
                    colLabels=[
                        "Noise"] + [f"Seed {s}" for s in piv.columns if s != "Median"] + ["Median"],
                    loc="center",
                )
                the_table.auto_set_font_size(False)
                the_table.set_fontsize(8)
                the_table.scale(1, 1.1)
                ax.set_title(
                    f"{pb} — {m} @ T={int(T)} (per noise: seeds + Median)", fontsize=9, pad=8)

                tname = fname.replace(".png", "_table.png")
                fig.tight_layout()
                fig.savefig(out_plots / tname, dpi=200)
                plt.close(fig)

                total += 1

    print(
        f"[per-method] erzeugt {total} per-method grouped Boxplot(s) + Tabellen (Nall).")
    return total


# def render_grouped_boxplots_by_method_noise(results_dir, decimals=3):
#     """
#     Erzeugt gruppierte Boxplots für JE Problem und JE Budget T:
#     - x-Achse: method
#     - mehrere Boxen je Methode: noise_pct Levels side-by-side
#     - Datei: bestvalue_box_<Problem>_T<T>_Nall.png
#     Dazu eine passende Tabelle (Median/IQR, 3 Nachkommastellen):
#     - Datei: bestvalue_box_<Problem>_T<T>_Nall_table.png

#     Erwartet Logs in results_dir/logs/*.csv mit Spalten:
#       problem, method, seed, noise_pct, iter, budget, regret (oder regret_norm/obs)
#     Nutzt standardmäßig 'regret' als Zielgröße. Wenn 'regret' fehlt, versucht 'regret_norm', dann 'best_so_far'.
#     """
#     import os
#     import glob
#     import numpy as np
#     import pandas as pd
#     import matplotlib.pyplot as plt
#     from matplotlib.ticker import MaxNLocator, FormatStrFormatter

#     logs_dir = os.path.join(results_dir, "logs")
#     files = sorted(glob.glob(os.path.join(logs_dir, "*.csv")))
#     if not files:
#         print("[grouped] keine Logs gefunden:", logs_dir)
#         return 0

#     df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
#     # Zielspalte robust wählen
#     target_col = None
#     for cand in ["regret", "regret_norm", "best_so_far"]:
#         if cand in df.columns:
#             target_col = cand
#             break
#     if target_col is None:
#         print(
#             "[grouped] keine geeignete Zielspalte gefunden (regret/regret_norm/best_so_far).")
#         return 0

#     # finale Iteration je (problem, method, seed, noise_pct, budget)
#     if "iter" in df.columns:
#         df["_rank_iter"] = df.groupby(["problem", "method", "seed", "noise_pct", "budget"], dropna=False)[
#             "iter"].transform("idxmax")
#         # Alternative robust: max iter je Gruppe
#         fin = df.sort_values("iter").groupby(
#             ["problem", "method", "seed", "noise_pct", "budget"], dropna=False).tail(1)
#     else:
#         # Fallback: nehme pro Gruppe die letzte Zeile
#         fin = df.groupby(["problem", "method", "seed",
#                          "noise_pct", "budget"], dropna=False).tail(1)

#     # nur sinnvolle Spalten
#     keep = ["problem", "method", "seed", "noise_pct", "budget", target_col]
#     fin = fin[keep].copy()

#     # Round noise for stable grouping/labels
#     if fin["noise_pct"].dtype.kind in "fc":
#         fin["noise_pct"] = fin["noise_pct"].round(3)

#     out_plots = os.path.join(results_dir, "plots")
#     os.makedirs(out_plots, exist_ok=True)

#     total = 0
#     # Für konsistente Reihenfolge
#     problems = sorted(fin["problem"].unique().tolist(), key=lambda s: str(s))
#     for pb in problems:
#         dfp = fin[fin["problem"] == pb]
#         budgets = sorted(dfp["budget"].unique().tolist())
#         for T in budgets:
#             dft = dfp[dfp["budget"] == T].copy()
#             methods = sorted(
#                 dft["method"].unique().tolist(), key=lambda s: str(s))
#             noises = sorted(dft["noise_pct"].unique().tolist())

#             # Vorbereitung für grouped boxplot: pro Methode die Daten je Noise
#             data_by_method_noise = {m: [] for m in methods}
#             for m in methods:
#                 for nz in noises:
#                     v = dft[(dft["method"] == m) & (dft["noise_pct"]
#                                                     == nz)][target_col].astype(float).values
#                     data_by_method_noise[m].append(v)

#             # ----- Plot -----
#             fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(methods)), 4.6))
#             # Platzierung: für jede Methode ein "Cluster", darin k=noises Boxen mit Offset
#             k = len(noises)
#             positions = []
#             labels = []
#             base_x = np.arange(len(methods))
#             width = 0.8
#             # Offsets zentrieren
#             if k > 1:
#                 offsets = np.linspace(-width/2 + width /
#                                       (2*k), width/2 - width/(2*k), k)
#             else:
#                 offsets = [0.0]

#             all_boxes = []
#             for mi, m in enumerate(methods):
#                 for ki, nz in enumerate(noises):
#                     x = base_x[mi] + offsets[ki]
#                     positions.append(x)
#                     labels.append(f"{m}\nN={nz:g}")
#                     vals = data_by_method_noise[m][ki]
#                     if len(vals) > 0:
#                         bp = ax.boxplot(vals, positions=[
#                                         x], widths=width/(k+0.2), patch_artist=True)
#                         all_boxes.append(bp)
#                     else:
#                         # leere Box: überspringen
#                         pass

#             ax.set_title(
#                 f"{pb} — grouped by method × noise @ T={T}", fontsize=11)
#             ax.set_xlabel("Methode × Noise")
#             ax.set_ylabel(target_col)
#             ax.xaxis.set_major_locator(MaxNLocator(integer=True))
#             # nur Methoden beschriften, Noise kommt in Legende
#             ax.set_xticks(base_x)
#             ax.set_xticklabels(methods, rotation=0, ha="center")

#             # Legende aus Noise-Handles bauen (einfach Marker simulieren)
#             from matplotlib.lines import Line2D
#             legend_elems = [
#                 Line2D([0], [0], color="black", lw=1, label=f"N={nz:g}") for nz in noises]
#             ax.legend(handles=legend_elems, title="Noise",
#                       loc="best", frameon=True)

#             ax.grid(True, axis="y", alpha=0.25)

#             plot_name = f"bestvalue_box_{pb}_T{int(T)}_Nall.png".replace(
#                 " ", "")
#             plot_path = os.path.join(out_plots, plot_name)
#             fig.tight_layout()
#             fig.savefig(plot_path, dpi=200)
#             plt.close(fig)

#             # ----- Tabelle (Median + IQR je Methode×Noise) -----
#             rows = []
#             for m in methods:
#                 row = {"method": m}
#                 for nz in noises:
#                     vals = dft[(dft["method"] == m) & (
#                         dft["noise_pct"] == nz)][target_col].astype(float)
#                     if len(vals) == 0:
#                         med = np.nan
#                         iqr = np.nan
#                     else:
#                         q1 = vals.quantile(0.25)
#                         med = vals.median()
#                         q3 = vals.quantile(0.75)
#                         iqr = q3 - q1
#                     row[f"N={nz:g} median"] = med
#                     row[f"N={nz:g} IQR"] = iqr
#                 rows.append(row)
#             tab = pd.DataFrame(rows)
#             # Runden
#             num_cols = [c for c in tab.columns if c != "method"]
#             for c in num_cols:
#                 tab[c] = tab[c].astype(float).round(decimals)

#             # Render als PNG
#             import matplotlib.pyplot as plt
#             fig, ax = plt.subplots(
#                 figsize=(max(5.5, 1.2 * len(tab.columns)), 0.5 * (len(tab) + 2)))
#             ax.axis("off")
#             the_table = ax.table(cellText=tab.values,
#                                  colLabels=tab.columns,
#                                  loc="center")
#             the_table.auto_set_font_size(False)
#             the_table.set_fontsize(8)
#             the_table.scale(1, 1.1)
#             ax.set_title(
#                 f"{pb} @ T={int(T)} — grouped table (median/IQR)", fontsize=9, pad=8)

#             table_name = plot_name.replace(".png", "_table.png")
#             table_path = os.path.join(out_plots, table_name)
#             fig.tight_layout()
#             fig.savefig(table_path, dpi=200)
#             plt.close(fig)

#             total += 1

#     print(f"[grouped] erzeugt {total} grouped boxplot(s) + Tabellen (Nall).")
#     return total


def plot_box_regret_norm_at_T(
    df: pd.DataFrame,
    outdir: Union[str, Path],
    method_order: Optional[List[str]] = None,
    highlight_seed: Optional[int] = 0,
    show_all_seeds: bool = True,
    jitter: float = 0.12,
    # backward compatibility:
    only_seed: Optional[int] = None,
) -> None:
    """Boxplot of final normalized regret (0..1). One fig per (problem,budget,noise)."""
    if only_seed is not None and highlight_seed is None:
        highlight_seed = int(only_seed)

    out = _ensure_outdir(Path(outdir) / "plots" / "regret_norm")
    dfn = df.copy()
    if "regret_norm" not in dfn.columns:
        return
    dfn = dfn.dropna(subset=["regret_norm"]).copy()
    if "noise_pct" not in dfn.columns:
        dfn["noise_pct"] = 0.0
    df_final = _final_rows_per_run(dfn)

    for p in sorted(df_final["problem"].unique()):
        dfp = df_final[df_final["problem"] == p]
        for T in sorted(dfp["budget"].unique()):
            dfpt = dfp[dfp["budget"] == T]
            for n in sorted(dfpt["noise_pct"].unique()):
                sub = dfpt[dfpt["noise_pct"] == n]
                if sub.empty:
                    continue

                present = sorted(sub["method"].unique())
                methods = [m for m in (
                    method_order or present) if m in present]
                data = [sub.loc[sub["method"] == m, "regret_norm"].to_numpy()
                        for m in methods]
                n_seeds = int(sub["seed"].nunique())

                fig, ax = plt.subplots(
                    figsize=(8, 4.8), )
                bp = ax.boxplot(data, labels=methods, showfliers=False)

                # Punkte
                if show_all_seeds:
                    for i, m in enumerate(methods, start=1):
                        yvals = sub.loc[sub["method"] ==
                                        m, "regret_norm"].to_numpy()
                        if yvals.size == 0:
                            continue
                        xs = np.array([i]) if yvals.size == 1 else i + \
                            np.linspace(-jitter, jitter, yvals.size)
                        ax.scatter(xs, yvals, s=20, c="0.6",
                                   zorder=3, label=None)

                if highlight_seed is not None:
                    hs = sub[sub["seed"] == int(highlight_seed)]
                    for i, m in enumerate(methods, start=1):
                        yvals = hs.loc[hs["method"] ==
                                       m, "regret_norm"].to_numpy()
                        if yvals.size > 0:
                            ax.scatter(np.repeat(i, yvals.size), yvals,
                                       s=36, c="black", zorder=4, label=None)

                handles = []
                if show_all_seeds:
                    handles.append(plt.Line2D([0], [0], marker='o', linestyle='None', markersize=6,
                                              color='0.6', label='All seeds'))
                if highlight_seed is not None:
                    handles.append(plt.Line2D([0], [0], marker='o', linestyle='None', markersize=6,
                                              color='black', label=f'Seed {highlight_seed}'))
                if handles:
                    ax.legend(handles=handles, loc="upper right")

                ax.set_ylim(0, 1)
                ax.set_ylabel("Final normalized regret @ T")
                ax.set_title(
                    f"{p}: Normalized regret at final budget (T={int(T)}, seeds={n_seeds}), noise={int(round(100*n))}%")

                # Median-Zahl neben Linie
                _annotate_median_labels_next_to_lines(
                    ax, bp, data_len=len(methods))

                plt.savefig(
                    out / f"regret_norm_box_{p}_T{int(T)}_N{int(round(100*n))}.png", dpi=150)
                plt.close()


def plot_box_best_value_at_T(
    df: pd.DataFrame,
    outdir: Union[str, Path],
    method_order: Optional[List[str]] = None,
    highlight_seed: Optional[int] = 0,
    show_all_seeds: bool = True,
    jitter: float = 0.12,
    metric: str = "best_so_far",    # or "best_so_far_obs"
    # backward compatibility:
    only_seed: Optional[int] = None,
) -> None:
    """Boxplot of final best value per method. One fig per (problem,budget,noise)."""
    if only_seed is not None and highlight_seed is None:
        highlight_seed = int(only_seed)

    value_col = "best_so_far_obs" if metric == "best_so_far_obs" else "best_so_far"
    ylabel = "Final observed best value @ T" if metric.endswith(
        "_obs") else "Final best value @ T"

    out = _ensure_outdir(Path(outdir) / "plots" / ("best_value" +
                         ("_obs" if metric.endswith("_obs") else "")))
    base = _final_rows_per_run(df)
    if base.empty:
        return
    if "noise_pct" not in base.columns:
        base = base.copy()
        base["noise_pct"] = 0.0

    for p in sorted(base["problem"].unique()):
        dfp = base[base["problem"] == p]
        for T in sorted(dfp["budget"].unique()):
            dfpt = dfp[dfp["budget"] == T]
            for n in sorted(dfpt["noise_pct"].unique()):
                sub = dfpt[dfpt["noise_pct"] == n]
                if sub.empty:
                    continue

                present = sorted(sub["method"].unique())
                methods = [m for m in (
                    method_order or present) if m in present]
                data = [sub.loc[sub["method"] == m, value_col].to_numpy()
                        for m in methods]
                n_seeds = int(sub["seed"].nunique())

                fig, ax = plt.subplots(
                    figsize=(8, 4.8), )
                bp = ax.boxplot(data, labels=methods, showfliers=False)

                # Punkte
                if show_all_seeds:
                    for i, m in enumerate(methods, start=1):
                        vals = sub.loc[sub["method"]
                                       == m, value_col].to_numpy()
                        if vals.size == 0:
                            continue
                        xs = i + (np.linspace(-jitter, jitter, vals.size)
                                  if vals.size > 1 else np.array([0.0]))
                        ax.scatter(xs, vals, s=20, c="0.6",
                                   zorder=3, label=None)

                if highlight_seed is not None:
                    hs = sub[sub["seed"] == int(highlight_seed)]
                    for i, m in enumerate(methods, start=1):
                        vals = hs.loc[hs["method"] == m, value_col].to_numpy()
                        if vals.size > 0:
                            ax.scatter(np.repeat(i, vals.size), vals,
                                       s=36, c="black", zorder=4, label=None)

                handles = []
                if show_all_seeds:
                    handles.append(plt.Line2D([0], [0], marker='o', linestyle='None', markersize=6,
                                              color='0.6', label='All seeds'))
                if highlight_seed is not None:
                    handles.append(plt.Line2D([0], [0], marker='o', linestyle='None', markersize=6,
                                              color='black', label=f'Seed {highlight_seed}'))
                if handles:
                    ax.legend(handles=handles, loc="upper right")

                ax.set_ylabel(ylabel)
                ax.set_title(
                    f"{p}: Best value at final budget (T={int(T)}, seeds={n_seeds}), noise={int(round(100*n))}%")

                # Median-Zahl neben Linie
                _annotate_median_labels_next_to_lines(
                    ax, bp, data_len=len(methods))

                fname = f"bestvalue_box_{p}_T{int(T)}_N{int(round(100*n))}{'_obs' if metric.endswith('_obs') else ''}.png"
                plt.savefig(out / fname, dpi=150, bbox_inches="tight")
                plt.close()


# ----- Plots: Heatmaps (rank / winrate) -------------------------------------

def plot_rank_heatmap(
    ranks: pd.DataFrame,
    group_cols: Sequence[str],
    outdir: Union[str, Path],
    title: str = "Average rank",
    filename: str = "rank_heatmap.png",
) -> None:
    """
    Heatmap der durchschnittlichen Ränge pro Gruppe × Methode.
    Wenn 'noise_pct' vorhanden ist → eine Heatmap pro Noise-Level (Suffix im Dateinamen).
    Erwartet: group_cols + ['method','rank'] (+ optional 'noise_pct').
    """
    out = _ensure_outdir(Path(outdir))
    need = set(group_cols).union({"method", "rank"})
    missing = [c for c in need if c not in ranks.columns]
    if missing:
        raise ValueError(f"plot_rank_heatmap(): missing columns: {missing}")

    has_noise = ("noise_pct" in ranks.columns)
    noise_levels = [None] if not has_noise else sorted(
        ranks["noise_pct"].unique())

    def _make_label(row) -> str:
        return _group_label(row, group_cols)

    for nlev in noise_levels:
        sub = ranks.copy()
        noise_txt, suf = "", ""
        if nlev is not None:
            sub = sub[np.isclose(sub["noise_pct"], nlev)]
            noise_txt, suf = _fmt_noise(nlev)
        if sub.empty:
            continue

        sub = sub.copy()
        sub["group"] = sub.apply(lambda r: _make_label(r), axis=1)

        pivot = sub.pivot_table(
            index="group", columns="method", values="rank", aggfunc="mean").sort_index()
        n_methods = max(1, pivot.shape[1])
        vmin, vmax = 1.0, float(n_methods)

        fig, ax = plt.subplots(
            figsize=(max(6, 1.1 * n_methods + 2),
                     max(4, 0.5 * len(pivot.index) + 2)),
        )
        im = ax.imshow(pivot.values, cmap="viridis",
                       vmin=vmin, vmax=vmax, aspect="auto")

        ax.set_xticks(np.arange(pivot.shape[1]))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
        ax.set_yticks(np.arange(pivot.shape[0]))
        ax.set_yticklabels(pivot.index)

        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.2f}", ha="center",
                            va="center", color="white", fontsize=8)

        cb = fig.colorbar(im, ax=ax)
        cb.set_label("Avg. rank (lower is better)")

        ttl = title if nlev is None else f"{title} — noise={noise_txt}"
        ax.set_title(ttl)

        fname = _add_suffix_to_filename(filename, suf)
        fig.savefig(out / fname, dpi=150)
        plt.close(fig)


def plot_winrate_heatmap(
    wins: pd.DataFrame,
    group_cols: Sequence[str],
    outdir: Union[str, Path],
    title: str = "Win rate",
    filename: str = "winrate_heatmap.png",
) -> None:
    """
    Heatmap der Winrate (= Anteil Seeds mit Best-Rank).
    Erwartet: group_cols + ['method','win_rate'] (+ optional 'noise_pct','n_seeds').
    """
    out = _ensure_outdir(Path(outdir))
    need = set(group_cols).union({"method", "win_rate"})
    missing = [c for c in need if c not in wins.columns]
    if missing:
        raise ValueError(f"plot_winrate_heatmap(): missing columns: {missing}")

    has_noise = ("noise_pct" in wins.columns)
    noise_levels = [None] if not has_noise else sorted(
        wins["noise_pct"].unique())

    def _make_label(row) -> str:
        return _group_label(row, group_cols)

    for nlev in noise_levels:
        sub = wins.copy()
        noise_txt, suf = "", ""
        if nlev is not None:
            sub = sub[np.isclose(sub["noise_pct"], nlev)]
            noise_txt, suf = _fmt_noise(nlev)
        if sub.empty:
            continue

        sub = sub.copy()
        sub["group"] = sub.apply(lambda r: _make_label(r), axis=1)

        pivot = sub.pivot_table(
            index="group", columns="method", values="win_rate", aggfunc="mean").sort_index()

        fig, ax = plt.subplots(
            figsize=(max(6, 1.1 * pivot.shape[1] + 2),
                     max(4, 0.5 * len(pivot.index) + 2)),
        )
        im = ax.imshow(pivot.values, cmap="viridis",
                       vmin=0.0, vmax=1.0, aspect="auto")

        ax.set_xticks(np.arange(pivot.shape[1]))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
        ax.set_yticks(np.arange(pivot.shape[0]))
        ax.set_yticklabels(pivot.index)

        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.2f}", ha="center",
                            va="center", color="white", fontsize=8)

        cb = fig.colorbar(im, ax=ax)
        cb.set_label("Win rate (0..1)")

        ttl = title if nlev is None else f"{title} — noise={noise_txt}"
        ax.set_title(ttl)

        fname = _add_suffix_to_filename(filename, suf)
        fig.savefig(out / fname, dpi=150)
        plt.close(fig)


# ----- Tables & End-to-end evaluation ---------------------------------------

def export_best_at_T_table(
    df: pd.DataFrame,
    outdir: Union[str, Path],
    kind: str = "true",
    decimals: Optional[int] = None,
) -> Path:
    """
    Schreibe eine Tabelle mit Bestwerten @T je (problem, method, seed [,noise_pct,budget]).
    kind:
      - "true" -> best_so_far
      - "obs"  -> best_so_far_obs
      - "norm" -> regret_norm (falls vorhanden)
    decimals: Anzahl Nachkommastellen für Float-Spalten (Default: global TABLE_DECIMALS oder 3).
    """
    outdir = Path(outdir) / "tables"
    outdir.mkdir(parents=True, exist_ok=True)

    # Spalte bestimmen
    if kind == "true":
        val_col = "best_so_far"
        out_name = "best_at_T_by_seed_method_true.csv"
    elif kind == "obs":
        val_col = "best_so_far_obs"
        out_name = "best_at_T_by_seed_method_obs.csv"
    elif kind == "norm":
        val_col = "regret_norm"
        out_name = "best_at_T_by_seed_method_norm.csv"
    else:
        raise ValueError(f"Unsupported kind: {kind}")

    # Index-Spalten, nur wenn vorhanden
    idx_cols = [c for c in ("problem", "method", "seed",
                            "noise_pct", "budget") if c in df.columns]

    if "iter" not in df.columns:
        raise ValueError(
            "export_best_at_T_table(): DataFrame must contain an 'iter' column.")

    rows = []
    for k, sub in df.groupby(idx_cols, dropna=False):
        sub = sub.sort_values("iter")
        T = int(sub["iter"].max())
        fin = sub[sub["iter"] == T]

        rec = {}
        if isinstance(k, tuple):
            for name, val in zip(idx_cols, k):
                rec[name] = val
        else:
            if idx_cols:
                rec[idx_cols[0]] = k
        rec["T"] = T

        if val_col in fin.columns and not fin.empty:
            # falls mehrere Zeilen bei gleichem T -> min wählen (robust)
            try:
                rec[f"{val_col}@T"] = float(fin[val_col].min())
            except Exception:
                # z.B. wenn Spalte nicht numerisch ist
                rec[f"{val_col}@T"] = np.nan
        else:
            rec[f"{val_col}@T"] = np.nan

        rows.append(rec)

    tab = pd.DataFrame(rows)

    # Spaltenreihenfolge hübsch anordnen
    preferred_order = [c for c in (
        "problem", "method", "seed", "noise_pct", "budget", "T") if c in tab.columns]
    value_col_out = f"{val_col}@T"
    other_cols = [
        c for c in tab.columns if c not in preferred_order + [value_col_out]]
    tab = tab[[*preferred_order, value_col_out, *other_cols]]

    # Sortierung
    sort_cols = [c for c in (
        "problem", "budget", "method", "seed", "noise_pct") if c in tab.columns]
    if sort_cols:
        tab = tab.sort_values(
            sort_cols, kind="mergesort").reset_index(drop=True)

    # Rundung / Formatierung
    do_round = bool(globals().get("ROUND_BEST_AT_T", True))
    dec = int(decimals if decimals is not None else globals().get(
        "TABLE_DECIMALS", 3))
    float_fmt = None
    if do_round:
        num_cols = tab.select_dtypes(
            include=["float64", "float32", "float16", "float", "double"]).columns
        if len(num_cols):
            tab[num_cols] = tab[num_cols].round(dec)
        float_fmt = f"%.{dec}f"

    path = outdir / out_name
    tab.to_csv(path, index=False, float_format=float_fmt)
    return path


def save_rank_tables(
    df: pd.DataFrame,
    outdir: Union[str, Path],
    group_cols: Sequence[str] = ("problem", "budget"),
    metrics: Sequence[str] = ("final_regret", "auc_regret"),
) -> Dict[str, str]:
    """
    Write CSVs for rank and winrate per group.
    """
    outdir = Path(outdir) / "tables"
    outdir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for metric in metrics:
        ranks = rank_by_group(
            df, metric=metric, group_cols=group_cols, seed_agg="median")
        winr = winrate_by_group(df, metric=metric, group_cols=group_cols)
        p1 = outdir / f"rank_by_group_{metric}.csv"
        p2 = outdir / f"winrate_by_group_{metric}.csv"
        ranks.to_csv(p1, index=False)
        winr.to_csv(p2, index=False)
        paths[f"rank_{metric}"] = str(p1)
        paths[f"winrate_{metric}"] = str(p2)
    return paths


def export_auc_table_from_agg(agg_df: pd.DataFrame, outdir: Path, label: str = "true",
                              value_col: str = "median") -> pd.DataFrame:
    """
    Compute AUC from an aggregated curves dataframe and save CSV + PNG.
    - agg_df must contain columns: iter, method, and grouping keys like problem, noise_pct, budget.
    - value_col is typically "median".
    """
    outdir = Path(outdir) / "tables"
    outdir.mkdir(parents=True, exist_ok=True)

    group_keys = [k for k in ["problem", "noise_pct",
                              "budget", "method"] if k in agg_df.columns]
    rows = []
    for keys, g in agg_df.groupby(group_keys):
        g = g.sort_values("iter")
        x = g["iter"].to_numpy()
        y = g[value_col].to_numpy()
        auc = float(np.trapz(y, x)) if len(x) > 1 else (
            float(y[-1]) if len(y) else float("nan"))
        row = dict(zip(group_keys, keys))
        row[f"AUC_{label}"] = auc
        rows.append(row)

    df_auc = pd.DataFrame(rows).sort_values(group_keys, ignore_index=True)
    csv_path = outdir / f"auc_{label}.csv"
    df_auc.to_csv(csv_path, index=False)

    # Render a compact PNG table for quick inspection
    try:
        fig, ax = plt.subplots(
            figsize=(max(6, 0.9*len(df_auc.columns)), max(1.6, 0.38*len(df_auc))))
        ax.axis("off")
        ax.table(cellText=df_auc.values, colLabels=df_auc.columns,
                 loc="center", cellLoc="right")
        fig.tight_layout()
        fig.savefig(outdir / f"auc_{label}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print("[warn] AUC PNG render:", e)

    return df_auc


def save_tables(results_dir: Union[str, Path], df: pd.DataFrame) -> Dict[str, str]:
    """
    Write CSV tables for AUC(regret) summary and average ranks.
    """
    out_tables = Path(results_dir) / "tables"
    out_tables.mkdir(parents=True, exist_ok=True)

    auc_runs = compute_auc_regret(df, normalize_by_T=True)
    auc_summary = summarize_auc_regret(auc_runs)
    auc_runs.to_csv(out_tables / "auc_regret_runs.csv", index=False)
    auc_summary.to_csv(out_tables / "auc_regret_summary.csv", index=False)

    avg_rank_final = compute_average_rank(df, metric="final_regret")
    avg_rank_auc = compute_average_rank(df, metric="auc_regret")
    avg_rank_final.to_csv(
        out_tables / "average_rank_final_regret.csv", index=False)
    avg_rank_auc.to_csv(
        out_tables / "average_rank_auc_regret.csv", index=False)

    return {
        "auc_runs": str(out_tables / "auc_regret_runs.csv"),
        "auc_summary": str(out_tables / "auc_regret_summary.csv"),
        "avg_rank_final": str(out_tables / "average_rank_final_regret.csv"),
        "avg_rank_auc": str(out_tables / "average_rank_auc_regret.csv"),
    }


def evaluate_folder(results_dir: Union[str, Path]) -> Dict[str, Any]:
    """
    End-to-end: Load logs and produce standard plots next to the CSVs.
    Writes:
      plots/regret_curves/*, plots/regret_norm_curves/*,
      plots/regret/*, plots/regret_norm/*, plots/best_value/*, plots/ybest/*
      tables/*
    """
    results_dir = Path(results_dir)
    df = load_logs(results_dir)

    n_seeds_total = int(df["seed"].nunique()) if "seed" in df.columns else None

    # Curves
    agg_r = aggregate_regret(df)
    plot_regret_curves(agg_r, results_dir, n_seeds=n_seeds_total)

    df = add_normalized_regret(df)
    agg_rn = aggregate_regret_norm(df)
    plot_regret_norm_curves(agg_rn, results_dir, n_seeds=n_seeds_total)

    # Boxplots
    METHOD_ORDER = ["BO_EI", "BO_KG", "DOE_LHS", "DOE_FF", "Random"]
    plot_box_regret_at_T(
        df, results_dir, method_order=METHOD_ORDER, highlight_seed=0)
    plot_box_regret_norm_at_T(
        df, results_dir, method_order=METHOD_ORDER, highlight_seed=0)
    plot_box_best_value_at_T(
        df, results_dir, method_order=METHOD_ORDER, highlight_seed=0)

    # y_best
    plot_ybest_curves(df, results_dir, n_seeds=n_seeds_total)

    # tables
    save_tables(results_dir, df)

    return {"plots": str(results_dir / "plots"), "tables": str(results_dir / "tables")}


def save_csv_as_table_png(csv_path: Union[str, Path],
                          out_png: Union[str, Path],
                          title: Optional[str] = None,
                          max_rows: int = 60,
                          fontsize: int = 9) -> Path:
    """
    Rendert eine CSV als PNG-Tabelle (matplotlib.table).
    - max_rows: begrenze Darstellung (Rest wird abgeschnitten, CSV bleibt vollständig)
    """
    csv_path = Path(csv_path)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    show_df = df.head(max_rows).copy()

    # einfache Breiten-/Höhen-Heuristik
    n_rows, n_cols = show_df.shape
    cell_w = 1.2   # Zoll pro Spalte
    cell_h = 0.35  # Zoll pro Zeile
    fig_w = max(6.0, min(22.0, 0.3 + cell_w * n_cols))
    fig_h = max(2.0, min(18.0, 0.6 + cell_h * (n_rows + (1 if title else 0))))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), )
    ax.axis("off")

    if title:
        ax.set_title(title, fontsize=fontsize+1, pad=8)

    table = ax.table(cellText=show_df.values,
                     colLabels=list(show_df.columns),
                     loc="upper left",
                     cellLoc="left",
                     colLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    # Spaltenbreiten auto (leicht größer)
    for k, cell in table.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if k[0] == 0:
            cell.set_text_props(weight="bold")

    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    return out_png


def save_best_at_T_tables_as_png(
    results_dir: Union[str, Path],
    decimals: Optional[int] = 3,
    max_rows: int = 200,
    fontsize: int = 9,
    dpi: int = 150,
) -> Dict[str, str]:
    """
    Rendert die Best-at-T CSV-Tabellen als PNG mit kontrollierter Dezimalzahl.

    Erwartete CSV-Dateien (liegen unter <results_dir>/tables):
      - best_at_T_by_seed_method_true.csv
      - best_at_T_by_seed_method_obs.csv
      - best_at_T_by_seed_method_norm.csv   (optional)

    Parameter
    ---------
    decimals : Anzahl Nachkommastellen für Float-Spalten (Default 3)
    max_rows : maximale Zeilen pro PNG (zur Sicherheit)
    fontsize : Tabellen-Fontgröße
    dpi      : Ausgabe-DPI

    Returns
    -------
    dict: Mapping { "true": <png_path>, "obs": <png_path>, "norm": <png_path> } für vorhandene Tabellen.
    """
    results_dir = Path(results_dir)
    tdir = results_dir / "tables"
    tdir.mkdir(parents=True, exist_ok=True)

    csvs = {
        "true": tdir / "best_at_T_by_seed_method_true.csv",
        "obs":  tdir / "best_at_T_by_seed_method_obs.csv",
        "norm": tdir / "best_at_T_by_seed_method_norm.csv",
    }

    pngs: Dict[str, str] = {}

    def _format_df(df: pd.DataFrame, dec: int) -> pd.DataFrame:
        df = df.copy()

        # int-Spalten hübsch casten (falls als float eingelesen)
        for col in ("seed", "budget", "T"):
            if col in df.columns:
                with np.errstate(invalid="ignore"):
                    if pd.api.types.is_float_dtype(df[col]):
                        # nur ganze Werte nach int, sonst string beibehalten
                        as_int = df[col].dropna().astype(float)
                        if np.all(np.isclose(as_int, np.round(as_int))):
                            df[col] = df[col].round(0).astype("Int64")

        # Floatspalten runden/als String formatieren
        float_cols = [
            c for c in df.columns if pd.api.types.is_float_dtype(df[c])]
        fmt = f"{{:.{dec}f}}"
        for c in float_cols:
            df[c] = df[c].map(lambda x: fmt.format(x) if pd.notna(x) else "")

        # noise_pct als Anteil belassen (0.500 -> "0.500"), nicht in %
        # Wenn du lieber Prozent möchtest, entkommentieren:
        # if "noise_pct" in df.columns:
        #     df["noise_pct"] = df["noise_pct"].map(lambda x: f"{100*x:.0f}%" if x != "" else "")

        return df

    def _df_to_png(df: pd.DataFrame, title: str, out_path: Path):
        # dynamische Größe (einfach, aber robust)
        n_rows, n_cols = df.shape
        n_rows = min(max_rows, n_rows)
        cell_h = 0.33  # inch
        cell_w = 0.85  # inch
        pad_h = 1.0    # Titel/Abstände
        pad_w = 0.6
        width = min(28, max(6, pad_w*2 + n_cols*cell_w))
        height = min(28, max(2.5, pad_h + n_rows*cell_h))

        fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
        ax.axis("off")

        the_table = ax.table(
            cellText=df.values[:n_rows],
            colLabels=list(df.columns),
            loc="upper center",
            cellLoc="center",
        )
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(fontsize)
        # etwas Luft zwischen den Zeilen
        the_table.scale(1.0, 1.25)

        # Kopfzeile fetten
        for (row, col), cell in the_table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight="bold")

        fig.suptitle(title, y=0.995, fontsize=fontsize+2)
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight", dpi=dpi)
        plt.close(fig)

    for kind, csv_path in csvs.items():
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path)
            df = _format_df(df, decimals if decimals is not None else 3)

            title = {
                "true": "Best-at-T (true best_so_far)",
                "obs":  "Best-at-T (observed best_so_far_obs)",
                "norm": "Best-at-T (normalized regret)",
            }.get(kind, "Best-at-T")

            png_path = csv_path.with_suffix(".png")
            _df_to_png(df, title, png_path)
            pngs[kind] = str(png_path)
        except Exception as e:
            print(f"[warn] Could not render PNG for '{csv_path.name}': {e}")

    return pngs


def save_per_boxplot_side_tables(df_best_at_T, out_dir, by=("problem", "noise_pct"), value_cols=("regret", "regret_obs", "regret_norm")):
    """
    Erzeugt zu jedem Boxplot (gruppiert nach `by`) eine kleine Tabelle als PNG.
    Rundung: 3 Nachkommastellen. Speichert unter out_dir / "tables" / f"box_{group}.png".
    Erwartet df_best_at_T mit Spalten: problem, method, seed, noise_pct, regret, regret_obs, regret_norm, ...
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from textwrap import shorten

    os.makedirs(os.path.join(out_dir, "tables"), exist_ok=True)
    count = 0

    # Gruppiere analog zu den Boxplots (z. B. je Problem × Noise)
    grouped = df_best_at_T.groupby(list(by), dropna=False)
    for group_key, sub in grouped:
        # Aggregiere median/mean (oder was in deinen Boxplots gezeigt wird)
        # Hier: median + IQR, gerundet auf 3 Nachkommastellen
        rows = []
        for m in sorted(sub["method"].unique()):
            row = {"method": m}
            subm = sub[sub["method"] == m]
            for col in value_cols:
                if col not in subm.columns:
                    continue
                q1 = subm[col].quantile(0.25)
                md = subm[col].median()
                q3 = subm[col].quantile(0.75)
                row[f"{col}_median"] = md
                row[f"{col}_IQR"] = q3 - q1
            rows.append(row)
        if not rows:
            continue
        tab = pd.DataFrame(rows)

        # runden
        for c in tab.columns:
            if c != "method" and np.issubdtype(tab[c].dtype, np.number):
                tab[c] = tab[c].astype(float).round(3)

        # sinnvollen Dateinamen bauen
        if isinstance(group_key, tuple):
            gname = "_".join([str(g) for g in group_key])
        else:
            gname = str(group_key)
        gname = gname.replace(" ", "").replace("/", "-")

        # Render als kleine Figure neben Boxplot
        fig, ax = plt.subplots(figsize=(5.2, 0.45 * (len(tab) + 2)))  # kompakt
        ax.axis("off")

        # schöne Spaltenüberschriften
        nice_cols = ["method"] + [c for c in tab.columns if c != "method"]
        the_table = ax.table(cellText=tab[nice_cols].values,
                             colLabels=nice_cols,
                             loc="center")
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(8)
        the_table.scale(1, 1.1)

        title = " | ".join([f"{k}={v}" for k, v in zip(
            by, group_key if isinstance(group_key, tuple) else (group_key,))])
        ax.set_title(shorten(title, width=80, placeholder="…"),
                     fontsize=9, pad=8)

        out_path = os.path.join(out_dir, "tables", f"box_{gname}.png")
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        count += 1

    return count


# ----- Manual smoke test -----------------------------------------------------
if __name__ == "__main__":
    """
    Minimal smoke test:
    - Pick latest results/<timestamp>_* folder
    - Load all_runs.csv (or concat log_*.csv)
    - Produce all plots & tables into that folder
    """
    results_root = Path("results").resolve()
    if not results_root.exists():
        raise SystemExit("No 'results' folder found. Run main.py first.")
    try:
        latest = max(results_root.glob("*_*"), key=lambda p: p.stat().st_mtime)
    except ValueError:
        raise SystemExit(
            "No results subfolder found under 'results/'. Run main.py first.")

    print("[manual] Using results folder:", latest)
    df = load_logs(latest)
    n_seeds_total = int(df["seed"].nunique()) if "seed" in df.columns else None
    print(f"[manual] Rows: {len(df)}, seeds: {n_seeds_total}")

    agg_r = aggregate_regret(df)
    plot_regret_curves(agg_r, latest, n_seeds=n_seeds_total)

    df = add_normalized_regret(df)
    agg_rn = aggregate_regret_norm(df)
    plot_regret_norm_curves(agg_rn, latest, n_seeds=n_seeds_total)

    METHOD_ORDER = ["BO_EI", "BO_KG", "DOE_LHS", "DOE_FF", "Random"]
    plot_box_regret_at_T(
        df, latest, method_order=METHOD_ORDER, highlight_seed=0)
    plot_box_regret_norm_at_T(
        df, latest, method_order=METHOD_ORDER, highlight_seed=0)
    plot_box_best_value_at_T(
        df, latest, method_order=METHOD_ORDER, highlight_seed=0)

    plot_ybest_curves(df, latest, n_seeds=n_seeds_total)

    save_tables(latest, df)
    print("[manual] Done. Check:", latest)


def render_boxplot_side_tables(
    results_dir: Union[str, Path],
    decimals: int = 3,
    fontsize: int = 9,
    dpi: int = 150,
) -> List[Path]:
    """
    Für jeden Boxplot unter results_dir/plots/** wird eine passende Teil-Tabelle
    (nur dieses Problem × Budget T × Noise) als PNG neben den Plot geschrieben:
        <plotname>.png  -->  <plotname>_table.png

    Zeilen = Methoden, Spalten = Seeds, plus 'Median'.
    Unterstützte Plots (per Dateiname):
      - regret_box_<PROB>_T<T>_N<N>[_obs].png
      - regret_norm_box_<PROB>_T<T>_N<N>.png
      - bestvalue_box_<PROB>_T<T>_N<N>[_obs].png
    """
    results_dir = Path(results_dir)
    plots_root = results_dir / "plots"

    # Logs laden
    df = load_logs(results_dir)

    # Noise-Spalte absichern
    if "noise_pct" not in df.columns:
        df = df.copy()
        df["noise_pct"] = 0.0

    # regret_norm ggf. berechnen
    if "regret_norm" not in df.columns:
        try:
            df = add_normalized_regret(df)
        except Exception:
            pass  # wenn nicht möglich, werden regret_norm-Tabellen einfach übersprungen

    # Finalzeile je Run
    keys = ["problem", "method", "seed", "budget", "noise_pct"]
    dff = df.sort_values("iter").groupby(keys, as_index=False).tail(1).copy()

    # Alle relevanten Boxplots einsammeln
    boxplots = list(plots_root.rglob("*box_*.png"))

    # Regexe für Parsing
    rx_regret = re.compile(
        r"^regret_box_(?P<p>.+?)_T(?P<T>\d+)_N(?P<N>\d+)(?P<obs>_obs)?\.png$", re.I)
    rx_regret_norm = re.compile(
        r"^regret_norm_box_(?P<p>.+?)_T(?P<T>\d+)_N(?P<N>\d+)\.png$", re.I)
    rx_bestvalue = re.compile(
        r"^bestvalue_box_(?P<p>.+?)_T(?P<T>\d+)_N(?P<N>\d+)(?P<obs>_obs)?\.png$", re.I)

    out_paths: List[Path] = []

    def _fmt_df_for_png(wide: pd.DataFrame) -> pd.DataFrame:
        # Methoden sortieren
        wide = wide.sort_index()
        # numerische Seed-Spalten sortieren (alles außer 'Median')
        seed_cols = [c for c in wide.columns if c != "Median"]
        seed_cols_sorted = sorted(
            seed_cols, key=lambda x: (isinstance(x, str), x))
        cols = seed_cols_sorted + \
            (["Median"] if "Median" in wide.columns else [])
        wide = wide.reindex(columns=cols)

        # Runden/als String formatieren
        fmt = f"{{:.{decimals}f}}"

        def _fmt_val(v):
            try:
                if pd.isna(v):
                    return ""
                return fmt.format(float(v))
            except Exception:
                return str(v)
        return _df_apply_elementwise(wide, _fmt_val)

    def _save_table_png(wide_fmt: pd.DataFrame, title: str, out_png: Path):
        n_rows, n_cols = wide_fmt.shape
        cell_h = 0.35
        cell_w = 0.95
        pad_h = 1.0
        pad_w = 0.6
        width = min(28, max(6, pad_w*2 + n_cols*cell_w))
        height = min(28, max(2.6, pad_h + n_rows*cell_h))

        fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
        ax.axis("off")
        tbl = ax.table(cellText=wide_fmt.values,
                       rowLabels=wide_fmt.index.tolist(),
                       colLabels=[str(c) for c in wide_fmt.columns],
                       loc="upper center", cellLoc="center", rowLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(fontsize)
        tbl.scale(1.0, 1.18)
        # Header fett
        for (r, c), cell in tbl.get_celld().items():
            if r == 0:
                cell.set_text_props(weight="bold")
        fig.suptitle(title, y=0.995, fontsize=fontsize+2)
        fig.tight_layout()
        fig.savefig(out_png, bbox_inches="tight", dpi=dpi)
        plt.close(fig)

    for png in boxplots:
        name = png.name
        m = rx_regret.match(name) or rx_regret_norm.match(
            name) or rx_bestvalue.match(name)
        if not m:
            continue

        p = m.group("p")
        T = int(m.group("T"))
        N = int(m.group("N"))
        nlv = N / 100.0

        # Welches Value-Feld?
        val_col = None
        title = ""
        if rx_regret.match(name):
            is_obs = bool(m.group("obs"))
            val_col = "regret_obs" if is_obs else "regret"
            title = f"{p} — Regret @T={T}, noise={N}%"
        elif rx_regret_norm.match(name):
            val_col = "regret_norm"
            title = f"{p} — Normalized Regret @T={T}, noise={N}%"
            if val_col not in dff.columns:
                continue
        elif rx_bestvalue.match(name):
            is_obs = bool(m.group("obs"))
            val_col = "best_so_far_obs" if is_obs else "best_so_far"
            title = f"{p} — Best Value @T={T}, noise={N}%"

        # Filter auf diese Facette
        sub = dff[
            (dff["problem"].astype(str) == str(p)) &
            (dff["budget"].astype(int) == T) &
            (np.isclose(dff["noise_pct"].astype(float), nlv))
        ][["method", "seed", val_col]].copy()

        if sub.empty or val_col not in sub.columns:
            continue

        # Pivot: rows=method, cols=seed, values=val_col
        wide = sub.pivot_table(
            index="method", columns="seed", values=val_col, aggfunc="first")
        # Median über Seeds
        med = sub.groupby("method")[val_col].median()
        wide["Median"] = med

        wide_fmt = _fmt_df_for_png(wide)

        out_png = png.with_name(png.stem + "_table.png")
        _save_table_png(wide_fmt, title, out_png)
        out_paths.append(out_png)

    print(f"[tables] generated {len(out_paths)} per-plot tables.")
    return out_paths


# ==== RSM (quadratic) utilities + evaluator (DROP-IN) ======================
# Einfügen ans Dateiende von evaluator.py


def _rsm__x_column(df: pd.DataFrame) -> str:
    """
    Bevorzugt UV-Spalten. Fallbacks vorhanden.
    Reihenfolge: x_uv > x_xy > x
    """
    for c in ("x_uv", "x_xy", "x"):
        if c in df.columns:
            return c
    raise KeyError("Keine X-Spalte (x_uv/x_xy/x) gefunden.")


def _rsm__to_U01_from_series(s: pd.Series) -> np.ndarray:
    X = np.vstack(s.apply(_rsm__to_array_row).to_numpy())
    return np.asarray(X, dtype=float)


def _rsm__is_u01(U: np.ndarray) -> bool:
    return np.isfinite(U).all() and (U.min() >= -1e-9) and (U.max() <= 1.0 + 1e-9)


def _rsm__map_x_to_u01(problem_name: str, X: np.ndarray) -> np.ndarray:
    """
    Mappt Originaldomäne -> [0,1]^d für gängige Benchmarks.
    Falls unbekannt: letzte Rettung = per-Dim min-max (nur Anzeigezwecke).
    """
    name = (problem_name or "").lower()
    U = X.copy().astype(float)
    d = U.shape[1]

    def _affine(x, lo, hi):  # map [lo,hi] -> [0,1]
        return (x - lo) / (hi - lo)

    if "branin" in name and d == 2:
        U[:, 0] = _affine(U[:, 0], -5.0, 10.0)
        U[:, 1] = _affine(U[:, 1],  0.0, 15.0)
        return np.clip(U, 0.0, 1.0)
    if "rosenbrock" in name:
        U = _affine(U, -2.0, 2.0)
        return np.clip(U, 0.0, 1.0)
    if "rastrigin" in name:
        U = _affine(U, -5.12, 5.12)
        return np.clip(U, 0.0, 1.0)
    if "sphere" in name:
        U = _affine(U, -5.0, 5.0)
        return np.clip(U, 0.0, 1.0)
    if "ackley" in name:
        U = _affine(U, -5.0, 5.0)
        return np.clip(U, 0.0, 1.0)
    if "goldstein" in name and d == 2:
        U = _affine(U, -2.0, 2.0)
        return np.clip(U, 0.0, 1.0)
    if "hartmann6" in name or ("hartmann" in name and d == 6):
        return np.clip(U, 0.0, 1.0)  # schon in [0,1]^6

    # unbekanntes Problem → per-Dim min-max als Fallback (nur Anzeige)
    lo = np.nanmin(U, axis=0)
    hi = np.nanmax(U, axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    U = (U - lo) / span
    return np.clip(U, 0.0, 1.0)


def _rsm__get_U01(g: pd.DataFrame) -> np.ndarray:
    """
    Liefert immer U in [0,1]^d aus der Gruppe g.
    Nutzt bevorzugt x_uv/x_xy, mappt nötigenfalls x -> u.
    """
    pname = str(g["problem"].iloc[0])
    # bevorzugte Spalten
    for c in ("x_uv", "x_xy", "x"):
        if c in g.columns:
            U = _rsm__to_U01_from_series(g[c])
            if _rsm__is_u01(U):
                return np.clip(U, 0.0, 1.0)
            # wenn nicht in [0,1] → versuchen zu mappen
            return _rsm__map_x_to_u01(pname, U)
    raise KeyError("Keine X-Spalte (x_uv/x_xy/x) gefunden.")


def _rsm__to_array_row(v):
    if isinstance(v, str):
        try:
            return np.array(eval(v), dtype=float)
        except Exception:
            return np.array([np.nan], dtype=float)
    return np.array(v, dtype=float)


def _rsm__features_batch(U01: np.ndarray) -> np.ndarray:
    """
    Quadratische RSM-Features auf z=2u-1:
      [1, z_i, z_i^2, z_i*z_j (i<j)]
    """
    U01 = np.asarray(U01, dtype=float)
    Z = 2.0 * U01 - 1.0
    n, d = Z.shape
    feats = [np.ones((n, 1), dtype=float)]
    # linear
    for j in range(d):
        feats.append(Z[:, [j]])
    # quadratisch
    for j in range(d):
        feats.append((Z[:, [j]] ** 2))
    # wechselwirkungen
    for i in range(d):
        for j in range(i + 1, d):
            feats.append(Z[:, [i]] * Z[:, [j]])
    return np.hstack(feats)  # (n, p)


def fit_quadratic_rsm(X01: np.ndarray, y: np.ndarray):
    Phi = _rsm__features_batch(X01)
    beta, *_ = lstsq(Phi, y, rcond=None)
    resid = y - Phi @ beta
    dof = max(1, len(y) - Phi.shape[1])
    sigma2 = float((resid @ resid) / dof)
    XtX = Phi.T @ Phi
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(XtX)
    cov_beta = sigma2 * XtX_inv
    return beta, cov_beta, sigma2


def rsm_predict_mean_var(beta: np.ndarray, cov_beta: np.ndarray, X01: np.ndarray):
    Phi = _rsm__features_batch(X01)
    mu = Phi @ beta
    # diag(Phi @ cov @ Phi^T)
    var = np.einsum("ij,jk,ik->i", Phi, cov_beta, Phi)
    var = np.maximum(var, 0.0)
    return mu, var


def rsm_argmin(beta: np.ndarray, d: int, n_starts: int = 32):
    bounds = [(0.0, 1.0)] * d

    def f(u):
        U = np.asarray(u, dtype=float).reshape(1, d)
        return float(_rsm__features_batch(U) @ beta)
    best = (np.inf, None)
    rng = np.random.default_rng(0)
    inits = [np.zeros(d), np.ones(d), 0.5*np.ones(d)]
    inits += [rng.random(d) for _ in range(n_starts)]
    for x0 in inits:
        res = minimize(f, x0=np.clip(x0, 0, 1),
                       method="L-BFGS-B", bounds=bounds)
        if res.fun < best[0]:
            best = (res.fun, res.x)
    return best[1], best[0]


def rsm_progress_over_time(sub: pd.DataFrame, use_observed: bool = True) -> pd.DataFrame:
    """
    Input: ein Run (Problem×Methode×Seed×Noise), sortiert nach iter.
    Output: pro Iteration t: RSM(u*; ≤t), ŷ*(t), y_true(u*), best_so_far(t), best_so_far_obs(t)
    """
    sub = sub.sort_values("iter").reset_index(drop=True)
    rows = []
    from evaluator import _rsm__get_U01, fit_quadratic_rsm, rsm_argmin, rsm_predict_mean_var, _rsm__true_eval

    ycol = "y_noisy" if (use_observed and "y_noisy" in sub.columns) else "y"
    for t in range(len(sub)):
        part = sub.iloc[:t+1]
        X = _rsm__get_U01(part)
        y = part[ycol].astype(float).to_numpy()
        if X.shape[0] < (X.shape[1] + 3):  # etwas Guard für Quadratik
            continue
        beta, cov, _ = fit_quadratic_rsm(X, y)
        d = X.shape[1]
        u_star, y_pred = rsm_argmin(beta, d)
        y_true_star = _rsm__true_eval(str(part["problem"].iloc[0]), u_star)
        rows.append({
            "iter": int(part["iter"].iloc[-1]),
            "n_points": int(X.shape[0]),
            "u_star": json.dumps(u_star.tolist()),
            "y_pred": float(y_pred),
            "y_true_at_ustar": float(y_true_star),
            "best_so_far": float(part["best_so_far"].min()) if "best_so_far" in part.columns else np.nan,
            "best_so_far_obs": float(part["best_so_far_obs"].min()) if "best_so_far_obs" in part.columns else np.nan,
        })
    return pd.DataFrame(rows)


# ==== BEGIN: True-eval fallback for common benchmarks (UV->[domain]) ====


def _rsm__true_eval(problem_name: str, u01) -> float | float("nan"):
    """
    Liefert f_true(u) aus dem UV-Raum [0,1]^d.
    1) Versucht problems.* Factory -> prob.evaluate(u, noise=None)
    2) Fällt zurück auf eingebaute Formeln (Branin, Rosenbrock, Rastrigin, Sphere, Ackley, Goldstein-Price, Hartmann6).
    """
    u = np.asarray(u01, dtype=float).ravel()
    name = (problem_name or "").lower()

    # 1) Versuche externe Factory (sofern vorhanden)
    try:
        factory = _rsm__problem_factory()
        if factory is not None:
            try:
                prob = factory(problem_name)
                return float(prob.evaluate(u, noise=None))
            except Exception:
                pass
    except Exception:
        pass

    # 2) Eingebaute Formeln (UV->Domain Mapping + Standardformeln)

    def _branin(u):
        # Domain: x1 in [-5,10], x2 in [0,15]
        x1 = -5.0 + 15.0 * u[0]
        x2 = 0.0 + 15.0 * u[1]
        a = 1.0
        b = 5.1 / (4.0 * np.pi**2)
        c = 5.0 / np.pi
        r = 6.0
        s = 10.0
        t = 1.0 / (8.0 * np.pi)
        return (x2 - b * x1**2 + c * x1 - r)**2 + s * (1.0 - t) * np.cos(x1) + s

    def _rosenbrock(u):
        # Domain: [-2,2]^d
        x = -2.0 + 4.0 * np.asarray(u, dtype=float)
        return float(np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1.0 - x[:-1])**2))

    def _rastrigin(u):
        # Domain: [-5.12,5.12]^d
        x = -5.12 + 10.24 * np.asarray(u, dtype=float)
        A = 10.0
        d = x.size
        return float(A * d + np.sum(x**2 - A * np.cos(2.0 * np.pi * x)))

    def _sphere(u):
        # Domain: [-5,5]^d
        x = -5.0 + 10.0 * np.asarray(u, dtype=float)
        return float(np.sum(x**2))

    def _ackley(u):
        # Domain: [-5,5]^d
        x = -5.0 + 10.0 * np.asarray(u, dtype=float)
        d = x.size
        return float(
            -20.0 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / d))
            - np.exp(np.sum(np.cos(2.0 * np.pi * x)) / d)
            + 20.0 + np.e
        )

    def _goldstein_price(u):
        # Domain: [-2,2]^2
        x = -2.0 + 4.0 * np.asarray(u, dtype=float)
        x1, x2 = x[0], x[1]
        term1 = 1 + (x1 + x2 + 1)**2 * (19 - 14*x1 + 3 *
                                        x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2)
        term2 = 30 + (2*x1 - 3*x2)**2 * (18 - 32*x1 + 12 *
                                         x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2)
        return float(term1 * term2)

    def _hartmann6(u):
        # Domain: [0,1]^6 (UV = Domain)
        x = np.asarray(u, dtype=float)
        alpha = np.array([1.0, 1.2, 3.0, 3.2])
        A = np.array([
            [10, 3, 17, 3, 1.7, 8],
            [0.05, 10, 17, 0.1, 8, 14],
            [3, 3.5, 1.7, 10, 17, 8],
            [17, 8, 0.05, 10, 0.1, 14]
        ], dtype=float)
        P = 1e-4 * np.array([
            [1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381]
        ], dtype=float)
        inner = np.sum(A * (x - P)**2, axis=1)
        return -float(np.dot(alpha, np.exp(-inner)))

    if "branin" in name:
        return float(_branin(u))
    if "rosenbrock" in name:
        return float(_rosenbrock(u))
    if "rastrigin" in name:
        return float(_rastrigin(u))
    if "sphere" in name:
        return float(_sphere(u))
    if "ackley" in name:
        return float(_ackley(u))
    if "goldstein" in name:
        return float(_goldstein_price(u))
    if "hartmann6" in name or ("hartmann" in name and u.size == 6):
        return float(_hartmann6(u))

    # keine bekannte Formel -> kein True-eval möglich
    return float("nan")
# ==== END: True-eval fallback ==============================================


def _rsm__problem_factory():
    """
    Versucht problems.make_problem oder problems.get_problem zu laden.
    Gibt Callable(name)->problem oder None zurück.
    """
    try:
        from problems import make_problem as _factory
        return _factory
    except Exception:
        try:
            from problems import get_problem as _factory
            return _factory
        except Exception:
            return None


def _rsm__group_fstar(g: pd.DataFrame) -> float | float("nan"):
    """
    Rekonstruiert f* aus vorhandenen Spalten: f* = y - regret (median zur Robustheit).
    """
    if "regret" in g.columns and "y" in g.columns:
        tmp = g[["y", "regret"]].dropna()
        if not tmp.empty:
            vals = (tmp["y"].astype(float) -
                    tmp["regret"].astype(float)).values
            if vals.size:
                return float(np.median(vals))
    return np.nan


def compute_rsm_opt_for_group(g: pd.DataFrame, use_observed: bool = True, prob_factory=None) -> dict:
    """
    Erwartet g als eine Gruppe nach (problem, method, seed, noise_pct_r).
    Liefert dict mit RSM-Optimum, UQ u.a.
    """
    g = g.sort_values("iter")
    xcol = _rsm__x_column(g)
    X = _rsm__get_U01(g)
    # Ziel
    if (not use_observed) and ("y" in g.columns):
        y = g["y"].astype(float).to_numpy()
    else:
        ycol = "y_noisy" if "y_noisy" in g.columns else "y"
        y = g[ycol].astype(float).to_numpy()

    beta, cov, sigma2 = fit_quadratic_rsm(X, y)
    d = X.shape[1]
    u_star, y_pred = rsm_argmin(beta, d)
    _, var_mean = rsm_predict_mean_var(beta, cov, u_star[None, :])
    se_pred = float(np.sqrt(var_mean[0]))

    # True re-eval am RSM-Optimum (robust, mit Fallback-Formeln)
    y_true = np.nan
    try:
        y_true = _rsm__true_eval(str(g["problem"].iloc[0]), u_star)
    except Exception:
        y_true = np.nan

    fstar = _rsm__group_fstar(g)
    regret_true = (y_true - fstar) if (not np.isnan(y_true)
                                       and not np.isnan(fstar)) else np.nan

    return {
        "u_star": json.dumps(u_star.tolist()),
        "y_pred": float(y_pred),
        "se_pred": se_pred,
        "y_true": float(y_true) if not np.isnan(y_true) else np.nan,
        "regret_true": float(regret_true) if not np.isnan(regret_true) else np.nan,
        "sigma2_resid": float(sigma2),
        "n_points": int(X.shape[0]),
    }


def evaluate_doe_with_rsm(
    results_dir: str,
    methods: tuple[str, ...] = (
        "DOE_LHS", "DOE_CCD", "DOE_FF", "OD", "Random"),
    use_observed: bool = True,
    save_csv: bool = True,
) -> pd.DataFrame:
    """
    Batch: pro (problem, method in methods, seed, noise_pct_r) Quadratik-RSM fitten,
    RSM-Optimum & UQ berechnen. Optional als tables/rsm_summary.csv speichern.
    """
    import os
    csv_path = os.path.join(results_dir, "all_runs.csv")
    df = pd.read_csv(csv_path)

    # noise_pct_r sicherstellen
    if "noise_pct_r" not in df.columns and "noise_pct" in df.columns:
        def _to_float(v):
            try:
                if isinstance(v, str):
                    v = v.replace(",", ".")
                return float(v)
            except Exception:
                return np.nan
        df["noise_pct_r"] = df["noise_pct"].apply(_to_float).round(3)

    df = df[df["method"].isin(methods)].copy()
    if df.empty:
        return pd.DataFrame()

    prob_factory = _rsm__problem_factory()

    rows = []
    key_cols = ["problem", "method", "seed", "noise_pct_r"]
    for key, grp in df.groupby(key_cols, dropna=False):
        try:
            res = compute_rsm_opt_for_group(
                grp, use_observed=use_observed, prob_factory=prob_factory)
            res.update({
                "problem": key[0],
                "method": key[1],
                "seed": int(key[2]),
                "noise": float(key[3]),
                "budget": int(grp["budget"].max()) if "budget" in grp.columns else np.nan
            })
            rows.append(res)
        except Exception:
            # schluckt einzelne fehlerhafte Gruppen
            continue

    out = pd.DataFrame(rows)
    if save_csv:
        out_dir = os.path.join(results_dir, "tables")
        os.makedirs(out_dir, exist_ok=True)
        out.to_csv(os.path.join(out_dir, "rsm_summary.csv"), index=False)
    return out


def rsm_progress_over_time(sub: pd.DataFrame, use_observed: bool = True) -> pd.DataFrame:
    sub = sub.sort_values("iter").reset_index(drop=True)
    rows = []
    from evaluator import _rsm__get_U01, fit_quadratic_rsm, rsm_argmin, rsm_predict_mean_var, _rsm__true_eval
    ycol = "y_noisy" if (use_observed and "y_noisy" in sub.columns) else "y"
    bst_pred = np.inf
    bst_true = np.inf
    for t in range(len(sub)):
        part = sub.iloc[:t+1]
        X = _rsm__get_U01(part)
        y = part[ycol].astype(float).to_numpy()
        # etwas Guard: Quadratik braucht mind. lineare + quadratische Terme
        if X.shape[0] < (X.shape[1] + 3):
            continue
        beta, cov, _ = fit_quadratic_rsm(X, y)
        d = X.shape[1]
        u_star, y_pred = rsm_argmin(beta, d)
        y_true_star = _rsm__true_eval(str(part["problem"].iloc[0]), u_star)
        bst_pred = float(min(bst_pred, y_pred))
        bst_true = float(min(bst_true, y_true_star)) if not np.isnan(
            y_true_star) else bst_true
        rows.append({
            "problem": part["problem"].iloc[0],
            "method": part["method"].iloc[0],
            "seed": int(part["seed"].iloc[0]),
            "noise_pct_r": float(part["noise_pct_r"].iloc[0]) if "noise_pct_r" in part.columns else np.nan,
            "iter": int(part["iter"].iloc[-1]),
            "n_points": int(X.shape[0]),
            "u_star": json.dumps(u_star.tolist()),
            "y_pred": float(y_pred),
            "y_true_at_ustar": float(y_true_star) if not np.isnan(y_true_star) else np.nan,
            "best_so_far_true": float(part["best_so_far"].min()) if "best_so_far" in part.columns else np.nan,
            "best_so_far_obs": float(part["best_so_far_obs"].min()) if "best_so_far_obs" in part.columns else np.nan,
            "best_so_far_model_pred": float(bst_pred),
            "best_so_far_model_true": float(bst_true) if np.isfinite(bst_true) else np.nan,
        })
    return pd.DataFrame(rows)


# ==== /RSM DROP-IN =========================================================
# ==== RSM VARIANTS DROP-IN (RSM2 quadratic, RSM3 cubic) ======================


def _rsm__features_poly(U01: np.ndarray, order: int = 2) -> np.ndarray:
    """
    Polynomial features on z = 2u-1 up to given order.
    order=2 -> [1, z_i, z_i^2, z_i*z_j]
    order=3 -> + [z_i^3, z_i^2*z_j (i!=j), z_i*z_j*z_k (i<j<k)]
    """
    U01 = np.asarray(U01, dtype=float)
    Z = 2.0 * U01 - 1.0
    n, d = Z.shape

    feats = [np.ones((n, 1), dtype=float)]                # bias

    # degree 1
    feats += [Z[:, [j]] for j in range(d)]

    if order >= 2:
        # degree 2: z_i^2
        feats += [(Z[:, [j]] ** 2) for j in range(d)]
        # degree 2: pairwise interactions z_i*z_j
        for i in range(d):
            for j in range(i+1, d):
                feats.append((Z[:, [i]] * Z[:, [j]]))

    if order >= 3:
        # degree 3: z_i^3
        feats += [(Z[:, [j]] ** 3) for j in range(d)]
        # degree 3: z_i^2 * z_j (i != j)
        for i in range(d):
            for j in range(d):
                if i == j:
                    continue
                feats.append((Z[:, [i]]**2 * Z[:, [j]]))
        # degree 3: triple interactions z_i*z_j*z_k
        for i in range(d):
            for j in range(i+1, d):
                for k in range(j+1, d):
                    feats.append(Z[:, [i]] * Z[:, [j]] * Z[:, [k]])

    return np.hstack(feats)


def fit_polynomial_rsm(U01: np.ndarray, y: np.ndarray, order: int = 2, ridge: float = 1e-8):
    """
    Ridge-LSQ fit: beta, cov(beta), sigma2.
    Works even if underdetermined; cov is pseudo-inverse regularized.
    """
    from numpy.linalg import lstsq, inv
    U01 = np.asarray(U01, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    Phi = _rsm__features_poly(U01, order=order)

    # ridge on XtX
    XtX = Phi.T @ Phi
    XtX_r = XtX + ridge * np.eye(XtX.shape[0])
    Xty = Phi.T @ y
    beta = np.linalg.solve(XtX_r, Xty)

    resid = y - Phi @ beta
    dof = max(1, len(y) - Phi.shape[1])
    sigma2 = float((resid @ resid) / dof)

    try:
        cov_beta = sigma2 * np.linalg.inv(XtX_r)
    except np.linalg.LinAlgError:
        cov_beta = sigma2 * np.linalg.pinv(XtX_r)

    return beta, cov_beta, sigma2


def rsm_predict_mean_var_poly(U01: np.ndarray, beta: np.ndarray, cov_beta: np.ndarray, order: int = 2):
    U01 = np.asarray(U01, dtype=float)
    Phi = _rsm__features_poly(U01, order=order)
    mu = Phi @ beta
    var = np.einsum("ij,jk,ik->i", Phi, cov_beta, Phi)
    var = np.maximum(var, 0.0)
    return mu, var


def rsm_argmin_poly(beta: np.ndarray, d: int, order: int = 2, n_starts: int = 48):
    """
    Argmin of polynomial mean prediction on [0,1]^d via multi-start L-BFGS-B.
    """
    import numpy as np
    from scipy.optimize import minimize

    bounds = [(0.0, 1.0)] * d

    def f(u):
        U = np.asarray(u, dtype=float).reshape(1, d)
        Phi = _rsm__features_poly(U, order=order)
        return float(Phi @ beta)

    best = (np.inf, None)
    rng = np.random.default_rng(42)

    # corners
    for bits in range(min(1 << min(d, 12), 1 << d)):
        u0 = np.array([(bits >> j) & 1 for j in range(d)], dtype=float)
        res = minimize(f, u0, bounds=bounds, method="L-BFGS-B")
        if res.fun < best[0]:
            best = (res.fun, res.x)

    # random starts
    for _ in range(n_starts):
        u0 = rng.random(d)
        res = minimize(f, u0, bounds=bounds, method="L-BFGS-B")
        if res.fun < best[0]:
            best = (res.fun, res.x)

    return best[1], best[0]


def _rsmv__get_group_U01(g: pd.DataFrame) -> np.ndarray:
    """Reuse evaluator internals to obtain U01 from a log group."""
    from evaluator import _rsm__get_U01  # already in this module
    return _rsm__get_U01(g)


def _rsmv__true_eval(problem_name: str, u01) -> float:
    from evaluator import _rsm__true_eval  # already in this module
    return _rsm__true_eval(problem_name, u01)


def _rsmv__fstar_from_group(g: pd.DataFrame) -> float:
    from evaluator import _rsm__group_fstar   # already in this module
    return _rsm__group_fstar(g)


def rsm_variants_progress_for_group(
    g: pd.DataFrame,
    variant: str,
    use_observed: bool = True,
    ridge: float = 1e-8,
) -> pd.DataFrame:
    """
    Build progression over iterations for a single (problem, method, seed, noise) group
    using variant in {"RSM2","RSM3"}.
    """
    order = 2 if variant.upper() == "RSM2" else 3
    cols = ["iter", "n_points", "u_star", "y_pred", "y_true_at_ustar",
            "best_so_far", "best_so_far_obs",
            "best_so_far_model_pred", "best_so_far_model_true"]
    rows = []

    g = g.sort_values("iter")
    iters = g["iter"].unique().tolist()

    bst_pred, bst_true = np.inf, np.inf
    for t in iters:
        part = g[g["iter"] <= t]
        X = _rsmv__get_group_U01(part)
        ycol = "y_noisy" if use_observed and (
            "y_noisy" in part.columns) else "y"
        y = part[ycol].astype(float).values

        if X.size == 0 or y.size == 0:
            continue

        beta, cov, _ = fit_polynomial_rsm(X, y, order=order, ridge=ridge)
        d = X.shape[1]
        u_star, y_pred = rsm_argmin_poly(beta, d, order=order)
        y_true_star = _rsmv__true_eval(str(part["problem"].iloc[0]), u_star)

        bst_pred = min(bst_pred, float(y_pred))
        if np.isfinite(y_true_star):
            bst_true = min(bst_true, float(y_true_star))

        rows.append({
            "iter": int(part["iter"].iloc[-1]),
            "n_points": int(X.shape[0]),
            "u_star": json.dumps(np.asarray(u_star, dtype=float).tolist()),
            "y_pred": float(y_pred),
            "y_true_at_ustar": float(y_true_star) if np.isfinite(y_true_star) else np.nan,
            "best_so_far": float(part["best_so_far"].min()) if "best_so_far" in part.columns else np.nan,
            "best_so_far_obs": float(part["best_so_far_obs"].min()) if "best_so_far_obs" in part.columns else np.nan,
            "best_so_far_model_pred": float(bst_pred),
            "best_so_far_model_true": float(bst_true) if np.isfinite(bst_true) else np.nan,
        })
    out = pd.DataFrame(rows, columns=cols)
    return out


def evaluate_doe_with_rsm_variants(
    results_dir: str,
    methods: Tuple[str, ...] = ("DOE_LHS",),
    variants: Tuple[str, ...] = ("RSM2", "RSM3"),
    use_observed: bool = True,
    ridge: float = 1e-8,
    save_csv: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each (problem, DOE method, seed, noise), fit RSM2/RSM3 on prefixes and
    write both summary @T and full progress. Returns (summary, progress).
    """
    import os
    csv_path = os.path.join(results_dir, "all_runs.csv")
    df = pd.read_csv(csv_path)

    # normalize noise column
    if "noise_pct_r" not in df.columns and "noise_pct" in df.columns:
        def _to_float(v):
            try:
                if isinstance(v, str):
                    v = v.replace(",", ".")
                return float(v)
            except Exception:
                return np.nan
        df["noise_pct_r"] = df["noise_pct"].apply(_to_float).round(3)

    df = df[df["method"].isin(methods)].copy()
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    summaries = []
    progress_all = []

    gcols = ["problem", "method", "seed", "noise_pct_r"]
    for key, grp in df.groupby(gcols, dropna=False):
        problem = str(key[0])
        fstar = _rsmv__fstar_from_group(grp)

        for variant in variants:
            prog = rsm_variants_progress_for_group(
                grp, variant=variant, use_observed=use_observed, ridge=ridge
            )
            if prog.empty:
                continue
            # add identifying columns
            prog["problem"] = problem
            prog["doe_method"] = str(key[1])
            prog["seed"] = int(key[2])
            prog["noise_pct_r"] = float(key[3]) if key[3] == key[3] else np.nan
            prog["variant"] = variant
            prog["budget"] = int(grp["budget"].max()) if "budget" in grp.columns else int(
                prog["iter"].max())

            # final @T row
            last = prog.iloc[-1].copy()
            y_true = float(last.get("y_true_at_ustar", np.nan))
            regret_true = (y_true - fstar) if (np.isfinite(y_true)
                                               and np.isfinite(fstar)) else np.nan

            summaries.append({
                "problem": problem,
                "variant": variant,
                "doe_method": str(key[1]),
                "seed": int(key[2]),
                "noise_pct_r": float(key[3]) if key[3] == key[3] else np.nan,
                "iter": int(last["iter"]),
                "budget": int(prog["budget"].iloc[-1]),
                "u_star": last["u_star"],
                "y_pred": float(last["y_pred"]),
                "y_true": y_true,
                "regret_true": float(regret_true) if np.isfinite(regret_true) else np.nan,
                "best_so_far_model_true": float(last.get("best_so_far_model_true", np.nan)),
            })

            progress_all.append(prog)

    summary_df = pd.DataFrame(summaries)
    progress_df = pd.concat(
        progress_all, ignore_index=True) if progress_all else pd.DataFrame()

    if save_csv:
        out_dir = Path(results_dir) / "tables"
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(out_dir / "rsm_variants_summary.csv", index=False)
        progress_df.to_csv(out_dir / "rsm_variants_progress.csv", index=False)

    return summary_df, progress_df


def rsm_variants_boxplots(
    results_dir: Union[str, Path],
    variants: Sequence[str] = ("RSM2", "RSM3"),
    dpi: int = 160
) -> list[Path]:
    """
    Make standard 'regret at T' boxplots for RSM2/RSM3 using existing plotting code.
    """
    from evaluator import plot_box_regret_at_T
    results_dir = Path(results_dir)
    tbl = results_dir / "tables" / "rsm_variants_summary.csv"
    if not tbl.exists():
        return []
    df = pd.read_csv(tbl)

    # convert to evaluator-like schema
    df_box = pd.DataFrame({
        "problem": df["problem"],
        "method": df["variant"],              # show RSM2 vs RSM3 as "methods"
        "seed": df["seed"],
        "budget": df["budget"],
        "iter": df["iter"],
        "noise_pct": df["noise_pct_r"],       # evaluator expects noise_pct
        # we compare on true regret at model optimum
        "regret": df["regret_true"],
    })

    outdir = results_dir / "plots"
    outdir.mkdir(parents=True, exist_ok=True)

    plot_box_regret_at_T(
        df=df_box,
        outdir=outdir,
        method_order=list(variants),
        metric="regret",
        highlight_seed=None,
        show_all_seeds=True
    )
    # Files follow evaluator's naming scheme; return all PNGs that match
    pngs = [p for p in outdir.glob("*.png") if "regret_box_" in p.name.lower()]
    return pngs
# ==== END RSM VARIANTS DROP-IN ===============================================

"""
main.py
=======

Purpose
-------
Entry point für einen kompletten Benchmark-Lauf über
problems × methods × budgets × seeds (inkl. optionaler Plots & Sample-Galerie).

Workflow (kurz)
---------------
1) Ausgabe-Ordner results/<timestamp>_<EXP_NAME>
2) Plan definieren (unten im Block "USER-KONFIG"): PROBLEMS, METHODS, BUDGETS, SEEDS, NOISE_PCTS, INIT_FACTOR
3) Für jede Kombination run_one(cfg) aufrufen → CSV-Logs
4) Alle Logs zu all_runs.csv aggregieren
5) (optional) Plots direkt hier erzeugen (PLOT_IN_MAIN), Tabellen als PNG rendern, HTML-Index bauen
6) (optional) 2D/3D Sample-Plots (nur für 2D-Probleme), @T-only um Informationsflut zu reduzieren

Wichtig
-------
- Algorithmische Hyperparameter bleiben in methods.py.
- Noise (noise_pct) skaliert die additive Beobachtungs-Std relativ zu std(f) auf [0,1]^d.
"""

from __future__ import annotations
import collections.abc
import importlib

import subprocess
import platform
import sys
import json
import itertools
import time
import os
import html
from pathlib import Path
from typing import Dict, List, Sequence, Union, Optional
from evaluator import render_boxplot_side_tables, save_best_at_T_tables_as_png
# (falls save_best_at_T_tables_as_png schon importiert ist, nur render_boxplot_side_tables ergänzen)


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning
import warnings

from runner import run_one
from problems import get_problem
from evaluator import (
    aggregate_regret, plot_regret_curves, add_normalized_regret,
    aggregate_regret_norm, plot_regret_norm_curves, plot_box_regret_at_T,
    plot_box_regret_norm_at_T, plot_box_best_value_at_T, plot_ybest_curves,
    rank_by_group, winrate_by_group, plot_rank_heatmap, plot_winrate_heatmap,
    save_rank_tables, save_tables, export_best_at_T_table,
    save_best_at_T_tables_as_png,
)

# Optional: Sample-Visualisierungen (2D/3D)
try:
    from viz_samples import plot_samples_every_k
    _HAS_SAMPLES = True
except Exception:
    _HAS_SAMPLES = False

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# # --- Minimal-Profile-Lader (robust für dict ODER TBConfig/Objekt) ---


# def _as_profile_dict(P):
#     # 1) dict direkt
#     if isinstance(P, dict):
#         return P

#     # 2) generisches Objekt (z.B. TBConfig / dataclass / pydantic / SimpleNamespace …)
#     def pick(obj, *names, default=None):
#         for n in names:
#             if hasattr(obj, n):
#                 return getattr(obj, n)
#         return default

#     prof = {
#         "exp_name": pick(P, "exp_name", "EXP_NAME", "name"),
#         "problems": pick(P, "problems", "PROBLEMS"),
#         "methods": pick(P, "methods", "METHODS"),
#         # budget vs. budget_factors: akzeptiere beides
#         "budget": pick(P, "budget", "BUDGET", "budget_factor"),
#         "budget_factors": pick(P, "budget_factors", "BUDGETS"),
#         "seeds": pick(P, "seeds", "SEEDS"),
#         "noise": pick(P, "noise", "NOISE", "noise_pcts", "NOISE_PCTS"),
#         "init_factor": pick(P, "init_factor", "INIT_FACTOR"),
#         "plot_in_main": pick(P, "plot_in_main", "PLOT_IN_MAIN"),
#         "generate_samples": pick(P, "generate_samples", "GENERATE_SAMPLES"),
#         "final_samples_at_t": pick(P, "final_samples_at_t", "FINAL_SAMPLES_AT_T"),
#         "k_every": pick(P, "k_every", "K_EVERY"),
#     }
#     return prof


# try:
#     if "--profile" in sys.argv:
#         prof_name = sys.argv[sys.argv.index("--profile")+1]
#         P_raw = getattr(importlib.import_module("tb_profiles"), prof_name)

#         # Falls versehentlich eine Fabrikfunktion übergeben wurde, NICHT aufrufen – wir lesen nur Attribute.
#         P = _as_profile_dict(P_raw)

#         # EXP_NAME
#         if P.get("exp_name") is not None:
#             EXP_NAME = P["exp_name"]
#         else:
#             EXP_NAME = prof_name  # Fallback

#         # PROBLEMS / METHODS
#         if P.get("problems") is not None:
#             PROBLEMS = P["problems"]
#         if P.get("methods") is not None:
#             METHODS = P["methods"]

#         # BUDGETS (aus budget ODER budget_factors)
#         if P.get("budget_factors") is not None:
#             bf = P["budget_factors"]
#             BUDGETS = list(bf) if isinstance(
#                 bf, collections.abc.Sequence) else [int(bf)]
#         elif P.get("budget") is not None:
#             BUDGETS = [int(P["budget"])]

#         # SEEDS / NOISE / INIT_FACTOR / Komfortflags
#         if P.get("seeds") is not None:
#             SEEDS = list(P["seeds"])
#         if P.get("noise") is not None:
#             NOISE_PCTS = list(P["noise"])
#         if P.get("init_factor") is not None:
#             INIT_FACTOR = int(P["init_factor"])
#         if P.get("plot_in_main") is not None:
#             PLOT_IN_MAIN = bool(P["plot_in_main"])
#         if P.get("generate_samples") is not None:
#             GENERATE_SAMPLES = bool(P["generate_samples"])
#         if P.get("final_samples_at_t") is not None:
#             FINAL_SAMPLES_AT_T = bool(P["final_samples_at_t"])
#         if P.get("k_every") is not None:
#             K_EVERY = int(P["k_every"])

#         print(
#             f"[main] Loaded profile '{prof_name}' → EXP_NAME={EXP_NAME}, METHODS={METHODS}, PROBLEMS={PROBLEMS}, BUDGETS={BUDGETS}, SEEDS={SEEDS}, NOISE={NOISE_PCTS}")
# except Exception as e:
#     print("[main] WARN: profile loading failed →", e)


# =============================================================================
# USER-KONFIG – einfach anpassen
# =============================================================================
# EXP_NAME: freier Kurzname für den Lauf → beeinflusst den Ausgabeordner
EXP_NAME: str = "TB2_Noise-leiter"

# PROBLEMS:
# - Fixe 2D: "Branin", "GoldsteinPrice"
# - Fixe 6D: "Hartmann6"
# - Variable Dimensionen (beliebige Schreibweisen werden vom Factory geparst):
#   "Sphere3", "sphere 5", "Sphere_7"
#   "Rosenbrock2", "rosenbrock 10", "Rosenbrock_20"
#   "Rastrigin2", "rastrigin 5", "Rastrigin_7"
# Beispiele:
# PROBLEMS = ["Branin", "Rastrigin2"]                     # 2D-Sanity
# PROBLEMS = ["Sphere3", "Rosenbrock5", "Rastrigin7"]     # variable d
# PROBLEMS = ["GoldsteinPrice", "Hartmann6"]
PROBLEMS: Sequence[str] = ["Branin"]

# METHODS (müssen in methods.py implementiert sein):
#   "BO_EI", "BO_KG",
#   "DOE_LHS", "DOE_FF", "DOE_CCD", "OD",
#   "Random"
# Hinweise:
# - DOE_FF nutzt Cranley–Patterson Shift (Fairness-Policy) falls in methods.py aktiviert.
# - BO_* respektiert INIT_FACTOR (Warmup: init_n = INIT_FACTOR * d).
METHODS: Sequence[str] = ["BO_EI", "DOE_LHS"]

# BUDGETS (Iterationen = factor * dim). Beispiel: factor 10 → T = 10*d
# Beispiele:
# BUDGETS = [10]           # schlank
# BUDGETS = [10, 20, 40]   # mehrere Budgets
BUDGETS: Sequence[int] = [10]

# SEEDS (Optimizer-Seeds). Beispiele:
# SEEDS = [0,1,2,3,4]
# SEEDS = list(range(10))
SEEDS: Sequence[int] = list(range(10))

# NOISE_PCTS (relativ zu std(f) auf [0,1]^d):
#   0.0 = noisefrei, 0.5 = 50% der std(f), 1.5 = 150% der std(f)
# Beispiele:
# NOISE_PCTS = [0.0]
# NOISE_PCTS = [0.0, 0.5, 1.5]
NOISE_PCTS: Sequence[float] = [0.00, 0.50, 1.50]

# INIT_FACTOR (nur für BO_*): init_n = INIT_FACTOR * d
INIT_FACTOR: int = 2

# ---------------------------------------------------------------------------
# Komfort-Flags (optional)
# ---------------------------------------------------------------------------
# Plots schon in main.py erzeugen? (ansonsten evaluate_latest.py nutzen)
PLOT_IN_MAIN: bool = True

# Sample-Visualisierungen erzeugen? (nur 2D-Probleme; benötigt viz_samples.py)
GENERATE_SAMPLES: bool = False

# Nur @T (letzte Iteration) visualisieren, um Informationsflut zu reduzieren
FINAL_SAMPLES_AT_T: bool = False

# Falls FINAL_SAMPLES_AT_T=False: alle k Iterationen speichern
K_EVERY: int = 5
# =============================================================================


# -----------------------------------------------------------------------------
# Kleine Helfer (Repro / Index / Utils)
# -----------------------------------------------------------------------------
def dump_effective_config(
    outdir: Path, *, timestamp: str, exp_name: str,
    problems, methods, budgets, seeds,
    init_factor, noise_pcts, resolved_budgets: dict
):
    """Write a JSON mit dem exakten Plan und aufgelösten Budgets pro Problem."""
    cfg = {
        "timestamp": timestamp,
        "exp_name": exp_name,
        "problems": list(problems),
        "methods": list(methods),
        "budgets": list(budgets),
        "resolved_budgets": resolved_budgets,
        "seeds": list(seeds),
        "init_factor": init_factor,
        "noise_pcts": [float(x) for x in noise_pcts],
    }
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def dump_env_info(outdir: Path):
    """Environment info: Python, Platform, pip freeze, git hash/dirty (falls Repo)."""
    info = {
        "python": sys.version,
        "platform": platform.platform(),
    }
    try:
        pip = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                             capture_output=True, text=True, check=False)
        info["pip_freeze"] = pip.stdout.splitlines()
    except Exception as e:
        info["pip_freeze_error"] = repr(e)
    try:
        git_hash = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                  capture_output=True, text=True)
        if git_hash.returncode == 0:
            info["git_hash"] = git_hash.stdout.strip()
        git_stat = subprocess.run(["git", "status", "--porcelain", "-uno"],
                                  capture_output=True, text=True)
        if git_stat.returncode == 0:
            info["git_dirty"] = bool(git_stat.stdout.strip())
    except Exception as e:
        info["git_info_error"] = repr(e)
    (outdir / "env.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


def _rel(p: Path, root: Path) -> str:
    return os.path.relpath(p, root).replace("\\", "/")


def build_benchmark_report(results_dir: Path) -> Path:
    """
    Klickbare index.html mit:
      - plots/*/*.png   (Kurven, Boxplots, Heatmaps, ybest)
      - samples/*/*/*.png (Samples 2D/3D)
      - tables/*.png    (Tabellen als Bilder)
      - NEU: „Boxplots + Tabellen (nebeneinander)“
    """
    plots_dir = results_dir / "plots"
    samples_dir = results_dir / "samples"
    tables_dir = results_dir / "tables"
    out_html = results_dir / "benchmark_report.html"

    def _rel(p: Path) -> str:
        return os.path.relpath(p, results_dir).replace("\\", "/")

    def _section(title: str, content_html: str) -> str:
        return f"<section><h2>{html.escape(title)}</h2>{content_html}</section>"

    # --- Alle Plots (Kachel-Galerie) ---
    plot_cards = []
    if plots_dir.exists():
        for p in sorted(plots_dir.rglob("*.png")):
            cap = f"{p.parent.name} — {p.name}"
            plot_cards.append(f"""
            <div class="card">
              <a href="{_rel(p)}" target="_blank"><img src="{_rel(p)}" loading="lazy"></a>
              <div class="cap">{html.escape(cap)}</div>
            </div>""")
    plots_html = "<div class='grid'>" + "".join(plot_cards) + "</div>"

    # --- Samples (facettiert nach Problem / Run-Ordner) ---
    samples_html = ""
    if samples_dir.exists():
        for prob_dir in sorted(p for p in samples_dir.iterdir() if p.is_dir()):
            groups = []
            for run_dir in sorted(p for p in prob_dir.iterdir() if p.is_dir()):
                imgs = sorted(run_dir.glob("*.png"))
                if not imgs:
                    continue
                thumbs = "".join(
                    f"""<a href="{_rel(img)}" target="_blank"><img src="{_rel(img)}" loading="lazy"></a>"""
                    for img in imgs
                )
                groups.append(f"""
                <details open>
                  <summary>{html.escape(run_dir.name)}</summary>
                  <div class="thumbs">{thumbs}</div>
                </details>""")
            if groups:
                samples_html += _section(
                    f"Samples – {prob_dir.name}", "".join(groups))

    # --- Tabellen (PNG) ---
    table_pngs = []
    if tables_dir.exists():
        for p in sorted(tables_dir.glob("*.png")):
            table_pngs.append(p)

    tables_cards = "".join(
        f"""<div class="card"><a href="{_rel(p)}" target="_blank"><img src="{_rel(p)}" loading="lazy"></a>
            <div class="cap">{html.escape(p.name)}</div></div>"""
        for p in table_pngs
    )
    tables_html = "<div class='grid'>" + tables_cards + "</div>"

    # --- Boxplots + Tabellen (nebeneinander) ---
    # Links: alle Boxplots; Rechts: die Tabellen-PNGs (sticky)
    def _is_boxplot(path: Path) -> bool:
        parts = path.parts
        if "plots" not in parts:
            return False
        # Ordnernamen für Boxplots
        return any(seg in ("regret", "regret_norm", "best_value", "regret_obs", "best_value_obs")
                   for seg in parts)

    box_imgs = []
    if plots_dir.exists():
        for p in sorted(plots_dir.rglob("*.png")):
            if _is_boxplot(p):
                box_imgs.append(p)

    left_col = "".join(
        f"""<div class="card"><a href="{_rel(p)}" target="_blank"><img src="{_rel(p)}" loading="lazy"></a>
             <div class="cap">{html.escape(p.parent.name)} — {html.escape(p.name)}</div></div>"""
        for p in box_imgs
    )
    right_col = "".join(
        f"""<div class="card"><a href="{_rel(p)}" target="_blank"><img src="{_rel(p)}" loading="lazy"></a>
             <div class="cap">{html.escape(p.name)}</div></div>"""
        for p in table_pngs
    )
    side_by_side = f"""
    <div class="twocol">
      <div class="leftcol">{left_col if left_col else "<i>Keine Boxplots gefunden.</i>"}</div>
      <div class="rightcol sticky">{right_col if right_col else "<i>Keine Tabellenbilder gefunden.</i>"}</div>
    </div>
    """

    html_s = f"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="utf-8" />
<title>Results Index – {html.escape(results_dir.name)}</title>
<style>
body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; }}
h1 {{ margin-bottom: 0.2rem; }}
h2 {{ margin: 2rem 0 0.6rem; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 12px; }}
.card {{ border: 1px solid #ddd; border-radius: 10px; overflow: hidden; background:#fff; }}
.card img {{ width: 100%; height: auto; display: block; }}
.card .cap {{ font-size: 12px; padding: 8px; color: #333; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
section {{ margin-bottom: 28px; }}
details summary {{ cursor: pointer; font-weight: 600; margin: 6px 0; }}
.thumbs img {{ width: 320px; height: auto; margin: 4px; border: 1px solid #ddd; border-radius: 8px; }}

.twocol {{ display: grid; grid-template-columns: 1fr 420px; gap: 16px; align-items: start; }}
.leftcol .card {{ margin-bottom: 12px; }}
.rightcol .card {{ margin-bottom: 12px; }}
.sticky {{ position: sticky; top: 12px; }}

.small {{ color:#666; font-size: 12px; }}
</style>
</head>
<body>
  <h1>Results Index</h1>
  <div class="small">{html.escape(str(results_dir))}</div>

  { _section("Boxplots + Tabellen (nebeneinander)", side_by_side) }
  { _section("Plots – Gesamtübersicht", plots_html if plot_cards else "<i>Keine Plots gefunden.</i>") }
  { _section("Samples (facettiert)", samples_html if samples_html else "<i>Keine Samples gefunden.</i>") }
  { _section("Tabellen (PNG)", tables_html if table_pngs else "<i>Keine Tabellenbilder gefunden.</i>") }

</body>
</html>"""
    out_html.write_text(html_s, encoding="utf-8")
    return out_html


def _detect_2d_problems_used(df: pd.DataFrame) -> List[str]:
    """Filtert Probleme aus df, deren Dimension = 2 ist (gemäß problems.get_problem)."""
    probs = sorted(pd.unique(df["problem"].astype(str)))
    out = []
    for p in probs:
        try:
            if get_problem(p).dim == 2:
                out.append(p)
        except Exception:
            pass
    return out


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    # --- Output-Ordner ----------------------------------------------------
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    outdir = Path(f"results/{timestamp}_{EXP_NAME}")
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Repro-Infos ------------------------------------------------------
    resolved = {p: [int(b) for b in BUDGETS] for p in PROBLEMS}
    dump_effective_config(outdir,
                          timestamp=timestamp, exp_name=EXP_NAME,
                          problems=PROBLEMS, methods=METHODS,
                          budgets=BUDGETS, seeds=SEEDS,
                          init_factor=INIT_FACTOR, noise_pcts=NOISE_PCTS,
                          resolved_budgets=resolved)
    dump_env_info(outdir)

    # --- Runs -------------------------------------------------------------
    errors = []
    total_planned = 0
    total_done = 0

    if not SEEDS:
        raise SystemExit(
            "No seeds specified. Set SEEDS = [0] oder list(range(N)).")

    for p in PROBLEMS:
        d = get_problem(p).dim  # dim kept for reference, not used for budgets
        budgets_for_problem = [int(b) for b in BUDGETS]
        for m, b, s, n_pct in itertools.product(METHODS, budgets_for_problem, SEEDS, NOISE_PCTS):
            total_planned += 1
            cfg = {
                "problem": p,
                "method": m,
                "seed": int(s),
                # separater RNG Stream für Noise
                "noise_seed": int(s) + 777_000,
                "noise_pct": float(n_pct),
                "budget": int(b),
                "outdir": str(outdir),
                "exp_name": EXP_NAME,
                "init_factor": INIT_FACTOR,
            }
            print(f"[run] {p} | {m} | seed={s} | T={b} | noise={n_pct:.2f}")
            try:
                run_one(cfg)
                total_done += 1
            except Exception as e:
                print(f"[SKIP] {p}/{m}/seed={s}/B={b}/N={n_pct}: {e}")
                errors.append((cfg, repr(e)))

    # --- glob logs from dedicated folder ---------------------------------
    logs_dir = outdir / "logs"
    logs = sorted(logs_dir.glob("log_*.csv"))
    if not logs:
        # legacy fallback
        logs = sorted(outdir.glob("log_*.csv"))

    if not logs:
        print(
            f"[debug] contents of {outdir}: {[p.name for p in outdir.iterdir()]}")
        print(
            f"[debug] contents of {logs_dir}: {([p.name for p in logs_dir.iterdir()] if logs_dir.exists() else 'n/a')}")
        raise SystemExit(f"No logs found to aggregate in {outdir}.")

    df = pd.concat([pd.read_csv(p) for p in logs], ignore_index=True)
    # Haupt-All-in-One an die Wurzel (Kompatibilität); optional Kopie in logs/
    df.to_csv(outdir / "all_runs.csv", index=False)
    try:
        df.to_csv(logs_dir / "all_runs.csv", index=False)
    except Exception:
        pass

    # ---------------------------------------------------------------------
    # Optional: Plots direkt in main (ansonsten evaluate_latest.py verwenden)
    # ---------------------------------------------------------------------
    METHOD_ORDER = ["BO_EI", "BO_KG", "DOE_CCD",
                    "OD", "DOE_FF", "DOE_LHS", "Random"]

    if PLOT_IN_MAIN:
        # Kurven
        agg_true = aggregate_regret(df, use_observed=False)
        plot_regret_curves(
            agg_true, outdir, ylabel="True regret (median, IQR)", suffix="")

        agg_obs = aggregate_regret(df, use_observed=True)
        plot_regret_curves(
            agg_obs, outdir, ylabel="Observed regret (median, IQR)", suffix="_obs")

        dfn = add_normalized_regret(df.copy())
        aggn = aggregate_regret_norm(dfn)
        plot_regret_norm_curves(aggn, outdir)

        # y_best
        plot_ybest_curves(dfn, outdir)

        # Boxplots
        plot_box_regret_at_T(df, outdir, method_order=METHOD_ORDER,
                             highlight_seed=0, show_all_seeds=True, metric="regret")
        plot_box_regret_at_T(df, outdir, method_order=METHOD_ORDER,
                             highlight_seed=0, show_all_seeds=True, metric="regret_obs")
        plot_box_regret_norm_at_T(dfn, outdir, method_order=METHOD_ORDER,
                                  highlight_seed=0, show_all_seeds=True)
        plot_box_best_value_at_T(df, outdir, method_order=METHOD_ORDER,
                                 highlight_seed=0, show_all_seeds=True, metric="best_so_far")
        plot_box_best_value_at_T(df, outdir, method_order=METHOD_ORDER,
                                 highlight_seed=0, show_all_seeds=True, metric="best_so_far_obs")

        # Heatmaps
        heatdir = outdir / "plots" / "heatmaps"
        heatdir.mkdir(parents=True, exist_ok=True)
        GROUP_COLS = tuple(
            [c for c in ("problem", "budget", "noise_pct") if c in df.columns])
        ranks_auc = rank_by_group(
            df, metric="auc_regret", group_cols=GROUP_COLS, seed_agg="median")
        plot_rank_heatmap(ranks_auc, GROUP_COLS, heatdir,
                          title="Average rank (AUC regret)", filename="rank_heatmap_auc.png")
        win_auc = winrate_by_group(
            df, metric="auc_regret", group_cols=GROUP_COLS)
        plot_winrate_heatmap(win_auc, GROUP_COLS, heatdir,
                             title="Win rate (AUC regret)", filename="winrate_heatmap_auc.png")
        ranks_fin = rank_by_group(
            df, metric="final_regret", group_cols=GROUP_COLS, seed_agg="median")
        plot_rank_heatmap(ranks_fin, GROUP_COLS, heatdir,
                          title="Average rank (final regret)", filename="rank_heatmap_final.png")
        win_fin = winrate_by_group(
            df, metric="final_regret", group_cols=GROUP_COLS)
        plot_winrate_heatmap(win_fin, GROUP_COLS, heatdir,
                             title="Win rate (final regret)", filename="winrate_heatmap_final.png")

        # Tabellen (CSV) basierend auf dfn (da regret_norm enthalten)
        save_rank_tables(df, outdir, group_cols=GROUP_COLS,
                         metrics=("final_regret", "auc_regret"))
        save_tables(outdir, add_normalized_regret(df.copy()))

    # ---------------------------------------------------------------------
    # Immer sinnvoll: Best-at-T Tabellen + PNGs (für Thesis-Layouts)
    # ---------------------------------------------------------------------
    # CSVs
    try:
        export_best_at_T_table(df, outdir, kind="true")
        export_best_at_T_table(df, outdir, kind="obs")
        dfn = add_normalized_regret(df.copy())
        if "regret_norm" in dfn.columns:
            export_best_at_T_table(dfn, outdir, kind="norm")
    except Exception as e:
        print("[warn] best_at_T CSV export failed:", e)

    # PNG-Renderings der *globalen* Tabellen (mit 3 Nachkommastellen)
    try:
        pngs = save_best_at_T_tables_as_png(
            outdir, decimals=3)  # <<< NEU: decimals=3
        print("Table PNGs:", pngs)
    except Exception as e:
        print("[warn] table->png failed:", e)

    # Zusätzliche *per-Plot* Teil-Tabellen direkt neben den Boxplots erzeugen
    try:
        render_boxplot_side_tables(outdir, decimals=3)  # <<< NEU
    except Exception as e:
        print("[warn] render_boxplot_side_tables failed:", e)

    # ---------------------------------------------------------------------
    # Optional: Sample-Visualisierungen (nur wenn viz_samples importierbar)
    # ---------------------------------------------------------------------
    if GENERATE_SAMPLES and _HAS_SAMPLES:
        try:
            print("[samples] Generating sample figures...")
            df_local = df.copy()
            seeds_used = sorted(df_local["seed"].astype(int).unique().tolist())
            probs_2d = _detect_2d_problems_used(df_local)
            methods_used = sorted(
                df_local["method"].astype(str).unique().tolist())
            noise_levels: List[Optional[float]] = [None]
            if "noise_pct" in df_local.columns:
                nlev = sorted(df_local["noise_pct"].dropna().astype(
                    float).unique().tolist())
                noise_levels = nlev if nlev else [None]

            k_eff = (10**9) if FINAL_SAMPLES_AT_T else int(max(1, K_EVERY))

            for p in probs_2d:
                for m in methods_used:
                    for s in seeds_used:
                        for n in noise_levels:
                            try:
                                plot_samples_every_k(
                                    df=df_local, problem_name=p, method=m, seed=int(
                                        s),
                                    k=k_eff, outdir=outdir, noise_pct=n
                                )
                                print(
                                    f"  ok: {p} | {m} | seed={s} | noise={n if n is not None else 0.0}")
                            except Exception as e:
                                print(
                                    f"  skip: {p} | {m} | seed={s} | noise={n}: {e}")

            print(f"[samples] Done. Figures under: {outdir/'samples'}")
        except Exception as e:
            print("[warn] sample generation skipped:", e)

    # ---------------------------------------------------------------------
    # HTML Index (Galerie) bauen
    # ---------------------------------------------------------------------
    try:
        html_path = build_benchmark_report(outdir)
        print("Report:", html_path)
    except Exception as e:
        print("[warn] building benchmark_report.html failed:", e)

    # ---------------------------------------------------------------------
    # Finale Logs
    # ---------------------------------------------------------------------
    print(f"[debug] all_runs shape: {df.shape}, columns: {sorted(df.columns)}")
    print(f"Artifacts written to: {outdir}")
    if errors:
        print(f"Completed with {len(errors)} skipped runs (see console).")


if __name__ == "__main__":
    main()

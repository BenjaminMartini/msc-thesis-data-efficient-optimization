"""
runner.py
=========

Purpose
-------
Core execution of ONE experiment run:
- Build the problem
- Instantiate the method with parameters from the experiment plan
- Run optimization for the given budget
- Log results to CSV

Logging format (one row per iteration)
--------------------------------------
Columns:
  problem, method, seed, noise_seed, noise_pct, sigma_noise,
  budget, iter, x, y, y_noisy, best_so_far, best_so_far_obs,
  regret, regret_obs, time_s

Notes
-----
- Methods are asked in canonical UV-space ([0,1]^d); we log X in *original*
  problem coordinates for transparent plotting.
- BO learns from *observed* values (y_noisy) so Noise wirkt tatsächlich auf
  die Optimierung. Für Reporting bleiben "true" Metriken parallel erhalten.
"""

from __future__ import annotations
import time
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from problems import get_problem, Problem
from methods import BO_EI, BO_KG, DOE_LHS, DOE_FF, DOE_CCD, OD, Random, Method


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _estimate_function_std(prob: Problem,
                           rng: np.random.Generator,
                           n: int = 4000) -> float:
    """Monte-Carlo Schätzung der std(f) im UV-Raum ([0,1]^d)."""
    U = rng.random((n, prob.dim))
    vals = np.array([prob.f_uv(u) for u in U], dtype=float)
    return float(vals.std(ddof=1))


# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------

def make_method(method_name: str, budget: int, dim: int, config: Dict) -> Method:
    """
    Instantiate method from a plan entry.
    - For BO_*: derive init_n from config['init_n'] or init_factor * dim.
    - For DOE_*: pass target size; random_shift wird versucht (Fallback ohne).
    """
    name = str(method_name).lower()

    # init_n für BO
    init_n = config.get("init_n")
    if init_n is None:
        init_factor = config.get("init_factor")
        if init_factor is not None:
            init_n = max(1, int(init_factor * int(dim)))

    if name in ("bo_ei", "ei", "boei", "bo"):
        return BO_EI(init_n=int(init_n) if init_n is not None else BO_EI.init_n)

    if name in ("bo_kg", "kg", "bokg"):
        return BO_KG(init_n=int(init_n) if init_n is not None else BO_KG.init_n)

    if name in ("doe_lhs", "lhs"):
        # LHS erwartet n_init
        return DOE_LHS(n_init=int(budget))

    if name in ("doe_ff", "doeff", "fullfact", "factorial"):
        try:
            return DOE_FF(n_target=int(budget), random_shift=True)
        except TypeError:
            return DOE_FF(n_target=int(budget))

    if name in ("doe_ccd", "ccd"):
        try:
            return DOE_CCD(n_target=int(budget), random_shift=True)
        except TypeError:
            return DOE_CCD(n_target=int(budget))

    if name in ("od", "doe_od", "d_opt", "doptimal", "dopt"):
        try:
            return OD(n_target=int(budget), random_shift=True)
        except TypeError:
            return OD(n_target=int(budget))

    if name in ("random", "rand"):
        return Random()

    raise ValueError(f"Unknown method: {method_name}")


# -----------------------------------------------------------------------------
# One run
# -----------------------------------------------------------------------------

def run_one(config: Dict) -> str:
    """
    Execute a single run and return the path to its CSV log file.

    Required in config:
      problem, method, seed, budget, outdir
    Optional in config:
      exp_name, init_n, init_factor, noise_pct, noise_seed
    """
    problem_name = config["problem"]
    method_name = config["method"]
    seed = int(config["seed"])
    budget = int(config["budget"])
    outdir = Path(config["outdir"])
    exp_name = str(config.get("exp_name", "exp"))
    init_factor = config.get("init_factor", None)

    rng = np.random.default_rng(seed)

    # Noise-Konfiguration
    # z.B. 0.0, 0.05, 0.5, 1.5
    noise_pct = float(config.get("noise_pct", 0.0))
    noise_seed = int(config.get("noise_seed", seed + 99991))
    rng_noise = np.random.default_rng(noise_seed)

    # Problem bauen
    prob: Problem = get_problem(problem_name)
    dim = int(prob.dim)

    # Noise-Skala (sigma) über std(f) im UV-Raum
    sigma_noise = 0.0
    if noise_pct > 0.0:
        n_mc = 6000 if prob.dim >= 4 else 4000
        sigma_noise = noise_pct * \
            _estimate_function_std(prob, rng_noise, n=n_mc)

    # Methode bauen + starten
    meth: Method = make_method(method_name, budget, prob.dim, config)
    meth.start(prob.dim, rng)

    # Logging
    rows = []
    t0 = time.time()

    best_true = np.inf
    best_obs = np.inf
    f_star = prob.f_star if getattr(prob, "f_star", None) is not None else None

    for it in range(budget):
        # 1) Kandidat im UV-Raum [0,1]^d
        x_uv = np.asarray(
            meth.ask(n_remaining=budget - it, rng=rng), dtype=float)
        # 2) in Originalraum für Auswertung/Plot
        x_xy = np.asarray(prob.to_xy(x_uv), dtype=float)

        # 3) wahrer Funktionswert
        y_true = float(prob.f_true(x_xy))

        # 4) beobachteter Wert (mit Noise)
        if sigma_noise > 0.0:
            eps = float(rng_noise.normal(loc=0.0, scale=sigma_noise))
            y_obs = y_true + eps
        else:
            y_obs = y_true

        # 5) Optimierer lernt aus *beobachtetem* Wert
        meth.tell(x_uv, y_obs)

        # 6) Laufende Minima (true/obs)
        best_true = min(best_true, y_true)
        best_obs = min(best_obs,  y_obs)

        # 7) Regrets relativ zu f*
        regret_true = (best_true - f_star) if f_star is not None else np.nan
        regret_obs = (best_obs - f_star) if f_star is not None else np.nan

        # 8) Logzeile
        rows.append({
            # --- Meta ---
            "exp_name":        exp_name,
            "problem":         prob.name,
            "method":          meth.name(),
            "dim":             dim,
            "seed":            seed,           # Optimizer RNG
            "noise_seed":      noise_seed,     # separater Noise-RNG
            "noise_pct":       noise_pct,      # z.B. 0.0 / 0.05 / 0.5 / 1.5
            "sigma_noise":     sigma_noise,
            "budget":          budget,
            # 0-basiert geloggt (Plots können bei it=1 starten)
            "iter":            it,
            "init_factor":     init_factor,

            # --- Inputs (beide Räume) ---
            "x_uv":            json.dumps(list(map(float, x_uv))),  # [0,1]^d
            # Originalraum
            "x_xy":            json.dumps(list(map(float, x_xy))),
            # Backcompat-Alias, wie bisher
            "x":               json.dumps(list(map(float, x_xy))),

            # --- Werte ---
            "y":               y_true,          # TRUE
            "y_noisy":         y_obs,           # beobachtet (mit Noise)
            "best_so_far":     best_true,       # TRUE-Bestwert
            "best_so_far_obs": best_obs,        # beobachteter Bestwert
            "regret":          regret_true,     # TRUE-Regret
            "regret_obs":      regret_obs,      # beobachteter Regret
            "time_s":          time.time() - t0,
        })

    # CSV schreiben (in gemeinsamen logs/ Ordner)
    df = pd.DataFrame(rows)

    logs_dir = outdir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Dateiname: Problem_Methode_Seed_T{budget}_N{xx}
    n_pct_int = int(round(100 * noise_pct))  # 0..150 etc.
    fname = f"log_{prob.name}_{meth.name()}_seed{seed}_T{budget}_N{n_pct_int}.csv"
    path = logs_dir / fname
    df.to_csv(path, index=False)
    return str(path)


# -----------------------------------------------------------------------------
# Manual test
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Minimaler Smoke-Test
    cfg = {
        "problem":     "Branin",
        "method":      "BO_EI",
        "seed":        0,
        "budget":      10,
        "outdir":      "results/test_run",
        "exp_name":    "debug",
        "init_factor": 4,
        "noise_pct":   0.5,  # 50% der std(f)
    }
    Path(cfg["outdir"]).mkdir(parents=True, exist_ok=True)
    p = run_one(cfg)
    print("Wrote log to:", p)

"""
viz_samples.py
--------------

Visual diagnostics for 2D problems:
- 2D: contour + samples + past/current best + arrow path
- 3D: surface (leicht blau/transparent) + samples + past/current best + path line
- Starts at iter=1 (nicht bei 0) in Darstellungen (Titel/Dateinamen sind 1-basig).
- Final-only Modus via plot_samples_every_k(..., k >= T+1) -> nur @T mit Suffix "_Tfinal".

Alle Plots im normalisierten UV-Raum [0,1]^2, damit Vergleiche zu anderen Plots passen.
"""
from __future__ import annotations

try:
    from tb_style import apply as _apply_style
    _apply_style('thesis_v1')
except Exception:
    pass
from matplotlib.patches import Patch

# Stdlib
import json
import ast
from pathlib import Path
from typing import Optional, Union, Sequence

# Third-party
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.lines import Line2D

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Project
from problems import get_problem

# ---------------------------------------------------------------------
# Robust parser for 'x' column (JSON-like to numpy array)


def _parse_x_to_np(val):
    if isinstance(val, (list, tuple, np.ndarray)):
        return np.asarray(val, dtype=float)
    try:
        return np.asarray(json.loads(val), dtype=float)
    except Exception:
        try:
            return np.asarray(ast.literal_eval(val), dtype=float)
        except Exception:
            raise ValueError(f"Cannot parse x='{val}' to numeric array.")
# ---------------------------------------------------------------------


# --- NEU/ERGÄNZT: Imports für Colormaps/Norm ---

# --- NEU: Globale Farb-Knobs (leicht anpassen) ---
COLORMAP_BASE = "Blues"  # gleiche Skala für 2D und 3D
# schneidet die fast-weißen Tiefen weg (0.20..0.35 üblich)
COLOR_CUT_LOW = 0.28
# <1 → tiefe Werte werden relativ dunkler (kräftigeres Blau)
COLOR_GAMMA_COLOR = 0.75
ALPHA_LO = 0.18     # 3D: Alpha bei niedrigen Z
ALPHA_HI = 0.55     # 3D: Alpha bei hohen Z
ALPHA_GAMMA = 1.00     # 3D: Form der Alpha-Kurve (1.0 = linear)


def _cmap_cut_low(name: str = "Blues", cut_low: float = 0.1) -> LinearSegmentedColormap:
    """Schneidet die unteren cut_low-Anteile aus der Colormap ab (verhindert weiß)."""
    base = plt.get_cmap(name)
    cut_low = float(np.clip(cut_low, 0.0, 0.8))
    colors = base(np.linspace(cut_low, 1.0, 256))
    return LinearSegmentedColormap.from_list(f"{name}_cut{int(cut_low*100)}", colors)


def _make_cmap_and_norm(Z: np.ndarray):
    """Liefert (cmap_cut, norm) basierend auf Z-Min/Max – für 2D und 3D identisch."""
    z_min, z_max = float(np.nanmin(Z)), float(np.nanmax(Z))
    norm = Normalize(vmin=z_min, vmax=z_max)
    cmap_cut = _cmap_cut_low(COLORMAP_BASE, cut_low=COLOR_CUT_LOW)
    return cmap_cut, norm


# -------------------------- Style (tweak hier) -------------------------------
STYLE = {
    # 2D & 3D markers
    "samples":   {"face": "white",    "edge": "black", "size2d": 28,  "size3d": 36,  "lw3d": 0.9},
    # hohl
    "past":      {"edge": "#ff7f0e",  "size2d": 70,    "size3d": 80,  "marker": "o", "lw": 1.2},
    "current":   {"face": "#ff7f0e",  "edge": "black", "size2d": 120, "size3d": 160, "marker": "*", "lw3d": 1.0},
    "known_min": {"face": "crimson",  "edge": "black", "size2d": 90,  "size3d": 140, "marker": "P", "lw3d": 0.8},

    # Pfad (wieder orange)
    "path":      {"color": "#ff7f0e", "lw2d": 1.8, "lw3d": 2.0, "alpha": 1.0, "arrowstyle": "->"},

    # 3D surface
    "surface": {
        "cmap": "Blues_r",
        "alpha": 0.55,
        "antialiased": True,
    },

    "view":      {"elev": 28, "azim": 135},
    "cbar_label": "f(x)",
}

STYLE["surface"].update({
    "cmap": "Blues",    # light low, dark high
    "cut_low": 0.12,    # cut the very white bottom to keep low values visible
    "alpha_hi": 0.55,   # alpha at high values
    "alpha_lo": 0.18,   # alpha at low values (more transparent)
    "antialiased": True,
    "zsort": "min",
})

STYLE["path"].update({
    "halo_lw3d": 3.6,  # schwarze Halo-Breite
})
# Optional: Bodenprojektion der Konturen
STYLE["floor"] = {"enabled": True, "offset_frac": 0.03}  # 3% unter z_min

SHOW_WIREFRAME = False
SHOW_FLOOR_CONTOURS = False


# ----------------------------- Name mapping ----------------------------------

def _canonical_problem_name(logged_name: str) -> str:
    s = logged_name.strip().lower()
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
    return logged_name


# --------------------------- Helpers / data prep -----------------------------

def _grid_2d(prob: "Problem", n: int = 160):
    assert prob.dim == 2, "Only 2D problems supported."
    x1 = np.linspace(prob.bounds_lo[0], prob.bounds_hi[0], n)
    x2 = np.linspace(prob.bounds_lo[1], prob.bounds_hi[1], n)
    X1, X2 = np.meshgrid(x1, x2)
    XY = np.c_[X1.ravel(), X2.ravel()]
    Z = np.array([prob.f_true(x) for x in XY], dtype=float).reshape(X1.shape)
    return X1, X2, Z


def _known_minima(prob: "Problem"):
    """
    Versucht bekannte Minima aus dem Problem zu ziehen. Fallback:
    Branin: 3 globale Minima im Originalraum (klassische Koordinaten).
    """
    mins = []
    if hasattr(prob, "x_stars"):
        try:
            mins = [np.asarray(xi, dtype=float) for xi in prob.x_stars]
        except Exception:
            mins = []
    elif hasattr(prob, "x_star"):
        try:
            mins = [np.asarray(prob.x_star, dtype=float)]
        except Exception:
            mins = []

    # Fallback für Branin, falls im Problem nichts hinterlegt ist
    name_low = str(prob.name).lower()
    if not mins and name_low.startswith("branin"):
        mins = [
            np.array([-np.pi,   12.275], dtype=float),
            np.array([np.pi,    2.275], dtype=float),
            np.array([3*np.pi,  2.475], dtype=float),
        ]
    return mins if mins else None


def _extract_run(df: pd.DataFrame,
                 problem: str,
                 method: str,
                 seed: int,
                 noise_pct: Optional[float] = None) -> pd.DataFrame:
    sub = df[(df["problem"].astype(str) == str(problem)) &
             (df["method"].astype(str) == str(method)) &
             (df["seed"].astype(int) == int(seed))]
    if noise_pct is not None and "noise_pct" in sub.columns:
        nv = float(noise_pct)
        sub = sub[np.isclose(sub["noise_pct"].astype(float), nv)]
    sub = sub.sort_values("iter").copy()
    if "x_arr" not in sub.columns and "x" in sub.columns:
        sub["x_arr"] = sub["x"].apply(_parse_x_to_np)
    return sub


def _noise_tag_for_folder(noise_pct: Optional[float], df_like: Optional[pd.DataFrame] = None) -> str:
    if noise_pct is None:
        if df_like is not None and "noise_pct" in df_like.columns:
            vals = pd.unique(df_like["noise_pct"].astype(float))
            if len(vals) == 1:
                noise_pct = float(vals[0])
    if noise_pct is None:
        return ""
    return f"_noise{int(round(float(noise_pct) * 100)):02d}pct"


def _cmap_cut_low(name: str = "Blues", cut_low: float = 0.12) -> LinearSegmentedColormap:
    """
    Colormap, die die unteren (nahe Weiß) Abschneidet, um tiefe Bereiche sichtbar zu halten.
    Beispiel: cut_low=0.12 → nimmt Farben aus [12%, 100%] des Originals.
    (Aktuell nicht zwingend genutzt; Blues_r liefert bereits dunklere Minima.)
    """
    base = plt.get_cmap(name)
    cut_low = float(np.clip(cut_low, 0.0, 0.4))
    colors = base(np.linspace(cut_low, 1.0, 256))
    return LinearSegmentedColormap.from_list(f"{name}_cut{int(cut_low*100)}", colors)


# ----------------------------- 2D plot ---------------------------------------

def _draw_incumbent_path_2d(ax, traj_pts: np.ndarray):
    if traj_pts.shape[0] < 2:
        return
    for a, b in zip(traj_pts[:-1], traj_pts[1:]):
        ax.annotate(
            "", xy=(b[0], b[1]), xytext=(a[0], a[1]),
            arrowprops=dict(arrowstyle=STYLE["path"]["arrowstyle"],
                            color=STYLE["path"]["color"],
                            lw=STYLE["path"]["lw2d"],
                            shrinkA=4, shrinkB=4, alpha=STYLE["path"]["alpha"])
        )


def plot_samples_2d(df: pd.DataFrame,
                    problem_name: str,
                    method: str,
                    seed: int,
                    iter_k: int,
                    outdir: Union[str, Path],
                    grid_n: int = 160,
                    noise_pct: Optional[float] = None,
                    is_final: bool = False) -> Path:
    """
    2D contour plot (UV space [0,1]^2) with all sampled points up to iteration `iter_k`.
    Einheitliche Farbskala mit 3D; dunklere Tiefen (kein weiß), Gamma-Formung.
    """

    # ---- kleine lokale Farb-Helpers (unabhängig vom Rest der Datei) ----
    from matplotlib.colors import Normalize, LinearSegmentedColormap

    def _cmap_cut_low(name: str = "Blues", cut_low: float = 0.28) -> LinearSegmentedColormap:
        base = plt.get_cmap(name)
        cut_low = float(np.clip(cut_low, 0.0, 0.8))
        colors = base(np.linspace(cut_low, 1.0, 256))
        return LinearSegmentedColormap.from_list(f"{name}_cut{int(cut_low*100)}", colors)

    def _make_cmap_and_norm(Z: np.ndarray, cmap_name: str = "Blues", cut_low: float = 0.28):
        z_min, z_max = float(np.nanmin(Z)), float(np.nanmax(Z))
        norm = Normalize(vmin=z_min, vmax=z_max)
        cmap_cut = _cmap_cut_low(cmap_name, cut_low=cut_low)
        return cmap_cut, norm

    def _gamma_cmap(cmap: LinearSegmentedColormap, gamma: float = 0.75) -> LinearSegmentedColormap:
        t = np.linspace(0, 1, 256)
        t_g = np.power(t, float(gamma))
        cols = cmap(t_g)
        return LinearSegmentedColormap.from_list("gamma_map", cols)

    # --- Problem + Run-Filter ---------------------------------------------------
    prob = get_problem(problem_name)
    if prob.dim != 2:
        raise ValueError("plot_samples_2d supports 2D problems only.")

    run = df.copy()
    if "problem" in run.columns:
        run = run[run["problem"].astype(str) == str(prob.name)]
    if "method" in run.columns:
        run = run[run["method"].astype(str) == str(method)]
    if "seed" in run.columns:
        run = run[run["seed"].astype(int) == int(seed)]
    if noise_pct is not None and "noise_pct" in run.columns:
        run = run[np.isclose(run["noise_pct"].astype(float), float(noise_pct))]

    if run.empty:
        raise ValueError(
            "No rows found for the requested (problem, method, seed[, noise]).")

    run = run.sort_values("iter")
    if "iter" not in run.columns:
        raise ValueError("DataFrame must contain an 'iter' column.")

    # clamp iter_k to max available
    T = int(run["iter"].max())
    iter_k = int(min(max(0, iter_k), T))
    up_to_k = run[run["iter"] <= iter_k].copy()
    if up_to_k.empty:
        raise ValueError("No rows up to the requested iteration.")

    # --- UV-Koordinaten ---------------------------------------------------------
    def _row_xy_to_uv(row) -> np.ndarray:
        x_xy = _parse_x_to_np(row["x"])
        return np.asarray(prob.to_uv(x_xy), dtype=float)

    up_to_k["uv"] = up_to_k.apply(_row_xy_to_uv, axis=1)
    uv = np.stack(up_to_k["uv"].to_numpy(), axis=0)  # (N,2)
    y_true = up_to_k["y"].to_numpy(dtype=float)

    # --- Best-Pfad (Verbesserungen) --------------------------------------------
    best_path_uv = []
    best_val = np.inf
    for (uvi, yi) in zip(uv, y_true):
        if yi < best_val - 1e-15:
            best_val = yi
            best_path_uv.append(uvi.copy())
    best_uv = best_path_uv[-1]

    # --- Grid im UV-Raum + f werten --------------------------------------------
    u = np.linspace(0.0, 1.0, int(grid_n))
    v = np.linspace(0.0, 1.0, int(grid_n))
    U, V = np.meshgrid(u, v)
    pts_uv = np.stack([U.ravel(), V.ravel()], axis=1)
    Z = np.empty(pts_uv.shape[0], dtype=float)
    for i, (uu, vv) in enumerate(pts_uv):
        x_xy = prob.to_xy(np.array([uu, vv], dtype=float))
        Z[i] = float(prob.f_true(x_xy))
    Z = Z.reshape(U.shape)

    # --- Einheitliche Farbskala (wie 3D) ---------------------------------------
    cmap_cut, norm = _make_cmap_and_norm(Z, cmap_name="Blues", cut_low=0.28)
    cmap_gamma = _gamma_cmap(cmap_cut, gamma=0.75)
    n_levels = 22

    # --- Plot -------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
    cf = ax.contourf(U, V, Z, levels=n_levels, cmap=cmap_gamma, norm=norm)
    cb = fig.colorbar(cf, ax=ax)
    cb.set_label("f(x)")

    # Alle Samples (zweifach für Kontrast)
    ax.scatter(uv[:, 0], uv[:, 1], s=22, c="white",
               edgecolors="none", zorder=3)
    ax.scatter(uv[:, 0], uv[:, 1], s=16, c="0.35",
               edgecolors="black", linewidths=0.25, zorder=4)

    # Past → Pfad mit Halo + Orange
    if len(best_path_uv) >= 2:
        path = np.stack(best_path_uv, axis=0)
        ax.plot(path[:, 0], path[:, 1], color="black", lw=3.0, zorder=5)
        ax.plot(path[:, 0], path[:, 1], color="#ff7f0e", lw=2.0, zorder=6)

    # Aktueller Best
    ax.scatter(best_uv[0], best_uv[1], s=140, marker="*", edgecolors="black",
               linewidths=0.9, c="#ff7f0e", zorder=7, label="current best")

    # Bekannte Minima
    minima_xy = _known_minima(prob)
    if minima_xy is not None and len(minima_xy):
        mins_uv = np.stack([np.asarray(prob.to_uv(xi), dtype=float)
                           for xi in minima_xy], axis=0)
        ax.scatter(mins_uv[:, 0], mins_uv[:, 1], s=90, marker="P",
                   edgecolors="black", linewidths=0.8, c="crimson",
                   zorder=8, label="known minima")

    # Achsen + Titel
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("u1")
    ax.set_ylabel("u2")
    iter_k_display = int(iter_k) + 1  # 1-basig
    n_pct = noise_pct
    if n_pct is None and "noise_pct" in up_to_k.columns:
        uniq = pd.unique(up_to_k["noise_pct"].astype(float))
        n_pct = float(uniq[0]) if len(uniq) == 1 else None
    noise_txt = f", noise={int(round(100*float(n_pct)))}%" if n_pct is not None else ""
    ax.set_title(f"{prob.name} — samples up to iter {iter_k_display}\n"
                 f"method={method}, seed={seed}{noise_txt}")

    # Legende
    ax.legend(loc="upper right", frameon=True)

    # --- Datei ------------------------------------------------------------------
    noise_tag = _noise_tag_for_folder(n_pct, up_to_k)
    out_dir = Path(outdir) / "samples" / \
        f"{prob.name}" / f"{method}_seed{seed}{noise_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_Tfinal" if is_final else ""
    path = out_dir / f"samples2d_it{iter_k_display}{suffix}.png"

    try:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    finally:
        plt.close(fig)
    return path


# ----------------------------- 3D plot ---------------------------------------

def plot_samples_3d(df: pd.DataFrame,
                    problem_name: str,
                    method: str,
                    seed: int,
                    iter_k: int,
                    outdir: Union[str, Path],
                    grid_n: int = 160,
                    noise_pct: Optional[float] = None,
                    is_final: bool = False) -> Path:
    """
    3D surface plot (UV space [0,1]^2) with unified colormap to 2D, darker lows, per-face alpha.
    """

    from matplotlib.colors import Normalize, LinearSegmentedColormap
    from matplotlib.lines import Line2D

    def _cmap_cut_low(name: str = "Blues", cut_low: float = 0.28) -> LinearSegmentedColormap:
        base = plt.get_cmap(name)
        cut_low = float(np.clip(cut_low, 0.0, 0.8))
        colors = base(np.linspace(cut_low, 1.0, 256))
        return LinearSegmentedColormap.from_list(f"{name}_cut{int(cut_low*100)}", colors)

    def _make_cmap_and_norm(Z: np.ndarray, cmap_name: str = "Blues", cut_low: float = 0.28):
        z_min, z_max = float(np.nanmin(Z)), float(np.nanmax(Z))
        norm = Normalize(vmin=z_min, vmax=z_max)
        cmap_cut = _cmap_cut_low(cmap_name, cut_low=cut_low)
        return cmap_cut, norm

    # Farb-Parameter (gleichen denen im 2D-Plot)
    COLOR_GAMMA_COLOR = 0.75  # <1 → tiefe Werte dunkler
    ALPHA_LO, ALPHA_HI = 0.18, 0.55
    ALPHA_GAMMA = 1.0

    # --- Problem + Run-Filter ---------------------------------------------------
    prob = get_problem(problem_name)
    if prob.dim != 2:
        raise ValueError("plot_samples_3d supports 2D problems only.")

    run = df.copy()
    if "problem" in run.columns:
        run = run[run["problem"].astype(str) == str(prob.name)]
    if "method" in run.columns:
        run = run[run["method"].astype(str) == str(method)]
    if "seed" in run.columns:
        run = run[run["seed"].astype(int) == int(seed)]
    if noise_pct is not None and "noise_pct" in run.columns:
        run = run[np.isclose(run["noise_pct"].astype(float), float(noise_pct))]

    if run.empty:
        raise ValueError(
            "No rows found for the requested (problem, method, seed[, noise]).")

    run = run.sort_values("iter")
    T = int(run["iter"].max())
    iter_k = int(min(max(0, iter_k), T))
    up_to_k = run[run["iter"] <= iter_k].copy()
    if up_to_k.empty:
        raise ValueError("No rows up to the requested iteration.")

    # --- UV & y_true ------------------------------------------------------------
    def _row_xy_to_uv(row) -> np.ndarray:
        x_xy = _parse_x_to_np(row["x"])
        return np.asarray(prob.to_uv(x_xy), dtype=float)

    up_to_k["uv"] = up_to_k.apply(_row_xy_to_uv, axis=1)
    uv = np.stack(up_to_k["uv"].to_numpy(), axis=0)
    y_true = up_to_k["y"].to_numpy(dtype=float)

    # --- Verbesserungs-Pfad -----------------------------------------------------
    path_uv, path_y = [], []
    best_val = np.inf
    for (uvi, yi) in zip(uv, y_true):
        if yi < best_val - 1e-15:
            best_val = yi
            path_uv.append(uvi.copy())
            path_y.append(float(yi))
    best_uv = path_uv[-1]
    best_y = path_y[-1]
    past_uv = np.stack(path_uv[:-1], axis=0) if len(path_uv) > 1 else None
    past_y = np.asarray(path_y[:-1], dtype=float) if len(path_y) > 1 else None

    # --- Surface: UV-Grid -> f --------------------------------------------------
    u = np.linspace(0.0, 1.0, int(grid_n))
    v = np.linspace(0.0, 1.0, int(grid_n))
    U, V = np.meshgrid(u, v)
    pts_uv = np.stack([U.ravel(), V.ravel()], axis=1)
    Z = np.empty(pts_uv.shape[0], dtype=float)
    for i, (uu, vv) in enumerate(pts_uv):
        x_xy = prob.to_xy(np.array([uu, vv], dtype=float))
        Z[i] = float(prob.f_true(x_xy))
    Z = Z.reshape(U.shape)

    # --- Einheitliche Farbskala (wie 2D) + per-face Alpha ----------------------
    cmap_cut, norm = _make_cmap_and_norm(Z, cmap_name="Blues", cut_low=0.28)
    Z_norm = norm(Z)
    z_for_col = np.power(Z_norm, COLOR_GAMMA_COLOR)   # Gamma auf Farben
    facecolors = cmap_cut(z_for_col)
    # Gamma auf Alpha (linear default)
    z_for_alpha = np.power(Z_norm, ALPHA_GAMMA)
    facecolors[..., 3] = ALPHA_LO + (ALPHA_HI - ALPHA_LO) * z_for_alpha

    fig = plt.figure(figsize=(8.2, 6.8))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(U, V, Z,
                           rstride=1, cstride=1,
                           facecolors=facecolors,
                           linewidth=0, antialiased=True,
                           shade=False, zorder=1)
    try:
        surf.set_zsort("min")  # Punkte/Pfad davor rendern
    except Exception:
        pass

    # --- Punkte sichtbar halten (zweifaches Scatter) ---------------------------
    ax.scatter(uv[:, 0], uv[:, 1], y_true,
               s=22, c="white", edgecolors="none", depthshade=False, zorder=3)
    ax.scatter(uv[:, 0], uv[:, 1], y_true,
               s=16, c="0.35", edgecolors="black", linewidths=0.25,
               depthshade=False, zorder=4)

    # Past bests
    if past_uv is not None and past_uv.size:
        ax.scatter(past_uv[:, 0], past_uv[:, 1], past_y,
                   s=180, facecolors="none",
                   edgecolors="black", linewidths=1.2,
                   depthshade=False, zorder=5)

    # Pfad: schwarzer Halo + orange
    if len(path_uv) >= 2:
        path_uv_arr = np.stack(path_uv, axis=0)
        path_y_arr = np.asarray(path_y, dtype=float)
        ax.plot(path_uv_arr[:, 0], path_uv_arr[:, 1], path_y_arr,
                color="black", lw=3.6, zorder=6)
        ax.plot(path_uv_arr[:, 0], path_uv_arr[:, 1], path_y_arr,
                color="#ff7f0e", lw=2.4, zorder=7)

    # Current best + Label
    ax.scatter(best_uv[0], best_uv[1], best_y, s=160, marker="o",
               edgecolors="black", linewidths=0.9, c="#d62728",
               depthshade=False, zorder=8, label="current best")
    ax.text(best_uv[0], best_uv[1], best_y, f"{best_y:.3g}",
            color="#d62728", fontsize=10, fontweight="bold", zorder=9)

    # Bekannte Minima
    minima_xy = _known_minima(prob)
    if minima_xy is not None and len(minima_xy):
        mins_uv = np.stack([np.asarray(prob.to_uv(xi), dtype=float)
                           for xi in minima_xy], axis=0)
        mins_z = np.array([float(prob.f_true(xi))
                          for xi in minima_xy], dtype=float)
        ax.scatter(mins_uv[:, 0], mins_uv[:, 1], mins_z,
                   s=140, marker="P", edgecolors="black", linewidths=0.8,
                   c="crimson", depthshade=False, zorder=8, label="known minima")

    # Achsen & Titel
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("u1")
    ax.set_ylabel("u2")
    ax.set_zlabel("f(x)")
    iter_k_display = int(iter_k) + 1
    n_pct = noise_pct
    if n_pct is None and "noise_pct" in up_to_k.columns:
        uniq = pd.unique(up_to_k["noise_pct"].astype(float))
        n_pct = float(uniq[0]) if len(uniq) == 1 else None
    noise_txt = f", noise={int(round(100*float(n_pct)))}%" if n_pct is not None else ""
    ax.set_title(f"{prob.name} — samples up to iter {iter_k_display}\n"
                 f"method={method}, seed={seed}{noise_txt}")

    # Gemeinsame Colorbar (wie 2D)
    m = plt.cm.ScalarMappable(norm=norm, cmap=cmap_cut)
    m.set_array([])
    cb = fig.colorbar(m, ax=ax, shrink=0.7, pad=0.05)
    cb.set_label("f(x)")

    # Legende
    handles = [
        Line2D([0], [0], marker='o', linestyle='None', markersize=6,
               markerfacecolor='0.35', markeredgecolor='black', label='samples ≤ k'),
        Line2D([0], [0], marker='o', linestyle='None', markersize=8,
               markerfacecolor='none', markeredgecolor='black', label='past bests'),
        Line2D([0], [0], marker='o', linestyle='None', markersize=8,
               markerfacecolor='#d62728', markeredgecolor='black', label='current best'),
        Line2D([0], [0], marker='P', linestyle='None', markersize=8,
               markerfacecolor='crimson', markeredgecolor='black', label='known minima'),
        Line2D([0], [0], color="#ff7f0e", lw=2.4, label='incumbent path'),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True)

    # Datei
    noise_tag = _noise_tag_for_folder(n_pct, up_to_k)
    out_dir = Path(outdir) / "samples" / \
        f"{prob.name}" / f"{method}_seed{seed}{noise_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_Tfinal" if is_final else ""
    path = out_dir / f"samples3d_it{iter_k_display}{suffix}.png"

    try:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    finally:
        plt.close(fig)
    return path


# ------------------------- Batch export by checkpoints -----------------------

def plot_samples_every_k(df: pd.DataFrame,
                         problem_name: str,
                         method: str,
                         seed: int,
                         k: int,
                         outdir: Union[str, Path],
                         grid_n_2d: int = 160,
                         grid_n_3d: int = 100,
                         noise_pct: Optional[float] = None) -> Sequence[Path]:
    """
    Generate 2D and 3D plots at selected iterations for a single run.
    - Falls k >= (T+1): nur @T (final).
    - Sonst: 1-basige Auswahl: plotte alle i, deren (i+1) % k == 0; T wird immer inkludiert.
    """
    prob = get_problem(_canonical_problem_name(problem_name))
    if prob.dim != 2:
        raise ValueError("Only 2D problems supported for sample plots.")

    run = _extract_run(df, prob.name, method, seed, noise_pct=noise_pct)
    iters = sorted(run["iter"].unique())
    if not iters:
        return []

    T = max(iters)  # 0-basiert
    # Nur final @T?
    if k >= (T + 1):
        checkpoints = [T]
    else:
        # 1-basig auswählen: (i+1) % k == 0
        checkpoints = [i for i in range(0, T + 1) if ((i + 1) % k == 0)]
        if not checkpoints or checkpoints[-1] != T:
            checkpoints.append(T)

    out_paths: list[Path] = []
    for it in checkpoints:
        is_final = (it == T)
        out_paths.append(
            plot_samples_2d(run, prob.name, method, seed, it, outdir,
                            grid_n=grid_n_2d, noise_pct=noise_pct, is_final=is_final)
        )
        out_paths.append(
            plot_samples_3d(run, prob.name, method, seed, it, outdir,
                            grid_n=grid_n_3d, noise_pct=noise_pct, is_final=is_final)
        )
    return out_paths


# ==== 2D Overlay: True vs. RSM + Punkte (DROP-IN) ==========================
# Einfügen ans Dateiende von viz_samples.py


def plot_rsm_vs_true_surface(
    problem_name: str,
    method: str,
    seed: int,
    noise: float,
    X: np.ndarray,             # Samples in u ∈ [0,1]^2 (shape: N×2)
    Gx: np.ndarray,            # Grid X (1D array length g)
    Gy: np.ndarray,            # Grid Y (1D array length g)
    Z_true: np.ndarray,        # True function on grid (g×g)
    mu_rsm: np.ndarray,        # RSM mean on grid (g×g)
    u_star: np.ndarray,        # RSM predicted optimum in u-space (2,)
    y_pred: float,             # ŷ* at u_star (RSM prediction)
    # optional: aktuelle Iteration (wird im Titel angezeigt)
    iter_idx: int = None,
    # optional: bekannte Minima in u-Koordinaten (k×2)
    known_minima_uv: np.ndarray = None,
    levels: int = 18
):
    """
    2D-Overlay (Contours) von Benchmark-Funktion vs. RSM inkl. Samples, Known-Optima und RSM-Optimum.

    Änderungen gegenüber der alten Version:
    - Legende zeigt den *Funktionsnamen* (z.B. "Branin2D") statt "True".
    - Known-Optima werden eingezeichnet (gleicher Stil wie in den Sample-3D-Plots: Marker 'P', crimson).
    - Titel enthält Methode, seed, noise, N und optional Iterationsindex.
    - RSM-Optimum heißt jetzt "RSM predicted optimum" und zeigt Zahlenwerte (u*, ŷ*).
    - RSM-Linienstil konsistent gestrichelt in der Legende.
    """

    import json
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    # --- Robustness ---
    assert X.ndim == 2 and X.shape[1] == 2, "X must be N×2 in u-space [0,1]^2"
    assert Gx.ndim == 1 and Gy.ndim == 1, "Gx/Gy must be 1D grid vectors"
    assert Z_true.shape == (
        Gy.size, Gx.size), "Z_true must be (len(Gy)×len(Gx))"
    assert mu_rsm.shape == (
        Gy.size, Gx.size), "mu_rsm must be (len(Gy)×len(Gx))"
    assert u_star.shape == (2,), "u_star must be length-2 in u-space"

    # --- Figure/Axes ---
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 6.1))

    # --- Contours: True (mit Funktionsnamen) ---
    true_contours = ax.contour(Gx, Gy, Z_true, levels=levels, linewidths=1.0)
    # Legendeneintrag für True = Funktionsname
    legend_handles = [
        Line2D([0], [0], linestyle='-', color='black', label=f"{problem_name}")
    ]

    # --- Contours: RSM (gestrichelt, konsistent) ---
    rsm_contours = ax.contour(
        Gx, Gy, mu_rsm, levels=levels, linestyles='--', linewidths=1.0)
    legend_handles.append(
        Line2D([0], [0], linestyle='--',
               color='black', label='RSM (quadratic)')
    )

    # --- Samples (schwarz) ---
    ax.scatter(X[:, 0], X[:, 1], s=20, c="k", alpha=0.85, label="Samples")
    legend_handles.append(
        Line2D([0], [0], linestyle='None',
               marker='o', color='k', label='Samples')
    )

    # --- Known minima (gleicher Stil wie 3D-Plots: Marker 'P', crimson) ---
    mins_uv = None
    if known_minima_uv is not None and len(known_minima_uv) > 0:
        mins_uv = np.asarray(known_minima_uv, dtype=float)
    else:
        # Optionaler Auto-Lookup über problems.get_problem (falls vorhanden)
        try:
            from problems import get_problem
            prob = get_problem(problem_name)
            if getattr(prob, "x_star_xy", None) is not None and hasattr(prob, "to_uv"):
                # prob.x_star_xy kann Liste oder Array sein
                xs = prob.x_star_xy
                if not isinstance(xs, (list, tuple, np.ndarray)):
                    xs = [xs]
                mins_uv = np.vstack([prob.to_uv(x) for x in xs])
        except Exception:
            mins_uv = None

    if mins_uv is not None and len(mins_uv) > 0:
        mins_uv = np.asarray(mins_uv).reshape(-1, 2)
        # Optional: True-Wert am bekannten Minimum (zeigt min. Wert, falls prob.f_true verfügbar)
        label_known = "known minima"
        try:
            from problems import get_problem
            prob = get_problem(problem_name)
            if hasattr(prob, "to_xy") and hasattr(prob, "f_true"):
                vals = []
                for m in mins_uv:
                    xy = prob.to_xy(m)
                    vals.append(float(prob.f_true(xy)))
                if len(vals):
                    label_known = f"known minima\nmin={min(vals):.4g}"
        except Exception:
            pass

        ax.scatter(
            mins_uv[:, 0], mins_uv[:, 1],
            s=90, marker="P",
            edgecolors="black", linewidths=0.8,
            c="crimson", zorder=6
        )
        legend_handles.append(
            Line2D([0], [0], linestyle='None', marker='P', markerfacecolor='crimson',
                   markeredgecolor='black', label=label_known)
        )

    # --- RSM predicted optimum (Stern, orange, mit Zahlen) ---
    label_rsm_opt = f"RSM predicted optimum\nu*={u_star[0]:.3f},{u_star[1]:.3f} | ŷ*={y_pred:.4g}"
    ax.scatter(
        [u_star[0]], [u_star[1]],
        s=110, marker="*",
        edgecolors="k", facecolors="orange",
        zorder=7
    )
    legend_handles.append(
        Line2D([0], [0], linestyle='None', marker='*',
               markerfacecolor='orange', markeredgecolor='black',
               label=label_rsm_opt)
    )

    # --- Titel inkl. Iter und N ---
    N = X.shape[0]
    iter_txt = f" | iter={iter_idx}" if iter_idx is not None else ""
    ax.set_title(
        f"{problem_name} | {method} | seed={seed} | noise={noise:g} | N={N}{iter_txt}")

    # --- Achsenformatierung ---
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("u1")
    ax.set_ylabel("u2")
    ax.grid(True, alpha=0.25)

    # --- Legende ---
    ax.legend(handles=legend_handles, loc="best", frameon=True)

    plt.tight_layout()
    plt.show()


# ==== /DROP-IN =============================================================

# ==== 3D Overlay: True vs. RSM + Punkte (DROP-IN) ==========================
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (aktiviert 3D)


def plot_rsm_vs_true_surface_3d(
    problem_name: str,
    method: str,
    seed: int,
    noise: float,
    X: np.ndarray,             # Samples in u ∈ [0,1]^2 (shape: N×2)
    Gx: np.ndarray,            # Grid X (1D array length g)
    Gy: np.ndarray,            # Grid Y (1D array length g)
    Z_true: np.ndarray,        # True function on grid (g×g)
    mu_rsm: np.ndarray,        # RSM mean on grid (g×g)
    u_star: np.ndarray,        # RSM predicted optimum in u-space (2,)
    y_pred: float,             # ŷ* at u_star (RSM prediction)
    # optional: aktuelle Iteration (wird im Titel angezeigt)
    iter_idx: int = None,
    # optionale bekannte Minima in u-Koordinaten (k×2)
    known_minima_uv: np.ndarray = None,
    elev: float = 25.0,        # Blickwinkel (Elevation)
    azim: float = -60.0,       # Blickwinkel (Azimut)
    stride: int = 2,           # Oberflächen-Striding
    alpha_true: float = 0.70,  # Transparenz True-Oberfläche
    alpha_rsm: float = 0.55    # Transparenz RSM-Oberfläche
):
    """
    3D-Overlay (Surfaces) von Benchmark-Funktion vs. RSM inkl. Samples, Known-Optima und RSM-Optimum.

    Änderungen ggü. älteren Versionen:
    - Legendeneintrag nutzt den *Funktionsnamen* (z.B. "Branin2D") statt "True".
    - Known-Optima werden eingezeichnet (Marker 'P', crimson) und optional mit True-Wert beschriftet.
    - Titel enthält Methode, seed, noise, N und optional Iterationsindex.
    - RSM-Optimum heißt "RSM predicted optimum" und zeigt Zahlenwerte (u*, ŷ*).
    - Samples schwarz; RSM/True als zwei halbtransparente Flächen mit eigener Legende.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    # --- Robustness ---
    assert X.ndim == 2 and X.shape[1] == 2, "X must be N×2 in u-space [0,1]^2"
    assert Gx.ndim == 1 and Gy.ndim == 1, "Gx/Gy must be 1D grid vectors"
    assert Z_true.shape == (
        Gy.size, Gx.size), "Z_true must be (len(Gy)×len(Gx))"
    assert mu_rsm.shape == (
        Gy.size, Gx.size), "mu_rsm must be (len(Gy)×len(Gx))"
    assert u_star.shape == (2,), "u_star must be length-2 in u-space"

    # --- Meshgrid ---
    GX, GY = np.meshgrid(Gx, Gy)  # shape (g,g)

    # --- Figure/Axes ---
    fig = plt.figure(figsize=(8.8, 6.9))
    ax = fig.add_subplot(111, projection='3d')

    # --- True Surface ---
    true_surf = ax.plot_surface(
        GX, GY, Z_true,
        rstride=stride, cstride=stride,
        linewidth=0, antialiased=True, alpha=alpha_true
    )

    # --- RSM Surface ---
    rsm_surf = ax.plot_surface(
        GX, GY, mu_rsm,
        rstride=stride, cstride=stride,
        linewidth=0, antialiased=True, alpha=alpha_rsm
    )

    # --- Legend Proxies (für Flächen) ---
    legend_handles = [
        Patch(label=f"{problem_name}"),
        Patch(label="RSM (quadratic)")
    ]

    # --- Hilfsfunktionen für z-Werte an Punkten ---
    def _nearest_z_from_grid(u: np.ndarray, GX: np.ndarray, GY: np.ndarray, Z: np.ndarray) -> float:
        """Nächster Gitterpunkt (ohne Interpolation)."""
        ix = np.clip(np.searchsorted(Gx, u[0]) - 1, 0, len(Gx) - 1)
        iy = np.clip(np.searchsorted(Gy, u[1]) - 1, 0, len(Gy) - 1)
        return float(Z[iy, ix])

    def _true_val(u: np.ndarray) -> float:
        """True-Wert an u, wenn möglich über problems.f_true, sonst nächster Gridpunkt."""
        try:
            from problems import get_problem
            prob = get_problem(problem_name)
            if hasattr(prob, "to_xy") and hasattr(prob, "f_true"):
                return float(prob.f_true(prob.to_xy(u)))
        except Exception:
            pass
        return _nearest_z_from_grid(u, GX, GY, Z_true)

    # --- Samples (schwarz) auf True-Höhe (falls möglich) ---
    if X.size > 0:
        z_samps = np.array([_true_val(u) for u in X])
        ax.scatter(
            X[:, 0], X[:, 1], z_samps,
            s=16, c="k", depthshade=True, alpha=0.95, label="Samples", zorder=5
        )
        legend_handles.append(
            Line2D([0], [0], marker='o', color='k', linestyle='None', label='Samples'))

    # --- Known minima (Marker 'P', crimson) ---
    mins_uv = None
    if known_minima_uv is not None and len(known_minima_uv) > 0:
        mins_uv = np.asarray(known_minima_uv, dtype=float)
    else:
        try:
            from problems import get_problem
            prob = get_problem(problem_name)
            if getattr(prob, "x_star_xy", None) is not None and hasattr(prob, "to_uv"):
                xs = prob.x_star_xy
                if not isinstance(xs, (list, tuple, np.ndarray)):
                    xs = [xs]
                mins_uv = np.vstack([prob.to_uv(x) for x in xs])
        except Exception:
            mins_uv = None

    if mins_uv is not None and len(mins_uv) > 0:
        mins_uv = np.asarray(mins_uv).reshape(-1, 2)
        z_mins = np.array([_true_val(u) for u in mins_uv])
        ax.scatter(
            mins_uv[:, 0], mins_uv[:, 1], z_mins,
            s=60, marker="P", edgecolors="black", linewidths=0.8,
            c="crimson", depthshade=True, zorder=6
        )
        # Optional numerischer Minimalwert
        label_known = "known minima"
        try:
            label_known = f"known minima\nmin={np.min(z_mins):.4g}"
        except Exception:
            pass
        legend_handles.append(
            Line2D([0], [0], marker='P', markerfacecolor='crimson',
                   markeredgecolor='black', linestyle='None', label=label_known)
        )

    # --- RSM predicted optimum (Stern, orange) ---
    ax.scatter(
        [u_star[0]], [u_star[1]], [y_pred],
        s=110, marker="*",
        edgecolors="k", facecolors="orange",
        depthshade=True, zorder=7
    )
    legend_handles.append(
        Line2D([0], [0], marker='*', markerfacecolor='orange',
               markeredgecolor='black', linestyle='None',
               label=f"RSM predicted optimum\nu*={u_star[0]:.3f},{u_star[1]:.3f} | ŷ*={y_pred:.4g}")
    )

    # --- Achsen/Ansicht/Titel ---
    N = X.shape[0]
    iter_txt = f" | iter={iter_idx}" if iter_idx is not None else ""
    ax.set_title(
        f"{problem_name} | {method} | seed={seed} | noise={noise:g} | N={N}{iter_txt}")
    ax.set_xlabel("u1")
    ax.set_ylabel("u2")
    ax.set_zlabel("f")
    try:
        ax.set_box_aspect((1, 1, 0.6))
    except Exception:
        pass
    ax.view_init(elev=elev, azim=azim)

    # --- Legende ---
    ax.legend(handles=legend_handles, loc="upper right", frameon=True)

    plt.tight_layout()
    plt.show()

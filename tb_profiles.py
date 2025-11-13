"""
tb_profiles.py — unified profiles for Thesis Benchmarks (TB1..TB6 + TB_RSM)

Usage in notebooks:
    import tb_profiles as TB
    # Start a run using a profile (override anything as needed)
    TB.apply_and_run("TB1", SEEDS=[0,1,2])

Design:
- TBConfig dataclass stores all configuration knobs used by main.py.
- PROFILES dict defines canonical benchmark suites (TB1..TB6) plus TB_RSM.
- apply_and_run loads/reloads main.py, pushes config as module attributes, then calls main().
"""

from __future__ import annotations
from dataclasses import dataclass, asdict, replace
from typing import Sequence, Dict, Any, Optional
import importlib

# --------------------------- Dataclass ----------------------------------------


@dataclass(frozen=True)
class TBConfig:
    EXP_NAME: str
    PROBLEMS: Sequence[str]
    METHODS: Sequence[str]
    BUDGET_FACTORS: Sequence[int]
    SEEDS: Sequence[int]
    NOISE_PCTS: Sequence[float]
    # Optional/common flags used in your pipeline (keep defaults consistent)
    INIT_FACTOR: float = 4.0
    PLOT_IN_MAIN: bool = False
    DO_SAMPLES: bool = True

# --------------------------- Profiles -----------------------------------------
# NOTE: Adjust PROBLEMS/METHODS names to match exactly those used in your
#       problems.py and methods.py factories.


PROFILES: Dict[str, TBConfig] = {
    # TB1 — Sanity-Check 2D, noisefrei
    "TB1": TBConfig(
        EXP_NAME="TB1",
        PROBLEMS=["Branin", "Rastrigin2"],
        METHODS=["BO_EI", "DOE_LHS", "Random"],
        BUDGET_FACTORS=[10],            # T = 10 * d
        SEEDS=list(range(5)),
        NOISE_PCTS=[0.0],
    ),

    # --- TB2 — Noise-Leiter (Robustheit) -----------------------------------------
    "TB2": TBConfig(
        EXP_NAME="TB2_Noise-Ladder",
        PROBLEMS=["Branin"],
        METHODS=["BO_EI", "DOE_LHS"],
        BUDGET_FACTORS=[10, 25, 50],         # explicit budgets for the ladder
        SEEDS=list(range(20)),         # 20 repeats
        NOISE_PCTS=[0.0, 0.5, 1.5],    # scaled by std(f) on [0,1]^d
        INIT_FACTOR=2,                 # BO warmup: init_n = INIT_FACTOR * d
        # (if supported) enforce local problems/methods only
    ),


    # TB3 — Budget-Skalierung (Data-Efficiency)
    "TB3": TBConfig(
        EXP_NAME="TB3",
        PROBLEMS=["Branin", "Rosenbrock2"],
        METHODS=["BO_EI", "DOE_LHS"],
        BUDGET_FACTORS=[5, 10, 20],
        SEEDS=list(range(10)),
        NOISE_PCTS=[0.0],
    ),

    # TB4 — Dimensions-Skalierung (2D → 6D)
    "TB4": TBConfig(
        EXP_NAME="TB4",
        PROBLEMS=["Sphere2", "Rosenbrock5", "Hartmann6"],
        METHODS=["BO_EI", "DOE_LHS"],
        BUDGET_FACTORS=[10],
        SEEDS=list(range(10)),
        NOISE_PCTS=[0.0],
    ),

    # TB5 — KG vs. EI (Bayesian-only Bake-off)
    "TB5": TBConfig(
        EXP_NAME="TB5",
        PROBLEMS=["Branin", "Rosenbrock5"],
        METHODS=["BO_EI", "BO_KG"],
        BUDGET_FACTORS=[10, 20],
        SEEDS=list(range(15)),
        NOISE_PCTS=[0.0, 0.5],
    ),

    # TB6 — DOE_FF Fairness-Policy (Random-Shift)
    "TB6": TBConfig(
        EXP_NAME="TB6",
        PROBLEMS=["Branin", "Rastrigin"],
        METHODS=["DOE_FF", "DOE_LHS"],
        BUDGET_FACTORS=[10],
        SEEDS=list(range(20)),
        NOISE_PCTS=[0.0],
    ),

    # --- TB_RSM: kurze, mittlere, lange Läufe -----------------------------
    "TB_RSM_B20": TBConfig(
        EXP_NAME="TB_RSM_B20",
        PROBLEMS=["Branin"],
        METHODS=["DOE_LHS", "Random"],
        SEEDS=list(range(3)),         # gern erhöhen
        NOISE_PCTS=[0.0],
        BUDGET_FACTORS=[10],          # → T = 10 * d = 20 (bei d=2)
        INIT_FACTOR=2.0,
        DO_SAMPLES=False,
    ),

    "TB_RSM_B100": TBConfig(
        EXP_NAME="TB_RSM_B100",
        PROBLEMS=["Branin"],
        METHODS=["DOE_LHS", "Random"],
        SEEDS=list(range(3)),
        NOISE_PCTS=[0.0],
        BUDGET_FACTORS=[50],          # → T = 100
        INIT_FACTOR=2.0,
        DO_SAMPLES=False,
    ),

    "TB_RSM_B1000": TBConfig(
        EXP_NAME="TB_RSM_B1000",
        PROBLEMS=["Branin"],
        METHODS=["DOE_LHS", "Random"],
        SEEDS=list(range(3)),
        NOISE_PCTS=[0.0],
        BUDGET_FACTORS=[500],         # → T = 1000
        INIT_FACTOR=2.0,
        DO_SAMPLES=False,
    ),


}

# --------------------------- Helpers ------------------------------------------


def list_profiles() -> Sequence[str]:
    return sorted(PROFILES.keys())


def get_profile(name: str) -> TBConfig:
    if name not in PROFILES:
        raise KeyError(
            f"Unknown profile: {name}. Available: {list_profiles()}")
    return PROFILES[name]


def _with_overrides(cfg: TBConfig, **overrides: Any) -> TBConfig:
    """Return a new TBConfig with supported fields overridden."""
    valid_fields = set(asdict(cfg).keys())
    clean = {k: v for k, v in overrides.items() if k in valid_fields}
    if not clean:
        return cfg
    return replace(cfg, **clean)


def _apply_to_module(cfg: TBConfig, module) -> None:
    """Push config as module-level attributes into 'main' (or similar)."""
    for k, v in asdict(cfg).items():
        setattr(module, k, v)

# --------------------------- Entry point --------------------------------------


def apply_and_run(profile: str, **overrides: Any):
    """Load 'main' module, apply profile (+overrides), and call main().
    Returns whatever main.main() returns (e.g., a results path or None).
    """
    cfg = get_profile(profile)
    if overrides:
        cfg = _with_overrides(cfg, **overrides)

    M = importlib.import_module("main")
    M = importlib.reload(M)
    _apply_to_module(cfg, M)

    # Call pipeline entrypoint
    if not hasattr(M, "main"):
        raise AttributeError("main.py has no function 'main()'")
    return M.main()


if __name__ == "__main__":
    print("Available profiles:", ", ".join(list_profiles()))

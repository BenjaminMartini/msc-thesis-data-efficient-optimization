"""
problems.py
===========

Purpose
-------
Defines benchmark optimization problems and a lightweight `Problem` dataclass.
Objectives are noise-free here; noise wird in höheren Schichten (runner) hinzugefügt.

Key points
----------
- Original domain bounds in R^d + affine mappings <-> [0,1]^d
- f_true(x_xy) erwartet ORIGINAL-Koordinaten; f_uv(u) ist Komfort-Wrapper
- Variable Dimensionen:
    * Sphere(d)       : any d >= 1, bounds [-5,5]^d, f* = 0 @ 0
    * Rosenbrock(d)   : any d >= 2, bounds [-2.048,2.048]^d, f* = 0 @ 1
    * Rastrigin(d)    : any d >= 1, bounds [-5.12,5.12]^d, f* = 0 @ 0
  Fixe Dimensionen:
    * Branin2D, GoldsteinPrice2D, Hartmann6D

Accepted names (case-insensitive, flexible spacing/underscores):
  "sphere", "sphere2", "sphere 5", "sphere5d"
  "rosenbrock", "rosenbrock10", ...
  "rastrigin7", "rastrigin 3", ...
  "branin", "goldsteinprice", "hartmann6"
"""

from __future__ import annotations

import re
import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional

Array = np.ndarray  # readability alias


# -----------------------------------------------------------------------------
# Dataclass
# -----------------------------------------------------------------------------

@dataclass
class Problem:
    name: str
    bounds_lo: Array
    bounds_hi: Array
    dim: int
    f_true: Callable[[Array], float]
    f_star: Optional[float] = None
    x_star_xy: Optional[Array] = None

    # ---- canonical cube mappings ------------------------------------------------
    def to_uv(self, x_xy: Array) -> Array:
        x_xy = np.asarray(x_xy, dtype=float)
        return (x_xy - self.bounds_lo) / (self.bounds_hi - self.bounds_lo)

    def to_xy(self, x_uv: Array) -> Array:
        x_uv = np.asarray(x_uv, dtype=float)
        return self.bounds_lo + x_uv * (self.bounds_hi - self.bounds_lo)

    def f_uv(self, x_uv: Array) -> float:
        return float(self.f_true(self.to_xy(x_uv)))


# -----------------------------------------------------------------------------
# 2D Benchmarks (original coords)
# -----------------------------------------------------------------------------

def branin_xy(xy: Array) -> float:
    xy = np.asarray(xy, dtype=float).ravel()
    x1, x2 = xy[0], xy[1]
    a = 1.0
    b = 5.1 / (4 * np.pi**2)
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * np.pi)
    return a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s


def goldstein_price_xy(xy: Array) -> float:
    x, y = np.asarray(xy, dtype=float).ravel()
    a = (1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2))
    b = (30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))
    return float(a * b)


# -----------------------------------------------------------------------------
# Variable-dimensional benchmarks (original coords)
# -----------------------------------------------------------------------------

def sphere_nd(xy: Array) -> float:
    xy = np.asarray(xy, dtype=float).ravel()
    return float(np.sum(xy**2))


def rosenbrock_nd(xy: Array, a: float = 1.0, b: float = 100.0) -> float:
    x = np.asarray(xy, dtype=float).ravel()
    # sum_{i=1..d-1} [(a - x_i)^2 + b * (x_{i+1} - x_i^2)^2]
    s1 = np.sum((a - x[:-1])**2)
    s2 = np.sum(b * (x[1:] - x[:-1]**2)**2)
    return float(s1 + s2)


def rastrigin_nd(xy: Array, A: float = 10.0) -> float:
    x = np.asarray(xy, dtype=float).ravel()
    d = x.size
    return float(A * d + np.sum(x**2 - A * np.cos(2 * np.pi * x)))


# -----------------------------------------------------------------------------
# Hartmann-6 on [0,1]^6 (original == canonical here)
# -----------------------------------------------------------------------------

def hartmann6_uv(u: Array) -> float:
    u = np.asarray(u, dtype=float).ravel()
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([
        [10,   3, 17,  3.5,  1.7,  8],
        [0.05, 10, 17,  0.1,  8, 14],
        [3,  3.5,  1.7, 10, 17,  8],
        [17,   8,  0.05, 10,  0.1, 14],
    ], dtype=float)
    P = 1e-4 * np.array([
        [1312, 1696, 5569, 124, 8283, 5886],
        [2329, 4135, 8307, 3736, 1004, 9991],
        [2348, 1451, 3522, 2883, 3047, 6650],
        [4047, 8828, 8732, 5743, 1091, 381],
    ], dtype=float)

    total = 0.0
    for i in range(4):
        inner = np.sum(A[i] * (u - P[i])**2)
        total += alpha[i] * np.exp(-inner)
    return -float(total)


# -----------------------------------------------------------------------------
# Name parser (supports "sphere 5", "rastrigin10", "rosenbrock7d", ...)
# -----------------------------------------------------------------------------

def _parse_base_and_dim(raw: str) -> tuple[str, Optional[int]]:
    """
    Return (base, dim) where base ∈ {'branin','sphere','rosenbrock','rastrigin',
                                      'goldsteinprice','hartmann6'}
    Dim is None if not specified (defaults applied in factory).
    """
    s = re.sub(r"[\s_\-]+", "", raw.strip().lower())

    # Hartmann6: fixed name (allow variants)
    if s.startswith("hartmann6"):
        return "hartmann6", 6

    # GoldsteinPrice: treat 'goldstein' as alias
    if s.startswith("goldsteinprice") or s.startswith("goldstein"):
        # optional trailing digits are ignored (function is 2D only)
        return "goldsteinprice", None

    # Branin: 2D only
    if s.startswith("branin"):
        return "branin", None

    # Generic pattern: base + optional digits + optional 'd'
    m = re.match(r"^(sphere|rosenbrock|rastrigin)(\d+)?d?$", s)
    if m:
        base = m.group(1)
        dim = int(m.group(2)) if m.group(2) is not None else None
        return base, dim

    # Fallbacks (plain names)
    if s in {"sphere", "rosenbrock", "rastrigin"}:
        return s, None

    raise ValueError(f"Unknown problem name format: {raw!r}")


# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------

def get_problem(name: str) -> Problem:
    """
    Build a `Problem` instance by (possibly dimensioned) name.

    Fixed problems:
      - "Branin"         -> 2D
      - "GoldsteinPrice" -> 2D
      - "Hartmann6"      -> 6D  (domain [0,1]^6)

    Variable-d problems (accept 'base', 'baseK', 'base K', 'baseKd'):
      - "Sphere"       : any d >= 1, default d=2 if omitted
      - "Rosenbrock"   : any d >= 2, default d=2 if omitted
      - "Rastrigin"    : any d >= 1, default d=2 if omitted
    """
    base, dim = _parse_base_and_dim(name)

    if base == "branin":
        lo = np.array([-5.0, 0.0], dtype=float)
        hi = np.array([10.0, 15.0], dtype=float)
        return Problem(
            name="Branin2D",
            bounds_lo=lo,
            bounds_hi=hi,
            dim=2,
            f_true=branin_xy,
            f_star=0.397887,
            x_star_xy=np.array([np.pi, 2.275], dtype=float),
        )

    if base == "goldsteinprice":
        lo = np.array([-2.0, -2.0], dtype=float)
        hi = np.array([2.0,  2.0], dtype=float)
        return Problem(
            name="GoldsteinPrice2D",
            bounds_lo=lo,
            bounds_hi=hi,
            dim=2,
            f_true=goldstein_price_xy,
            f_star=3.0,
            x_star_xy=np.array([0.0, -1.0], dtype=float),
        )

    if base == "hartmann6":
        lo = np.zeros(6, dtype=float)
        hi = np.ones(6, dtype=float)
        x_star = np.array([0.20169, 0.150011, 0.476874,
                          0.275332, 0.311652, 0.6573], dtype=float)
        return Problem(
            name="Hartmann6D",
            bounds_lo=lo, bounds_hi=hi, dim=6,
            f_true=hartmann6_uv,  # original domain == [0,1]^6
            f_star=-3.32237,
            x_star_xy=x_star,
        )

    # ----- Variable-d problems -----

    if base == "sphere":
        d = int(dim) if dim is not None else 2
        if d < 1:
            raise ValueError("Sphere(d): require d >= 1.")
        lo = np.full(d, -5.0, dtype=float)
        hi = np.full(d,  5.0, dtype=float)
        return Problem(
            name=f"Sphere{d}D",
            bounds_lo=lo, bounds_hi=hi, dim=d,
            f_true=sphere_nd,
            f_star=0.0,
            x_star_xy=np.zeros(d, dtype=float),
        )

    if base == "rosenbrock":
        d = int(dim) if dim is not None else 2
        if d < 2:
            raise ValueError("Rosenbrock(d): require d >= 2.")
        lo = np.full(d, -2.048, dtype=float)
        hi = np.full(d,  2.048, dtype=float)
        return Problem(
            name=f"Rosenbrock{d}D",
            bounds_lo=lo, bounds_hi=hi, dim=d,
            f_true=rosenbrock_nd,
            f_star=0.0,
            x_star_xy=np.ones(d, dtype=float),
        )

    if base == "rastrigin":
        d = int(dim) if dim is not None else 2
        if d < 1:
            raise ValueError("Rastrigin(d): require d >= 1.")
        lo = np.full(d, -5.12, dtype=float)
        hi = np.full(d,  5.12, dtype=float)
        return Problem(
            name=f"Rastrigin{d}D",
            bounds_lo=lo, bounds_hi=hi, dim=d,
            f_true=rastrigin_nd,
            f_star=0.0,
            x_star_xy=np.zeros(d, dtype=float),
        )

    # Should never happen due to parser
    raise ValueError(f"Unknown problem: {name!r}")

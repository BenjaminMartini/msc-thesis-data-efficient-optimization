"""
methods.py
==========

Optimization methods behind a unified interface.

Included:
- BO_EI    : Bayesian Optimization with Expected Improvement.
- BO_KG    : Bayesian Optimization with Knowledge Gradient (MC approximation).
- DOE_LHS  : Latin Hypercube Sampling (non-adaptive design).
- Random   : Pure random baseline.

All methods operate in the canonical cube [0,1]^d.
"""

from __future__ import annotations
from typing import Optional, List, Tuple
from math import ceil

from dataclasses import dataclass
import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel
from scipy.stats import norm
from typing import Optional, List, Sequence, Tuple, Union

try:
    from numpy.typing import NDArray
    Array = NDArray[np.float64]
except Exception:
    Array = np.ndarray

# (falls noch nicht vorhanden)

# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------


class Method:
    """Lightweight protocol for optimization methods."""

    def start(self, dim: int, rng: np.random.Generator) -> None:
        raise NotImplementedError

    def ask(self, n_remaining: int, rng: np.random.Generator) -> Array:
        raise NotImplementedError

    def tell(self, x_uv: Array, y: float) -> None:
        return

    def name(self) -> str:
        return self.__class__.__name__


def _rand_points(n: int, dim: int, rng: np.random.Generator) -> Array:
    """Uniform random points in [0,1]^d."""
    return rng.random((n, dim))


# ---------------------------------------------------------------------------
# Bayesian Optimization with Expected Improvement (EI)
# ---------------------------------------------------------------------------

@dataclass
class BO_EI(Method):
    """Bayesian Optimization using Expected Improvement."""

    init_n: int = 5
    n_cand: int = 1024
    xi: float = 0.01
    normalize_y: bool = True

    dim: Optional[int] = None
    X: Optional[list] = None
    Y: Optional[list] = None
    gp: Optional[GaussianProcessRegressor] = None

    def start(self, dim: int, rng: np.random.Generator) -> None:
        self.dim = dim
        self.X, self.Y = [], []
        kernel = C(1.0, (1e-3, 1e3)) * \
            Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(1e-6)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.0,
            normalize_y=self.normalize_y,
            n_restarts_optimizer=2,
            random_state=int(rng.integers(0, 2**31 - 1)),
        )

    def _fit(self) -> None:
        if self.X:
            self.gp.fit(np.asarray(self.X), np.asarray(self.Y))

    def ask(self, n_remaining: int, rng: np.random.Generator) -> Array:
        if len(self.X) < self.init_n:
            return _rand_points(1, self.dim, rng)[0]
        self._fit()
        Xc = _rand_points(self.n_cand, self.dim, rng)
        mu, std = self.gp.predict(Xc, return_std=True)
        y_best = float(np.min(self.Y))
        imp = y_best - mu - self.xi
        with np.errstate(divide="ignore", invalid="ignore"):
            Z = np.where(std > 0, imp / std, 0.0)
            ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
            ei = np.where(std > 0, ei, 0.0)
        return Xc[int(np.argmax(ei))]

    def tell(self, x_uv: Array, y: float) -> None:
        self.X.append(np.asarray(x_uv))
        self.Y.append(float(y))


# ---------------------------------------------------------------------------
# Bayesian Optimization with Knowledge Gradient (approx)
# ---------------------------------------------------------------------------

@dataclass
class BO_KG(Method):
    """
    Bayesian Optimization using Knowledge Gradient.

    Approximation:
    - For each candidate, sample several possible outcomes from N(mu, sigma^2).
    - For each sample, imagine updating the GP mean (approximate).
    - Estimate the expected improvement in the max predicted mean.
    """

    init_n: int = 5
    n_cand: int = 1024
    n_mc: int = 20
    normalize_y: bool = True

    dim: Optional[int] = None
    X: Optional[list] = None
    Y: Optional[list] = None
    gp: Optional[GaussianProcessRegressor] = None

    def start(self, dim: int, rng: np.random.Generator) -> None:
        self.dim = dim
        self.X, self.Y = [], []
        kernel = C(1.0, (1e-3, 1e3)) * \
            Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(1e-6)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.0,
            normalize_y=self.normalize_y,
            n_restarts_optimizer=2,
            random_state=int(rng.integers(0, 2**31 - 1)),
        )

    def _fit(self) -> None:
        if self.X:
            self.gp.fit(np.asarray(self.X), np.asarray(self.Y))

    def ask(self, n_remaining: int, rng: np.random.Generator) -> Array:
        if len(self.X) < self.init_n:
            return _rand_points(1, self.dim, rng)[0]

        self._fit()
        Xc = _rand_points(self.n_cand, self.dim, rng)
        mu, std = self.gp.predict(Xc, return_std=True)

        # Current best predicted mean
        mu_current_best = float(np.min(mu))

        # Approximate KG via MC sampling at each candidate
        kg_vals = np.zeros(len(Xc))
        for i, (m, s) in enumerate(zip(mu, std)):
            if s <= 1e-12:
                kg_vals[i] = 0.0
                continue
            # Sample possible outcomes from predictive distribution
            y_samp = rng.normal(loc=m, scale=s, size=self.n_mc)
            # For each hypothetical observation, compute hypothetical best mean
            # (approximate: we assume update only shifts this candidate)
            mu_new_best = np.minimum(mu_current_best, np.min(y_samp))
            kg_vals[i] = mu_current_best - mu_new_best

        return Xc[int(np.argmax(kg_vals))]

    def tell(self, x_uv: Array, y: float) -> None:
        self.X.append(np.asarray(x_uv))
        self.Y.append(float(y))

# ---------------------------------------------------------------------------
# DOE: Full Factorial Grid (exactly n_target points)
# ---------------------------------------------------------------------------


@dataclass
class DOE_FF(Method):
    """Full Factorial DOE on [0,1]^d; generates exactly n_target points.
    Optional Cranley–Patterson random shift to avoid symmetry hits.
    """
    n_target: Optional[int] = None
    random_shift: bool = False      # <-- NEW
    dim: Optional[int] = None
    plan: Optional[Array] = None
    ptr: int = 0

    def start(self, dim: int, rng: np.random.Generator) -> None:
        if self.n_target is None:
            raise ValueError("DOE_FF requires n_target (use budget).")
        self.dim = dim
        self.ptr = 0

        # choose a minimal levels-per-dimension L so that L^d >= n_target
        from math import ceil
        L = max(2, int(ceil(self.n_target ** (1.0 / dim))))
        axes = [np.linspace(0.0, 1.0, L) for _ in range(dim)]
        mesh = np.meshgrid(*axes, indexing="xy")
        pts = np.stack([m.ravel() for m in mesh], axis=-1)  # (L^d, d)

        # --- NEW: Cranley–Patterson shift u' = (u + delta) mod 1
        if self.random_shift:
            delta = rng.random(dim)
            pts = (pts + delta) % 1.0

        # ensure >= n_target; then shuffle & cut to exactly n_target
        if len(pts) < self.n_target:
            reps = int(np.ceil(self.n_target / len(pts)))
            pts = np.tile(pts, (reps, 1))
        rng.shuffle(pts)
        self.plan = pts[: self.n_target]

    def ask(self, n_remaining: int, rng: np.random.Generator) -> Array:
        if self.plan is None:
            raise RuntimeError("DOE_FF.start() must be called first.")
        if self.ptr >= len(self.plan):
            raise StopIteration("DOE_FF plan exhausted.")
        x = self.plan[self.ptr]
        self.ptr += 1
        return x

# ---------------------------------------------------------------------------
# DOE: Central Composite Design (face-centered) on [0,1]^d
# ---------------------------------------------------------------------------


Array = np.ndarray


def _lhs(n: int, d: int, rng: np.random.Generator) -> Array:
    """Simple Latin Hypercube in [0,1]^d."""
    X = np.empty((n, d), dtype=float)
    for j in range(d):
        perm = rng.permutation(n)
        X[:, j] = (perm + rng.random(n)) / n
    return X


@dataclass
class DOE_CCD(Method):
    """
    Face-centered Central Composite Design in [0,1]^d.
    Ensures exactly n_target points (truncates or tops-up with LHS).
    Optional Cranley–Patterson shift breaks symmetry.
    """
    n_target: Optional[int] = None
    center_reps: Optional[int] = None   # default chosen from dim in start()
    random_shift: bool = True
    jitter_center: bool = True
    jitter_scale: float = 1e-6
    plan: Optional[Array] = None
    dim: Optional[int] = None
    ptr: int = 0

    def _build_ccd(self, d: int, rng: np.random.Generator) -> Array:
        # Face-centered CCD (alpha = 1) in [-1,1]^d mapped to [0,1]^d via (z+1)/2
        # 1) factorial points: corners {-1, +1}^d  => in [0,1]^d {0,1}^d
        corners = np.array(np.meshgrid(*([[-1.0, 1.0]] * d), indexing="xy"))
        corners = corners.reshape(d, -1).T  # (2^d, d)
        # 2) axial points: +/- along each axis, others 0
        axes = []
        for j in range(d):
            e = np.zeros(d, dtype=float)
            e[j] = 1.0
            axes.append(+e)
            axes.append(-e)
        axes = np.array(axes, dtype=float)  # (2d, d)
        # 3) center points
        n_center = self.center_reps if self.center_reps is not None else max(
            1, d // 2)
        center = np.zeros((n_center, d), dtype=float)

        Z = np.vstack([corners, axes, center])  # in [-1,1]^d
        U = (Z + 1.0) * 0.5  # -> [0,1]^d

        # optional: tiny jitter for center replicates
        if self.jitter_center and n_center > 0:
            U[-n_center:, :] = np.clip(U[-n_center:, :] + rng.normal(
                0.0, self.jitter_scale, size=(n_center, d)), 0.0, 1.0)

        return U

    def start(self, dim: int, rng: np.random.Generator) -> None:
        if self.n_target is None:
            raise ValueError("DOE_CCD requires n_target (use budget).")
        self.dim = dim
        self.ptr = 0

        pts = self._build_ccd(dim, rng)

        # optional Cranley–Patterson shift to break symmetry
        if self.random_shift:
            delta = rng.random(dim)
            pts = (pts + delta) % 1.0

        # shuffle for robustness
        rng.shuffle(pts)

        # top-up or truncate to exactly n_target
        if len(pts) < self.n_target:
            k = int(self.n_target - len(pts))
            pts = np.vstack([pts, _lhs(k, dim, rng)])
        elif len(pts) > self.n_target:
            pts = pts[: self.n_target]

        self.plan = np.asarray(pts, dtype=float)

    def ask(self, n_remaining: int, rng: np.random.Generator) -> Array:
        if self.plan is None:
            raise RuntimeError("DOE_CCD.start() must be called first.")
        if self.ptr >= len(self.plan):
            raise StopIteration("DOE_CCD plan exhausted.")
        x = self.plan[self.ptr]
        self.ptr += 1
        return x


# ---------------------------------------------------------------------------
# DOE: D-optimal design (quadratic RSM) via greedy log-det (sequential)
# ---------------------------------------------------------------------------

def _quad_features(u: Array) -> Array:
    """
    Quadratic RSM features on centered coords z = 2u-1:
    [1, z_i, z_i^2, z_i*z_j (i<j)]
    """
    z = 2.0 * u - 1.0  # center to [-1,1]
    d = z.shape[-1]
    feats: List[float] = [1.0]
    # linear
    feats.extend(z.tolist())
    # pure quadratics
    feats.extend((z * z).tolist())
    # interactions
    for i in range(d):
        for j in range(i + 1, d):
            feats.append(float(z[i] * z[j]))
    return np.asarray(feats, dtype=float)


def _build_rsm_matrix(U: Array) -> Array:
    """Design matrix Phi with quadratic features for all candidate points."""
    Phi = np.vstack([_quad_features(u) for u in U])
    return Phi  # (N, p)


@dataclass
class OD(Method):
    """
    D-optimal design for quadratic RSM on [0,1]^d.
    Greedy sequential selection using the matrix determinant lemma.
    """
    n_target: Optional[int] = None
    n_candidates: Optional[int] = None   # default: max(50*d, 3*n_target)
    random_shift: bool = True            # shift candidate set to break symmetry
    regularization: float = 1e-10        # for initial M_inv = (1/eps) I
    plan: Optional[Array] = None
    dim: Optional[int] = None
    ptr: int = 0

    def _candidate_set(self, d: int, rng: np.random.Generator) -> Array:
        # Union of CCD points (face-centered) and LHS cloud
        n_cand = self.n_candidates or max(50 * d, 3 * int(self.n_target or 0))
        n_cand = max(n_cand, int(self.n_target or 0))
        ccd = DOE_CCD(n_target=None, center_reps=max(1, d // 2), random_shift=self.random_shift,
                      jitter_center=True, jitter_scale=1e-6)
        # build full CCD (without truncation)
        ccd_pts = ccd._build_ccd(d, rng)
        if self.random_shift:
            delta = rng.random(d)
            ccd_pts = (ccd_pts + delta) % 1.0

        lhs_pts = _lhs(n_cand, d, rng)
        U = np.vstack([ccd_pts, lhs_pts])
        # optional: deduplicate roughly
        U = np.clip(U, 0.0, 1.0)
        return U

    def start(self, dim: int, rng: np.random.Generator) -> None:
        if self.n_target is None:
            raise ValueError("OD requires n_target (use budget).")
        self.dim = dim
        self.ptr = 0

        Ucand = self._candidate_set(dim, rng)
        Phi = _build_rsm_matrix(Ucand)  # (N, p)
        N, p = Phi.shape

        # Greedy sequential D-opt using Sherman–Morrison updates on M_inv
        eps = float(self.regularization)
        M_inv = np.eye(p, dtype=float) / eps
        chosen: List[int] = []
        available = np.ones(N, dtype=bool)

        for _ in range(int(self.n_target)):
            # score s_j = 1 + phi_j^T M_inv phi_j  (maximizes det increment)
            # compute in chunks for memory safety
            best_idx, best_score = -1, -np.inf
            # vectorized scoring
            Phi_av = Phi[available]
            quad = np.einsum("ij,jk,ik->i", Phi_av, M_inv,
                             Phi_av)  # phi @ M_inv @ phi
            scores = 1.0 + quad
            # argmax among available
            rel_idx = int(np.argmax(scores))
            best_score = float(scores[rel_idx])
            # translate back to global index
            cand_indices = np.flatnonzero(available)
            j = int(cand_indices[rel_idx])

            # update M_inv via Sherman–Morrison with u = phi_j
            u = Phi[j:j+1, :].T  # (p,1)
            denom = 1.0 + float(u.T @ M_inv @ u)
            K = (M_inv @ u) @ (u.T @ M_inv) / denom
            M_inv = M_inv - K

            chosen.append(j)
            available[j] = False

        self.plan = Ucand[chosen, :]

    def ask(self, n_remaining: int, rng: np.random.Generator) -> Array:
        if self.plan is None:
            raise RuntimeError("OD.start() must be called first.")
        if self.ptr >= len(self.plan):
            raise StopIteration("OD plan exhausted.")
        x = self.plan[self.ptr]
        self.ptr += 1
        return x


# ---------------------------------------------------------------------------
# DOE: Latin Hypercube Sampling
# ---------------------------------------------------------------------------

@dataclass
class DOE_LHS(Method):
    """Latin Hypercube Sampling (non-adaptive)."""

    n_init: int
    dim: Optional[int] = None
    plan: Optional[Array] = None
    ptr: int = 0

    def start(self, dim: int, rng: np.random.Generator) -> None:
        self.dim = dim
        self.ptr = 0
        n = int(self.n_init)
        cut = np.linspace(0.0, 1.0, n + 1)
        u = rng.random((n, dim))
        X = np.empty((n, dim), dtype=float)
        for j in range(dim):
            X[:, j] = u[:, j] * (cut[1:] - cut[:-1]) + cut[:-1]
            rng.shuffle(X[:, j])
        self.plan = X

    def ask(self, n_remaining: int, rng: np.random.Generator) -> Array:
        if self.plan is None:
            raise RuntimeError("DOE_LHS.start() must be called before ask().")
        if self.ptr >= len(self.plan):
            raise StopIteration("DOE_LHS plan exhausted.")
        x = self.plan[self.ptr]
        self.ptr += 1
        return x


# ---------------------------------------------------------------------------
# Random baseline
# ---------------------------------------------------------------------------

@dataclass
class Random(Method):
    """Pure random sampling baseline in [0,1]^d."""
    dim: Optional[int] = None

    def start(self, dim: int, rng: np.random.Generator) -> None:
        self.dim = dim

    def ask(self, n_remaining: int, rng: np.random.Generator) -> Array:
        return _rand_points(1, self.dim, rng)[0]

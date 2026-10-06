"""Type I error experiment for the Wasserstein practical-significance test.

Implements the spec in wasserstein_type1_short.md:
  - tie-corrected transportation-budget PIT bounds (empirical F_t),
  - exact PIT bounds for a normal F_t (R = infinity baseline),
  - exact W*(delta) via linear assignment,
  - Monte Carlo critical value c_alpha,
  - null scenarios N0-N5 and the macroreplication loop.
"""
import itertools

import numpy as np
from scipy.optimize import linear_sum_assignment, linprog
from scipy.special import ndtr

SQRT_2_OVER_PI = np.sqrt(2.0 / np.pi)
_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


def _npdf(z):
    return _INV_SQRT_2PI * np.exp(-0.5 * z * z)


# ---------------------------------------------------------------------------
# PIT bounds
# ---------------------------------------------------------------------------
def _greedy_moved_mass(dist, delta, R):
    """Mass that can be moved across y_t with budget delta.

    dist: (T, R) array of nonnegative distances, np.inf for atoms that cannot move.
    Returns (T,) mass moved = k/R + eta.
    """
    d = np.sort(dist, axis=1)
    cum = np.cumsum(d, axis=1) / R  # cost of moving the first j atoms
    k = np.sum(cum <= delta, axis=1)  # whole atoms moved
    rows = np.arange(d.shape[0])
    spent = np.where(k > 0, cum[rows, np.maximum(k - 1, 0)], 0.0)
    has_next = k < d.shape[1]
    d_next = np.where(has_next, d[rows, np.minimum(k, d.shape[1] - 1)], np.inf)
    with np.errstate(invalid="ignore", divide="ignore"):
        eta = np.where(np.isfinite(d_next), (delta - spent) / d_next, 0.0)
    eta = np.clip(eta, 0.0, 1.0 / R)
    return k / R + eta


def pit_bounds_empirical(y, Ysim, delta):
    """Tie-corrected PIT interval for empirical F_t.

    y: (T,) observations; Ysim: (T, R) candidate replications.
    """
    T, R = Ysim.shape
    Fhat = np.mean(Ysim <= y[:, None], axis=1)
    if delta == 0:
        return Fhat.copy(), Fhat.copy()
    d_low = y[:, None] - Ysim
    d_low = np.where(d_low >= 0, d_low, np.inf)  # atoms <= y, ties included
    d_up = Ysim - y[:, None]
    d_up = np.where(d_up > 0, d_up, np.inf)  # atoms > y
    lo = Fhat - _greedy_moved_mass(d_low, delta, R)
    hi = Fhat + _greedy_moved_mass(d_up, delta, R)
    return np.clip(lo, 0.0, 1.0), np.clip(hi, 0.0, 1.0)


def pit_bounds_normal(y, mu, sigma, delta, iters=50):
    """Exact PIT interval for F_t = N(mu_t, sigma^2) (R = infinity baseline)."""
    zy = (y - mu) / sigma
    Fy = ndtr(zy)
    if delta == 0:
        return Fy.copy(), Fy.copy()
    phy = _npdf(zy)

    def c_minus(za):  # cost of pushing mass in [a, y] past y
        return (y - mu) * (Fy - ndtr(za)) + sigma * (phy - _npdf(za))

    def c_plus(zb):  # cost of pulling mass in [y, b] back to y
        return (mu - y) * (ndtr(zb) - Fy) + sigma * (phy - _npdf(zb))

    # lower: find za <= zy with c_minus(za) = delta (c_minus decreasing in za)
    lo_z, hi_z = np.full_like(zy, -40.0), zy.copy()
    for _ in range(iters):
        mid = 0.5 * (lo_z + hi_z)
        too_much = c_minus(mid) > delta
        lo_z = np.where(too_much, mid, lo_z)
        hi_z = np.where(too_much, hi_z, mid)
    lower = np.where(c_minus(-40.0 * np.ones_like(zy)) <= delta, 0.0, ndtr(hi_z))

    # upper: find zb >= zy with c_plus(zb) = delta (c_plus increasing in zb)
    lo_z, hi_z = zy.copy(), np.full_like(zy, 40.0)
    for _ in range(iters):
        mid = 0.5 * (lo_z + hi_z)
        too_much = c_plus(mid) > delta
        hi_z = np.where(too_much, mid, hi_z)
        lo_z = np.where(too_much, lo_z, mid)
    upper = np.where(c_plus(40.0 * np.ones_like(zy)) <= delta, 1.0, ndtr(lo_z))
    return lower, upper


# ---------------------------------------------------------------------------
# W1 to Uniform(0,1) and W*(delta)
# ---------------------------------------------------------------------------
def _slot_cost(d, T):
    """Exact integral of |u - s| over a slot of width 1/T, u at distance d from its midpoint."""
    w = 1.0 / T
    return np.where(d <= w / 2, w * w / 4 + d * d, w * d)


def w1_uniform(u):
    """Exact W1 between the empirical distribution of u (last axis) and U(0,1)."""
    u = np.sort(u, axis=-1)
    T = u.shape[-1]
    m = (2 * np.arange(1, T + 1) - 1) / (2 * T)
    return _slot_cost(np.abs(u - m), T).sum(axis=-1)


def w_star(lo, hi, return_points=False):
    """Exact W*(delta) = min over u_t in [lo_t, hi_t] of W1(empirical, U(0,1))."""
    T = lo.shape[0]
    m = (2 * np.arange(1, T + 1) - 1) / (2 * T)
    d = np.maximum(0.0, np.maximum(lo[:, None] - m[None, :], m[None, :] - hi[:, None]))
    C = _slot_cost(d, T)
    rows, cols = linear_sum_assignment(C)
    val = C[rows, cols].sum()
    if return_points:
        return val, np.clip(m[cols], lo[rows], hi[rows])
    return val


def w_star_sort_clip(lo, hi):
    """Heuristic: sort intervals by midpoint, clip slot midpoints into them."""
    T = lo.shape[0]
    m = (2 * np.arange(1, T + 1) - 1) / (2 * T)
    order = np.argsort(0.5 * (lo + hi), kind="stable")
    u = np.clip(m, lo[order], hi[order])
    return w1_uniform(u)


def w_star_bruteforce(lo, hi):
    T = lo.shape[0]
    m = (2 * np.arange(1, T + 1) - 1) / (2 * T)
    best = np.inf
    for perm in itertools.permutations(range(T)):
        u = np.clip(m, lo[list(perm)], hi[list(perm)])
        best = min(best, w1_uniform(u))
    return best


def critical_value(T, alpha=0.05, n_draws=100_000, seed=0, chunk=20_000):
    rng = np.random.default_rng(seed)
    stats = [w1_uniform(rng.random((min(chunk, n_draws - i), T))) for i in range(0, n_draws, chunk)]
    return np.quantile(np.concatenate(stats), 1 - alpha)


# ---------------------------------------------------------------------------
# LP check of the PIT bounds (small R)
# ---------------------------------------------------------------------------
def pit_bounds_lp(y, x, delta, eps=1e-9, n_grid=60):
    """PIT bounds by LP over transport plans from atoms x (weight 1/R) to a grid."""
    R = len(x)
    grid = np.unique(np.concatenate([x, [y, y + eps],
                                     np.linspace(x.min() - 1, x.max() + 1, n_grid)]))
    G = len(grid)
    cost = np.abs(x[:, None] - grid[None, :]).ravel()
    A_eq = np.kron(np.eye(R), np.ones(G))
    b_eq = np.full(R, 1.0 / R)
    below = np.tile((grid <= y).astype(float), R)
    out = []
    for sign in (1.0, -1.0):  # minimise then maximise F(y)
        res = linprog(sign * below, A_ub=cost[None, :], b_ub=[delta],
                      A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
        out.append(below @ res.x)
    return out[0], out[1]


# ---------------------------------------------------------------------------
# Scenarios and macroreplications
# ---------------------------------------------------------------------------
def candidate_means(T):
    t = np.arange(1, T + 1)
    return 5.0 + 2.0 * np.sin(2 * np.pi * t / 25)


def scenario_params(scenario, T, delta, rng):
    """Per-period (shift, scale) of G_t = N(mu_t + shift_t, scale_t^2), with F_t = N(mu_t, 1)."""
    shift, scale = np.zeros(T), np.ones(T)
    if scenario == "N0":
        pass
    elif scenario == "N1":
        shift[:] = delta
    elif scenario == "N2":
        shift[:] = -delta
    elif scenario == "N3":
        scale[:] = 1 + delta / SQRT_2_OVER_PI
    elif scenario == "N4":
        scale[:] = 1 - delta / SQRT_2_OVER_PI
    elif scenario == "N5":
        dt = delta * (1 - rng.random(T))  # U(0, delta]
        kind = rng.integers(0, 4, T)
        shift = np.select([kind == 0, kind == 1], [dt, -dt], 0.0)
        scale = np.select([kind == 2, kind == 3],
                          [1 + dt / SQRT_2_OVER_PI, 1 - dt / SQRT_2_OVER_PI], 1.0)
    else:
        raise ValueError(scenario)
    assert np.all(scale > 0)
    return shift, scale


def true_w1(shift, scale):
    """W1(G_t, F_t) per period for the shift-only or scale-only perturbations used here."""
    return np.abs(shift) + np.abs(scale - 1) * SQRT_2_OVER_PI


def run_cell(scenario, T, R, delta, c_alpha, M, seed, gen_delta=None):
    """M macroreplications of one design cell. R=None means exact F_t.

    The test is run at radius delta; G_t is generated at radius gen_delta
    (default delta, i.e. the null boundary; gen_delta > delta gives power).

    Returns dict with the rejection count and the distribution of W*.
    """
    rng = np.random.default_rng(seed)
    mu = candidate_means(T)
    gen_delta = delta if gen_delta is None else gen_delta
    shift, scale = scenario_params(scenario, T, gen_delta, rng)  # fixed within the cell
    wstars = np.empty(M)
    for i in range(M):
        y = mu + shift + scale * rng.standard_normal(T)
        if R is None:
            lo, hi = pit_bounds_normal(y, mu, 1.0, delta)
        else:
            Ysim = mu[:, None] + rng.standard_normal((T, R))
            lo, hi = pit_bounds_empirical(y, Ysim, delta)
        wstars[i] = w_star(lo, hi)
    rej = int(np.sum(wstars > c_alpha))
    return dict(scenario=scenario, T=T, R=("inf" if R is None else R), delta=delta,
                gen_delta=gen_delta,
                M=M, rejections=rej, alpha_hat=rej / M,
                max_true_w1=float(true_w1(shift, scale).max()),
                mean_wstar=float(wstars.mean()))


def wilson_ci(k, n, z=1.959963984540054):
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half

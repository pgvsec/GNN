"""Batched fixed-point solver for the Hill GRN ODE.

    dx_i/dt = P_i * prod_{j->i} H_ji(x_j) - K_i * x_i

Sections
  1. SolverConfig, parameter preparation, seeding
  2. Newton iteration in log coordinates with a vectorized line search
  3. Deduplication of converged points
  4. Spectral properties of each fixed point and the Brouwer certificate
  5. solve_draws: rounds until the index sum is 1 and fresh seeds find nothing new

The physics is imported from model.py, so the generator and the training loss evaluate the
same ODE and the same analytic Jacobian.

Shapes
  B draws of one gene count N, each with its own graph and parameters, are solved together.
  Seeds and iterates are [B, K, N] (K seeds per draw); the Jacobian is [B, K, N, N] with
  J[..., i, j] = d f_i / d x_j. Draws are grouped by N, so no padding is ever solved.
  Everything runs in float64 on the chosen device; eigenvalues are computed on the CPU.

Why these choices
  Newton runs on y = log x. Fixed points of switch-like networks sit at expression levels that
  span many decades, positivity is automatic, and steps are relative. Rows of the linear system
  are divided by |P R| + K x of that gene, so every gene is converged to a relative tolerance
  instead of an absolute one: a gene sitting at 1e-9 of its box is resolved as precisely as
  one at the top of the box.
  Completeness: all fixed points lie in the positively invariant box and the vector field
  points inward on its boundary, so the Brouwer degree of -f on the box is 1. When all fixed
  points are hyperbolic, the indices sgn det(-J) of ALL fixed points sum to 1. The sum over
  the points found therefore detects a missing root only when the missing indices do not
  cancel. A missed saddle-node pair (index +1 and -1) leaves the sum at 1 and is invisible to
  it. The index sum is a necessary condition, not a proof of completeness.
  Stopping rule: a draw is finished when its index sum is 1, every point is hyperbolic, and a
  confirmation round of fresh seeds found nothing new. A draw that fails either test is
  reseeded with twice as many seeds, up to `rounds`. generate.py --mode audit measures how
  many fixed points a much larger seed budget still adds.
"""
import itertools
import math
from dataclasses import dataclass

import numpy as np
import torch

from model import hill_jacobian, ode_rhs

STEP_CLIP = 2.0                  # largest log-space Newton step per iteration
ALPHAS = (1.0, 0.5, 0.2, 0.05)   # line-search damping factors
WARMUP = 12                      # Newton iterations on the full batch before unfinished systems are re-batched
CLUSTER_BIN = 5e-6               # pre-clustering bin width in log x
CLUSTER_TOL = 1e-5               # points closer than this in log x are the same fixed point
MAX_CLUSTERS = 300               # cap on distinct candidates kept per draw
HYPER_TOL = 1e-9                 # |Re lambda| below this fraction of the spectral radius: non-hyperbolic
FIELDS = ("state", "charpoly", "stab", "bidx", "hopf", "zeroeig", "resid")


@dataclass(frozen=True)
class SolverConfig:
    seeds: tuple = (96, 128, 160, 192, 224, 256, 288, 320, 352)   # random seeds in the first two rounds, n = 2 .. 10
    rounds: int = 5                # rounds 0 and 1 use seeds / 2 each, round r >= 2 uses seeds / 2 * 2**(r - 1)
    newton_steps: int = 60
    tol: float = 1e-12             # iteration stops for a system below this relative residual
    accept_tol: float = 1e-9       # a converged point must be below this to be kept
    eps_box: float = 0.05
    max_corners: int = 64          # corner seeds per draw in round 0
    confirm: bool = True           # require a second round that finds nothing new before a draw is finished


def corner_count(n, cap):
    return 2 ** n if 2 ** n <= cap else cap


# ---------------------------------------------------------------------------
# 1. Parameters and seeds
# ---------------------------------------------------------------------------
def prepare_params(par, eps_box, device):
    """Numpy parameter dict -> float64 torch dict on the device, with the invariant-box bound.

    par: P, K [B, N]; adj, T, nh, F [B, N, N] indexed [source, target]; neutral entries
    (adj = 0, T = nh = F = 1) where there is no edge."""
    def t(a):
        return torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float64, device=device)

    P, Kd, adj, T, nh, F = (t(par[k]) for k in ("P", "K", "adj", "T", "nh", "F"))
    gain = torch.where(adj != 0, F.clamp(min=1.0), torch.ones_like(F)).prod(dim=1)
    xmax = (1.0 + eps_box) * P / Kd * gain
    return {"P": P, "K": Kd, "adj": adj, "T": T, "nh": nh, "F": F, "xmax": xmax}


def subset(p, idx):
    return {k: v[idx] for k, v in p.items()}


def make_seeds(p, n_rand, gen, max_corners, warm=None):
    """[B, corners + n_rand + W, N] starting points.

    Corner seeds (up to max_corners; none if 0) put every gene at the lowest or highest level
    its regulation allows (P/K times the product of min or max fold-changes), with log-normal
    jitter. Multistable switches have their states near such corners. The random seeds are
    half uniform in the box and half log-uniform over seven decades below the box bound,
    which reaches low-expression states that uniform sampling almost never lands near.
    warm [B, W, N], if given, holds known points (the fixed points of a nearby parameter
    vector) that are added as seeds."""
    P, Kd, xmax, adj, F = p["P"], p["K"], p["xmax"], p["adj"], p["F"]
    B, N = P.shape
    dev, dt = P.device, P.dtype
    parts = []
    if max_corners > 0:
        edge = adj != 0
        gain_hi = torch.where(edge, F.clamp(min=1.0), torch.ones_like(F)).prod(dim=1)
        gain_lo = torch.where(edge, F.clamp(max=1.0), torch.ones_like(F)).prod(dim=1)
        hi, lo = P / Kd * gain_hi, P / Kd * gain_lo
        if 2 ** N <= max_corners:
            pat = torch.tensor(list(itertools.product([False, True], repeat=N)), device=dev)
        else:
            pat = torch.rand(max_corners, N, generator=gen, device=dev, dtype=dt) < 0.5
        corner = torch.where(pat.unsqueeze(0), hi.unsqueeze(1), lo.unsqueeze(1))
        parts.append(corner * torch.exp(0.25 * torch.randn(corner.shape, generator=gen, device=dev, dtype=dt)))
    n_lin = n_rand // 2
    parts.append(torch.rand(B, n_lin, N, generator=gen, device=dev, dtype=dt) * xmax.unsqueeze(1))
    u = torch.rand(B, n_rand - n_lin, N, generator=gen, device=dev, dtype=dt)
    parts.append(xmax.unsqueeze(1) * torch.exp(u * math.log(1e-7)))
    if warm is not None:
        parts.append(warm)
    seeds = torch.minimum(torch.cat(parts, dim=1), xmax.unsqueeze(1))
    return torch.maximum(seeds, 1e-12 * xmax.unsqueeze(1))


def pad_warm(warm, xmax, device):
    """[B, W, N] tensor from per-draw arrays of known points. Rows are padded by repeating the
    first point; a draw without any point gets a point inside the box."""
    B, N = xmax.shape
    W = max(1, max(w.shape[0] for w in warm))
    out = np.empty((B, W, N))
    for i, w in enumerate(warm):
        w = w if w.shape[0] > 0 else 0.1 * xmax[i][None]
        out[i, :w.shape[0]] = w
        out[i, w.shape[0]:] = w[0]
    return torch.as_tensor(out, dtype=torch.float64, device=device)


# ---------------------------------------------------------------------------
# 2. Newton iteration
# ---------------------------------------------------------------------------
def newton(x, p, steps, tol):
    """Damped Newton in y = log x on every (draw, seed) system at once.

    x: [B, K, N] positive starting points. Returns the final points and the relative residual
    max_i |f_i| / (|P R|_i + K_i x_i) of every system, [B, K].

    The first WARMUP iterations run on the full batch. Most seeds have converged by then, and
    a large share of the rest are dead (no damped step reduces their residual, so the iterate
    can never change). Only the systems that are neither converged nor dead continue, each
    with its own gathered parameters as a batch of single-seed draws, so the long tail of
    slowly converging seeds no longer drags the whole batch through every iteration."""
    first = min(WARMUP, steps)
    x, r, dead = newton_steps(x, p, first, tol)
    if steps > first:
        idx = ((r >= tol) & ~dead).nonzero()
        if idx.shape[0] > 0:
            b, k = idx[:, 0], idx[:, 1]
            xs, rs, _ = newton_steps(x[b, k].unsqueeze(1), subset(p, b), steps - first, tol)
            x, r = x.clone(), r.clone()
            x[b, k], r[b, k] = xs[:, 0], rs[:, 0]
    return x, r


def newton_steps(x, p, steps, tol):
    """`steps` damped Newton iterations. Returns points, relative residuals and the dead mask.

    A step that would not reduce the residual is rejected, so a system can stall but never
    diverges. A system for which no damping factor improves the residual is dead: the next
    iteration would repeat the same computation."""
    P, Kd, adj, T, nh, F = p["P"], p["K"], p["adj"], p["T"], p["nh"], p["F"]
    hi = p["xmax"].unsqueeze(1)

    def evaluate(z):
        f = ode_rhs(z, P, Kd, adj, T, nh, F)
        s = (f + 2.0 * Kd.unsqueeze(1) * z).clamp_min(1e-300)       # |P R| + K x, always positive
        return f / s, s

    f, s = evaluate(x)
    r = f.abs().amax(dim=-1)
    dead = torch.zeros_like(r, dtype=torch.bool)
    for _ in range(steps):
        active = (r >= tol) & ~dead
        if not bool(active.any()):
            break
        J = hill_jacobian(x, P, Kd, adj, T, nh, F)
        A = J * x.unsqueeze(2) / s.unsqueeze(-1)                    # d f_i / d y_j, rows scaled by s_i
        dy, info = torch.linalg.solve_ex(A, -f.unsqueeze(-1))
        dy = torch.nan_to_num(dy.squeeze(-1), nan=0.0, posinf=0.0, neginf=0.0)
        dy = dy.clamp(-STEP_CLIP, STEP_CLIP) * (info == 0).unsqueeze(-1)
        bx, bf, bs, br = x, f, s, r
        for a in ALPHAS:
            xn = torch.minimum(x * torch.exp(a * dy), hi)
            fn, sn = evaluate(xn)
            rn = fn.abs().amax(dim=-1)
            better = rn < br
            m = better.unsqueeze(-1)
            bx, bf, bs = torch.where(m, xn, bx), torch.where(m, fn, bf), torch.where(m, sn, bs)
            br = torch.where(better, rn, br)
        dead = dead | (active & ~(br < r))
        x, f, s, r = bx, bf, bs, br
    return x, r, dead


# ---------------------------------------------------------------------------
# 3. Deduplication
# ---------------------------------------------------------------------------
def cluster_roots(points):
    """Unique fixed points among converged points of one draw. points: [m, N] positive.

    Points are first binned in log x (fine bins, so identical roots almost always share a bin),
    then bin means closer than CLUSTER_TOL in every coordinate are merged by label propagation,
    which also repairs roots that straddled a bin boundary."""
    if points.shape[0] <= 1:
        return points
    lg = np.log(points)
    q = np.round(lg / CLUSTER_BIN).astype(np.int64)
    _, inv, cnt = np.unique(q, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    c = cnt.shape[0]
    mean = np.zeros((c, lg.shape[1]))
    np.add.at(mean, inv, lg)
    mean /= cnt[:, None]
    if c > MAX_CLUSTERS:
        keep = np.argsort(-cnt)[:MAX_CLUSTERS]
        mean, c = mean[keep], MAX_CLUSTERS
    close = np.abs(mean[:, None, :] - mean[None, :, :]).max(axis=-1) < CLUSTER_TOL
    lab = np.arange(c)
    for _ in range(c):
        new = np.where(close, lab[None, :], c).min(axis=1)
        if np.array_equal(new, lab):
            break
        lab = new
    return np.exp(mean[np.unique(lab)])


# ---------------------------------------------------------------------------
# 4. Spectral properties
# ---------------------------------------------------------------------------
def charpoly_from_eigenvalues(ev):
    """Coefficients c_1 .. c_n of det(lambda I - J) = lambda^n + c_1 lambda^(n-1) + ... + c_n.

    ev: [M, N] complex. The last coefficient is c_n = det(-J), the Brouwer index sign."""
    M, N = ev.shape
    c = np.zeros((M, N + 1), dtype=np.complex128)
    c[:, 0] = 1.0
    for i in range(N):
        c[:, 1:i + 2] = c[:, 1:i + 2] - ev[:, i:i + 1] * c[:, 0:i + 1]
    return c[:, 1:].real


def empty_properties(N):
    return {"counts": np.zeros(0, np.int64), "state": np.zeros((0, N)), "charpoly": np.zeros((0, N)),
            "stab": np.zeros(0, np.uint8), "bidx": np.zeros(0, np.int8), "hopf": np.zeros(0, np.float32),
            "zeroeig": np.zeros(0, np.float32), "resid": np.zeros(0, np.float32)}


def root_properties(roots, p):
    """Properties of the fixed points of B draws.

    roots: list of B arrays [c_b, N]; p: parameters of the same B draws. Returns the flat
    per-point arrays in FIELDS plus counts [B], bsum [B] (Brouwer index sum) and cons [B].

    Stability classes: 0 stable node, 1 stable spiral, 2 unstable node, 3 unstable spiral,
    4 saddle, 5 non-hyperbolic. Index sgn det(-J) = (-1)^(number of real positive eigenvalues);
    complex pairs contribute a positive factor. A draw is consistent when its indices sum to
    1 and no point is non-hyperbolic."""
    B, N = p["P"].shape
    counts = np.array([r.shape[0] for r in roots], dtype=np.int64)
    out = empty_properties(N)
    out["counts"] = counts
    out["bsum"] = np.zeros(B, np.int64)
    out["cons"] = np.zeros(B, dtype=bool)
    if counts.sum() == 0:
        return out
    did_np = np.repeat(np.arange(B), counts)
    dev = p["P"].device
    did = torch.as_tensor(did_np, device=dev)
    X = torch.as_tensor(np.concatenate(roots, axis=0), dtype=torch.float64, device=dev).unsqueeze(1)
    g = {k: v[did] for k, v in p.items()}
    f = ode_rhs(X, g["P"], g["K"], g["adj"], g["T"], g["nh"], g["F"])
    resid = (f.abs() / (f + 2.0 * g["K"].unsqueeze(1) * X)).amax(dim=-1)[:, 0].cpu().numpy()
    J = hill_jacobian(X, g["P"], g["K"], g["adj"], g["T"], g["nh"], g["F"])[:, 0].cpu().numpy()
    finite = np.isfinite(J).all(axis=(1, 2))
    J[~finite] = -np.eye(N)                       # placeholder; these points are flagged below
    ev = np.linalg.eigvals(J)

    re, im = ev.real, ev.imag
    rho = np.abs(ev).max(axis=1)
    tol = HYPER_TOL * rho[:, None]
    nonhyp = (np.abs(re) <= tol).any(axis=1) | ~finite
    cplx = (np.abs(im) > tol).any(axis=1)
    all_neg, all_pos = (re < 0).all(axis=1), (re > 0).all(axis=1)
    stab = np.full(re.shape[0], 4, dtype=np.uint8)
    stab[all_neg & ~cplx], stab[all_neg & cplx] = 0, 1
    stab[all_pos & ~cplx], stab[all_pos & cplx] = 2, 3
    stab[nonhyp] = 5
    n_pos_real = ((re > 0) & (np.abs(im) <= tol)).sum(axis=1)
    bidx = np.where(nonhyp, 0, 1 - 2 * (n_pos_real % 2)).astype(np.int8)

    cm = np.abs(im) > tol
    rows = np.arange(re.shape[0])
    pick = np.where(cm, np.abs(re), np.inf).argmin(axis=1)
    hopf = np.where(cm.any(axis=1), np.abs(re[rows, pick]) / np.maximum(np.abs(im[rows, pick]), 1e-300), np.nan)

    out.update({"state": np.concatenate(roots, axis=0), "charpoly": charpoly_from_eigenvalues(ev),
                "stab": stab, "bidx": bidx, "hopf": hopf.astype(np.float32),
                "zeroeig": (np.abs(ev).min(axis=1) / rho).astype(np.float32),
                "resid": resid.astype(np.float32)})
    out["bsum"] = np.bincount(did_np, weights=bidx, minlength=B).astype(np.int64)
    bad = np.bincount(did_np, weights=nonhyp.astype(np.float64), minlength=B) > 0
    out["cons"] = (out["bsum"] == 1) & ~bad
    return out


def split_properties(pr):
    """Per-draw views of the flat arrays returned by root_properties."""
    cuts = np.cumsum(pr["counts"])[:-1]
    cols = {f: np.split(pr[f], cuts) for f in FIELDS}
    return [{f: cols[f][i] for f in FIELDS} for i in range(pr["counts"].shape[0])]


# ---------------------------------------------------------------------------
# 5. Draw solver
# ---------------------------------------------------------------------------
def solve_draws(par, scfg, n, gen, device, warm=None):
    """All fixed points of B parameter draws that share a gene count n.

    Round 0 uses corner seeds, seeds / 2 random seeds, and the optional warm points (a list of
    B arrays [c, N] of known points, for example the fixed points of a nearby draw). Round 1
    uses seeds / 2 fresh random seeds, round r >= 2 uses seeds / 2 * 2**(r - 1). New points are
    merged with the old ones. A draw is finished when its index sum is 1, no point is
    non-hyperbolic, and the round just run (not the first) found no new point; the other
    draws continue until `rounds` is exhausted. With scfg.confirm False one round suffices.
    Returns counts [B], bsum [B], cons [B], xmax [B, N] and the flat per-point arrays in
    FIELDS ordered by draw, plus seed_ok (fraction of round-0 seeds that converged) and
    rounds_used."""
    p = prepare_params(par, scfg.eps_box, device)
    B, N = p["P"].shape
    xmax_np = p["xmax"].cpu().numpy()
    warm_t = pad_warm(warm, xmax_np, device) if warm is not None else None
    roots = [np.zeros((0, N)) for _ in range(B)]
    store = [None] * B
    bsum = np.zeros(B, np.int64)
    cons = np.zeros(B, dtype=bool)
    active = np.arange(B)
    seed_ok = 0.0
    min_rounds = 2 if scfg.confirm else 1
    for rnd in range(scfg.rounds):
        pa = p if active.size == B else subset(p, torch.as_tensor(active, device=device))
        n_rand = max(scfg.seeds[n - 2] // 2, 8) * 2 ** max(rnd - 1, 0)
        x0 = make_seeds(pa, n_rand, gen, scfg.max_corners if rnd == 0 else 0, warm_t if rnd == 0 else None)
        x, r = newton(x0, pa, scfg.newton_steps, scfg.tol)
        ok = (r < scfg.accept_tol).cpu().numpy()
        if rnd == 0:
            seed_ok = float(ok.mean())             # fraction of round-0 seeds that converged
        xs = x.cpu().numpy()
        grew = np.zeros(active.size, dtype=bool)
        for i, b in enumerate(active):
            before = roots[b].shape[0]
            roots[b] = cluster_roots(np.concatenate([roots[b], xs[i][ok[i]]], axis=0))
            grew[i] = roots[b].shape[0] > before
        pr = root_properties([roots[b] for b in active], pa)
        pieces = split_properties(pr)
        for i, b in enumerate(active):
            store[b] = pieces[i]
            bsum[b] = pr["bsum"][i]
            cons[b] = pr["cons"][i]
        active = active[~(pr["cons"] & ~grew & (rnd >= min_rounds - 1))]
        if active.size == 0:
            break
    res = {"counts": np.array([s["bidx"].shape[0] for s in store], dtype=np.int64),
           "bsum": bsum, "cons": cons, "xmax": xmax_np,
           "seed_ok": seed_ok, "rounds_used": rnd + 1}
    for f in FIELDS:
        res[f] = np.concatenate([s[f] for s in store], axis=0)
    return res


def summarize_solution(res):
    """Monitoring statistics of a solved batch, for the generator's per-shard log.

    n_draws, mean and maximum fixed-point count, fraction of draws certified by the Brouwer
    sum, fraction of draws with at least two stable points (bistable or more), the share of
    each stability class, and the fraction of first-round seeds that converged."""
    n_draws = int(res["counts"].shape[0])
    did = np.repeat(np.arange(n_draws), res["counts"])
    stable = np.isin(res["stab"], (0, 1))
    n_stable = np.bincount(did, weights=stable.astype(np.float64), minlength=n_draws)
    total = max(int(res["stab"].shape[0]), 1)
    return {
        "n_draws": n_draws,
        "mean_fp": float(res["counts"].mean()) if n_draws else 0.0,
        "max_fp": int(res["counts"].max()) if n_draws else 0,
        "certified": float(res["cons"].mean()) if n_draws else 0.0,
        "multistable": float((n_stable >= 2).mean()) if n_draws else 0.0,
        "class_share": [float((res["stab"] == c).sum()) / total for c in range(6)],
        "seed_ok": float(res["seed_ok"]),
    }


# ---------------------------------------------------------------------------
# 6. Independent verification
# ---------------------------------------------------------------------------
def ode_numpy(x, P, Kd, adj, T, nh, F):
    """ODE right-hand side for one point, written independently of model.py.

    x [N]; P, Kd [N]; adj, T, nh, F [N, N] indexed [source, target].
    Returns f [N] and the gene-wise scale |P R| + K x used for relative residuals."""
    r = (np.maximum(x, 1e-300)[:, None] / T) ** nh
    H = np.where(adj != 0, (F * r + 1.0) / (r + 1.0), 1.0)
    production = P * H.prod(axis=0)
    return production - Kd * x, production + Kd * x


def verify_batch(par, res, scfg, device, max_points=256, seed=0):
    """Independent checks of a solved batch. Raises RuntimeError naming the first failure.

    1. The residual, recomputed in numpy, is small at every sampled fixed point.
    2. The analytic Jacobian agrees with central finite differences within an entry-wise
       error bound (truncation plus roundoff), which checks the hand-derived Hill derivative.
    3. Every point lies inside the invariant box.
    4. The stored Brouwer index of a hyperbolic point equals the sign of the last
       characteristic polynomial coefficient, c_n = det(-J).
    5. The stored index sums and consistency flags are reproduced from the stored points."""
    counts = res["counts"]
    if counts.sum() == 0:
        return
    B, N = par["P"].shape
    did = np.repeat(np.arange(B), counts)
    if np.bincount(did, weights=res["bidx"], minlength=B).astype(np.int64).tolist() != res["bsum"].tolist():
        raise RuntimeError("stored Brouwer sums do not equal the sum of stored indices")
    p = prepare_params(par, scfg.eps_box, device)
    pick = np.random.default_rng(seed).choice(did.shape[0], size=min(max_points, did.shape[0]), replace=False)
    worst_res, worst_jac = 0.0, 0.0
    for i in pick:
        b = int(did[i])
        x = res["state"][i]
        args = (par["P"][b], par["K"][b], par["adj"][b], par["T"][b], par["nh"][b], par["F"][b])
        f, scale = ode_numpy(x, *args)
        worst_res = max(worst_res, float((np.abs(f) / scale).max()))
        if (x > res["xmax"][b] * (1.0 + 1e-9)).any():
            raise RuntimeError("a fixed point lies outside the invariant box")
        xt = torch.as_tensor(x, dtype=torch.float64, device=device).view(1, 1, N)
        gb = subset(p, torch.as_tensor([b], device=device))
        J = hill_jacobian(xt, gb["P"], gb["K"], gb["adj"], gb["T"], gb["nh"], gb["F"])[0, 0].cpu().numpy()
        J_fd = np.zeros((N, N))
        h = 1e-6 * x
        for j in range(N):
            xp, xm = x.copy(), x.copy()
            xp[j] += h[j]
            xm[j] -= h[j]
            J_fd[:, j] = (ode_numpy(xp, *args)[0] - ode_numpy(xm, *args)[0]) / (2.0 * h[j])
        # Entry-wise error bound: truncation (1e-5 of the entry) plus the roundoff of the two
        # evaluations of f_i, which is about eps * scale_i / h_j and is large for a gene whose
        # expression is far below the others. A wrong derivative exceeds this bound by orders
        # of magnitude; a correct one stays within it.
        bound = 1e-5 * np.abs(J) + 64.0 * np.finfo(np.float64).eps * scale[:, None] / h[None, :]
        worst_jac = max(worst_jac, float((np.abs(J - J_fd) / bound).max()))
        if res["stab"][i] != 5 and np.sign(res["charpoly"][i, -1]) != res["bidx"][i]:
            raise RuntimeError("Brouwer index disagrees with the sign of c_n")
    if worst_res > 1e-8:
        raise RuntimeError(f"fixed-point residual {worst_res:.2e} exceeds 1e-8")
    if worst_jac > 1.0:
        raise RuntimeError(f"analytic Jacobian exceeds the finite-difference error bound by a factor {worst_jac:.2e}")
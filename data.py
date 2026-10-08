"""Corpus A access layer.

Sections
  1. Constants shared by all modules
  2. build_cache: generator shards -> flat, memory-mappable numpy arrays, then verification
  3. SccDataset, collate, make_loader, describe_dataset
  4. scc_to_batch: batch of one SCC from raw parameters (inference / composition)

Tensor conventions
  Graph tensors are dense and indexed [source, target].
  adj    [B, N, N]  sign of the edge (+1 / -1), 0 where there is no edge
  T, nh, F [B, N, N] Hill threshold, coefficient, fold-change; neutral (1) where no edge
  P, K, xmax [B, N]  production, degradation, invariant-box bound
  fp_*   [B, Kt, ...] fixed-point targets padded to the largest count in the batch

Cache layout (written by build_cache, read by SccDataset). Keys are grouped by prefix:
  g_*  per graph        e_*  per edge of a graph      s_*  per sample
  ep_* per edge of a sample (edge parameters)        fp_* per fixed point
Pointer arrays (g_edge_ptr, s_graph, s_ep_ptr, s_fp_ptr) index into the group they name.
"""
import glob
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# ---------------------------------------------------------------------------
# 1. Constants
# ---------------------------------------------------------------------------
N_MAX = 10
NUM_STABILITY = 7              # stability codes 0..5 as stored; class 6 is reserved for unknown
UNKNOWN_STABILITY_CODE = 255
UNKNOWN_STABILITY_CLASS = 6
EPS_BOX = 0.05                 # same epsilon as the generator's invariant box
SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST = 1, 2, 3

SHARD_GLOB = "shard_[0-9][0-9][0-9][0-9].npz"
GROUP_ORDER = ("g", "e", "s", "ep", "fp")      # graphs, edges, samples, edge parameters, fixed points
POINTER_KEYS = {"g_edge_ptr": "e", "s_graph": "g", "s_ep_ptr": "ep", "s_fp_ptr": "fp"}
REPORT_KEYS = ("g_n", "s_graph", "s_n_fp", "s_cons", "s_phase", "fp_state")
DATASET_KEYS = [
    "g_n", "g_n_edges", "g_pos_fb", "g_split", "g_edge_ptr",
    "e_src", "e_tgt", "e_sgn",
    "s_graph", "s_P", "s_K", "s_xmax", "s_n_fp", "s_cons", "s_phase",
    "s_ep_ptr", "s_fp_ptr",
    "ep_T", "ep_nh", "ep_F",
    "fp_state", "fp_charpoly", "fp_stab",
]


# ---------------------------------------------------------------------------
# 2. Cache construction and verification
# ---------------------------------------------------------------------------
def report_cache(out, max_fp):
    n_fp = np.asarray(out["s_n_fp"])
    g_n = np.asarray(out["g_n"])[np.asarray(out["s_graph"])]
    cons = np.asarray(out["s_cons"])
    phase = np.asarray(out["s_phase"])
    print(f"graphs   : {out['g_n'].shape[0]}")
    print(f"samples  : {n_fp.shape[0]}")
    print(f"fp rows  : {out['fp_state'].shape[0]}")
    print(f"Brouwer-certified samples : {100.0 * cons.mean():.1f}%")
    print(f"samples with zero FPs     : {100.0 * (n_fp == 0).mean():.2f}%")
    print("per n:  samples | certified % | mean n_fp")
    for n in range(2, N_MAX + 1):
        m = g_n == n
        if m.any():
            print(f"  n={n:2d}: {int(m.sum()):9d} | {100.0 * cons[m].mean():6.1f} | {n_fp[m].mean():.2f}")
    pct = np.percentile(n_fp, [50, 90, 99, 99.9])
    print(f"n_fp percentiles 50/90/99/99.9 : {pct[0]:.0f} / {pct[1]:.0f} / {pct[2]:.0f} / {pct[3]:.0f}"
          f"   max {n_fp.max()}")
    print(f"suggested max_fp (p99 + 3)     : {int(np.ceil(pct[2])) + 3}")
    print(f"dropped at max_fp={max_fp}          : {100.0 * (n_fp > max_fp).mean():.2f}% of samples")
    for ph, name in enumerate(("broad", "switch", "boundary")):
        m = phase == ph
        if m.any():
            print(f"  phase {ph} ({name:8s}): {int(m.sum()):9d} samples | mean n_fp {n_fp[m].mean():.2f}"
                  f" | certified {100.0 * cons[m].mean():.1f}%")


def ode_residual_numpy(x, P, Kd, xmax, adj, T, nh, F):
    """Relative ODE residual f_i / (K_i * xmax_i) of K points x [K, N] for one graph.

    Written independently of model.py so that the cache check does not depend on the code
    that the training loss uses. Parameter arrays: P, Kd, xmax [N]; adj, T, nh, F [N, N]."""
    logr = nh[None] * (np.log(np.maximum(x, 1e-30))[:, :, None] - np.log(T)[None])
    zero = np.zeros_like(logr)
    log_h = np.logaddexp(np.log(F)[None] + logr, zero) - np.logaddexp(logr, zero)
    prod = np.exp((log_h * (adj != 0)[None]).sum(axis=1))
    return (P[None] * prod - Kd[None] * x) / (Kd * xmax)[None]


def check_structure(cache_dir):
    """Structural integrity of the merged arrays; raises RuntimeError on the first problem.

    Every pointer must land inside the group it names, and the per-sample blocks of edge
    parameters and of fixed points, as well as the per-graph blocks of edges, must tile their
    arrays exactly with no gap and no overlap. A wrong offset during the merge would otherwise
    show up only as silently misaligned training data."""
    a = open_cache(cache_dir)
    n_graph = a["g_n"].shape[0]
    s_graph = np.asarray(a["s_graph"]).astype(np.int64)
    if s_graph.min() < 0 or s_graph.max() >= n_graph:
        raise RuntimeError("s_graph points outside the graph array")

    def tiles(ptr, length, total, name):
        ptr = np.asarray(ptr).astype(np.int64)
        if ptr[0] != 0 or not np.array_equal(ptr[1:], (ptr + length)[:-1]) or ptr[-1] + length[-1] != total:
            raise RuntimeError(f"{name} blocks do not tile their array")

    n_edges = np.asarray(a["g_n_edges"]).astype(np.int64)
    tiles(a["g_edge_ptr"], n_edges, a["e_src"].shape[0], "edge")
    tiles(a["s_ep_ptr"], n_edges[s_graph], a["ep_T"].shape[0], "edge-parameter")
    tiles(a["s_fp_ptr"], np.asarray(a["s_n_fp"]).astype(np.int64), a["fp_state"].shape[0], "fixed-point")
    split = np.asarray(a["g_split"])
    for code in (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST):
        if not (split == code).any():
            raise RuntimeError(f"split {code} contains no graphs")
    if int(np.asarray(a["e_src"]).max()) >= N_MAX or int(np.asarray(a["e_tgt"]).max()) >= N_MAX:
        raise RuntimeError("edge endpoint outside N_MAX")
    print("structure: pointers tile every array, all three splits are populated")


def verify_cache(cache_dir, n_check=2000, seed=0):
    """Re-derive label quality from the merged cache for a random sample of samples.

    Reports the ODE residual of the stored fixed points under the stored parameters, how many
    points lie outside the invariant box, and how often the stored certification flag
    disagrees with the stored Brouwer indices. Raises if the labels are not fixed points."""
    a = open_cache(cache_dir)
    n_fp = np.asarray(a["s_n_fp"])
    cand = np.nonzero(n_fp >= 1)[0]
    pick = np.random.default_rng(seed).choice(cand, size=min(n_check, cand.size), replace=False)
    bidx = np.load(os.path.join(cache_dir, "fp_bidx.npy"), mmap_mode="r")
    worst, outside, total, flag_mismatch = [], 0, 0, 0
    for s in pick:
        k = int(n_fp[s])
        d = load_sample(a, int(s), k)
        n = d["n"]
        x = d["fp_state"][:k, :n].astype(np.float64)
        xmax = d["xmax"][:n].astype(np.float64)
        r = ode_residual_numpy(x, d["P"][:n].astype(np.float64), d["K"][:n].astype(np.float64), xmax,
                               d["adj"][:n, :n], d["T"][:n, :n].astype(np.float64),
                               d["nh"][:n, :n].astype(np.float64), d["F"][:n, :n].astype(np.float64))
        worst.append(np.abs(r).max(axis=1))
        outside += int((x > xmax * (1.0 + 1e-6)).any(axis=1).sum())
        total += k
        f0 = int(a["s_fp_ptr"][s])
        flag_mismatch += int((int(np.asarray(bidx[f0:f0 + k]).sum()) == 1) != bool(a["s_cons"][s]))
    w = np.concatenate(worst)
    print(f"verify: {pick.size} samples, {total} fixed points | |residual| median {np.median(w):.2e}"
          f"  p99 {np.percentile(w, 99):.2e}  max {w.max():.2e} | outside box {outside}"
          f" | certified flag differs from index sum in {flag_mismatch} samples (non-hyperbolic points)")
    if np.median(w) > 1e-4 or outside > 0:
        raise RuntimeError("cache labels are not fixed points of the stored parameters")


def build_cache(corpus_dir, cache_dir, max_fp):
    """Merge the generator's shards into one memory-mapped array per key.

    The final arrays are allocated at their exact total length and the shards are copied in
    one at a time, shifting every pointer array by the running offset of the group it points
    into, so peak memory is one shard regardless of corpus size."""
    shards = sorted(glob.glob(os.path.join(corpus_dir, "shards", SHARD_GLOB)))
    if not shards:
        raise FileNotFoundError(f"no shards under {corpus_dir}/shards")
    os.makedirs(cache_dir, exist_ok=True)
    sizes = []
    for path in shards:
        with np.load(path) as z:
            sizes.append(z["_sizes"])
    totals = {g: int(sum(s[j] for s in sizes)) for j, g in enumerate(GROUP_ORDER)}

    out = {}
    with np.load(shards[0]) as first:
        for k in first.files:
            if k == "_sizes":
                continue
            arr = first[k]
            shape = (totals[k.split("_")[0]],) + arr.shape[1:]
            out[k] = np.lib.format.open_memmap(
                os.path.join(cache_dir, k + ".npy"), mode="w+", dtype=arr.dtype, shape=shape)

    pos = {g: 0 for g in GROUP_ORDER}
    for i, path in enumerate(shards):
        with np.load(path) as z:
            for k in out:
                v = z[k]
                if k in POINTER_KEYS:
                    v = v + pos[POINTER_KEYS[k]]
                g = k.split("_")[0]
                out[k][pos[g]:pos[g] + v.shape[0]] = v
        for j, g in enumerate(GROUP_ORDER):
            pos[g] += int(sizes[i][j])
        if (i + 1) % 20 == 0 or i + 1 == len(shards):
            print(f"merged {i + 1}/{len(shards)} shards", flush=True)

    for v in out.values():
        v.flush()
    del out
    with open(os.path.join(cache_dir, "meta.json"), "w") as fh:
        json.dump({"n_shards": len(shards), **totals}, fh)
    report_cache({k: np.load(os.path.join(cache_dir, k + ".npy"), mmap_mode="r")
                  for k in REPORT_KEYS}, max_fp)
    check_structure(cache_dir)
    verify_cache(cache_dir)


# ---------------------------------------------------------------------------
# 3. Dataset
# ---------------------------------------------------------------------------
def open_cache(cache_dir):
    return {k: np.load(os.path.join(cache_dir, k + ".npy"), mmap_mode="r") for k in DATASET_KEYS}


def load_sample(a, s, max_fp):
    """One sample as padded numpy arrays. Requires n_fp(s) <= max_fp."""
    g = int(a["s_graph"][s])
    n = int(a["g_n"][g])
    ne = int(a["g_n_edges"][g])
    e0 = int(a["g_edge_ptr"][g])
    p0 = int(a["s_ep_ptr"][s])
    f0 = int(a["s_fp_ptr"][s])
    k = int(a["s_n_fp"][s])

    P = np.ones(N_MAX, np.float32)
    Kd = np.ones(N_MAX, np.float32)
    xmax = np.ones(N_MAX, np.float32)
    P[:n] = a["s_P"][s, :n]
    Kd[:n] = a["s_K"][s, :n]
    xmax[:n] = a["s_xmax"][s, :n]

    src = np.asarray(a["e_src"][e0:e0 + ne], dtype=np.int64)
    tgt = np.asarray(a["e_tgt"][e0:e0 + ne], dtype=np.int64)
    adj = np.zeros((N_MAX, N_MAX), np.float32)
    T = np.ones((N_MAX, N_MAX), np.float32)
    nh = np.ones((N_MAX, N_MAX), np.float32)
    fold = np.ones((N_MAX, N_MAX), np.float32)
    adj[src, tgt] = a["e_sgn"][e0:e0 + ne]
    T[src, tgt] = a["ep_T"][p0:p0 + ne]
    nh[src, tgt] = a["ep_nh"][p0:p0 + ne]
    fold[src, tgt] = a["ep_F"][p0:p0 + ne]

    fp_state = np.zeros((max_fp, N_MAX), np.float32)
    fp_cp = np.zeros((max_fp, N_MAX), np.float32)
    fp_state[:k, :n] = a["fp_state"][f0:f0 + k, :n]
    fp_cp[:k, :n] = np.nan_to_num(a["fp_charpoly"][f0:f0 + k, :n])
    fp_stab = np.full(max_fp, -1, np.int64)
    st = np.asarray(a["fp_stab"][f0:f0 + k], dtype=np.int64)
    st[st == UNKNOWN_STABILITY_CODE] = UNKNOWN_STABILITY_CLASS
    fp_stab[:k] = st

    return {
        "n": n, "k": k,
        "P": np.nan_to_num(P, nan=1.0), "K": np.nan_to_num(Kd, nan=1.0),
        "xmax": np.nan_to_num(xmax, nan=1.0),
        "adj": adj, "T": T, "nh": nh, "F": fold,
        "fp_state": fp_state, "fp_charpoly": fp_cp, "fp_stab": fp_stab,
        "has_pos_fb": bool(a["g_pos_fb"][g]),
        "consistent": bool(a["s_cons"][s]),
        "phase": int(a["s_phase"][s]),
    }


class SccDataset(Dataset):
    """One item = one parameterized SCC realization with all of its fixed points.

    Samples with no fixed points (a failed solve; the Brouwer degree guarantees at least one)
    and samples with more than max_fp fixed points are excluded."""

    def __init__(self, cache_dir, split, max_fp, max_samples=0, seed=0):
        self.cache_dir = cache_dir
        self.max_fp = max_fp
        a = open_cache(cache_dir)
        s_graph = np.asarray(a["s_graph"])
        n_fp = np.asarray(a["s_n_fp"])
        g_split = np.asarray(a["g_split"])
        keep = (g_split[s_graph] == split) & (n_fp >= 1) & (n_fp <= max_fp)
        index = np.nonzero(keep)[0]
        if max_samples > 0 and index.shape[0] > max_samples:
            rng = np.random.default_rng(seed)
            index = np.sort(rng.choice(index, size=max_samples, replace=False))
        self.index = index
        self.sample_n = np.asarray(a["g_n"])[s_graph[index]].astype(np.int64)
        self.sample_cons = np.asarray(a["s_cons"])[index]
        self.sample_nfp = n_fp[index].astype(np.int64)
        self._a = None   # opened lazily inside each worker; memmaps must not be pickled

    def __len__(self):
        return int(self.index.shape[0])

    def __getitem__(self, i):
        if self._a is None:
            self._a = open_cache(self.cache_dir)
        return load_sample(self._a, int(self.index[i]), self.max_fp)


def collate(batch):
    n = torch.tensor([b["n"] for b in batch], dtype=torch.long)
    k = torch.tensor([b["k"] for b in batch], dtype=torch.long)
    nb = int(n.max())
    kt = int(k.max())

    def stack(key):
        return torch.from_numpy(np.stack([b[key] for b in batch]))

    out = {
        "n": n,
        "k": k,
        "phase": torch.tensor([b["phase"] for b in batch], dtype=torch.long),
        "P": stack("P")[:, :nb].contiguous(),
        "K": stack("K")[:, :nb].contiguous(),
        "xmax": stack("xmax")[:, :nb].contiguous(),
        "adj": stack("adj")[:, :nb, :nb].contiguous(),
        "T": stack("T")[:, :nb, :nb].contiguous(),
        "nh": stack("nh")[:, :nb, :nb].contiguous(),
        "F": stack("F")[:, :nb, :nb].contiguous(),
        "fp_state": stack("fp_state")[:, :kt, :nb].contiguous(),
        "fp_charpoly": stack("fp_charpoly")[:, :kt, :nb].contiguous(),
        "fp_stab": stack("fp_stab")[:, :kt].contiguous(),
        "has_pos_fb": torch.tensor([b["has_pos_fb"] for b in batch], dtype=torch.bool),
        "consistent": torch.tensor([b["consistent"] for b in batch], dtype=torch.bool),
    }
    out["node_mask"] = torch.arange(nb).unsqueeze(0) < n.unsqueeze(1)
    out["fp_mask"] = torch.arange(kt).unsqueeze(0) < k.unsqueeze(1)
    return out


def make_loader(cfg, split, train, max_samples=None):
    """Training loader draws steps_per_epoch * batch_size samples per epoch with replacement,
    weighting each sample by count(n)^-sampler_power so small graphs are not drowned out."""
    if max_samples is None:
        max_samples = 0 if train else cfg.max_val_samples
    ds = SccDataset(cfg.cache_dir, split, cfg.max_fp, max_samples=max_samples, seed=cfg.seed)
    kwargs = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.num_workers > 0,
    )
    if cfg.num_workers > 0:
        kwargs["prefetch_factor"] = 4
    if train:
        counts = np.bincount(ds.sample_n)
        w = counts[ds.sample_n].astype(np.float64) ** (-cfg.sampler_power)
        sampler = WeightedRandomSampler(
            torch.from_numpy(w), num_samples=cfg.steps_per_epoch * cfg.batch_size, replacement=True)
        return ds, DataLoader(ds, sampler=sampler, drop_last=True, **kwargs)
    return ds, DataLoader(ds, shuffle=False, drop_last=False, **kwargs)


def describe_dataset(ds, name):
    print(f"[{name}] samples {len(ds)}  certified {100.0 * ds.sample_cons.mean():.1f}%"
          f"  mean n_fp {ds.sample_nfp.mean():.2f}")
    for n in range(2, N_MAX + 1):
        m = ds.sample_n == n
        if m.any():
            print(f"   n={n:2d}: {int(m.sum()):8d} samples | certified {100.0 * ds.sample_cons[m].mean():5.1f}%"
                  f" | mean n_fp {ds.sample_nfp[m].mean():.2f}")


# ---------------------------------------------------------------------------
# 4. Single-SCC batches
# ---------------------------------------------------------------------------
def scc_to_batch(n, edges, P, K, T, n_hill, F, device="cpu"):
    """Batch of one SCC from raw parameters, ready for SccFixedPointNet.forward.

    n      : number of genes
    edges  : list of (source, target) pairs, 0-based, no duplicates
    P, K   : length-n production and degradation rates
    T, n_hill, F : per-edge threshold, Hill coefficient, fold-change, in the order of `edges`
    The sign of an edge follows from F (F > 1 activation, F < 1 repression).
    xmax uses the product bound (1 + eps) * P / K * prod_j max(1, F_ji)."""
    if n > N_MAX:
        raise ValueError(f"n={n} exceeds N_MAX={N_MAX}")
    P = np.asarray(P, np.float32)
    K = np.asarray(K, np.float32)
    T = np.asarray(T, np.float32)
    n_hill = np.asarray(n_hill, np.float32)
    F = np.asarray(F, np.float32)
    src = np.array([e[0] for e in edges], dtype=np.int64)
    tgt = np.array([e[1] for e in edges], dtype=np.int64)
    if len({(int(s), int(t)) for s, t in zip(src, tgt)}) != len(edges):
        raise ValueError("duplicate edges")

    adj = np.zeros((n, n), np.float32)
    Tm = np.ones((n, n), np.float32)
    nhm = np.ones((n, n), np.float32)
    Fm = np.ones((n, n), np.float32)
    adj[src, tgt] = np.where(F > 1.0, 1.0, -1.0)
    Tm[src, tgt] = T
    nhm[src, tgt] = n_hill
    Fm[src, tgt] = F

    bound = np.where(adj != 0, np.maximum(1.0, Fm), 1.0).prod(axis=0)    # product over sources
    xmax = (1.0 + EPS_BOX) * P / K * bound

    def put(a):
        return torch.from_numpy(np.ascontiguousarray(a)).unsqueeze(0).to(device)

    return {
        "n": torch.tensor([n], dtype=torch.long, device=device),
        "P": put(P), "K": put(K), "xmax": put(xmax.astype(np.float32)),
        "adj": put(adj), "T": put(Tm), "nh": put(nhm), "F": put(Fm),
        "node_mask": torch.ones(1, n, dtype=torch.bool, device=device),
    }

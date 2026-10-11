"""Corpus B access layer — constant forced-response dataset.

One sample = one parameterized SCC with a forcing specification and all of its
fixed points under that forcing.

Cache layout (built by build_cache_b). Keys are grouped by prefix:
  g_*  per graph          e_*  per directed edge
  s_*  per sample         ep_* per edge-parameter row (one per edge per sample)
  fp_* per fixed point    cb_* per forcing port (variable count per sample)

Pointer arrays and the groups they index:
  g_edge_ptr -> e (graph start in edge table)
  s_graph    -> g (absolute graph index)
  s_ep_ptr   -> ep (sample start in edge-parameter table)
  s_fp_ptr   -> fp (sample start in fixed-point table)
  s_cb_ptr   -> cb (sample start in port table)

Requires shards generated with corpus_b_generator.py _sizes of length 6.
Shards with only 5 _sizes entries have the stratum-1/2 mislabelling bug and
must be regenerated.
"""
import glob
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from data import (EPS_BOX, N_MAX, NUM_STABILITY,
                  SHARD_GLOB, SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST,
                  UNKNOWN_STABILITY_CLASS, UNKNOWN_STABILITY_CODE)

# ---------------------------------------------------------------------------
# 1. Constants
# ---------------------------------------------------------------------------
MAX_PORTS = 3
SHARD_GLOB_B = "shard_b_[0-9][0-9][0-9][0-9].npz"

GROUP_ORDER_B = ("g", "e", "s", "ep", "fp", "cb")

POINTER_KEYS_B = {
    "g_edge_ptr": "e",
    "s_graph":    "g",
    "s_ep_ptr":   "ep",
    "s_fp_ptr":   "fp",
    "s_cb_ptr":   "cb",
}

DATASET_KEYS_B = [
    "g_n", "g_n_edges", "g_tc", "g_pos_fb", "g_neg_fb", "g_split", "g_edge_ptr",
    "e_src", "e_tgt", "e_sgn",
    "s_graph", "s_P", "s_K", "s_xmax", "s_n_fp", "s_cons", "s_bsum",
    "s_ep_ptr", "s_fp_ptr", "s_cb_ptr",
    "cb_n_ports", "cb_stratum", "cb_bif_type", "cb_bif_u", "cb_bif_crossed",
    "cb_port_gene", "cb_port_u", "cb_port_F", "cb_port_T", "cb_port_nh",
    "ep_T", "ep_nh", "ep_F",
    "fp_state", "fp_charpoly", "fp_stab",
]


def _group_of(key):
    """Return the GROUP_ORDER_B prefix of a shard key ('ep_T' -> 'ep', 'cb_port_u' -> 'cb')."""
    for g in sorted(GROUP_ORDER_B, key=len, reverse=True):
        if key.startswith(g + "_"):
            return g
    raise ValueError(f"Cannot determine group for key: {key!r}")


# ---------------------------------------------------------------------------
# 2. Cache construction
# ---------------------------------------------------------------------------
def build_cache_b(corpus_dir, cache_dir):
    """Merge Corpus B shard .npz files into flat memory-mapped numpy arrays."""
    shards = sorted(glob.glob(os.path.join(corpus_dir, "shards", SHARD_GLOB_B)))
    if not shards:
        raise FileNotFoundError(f"No shards under {corpus_dir}/shards")
    os.makedirs(cache_dir, exist_ok=True)

    sizes = []
    for path in shards:
        with np.load(path) as z:
            s = z["_sizes"]
            if s.shape[0] != len(GROUP_ORDER_B):
                raise ValueError(
                    f"{path}: _sizes has {s.shape[0]} entries, expected "
                    f"{len(GROUP_ORDER_B)} (groups {GROUP_ORDER_B}).  "
                    "Regenerate shards with the updated corpus_b_generator.py.")
            sizes.append(s)

    totals = {g: int(sum(s[j] for s in sizes)) for j, g in enumerate(GROUP_ORDER_B)}

    out = {}
    with np.load(shards[0]) as first:
        for k in first.files:
            if k == "_sizes":
                continue
            arr = first[k]
            g = _group_of(k)
            shape = (totals[g],) + arr.shape[1:]
            out[k] = np.lib.format.open_memmap(
                os.path.join(cache_dir, k + ".npy"),
                mode="w+", dtype=arr.dtype, shape=shape)

    pos = {g: 0 for g in GROUP_ORDER_B}
    for i, path in enumerate(shards):
        with np.load(path) as z:
            for k in out:
                v = z[k]
                if k in POINTER_KEYS_B:
                    v = v + pos[POINTER_KEYS_B[k]]
                g = _group_of(k)
                out[k][pos[g]:pos[g] + v.shape[0]] = v
        for j, g in enumerate(GROUP_ORDER_B):
            pos[g] += int(sizes[i][j])
        if (i + 1) % 20 == 0 or i + 1 == len(shards):
            print(f"merged {i + 1}/{len(shards)} shards", flush=True)

    for v in out.values():
        v.flush()
    del out
    with open(os.path.join(cache_dir, "meta.json"), "w") as fh:
        json.dump({"n_shards": len(shards), **totals}, fh)
    _report_cache_b({k: np.load(os.path.join(cache_dir, k + ".npy"), mmap_mode="r")
                     for k in ["g_n", "s_graph", "s_n_fp", "s_cons", "s_cb_ptr", "cb_stratum"]})


def _report_cache_b(a):
    n_fp    = np.asarray(a["s_n_fp"])
    g_n     = np.asarray(a["g_n"])[np.asarray(a["s_graph"])]
    cons    = np.asarray(a["s_cons"])
    # cb_stratum is per-port; pick the first port of each sample via s_cb_ptr
    stratum = np.asarray(a["cb_stratum"])[np.asarray(a["s_cb_ptr"])]
    print(f"graphs  : {a['g_n'].shape[0]}")
    print(f"samples : {n_fp.shape[0]}")
    print(f"Brouwer-certified: {100.0 * cons.mean():.1f}%")
    strat_names = {0: "broad", 1: "near-bif", 2: "opp-regime"}
    for st, nm in strat_names.items():
        m = stratum == st
        if m.any():
            print(f"  stratum {st} ({nm:11s}): {int(m.sum()):9d} samples"
                  f" | certified {100.0 * cons[m].mean():.1f}%"
                  f" | mean n_fp {n_fp[m].mean():.2f}")
    for n in range(2, N_MAX + 1):
        m = g_n == n
        if m.any():
            print(f"  n={n:2d}: {int(m.sum()):9d} | certified {100.0*cons[m].mean():.1f}%"
                  f" | mean n_fp {n_fp[m].mean():.2f}")


# ---------------------------------------------------------------------------
# 3. Dataset
# ---------------------------------------------------------------------------
def open_cache_b(cache_dir):
    return {k: np.load(os.path.join(cache_dir, k + ".npy"), mmap_mode="r")
            for k in DATASET_KEYS_B}


def load_sample_b(a, s, max_fp, max_ports=MAX_PORTS):
    """One Corpus B sample as padded numpy arrays. Requires n_fp(s) <= max_fp."""
    g   = int(a["s_graph"][s])
    n   = int(a["g_n"][g])
    ne  = int(a["g_n_edges"][g])
    e0  = int(a["g_edge_ptr"][g])
    p0  = int(a["s_ep_ptr"][s])
    f0  = int(a["s_fp_ptr"][s])
    c0  = int(a["s_cb_ptr"][s])
    k   = int(a["s_n_fp"][s])
    np_ = int(a["cb_n_ports"][s])

    P    = np.ones(N_MAX, np.float32);  P[:n]    = a["s_P"][s, :n]
    Kd   = np.ones(N_MAX, np.float32);  Kd[:n]   = a["s_K"][s, :n]
    xmax = np.ones(N_MAX, np.float32);  xmax[:n] = a["s_xmax"][s, :n]

    src = np.asarray(a["e_src"][e0:e0 + ne], dtype=np.int64)
    tgt = np.asarray(a["e_tgt"][e0:e0 + ne], dtype=np.int64)
    adj  = np.zeros((N_MAX, N_MAX), np.float32)
    T    = np.ones((N_MAX, N_MAX), np.float32)
    nh   = np.ones((N_MAX, N_MAX), np.float32)
    fold = np.ones((N_MAX, N_MAX), np.float32)
    adj[src, tgt]  = a["e_sgn"][e0:e0 + ne]
    T[src, tgt]    = a["ep_T"][p0:p0 + ne]
    nh[src, tgt]   = a["ep_nh"][p0:p0 + ne].astype(np.float32)  # stored uint8
    fold[src, tgt] = a["ep_F"][p0:p0 + ne]

    fp_state = np.zeros((max_fp, N_MAX), np.float32)
    fp_cp    = np.zeros((max_fp, N_MAX), np.float32)
    fp_stab  = np.full(max_fp, -1, np.int64)
    if k > 0:
        fp_state[:k, :n] = a["fp_state"][f0:f0 + k, :n]
        fp_cp[:k, :n]    = np.nan_to_num(a["fp_charpoly"][f0:f0 + k, :n])
        st = np.asarray(a["fp_stab"][f0:f0 + k], dtype=np.int64)
        st[st == UNKNOWN_STABILITY_CODE] = UNKNOWN_STABILITY_CLASS
        fp_stab[:k] = st

    np_use   = min(np_, max_ports)
    port_gene = np.full(max_ports, -1, np.int64)
    port_u    = np.zeros(max_ports, np.float32)
    port_F    = np.ones(max_ports, np.float32)
    port_T    = np.ones(max_ports, np.float32)
    port_nh   = np.ones(max_ports, np.float32)
    port_gene[:np_use] = a["cb_port_gene"][c0:c0 + np_use].astype(np.int64)
    port_u[:np_use]    = a["cb_port_u"][c0:c0 + np_use]
    port_F[:np_use]    = a["cb_port_F"][c0:c0 + np_use]
    port_T[:np_use]    = a["cb_port_T"][c0:c0 + np_use]
    port_nh[:np_use]   = a["cb_port_nh"][c0:c0 + np_use].astype(np.float32)

    return {
        "n": n, "k": k, "n_ports": np_,
        "P":    np.nan_to_num(P,    nan=1.0),
        "K":    np.nan_to_num(Kd,   nan=1.0),
        "xmax": np.nan_to_num(xmax, nan=1.0),
        "adj": adj, "T": T, "nh": nh, "F": fold,
        "fp_state": fp_state, "fp_charpoly": fp_cp, "fp_stab": fp_stab,
        "port_gene": port_gene, "port_u": port_u,
        "port_F": port_F, "port_T": port_T, "port_nh": port_nh,
        "has_pos_fb":  bool(a["g_pos_fb"][g]),
        "consistent":  bool(a["s_cons"][s]),
        "stratum":     int(a["cb_stratum"][s]),
        "bif_type":    int(a["cb_bif_type"][s]),
        "bif_crossed": bool(a["cb_bif_crossed"][s]),
        "tc":          int(a["g_tc"][g]),
    }


class CorpusBDataset(Dataset):
    """One item = one forced SCC realization with all of its fixed points.

    Samples with no fixed points (failed solve) and samples with more than
    max_fp fixed points are excluded."""

    def __init__(self, cache_dir, split, max_fp, max_ports=MAX_PORTS,
                 max_samples=0, seed=0):
        self.cache_dir = cache_dir
        self.max_fp    = max_fp
        self.max_ports = max_ports
        a = open_cache_b(cache_dir)
        s_graph = np.asarray(a["s_graph"])
        n_fp    = np.asarray(a["s_n_fp"])
        g_split = np.asarray(a["g_split"])
        keep  = (g_split[s_graph] == split) & (n_fp >= 1) & (n_fp <= max_fp)
        index = np.nonzero(keep)[0]
        if max_samples > 0 and index.shape[0] > max_samples:
            rng   = np.random.default_rng(seed)
            index = np.sort(rng.choice(index, size=max_samples, replace=False))
        self.index        = index
        self.sample_n     = np.asarray(a["g_n"])[s_graph[index]].astype(np.int64)
        self.sample_cons  = np.asarray(a["s_cons"])[index]
        self.sample_nfp   = n_fp[index].astype(np.int64)
        self.sample_strat = np.asarray(a["cb_stratum"])[index]
        self._a = None

    def __len__(self):
        return int(self.index.shape[0])

    def __getitem__(self, i):
        if self._a is None:
            self._a = open_cache_b(self.cache_dir)
        return load_sample_b(self._a, int(self.index[i]), self.max_fp, self.max_ports)


def collate_b(batch):
    n   = torch.tensor([b["n"]       for b in batch], dtype=torch.long)
    k   = torch.tensor([b["k"]       for b in batch], dtype=torch.long)
    np_ = torch.tensor([b["n_ports"] for b in batch], dtype=torch.long)
    nb  = int(n.max())
    kt  = int(k.max())

    def stack(key):
        return torch.from_numpy(np.stack([b[key] for b in batch]))

    out = {
        "n": n, "k": k, "n_ports": np_,
        "stratum":     torch.tensor([b["stratum"]     for b in batch], dtype=torch.long),
        "bif_type":    torch.tensor([b["bif_type"]    for b in batch], dtype=torch.long),
        "bif_crossed": torch.tensor([b["bif_crossed"] for b in batch], dtype=torch.bool),
        "tc":          torch.tensor([b["tc"]          for b in batch], dtype=torch.long),
        "P":    stack("P")[:, :nb].contiguous(),
        "K":    stack("K")[:, :nb].contiguous(),
        "xmax": stack("xmax")[:, :nb].contiguous(),
        "adj":  stack("adj")[:, :nb, :nb].contiguous(),
        "T":    stack("T")[:, :nb, :nb].contiguous(),
        "nh":   stack("nh")[:, :nb, :nb].contiguous(),
        "F":    stack("F")[:, :nb, :nb].contiguous(),
        "fp_state":    stack("fp_state")[:, :kt, :nb].contiguous(),
        "fp_charpoly": stack("fp_charpoly")[:, :kt, :nb].contiguous(),
        "fp_stab":     stack("fp_stab")[:, :kt].contiguous(),
        "port_gene": stack("port_gene").contiguous(),
        "port_u":    stack("port_u").contiguous(),
        "port_F":    stack("port_F").contiguous(),
        "port_T":    stack("port_T").contiguous(),
        "port_nh":   stack("port_nh").contiguous(),
        "has_pos_fb": torch.tensor([b["has_pos_fb"] for b in batch], dtype=torch.bool),
        "consistent": torch.tensor([b["consistent"] for b in batch], dtype=torch.bool),
    }
    out["node_mask"] = torch.arange(nb).unsqueeze(0) < n.unsqueeze(1)
    out["fp_mask"]   = torch.arange(kt).unsqueeze(0) < k.unsqueeze(1)
    return out


def make_loader_b(cfg, split, train, max_samples=None):
    """Training loader with size-stratified sampling, matching make_loader in data.py."""
    if max_samples is None:
        max_samples = 0 if train else cfg.max_val_samples
    ds = CorpusBDataset(cfg.cache_dir_b, split, cfg.max_fp,
                        max_samples=max_samples, seed=cfg.seed)
    kwargs = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        collate_fn=collate_b,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.num_workers > 0,
    )
    if cfg.num_workers > 0:
        kwargs["prefetch_factor"] = 4
    if train:
        counts = np.bincount(ds.sample_n)
        w = counts[ds.sample_n].astype(np.float64) ** (-cfg.sampler_power)
        sampler = WeightedRandomSampler(
            torch.from_numpy(w),
            num_samples=cfg.steps_per_epoch * cfg.batch_size,
            replacement=True)
        return ds, DataLoader(ds, sampler=sampler, drop_last=True, **kwargs)
    return ds, DataLoader(ds, shuffle=False, drop_last=False, **kwargs)


def describe_dataset_b(ds, name):
    print(f"[{name}] samples {len(ds)}  certified {100.0 * ds.sample_cons.mean():.1f}%"
          f"  mean n_fp {ds.sample_nfp.mean():.2f}")
    for n in range(2, N_MAX + 1):
        m = ds.sample_n == n
        if m.any():
            print(f"   n={n:2d}: {int(m.sum()):8d} samples | certified"
                  f" {100.0 * ds.sample_cons[m].mean():5.1f}%"
                  f" | mean n_fp {ds.sample_nfp[m].mean():.2f}")
    strat_names = {0: "broad", 1: "near-bif", 2: "opp-regime"}
    for st, nm in strat_names.items():
        m = ds.sample_strat == st
        if m.any():
            print(f"   stratum {st} ({nm:11s}): {int(m.sum()):8d} samples"
                  f" | certified {100.0 * ds.sample_cons[m].mean():5.1f}%"
                  f" | mean n_fp {ds.sample_nfp[m].mean():.2f}")

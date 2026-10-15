"""Corpus B access layer: single-SCC constant forced response.

Sections
  1. Constants and shard keys
  2. Cache construction (merge Corpus B shards)
  3. CorpusBDataset, collate_b, make_loader_b

Shard key layout (written by CorpusBShardBuilder.finalize):
  g_*        per graph      (same topology keys as Corpus A)
  e_*        per edge of a graph
  s_*        per sample     (includes s_graph, s_P, s_K, s_xmax, s_n_fp, ...)
  ep_*       per edge of a sample (forced Hill parameters)
  fp_*       per fixed point
  cb_*       Corpus-B-specific metadata and port arrays
  s_cb_ptr   pointer into the cb_port_* arrays (variable length per sample)

Tensor conventions
  Same as data.py for the graph tensors.
  Additionally, per sample:
    port_gene   [L]     index of the forced gene (0-based, in [0, N))
    port_u      [L]     forcing level (Hill function output)
    port_F      [L]     fold-change of the synthetic forcing edge
    port_T      [L]     threshold of the synthetic forcing edge
    port_nh     [L]     Hill coefficient of the synthetic forcing edge
    P_forced    [N]     production rates after absorbing all port forcings
  where L = cb_n_ports[s] <= MAX_PORTS.
"""
import glob
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from data import (N_MAX, NUM_STABILITY, UNKNOWN_STABILITY_CLASS,
                  UNKNOWN_STABILITY_CODE, SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST)

# ---------------------------------------------------------------------------
# 1. Constants
# ---------------------------------------------------------------------------
MAX_PORTS = 8          # cap on the number of simultaneous forced genes per sample
SHARD_GLOB_B = "shard_b_[0-9][0-9][0-9][0-9].npz"   # Corpus B shard naming convention

# All keys that appear in every Corpus B shard's finalize() output
# (minus the _sizes sentinel and the pointer arrays, which are derived here)
CORPUS_B_GRAPH_KEYS = [
    "g_n", "g_n_edges", "g_split", "g_edge_ptr",
    "e_src", "e_tgt", "e_sgn",
]
CORPUS_B_SAMPLE_KEYS = [
    "s_graph", "s_P", "s_K", "s_xmax", "s_n_fp", "s_cons", "s_bsum", "s_phase",
    "s_ep_ptr", "s_fp_ptr", "s_cb_ptr",
    "cb_n_ports", "cb_stratum", "cb_bif_type", "cb_bif_crossed", "cb_auto_nfp",
    "ep_T", "ep_nh", "ep_F",
    "fp_state", "fp_charpoly", "fp_stab",
    "cb_port_gene", "cb_port_u", "cb_port_F", "cb_port_T", "cb_port_nh",
]
CORPUS_B_ALL_KEYS = CORPUS_B_GRAPH_KEYS + CORPUS_B_SAMPLE_KEYS

# Groups for the _sizes sentinel: [graphs, edges, samples, edge-params, fixed-points]
_GROUP_ORDER = ("g", "e", "s", "ep", "fp")
_POINTER_KEYS = {
    "g_edge_ptr": "e",
    "s_ep_ptr":   "ep",
    "s_fp_ptr":   "fp",
    "s_cb_ptr":   "cb",   # cb_port_* arrays are a 5th group
}


# ---------------------------------------------------------------------------
# 2. Cache construction
# ---------------------------------------------------------------------------
def build_cache_b(corpus_b_dir, cache_dir):
    """Merge Corpus B shards into flat memory-mapped arrays."""
    shards = sorted(glob.glob(os.path.join(corpus_b_dir, "shards", SHARD_GLOB_B)))
    if not shards:
        raise FileNotFoundError(f"no shards under {corpus_b_dir}/shards")
    os.makedirs(cache_dir, exist_ok=True)

    # Collect sizes: _sizes = [n_graphs, n_edges, n_samples, n_ep, n_fp]
    sizes = []
    for path in shards:
        with np.load(path, allow_pickle=False) as z:
            sizes.append(z["_sizes"].tolist())

    # We also need the total length of cb_port_* arrays — not in _sizes.
    # Derive it from cb_n_ports across all shards.
    n_cb_total = 0
    for path in shards:
        with np.load(path, allow_pickle=False) as z:
            if "cb_n_ports" in z.files:
                n_cb_total += int(z["cb_n_ports"].sum())

    totals = {g: int(sum(s[j] for s in sizes)) for j, g in enumerate(_GROUP_ORDER)}
    totals["cb"] = n_cb_total

    # Allocate memmapped arrays for every key we care about
    with np.load(shards[0], allow_pickle=False) as first:
        all_keys = [k for k in first.files if k != "_sizes"]

    out = {}
    with np.load(shards[0], allow_pickle=False) as first:
        for k in all_keys:
            arr = first[k]
            prefix = k.split("_")[0]
            if prefix == "cb" and k not in ("cb_port_gene", "cb_port_u",
                                             "cb_port_F", "cb_port_T", "cb_port_nh"):
                n = totals["s"]   # per-sample cb_* metadata
            elif k in ("cb_port_gene", "cb_port_u",
                       "cb_port_F", "cb_port_T", "cb_port_nh"):
                n = totals["cb"]
            elif prefix in totals:
                n = totals[prefix]
            else:
                # fallback for s_cb_ptr (prefix "s" but size is n_samples)
                n = totals.get(prefix, arr.shape[0])
            shape = (n,) + arr.shape[1:]
            out[k] = np.lib.format.open_memmap(
                os.path.join(cache_dir, k + ".npy"),
                mode="w+", dtype=arr.dtype, shape=shape)

    pos = {g: 0 for g in list(_GROUP_ORDER) + ["cb"]}
    for i, path in enumerate(shards):
        with np.load(path, allow_pickle=False) as z:
            s = sizes[i]
            n_g, n_e, n_s, n_ep, n_fp = s[:5]
            n_cb_shard = int(z["cb_n_ports"].sum()) if "cb_n_ports" in z.files else 0
            for k in out:
                if k not in z.files:
                    continue
                v = np.asarray(z[k])
                # Shift pointer arrays
                if k == "g_edge_ptr":
                    v = v + pos["e"]
                elif k == "s_ep_ptr":
                    v = v + pos["ep"]
                elif k == "s_fp_ptr":
                    v = v + pos["fp"]
                elif k == "s_cb_ptr":
                    v = v + pos["cb"]
                elif k == "s_graph":
                    v = v + pos["g"]
                prefix = k.split("_")[0]
                if k in ("cb_port_gene", "cb_port_u",
                         "cb_port_F", "cb_port_T", "cb_port_nh"):
                    p = pos["cb"]
                    out[k][p:p + v.shape[0]] = v
                elif prefix == "cb":
                    p = pos["s"]
                    out[k][p:p + v.shape[0]] = v
                elif prefix in pos:
                    p = pos[prefix]
                    out[k][p:p + v.shape[0]] = v
        pos["g"] += n_g
        pos["e"] += n_e
        pos["s"] += n_s
        pos["ep"] += n_ep
        pos["fp"] += n_fp
        pos["cb"] += n_cb_shard
        if (i + 1) % 20 == 0 or i + 1 == len(shards):
            print(f"[corpus_b] merged {i + 1}/{len(shards)} shards", flush=True)

    for v in out.values():
        v.flush()
    del out
    with open(os.path.join(cache_dir, "meta_b.json"), "w") as fh:
        json.dump({"n_shards": len(shards), **totals}, fh)
    print(f"[corpus_b] cache done: {totals['s']} samples, "
          f"{totals['fp']} fixed points, {totals['cb']} port entries")


def open_cache_b(cache_dir):
    """Open all Corpus B arrays as memory-mapped read-only."""
    out = {}
    for path in glob.glob(os.path.join(cache_dir, "*.npy")):
        key = os.path.basename(path)[:-4]  # strip .npy
        out[key] = np.load(path, mmap_mode="r")
    return out


# ---------------------------------------------------------------------------
# 3. Dataset
# ---------------------------------------------------------------------------
def load_sample_b(a, s, max_fp):
    """One Corpus B sample as padded numpy arrays.

    Returns the same graph tensors as data.load_sample, plus:
      P_forced [N]        production rates after absorbing port forcings
      port_gene [L]       forced gene indices (0-based)
      port_u    [L]       forcing levels
      port_F/T/nh [L]     synthetic Hill edge parameters
      n_ports             number of active ports (L)
    where L = cb_n_ports[s].
    """
    g = int(a["s_graph"][s])
    n = int(a["g_n"][g])
    ne = int(a["g_n_edges"][g])
    e0 = int(a["g_edge_ptr"][g])
    p0 = int(a["s_ep_ptr"][s])
    f0 = int(a["s_fp_ptr"][s])
    c0 = int(a["s_cb_ptr"][s])
    k  = int(a["s_n_fp"][s])
    L  = int(a["cb_n_ports"][s])

    # --- kinetic parameters (already forced by the generator's apply_forcing) ---
    P      = np.ones(N_MAX, np.float32)
    Kd     = np.ones(N_MAX, np.float32)
    xmax   = np.ones(N_MAX, np.float32)
    P[:n]  = a["s_P"][s, :n]
    Kd[:n] = a["s_K"][s, :n]
    xmax[:n] = a["s_xmax"][s, :n]

    # --- graph topology ---
    src = np.asarray(a["e_src"][e0:e0 + ne], dtype=np.int64)
    tgt = np.asarray(a["e_tgt"][e0:e0 + ne], dtype=np.int64)
    adj  = np.zeros((N_MAX, N_MAX), np.float32)
    T    = np.ones((N_MAX, N_MAX), np.float32)
    nh   = np.ones((N_MAX, N_MAX), np.float32)
    fold = np.ones((N_MAX, N_MAX), np.float32)
    adj[src, tgt]  = a["e_sgn"][e0:e0 + ne]
    T[src, tgt]    = a["ep_T"][p0:p0 + ne]
    nh[src, tgt]   = a["ep_nh"][p0:p0 + ne]
    fold[src, tgt] = a["ep_F"][p0:p0 + ne]

    # --- fixed points ---
    fp_state  = np.zeros((max_fp, N_MAX), np.float32)
    fp_cp     = np.zeros((max_fp, N_MAX), np.float32)
    fp_stab   = np.full(max_fp, -1, np.int64)
    fp_state[:k, :n] = a["fp_state"][f0:f0 + k, :n]
    fp_cp[:k, :n]    = np.nan_to_num(a["fp_charpoly"][f0:f0 + k, :n])
    st = np.asarray(a["fp_stab"][f0:f0 + k], dtype=np.int64)
    st[st == UNKNOWN_STABILITY_CODE] = UNKNOWN_STABILITY_CLASS
    fp_stab[:k] = st

    # --- port forcing ---
    L_clip = min(L, MAX_PORTS)
    port_gene = np.zeros(MAX_PORTS, np.int64)
    port_u    = np.ones(MAX_PORTS,  np.float32)
    port_F    = np.ones(MAX_PORTS,  np.float32)
    port_T    = np.ones(MAX_PORTS,  np.float32)
    port_nh   = np.ones(MAX_PORTS,  np.float32)
    port_gene[:L_clip] = a["cb_port_gene"][c0:c0 + L_clip]
    port_u[:L_clip]    = a["cb_port_u"][c0:c0 + L_clip]
    port_F[:L_clip]    = a["cb_port_F"][c0:c0 + L_clip]
    port_T[:L_clip]    = a["cb_port_T"][c0:c0 + L_clip]
    port_nh[:L_clip]   = a["cb_port_nh"][c0:c0 + L_clip]

    return {
        "n": n, "k": k, "n_ports": L_clip,
        "P": np.nan_to_num(P, nan=1.0),
        "K": np.nan_to_num(Kd, nan=1.0),
        "xmax": np.nan_to_num(xmax, nan=1.0),
        "adj": adj, "T": T, "nh": nh, "F": fold,
        "fp_state": fp_state, "fp_charpoly": fp_cp, "fp_stab": fp_stab,
        "port_gene": port_gene,
        "port_u":    port_u,
        "port_F":    port_F,
        "port_T":    port_T,
        "port_nh":   port_nh,
        "consistent":  bool(a["s_cons"][s]),
        "bif_crossed": bool(a["cb_bif_crossed"][s]),
        "stratum":     int(a["cb_stratum"][s]),
        "has_pos_fb":  bool(a["g_pos_fb"][g]),
    }


class CorpusBDataset(Dataset):
    """One item = one Corpus B sample (forced SCC realization).

    Filtered to samples where 1 <= n_fp <= max_fp.
    """

    def __init__(self, cache_dir, split, max_fp, max_samples=0, seed=0):
        self.cache_dir = cache_dir
        self.max_fp    = max_fp
        a = open_cache_b(cache_dir)
        s_graph = np.asarray(a["s_graph"])
        n_fp    = np.asarray(a["s_n_fp"])
        g_split = np.asarray(a["g_split"])
        keep    = (g_split[s_graph] == split) & (n_fp >= 1) & (n_fp <= max_fp)
        index   = np.nonzero(keep)[0]
        if max_samples > 0 and index.shape[0] > max_samples:
            rng   = np.random.default_rng(seed)
            index = np.sort(rng.choice(index, size=max_samples, replace=False))
        self.index       = index
        self.sample_n    = np.asarray(a["g_n"])[s_graph[index]].astype(np.int64)
        self.sample_cons = np.asarray(a["s_cons"])[index]
        self.sample_nfp  = n_fp[index].astype(np.int64)
        self._a          = None   # lazy open inside each worker

    def __len__(self):
        return int(self.index.shape[0])

    def __getitem__(self, i):
        if self._a is None:
            self._a = open_cache_b(self.cache_dir)
        return load_sample_b(self._a, int(self.index[i]), self.max_fp)


def collate_b(batch):
    """Collate a list of Corpus B samples into a batch dict.

    All graph tensors are trimmed to the max active nodes / fps in the batch.
    Port tensors are trimmed to the max active ports.
    """
    n = torch.tensor([b["n"] for b in batch], dtype=torch.long)
    k = torch.tensor([b["k"] for b in batch], dtype=torch.long)
    l = torch.tensor([b["n_ports"] for b in batch], dtype=torch.long)
    nb = int(n.max())
    kt = int(k.max())
    lt = int(l.max())

    def stack(key):
        return torch.from_numpy(np.stack([b[key] for b in batch]))

    out = {
        "n":       n,
        "k":       k,
        "n_ports": l,
        # Graph / kinetic tensors (forced P already absorbed by generator)
        "P":    stack("P")[:, :nb].contiguous(),
        "K":    stack("K")[:, :nb].contiguous(),
        "xmax": stack("xmax")[:, :nb].contiguous(),
        "adj":  stack("adj")[:, :nb, :nb].contiguous(),
        "T":    stack("T")[:, :nb, :nb].contiguous(),
        "nh":   stack("nh")[:, :nb, :nb].contiguous(),
        "F":    stack("F")[:, :nb, :nb].contiguous(),
        # Fixed-point labels
        "fp_state":   stack("fp_state")[:, :kt, :nb].contiguous(),
        "fp_charpoly":stack("fp_charpoly")[:, :kt, :nb].contiguous(),
        "fp_stab":    stack("fp_stab")[:, :kt].contiguous(),
        # Port forcing
        "port_gene": stack("port_gene")[:, :lt].contiguous(),
        "port_u":    stack("port_u")[:, :lt].contiguous(),
        "port_F":    stack("port_F")[:, :lt].contiguous(),
        "port_T":    stack("port_T")[:, :lt].contiguous(),
        "port_nh":   stack("port_nh")[:, :lt].contiguous(),
        # Metadata
        "consistent":  torch.tensor([b["consistent"]  for b in batch], dtype=torch.bool),
        "bif_crossed": torch.tensor([b["bif_crossed"] for b in batch], dtype=torch.bool),
        "stratum":     torch.tensor([b["stratum"]     for b in batch], dtype=torch.long),
        "has_pos_fb":  torch.tensor([b["has_pos_fb"]  for b in batch], dtype=torch.bool),
    }
    out["node_mask"] = torch.arange(nb).unsqueeze(0) < n.unsqueeze(1)
    out["fp_mask"]   = torch.arange(kt).unsqueeze(0) < k.unsqueeze(1)
    out["port_mask"] = torch.arange(lt).unsqueeze(0) < l.unsqueeze(1)
    return out


def make_loader_b(cfg, split, train, max_samples=None):
    """Build a DataLoader for Corpus B.

    cfg must expose: cache_dir_b, max_fp, max_val_samples, batch_size,
    num_workers, steps_per_epoch, sampler_power, seed.
    """
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

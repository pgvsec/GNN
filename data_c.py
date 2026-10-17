"""Corpus C access layer — DAG-structured multi-SCC dataset.

One sample = one branch of one SCC (an SCC under a specific choice of stable
parent fixed points). This is the natural training unit: given the intra-SCC
topology and the forcing signal from parent SCCs, predict the fixed-point
portrait of this SCC.

Cache layout (built by build_cache_c). Keys are grouped by prefix:
  dag_*    per DAG            scc_*  per SCC (across all DAGs)
  iedge_*  per inter-SCC edge branch_* per branch
  link_*   per branch-parent link    bfp_*  per branch fixed point

Pointer/index arrays that are shifted during cache merge:
  dag_scc_ptr     : [n_dags+1]  CSR sentinels into scc
  dag_edge_ptr    : [n_dags+1]  CSR sentinels into iedge
  scc_branch_ptr  : [n_sccs+1]  CSR sentinels into branch
  branch_fp_ptr   : [n_branches+1]  CSR sentinels into bfp
  branch_link_ptr : [n_branches+1]  CSR sentinels into link
  link_parent_branch : [n_links]    absolute branch index
  iedge_src_scc, iedge_tgt_scc : [n_iedges]  absolute scc index
  branch_scc    : [n_branches]  absolute scc index
  scc_dag_id    : [n_sccs]      absolute dag index
"""
import glob
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from data import (EPS_BOX, N_MAX, NUM_STABILITY, SHARD_GLOB,
                  UNKNOWN_STABILITY_CLASS, UNKNOWN_STABILITY_CODE)

# ---------------------------------------------------------------------------
# 1. Constants
# ---------------------------------------------------------------------------
GROUP_ORDER_C = ("dag", "scc", "iedge", "branch", "link", "bfp")

# CSR sentinel pointer arrays: shape is (source_group_size + 1,).
# ptr[i]..ptr[i+1] are the target-group entries for source row i.
# Last element equals total items in target group.
# SENTINEL_PTRS_C maps ptr_name -> SOURCE group (determines array length).
# SENTINEL_TGTS_C maps ptr_name -> TARGET group (determines pointer offset).
SENTINEL_PTRS_C = {
    "dag_scc_ptr":     "dag",
    "dag_edge_ptr":    "dag",
    "scc_branch_ptr":  "scc",
    "branch_fp_ptr":   "branch",
    "branch_link_ptr": "branch",
}
SENTINEL_TGTS_C = {
    "dag_scc_ptr":     "scc",
    "dag_edge_ptr":    "iedge",
    "scc_branch_ptr":  "branch",
    "branch_fp_ptr":   "bfp",
    "branch_link_ptr": "link",
}

# Plain index arrays whose values are absolute indices into another group.
ABS_INDEX_C = {
    "link_parent_branch": "branch",
    "iedge_src_scc":      "scc",
    "iedge_tgt_scc":      "scc",
    "branch_scc":         "scc",
    "scc_dag_id":         "dag",
}

# All arrays read by the dataset (excludes schema_version scalar and _sizes).
DATASET_KEYS_C = [
    # DAG level
    "dag_n_sccs", "dag_n_edges", "dag_family", "dag_shape", "dag_depth",
    "dag_max_fanin", "dag_n_reconv", "dag_n_roots", "dag_all_solved",
    "dag_scc_ptr", "dag_edge_ptr",
    # SCC level
    "scc_n", "scc_tc", "scc_dag_id", "scc_topo_rank", "scc_depth",
    "scc_n_parents", "scc_truncated", "scc_branch_ptr",
    "scc_sign", "scc_P", "scc_K", "scc_ep_T", "scc_ep_nh", "scc_ep_F",
    # Inter-SCC edges
    "iedge_src_scc", "iedge_src_gene", "iedge_tgt_scc", "iedge_tgt_gene",
    "iedge_T", "iedge_nh", "iedge_F",
    # Branch level
    "branch_scc", "branch_key", "branch_cons", "branch_bsum", "branch_u",
    "branch_fp_ptr", "branch_link_ptr",
    # Links
    "link_parent_branch", "link_parent_fp",
    # Branch fixed points
    "bfp_x", "bfp_charpoly", "bfp_stab", "bfp_hopf", "bfp_zeroeig", "bfp_resid",
]


def _group_of_c(key):
    """Group prefix of a Corpus C shard key ('bfp_stab' -> 'bfp', 'dag_n_sccs' -> 'dag')."""
    for g in sorted(GROUP_ORDER_C, key=len, reverse=True):
        if key.startswith(g + "_"):
            return g
    raise ValueError(f"Cannot determine group for key: {key!r}")


# ---------------------------------------------------------------------------
# 2. Cache construction
# ---------------------------------------------------------------------------
def build_cache_c(corpus_dir, cache_dir):
    """Merge Corpus C shard .npz files into flat memory-mapped numpy arrays.

    Handles both plain index arrays (ABS_INDEX_C) and CSR sentinel pointer
    arrays (SENTINEL_PTRS_C).  Sentinel arrays are merged by appending each
    shard's interior elements (arr[1:]) with the running group offset, so the
    merged array always has shape (total_group_size + 1,)."""
    SHARD_GLOB_C = "corpus_c_shard_[0-9][0-9][0-9][0-9].npz"
    shards = sorted(glob.glob(os.path.join(corpus_dir, "shards", SHARD_GLOB_C)))
    if not shards:
        # Fall back to corpus_dir itself (generator writes directly there).
        shards = sorted(glob.glob(os.path.join(corpus_dir, SHARD_GLOB_C)))
    if not shards:
        raise FileNotFoundError(
            f"No shards found in {corpus_dir}/shards or {corpus_dir}")
    os.makedirs(cache_dir, exist_ok=True)

    sizes = []
    for path in shards:
        with np.load(path) as z:
            s = z["_sizes"]
            if s.shape[0] != len(GROUP_ORDER_C):
                raise ValueError(
                    f"{path}: _sizes has {s.shape[0]} entries, expected "
                    f"{len(GROUP_ORDER_C)} (groups {GROUP_ORDER_C}).  "
                    "Regenerate shards with the updated corpus_c_generator.py.")
            sizes.append(s)

    totals = {g: int(sum(s[j] for s in sizes)) for j, g in enumerate(GROUP_ORDER_C)}

    # Allocate output arrays.  Sentinel ptrs get +1 for the trailing sentinel.
    out = {}
    with np.load(shards[0]) as first:
        for k in first.files:
            if k in ("_sizes", "schema_version"):
                continue
            arr = first[k]
            if k in SENTINEL_PTRS_C:
                src = SENTINEL_PTRS_C[k]
                shape = (totals[src] + 1,) + arr.shape[1:]
            else:
                g = _group_of_c(k)
                shape = (totals[g],) + arr.shape[1:]
            out[k] = np.lib.format.open_memmap(
                os.path.join(cache_dir, k + ".npy"),
                mode="w+", dtype=arr.dtype, shape=shape)

    # Initialize leading zeros for sentinel arrays
    for k in SENTINEL_PTRS_C:
        out[k][0] = 0

    pos = {g: 0 for g in GROUP_ORDER_C}
    ptr_pos = {k: 1 for k in SENTINEL_PTRS_C}  # next write position (after the leading 0)

    for i, path in enumerate(shards):
        with np.load(path) as z:
            for k in out:
                if k in SENTINEL_PTRS_C:
                    tgt = SENTINEL_TGTS_C[k]
                    v = z[k][1:] + pos[tgt]   # drop shard's leading 0, add running offset
                    n_write = v.shape[0]
                    out[k][ptr_pos[k]:ptr_pos[k] + n_write] = v
                    ptr_pos[k] += n_write
                else:
                    v = z[k]
                    if k in ABS_INDEX_C:
                        v = v + pos[ABS_INDEX_C[k]]
                    g = _group_of_c(k)
                    out[k][pos[g]:pos[g] + v.shape[0]] = v

        for j, g in enumerate(GROUP_ORDER_C):
            pos[g] += int(sizes[i][j])
        if (i + 1) % 20 == 0 or i + 1 == len(shards):
            print(f"merged {i + 1}/{len(shards)} shards", flush=True)

    for v in out.values():
        v.flush()
    del out
    with open(os.path.join(cache_dir, "meta.json"), "w") as fh:
        json.dump({"n_shards": len(shards), **totals}, fh)
    _report_cache_c(totals)


def _report_cache_c(totals):
    print("Corpus C cache built:")
    for g, total in totals.items():
        print(f"  {g:10s}: {total:12d}")


# ---------------------------------------------------------------------------
# 3. Single-sample access
# ---------------------------------------------------------------------------
def open_cache_c(cache_dir):
    return {k: np.load(os.path.join(cache_dir, k + ".npy"), mmap_mode="r")
            for k in DATASET_KEYS_C}


def load_branch(a, b, max_fp):
    """One branch as padded numpy arrays. Requires n_fp(b) <= max_fp.

    Returns features for one (SCC, forcing-context) pair:
      n         : number of genes in the SCC
      k         : number of fixed points in this branch
      P, K      : production and degradation, padded to N_MAX
      adj       : sign matrix [N_MAX, N_MAX], 0 for absent edges
      T, nh, F  : intra-SCC edge parameters, neutral (1) for absent edges
      u         : forcing vector [N_MAX], multiplicative factor on P from parents
                  (1.0 = unforced gene, product of Hill functions for forced genes)
      fp_state, fp_charpoly, fp_stab : fixed-point arrays padded to max_fp
      consistent   : Brouwer-certified (index sum == 1)
      tc           : Thomas class of the SCC (1=TC_NEG, 2=TC_POS, 3=TC_BOTH)
      n_parents    : number of parent SCCs (0 for root)
      is_root      : convenience flag (n_parents == 0)
      dag_all_solved : True if every SCC in the parent DAG has >= 1 branch
    """
    scc_id = int(a["branch_scc"][b])
    dag_id = int(a["scc_dag_id"][scc_id])
    n      = int(a["scc_n"][scc_id])

    P  = np.ones(N_MAX, np.float32);  P[:n]  = a["scc_P"][scc_id, :n]
    Kd = np.ones(N_MAX, np.float32);  Kd[:n] = a["scc_K"][scc_id, :n]

    adj  = a["scc_sign"][scc_id, :n, :n].astype(np.float32)
    adj_pad = np.zeros((N_MAX, N_MAX), np.float32)
    adj_pad[:n, :n] = adj

    T    = np.ones((N_MAX, N_MAX), np.float32)
    nh   = np.ones((N_MAX, N_MAX), np.float32)
    fold = np.ones((N_MAX, N_MAX), np.float32)
    T[:n, :n]    = a["scc_ep_T"][scc_id, :n, :n]
    nh[:n, :n]   = a["scc_ep_nh"][scc_id, :n, :n]
    fold[:n, :n] = a["scc_ep_F"][scc_id, :n, :n]

    u_vec = np.ones(N_MAX, np.float32)
    u_vec[:n] = a["branch_u"][b, :n]

    f0 = int(a["branch_fp_ptr"][b])
    f1 = int(a["branch_fp_ptr"][b + 1])
    k  = f1 - f0

    fp_state = np.zeros((max_fp, N_MAX), np.float32)
    fp_cp    = np.zeros((max_fp, N_MAX), np.float32)
    fp_stab  = np.full(max_fp, -1, np.int64)
    if k > 0:
        fp_state[:k, :n] = a["bfp_x"][f0:f1, :n]
        fp_cp[:k, :n]    = np.nan_to_num(a["bfp_charpoly"][f0:f1, :n])
        st = np.asarray(a["bfp_stab"][f0:f1], dtype=np.int64)
        st[st == UNKNOWN_STABILITY_CODE] = UNKNOWN_STABILITY_CLASS
        fp_stab[:k] = st

    n_parents = int(a["scc_n_parents"][scc_id])
    return {
        "n": n, "k": k,
        "P": np.nan_to_num(P, nan=1.0),
        "K": np.nan_to_num(Kd, nan=1.0),
        "adj": adj_pad, "T": T, "nh": nh, "F": fold,
        "u": u_vec,
        "fp_state": fp_state, "fp_charpoly": fp_cp, "fp_stab": fp_stab,
        "consistent":    bool(a["branch_cons"][b]),
        "tc":            int(a["scc_tc"][scc_id]),
        "n_parents":     n_parents,
        "is_root":       n_parents == 0,
        "dag_all_solved": bool(a["dag_all_solved"][dag_id]),
    }


# ---------------------------------------------------------------------------
# 4. Dataset
# ---------------------------------------------------------------------------
class CorpusCDataset(Dataset):
    """One item = one branch (SCC + parent context) with all of its fixed points.

    Branches with zero fixed points or more than max_fp are excluded.
    Optionally restrict to branches whose DAG has dag_all_solved == True."""

    def __init__(self, cache_dir, max_fp, only_solved=False, max_samples=0, seed=0):
        self.cache_dir = cache_dir
        self.max_fp    = max_fp
        a = open_cache_c(cache_dir)
        n_branches  = a["branch_fp_ptr"].shape[0] - 1
        branch_nfp  = (np.asarray(a["branch_fp_ptr"][1:]) -
                       np.asarray(a["branch_fp_ptr"][:-1])).astype(np.int64)
        keep = (branch_nfp >= 1) & (branch_nfp <= max_fp)
        if only_solved:
            scc_ids   = np.asarray(a["branch_scc"])
            dag_ids   = np.asarray(a["scc_dag_id"])[scc_ids]
            dag_solved = np.asarray(a["dag_all_solved"])
            keep &= dag_solved[dag_ids]
        index = np.nonzero(keep)[0]
        if max_samples > 0 and index.shape[0] > max_samples:
            rng   = np.random.default_rng(seed)
            index = np.sort(rng.choice(index, size=max_samples, replace=False))
        self.index      = index
        scc_ids_idx     = np.asarray(a["branch_scc"])[index]
        self.sample_n   = np.asarray(a["scc_n"])[scc_ids_idx].astype(np.int64)
        self.sample_nfp = branch_nfp[index]
        self.sample_cons = np.asarray(a["branch_cons"])[index]
        self.sample_tc   = np.asarray(a["scc_tc"])[scc_ids_idx]
        self._a = None

    def __len__(self):
        return int(self.index.shape[0])

    def __getitem__(self, i):
        if self._a is None:
            self._a = open_cache_c(self.cache_dir)
        return load_branch(self._a, int(self.index[i]), self.max_fp)


def collate_c(batch):
    n = torch.tensor([b["n"] for b in batch], dtype=torch.long)
    k = torch.tensor([b["k"] for b in batch], dtype=torch.long)
    nb = int(n.max())
    kt = int(k.max())

    def stack(key):
        return torch.from_numpy(np.stack([b[key] for b in batch]))

    out = {
        "n": n, "k": k,
        "tc":          torch.tensor([b["tc"]          for b in batch], dtype=torch.long),
        "n_parents":   torch.tensor([b["n_parents"]   for b in batch], dtype=torch.long),
        "is_root":     torch.tensor([b["is_root"]     for b in batch], dtype=torch.bool),
        "P":    stack("P")[:, :nb].contiguous(),
        "K":    stack("K")[:, :nb].contiguous(),
        "adj":  stack("adj")[:, :nb, :nb].contiguous(),
        "T":    stack("T")[:, :nb, :nb].contiguous(),
        "nh":   stack("nh")[:, :nb, :nb].contiguous(),
        "F":    stack("F")[:, :nb, :nb].contiguous(),
        "u":    stack("u")[:, :nb].contiguous(),
        "fp_state":    stack("fp_state")[:, :kt, :nb].contiguous(),
        "fp_charpoly": stack("fp_charpoly")[:, :kt, :nb].contiguous(),
        "fp_stab":     stack("fp_stab")[:, :kt].contiguous(),
        "consistent":    torch.tensor([b["consistent"]    for b in batch], dtype=torch.bool),
        "dag_all_solved": torch.tensor([b["dag_all_solved"] for b in batch], dtype=torch.bool),
    }
    out["node_mask"] = torch.arange(nb).unsqueeze(0) < n.unsqueeze(1)
    out["fp_mask"]   = torch.arange(kt).unsqueeze(0) < k.unsqueeze(1)

    # has_pos_fb: True if SCC has a positive feedback loop (TC_POS=2 or TC_BOTH=3).
    # TC_NEG=1 means no positive feedback => at most one stable FP (Thomas criterion).
    out["has_pos_fb"] = out["tc"] != 1

    # xmax: invariant-box bound (1 + EPS_BOX) * P/K * prod_j max(1, F[j,i])
    # F[:, :nb, :nb] has fold-change for each edge (1.0 for absent edges).
    # Product over incoming edges (dim=1) of max(1, F) gives the bound factor.
    F_full = stack("F")[:, :nb, :nb].float()          # [B, nb, nb]
    bound  = F_full.clamp_min(1.0).prod(dim=1)         # [B, nb]
    P_full = out["P"].float()
    K_full = out["K"].float()
    out["xmax"] = ((1.0 + EPS_BOX) * P_full / K_full * bound).to(torch.float32)

    return out


def make_loader_c(cfg, train, only_solved=False, max_samples=None):
    """Training/validation loader for Corpus C, with size-stratified sampling."""
    if max_samples is None:
        max_samples = 0 if train else cfg.max_val_samples
    ds = CorpusCDataset(cfg.cache_dir_c, cfg.max_fp,
                        only_solved=only_solved,
                        max_samples=max_samples, seed=cfg.seed)
    kwargs = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        collate_fn=collate_c,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.num_workers > 0,
    )
    if cfg.num_workers > 0:
        kwargs["prefetch_factor"] = 4
    if train:
        counts = np.bincount(ds.sample_n)
        w = counts[ds.sample_n].astype(np.float64) ** (-cfg.sampler_power)
        kpower = getattr(cfg, "sampler_kpower", 0.0)
        if kpower > 0.0:
            # additionally upsample branches with more fixed points
            w *= ds.sample_nfp.astype(np.float64) ** kpower
        sampler = WeightedRandomSampler(
            torch.from_numpy(w),
            num_samples=cfg.steps_per_epoch * cfg.batch_size,
            replacement=True)
        return ds, DataLoader(ds, sampler=sampler, drop_last=True, **kwargs)
    return ds, DataLoader(ds, shuffle=False, drop_last=False, **kwargs)


def describe_dataset_c(ds, name):
    print(f"[{name}] branches {len(ds)}  certified {100.0 * ds.sample_cons.mean():.1f}%"
          f"  mean n_fp {ds.sample_nfp.mean():.2f}")
    tc_names = {1: "TC_NEG", 2: "TC_POS", 3: "TC_BOTH"}
    for tc, nm in tc_names.items():
        m = ds.sample_tc == tc
        if m.any():
            print(f"   {nm:8s}: {int(m.sum()):9d} branches"
                  f" | certified {100.0 * ds.sample_cons[m].mean():5.1f}%"
                  f" | mean n_fp {ds.sample_nfp[m].mean():.2f}")
    for n in range(2, N_MAX + 1):
        m = ds.sample_n == n
        if m.any():
            print(f"   n={n:2d}: {int(m.sum()):8d} branches"
                  f" | certified {100.0 * ds.sample_cons[m].mean():5.1f}%"
                  f" | mean n_fp {ds.sample_nfp[m].mean():.2f}")

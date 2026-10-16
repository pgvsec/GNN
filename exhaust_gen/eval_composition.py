"""Composition test: evaluate GRNPredictor on 6-gene graphs that decompose
into exactly two 3-gene SCCs, comparing predictions against Corpus A ground truth.

Usage (on gilbreth, from REPO_ROOT):
    python exhaust_gen/eval_composition.py \
        --l1_ckpt  /scratch/gilbreth/gupt1042/runs/exhaust_layer1/best.pt \
        --l3_ckpt  /scratch/gilbreth/gupt1042/runs/exhaust_layer3/best.pt \
        --corpus_cache /scratch/gilbreth/gupt1042/corpus_exhaust_A/cache \
        --out_csv   /scratch/gilbreth/gupt1042/runs/eval_composition.csv \
        --max_graphs 2000
"""

import argparse
import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_l1(path, device):
    from model import SccFixedPointNet
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = SimpleNamespace(**ck["cfg"])
    model = SccFixedPointNet(cfg)
    model.load_state_dict(ck.get("ema", ck.get("model")))
    model.to(device).eval()
    return model, cfg


def load_l3(path, l1_model, device):
    from model_c import CorpusCNet
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = SimpleNamespace(**ck["cfg"])
    model = CorpusCNet(l1_model, cfg)
    model.load_state_dict(ck.get("ema", ck.get("model")))
    model.to(device).eval()
    return model, cfg


def tarjan_sccs(n, adj_bool):
    """Returns list of SCCs as lists of node indices (reverse topo order)."""
    idx = [-1] * n
    low = [0] * n
    on_stack = [False] * n
    stack = []
    counter = [0]
    sccs = []

    def strongconnect(v):
        idx[v] = low[v] = counter[0]
        counter[0] += 1
        stack.append(v)
        on_stack[v] = True
        for w in range(n):
            if adj_bool[v, w]:
                if idx[w] == -1:
                    strongconnect(w)
                    low[v] = min(low[v], low[w])
                elif on_stack[w]:
                    low[v] = min(low[v], idx[w])
        if low[v] == idx[v]:
            scc = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                scc.append(w)
                if w == v:
                    break
            sccs.append(sorted(scc))

    for v in range(n):
        if idx[v] == -1:
            strongconnect(v)
    return sccs


def is_two_scc_of_size_k(adj, k=3):
    """True if the graph decomposes into exactly 2 SCCs each of size k."""
    n = adj.shape[0]
    if n != 2 * k:
        return False
    adj_bool = (adj != 0)
    sccs = tarjan_sccs(n, adj_bool)
    if len(sccs) != 2:
        return False
    return all(len(s) == k for s in sccs)


def count_exact_match(pred_n, gt_n):
    return int(pred_n == gt_n)


def fp_within_half_decade(pred_fps, gt_fps):
    """Fraction of gt FPs that have a predicted FP within 0.5 log10 decades."""
    if len(gt_fps) == 0:
        return float("nan")
    if len(pred_fps) == 0:
        return 0.0
    hits = 0
    for gfp in gt_fps:
        dists = [np.max(np.abs(np.log10(np.clip(pfp, 1e-12, None))
                               - np.log10(np.clip(gfp,  1e-12, None))))
                 for pfp in pred_fps]
        if min(dists) <= 0.5:
            hits += 1
    return hits / len(gt_fps)


def stable_binary_acc(pred_stab, gt_stab):
    """Accuracy of stable/unstable classification."""
    STABLE = {0, 1}
    if len(gt_stab) == 0:
        return float("nan")
    # match by count (lenient: compare multisets of is_stable)
    gt_stable = sorted([int(s in STABLE) for s in gt_stab])
    pred_stable = sorted([int(s in STABLE) for s in pred_stab])
    n = max(len(gt_stable), len(pred_stable))
    # pad shorter with 0
    gt_stable  += [0] * (n - len(gt_stable))
    pred_stable += [0] * (n - len(pred_stable))
    return sum(a == b for a, b in zip(gt_stable, pred_stable)) / n


# ---------------------------------------------------------------------------
# Load Corpus A cache (memmap)
# ---------------------------------------------------------------------------

def load_corpus_a_cache(cache_dir):
    cache_dir = Path(cache_dir)
    meta = np.load(cache_dir / "meta.npz")
    sizes = {k: int(meta[k]) for k in meta.files}

    def mm(name, dtype, shape):
        p = cache_dir / f"{name}.bin"
        return np.memmap(p, dtype=dtype, mode="r", shape=shape)

    n_g  = sizes["n_g"]
    n_e  = sizes["n_e"]
    n_s  = sizes["n_s"]
    n_ep = sizes["n_ep"]
    n_fp = sizes["n_fp"]

    # graph-level
    g_n        = mm("g_n",        np.int32,   (n_g,))
    g_n_edges  = mm("g_n_edges",  np.uint8,   (n_g,))
    g_edge_ptr = mm("g_edge_ptr", np.int64,   (n_g + 1,))
    g_split    = mm("g_split",    np.uint8,   (n_g,))

    # edge-level
    e_src = mm("e_src", np.int64,   (n_e,))
    e_tgt = mm("e_tgt", np.int64,   (n_e,))
    e_sgn = mm("e_sgn", np.float32, (n_e,))

    # sample-level — determine N_MAX from s_P shape
    N_MAX = sizes.get("N_MAX", 10)
    s_graph  = mm("s_graph",  np.int64,   (n_s,))
    s_P      = mm("s_P",      np.float32, (n_s, N_MAX))
    s_K      = mm("s_K",      np.float32, (n_s, N_MAX))
    s_n_fp   = mm("s_n_fp",   np.int32,   (n_s,))
    s_ep_ptr = mm("s_ep_ptr", np.int64,   (n_s + 1,))
    s_fp_ptr = mm("s_fp_ptr", np.int64,   (n_s + 1,))
    s_phase  = mm("s_phase",  np.uint8,   (n_s,))

    # edge-parameter-level
    ep_T  = mm("ep_T",  np.float32, (n_ep,))
    ep_nh = mm("ep_nh", np.uint8,   (n_ep,))
    ep_F  = mm("ep_F",  np.float32, (n_ep,))

    # fixed-point-level
    fp_state = mm("fp_state", np.float32, (n_fp, N_MAX))
    fp_stab  = mm("fp_stab",  np.uint8,   (n_fp,))

    return dict(
        n_g=n_g, n_e=n_e, n_s=n_s, n_ep=n_ep, n_fp=n_fp, N_MAX=N_MAX,
        g_n=g_n, g_n_edges=g_n_edges, g_edge_ptr=g_edge_ptr, g_split=g_split,
        e_src=e_src, e_tgt=e_tgt, e_sgn=e_sgn,
        s_graph=s_graph, s_P=s_P, s_K=s_K, s_n_fp=s_n_fp,
        s_ep_ptr=s_ep_ptr, s_fp_ptr=s_fp_ptr, s_phase=s_phase,
        ep_T=ep_T, ep_nh=ep_nh, ep_F=ep_F,
        fp_state=fp_state, fp_stab=fp_stab,
    )


def build_adj(n, e_src, e_tgt, e_sgn):
    adj = np.zeros((n, n), dtype=np.float32)
    for s, t, sg in zip(e_src, e_tgt, e_sgn):
        adj[s, t] = sg
    return adj


def build_params(n, N_MAX, s_P, s_K, e_src, e_tgt, ep_T, ep_nh, ep_F):
    P  = s_P[:n].copy()
    K  = s_K[:n].copy()
    T  = np.ones((n, n), dtype=np.float32)
    nh = np.ones((n, n), dtype=np.float32) * 2.0
    F  = np.ones((n, n), dtype=np.float32) * 2.0
    for i, (s, t) in enumerate(zip(e_src, e_tgt)):
        T[s, t]  = ep_T[i]
        nh[s, t] = float(ep_nh[i])
        F[s, t]  = ep_F[i]
    return P, K, T, nh, F


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l1_ckpt",      required=True)
    ap.add_argument("--l3_ckpt",      required=True)
    ap.add_argument("--corpus_cache", required=True)
    ap.add_argument("--out_csv",      default="eval_composition.csv")
    ap.add_argument("--scc_size",     type=int, default=3,
                    help="Size of each SCC (default 3, so graphs are 2*scc_size genes)")
    ap.add_argument("--max_graphs",   type=int, default=5000,
                    help="Max number of graphs to test (topology level)")
    ap.add_argument("--max_samples",  type=int, default=3,
                    help="Max parameter draws per topology")
    ap.add_argument("--split",        type=int, default=None,
                    help="0=train,1=val,2=test; None=all")
    ap.add_argument("--no_cuda",      action="store_true")
    args = ap.parse_args()

    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    print(f"Device: {device}")

    print("Loading models...")
    l1_model, _ = load_l1(args.l1_ckpt, device)
    l3_model, _ = load_l3(args.l3_ckpt, l1_model, device)

    from assemble import GRNPredictor
    predictor = GRNPredictor(l1_model, l3_model, device=device, refine_steps=5)

    print(f"Loading corpus cache from {args.corpus_cache}...")
    C = load_corpus_a_cache(args.corpus_cache)
    N_MAX = C["N_MAX"]
    target_n = 2 * args.scc_size

    print(f"Scanning for {target_n}-gene graphs with 2 SCCs of size {args.scc_size}...")

    # Find qualifying graph indices
    qualifying = []
    for gi in range(C["n_g"]):
        if int(C["g_n"][gi]) != target_n:
            continue
        if args.split is not None and int(C["g_split"][gi]) != args.split:
            continue
        ep0, ep1 = int(C["g_edge_ptr"][gi]), int(C["g_edge_ptr"][gi + 1])
        e_src = C["e_src"][ep0:ep1]
        e_tgt = C["e_tgt"][ep0:ep1]
        e_sgn = C["e_sgn"][ep0:ep1]
        adj = build_adj(target_n, e_src, e_tgt, e_sgn)
        if is_two_scc_of_size_k(adj, k=args.scc_size):
            qualifying.append(gi)
        if len(qualifying) >= args.max_graphs:
            break

    print(f"Found {len(qualifying)} qualifying topologies.")
    if len(qualifying) == 0:
        print("No qualifying graphs found. Try --scc_size 2 or check corpus_cache path.")
        sys.exit(1)

    # Build sample index: sample -> graph
    print("Building sample index...")
    s_graph_arr = C["s_graph"]
    qualifying_set = set(qualifying)
    # group samples by graph
    graph_to_samples = {gi: [] for gi in qualifying}
    for si in range(C["n_s"]):
        gi = int(s_graph_arr[si])
        if gi in qualifying_set:
            graph_to_samples[gi].append(si)

    # Run evaluation
    rows = []
    n_total = n_count_exact = n_fp_hit = n_fp_total = 0
    errors = 0

    for gi in qualifying:
        ep0, ep1 = int(C["g_edge_ptr"][gi]), int(C["g_edge_ptr"][gi + 1])
        e_src = C["e_src"][ep0:ep1]
        e_tgt = C["e_tgt"][ep0:ep1]
        e_sgn = C["e_sgn"][ep0:ep1]
        adj = build_adj(target_n, e_src, e_tgt, e_sgn)

        samples = graph_to_samples[gi][:args.max_samples]
        for si in samples:
            n_fp_gt = int(C["s_n_fp"][si])
            fp0, fp1 = int(C["s_fp_ptr"][si]), int(C["s_fp_ptr"][si + 1])
            gt_fps  = C["fp_state"][fp0:fp1, :target_n].copy()
            gt_stab = C["fp_stab"][fp0:fp1].copy()

            P_raw = C["s_P"][si]
            K_raw = C["s_K"][si]
            # edge-parameter ptr for this sample
            ep_s0, ep_s1 = int(C["s_ep_ptr"][si]), int(C["s_ep_ptr"][si + 1])
            ep_T  = C["ep_T"][ep_s0:ep_s1]
            ep_nh = C["ep_nh"][ep_s0:ep_s1]
            ep_F  = C["ep_F"][ep_s0:ep_s1]

            P, K, T, nh, F = build_params(
                target_n, N_MAX, P_raw, K_raw, e_src, e_tgt, ep_T, ep_nh, ep_F
            )

            try:
                with torch.no_grad():
                    result = predictor.predict(adj, P, K, T, nh, F, verbose=False)
            except Exception as exc:
                errors += 1
                print(f"  Error on graph {gi} sample {si}: {exc}")
                continue

            pred_fps  = result.get("fps_refined", result["global_fps"])
            pred_stab = result["global_stab"]
            n_pred = result["n_fps"]

            ce = count_exact_match(n_pred, n_fp_gt)
            hd = fp_within_half_decade(pred_fps, gt_fps)
            sa = stable_binary_acc(pred_stab, gt_stab)

            rows.append({
                "graph_idx":    gi,
                "sample_idx":   si,
                "n_genes":      target_n,
                "n_fp_gt":      n_fp_gt,
                "n_fp_pred":    n_pred,
                "count_exact":  ce,
                "fp_half_decade": hd,
                "stable_acc":   sa,
                "n_converged":  result.get("n_converged", -1),
            })

            n_total += 1
            n_count_exact += ce
            if not np.isnan(hd):
                n_fp_hit += hd * n_fp_gt
                n_fp_total += n_fp_gt

    print(f"\n{'='*60}")
    print(f"Evaluated {n_total} samples ({errors} errors)")
    if n_total > 0:
        print(f"count_exact          : {n_count_exact/n_total:.4f}  ({n_count_exact}/{n_total})")
    if n_fp_total > 0:
        print(f"fp_within_half_decade: {n_fp_hit/n_fp_total:.4f}")
    print(f"{'='*60}")

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results saved to {out}")


if __name__ == "__main__":
    main()

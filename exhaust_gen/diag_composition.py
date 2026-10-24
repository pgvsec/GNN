"""Diagnostic: check count_logit distribution and model FP predictions post-fix."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

def load_l1(path, device):
    from model import SccFixedPointNet
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = SimpleNamespace(**ck["cfg"])
    m = SccFixedPointNet(cfg); m.load_state_dict(ck.get("ema", ck.get("model")))
    return m.to(device).eval()

def load_l3(path, l1, device):
    from model_c import CorpusCNet
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = SimpleNamespace(**ck["cfg"])
    m = CorpusCNet(l1, cfg); m.load_state_dict(ck.get("ema", ck.get("model")))
    return m.to(device).eval()

ap = argparse.ArgumentParser()
ap.add_argument("--l1_ckpt", required=True)
ap.add_argument("--l3_ckpt", required=True)
ap.add_argument("--eval_dir", required=True)
ap.add_argument("--corpus_a_cache", default=None)
ap.add_argument("--n_cases", type=int, default=20)
args = ap.parse_args()

device = torch.device("cpu")
l1 = load_l1(args.l1_ckpt, device)
l3 = load_l3(args.l3_ckpt, l1, device)

# Check for NaN in model weights
l1_nan = sum(p.isnan().sum().item() for p in l1.parameters())
l3_nan = sum(p.isnan().sum().item() for p in l3.parameters())
l1_norm = sum(p.norm().item() for p in l1.parameters())
print(f"L1 weight NaN count: {l1_nan}, total norm: {l1_norm:.4f}")
print(f"L3 weight NaN count: {l3_nan}")
import inspect, assemble as _asm
print(f"assemble.py path: {inspect.getfile(_asm)}")

d = Path(args.eval_dir)
adj_all  = np.load(d/"adj.npy")
P_all    = np.load(d/"P.npy")
K_all    = np.load(d/"K.npy")
T_all    = np.load(d/"T.npy")
nh_all   = np.load(d/"nh.npy")
F_all    = np.load(d/"F.npy")
n_fp_arr = np.load(d/"n_fp.npy")

from assemble import decompose_grn, _make_scc_batch, _compute_forcing
from model import decode_slots

for i in range(args.n_cases):
    adj = adj_all[i]; P = P_all[i].astype(np.float32)
    K   = K_all[i].astype(np.float32); T = T_all[i].astype(np.float32)
    nh  = nh_all[i].astype(np.float32); F = F_all[i].astype(np.float32)
    n_fp_gt = int(n_fp_arr[i])

    dag = decompose_grn(adj, P, K, T, nh, F)
    print(f"\n=== Case {i} | GT n_fp={n_fp_gt} ===")

    for scc in dag.sccs:
        is_root = len(scc.parents) == 0
        u = np.ones(scc.n, dtype=np.float32)
        # Use zero parent FP as placeholder for forced SCC
        if not is_root:
            u_contrib = _compute_forcing(dag, scc.parents[0], scc.scc_id,
                                         np.zeros(dag.sccs[scc.parents[0]].n),
                                         adj, T, nh, F)
            u *= u_contrib

        batch = _make_scc_batch(scc, u, device)

        with torch.no_grad():
            if is_root:
                out = l1(batch)
                model_name = "L1"
            else:
                out, _ = l3(batch, freeze_backbone=True)
                model_name = "L3"

        cl = out["count_logit"].float()[0]   # [max_fp+1]
        el = out["exist_logit"].float()[0]   # [K_slots]
        ep = torch.sigmoid(el)

        # Find predicted k (odd indices only)
        even_mask = (torch.arange(cl.shape[0]) % 2 == 0)
        cl_odd = cl.clone(); cl_odd[even_mask] = float("-inf")
        k_pred = int(cl_odd.argmax().item())

        # Top-8 count logit values
        cl_top = [(int(j), float(cl[j].item())) for j in range(min(10, cl.shape[0]))]
        ep_top = [f"{float(ep[j].item()):.3f}" for j in range(min(8, ep.shape[0]))]

        n_slots_true = int((ep > 0.5).sum().item())

        print(f"  SCC {scc.scc_id} n={scc.n} {model_name} | "
              f"k_pred={k_pred} | count_logit[0..9]={[f'{v:.2f}' for _,v in cl_top]} | "
              f"exist_prob[:8]={ep_top} | slots>0.5: {n_slots_true}")

# --- Sanity check: does L1 work on a hand-crafted 3-gene TC_BOTH case? ---
print("\n\n=== L1 SANITY CHECK (3-gene bistable) ===")
# Create a simple 3-gene toggle network that should have 3 FPs
n = 3
adj_s = np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]], dtype=np.float32)
P_s = np.array([5.0, 5.0, 5.0], dtype=np.float32)
K_s = np.array([1.0, 1.0, 1.0], dtype=np.float32)
T_s = np.ones((n, n), dtype=np.float32)
nh_s = np.full((n, n), 2.0, dtype=np.float32)
F_s = np.where(adj_s != 0, np.where(adj_s > 0, 5.0, 0.2), 1.0).astype(np.float32)

from assemble import SccNode
scc_s = SccNode(0, list(range(n)), n, adj_s, P_s, K_s, T_s, nh_s, F_s, [], [])
u_s = np.ones(n, dtype=np.float32)
batch_s = _make_scc_batch(scc_s, u_s, device)
print(f"  batch xmax (first 6): {batch_s['xmax'][0,:6].tolist()}")
print(f"  batch P (first 6): {batch_s['P'][0,:6].tolist()}")
print(f"  batch K (first 6): {batch_s['K'][0,:6].tolist()}")
with torch.no_grad():
    out_s = l1(batch_s)
cl_s = out_s["count_logit"].float()[0]
el_s = out_s["exist_logit"].float()[0]
ep_s = torch.sigmoid(el_s)
even_mask = (torch.arange(cl_s.shape[0]) % 2 == 0)
cl_s_odd = cl_s.clone(); cl_s_odd[even_mask] = float("-inf")
k_pred_s = int(cl_s_odd.argmax().item())
print(f"  k_pred={k_pred_s} | count_logit[0..9]={[f'{cl_s[j].item():.2f}' for j in range(min(10, cl_s.shape[0]))]} | exist_prob[:4]={[f'{ep_s[j].item():.3f}' for j in range(4)]}")

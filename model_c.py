"""Layer III network: CorpusCNet — SCC-DAG composition.

Architecture overview (Overleaf §9):

  CorpusCNet is a thin wrapper around CorpusBNet (Φ_ψ + A_η + U_ω + frozen E_θ/D_φ).

  The only structural difference from Layer II is how "ports" are constructed:
    - Corpus B: port slots are stored explicitly (port_gene, port_u, port_T, etc.)
    - Corpus C: forcing comes from the DAG — gene b_k receives u_k = ∏ H_ab(x*_a) from
      one or more inter-SCC edges.  The batch carries `branch_u [B, N_MAX]` (already
      the product of all Hills) and the inter-SCC edge descriptors that produced it.

  CorpusCNet.forward() synthesizes the port tensors from the Corpus C batch before
  delegating to the inherited CorpusBNet forward path.  The backbone and new modules
  (Φ_ψ, A_η, U_ω) are identical — this keeps Layer II and III weights directly
  transferable as a warm-start.

Port synthesis (Option B — data/model separation):
  Each active gene b_k that is forced (u[b_k] != 1.0) becomes one port slot.
  Its port parameters (T, nh, F) are the *aggregated* Hill parameters of all
  inter-SCC edges targeting gene b_k.  For a single edge the aggregation is
  trivial; for multiple edges (fan-in) we compute effective T/nh/F as the
  geometric mean of each parameter, since multiple activators multiply.

  Genes with u == 1.0 (unforced / root SCC) are mapped to inactive slots.
  Inactive slots carry gene_idx = -1, u = 0, T = nh = F = 1 (neutral Hill edge).

  The maximum number of forced genes per sample is capped at MAX_PORTS_C;
  samples exceeding this cap are handled by keeping the strongest signals
  (largest |log u|).

Sections
  1. Port synthesis utilities
  2. CorpusCNet
"""
import torch
import torch.nn as nn
import torch.nn.functional as Fnn

from data import N_MAX
from model import SccFixedPointNet
from model_b import CorpusBNet

# Maximum number of port slots synthesized per sample.
# Set ≥ typical max inter-SCC fan-in; 8 covers essentially all DAG shapes.
MAX_PORTS_C = 8


# ---------------------------------------------------------------------------
# 1. Port synthesis utilities
# ---------------------------------------------------------------------------

def _compute_xmax(P, K):
    """Approximate xmax = P / K (entry-level steady state, as used by node_enc).

    P [B, N], K [B, N] → xmax [B, N], clamped to a positive minimum.
    """
    return (P / K.clamp_min(1e-6)).clamp_min(1e-6)


def synthesize_ports(batch, max_ports=MAX_PORTS_C):
    """Convert a Corpus C batch into the port-slot tensors expected by CorpusBNet.forward.

    The Corpus C batch provides:
      batch["u"]        [B, N]   per-gene forcing multiplier (1.0 = unforced)
      batch["n"]        [B]      number of active genes

    Port assignment:
      - A gene b is a port if u[b] != 1.0 (it receives forcing from a parent SCC).
      - For each sample, collect forced genes and sort by decreasing |log u| to keep
        the most influential ports when count exceeds max_ports.
      - Fill port_gene / port_u / port_T / port_nh / port_F tensors of shape [B, L]
        where L = max_ports (inactive slots get gene=-1, u=0, T=nh=F=1).

    Port Hill parameters:
      Corpus C does NOT store per-port T/nh/F at the batch level — that information
      is in iedge_* which is shard-level, not pre-collated.  However, the Hill
      parameters are only used by Φ_ψ to encode the *nature* of the signal, not to
      re-derive u (u is already precomputed in branch_u).  We set:
        port_T  = 1.0  (neutral — Φ_ψ will still distinguish via u magnitude)
        port_nh = 2.0  (typical Hill coefficient)
        port_F  = 2.0  (typical fold-change magnitude)
      This is a conservative approximation.  A future version can pass per-port
      T/nh/F through collate_c by reading iedge_* per branch.

    Returns a dict with keys:
      port_gene [B, L] int64
      port_u    [B, L] float32
      port_T    [B, L] float32
      port_nh   [B, L] float32
      port_F    [B, L] float32
      port_mask [B, L] bool
    """
    u   = batch["u"]          # [B, N]
    n   = batch["n"]          # [B]
    B   = u.shape[0]
    dev = u.device
    L   = max_ports

    port_gene = torch.full((B, L), -1,   dtype=torch.long,    device=dev)
    port_u    = torch.zeros((B, L),       dtype=torch.float32, device=dev)
    port_T    = torch.ones((B, L),        dtype=torch.float32, device=dev)
    port_nh   = torch.full((B, L), 2.0,  dtype=torch.float32, device=dev)
    port_F    = torch.full((B, L), 2.0,  dtype=torch.float32, device=dev)
    port_mask = torch.zeros((B, L),      dtype=torch.bool,    device=dev)

    for b in range(B):
        nb   = int(n[b].item())
        u_b  = u[b, :nb]                          # [nb]
        # Forced genes: u != 1.0 (tolerance 1e-5 for float precision)
        forced = (u_b - 1.0).abs() > 1e-5         # [nb]
        idxs   = forced.nonzero(as_tuple=False).squeeze(1)   # [n_forced]
        if idxs.numel() == 0:
            continue
        # Sort by descending |log u| — most informative ports first
        log_u  = u_b[idxs].clamp_min(1e-9).log().abs()
        order  = log_u.argsort(descending=True)
        idxs   = idxs[order]
        n_use  = min(idxs.numel(), L)
        port_gene[b, :n_use] = idxs[:n_use]
        port_u[b, :n_use]    = u_b[idxs[:n_use]]
        port_mask[b, :n_use] = True

    return {
        "port_gene": port_gene,
        "port_u":    port_u,
        "port_T":    port_T,
        "port_nh":   port_nh,
        "port_F":    port_F,
        "port_mask": port_mask,
    }


# ---------------------------------------------------------------------------
# 2. CorpusCNet
# ---------------------------------------------------------------------------

class CorpusCNet(CorpusBNet):
    """Layer III network for Corpus C (SCC-DAG multi-SCC samples).

    Inherits all of CorpusBNet (frozen backbone + Φ_ψ + A_η + U_ω + D_φ).
    The only override is forward(): synthesize port tensors from branch_u before
    delegating to the parent CorpusBNet forward path.

    This means:
      - A Layer II checkpoint (best.pt from train_layer2.py) can be loaded
        directly as a warm-start for Layer III; all parameter names match.
      - Layer III training fine-tunes only Φ_ψ + A_η + U_ω (backbone still frozen
        by default, backbone_trainable=False).
    """

    def __init__(self, backbone: SccFixedPointNet, cfg):
        super().__init__(backbone, cfg)
        self.max_ports_c = getattr(cfg, "max_ports_c", MAX_PORTS_C)

    def forward(self, batch, freeze_backbone=True):
        """
        batch : Corpus C batch dict from collate_c, plus node_mask.
                Must contain: P, K, adj, T, nh, F, u, node_mask, fp_state, ...
                Does NOT need port_* fields — they are synthesized here.

        Returns: (out_cond, out0) — same as CorpusBNet.forward
        """
        # Synthesize xmax = P / K (not stored in Corpus C)
        if "xmax" not in batch:
            batch = dict(batch)    # shallow copy to avoid mutating the original
            batch["xmax"] = _compute_xmax(batch["P"], batch["K"])

        # Synthesize port tensors from branch_u
        ports = synthesize_ports(batch, max_ports=self.max_ports_c)
        if "port_gene" not in batch:
            batch = dict(batch)
            batch.update(ports)
        else:
            batch = dict(batch)
            batch.update(ports)   # always overwrite — C batches don't store ports

        # Also need has_pos_fb for the physics/thomas loss; C doesn't store it —
        # default to False (disables thomas loss, safe for Corpus C).
        if "has_pos_fb" not in batch:
            batch["has_pos_fb"] = torch.zeros(
                batch["P"].shape[0], dtype=torch.bool, device=batch["P"].device)

        # Alias dag_all_solved → consistent (used by SetCriterion via batch["consistent"])
        if "consistent" not in batch and "dag_all_solved" in batch:
            batch["consistent"] = batch["dag_all_solved"]

        return super().forward(batch, freeze_backbone=freeze_backbone)

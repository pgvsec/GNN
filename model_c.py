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

Count head (ForcingAwareCountHead):
  The core difficulty at Layer III is predicting k when forcing u pushes the
  SCC across a bifurcation boundary (k=3 → k=1 saddle-node).  The correct
  inputs to the count head are:

    1. sorted sigmoid(exist_logit) [B, K]
       The SlotDecoder's slot-level evidence for how many FPs exist.  With
       w_exist_cnt > 0, sigmoid(exist_logit).sum() ≈ k_true during training,
       making this the strongest available count signal.

    2. Conditioned graph embedding z_cond [B, d]
       From U_ω after incorporating the forcing signal.  Encodes the forcing
       indirectly but also the full topology + kinetic context.

    3. Forcing statistics u_stats [B, 3]:
         (max |log u| over active genes,
          mean |log u| over active genes,
          n_forced / n)
       These directly encode how far the SCC has been pushed from its
       autonomous operating point.  High |log u| → SCC is near or past a
       bifurcation boundary.

    4. Thomas class one-hot [B, 3]
       tc ∈ {1=TC_NEG, 2=TC_POS, 3=TC_BOTH}.  TC_NEG → no positive feedback
       → multistability is topologically impossible → k_stable ≤ 1.  This is
       the strongest structural constraint on k and must reach the count head.

  All inputs are detached so count gradients do not flow back through U_ω.

Sections
  1. Port synthesis utilities
  2. ForcingAwareCountHead
  3. CorpusCNet
"""
import torch
import torch.nn as nn
import torch.nn.functional as Fnn

from data import N_MAX
from model import SccFixedPointNet, MLP
from model_b import CorpusBNet

# Maximum number of port slots synthesized per sample.
# Set ≥ typical max inter-SCC fan-in; 8 covers essentially all DAG shapes.
MAX_PORTS_C = 8

# Default maximum FP count the forcing-aware head can predict.
DEFAULT_MAX_COUNT_C = 32


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


def _forcing_stats(u, n, node_mask):
    """Compute three scalar forcing statistics per sample.

    Returns u_stats [B, 3]:
      col 0: max  |log u| over active genes (clamped to [-10, 10])
      col 1: mean |log u| over active genes (masked mean)
      col 2: n_forced / n  (fraction of genes that are forced)

    These directly encode how far the SCC has been pushed from its autonomous
    operating point.  The count head uses them to locate the SCC relative to
    its bifurcation boundaries.
    """
    dev    = u.device
    B, N   = u.shape
    mask_f = node_mask.float()                        # [B, N]
    n_f    = mask_f.sum(dim=1).clamp_min(1.0)        # [B]

    log_u_abs = (u - 1.0).abs().clamp_min(1e-9).log().clamp(-10.0, 10.0)  # [B, N]
    log_u_abs = log_u_abs * mask_f                    # zero out padding

    # Padded genes must not win the max (a zero beats the negative values of weakly forced
    # genes), or the feature would depend on how far the batch is padded.
    u_max_log  = log_u_abs.masked_fill(~node_mask, -10.0).max(dim=1).values   # [B]
    u_mean_log = log_u_abs.sum(dim=1) / n_f           # [B]

    forced      = ((u - 1.0).abs() > 1e-5) & node_mask   # [B, N]
    n_forced_norm = forced.float().sum(dim=1) / n_f       # [B]

    return torch.stack([u_max_log, u_mean_log, n_forced_norm], dim=1)  # [B, 3]


# ---------------------------------------------------------------------------
# 2. ForcingAwareCountHead
# ---------------------------------------------------------------------------

class ForcingAwareCountHead(nn.Module):
    """Count head for Corpus C forced SCCs (v1 — uses exist_logit).

    Reads four groups of features (all detached — no gradients flow back
    through U_ω or the backbone):

      1. sorted sigmoid(exist_logit) [B, K]
      2. Conditioned graph embedding z_cond [B, d]
      3. Forcing statistics u_stats [B, 3]
      4. Thomas class one-hot [B, 3]

    Total input: K + d_model + 3 + 3 → 3-layer MLP → count_logit[0..max_count].

    Note: coupling to exist_logit prevents decoder training (any decoder
    gradient shifts exist_logit and breaks count calibration).  Use
    ForcingAwareCountHeadV2 (cfg.use_v2_count_head=True) to decouple.
    """

    def __init__(self, d_model: int, K: int,
                 max_count: int = DEFAULT_MAX_COUNT_C,
                 dropout: float = 0.0):
        super().__init__()
        self.max_count = max_count
        d_in = K + d_model + 3 + 3   # exist_probs + z_cond + u_stats + tc_onehot
        self.mlp = MLP(d_in, d_model, max_count + 1, n_layers=3, dropout=dropout)

    def forward(self,
                exist_logit: torch.Tensor,   # [B, K]
                z_cond:      torch.Tensor,   # [B, d]
                u_stats:     torch.Tensor,   # [B, 3]
                tc_onehot:   torch.Tensor,   # [B, 3]
                ) -> torch.Tensor:           # [B, max_count+1]
        probs_sorted = torch.sigmoid(exist_logit).sort(dim=1, descending=True).values
        feat = torch.cat([probs_sorted, z_cond, u_stats, tc_onehot], dim=-1)
        return self.mlp(feat)


class ForcingAwareCountHeadV2(nn.Module):
    """Count head for Corpus C forced SCCs (v2 — decoupled from exist_logit).

    Reads only three feature groups (all detached):

      1. Conditioned graph embedding z_cond [B, d]
         A_η output encoding topology, kinetics, and the forcing signal.
      2. Forcing statistics u_stats [B, 3]
         (max|log u|, mean|log u|, n_forced/n)
      3. Thomas class one-hot [B, 3]

    Does NOT read exist_logit.  This breaks the count head's dependence on
    the decoder output, so the decoder (SlotDecoder + exist_head) can be
    trained without disrupting count calibration.

    Total input: d_model + 3 + 3 → 3-layer MLP → count_logit[0..max_count].
    """

    def __init__(self, d_model: int,
                 max_count: int = DEFAULT_MAX_COUNT_C,
                 dropout: float = 0.0):
        super().__init__()
        self.max_count = max_count
        d_in = d_model + 3 + 3   # z_cond + u_stats + tc_onehot
        self.mlp = MLP(d_in, d_model, max_count + 1, n_layers=3, dropout=dropout)

    def forward(self,
                z_cond:    torch.Tensor,   # [B, d]
                u_stats:   torch.Tensor,   # [B, 3]
                tc_onehot: torch.Tensor,   # [B, 3]
                ) -> torch.Tensor:         # [B, max_count+1]
        feat = torch.cat([z_cond, u_stats, tc_onehot], dim=-1)
        return self.mlp(feat)


# ---------------------------------------------------------------------------
# 3. CorpusCNet
# ---------------------------------------------------------------------------

class CorpusCNet(CorpusBNet):
    """Layer III network for Corpus C (SCC-DAG multi-SCC samples).

    Inherits all of CorpusBNet (frozen backbone + Φ_ψ + A_η + U_ω + D_φ).
    The only override is forward(): synthesize port tensors from branch_u before
    delegating to the parent CorpusBNet forward path, then replace count_logit
    with a forcing-aware prediction.

    This means:
      - A Layer II checkpoint (best.pt from train_layer2.py) can be loaded
        directly as a warm-start for Layer III; all parameter names match.
      - Layer III training fine-tunes only Φ_ψ + A_η + U_ω + forcing_count_head
        (backbone still frozen by default, backbone_trainable=False).
    """

    def __init__(self, backbone: SccFixedPointNet, cfg):
        super().__init__(backbone, cfg)
        self.max_ports_c = getattr(cfg, "max_ports_c", MAX_PORTS_C)

        max_count_c = getattr(cfg, "max_count_c", DEFAULT_MAX_COUNT_C)
        if getattr(cfg, "use_v2_count_head", False):
            self.forcing_count_head = ForcingAwareCountHeadV2(
                d_model=cfg.d_model,
                max_count=max_count_c,
                dropout=cfg.dropout,
            )
        else:
            self.forcing_count_head = ForcingAwareCountHead(
                d_model=cfg.d_model,
                K=cfg.max_fp,
                max_count=max_count_c,
                dropout=cfg.dropout,
            )

    def forward(self, batch, freeze_backbone=True):
        """
        batch : Corpus C batch dict from collate_c, plus node_mask.
                Must contain: P, K, adj, T, nh, F, u, n, tc, node_mask, fp_state, ...
                Does NOT need port_* fields — they are synthesized here.

        Returns: (out_cond, out0) — same as CorpusBNet.forward
        """
        # Synthesize xmax = P / K (not stored in Corpus C)
        if "xmax" not in batch:
            batch = dict(batch)
            batch["xmax"] = _compute_xmax(batch["P"], batch["K"])

        # Synthesize port tensors from branch_u
        batch = dict(batch)
        batch.update(synthesize_ports(batch, max_ports=self.max_ports_c))

        # has_pos_fb: C doesn't store it — default False (disables thomas loss).
        if "has_pos_fb" not in batch:
            batch["has_pos_fb"] = torch.zeros(
                batch["P"].shape[0], dtype=torch.bool, device=batch["P"].device)

        # Alias dag_all_solved → consistent (used by SetCriterion)
        if "consistent" not in batch and "dag_all_solved" in batch:
            batch["consistent"] = batch["dag_all_solved"]

        out_cond, out0 = super().forward(batch, freeze_backbone=freeze_backbone)

        # ------------------------------------------------------------------
        # Forcing-aware count head: replace backbone count_logit.
        #
        # The backbone count head reads z_cf and encodes forcing only
        # implicitly.  Here we give the count head explicit access to:
        #   - slot existence evidence  (already calibrated to sum to k)
        #   - conditioned graph embedding
        #   - forcing magnitude statistics (where the SCC sits relative to
        #     its bifurcation boundaries)
        #   - Thomas class (structural constraint on max stable FP count)
        # All inputs are detached — count gradients must not flow into U_ω.
        # ------------------------------------------------------------------
        z_cond = out_cond.pop("_z_cond")      # [B, d]
        out_cond.pop("_h_cond", None)          # not needed here

        node_mask = batch["node_mask"]         # [B, N]

        # Forcing statistics
        u_stats = _forcing_stats(batch["u"], batch["n"], node_mask)   # [B, 3]

        # Thomas class one-hot: tc ∈ {1, 2, 3} → indices 0, 1, 2
        tc = batch.get("tc", torch.ones(
            batch["P"].shape[0], dtype=torch.long, device=batch["P"].device))
        tc_onehot = Fnn.one_hot((tc - 1).clamp(0, 2), num_classes=3).float()  # [B, 3]

        if isinstance(self.forcing_count_head, ForcingAwareCountHeadV2):
            # V2: no exist_logit — decoder training cannot destabilise count head
            out_cond["count_logit"] = self.forcing_count_head(
                z_cond.detach(),
                u_stats,
                tc_onehot,
            )
        else:
            # V1: reads exist_logit (decoder changes break calibration)
            exist_logit = out_cond["exist_logit"].detach()   # [B, K]
            out_cond["count_logit"] = self.forcing_count_head(
                exist_logit,
                z_cond.detach(),
                u_stats,
                tc_onehot,
            )

        return out_cond, out0

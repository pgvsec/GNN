"""Layer II / III network: Φ_ψ, A_η, U_ω — conditioned fixed-point prediction.

Architecture (Overleaf §8, eq. 1169-1177):

  1. PhiPsi (Φ_ψ): message MLP
       m_k = Φ_ψ(u_k, z_j^0, h_{jb_k}^0, ξ_k)   ∀k ∈ ports
     Inputs: [scalar u_k | graph emb z^0 | node emb h_{b_k}^0 | edge params ξ_k=(T,n,F)]
     Output: m_k ∈ R^m_dim

  2. AEta (A_η): DeepSets aggregator (permutation-invariant over ports)
       M_j = A_η({m_k})
     Produces a single conditioning vector M_j ∈ R^m_dim.

  3. UOmega (U_ω): GNN conditioner
       (z_j^cond, {h_jv^cond}) = U_ω(G_j, z_j^0, {h_jv^0}, M_j, D_φ(z_j^0, {h_jv^0}))
     FiLM-style injection at the entry gene(s); message-passing updates all nodes.
     The autonomous phase portrait prediction D_φ(z^0, h^0) is passed in via
     exist_logit, stab_logit, pos tokens so U_ω can exploit proximity-to-bifurcation info.

  4. CorpusBNet: wraps SccFixedPointNet (frozen) + the three new modules.
     Forward:
       - Run frozen backbone to get z^0, {h^0}, slot_emb (slot_emb not used in II/III
         except for the D_φ prediction already embedded in backbone out dict)
       - Compute Φ_ψ messages for every port
       - Aggregate to M_j
       - Condition via U_ω to get z^cond, {h^cond}
       - Decode conditioned phase portrait via shared D_φ (backbone's SlotDecoder + heads)

Sections
  1. PhiPsi
  2. AEta
  3. UOmega
  4. CorpusBNet
"""
import torch
import torch.nn as nn
import torch.nn.functional as Fnn

from model import SccFixedPointNet, MLP, GraphNetBlock

# Dimension of message vectors passed from Φ_ψ through A_η into U_ω.
# Set equal to d_model so U_ω can add it without a projection at the injection site.
# (Can be overridden via Config.m_dim.)
DEFAULT_M_DIM = 128


# ---------------------------------------------------------------------------
# 1. Φ_ψ — signal MLP
# ---------------------------------------------------------------------------
class PhiPsi(nn.Module):
    """Message MLP for a single (source-branch, target-SCC) port.

    Input vector: [u_k (1) | z_j^0 (d) | h_{jb_k}^0 (d) | T_k (1) | nh_k (1) | F_k (1)]
    Total input dim: 1 + d + d + 3 = 2d + 4
    Output: m_k ∈ R^m_dim
    """

    def __init__(self, d_model, m_dim, n_layers=3, dropout=0.0):
        super().__init__()
        d_in = 2 * d_model + 4    # u, z^0, h_{b}^0, (T, nh, F)
        self.net = MLP(d_in, d_model, m_dim, n_layers=n_layers, dropout=dropout)

    def forward(self, u, z, h_b, port_T, port_nh, port_F):
        """
        u      [B, L]   forcing levels
        z      [B, d]   graph embedding z_j^0
        h_b    [B, L, d] node embeddings of each forced gene b_k
        port_T [B, L]   threshold
        port_nh[B, L]   Hill coefficient
        port_F [B, L]   fold-change
        returns m [B, L, m_dim]
        """
        B, L = u.shape
        d = z.shape[-1]
        # Broadcast z to [B, L, d]
        z_exp = z.unsqueeze(1).expand(B, L, d)
        # Log-scale the Hill edge parameters (same convention as PairEncoder in model.py)
        xi = torch.stack([
            port_T.clamp_min(1e-6).log() / 3.0,
            port_nh / 5.0,
            port_F.clamp_min(1e-6).log() / 5.0,
        ], dim=-1)                                  # [B, L, 3]
        x = torch.cat([u.unsqueeze(-1), z_exp, h_b, xi], dim=-1)   # [B, L, 2d+4]
        return self.net(x)                          # [B, L, m_dim]


# ---------------------------------------------------------------------------
# 2. A_η — DeepSets message aggregator
# ---------------------------------------------------------------------------
class AEta(nn.Module):
    """Permutation-invariant aggregator over a variable-length set of port messages.

    DeepSets: apply a per-element MLP (rho), aggregate with sum (masked), apply
    a global MLP (phi). Handles L=1 (single port) and L>1 (multi-port) uniformly.
    """

    def __init__(self, m_dim, dropout=0.0):
        super().__init__()
        self.rho = MLP(m_dim, m_dim, m_dim, n_layers=2, dropout=dropout)
        self.phi = MLP(m_dim, m_dim, m_dim, n_layers=2, dropout=dropout)
        self.norm = nn.LayerNorm(m_dim)

    def forward(self, m, port_mask):
        """
        m         [B, L, m_dim]  per-port messages
        port_mask [B, L]         True for active ports
        returns   M [B, m_dim]   aggregated conditioning vector
        """
        e = self.rho(m)                                         # [B, L, m_dim]
        msk = port_mask.unsqueeze(-1).float()
        agg = (e * msk).sum(1) / msk.sum(1).clamp_min(1.0)     # [B, m_dim] — masked mean
        return self.norm(self.phi(agg))                         # [B, m_dim]


# ---------------------------------------------------------------------------
# 3. U_ω — FiLM-conditioned GNN update
# ---------------------------------------------------------------------------
class FiLMInjection(nn.Module):
    """FiLM: affine-modulate node embeddings at the entry genes.

    Generates per-node scale (γ) and shift (β) from M_j, applied only at
    the active port genes. Non-port genes are left at their base embeddings.
    """

    def __init__(self, m_dim, d_model):
        super().__init__()
        self.gamma_mlp = nn.Linear(m_dim, d_model)
        self.beta_mlp  = nn.Linear(m_dim, d_model)

    def forward(self, h, M, port_gene, port_mask):
        """
        h         [B, N, d]   node embeddings
        M         [B, m_dim]  aggregated message
        port_gene [B, L]      gene indices to inject into (0-based)
        port_mask [B, L]      True for active ports
        returns   h           modulated node embeddings [B, N, d]
        """
        B, N, d = h.shape
        gamma = 1.0 + self.gamma_mlp(M)     # [B, d] — residual from 1
        beta  = self.beta_mlp(M)            # [B, d]
        # Scatter into entry genes: for each active port, modulate that gene.
        # Multiple ports targeting the same gene accumulate (additive FiLM).
        delta = torch.zeros_like(h)          # [B, N, d]
        msk = port_mask.float()              # [B, L]
        for k_idx in range(port_gene.shape[1]):
            gene_k = port_gene[:, k_idx]    # [B]
            active = msk[:, k_idx]          # [B]
            gene_k_safe = gene_k.clamp(min=0)
            # Per-gene modulation increment: (gamma - 1) * h[gene_k] + beta
            h_k = h[torch.arange(B, device=h.device), gene_k_safe]   # [B, d]
            inc = ((gamma - 1.0) * h_k + beta) * active.unsqueeze(-1)
            # Scatter-add
            idx = gene_k_safe.view(B, 1, 1).expand(B, 1, d)
            delta.scatter_add_(1, idx, inc.unsqueeze(1))
        return h + delta


class UOmega(nn.Module):
    """GNN conditioning update.

    Takes:
      - Base embeddings z^0 [B,d], h^0 [B,N,d]
      - Aggregated message M_j [B, m_dim]
      - Autonomous portrait tokens: exist_logit [B,K], stab_logit [B,K,S], pos [B,K,N]
        (concatenated into a [B, portrait_dim] summary and injected via a small MLP)
    Produces:
      - z^cond [B, d], h^cond [B, N, d]
    """

    def __init__(self, d_model, m_dim, max_fp, n_gnn_layers=2,
                 n_heads=8, dropout=0.0):
        super().__init__()
        from data import NUM_STABILITY, N_MAX

        # Portrait summary: compress exist [K] + stab [K,S] + pos [K,N] → portrait_dim
        portrait_raw_dim = max_fp * (1 + NUM_STABILITY + N_MAX)
        portrait_dim = d_model
        self.portrait_proj = MLP(portrait_raw_dim, d_model, portrait_dim, n_layers=2)

        # Fuse M_j + portrait summary → conditioning signal per node (broadcast)
        self.cond_proj = nn.Linear(m_dim + portrait_dim, d_model)

        # FiLM injection at entry genes
        self.film = FiLMInjection(m_dim, d_model)

        # GNN refinement layers (same architecture as backbone's GraphNetBlock)
        self.gnn_layers = nn.ModuleList([
            GraphNetBlock(d_model, dropout) for _ in range(n_gnn_layers)
        ])
        self.attn_norm = nn.LayerNorm(d_model)

        # Graph-level update (z^cond = z^0 + delta_z from mean-pooled updated nodes)
        self.graph_update = MLP(2 * d_model, d_model, d_model)
        self.graph_norm   = nn.LayerNorm(d_model)

    def forward(self, z0, h0, edge_emb, edge_mask, node_mask, M,
                exist_logit, stab_logit, pos, port_gene, port_mask):
        """
        z0          [B, d]
        h0          [B, N, d]
        edge_emb    [B, N, N, d]   — pair embeddings from backbone (reused, no-grad)
        edge_mask   [B, N, N] bool
        node_mask   [B, N]   bool
        M           [B, m_dim]
        exist_logit [B, K]
        stab_logit  [B, K, S]
        pos         [B, K, N]
        port_gene   [B, L]   int
        port_mask   [B, L]   bool
        """
        B, N, d = h0.shape

        # 1. Portrait summary (detach — not training the backbone)
        # Pad pos to N_MAX along the gene dimension so portrait_raw_dim is constant
        from data import N_MAX as _N_MAX
        K_port = exist_logit.shape[1]
        # Positions of padded genes are sigmoid outputs, not zeros; mask them so the portrait
        # of an SCC does not depend on how far its batch is padded.
        pos = pos * node_mask.unsqueeze(1).to(pos.dtype)
        if pos.shape[-1] < _N_MAX:
            pad = torch.zeros(B, K_port, _N_MAX - pos.shape[-1],
                              device=pos.device, dtype=pos.dtype)
            pos_pad = torch.cat([pos, pad], dim=-1)
        else:
            pos_pad = pos[..., :_N_MAX]
        portrait_flat = torch.cat([
            exist_logit,
            stab_logit.flatten(1),
            pos_pad.flatten(1),
        ], dim=-1).detach()   # [B, portrait_raw_dim]
        portrait = self.portrait_proj(portrait_flat)    # [B, portrait_dim]

        # 2. Node-level conditioning bias from M and portrait (broadcast)
        cond_bias = self.cond_proj(torch.cat([M, portrait], dim=-1))    # [B, d]
        h = h0 + cond_bias.unsqueeze(1)                                  # [B, N, d]

        # 3. FiLM injection at entry genes
        h = self.film(h, M, port_gene, port_mask)

        # 4. GNN refinement (mask out padded nodes/edges)
        nm = node_mask.unsqueeze(-1).float()
        for layer in self.gnn_layers:
            h, _ = layer(h, edge_emb, edge_mask, node_mask)

        # 5. Updated graph embedding: z^cond = LayerNorm(z^0 + MLP([z^0 | mean(h^cond)]))
        mean_h = (h * nm).sum(1) / nm.sum(1).clamp_min(1.0)   # [B, d]
        z_cond = self.graph_norm(z0 + self.graph_update(torch.cat([z0, mean_h], dim=-1)))

        return z_cond, h


# ---------------------------------------------------------------------------
# 4. CorpusBNet
# ---------------------------------------------------------------------------
class CorpusBNet(nn.Module):
    """Full Layer II/III network for Corpus B.

    Wraps a frozen (or optionally fine-tuned) SccFixedPointNet backbone with
    the three new trainable components Φ_ψ, A_η, U_ω.

    Forward:
      1. Run backbone (no_grad by default) → z^0, h^0, edge embeddings, autonomous portrait
      2. Φ_ψ: per-port messages
      3. A_η: aggregate to M_j
      4. U_ω: conditioned embeddings (z^cond, h^cond)
      5. D_φ: shared SlotDecoder + heads on (z^cond, h^cond) → conditioned portrait
    """

    def __init__(self, backbone: SccFixedPointNet, cfg):
        super().__init__()
        self.backbone = backbone
        d  = cfg.d_model
        m  = getattr(cfg, "m_dim", DEFAULT_M_DIM)
        K  = cfg.max_fp
        n_u_layers = getattr(cfg, "n_u_layers", 2)

        self.phi_psi = PhiPsi(d, m, n_layers=3, dropout=cfg.dropout)
        self.a_eta   = AEta(m, dropout=cfg.dropout)
        self.u_omega = UOmega(d, m, K,
                              n_gnn_layers=n_u_layers,
                              n_heads=cfg.n_heads,
                              dropout=cfg.dropout)

    # ------------------------------------------------------------------
    # Internal helper: run backbone and collect intermediate tensors
    # ------------------------------------------------------------------
    def _run_backbone(self, batch, freeze=True):
        """Run backbone forward, returning intermediate tensors needed by Φ_ψ/U_ω.

        Returns:
          out0         : dict from backbone (autonomous portrait + embeddings)
          h0           : node embeddings  [B, N, d]  (float32)
          z0           : graph embedding  [B, d]     (float32)
          edge_emb     : pair embeddings  [B, N, N, d] (float32, from last GNN layer)
          edge_mask    : [B, N, N] bool
        """
        ctx = torch.no_grad() if freeze else torch.enable_grad()
        with ctx:
            # We need the edge embeddings from inside the backbone too.
            # Rather than duplicating backbone code, we call a patched forward
            # that also returns edge_emb. We achieve this by calling internal
            # sub-modules directly on a copy of the computation.
            node_mask = batch["node_mask"]
            adj = batch["adj"]
            edge_mask = adj != 0

            h = self.backbone.node_enc(batch["P"], batch["K"], batch["xmax"], adj, node_mask)
            e = self.backbone.pair_enc(batch["T"], batch["nh"], batch["F"], adj, batch["xmax"])
            for blk in self.backbone.gnn:
                h, e = blk(h, e, edge_mask, node_mask)
            for blk in self.backbone.attn:
                h = blk(h, node_mask)

            g = self.backbone.readout(h, node_mask)
            s = self.backbone.decoder(g, h, node_mask)

        # Run heads in float32 (same as backbone.forward)
        with torch.autocast(device_type=h.device.type, enabled=False):
            sf, hf, gf = s.float(), h.float(), g.float()
            ef = e.float()
            gsf = gf.unsqueeze(1).expand_as(sf)
            exist_logit_0 = self.backbone.exist_head(sf).squeeze(-1)   # [B, K]
            has_pos_fb = batch.get("has_pos_fb",
                torch.ones(gf.shape[0], dtype=torch.bool, device=gf.device))
            out0 = {
                "exist_logit":       exist_logit_0,
                "stab_logit":        self.backbone.stab_head(sf),
                "pos":               torch.sigmoid(self.backbone.pos_head(sf, hf)),
                "charpoly":          self.backbone.charpoly_head(torch.cat([sf, gsf], dim=-1)),
                "count_logit":       self.backbone.count_head(
                    exist_logit_0.detach(), gf.detach(), has_pos_fb),
                "stable_count_logit":self.backbone.stable_count_head(
                    exist_logit_0.detach(), gf.detach(), has_pos_fb),
                "slot_emb": sf,
                "graph_emb": gf,
                "node_emb":  hf,
            }
        return out0, hf, gf, ef, edge_mask

    def forward(self, batch, freeze_backbone=True):
        """
        batch : Corpus B batch dict from collate_b, plus node_mask.
                Must contain port_gene, port_u, port_F, port_T, port_nh, port_mask.
        Returns:
          out_cond : dict with conditioned portrait predictions
                     (same keys as SccFixedPointNet.forward + "out0" for the autonomous portrait)
        """
        # --- Step 1: backbone ---
        out0, h0, z0, edge_emb, edge_mask = self._run_backbone(batch, freeze=freeze_backbone)

        node_mask = batch["node_mask"]
        B, N      = node_mask.shape
        d         = h0.shape[-1]

        port_gene = batch["port_gene"]   # [B, L]
        port_u    = batch["port_u"].float()
        port_F    = batch["port_F"].float()
        port_T    = batch["port_T"].float()
        port_nh   = batch["port_nh"].float()
        port_mask = batch["port_mask"]   # [B, L]
        L         = port_gene.shape[1]

        # --- Step 2: Φ_ψ messages ---
        # Gather node embeddings of forced genes: h_b [B, L, d]
        idx = port_gene.clamp_min(0).clamp_max(N - 1)  # safe indexing
        idx_exp = idx.unsqueeze(-1).expand(B, L, d)
        h_b = torch.gather(h0.unsqueeze(1).expand(B, L, N, d)
                           .reshape(B * L, N, d),
                           1,
                           idx.reshape(B * L, 1).unsqueeze(-1).expand(B * L, 1, d)
                           ).reshape(B, L, d)
        # Simpler equivalent:
        h_b = h0[torch.arange(B, device=h0.device).unsqueeze(1).expand(B, L),
                  idx]    # [B, L, d]

        m_ports = self.phi_psi(port_u, z0, h_b, port_T, port_nh, port_F)   # [B, L, m_dim]

        # --- Step 3: A_η aggregate ---
        M = self.a_eta(m_ports, port_mask)    # [B, m_dim]

        # --- Step 4: U_ω condition ---
        z_cond, h_cond = self.u_omega(
            z0, h0, edge_emb.detach(), edge_mask, node_mask, M,
            out0["exist_logit"].detach(),
            out0["stab_logit"].detach(),
            out0["pos"].detach(),
            port_gene, port_mask)

        # --- Step 5: D_φ decode conditioned portrait ---
        with torch.autocast(device_type=h_cond.device.type, enabled=False):
            s_cond = self.backbone.decoder(z_cond, h_cond, node_mask).float()
            z_cf   = z_cond.float()
            gs_c   = z_cf.unsqueeze(1).expand_as(s_cond)
            nb     = N
            exist_logit_cond = self.backbone.exist_head(s_cond).squeeze(-1)   # [B, K]
            has_pos_fb_cond = batch.get("has_pos_fb",
                torch.ones(z_cf.shape[0], dtype=torch.bool, device=z_cf.device))
            out_cond = {
                "exist_logit":        exist_logit_cond,
                "stab_logit":         self.backbone.stab_head(s_cond),
                "pos":                torch.sigmoid(self.backbone.pos_head(s_cond, h_cond.float())),
                "charpoly":           self.backbone.charpoly_head(torch.cat([s_cond, gs_c], dim=-1)),
                "count_logit":        self.backbone.count_head(
                    exist_logit_cond.detach(), z_cf.detach(), has_pos_fb_cond),
                "stable_count_logit": self.backbone.stable_count_head(
                    exist_logit_cond.detach(), z_cf.detach(), has_pos_fb_cond),
                # Embeddings for downstream use
                "slot_emb":  s_cond,
                "graph_emb": z_cf,
                "node_emb":  h_cond.float(),
                # Exposed for subclasses (e.g. CorpusCNet pairwise head)
                "_z_cond":   z_cf,
                "_h_cond":   h_cond.float(),
            }

        return out_cond, out0

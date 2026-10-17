"""Layer I network: parameterized SCC -> set of fixed points.

Sections
  1. Target transforms (position and characteristic polynomial encodings)
  2. GRN ODE: residual, analytic Jacobian, Newton refinement
  3. Feature encoders
  4. Network blocks
  5. SccFixedPointNet
  6. predict_fixed_points: inference with optional Newton refinement

The network proposes fixed points; the ODE functions in section 2 are used both as a
training signal (residual loss) and as an inference-time polish (Newton refinement), so the
same physics is applied in training and at test time.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as Fnn

from data import N_MAX, NUM_STABILITY

LOG_EPS = 1e-6                    # position encoding: log(x / xmax + LOG_EPS)
LOG_SPAN = -math.log(LOG_EPS)
CHARPOLY_SCALE = 10.0             # symlog charpoly targets are divided by this


# ---------------------------------------------------------------------------
# 1. Target transforms
# ---------------------------------------------------------------------------
def encode_position(x, xmax):
    """Fixed-point coordinate -> t in [0, 1]: log-scaled fraction of the invariant box.

    Fixed points of switch-like GRNs sit at expression levels spanning many decades, so a
    linear fraction of xmax would leave the low states indistinguishable from zero."""
    xn = (x / xmax).clamp_min(0.0)
    t = (torch.log(xn + LOG_EPS) - math.log(LOG_EPS)) / LOG_SPAN
    return t.clamp(0.0, 1.0)


def decode_position(t, xmax):
    xn = torch.exp(t * LOG_SPAN + math.log(LOG_EPS)) - LOG_EPS
    return xn.clamp_min(1e-9) * xmax


def symlog(x):
    return torch.sign(x) * torch.log1p(x.abs())


def inv_symlog(y):
    return torch.sign(y) * torch.expm1(y.abs())


def graph_rate_scale(Kdeg, node_mask_f):
    """Geometric-mean degradation rate per graph, [B]. Sets the time unit of the Jacobian."""
    logk = torch.log(Kdeg) * node_mask_f
    return torch.exp(logk.sum(-1) / node_mask_f.sum(-1).clamp_min(1.0))


def coeff_scale(rate, n_coef):
    """rate^m for m = 1..n_coef, [B, n_coef]. The coefficient c_m has units of rate^m."""
    m = torch.arange(1, n_coef + 1, device=rate.device, dtype=rate.dtype)
    return rate.unsqueeze(1) ** m.unsqueeze(0)


def encode_charpoly(c, rate):
    """c [B, K, N] -> symlog of rate-normalized coefficients, divided by CHARPOLY_SCALE."""
    scale = coeff_scale(rate, c.shape[-1]).unsqueeze(1)
    return symlog(c / scale) / CHARPOLY_SCALE


def decode_charpoly(y, rate):
    scale = coeff_scale(rate, y.shape[-1]).unsqueeze(1)
    return inv_symlog((y * CHARPOLY_SCALE).clamp(-60.0, 60.0)) * scale


# ---------------------------------------------------------------------------
# 2. GRN ODE:   dx_i/dt = P_i * prod_{j->i} H_ji(x_j) - K_i * x_i
#               H = (F r + 1) / (r + 1),   r = (x_j / T)^n
# All functions take K candidate points per graph: x is [B, K, N]; graph tensors are
# [B, N] (P, Kdeg, xmax) or [B, N, N] indexed [source, target] (adj, T, nh, fold).
# ---------------------------------------------------------------------------
def hill_log_terms(x, T, nh, fold):
    """log r and log H for every ordered pair, [B, K, source, target]."""
    logx = torch.log(x.clamp_min(1e-30))
    logr = nh.unsqueeze(1) * (logx.unsqueeze(3) - torch.log(T).unsqueeze(1))
    zero = torch.zeros_like(logr)
    log_h = torch.logaddexp(torch.log(fold).unsqueeze(1) + logr, zero) - torch.logaddexp(logr, zero)
    return logr, log_h


def ode_rhs(x, P, Kdeg, adj, T, nh, fold):
    _, log_h = hill_log_terms(x, T, nh, fold)
    edge = (adj != 0).unsqueeze(1).to(log_h.dtype)
    prod = torch.exp((log_h * edge).sum(dim=2))                       # product over sources
    return P.unsqueeze(1) * prod - Kdeg.unsqueeze(1) * x


def hill_residual(x, P, Kdeg, xmax, adj, T, nh, fold):
    """ODE right-hand side divided by K_i * xmax_i. Zero at a fixed point; |value| <= ~1
    anywhere in the invariant box because P_i * prod max(1, F) <= K_i * xmax_i."""
    return ode_rhs(x, P, Kdeg, adj, T, nh, fold) / (Kdeg * xmax).unsqueeze(1)


def hill_jacobian(x, P, Kdeg, adj, T, nh, fold):
    """Analytic Jacobian [B, K, N, N], J[i, j] = d f_i / d x_j.

    d f_i / d x_j = P_i * prod_{l != j} H_li * H'_ji(x_j) - K_i * delta_ij
    with H'(x) = (F - 1) * n * r / (x * (1 + r)^2)."""
    B, K, N = x.shape
    xc = x.clamp_min(1e-30)
    logr, log_h = hill_log_terms(x, T, nh, fold)
    r = torch.exp(logr)
    h = torch.exp(log_h)
    edge = (adj != 0).unsqueeze(1).to(h.dtype)
    prod = torch.exp((log_h * edge).sum(dim=2))                       # [B, K, target]
    dh = (fold.unsqueeze(1) - 1.0) * nh.unsqueeze(1) * r / (xc.unsqueeze(3) * (1.0 + r) ** 2)
    others = prod.unsqueeze(2) / h                                    # product excluding source j
    jt = P.view(B, 1, 1, N) * others * dh * edge                      # [B, K, source, target]
    eye = torch.eye(N, device=x.device, dtype=x.dtype)
    return jt.transpose(2, 3) - Kdeg.view(B, 1, N, 1) * eye


@torch.no_grad()
def newton_refine(x, P, Kdeg, xmax, adj, T, nh, fold, node_mask, steps):
    """Damped Newton iterations on candidate fixed points, kept inside the invariant box.

    Padded nodes are given an identity Jacobian row so they never move. Singular Jacobians
    produce a zero step instead of an error."""
    if steps <= 0:
        return x
    B, K, N = x.shape
    eye = torch.eye(N, device=x.device, dtype=x.dtype)
    nm = node_mask.unsqueeze(1).to(x.dtype)                           # [B, 1, N]
    pair = (node_mask.unsqueeze(2) & node_mask.unsqueeze(1)).unsqueeze(1).to(x.dtype)
    pad = (~node_mask).to(x.dtype).view(B, 1, N, 1) * eye
    hi = xmax.unsqueeze(1)
    lo = 1e-12 * hi
    for _ in range(steps):
        f = ode_rhs(x, P, Kdeg, adj, T, nh, fold) * nm
        J = hill_jacobian(x, P, Kdeg, adj, T, nh, fold) * pair + pad
        dx, _ = torch.linalg.solve_ex(J, (-f).unsqueeze(-1))
        dx = torch.nan_to_num(dx.squeeze(-1), nan=0.0, posinf=0.0, neginf=0.0) * nm
        lim = 0.5 * hi
        dx = torch.maximum(torch.minimum(dx, lim), -lim)
        x = torch.minimum(torch.maximum(x + dx, lo), hi)
    return x


# ---------------------------------------------------------------------------
# 3. Feature encoders (fixed scaling: usable on new SCCs without dataset statistics)
# ---------------------------------------------------------------------------
NODE_FEATURES = 11
PAIR_FEATURES = 11


class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, n_layers=2, dropout=0.0):
        super().__init__()
        layers = []
        d = d_in
        for _ in range(n_layers - 1):
            layers += [nn.Linear(d, d_hidden), nn.GELU(), nn.Dropout(dropout)]
            d = d_hidden
        layers.append(nn.Linear(d, d_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class NodeEncoder(nn.Module):
    """Per-gene features: log P, log K, log xmax, log P/K, in/out degree by sign, self-loop sign."""

    def __init__(self, d):
        super().__init__()
        self.mlp = MLP(NODE_FEATURES, d, d)

    def forward(self, P, Kdeg, xmax, adj, node_mask):
        valid = node_mask.to(P.dtype)
        act = (adj > 0).to(P.dtype)
        rep = (adj < 0).to(P.dtype)
        logp = torch.log(P)
        logk = torch.log(Kdeg)
        logx = torch.log(xmax)
        n = valid.sum(-1, keepdim=True).expand_as(P)
        feats = torch.stack([
            logp / 5.0, logk / 5.0, logx / 10.0, (logp - logk) / 5.0,
            act.sum(1) / 5.0, rep.sum(1) / 5.0,          # in-degree: sum over sources
            act.sum(2) / 5.0, rep.sum(2) / 5.0,          # out-degree: sum over targets
            torch.diagonal(act, dim1=1, dim2=2),
            torch.diagonal(rep, dim1=1, dim2=2),
            n / N_MAX,
        ], dim=-1)
        return self.mlp(feats) * valid.unsqueeze(-1)


class PairEncoder(nn.Module):
    """Per-pair features for every ordered (source, target); non-edges are masked downstream.

    log F is signed (positive for activation, negative for repression) and log(T / xmax_src)
    says whether the source can reach its own regulatory threshold."""

    def __init__(self, d):
        super().__init__()
        self.mlp = MLP(PAIR_FEATURES, d, d)

    def forward(self, T, nh, fold, adj, xmax):
        act = (adj > 0).to(T.dtype)
        rep = (adj < 0).to(T.dtype)
        logt = torch.log(T)
        logf = torch.log(fold)
        logx_src = torch.log(xmax).unsqueeze(2)
        onehot = Fnn.one_hot(nh.clamp(1, 5).long() - 1, 5).to(T.dtype)
        base = torch.stack([
            act, rep, logt / 3.0, (logt - logx_src) / 10.0, logf / 5.0, nh / 5.0,
        ], dim=-1)
        return self.mlp(torch.cat([base, onehot], dim=-1))


# ---------------------------------------------------------------------------
# 4. Blocks
# ---------------------------------------------------------------------------
class GraphNetBlock(nn.Module):
    """Directed edge-conditioned message passing on dense [B, N, N] pair tensors.

    Edge state e[b, i, j] belongs to the ordered pair (source i, target j). Forward messages
    go source -> target and reverse messages go target -> source, so each gene sees both its
    regulators and the genes it regulates. Sum aggregation matches the multiplicative
    structure of the production term (a product of Hill factors is a sum in log space)."""

    def __init__(self, d, dropout):
        super().__init__()
        self.trunk = MLP(3 * d, d, d)
        self.edge_out = nn.Linear(d, d)
        self.msg_fwd = nn.Linear(d, d)
        self.msg_bwd = nn.Linear(d, d)
        self.node_mlp = MLP(3 * d, d, d)
        self.norm_e = nn.LayerNorm(d)
        self.norm_h = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, h, e, edge_mask, node_mask):
        B, N, d = h.shape
        hs = h.unsqueeze(2).expand(B, N, N, d)
        ht = h.unsqueeze(1).expand(B, N, N, d)
        u = self.trunk(torch.cat([hs, ht, e], dim=-1))
        e = self.norm_e(e + self.drop(self.edge_out(u)))
        em = edge_mask.unsqueeze(-1).to(u.dtype)
        agg_in = (self.msg_fwd(u) * em).sum(dim=1)       # over sources -> [B, target, d]
        agg_out = (self.msg_bwd(u) * em).sum(dim=2)      # over targets -> [B, source, d]
        h = self.norm_h(h + self.drop(self.node_mlp(torch.cat([h, agg_in, agg_out], dim=-1))))
        return h * node_mask.unsqueeze(-1).to(h.dtype), e


class NodeAttentionBlock(nn.Module):
    """Full self-attention over the genes of one SCC. No positional encoding: genes are unordered."""

    def __init__(self, d, heads, dropout):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=heads, dim_feedforward=4 * d, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True)

    def forward(self, h, node_mask):
        h = self.layer(h, src_key_padding_mask=~node_mask)
        return h * node_mask.unsqueeze(-1).to(h.dtype)


class GraphReadout(nn.Module):
    """Mean and max pooling over genes plus the graph size, mixed by an MLP."""

    def __init__(self, d):
        super().__init__()
        self.mlp = MLP(2 * d + 1, d, d)

    def forward(self, h, node_mask):
        m = node_mask.unsqueeze(-1).to(h.dtype)
        mean = (h * m).sum(1) / m.sum(1).clamp_min(1.0)
        mx = h.masked_fill(~node_mask.unsqueeze(-1), -1e4).max(dim=1).values
        n = node_mask.sum(1, keepdim=True).to(h.dtype) / N_MAX
        return self.mlp(torch.cat([mean, mx, n], dim=-1))


class SlotDecoder(nn.Module):
    """K learned slot queries cross-attend to [graph token, gene tokens].

    Self-attention among the slots lets them divide the fixed points between themselves."""

    def __init__(self, d, heads, layers, n_slots, dropout):
        super().__init__()
        self.queries = nn.Embedding(n_slots, d)
        layer = nn.TransformerDecoderLayer(
            d_model=d, nhead=heads, dim_feedforward=4 * d, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, layers, norm=nn.LayerNorm(d))

    def forward(self, g, h, node_mask):
        B = h.shape[0]
        memory = torch.cat([g.unsqueeze(1), h], dim=1)
        no_pad = torch.zeros(B, 1, dtype=torch.bool, device=h.device)
        pad = torch.cat([no_pad, ~node_mask], dim=1)
        q = self.queries.weight.unsqueeze(0).expand(B, -1, -1) + g.unsqueeze(1)
        return self.decoder(q, memory, memory_key_padding_mask=pad)


class SlotNodeHead(nn.Module):
    """Scalar per (slot, gene). Equivariant in gene order and independent of graph size."""

    def __init__(self, d):
        super().__init__()
        self.mlp = MLP(3 * d, d, 1)

    def forward(self, s, h):
        B, K, d = s.shape
        N = h.shape[1]
        ss = s.unsqueeze(2).expand(B, K, N, d)
        hh = h.unsqueeze(1).expand(B, K, N, d)
        return self.mlp(torch.cat([ss, hh, ss * hh], dim=-1)).squeeze(-1)


# ---------------------------------------------------------------------------
# 5. Network
# ---------------------------------------------------------------------------
class SccFixedPointNet(nn.Module):
    """Per-slot outputs
         exist_logit : the slot holds a fixed point
         stab_logit  : stability class (NUM_STABILITY)
         pos         : log-scaled position in the invariant box, per gene, in [0, 1]
         charpoly    : symlog-normalized characteristic polynomial coefficients of the Jacobian
       Per-graph outputs
         count_logit, stable_count_logit : number of fixed points / of stable fixed points
    """

    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        self.node_enc = NodeEncoder(d)
        self.pair_enc = PairEncoder(d)
        self.gnn = nn.ModuleList([GraphNetBlock(d, cfg.dropout) for _ in range(cfg.n_gnn_layers)])
        self.attn = nn.ModuleList(
            [NodeAttentionBlock(d, cfg.n_heads, cfg.dropout) for _ in range(cfg.n_attn_layers)])
        self.readout = GraphReadout(d)
        self.decoder = SlotDecoder(d, cfg.n_heads, cfg.n_dec_layers, cfg.max_fp, cfg.dropout)
        self.exist_head = MLP(d, d, 1)
        self.stab_head = MLP(d, d, NUM_STABILITY)
        self.pos_head = SlotNodeHead(d)
        self.charpoly_head = MLP(2 * d, d, N_MAX)
        self.count_head = MLP(d, d, cfg.max_fp + 1)
        self.stable_count_head = MLP(d, d, cfg.max_fp + 1)

    def forward(self, batch):
        node_mask = batch["node_mask"]
        adj = batch["adj"]
        edge_mask = adj != 0

        h = self.node_enc(batch["P"], batch["K"], batch["xmax"], adj, node_mask)
        e = self.pair_enc(batch["T"], batch["nh"], batch["F"], adj, batch["xmax"])
        for blk in self.gnn:
            h, e = blk(h, e, edge_mask, node_mask)
        for blk in self.attn:
            h = blk(h, node_mask)

        g = self.readout(h, node_mask)
        s = self.decoder(g, h, node_mask)

        # Heads run in float32 even under bf16 autocast: a bf16 position has about 3 significant
        # digits, which is a position error of several percent after the log decoding and would
        # blur the ODE-residual terms of the loss.
        with torch.autocast(device_type=h.device.type, enabled=False):
            s, h, g = s.float(), h.float(), g.float()
            gs = g.unsqueeze(1).expand_as(s)
            # Augment graph embedding with the mean of post-decoder slot embeddings.
            # This lets the count head see what basins the decoder actually found
            # instead of counting purely from pre-decoder graph features.
            g_count = g + s.mean(dim=1)
            return {
                "exist_logit": self.exist_head(s).squeeze(-1),
                "stab_logit": self.stab_head(s),
                "pos": torch.sigmoid(self.pos_head(s, h)),
                "charpoly": self.charpoly_head(torch.cat([s, gs], dim=-1)),
                "count_logit": self.count_head(g_count),
                "stable_count_logit": self.stable_count_head(g_count),
                # Intermediate embeddings exposed for Layer II / III
                "slot_emb": s,    # branch tokens q^(r)  [B, K, d]
                "graph_emb": g,   # z_j^0               [B, d]
                "node_emb": h,    # h_jv^0              [B, N, d]
            }


# ---------------------------------------------------------------------------
# 6. Inference
# ---------------------------------------------------------------------------
def decode_slots(out):
    """Boolean [B, K] mask of the slots that form the predicted fixed-point set.

    The count head says how many fixed points there are and the existence head ranks the
    slots, so the kept set is the k most probable slots, with k the argmax of the count head
    over odd counts. A generic system has an odd number of hyperbolic fixed points (every
    index is +1 or -1 and they sum to 1), so even counts are excluded. Thresholding the
    existence probability is not used: it is overconfident in the middle range and keeps
    duplicate slots, and on held-out graphs it gave 64% exact counts against 79% for the
    count head."""
    logits = out["count_logit"].float()
    even = torch.arange(logits.shape[-1], device=logits.device) % 2 == 0
    k = logits.masked_fill(even, float("-inf")).argmax(dim=-1)
    order = out["exist_logit"].float().argsort(dim=1, descending=True)
    rank = order.argsort(dim=1)                               # position of every slot in the ranking
    return rank < k.unsqueeze(1)


@torch.no_grad()
def predict_fixed_points(model, batch, refine_steps=5):
    """Fixed-point set for each graph in the batch.

    The kept slots are chosen by decode_slots. Their positions are polished
    with `refine_steps` Newton iterations on the true ODE; `residual` is the maximum relative
    ODE residual over genes after polishing (a polished slot with residual below ~1e-6 is a
    genuine fixed point regardless of what the labels said).
    Returns tensors with leading dims [B, K]:
      exist [B,K] bool, x [B,K,N], residual [B,K], stability [B,K] long,
      charpoly [B,K,N] decoded from the network, n_fp [B], n_stable [B]
    """
    model.eval()
    out = {k: v.float() for k, v in model(batch).items()}
    node_mask = batch["node_mask"]
    nm = node_mask.float()
    nb = node_mask.shape[1]
    P, Kd, xmax = batch["P"].float(), batch["K"].float(), batch["xmax"].float()
    adj, T, nh, fold = batch["adj"], batch["T"].float(), batch["nh"].float(), batch["F"].float()

    exist = decode_slots(out)
    x = decode_position(out["pos"], xmax.unsqueeze(1))
    x = newton_refine(x, P, Kd, xmax, adj, T, nh, fold, node_mask, refine_steps)
    res = hill_residual(x, P, Kd, xmax, adj, T, nh, fold).abs() * nm.unsqueeze(1)
    stability = out["stab_logit"].argmax(-1)
    rate = graph_rate_scale(Kd, nm)
    return {
        "exist": exist,
        "x": x,
        "residual": res.max(dim=-1).values,
        "stability": stability,
        "charpoly": decode_charpoly(out["charpoly"][..., :nb], rate),
        "n_fp": exist.sum(1),
        "n_stable": (exist & (stability <= 1)).sum(1),
    }
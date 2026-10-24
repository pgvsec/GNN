"""Training objective and evaluation metrics for Layer I.

Sections
  1. match_slots: Hungarian assignment of target fixed points to decoder slots
  2. SetCriterion: the training loss
  3. batch_metrics: per-sample evaluation numerators and denominators
  4. MetricAccumulator: overall and grouped (n, draw phase, consistency, feedback) aggregation
  5. CalibrationTable: reliability of the slot existence probability
  6. label_residual: ODE residual of the stored labels (data integrity check)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn
from scipy.optimize import linear_sum_assignment

from data import N_MAX
from model import (CHARPOLY_SCALE, LOG_SPAN, decode_charpoly, decode_position,
                   decode_slots, encode_charpoly, encode_position, graph_rate_scale,
                   hill_residual, newton_refine)


# ---------------------------------------------------------------------------
# 1. Matching
# ---------------------------------------------------------------------------
@torch.no_grad()
def match_slots(cost, fp_mask):
    """Optimal one-to-one assignment of target fixed points to decoder slots.

    cost    : [B, K, Kt] float, lower is better
    fp_mask : [B, Kt] bool, valid targets
    returns : [B, K] long, index of the matched target per slot, -1 if the slot is unmatched
    """
    B, K, _ = cost.shape
    cost_np = np.nan_to_num(cost.float().cpu().numpy(), nan=1e6, posinf=1e6, neginf=-1e6)
    counts = fp_mask.sum(1).cpu().numpy()
    out = np.full((B, K), -1, dtype=np.int64)
    for b in range(B):
        kb = int(counts[b])
        if kb == 0:
            continue
        rows, cols = linear_sum_assignment(cost_np[b, :, :kb])
        out[b, rows] = cols
    return torch.from_numpy(out).to(cost.device)


# ---------------------------------------------------------------------------
# 2. Criterion
# ---------------------------------------------------------------------------
class SetCriterion(nn.Module):
    """Set-prediction loss for Layer I.

    exist   BCE on slot existence. When a sample's Brouwer sum is not 1 its label set is
            incomplete (a fixed point was missed by the solver), so an unmatched slot is
            penalized there only if it duplicates a labelled fixed point or has a large ODE
            residual. A slot that satisfies the ODE and is not a duplicate may be a real,
            unlabelled fixed point and is left alone.
    pos     Huber on the log-scaled position, matched slots.
    cp      Huber on the normalized characteristic polynomial coefficients, matched slots.
    stab    Cross-entropy on the stability class, matched slots.
    count   Cross-entropy on the total and stable fixed-point counts, Brouwer-consistent
            samples only (the only samples whose counts can be trusted).
    phys    Huber on the Hill ODE residual at the decoded position, matched slots.
    thomas  Expected number of stable fixed points above 1 when the graph has no positive
            feedback loop (Thomas necessary condition).
    The Brouwer index is not a loss term: sgn(det(-J)) has zero gradient. It is evaluated
    on predictions in batch_metrics.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, out, batch):
        with torch.autocast(device_type=batch["P"].device.type, enabled=False):
            return self._compute(out, batch)

    @staticmethod
    def _matched_mean(per_slot, mw):
        """Mean over matched slots within each sample, then over the batch."""
        return ((per_slot * mw).sum(1) / mw.sum(1).clamp_min(1.0)).mean()

    @torch.no_grad()
    def _match(self, exist_logit, stab_logit, pos, cp, tgt_pos, tgt_cp, tgt_stab, fp_mask, nm):
        cfg = self.cfg
        K = pos.shape[1]
        m = nm[:, None, None, :]
        den = m.sum(-1).clamp_min(1.0)
        c_pos = ((pos.unsqueeze(2) - tgt_pos.unsqueeze(1)).abs() * m).sum(-1) / den
        c_cp = ((cp.unsqueeze(2) - tgt_cp.unsqueeze(1)).abs() * m).sum(-1) / den
        p_exist = torch.sigmoid(exist_logit)
        prob = torch.softmax(stab_logit, dim=-1)
        idx = tgt_stab.clamp_min(0).unsqueeze(1).expand(-1, K, -1)
        p_cls = torch.gather(prob, 2, idx)
        cost = (-cfg.cost_exist * p_exist.unsqueeze(2) + cfg.cost_pos * c_pos
                + cfg.cost_cp * c_cp - cfg.cost_stab * p_cls)
        slot_target = match_slots(cost, fp_mask)
        dist_min = c_pos.masked_fill(~fp_mask.unsqueeze(1), float("inf")).min(dim=2).values
        return slot_target, dist_min

    def _compute(self, out, batch):
        cfg = self.cfg
        out = {k: v.float() for k, v in out.items()}
        P = batch["P"].float()
        Kd = batch["K"].float()
        xmax = batch["xmax"].float()
        T = batch["T"].float()
        nh = batch["nh"].float()
        fold = batch["F"].float()
        adj = batch["adj"]
        node_mask = batch["node_mask"]
        nm = node_mask.float()
        n_valid = nm.sum(-1).clamp_min(1.0)
        fp_mask = batch["fp_mask"]
        B, nb = node_mask.shape
        K = out["exist_logit"].shape[1]

        rate = graph_rate_scale(Kd, nm)
        tgt_pos = encode_position(batch["fp_state"].float(), xmax.unsqueeze(1))
        tgt_cp = encode_charpoly(batch["fp_charpoly"].float(), rate)
        tgt_stab = batch["fp_stab"]

        pos = out["pos"]
        cp = out["charpoly"][..., :nb]
        exist_logit = out["exist_logit"]
        stab_logit = out["stab_logit"]

        # slot <-> target assignment, then targets arranged in slot order
        slot_target, dist_min = self._match(
            exist_logit.detach(), stab_logit.detach(), pos.detach(), cp.detach(),
            tgt_pos, tgt_cp, tgt_stab, fp_mask, nm)
        matched = slot_target >= 0
        mw = matched.float()
        idx = slot_target.clamp_min(0)
        a_pos = torch.gather(tgt_pos, 1, idx.unsqueeze(-1).expand(-1, -1, nb))
        a_cp = torch.gather(tgt_cp, 1, idx.unsqueeze(-1).expand(-1, -1, nb))
        a_stab = torch.gather(tgt_stab, 1, idx).clamp_min(0)

        # physics: ODE residual at the decoded position of every slot
        # Forced samples (Corpus C) carry P_phys = P * u: the stored fixed points belong to
        # the forced system, so the residual must be taken there, not at the unforced P.
        x_hat = decode_position(pos, xmax.unsqueeze(1))
        P_res = batch["P_phys"].float() if "P_phys" in batch else P
        res = hill_residual(x_hat, P_res, Kd, xmax, adj, T, nh, fold)
        res_slot = (res.abs() * nm.unsqueeze(1)).sum(-1) / n_valid.unsqueeze(1)

        def slot_mean(per_node):
            per_slot = (per_node * nm.unsqueeze(1)).sum(-1) / n_valid.unsqueeze(1)
            return self._matched_mean(per_slot, mw)

        beta = cfg.huber_beta
        l_pos = slot_mean(Fnn.smooth_l1_loss(pos, a_pos, beta=beta, reduction="none"))
        l_cp = slot_mean(Fnn.smooth_l1_loss(cp, a_cp, beta=beta, reduction="none"))
        l_phys = slot_mean(Fnn.smooth_l1_loss(res, torch.zeros_like(res), beta=beta, reduction="none"))
        ce = Fnn.cross_entropy(
            stab_logit.reshape(B * K, -1), a_stab.reshape(-1), reduction="none").view(B, K)
        l_stab = self._matched_mean(ce, mw)

        # existence with residual-gated negatives on incomplete label sets
        bce = Fnn.binary_cross_entropy_with_logits(exist_logit, mw, reduction="none")
        cons = batch["consistent"].float().unsqueeze(1)
        true_negative = ((res_slot.detach() > cfg.gate_tau) | (dist_min < cfg.dup_tol)).float()
        neg_w = cfg.eos_coef * (cons + (1.0 - cons) * true_negative)
        w = torch.where(matched, torch.ones_like(neg_w), neg_w)
        l_exist = ((bce * w).sum(1) / w.sum(1).clamp_min(1e-6)).mean()

        # counts, trusted only where the Brouwer sum is 1
        cons1 = batch["consistent"].float()
        # Clamp to the count_logit output range (not to slot count K), so that
        # models with an extended count head (e.g. PairwiseCountHead in L3) can
        # supervise k values above K.  For L1/L2, count_logit has K+1 outputs so
        # this equals the old behaviour.
        k_true = batch["k"].clamp(max=out["count_logit"].shape[-1] - 1)
        stable_true = ((tgt_stab == 0) | (tgt_stab == 1)).sum(1).clamp(max=K)
        ce_k = Fnn.cross_entropy(out["count_logit"], k_true, reduction="none")
        ce_s = Fnn.cross_entropy(out["stable_count_logit"], stable_true, reduction="none")

        # Focal reweighting of count CE: (1 - p_correct)^gamma up-weights hard examples.
        # gamma=0 recovers plain CE. Useful when many Brouwer-consistent samples are
        # "easy" (k=1) and the model already gets them right, leaving the hard multi-basin
        # cases under-represented in the gradient.
        focal_gamma = getattr(cfg, "count_focal_gamma", 0.0)
        if focal_gamma > 0.0:
            with torch.no_grad():
                pt_k = torch.softmax(out["count_logit"].detach(), dim=-1).gather(
                    1, k_true.unsqueeze(1)).squeeze(1).clamp(1e-6, 1.0)
                focal_w = (1.0 - pt_k).pow(focal_gamma)
            ce_k = focal_w * ce_k

        kscale = getattr(cfg, "w_count_kscale", 0.0)
        k_wt = k_true.float() * kscale + 1.0  # 1.0 when kscale=0 (flat); k*s+1 otherwise
        denom = (cons1 * k_wt).sum().clamp_min(1.0)
        l_count = ((ce_k + ce_s) * cons1 * k_wt).sum() / denom

        # Existence soft-count consistency: force sigmoid(exist_logit).sum() ≈ k_true.
        # Unlike the slot-by-slot BCE, this is a global constraint that simultaneously
        # penalises over-activating empty slots AND suppresses the mean activation when
        # the model is counting too high.  Applied only to Brouwer-consistent samples
        # where k_true is reliable.
        soft_cnt = torch.sigmoid(exist_logit).sum(1)   # [B]
        n_cons = cons1.sum().clamp_min(1.0)
        l_exist_cnt = ((soft_cnt - k_true.float()).pow(2) * cons1).sum() / n_cons

        # Thomas: no positive feedback loop => at most one stable fixed point
        p_stable = torch.softmax(stab_logit, dim=-1)[..., :2].sum(-1)
        exp_stable = (torch.sigmoid(exist_logit) * p_stable).sum(1)
        nopos = (~batch["has_pos_fb"]).float()
        l_thomas = (torch.relu(exp_stable - 1.0) * nopos).sum() / nopos.sum().clamp_min(1.0)

        w_exist_cnt = getattr(cfg, "w_exist_cnt", 0.0)
        total = (cfg.w_exist * l_exist + cfg.w_pos * l_pos + cfg.w_cp * l_cp
                 + cfg.w_stab * l_stab + cfg.w_count * l_count
                 + cfg.w_phys * l_phys + cfg.w_thomas * l_thomas
                 + w_exist_cnt * l_exist_cnt)

        logs = {
            "total": total.detach(), "exist": l_exist.detach(), "pos": l_pos.detach(),
            "cp": l_cp.detach(), "stab": l_stab.detach(), "count": l_count.detach(),
            "phys": l_phys.detach(), "thomas": l_thomas.detach(),
            "exist_cnt": l_exist_cnt.detach(),
        }
        aux = {
            "matched": matched, "a_pos": a_pos.detach(), "a_cp": a_cp.detach(),
            "a_stab": a_stab, "res_slot": res_slot.detach(), "rate": rate,
        }
        return total, logs, aux


# ---------------------------------------------------------------------------
# 3. Metrics
# ---------------------------------------------------------------------------
@torch.no_grad()
def batch_metrics(out, batch, aux, refine_steps=0):
    """Per-sample (numerator, denominator) tensors of shape [B] for every metric.

    Count metrics use Brouwer-consistent samples only (complete label sets).
    Matched-slot metrics use every sample: a matched label is a real fixed point even when
    the label set is incomplete.

    Every slot metric describes the set chosen by decode_slots (the k most probable slots, k from
    the odd-restricted count head).

      count_exact / count_abs_err        decoded set size vs labelled count
      count_head_exact                   plain argmax of the count head vs labelled count
      stable_count_exact                 stable fixed points implied by the decoded set
      stable_count_head_exact            argmax of the stable-count head
      fp_recall                          matched slots that are in the decoded set
      slot_precision_consistent          decoded slots that matched a label (consistent only)
      count_exact_t03 / count_exact_t07  size of the set of slots above existence probability
                                         0.3 / 0.7, kept for comparison with thresholding
      regime_acc                         mono / bi / multistable (stable count clipped at 3)
      stab_acc                           stability class of matched slots
      stable_binary_acc                  stable vs not stable for matched slots
      pos_decades                        mean position error of matched slots in decades
      fp_within_half_decade              matched slots located within half a decade per gene
      cp_symlog_err                      characteristic polynomial error, symlog units
      brouwer_idx_acc                    sign(c_n) = Brouwer index of matched slots
      phys_residual                      ODE residual at the network's positions, matched slots
      brouwer_consistent                 predicted set has sum sgn(c_n) = 1, c_n = det(-J)
      thomas_violation                   no positive loop but more than one predicted stable FP
      refined_converged                  predicted slots whose Newton-refined residual < 1e-6
    """
    out = {k: v.float() for k, v in out.items()}
    node_mask = batch["node_mask"]
    nm = node_mask.float()
    B, nb = node_mask.shape
    K = out["exist_logit"].shape[1]
    n_valid = nm.sum(-1).clamp_min(1.0)

    cons = batch["consistent"]
    consf = cons.float()
    matched = aux["matched"]
    mf = matched.float()

    pred_exist = decode_slots(out)
    pred_cls = out["stab_logit"].argmax(-1)
    n_pred = pred_exist.sum(1)
    n_pred_stable = (pred_exist & (pred_cls <= 1)).sum(1)
    tgt_stab = batch["fp_stab"]
    stable_true = ((tgt_stab == 0) | (tgt_stab == 1)).sum(1)
    k_true = batch["k"]
    nopos = (~batch["has_pos_fb"]).float()

    def slot_error(pred, tgt):
        """Mean absolute error over genes for every slot, [B, K]."""
        return ((pred - tgt).abs() * nm.unsqueeze(1)).sum(-1) / n_valid.unsqueeze(1)

    def slot_abs(pred, tgt):
        return (slot_error(pred, tgt) * mf).sum(1)

    cp_pred = out["charpoly"][..., :nb]
    cp_dec = decode_charpoly(cp_pred, aux["rate"])
    n_idx = (batch["n"] - 1).view(B, 1, 1).expand(-1, K, 1)
    c_n = torch.gather(cp_dec, 2, n_idx).squeeze(-1)             # c_n = det(-J) = Brouwer index sign
    a_cn = torch.gather(aux["a_cp"], 2, n_idx).squeeze(-1)       # symlog keeps the sign
    brouwer_sum = (torch.sign(c_n) * pred_exist.float()).sum(1)

    probs = torch.sigmoid(out["exist_logit"])
    n_low = (probs > 0.3).sum(1)
    n_high = (probs > 0.7).sum(1)
    regime_pred = n_pred_stable.clamp(max=3)
    regime_true = stable_true.clamp(max=3)
    pos_decades = slot_error(out["pos"], aux["a_pos"]) * LOG_SPAN / 2.302585092994046

    n_matched = mf.sum(1)
    pairs = {
        "count_exact": (((n_pred == k_true) & cons).float(), consf),
        "count_exact_t03": (((n_low == k_true) & cons).float(), consf),
        "count_exact_t07": (((n_high == k_true) & cons).float(), consf),
        "regime_acc": (((regime_pred == regime_true) & cons).float(), consf),
        "stable_binary_acc": (
            (((pred_cls <= 1) == (aux["a_stab"] <= 1)) & matched).sum(1).float(), n_matched),
        "fp_within_half_decade": (((pos_decades < 0.5) & matched).sum(1).float(), n_matched),
        "brouwer_idx_acc": (((torch.sign(c_n) == torch.sign(a_cn)) & matched).sum(1).float(), n_matched),
        "count_abs_err": ((n_pred - k_true).abs().float() * consf, consf),
        "count_head_exact": (((out["count_logit"].argmax(-1) == k_true) & cons).float(), consf),
        "stable_count_exact": (((n_pred_stable == stable_true) & cons).float(), consf),
        "stable_count_head_exact": (
            ((out["stable_count_logit"].argmax(-1) == stable_true) & cons).float(), consf),
        "fp_recall": ((pred_exist & matched).sum(1).float(), n_matched),
        "slot_precision_consistent": (
            (pred_exist & matched).sum(1).float() * consf, pred_exist.sum(1).float() * consf),
        "stab_acc": (((pred_cls == aux["a_stab"]) & matched).sum(1).float(), n_matched),
        "pos_decades": (slot_abs(out["pos"], aux["a_pos"]) * LOG_SPAN / 2.302585092994046, n_matched),
        "cp_symlog_err": (slot_abs(cp_pred, aux["a_cp"]) * CHARPOLY_SCALE, n_matched),
        "phys_residual": ((aux["res_slot"] * mf).sum(1), n_matched),
        "brouwer_consistent": ((brouwer_sum == 1).float(), torch.ones(B, device=nm.device)),
        "thomas_violation": (((n_pred_stable > 1).float() * nopos), nopos),
    }

    if refine_steps > 0:
        xmax = batch["xmax"].float()
        P, Kd = batch["P"].float(), batch["K"].float()
        adj, T = batch["adj"], batch["T"].float()
        nh, fold = batch["nh"].float(), batch["F"].float()
        x = decode_position(out["pos"], xmax.unsqueeze(1))
        if "P_phys" in batch:
            P, xmax = batch["P_phys"].float(), batch["xmax_phys"].float()
        x = newton_refine(x, P, Kd, xmax, adj, T, nh, fold, node_mask, refine_steps)
        res = hill_residual(x, P, Kd, xmax, adj, T, nh, fold).abs() * nm.unsqueeze(1)
        converged = res.max(dim=-1).values < 1e-6
        pairs["refined_converged"] = (
            (pred_exist & converged).sum(1).float(), pred_exist.sum(1).float())
    return pairs


# ---------------------------------------------------------------------------
# 4. Aggregation
# ---------------------------------------------------------------------------
class MetricAccumulator:
    """Accumulates per-sample numerators / denominators, overall and per group.

    Groups: "n" (graph size), "phase" (draw phase 0 broad, 1 multistability-biased,
    2 sweep-guided), "consistent" (Brouwer sum equal to 1) and "has_pos_fb" (graph has a
    positive feedback loop). Overall values are ratios of totals, not means of batch means."""

    GROUP_SIZES = {"n": N_MAX + 1, "phase": 3, "consistent": 2, "has_pos_fb": 2}

    def __init__(self):
        self.num = {}
        self.den = {}

    def update(self, pairs, batch):
        labels = {g: batch[g].detach().cpu().numpy().astype(np.int64) for g in self.GROUP_SIZES}
        for name, (num, den) in pairs.items():
            num = num.detach().float().cpu().numpy().astype(np.float64)
            den = den.detach().float().cpu().numpy().astype(np.float64)
            for g, size in self.GROUP_SIZES.items():
                key = (name, g)
                n_part = np.bincount(labels[g], weights=num, minlength=size)
                d_part = np.bincount(labels[g], weights=den, minlength=size)
                self.num[key] = self.num.get(key, np.zeros(size)) + n_part
                self.den[key] = self.den.get(key, np.zeros(size)) + d_part

    def compute(self):
        """Returns (overall {metric: value}, grouped {metric: {group: {value: ratio}}})."""
        overall, grouped = {}, {}
        for (name, g), nu in self.num.items():
            de = self.den[(name, g)]
            if g == "n":
                overall[name] = float(nu.sum() / max(de.sum(), 1e-9))
            ratios = {int(v): float(nu[v] / de[v]) for v in range(len(nu)) if de[v] > 0}
            grouped.setdefault(name, {})[g] = ratios
        return overall, grouped


# ---------------------------------------------------------------------------
# 5. Calibration
# ---------------------------------------------------------------------------
class CalibrationTable:
    """Reliability of the slot existence probability.

    Only Brouwer-consistent samples are used, since their label sets are complete and an
    unmatched slot there is a genuine false positive. Bin b holds slots with probability in
    [b / bins, (b + 1) / bins). ECE is the count-weighted mean gap between the mean predicted
    probability and the matched fraction; use the table to choose the existence threshold."""

    def __init__(self, bins=10):
        self.bins = bins
        self.count = np.zeros(bins)
        self.positive = np.zeros(bins)
        self.prob_sum = np.zeros(bins)

    @torch.no_grad()
    def update(self, out, aux, batch):
        p = torch.sigmoid(out["exist_logit"].float())
        use = batch["consistent"].unsqueeze(1).expand_as(p)
        if not use.any():
            return
        b = (p * self.bins).long().clamp(max=self.bins - 1)[use].cpu().numpy()
        y = aux["matched"].float()[use].cpu().numpy()
        q = p[use].cpu().numpy().astype(np.float64)
        self.count += np.bincount(b, minlength=self.bins)
        self.positive += np.bincount(b, weights=y, minlength=self.bins)
        self.prob_sum += np.bincount(b, weights=q, minlength=self.bins)

    def compute(self):
        total = max(self.count.sum(), 1.0)
        rows, ece = [], 0.0
        for i in range(self.bins):
            if self.count[i] == 0:
                continue
            mean_p = self.prob_sum[i] / self.count[i]
            frac = self.positive[i] / self.count[i]
            ece += self.count[i] / total * abs(mean_p - frac)
            rows.append({"bin": i, "count": int(self.count[i]), "mean_prob": float(mean_p),
                         "matched_fraction": float(frac)})
        return {"ece": float(ece), "bins": rows}


# ---------------------------------------------------------------------------
# 6. Label integrity
# ---------------------------------------------------------------------------
@torch.no_grad()
def label_residual(batch):
    """Relative ODE residual at every stored fixed point under the stored parameters.

    Returns a flat vector over valid (fixed point, gene) entries. Values near single-precision
    round-off mean the labels are true fixed points of the model being trained and that edge
    parameters are aligned with fixed points; large values flag corrupted parameters."""
    res = hill_residual(
        batch["fp_state"].float().clamp_min(1e-30), batch.get("P_phys", batch["P"]).float(),
        batch["K"].float(),
        batch["xmax"].float(), batch["adj"], batch["T"].float(), batch["nh"].float(),
        batch["F"].float())
    valid = batch["fp_mask"].unsqueeze(-1) & batch["node_mask"].unsqueeze(1)
    return res.abs()[valid]
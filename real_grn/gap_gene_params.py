"""
Drosophila gap gene network parameters for GNN testing.

Two topologies (reference: Gap_Gene_Network_Reference.md):

  Model A — Reinitz/Manu 2009: single 4-gene SCC {Hb, Kr, Gt, Kni}
    Contains Kr⊣Hb (THE CRITICAL EDGE), Hb autoactivation.
    Expected: 2–4 coexisting attractors at central AP positions.

  Model B — Papatsenko-Levine 2011: two 2-gene toggle switches
    SCC_A = {Hb, Kni}  (mutual repressor, root SCC)
    SCC_B = {Kr, Gt}   (mutual repressor, forced by Hb from SCC_A)
    Kr⊣Hb is OMITTED — this is what makes it 2-SCC, not 1-SCC.
    Expected: product bistability (up to 4 attractors: 2×2 product).

Parameterisation:
  Hill-function ODE: dx_i/dt = P_i * prod_j H_{ji}(x_j) - K_i * x_i
  H(x, T, n, F) = (F * r + 1) / (r + 1),   r = (x/T)^n
  F > 1: activator,  F < 1: repressor.

AP-position encoding:
  External inputs (Bcd morphogen, Cad morphogen, Tll) drive differential gene
  production along the anterior-posterior axis.  Rather than computing P_eff
  from explicit Hill-function activations (which compounds large fold-changes
  and breaks internal toggle calibration), we encode the AP-position effect
  directly as EFFECTIVE PRODUCTION RATES P[gene] per position.

  This is biologically honest: at a given AP position, each gap gene has a net
  production rate determined by all external inputs at that position.  The P
  values below are chosen so that:
    1) All genes have P > 1.9  (required for bistability with T_rep=0.3)
    2) P ratios match the known biology (Bcd gradient → Hb/Kr high anteriorly;
       Cad gradient → Kni/Gt high posteriorly)
    3) T_rep = 0.3 gives clean bistability: x_on ≈ P/K ≈ 2–4,
       x_off = F_rep * P/K ≈ 0.01–0.04 ≪ T_rep

  Three AP positions tested:
    37% EL — Hb/Kr boundary (anterior): Bcd dominant → Hb(3.0), Kr(2.5)
    45% EL — Kr/Kni boundary (central): balanced → all ≈ 2.5
    55% EL — Kni/Gt domain (posterior): Cad dominant → Kni(3.5), Gt(3.0)

Node ordering:
  Model A: [0=Hb, 1=Kr, 2=Gt, 3=Kni]  (classical order)
  Model B: [0=Hb, 1=Kni, 2=Kr, 3=Gt]  (composition format: SCC_A first)

Bistability check (symmetric toggle with P=2, T=0.3, n=4, F=0.01):
  x_on ≈ P/K = 2.0
  x_off = F * P/K = 0.02   (fully repressed by partner)
  T_rep = 0.3 : 0.02 ≪ 0.3 ≪ 2.0  → clean bistable toggle ✓
  Critical threshold: P/K > 2T/F^(1/n) = 2*0.3/0.316 ≈ 1.90  → P=2.0 > 1.90 ✓
"""

import numpy as np

# ---------------------------------------------------------------------------
# 1. AP-position parameters: effective production rates per gene
# ---------------------------------------------------------------------------
# Key biology:
#   37% EL: near Hb/Kr boundary; Bcd still strong → Hb and Kr activated
#   45% EL: near Kr/Kni boundary; Bcd weaker, Cad growing → balanced
#   55% EL: deep in Kr/Kni overlap; Cad dominates → Kni/Gt activated

AP_POSITIONS = {
    "37pct": {
        "label": "37% EL — Hb/Kr boundary (anterior)",
        "P_genes": {"Hb": 3.0, "Kr": 2.5, "Gt": 2.0, "Kni": 2.0},
    },
    "45pct": {
        "label": "45% EL — Kr/Kni boundary (central)",
        "P_genes": {"Hb": 2.0, "Kr": 2.5, "Gt": 2.0, "Kni": 2.5},
    },
    "55pct": {
        "label": "55% EL — Kni/Gt domain (posterior)",
        "P_genes": {"Hb": 2.0, "Kr": 2.0, "Gt": 3.0, "Kni": 3.5},
    },
}

K_BASE = 1.0  # degradation rate (all genes)

# ---------------------------------------------------------------------------
# 2. Internal kinetics
# ---------------------------------------------------------------------------
# Strong mutual repression — both toggle switches, all confident gap–gap edges
T_REP    = 0.30  # half-saturation threshold
N_REP    = 4     # Hill coefficient (cooperative; n≥2 for bistability)
F_REP    = 0.01  # fold-change at saturation (1% residual production)

# Weak / disputed edges (Model A only: Kr⊣Hb, Kni⊣Kr)
# Jaeger 2011: "weak, possibly redundant"; Ashyraliyev 2009: T_Kr,hb ≈ 0 in 37/39 fitted solutions
T_WEAK   = 0.80  # higher threshold → weaker effective repression at typical expression
N_WEAK   = 3
F_WEAK   = 0.15  # mild repressor; significantly less than strong (F=0.01)

# Cross-module feedforward: Hb → Kr (confirmed strong), Hb → Gt (confirmed, slightly weaker)
T_CROSS_KR = 0.60  # threshold for Hb repression of Kr
N_CROSS_KR = 3
F_CROSS_KR = 0.01  # strong repression (same strength as toggle repressions)

T_CROSS_GT = 0.60  # threshold for Hb repression of Gt
N_CROSS_GT = 3
F_CROSS_GT = 0.05  # moderate repression (Gt can survive partial Hb expression)

# Hb autoactivation (P1 promoter; confirmed strong in Jaeger 2011)
T_SELF_HB  = 0.60
N_SELF_HB  = 3
F_SELF_HB  = 2.5   # 2.5× fold-gain at saturation; keeps Hb-ON robust without over-dominating

# ---------------------------------------------------------------------------
# 3. Topology / parameter builders
# ---------------------------------------------------------------------------
def _make_par(N):
    return (
        np.zeros((N, N), dtype=np.float32),   # adj
        np.ones((N, N), dtype=np.float64),    # T
        np.ones((N, N), dtype=np.float64),    # nh
        np.ones((N, N), dtype=np.float64),    # F
    )


def _edge(adj, T, nh, F, src, tgt, sign, t, n, f):
    adj[src, tgt] = sign
    T[src, tgt]   = t
    nh[src, tgt]  = n
    F[src, tgt]   = f


# ── Model A ─────────────────────────────────────────────────────────────────
# Node order: [0=Hb, 1=Kr, 2=Gt, 3=Kni]

MODEL_A_GENE_NAMES = ["Hb", "Kr", "Gt", "Kni"]
MODEL_A_IDX = {g: i for i, g in enumerate(MODEL_A_GENE_NAMES)}


def build_model_a(ap_key: str):
    """Model A (Reinitz/Manu 2009): single 4-gene SCC {Hb, Kr, Gt, Kni}.

    All confirmed gap–gap edges + Kr⊣Hb (THE CRITICAL EDGE, weak) + Hb autoactivation.
    Returns (P, K, adj, T, nh, F) with shapes [4], [4], [4,4], [4,4], [4,4], [4,4].
    """
    p_map = AP_POSITIONS[ap_key]["P_genes"]
    idx   = MODEL_A_IDX

    P   = np.array([p_map[g] for g in MODEL_A_GENE_NAMES], dtype=np.float64)
    K   = np.full(4, K_BASE, dtype=np.float64)
    adj, T, nh, Fm = _make_par(4)

    # {Hb(0), Kni(3)} mutual repression — strong toggle
    _edge(adj, T, nh, Fm, idx["Hb"],  idx["Kni"], -1, T_REP, N_REP, F_REP)
    _edge(adj, T, nh, Fm, idx["Kni"], idx["Hb"],  -1, T_REP, N_REP, F_REP)

    # {Kr(1), Gt(2)} mutual repression — strong toggle
    _edge(adj, T, nh, Fm, idx["Kr"], idx["Gt"], -1, T_REP, N_REP, F_REP)
    _edge(adj, T, nh, Fm, idx["Gt"], idx["Kr"], -1, T_REP, N_REP, F_REP)

    # Cross-module edges from Hb to {Kr, Gt} (confirmed strong)
    _edge(adj, T, nh, Fm, idx["Hb"], idx["Kr"], -1, T_CROSS_KR, N_CROSS_KR, F_CROSS_KR)
    _edge(adj, T, nh, Fm, idx["Hb"], idx["Gt"], -1, T_CROSS_GT, N_CROSS_GT, F_CROSS_GT)

    # THE CRITICAL EDGE: Kr⊣Hb — weak, creates the single 4-gene SCC
    # (absent in Model B; this edge is what makes A a 1-SCC, B a 2-SCC)
    _edge(adj, T, nh, Fm, idx["Kr"], idx["Hb"], -1, T_WEAK, N_WEAK, F_WEAK)

    # Weak: Kni⊣Kr (disputed, present in some fitted circuits)
    _edge(adj, T, nh, Fm, idx["Kni"], idx["Kr"], -1, T_WEAK, N_WEAK, F_WEAK)

    # Hb autoactivation via P1 promoter (confirmed strong)
    _edge(adj, T, nh, Fm, idx["Hb"], idx["Hb"], +1, T_SELF_HB, N_SELF_HB, F_SELF_HB)

    return P, K, adj, T, nh, Fm


# ── Model B ─────────────────────────────────────────────────────────────────
# Node order: [0=Hb, 1=Kni, 2=Kr, 3=Gt]  (composition format: SCC_A first)

MODEL_B_GENE_NAMES = ["Hb", "Kni", "Kr", "Gt"]
MODEL_B_IDX = {g: i for i, g in enumerate(MODEL_B_GENE_NAMES)}


def build_model_b(ap_key: str):
    """Model B (Papatsenko-Levine 2011): two 2-gene toggle switches.

    SCC_A = {Hb(0), Kni(1)} — mutual repressor (root SCC, no forcing)
    SCC_B = {Kr(2), Gt(3)}  — mutual repressor (forced by Hb from SCC_A)
    Kr⊣Hb OMITTED — this is what makes it 2-SCC, not 1-SCC.
    Hb autoactivation INCLUDED (Papatsenko-Levine includes it in their model).
    Returns (P, K, adj, T, nh, F) with shapes [4], [4], [4,4], [4,4], [4,4], [4,4].
    """
    p_map = AP_POSITIONS[ap_key]["P_genes"]
    idx   = MODEL_B_IDX

    P   = np.array([p_map[g] for g in MODEL_B_GENE_NAMES], dtype=np.float64)
    K   = np.full(4, K_BASE, dtype=np.float64)
    adj, T, nh, Fm = _make_par(4)

    # SCC_A: {Hb(0), Kni(1)} mutual repression
    _edge(adj, T, nh, Fm, idx["Hb"],  idx["Kni"], -1, T_REP, N_REP, F_REP)
    _edge(adj, T, nh, Fm, idx["Kni"], idx["Hb"],  -1, T_REP, N_REP, F_REP)

    # SCC_B: {Kr(2), Gt(3)} mutual repression
    _edge(adj, T, nh, Fm, idx["Kr"], idx["Gt"], -1, T_REP, N_REP, F_REP)
    _edge(adj, T, nh, Fm, idx["Gt"], idx["Kr"], -1, T_REP, N_REP, F_REP)

    # Cross-edges SCC_A → SCC_B: Hb represses Kr and Gt
    # Tarjan SCC: no B→A edges → clean 2-SCC structure
    _edge(adj, T, nh, Fm, idx["Hb"], idx["Kr"], -1, T_CROSS_KR, N_CROSS_KR, F_CROSS_KR)
    _edge(adj, T, nh, Fm, idx["Hb"], idx["Gt"], -1, T_CROSS_GT, N_CROSS_GT, F_CROSS_GT)

    # Hb autoactivation (Papatsenko-Levine include this)
    _edge(adj, T, nh, Fm, idx["Hb"], idx["Hb"], +1, T_SELF_HB, N_SELF_HB, F_SELF_HB)

    # Kr⊣Hb is intentionally OMITTED
    # Kni⊣Kr is intentionally OMITTED (Papatsenko minimalist design)

    return P, K, adj, T, nh, Fm

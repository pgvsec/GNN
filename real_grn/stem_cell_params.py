"""
Pluripotency → Hematopoiesis → Lymphocyte activation
Three-SCC depth-3 network for GNN stress-testing.

Network structure (DAG depth = 3):
  SCC₁ — Pluripotency triad:         OCT4 (POU5F1), SOX2, NANOG         [3 genes]
  SCC₂ — Hematopoietic TF network:   GATA1, GATA2, KLF1, SPI1, TAL1    [5 genes]
  SCC₃ — Lymphocyte activation:      BATF, IRF4                          [2 genes]

Inter-SCC edges (DAG forcing, NOT part of any SCC):
  SCC₁ → SCC₂ :  NANOG⊣GATA1, OCT4⊣GATA1  (pluripotency represses commitment)
  SCC₂ → SCC₃ :  SPI1→IRF4                  (PU.1 activates lymphocyte programme)

Sources:
  SCC₁ topology:  Chickarmane & Peterson 2008 (OCT4/SOX2/NANOG mutual activation)
  SCC₂ topology:  Huang et al. 2007 (GATA1/PU.1 bistable toggle) +
                  TRRUST v2 / DoRothEA A-confidence for GATA2/KLF1/TAL1 edges
  SCC₃ topology:  BATF/IRF4 mutual-activation loop (DoRothEA A: Glasmacher 2012)
  SCC₁→SCC₂:     OCT4⊣GATA1 (Babaie 2007), NANOG⊣GATA1 (Chambers 2007)
  SCC₂→SCC₃:     SPI1→IRF4 (DoRothEA A, Glasmacher 2010)

Hill-function ODE:  dx_i/dt = P_i * prod_j H_{ji}(x_j) − K_i * x_i
  H(x, T, n, F) = (F·r + 1)/(r + 1),   r = (x/T)^n
  F > 1 → activator,  F < 1 → repressor.

Three signaling contexts (analogous to gap-gene AP positions):
  "ESC"       — embryonic stem cell niche: OCT4/SOX2/NANOG signals dominant
  "progenitor" — transitional: pluripotency factors waning, GATA/PU.1 balanced
  "myeloid"   — hematopoietic cytokine-rich: low pluripotency, elevated SPI1

Node ordering (composition format — SCC₁ first, then SCC₂, then SCC₃):
  [0=OCT4, 1=SOX2, 2=NANOG,  3=GATA1, 4=GATA2, 5=KLF1, 6=SPI1, 7=TAL1,  8=BATF, 9=IRF4]

Bistability notes:
  SCC₁ (all mutual activation, F=3.0, T=0.5):
    x_on ≈ P/K ≈ 2–3; x_off ≈ 0 (no repressors in this SCC)
    Positive-feedback monostability or bistability depending on P.
    With P≥2, T=0.5, F=3.0: {all-ON} and {all-OFF} are both stable.

  SCC₂ GATA1/SPI1 toggle (F=0.01, T=0.3, n=4):
    Critical threshold: P/K > 2T/F^(1/n) ≈ 1.90 → P=2.0 ✓
    Erythroid state: GATA1+, GATA2+, KLF1+, TAL1+, SPI1−
    Myeloid state: SPI1+, GATA1−, GATA2−, KLF1−, TAL1−
    (GATA2/KLF1/TAL1 are erythroid co-factors: coactivated with GATA1)

  SCC₃ BATF/IRF4 (all mutual activation, F=3.0, T=0.5):
    Activated: {BATF+, IRF4+}
    Resting:   {BATF−, IRF4−}
"""

import numpy as np

# ---------------------------------------------------------------------------
# Kinetic constants
# ---------------------------------------------------------------------------
K_BASE = 1.0

# Strong mutual repression (GATA1/SPI1 toggle core, same as gap gene T_REP)
T_REP  = 0.30;  N_REP  = 4;  F_REP  = 0.01

# GATA switch: GATA1 ⊣ GATA2 (moderate — GATA1 displaces GATA2 at autoregulatory sites)
T_GATA_SW = 0.40;  N_GATA_SW = 3;  F_GATA_SW = 0.10

# Strong mutual activation (pluripotency triad, BATF/IRF4)
T_ACT_STR = 0.50;  N_ACT_STR = 3;  F_ACT_STR = 3.0

# Moderate activation (erythroid co-factor co-activation: GATA2/KLF1/TAL1 ↔ GATA1)
T_ACT_MOD = 0.50;  N_ACT_MOD = 3;  F_ACT_MOD = 2.5

# Inter-SCC forcing edges
T_PLURI_REP = 0.40;  N_PLURI_REP = 3;  F_PLURI_REP = 0.05  # OCT4/NANOG ⊣ GATA1
T_PU1_IRF4  = 0.40;  N_PU1_IRF4  = 3;  F_PU1_IRF4  = 3.0   # SPI1 → IRF4

# ---------------------------------------------------------------------------
# Signaling contexts
# ---------------------------------------------------------------------------
# Production rates per gene per context.  Values encode external signal strength
# (cytokines, growth factors, niche signals) biologically:
#   ESC       — LIF/BMP/FGF4 → OCT4/SOX2/NANOG high; hematopoietic low
#   progenitor — transitional niche: pluripotency waning, GATA/PU.1 balanced
#   myeloid    — SCF/EPO/GM-CSF present; pluripotency low; PU.1 elevated

CONTEXTS = {
    "ESC": {
        "label": "ESC niche — pluripotency factors dominant",
        "P_genes": {
            "OCT4": 3.0, "SOX2": 3.0, "NANOG": 3.0,
            "GATA1": 2.0, "GATA2": 2.0, "KLF1": 2.0, "SPI1": 2.0, "TAL1": 2.0,
            "BATF": 2.0, "IRF4": 2.0,
        },
    },
    "progenitor": {
        "label": "Progenitor — balanced / transitional",
        "P_genes": {
            "OCT4": 2.0, "SOX2": 2.0, "NANOG": 2.0,
            "GATA1": 2.5, "GATA2": 2.5, "KLF1": 2.0, "SPI1": 2.5, "TAL1": 2.0,
            "BATF": 2.0, "IRF4": 2.0,
        },
    },
    "myeloid": {
        "label": "Myeloid/lymphoid — hematopoietic cytokines dominant",
        "P_genes": {
            "OCT4": 2.0, "SOX2": 2.0, "NANOG": 2.0,
            "GATA1": 2.0, "GATA2": 2.0, "KLF1": 2.0, "SPI1": 3.0, "TAL1": 2.0,
            "BATF": 2.5, "IRF4": 2.5,
        },
    },
}

# ---------------------------------------------------------------------------
# Gene name lists (composition node order)
# ---------------------------------------------------------------------------
SCC1_GENES = ["OCT4", "SOX2", "NANOG"]
SCC2_GENES = ["GATA1", "GATA2", "KLF1", "SPI1", "TAL1"]
SCC3_GENES = ["BATF", "IRF4"]

ALL_GENES = SCC1_GENES + SCC2_GENES + SCC3_GENES   # 10 genes
ALL_IDX   = {g: i for i, g in enumerate(ALL_GENES)}

# SCC₁ local indices (within the 3-gene block)
_S1 = {g: i for i, g in enumerate(SCC1_GENES)}
# SCC₂ local indices (within the 5-gene block)
_S2 = {g: i for i, g in enumerate(SCC2_GENES)}
# SCC₃ local indices (within the 2-gene block)
_S3 = {g: i for i, g in enumerate(SCC3_GENES)}

# ---------------------------------------------------------------------------
# Parameter builder helpers
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


# ---------------------------------------------------------------------------
# Per-SCC builders  (return local arrays, not full 10-gene arrays)
# ---------------------------------------------------------------------------
def build_scc1(ctx_key: str):
    """SCC₁: OCT4/SOX2/NANOG — fully connected mutual activation (3 genes).

    Chickarmane & Peterson 2008: all three factors bind each other's promoters
    and autoregulate through the OCT4/SOX2/NANOG composite element.
    Returns (P[3], K[3], adj[3,3], T[3,3], nh[3,3], F[3,3]).
    """
    p_map = CONTEXTS[ctx_key]["P_genes"]
    P   = np.array([p_map[g] for g in SCC1_GENES], dtype=np.float64)
    K   = np.full(3, K_BASE, dtype=np.float64)
    adj, T, nh, Fm = _make_par(3)
    idx = _S1

    # Fully connected mutual activation among all 3
    for src in SCC1_GENES:
        for tgt in SCC1_GENES:
            if src != tgt:
                _edge(adj, T, nh, Fm, idx[src], idx[tgt], +1,
                      T_ACT_STR, N_ACT_STR, F_ACT_STR)
    return P, K, adj, T, nh, Fm


def build_scc2(ctx_key: str):
    """SCC₂: GATA1/GATA2/KLF1/SPI1/TAL1 — GATA/PU.1 toggle + erythroid co-factors (5 genes).

    Core toggle (Huang 2007):
      GATA1 ⊣ SPI1  (strong, T=0.30)
      SPI1  ⊣ GATA1 (strong, T=0.30)

    GATA switch (DoRothEA A):
      GATA2 → GATA1 (activation: GATA2 drives GATA1 during erythropoiesis)
      GATA1 ⊣ GATA2 (repression: GATA1 displaces GATA2 at autoregulatory sites)

    Erythroid co-factor loops (TRRUST / DoRothEA A/B):
      TAL1 ↔ GATA1 (mutual activation: erythroid complex)
      KLF1 ↔ GATA1 (mutual activation: KLF1 is a GATA1 target and activates GATA1)

    Returns (P[5], K[5], adj[5,5], T[5,5], nh[5,5], F[5,5]).
    """
    p_map = CONTEXTS[ctx_key]["P_genes"]
    P   = np.array([p_map[g] for g in SCC2_GENES], dtype=np.float64)
    K   = np.full(5, K_BASE, dtype=np.float64)
    adj, T, nh, Fm = _make_par(5)
    idx = _S2

    # Core GATA1/SPI1 toggle — strong mutual repression
    _edge(adj, T, nh, Fm, idx["GATA1"], idx["SPI1"],  -1, T_REP, N_REP, F_REP)
    _edge(adj, T, nh, Fm, idx["SPI1"],  idx["GATA1"], -1, T_REP, N_REP, F_REP)

    # GATA switch
    _edge(adj, T, nh, Fm, idx["GATA2"], idx["GATA1"], +1, T_ACT_MOD, N_ACT_MOD, F_ACT_MOD)
    _edge(adj, T, nh, Fm, idx["GATA1"], idx["GATA2"], -1, T_GATA_SW, N_GATA_SW, F_GATA_SW)

    # Erythroid co-factor loops (TAL1 ↔ GATA1, KLF1 ↔ GATA1)
    _edge(adj, T, nh, Fm, idx["TAL1"],  idx["GATA1"], +1, T_ACT_MOD, N_ACT_MOD, F_ACT_MOD)
    _edge(adj, T, nh, Fm, idx["GATA1"], idx["TAL1"],  +1, T_ACT_MOD, N_ACT_MOD, F_ACT_MOD)
    _edge(adj, T, nh, Fm, idx["KLF1"],  idx["GATA1"], +1, T_ACT_MOD, N_ACT_MOD, F_ACT_MOD)
    _edge(adj, T, nh, Fm, idx["GATA1"], idx["KLF1"],  +1, T_ACT_MOD, N_ACT_MOD, F_ACT_MOD)

    return P, K, adj, T, nh, Fm


def build_scc3(ctx_key: str):
    """SCC₃: BATF/IRF4 — mutual activation (lymphocyte activation loop, 2 genes).

    DoRothEA A: BATF and IRF4 form AP-1/IRF composite elements (AICEs) that
    mutually reinforce each other's expression (Glasmacher et al. 2012).
    Returns (P[2], K[2], adj[2,2], T[2,2], nh[2,2], F[2,2]).
    """
    p_map = CONTEXTS[ctx_key]["P_genes"]
    P   = np.array([p_map[g] for g in SCC3_GENES], dtype=np.float64)
    K   = np.full(2, K_BASE, dtype=np.float64)
    adj, T, nh, Fm = _make_par(2)
    idx = _S3

    _edge(adj, T, nh, Fm, idx["BATF"], idx["IRF4"], +1, T_ACT_STR, N_ACT_STR, F_ACT_STR)
    _edge(adj, T, nh, Fm, idx["IRF4"], idx["BATF"], +1, T_ACT_STR, N_ACT_STR, F_ACT_STR)

    return P, K, adj, T, nh, Fm


# ---------------------------------------------------------------------------
# Full 10-gene network builder
# ---------------------------------------------------------------------------
def build_full_network(ctx_key: str):
    """Assemble the full 10-gene network with inter-SCC forcing edges.

    Returns (P[10], K[10], adj[10,10], T[10,10], nh[10,10], F[10,10]).
    Node order: [OCT4, SOX2, NANOG, GATA1, GATA2, KLF1, SPI1, TAL1, BATF, IRF4]
    """
    p_map = CONTEXTS[ctx_key]["P_genes"]
    N = len(ALL_GENES)
    P   = np.array([p_map[g] for g in ALL_GENES], dtype=np.float64)
    K   = np.full(N, K_BASE, dtype=np.float64)
    adj, T, nh, Fm = _make_par(N)
    idx = ALL_IDX

    # --- SCC₁ internal edges ---
    for src in SCC1_GENES:
        for tgt in SCC1_GENES:
            if src != tgt:
                _edge(adj, T, nh, Fm, idx[src], idx[tgt], +1,
                      T_ACT_STR, N_ACT_STR, F_ACT_STR)

    # --- SCC₂ internal edges ---
    _edge(adj, T, nh, Fm, idx["GATA1"], idx["SPI1"],  -1, T_REP,       N_REP,       F_REP)
    _edge(adj, T, nh, Fm, idx["SPI1"],  idx["GATA1"], -1, T_REP,       N_REP,       F_REP)
    _edge(adj, T, nh, Fm, idx["GATA2"], idx["GATA1"], +1, T_ACT_MOD,   N_ACT_MOD,   F_ACT_MOD)
    _edge(adj, T, nh, Fm, idx["GATA1"], idx["GATA2"], -1, T_GATA_SW,   N_GATA_SW,   F_GATA_SW)
    _edge(adj, T, nh, Fm, idx["TAL1"],  idx["GATA1"], +1, T_ACT_MOD,   N_ACT_MOD,   F_ACT_MOD)
    _edge(adj, T, nh, Fm, idx["GATA1"], idx["TAL1"],  +1, T_ACT_MOD,   N_ACT_MOD,   F_ACT_MOD)
    _edge(adj, T, nh, Fm, idx["KLF1"],  idx["GATA1"], +1, T_ACT_MOD,   N_ACT_MOD,   F_ACT_MOD)
    _edge(adj, T, nh, Fm, idx["GATA1"], idx["KLF1"],  +1, T_ACT_MOD,   N_ACT_MOD,   F_ACT_MOD)

    # --- SCC₃ internal edges ---
    _edge(adj, T, nh, Fm, idx["BATF"], idx["IRF4"], +1, T_ACT_STR, N_ACT_STR, F_ACT_STR)
    _edge(adj, T, nh, Fm, idx["IRF4"], idx["BATF"], +1, T_ACT_STR, N_ACT_STR, F_ACT_STR)

    # --- Inter-SCC forcing edges (DAG edges, NOT in any SCC) ---
    # SCC₁ → SCC₂: NANOG⊣GATA1, OCT4⊣GATA1
    _edge(adj, T, nh, Fm, idx["NANOG"], idx["GATA1"], -1, T_PLURI_REP, N_PLURI_REP, F_PLURI_REP)
    _edge(adj, T, nh, Fm, idx["OCT4"],  idx["GATA1"], -1, T_PLURI_REP, N_PLURI_REP, F_PLURI_REP)
    # SCC₂ → SCC₃: SPI1→IRF4
    _edge(adj, T, nh, Fm, idx["SPI1"],  idx["IRF4"],  +1, T_PU1_IRF4,  N_PU1_IRF4,  F_PU1_IRF4)

    return P, K, adj, T, nh, Fm

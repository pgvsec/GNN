"""Quick visual sanity-check for Corpus B shards.

    python analyze_corpus_b.py --corpus_b_dir corpusB --corpus_a_dir corpusA
    python analyze_corpus_b.py --corpus_b_dir corpusB --corpus_a_dir corpusA --out_dir corpusB/analysis

Produces two figures (saved as PNGs + shown if a display is available):

Figure 1 — coverage_b.png   (6 panels, what went into the corpus)
  [0,0] Stratum distribution by n             (stacked bar: broad / near-bif / opposite)
  [0,1] Bifurcation type by n                 (stacked bar: none / saddle-node / Hopf)
  [1,0] bif_crossed % by n and stratum        (grouped bar: did forcing change n_fp?)
  [1,1] Forced n_fp distribution              (log-scale histogram vs autonomous n_fp)
  [2,0] Brouwer-certified % by n              (bar, target > 70%)
  [2,1] Forcing level u distribution by port count (log-x histogram, L=1/2/3)

Figure 2 — quality_b.png   (6 panels, label quality + alignment with Corpus A)
  [0,0] Mean bif_crossed by Thomas class      (are tc=1 / tc=2 / tc=3 behaving as expected?)
  [0,1] Forcing level u vs bif_crossed        (2-D density: does crossing cluster near u_bif?)
  [1,0] n_fp change: forced - autonomous      (histogram; should be symmetric around 0 overall)
  [1,1] hopf_ratio distribution (forced FPs)  (log histogram; near-zero = Hopf proximity)
  [2,0] zeroeig distribution (forced FPs)     (log histogram; near-zero = near saddle-node)
  [2,1] Certified % vs bif_crossed %  per graph-n  (scatter, one dot per n)

Numbers to look for:
  - Brouwer-certified > 70% for all n  (solver completeness)
  - bif_crossed between 25-50%         (forcing is diverse, not stuck on one side)
  - Stratum 1 share ≈ 35%              (near-bifurcation oversampling is active)
  - Saddle-node or Hopf bif_type > 0 for at least some shards (sweep is finding boundaries)
  - n_fp change histogram roughly centered (no systematic bias toward more/fewer FPs)
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

N_VALUES = list(range(2, 11))
STRATUM_NAMES  = ["broad", "near-bif", "opposite"]
STRATUM_COLORS = ["#0072BD", "#D95319", "#EDB120"]
BIF_TYPE_NAMES  = ["none", "saddle-node", "Hopf"]
BIF_TYPE_COLORS = ["#AAAAAA", "#D95319", "#7E2F8E"]
TC_NAMES  = ["neg-only (tc=1)", "pos-only (tc=2)", "both (tc=3)"]
TC_COLORS = ["#D95319", "#0072BD", "#EDB120"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_shards(shard_dir, max_shards=0):
    paths = sorted(glob.glob(os.path.join(shard_dir, "shard_b_*.npz")))
    if not paths:
        raise FileNotFoundError(f"No shard_b_*.npz files found under {shard_dir}")
    if max_shards > 0:
        paths = paths[:max_shards]
    print(f"Loading {len(paths)} shard(s) from {shard_dir} …")

    arrays = {}
    for p in paths:
        with np.load(p, allow_pickle=False) as z:
            for k in z.files:
                if k == "_sizes":
                    continue
                v = z[k]
                if k not in arrays:
                    arrays[k] = [v]
                else:
                    arrays[k].append(v)

    # Concatenate along axis 0 for every key
    out = {}
    for k, vs in arrays.items():
        try:
            out[k] = np.concatenate(vs, axis=0)
        except ValueError:
            out[k] = vs[0]   # single non-stackable array (shouldn't happen)
    return out


def load_corpus_a_nfp(corpus_a_dir):
    """Autonomous n_fp per graph from Corpus A cache (for comparison)."""
    cache = os.path.join(corpus_a_dir, "cache")
    s_graph = np.load(os.path.join(cache, "s_graph.npy"), mmap_mode="r")
    s_n_fp  = np.load(os.path.join(cache, "s_n_fp.npy"),  mmap_mode="r")
    g_n     = np.load(os.path.join(cache, "g_n.npy"),     mmap_mode="r")
    # Median autonomous n_fp per graph
    n_graphs = int(g_n.shape[0])
    med = np.zeros(n_graphs, dtype=np.float32)
    for g in range(n_graphs):
        mask = np.asarray(s_graph) == g
        if mask.any():
            med[g] = np.median(np.asarray(s_n_fp)[mask])
    return np.asarray(g_n), med   # shape (n_graphs,)


# ---------------------------------------------------------------------------
# Figure 1: coverage
# ---------------------------------------------------------------------------
def figure_coverage(d, out_path, dpi):
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    plt.rcParams.update({"font.size": 12, "axes.grid": True, "grid.alpha": 0.3})

    s_graph   = d["s_graph"]          # local graph index per sample
    g_n       = d["g_n"]              # gene count per graph
    sample_n  = g_n[s_graph]          # gene count per sample
    stratum   = d["cb_stratum"]
    bif_type  = d["cb_bif_type"]
    bif_cross = d["cb_bif_crossed"].astype(bool)
    forced_fp = d["s_n_fp"].astype(np.int64)
    cons      = d["s_cons"].astype(bool)
    auto_nfp  = d["cb_auto_nfp"].astype(np.int64)

    # [0,0] Stratum by n
    ax = axes[0, 0]
    bottom = np.zeros(len(N_VALUES))
    for st, name, color in zip(range(3), STRATUM_NAMES, STRATUM_COLORS):
        vals = np.array([(sample_n[stratum == st] == k).sum() for k in N_VALUES])
        ax.bar(N_VALUES, vals, bottom=bottom, color=color, edgecolor="k",
               linewidth=0.3, label=name)
        bottom += vals
    ax.legend(fontsize=10)
    ax.set(xlabel="n (gene count)", ylabel="Samples", title="Stratum distribution by n")

    # [0,1] Bifurcation type by n
    ax = axes[0, 1]
    bottom = np.zeros(len(N_VALUES))
    for bt, name, color in zip(range(3), BIF_TYPE_NAMES, BIF_TYPE_COLORS):
        vals = np.array([(sample_n[bif_type == bt] == k).sum() for k in N_VALUES])
        ax.bar(N_VALUES, vals, bottom=bottom, color=color, edgecolor="k",
               linewidth=0.3, label=name)
        bottom += vals
    ax.legend(fontsize=10)
    ax.set(xlabel="n", ylabel="Samples", title="Bifurcation type by n")

    # [1,0] bif_crossed % by n and stratum
    ax = axes[1, 0]
    w = 0.25
    xs = np.array(N_VALUES, dtype=float)
    for si, (name, color) in enumerate(zip(STRATUM_NAMES, STRATUM_COLORS)):
        m_st = stratum == si
        pct = np.array([
            100 * bif_cross[m_st & (sample_n == k)].mean()
            if (m_st & (sample_n == k)).any() else np.nan
            for k in N_VALUES])
        ax.bar(xs + (si - 1) * w, pct, width=w, color=color,
               edgecolor="k", linewidth=0.3, label=name)
    ax.legend(fontsize=10)
    ax.set(xlabel="n", ylabel="bif_crossed %",
           title="Forcing crossed a bifurcation (by stratum)")
    ax.axhline(25, color="k", ls="--", lw=1, alpha=0.5)
    ax.axhline(50, color="k", ls="--", lw=1, alpha=0.5)

    # [1,1] Forced vs autonomous n_fp distribution
    ax = axes[1, 1]
    bins = np.arange(0, 21) - 0.5
    ax.hist(forced_fp[forced_fp <= 20], bins=bins, color="#0072BD",
            edgecolor="k", linewidth=0.3, alpha=0.7, label="forced n_fp")
    ax.hist(auto_nfp[auto_nfp <= 20], bins=bins, color="#D95319",
            edgecolor="k", linewidth=0.3, alpha=0.7, label="autonomous n_fp (stored)")
    ax.set_yscale("log")
    ax.legend(fontsize=10)
    ax.set(xlabel="n_fp", ylabel="Count (log)", title="Forced vs autonomous fixed-point count")

    # [2,0] Brouwer-certified % by n
    ax = axes[2, 0]
    pct = np.array([100 * cons[sample_n == k].mean()
                    if (sample_n == k).any() else np.nan for k in N_VALUES])
    bars = ax.bar(N_VALUES, pct, color="#4DB366", edgecolor="k", linewidth=0.3)
    ax.axhline(70, color="r", ls="--", lw=1.5, label="70% target")
    for bar, v in zip(bars, pct):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5,
                    f"{v:.0f}%", ha="center", va="bottom", fontsize=9)
    ax.legend(fontsize=10)
    ax.set(xlabel="n", ylabel="Certified %", title="Brouwer-certified samples by n",
           ylim=(0, 105))

    # [2,1] Forcing level u distribution by port count
    ax = axes[2, 1]
    # cb_port_u is flat [n_samples * n_ports]; cb_n_ports tells us how many ports per sample
    n_ports = d["cb_n_ports"].astype(np.int64)
    cb_u    = d["cb_port_u"].astype(np.float32)
    # Reconstruct per-port u via cumsum of n_ports
    ptr = np.concatenate([[0], np.cumsum(n_ports)])
    log_u_by_L = {1: [], 2: [], 3: []}
    for i, (n_p, p0) in enumerate(zip(n_ports, ptr[:-1])):
        u_i = cb_u[p0:p0 + n_p]
        lv = int(min(n_p, 3))
        log_u_by_L.setdefault(lv, []).append(np.log10(np.maximum(u_i, 1e-6)))
    colors_L = {1: "#0072BD", 2: "#D95319", 3: "#EDB120"}
    for lv in (1, 2, 3):
        data = log_u_by_L.get(lv, [])
        if data:
            ax.hist(np.concatenate(data), bins=60, density=True,
                    alpha=0.55, color=colors_L[lv],
                    edgecolor="none", label=f"L={lv} ports")
    ax.axvline(0, color="k", ls="--", lw=1, alpha=0.5, label="u=1 (neutral)")
    ax.legend(fontsize=10)
    ax.set(xlabel="log10(u) — forcing level", ylabel="Density",
           title="Forcing level distribution by port count")

    fig.suptitle(f"Corpus B Coverage  ({d['s_n_fp'].shape[0]:,} samples)",
                 fontsize=18, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# Figure 2: quality
# ---------------------------------------------------------------------------
def figure_quality(d, g_n_ca, auto_nfp_ca, out_path, dpi):
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))

    s_graph   = d["s_graph"]
    g_n       = d["g_n"]
    g_tc      = d.get("g_tc", None)
    sample_n  = g_n[s_graph]
    stratum   = d["cb_stratum"]
    bif_cross = d["cb_bif_crossed"].astype(bool)
    bif_u     = d["cb_bif_u"].astype(np.float32)
    forced_fp = d["s_n_fp"].astype(np.int64)
    auto_fp   = d["cb_auto_nfp"].astype(np.int64)
    cons      = d["s_cons"].astype(bool)

    # Flat FP arrays
    fp_hopf    = d.get("fp_hopf",    np.array([], np.float32))
    fp_zeroeig = d.get("fp_zeroeig", np.array([], np.float32))

    # [0,0] bif_crossed by Thomas class
    ax = axes[0, 0]
    if g_tc is not None:
        tc_sample = g_tc[s_graph]
        for ti, (name, color) in enumerate(zip(TC_NAMES, TC_COLORS), start=1):
            m = tc_sample == ti
            if m.any():
                pct = 100 * bif_cross[m].mean()
                ax.bar(name, pct, color=color, edgecolor="k", linewidth=0.3)
                ax.text(name, pct + 0.5, f"{pct:.1f}%", ha="center",
                        va="bottom", fontsize=10)
        ax.set(ylabel="bif_crossed %", title="Bifurcation crossed by Thomas class",
               ylim=(0, 100))
        ax.axhline(25, color="k", ls="--", lw=1, alpha=0.5)
    else:
        ax.text(0.5, 0.5, "g_tc not in shards", ha="center",
                va="center", transform=ax.transAxes)
        ax.set_title("bif_crossed by Thomas class (unavailable)")

    # [0,1] Forcing level u vs bif_crossed (2-D scatter density)
    ax = axes[0, 1]
    # Use per-sample first port u as a proxy
    n_ports = d["cb_n_ports"].astype(np.int64)
    cb_u    = d["cb_port_u"].astype(np.float32)
    ptr = np.concatenate([[0], np.cumsum(n_ports)])
    u_first = np.array([cb_u[ptr[i]] if n_ports[i] > 0 else np.nan
                        for i in range(len(n_ports))], dtype=np.float32)
    valid = np.isfinite(u_first) & (u_first > 0)
    log_u = np.log10(u_first[valid])
    cross = bif_cross[valid].astype(np.float32)
    ax.scatter(log_u[~cross.astype(bool)], np.random.default_rng(0).uniform(-0.1, 0.1, (~cross.astype(bool)).sum()),
               s=2, alpha=0.15, color="#0072BD", rasterized=True, label="not crossed")
    ax.scatter(log_u[cross.astype(bool)], np.random.default_rng(1).uniform(0.9, 1.1, cross.astype(bool).sum()),
               s=2, alpha=0.15, color="#D95319", rasterized=True, label="crossed")
    ax.set(xlabel="log10(u) — first port forcing level",
           ylabel="bif_crossed (jittered)", title="Forcing level vs bifurcation crossing",
           ylim=(-0.5, 1.5), yticks=[0, 1], yticklabels=["not crossed", "crossed"])
    ax.axvline(0, color="k", ls="--", lw=1, alpha=0.4)
    ax.legend(markerscale=4, fontsize=10)

    # [1,0] n_fp change histogram
    ax = axes[1, 0]
    delta = forced_fp - auto_fp
    lo, hi = int(delta.min()), int(delta.max())
    bins = np.arange(max(lo, -10) - 0.5, min(hi, 10) + 1.5)
    ax.hist(delta, bins=bins, color="#7FBFE0", edgecolor="k", linewidth=0.3)
    ax.axvline(0, color="r", ls="--", lw=1.5, label="no change")
    ax.set_yscale("log")
    ax.legend(fontsize=10)
    ax.set(xlabel="forced n_fp − autonomous n_fp",
           ylabel="Count (log)", title="Fixed-point count change due to forcing")
    frac_changed = (delta != 0).mean()
    ax.text(0.97, 0.97, f"Changed: {100*frac_changed:.1f}%",
            transform=ax.transAxes, ha="right", va="top", fontsize=11)

    # [1,1] hopf_ratio distribution (forced FPs)
    ax = axes[1, 1]
    hr = fp_hopf[np.isfinite(fp_hopf) & (fp_hopf > 0)]
    if hr.size > 0:
        ax.hist(np.log10(hr), bins=80, color="#9B4DB3", edgecolor="k", linewidth=0.2)
        ax.set(xlabel="log10(hopf_ratio = |α_lead| / |β_lead|)",
               ylabel="Count", title="Hopf proximity of forced fixed points")
        ax.axvline(np.log10(0.3), color="r", ls="--", lw=1.5,
                   label="ratio=0.3 (near-Hopf)")
        ax.legend(fontsize=10)
    else:
        ax.text(0.5, 0.5, "No complex eigenvalues found", ha="center",
                va="center", transform=ax.transAxes)
        ax.set_title("Hopf proximity (no data)")

    # [2,0] zeroeig distribution (forced FPs, near-saddle-node indicator)
    ax = axes[2, 0]
    ze = fp_zeroeig[np.isfinite(fp_zeroeig) & (fp_zeroeig > 0)]
    if ze.size > 0:
        ax.hist(np.log10(ze), bins=80, color="#E69A1A", edgecolor="k", linewidth=0.2)
        ax.set(xlabel="log10(min|λ| / spectral_radius)",
               ylabel="Count", title="Saddle-node proximity of forced FPs")
        ax.axvline(np.log10(1e-3), color="r", ls="--", lw=1.5,
                   label="ratio < 1e-3 (near SN)")
        ax.legend(fontsize=10)
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("Saddle-node proximity (no data)")

    # [2,1] Certified % vs bif_crossed % per n (scatter)
    ax = axes[2, 1]
    cert_n  = [100 * cons[sample_n == k].mean()
               if (sample_n == k).any() else np.nan for k in N_VALUES]
    cross_n = [100 * bif_cross[sample_n == k].mean()
               if (sample_n == k).any() else np.nan for k in N_VALUES]
    for n, c, b in zip(N_VALUES, cert_n, cross_n):
        if not (np.isnan(c) or np.isnan(b)):
            ax.scatter(b, c, s=90, zorder=3, color="#0072BD")
            ax.text(b + 0.5, c + 0.5, f"n={n}", fontsize=9)
    ax.axhline(70, color="r", ls="--", lw=1, alpha=0.6, label="cert 70% target")
    ax.axvline(25, color="orange", ls="--", lw=1, alpha=0.6, label="cross 25% floor")
    ax.legend(fontsize=10)
    ax.set(xlabel="bif_crossed %", ylabel="Brouwer-certified %",
           title="Quality: certified vs bif_crossed per n")

    fig.suptitle(f"Corpus B Quality  ({d['s_n_fp'].shape[0]:,} samples)",
                 fontsize=18, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------------
def text_summary(d):
    n_samples = int(d["s_n_fp"].shape[0])
    cons      = d["s_cons"].astype(bool)
    bif_cross = d["cb_bif_crossed"].astype(bool)
    stratum   = d["cb_stratum"]
    bif_type  = d["cb_bif_type"]
    s_graph   = d["s_graph"]
    g_n       = d["g_n"]
    sample_n  = g_n[s_graph]

    lines = [
        "=" * 62,
        f"Corpus B summary  ({n_samples:,} samples)",
        "=" * 62,
        f"  Brouwer-certified   : {100*cons.mean():.1f}%   (target > 70%)",
        f"  bif_crossed         : {100*bif_cross.mean():.1f}%   (target 25-50%)",
        f"  Stratum 0 (broad)   : {100*(stratum==0).mean():.1f}%",
        f"  Stratum 1 (near-bif): {100*(stratum==1).mean():.1f}%   (target ~35%)",
        f"  Stratum 2 (opposite): {100*(stratum==2).mean():.1f}%   (target ~25%)",
        f"  Bif type saddle-node: {100*(bif_type==1).mean():.1f}%",
        f"  Bif type Hopf       : {100*(bif_type==2).mean():.1f}%",
        "",
        "  per n:  samples | certified% | bif_crossed%",
    ]
    for k in N_VALUES:
        m = sample_n == k
        if m.any():
            lines.append(
                f"    n={k:2d}: {int(m.sum()):7,d} | "
                f"{100*cons[m].mean():9.1f}% | "
                f"{100*bif_cross[m].mean():11.1f}%"
            )
    lines.append("=" * 62)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus_b_dir", default="corpusB")
    p.add_argument("--corpus_a_dir", default="corpusA")
    p.add_argument("--out_dir",      default="")
    p.add_argument("--max_shards",   type=int, default=0,
                   help="0 = load all shards")
    p.add_argument("--dpi",          type=int, default=150)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = args.out_dir or os.path.join(args.corpus_b_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    shard_dir = os.path.join(args.corpus_b_dir, "shards")
    d = load_shards(shard_dir, args.max_shards)

    # Try to load Corpus A for comparison
    g_n_ca, auto_nfp_ca = None, None
    ca_cache = os.path.join(args.corpus_a_dir, "cache")
    if os.path.exists(ca_cache):
        try:
            g_n_ca, auto_nfp_ca = load_corpus_a_nfp(args.corpus_a_dir)
        except Exception as e:
            print(f"  (Corpus A cache not loaded: {e})")

    summary = text_summary(d)
    print(summary)
    with open(os.path.join(out_dir, "summary_b.txt"), "w") as fh:
        fh.write(summary + "\n")

    figure_coverage(d, os.path.join(out_dir, "coverage_b.png"), args.dpi)
    figure_quality(d, g_n_ca, auto_nfp_ca,
                   os.path.join(out_dir, "quality_b.png"), args.dpi)
    print(f"\nAll outputs in {out_dir}")


if __name__ == "__main__":
    main()

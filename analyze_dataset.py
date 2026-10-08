"""Dataset analysis for Corpus A.

    python analyze_dataset.py --cache_dir corpusA/cache
    python analyze_dataset.py --cache_dir corpusA/cache --matlab_png highResPlot__1_.png

Writes into <out_dir> (default: <cache_dir>/../analysis)
  coverage_report.png   12 panels in the layout of the MATLAB "Corpus A Coverage Report (Rev 2)"
  data_properties.png   8 panels on label quality and on the parameter prior
  summary.txt           the numbers behind both figures
  comparison.png        with --matlab_png: the MATLAB report above this one

Conventions, chosen so that panels can be read next to the MATLAB figure
  Thomas class follows the MATLAB coding: 0 no loops, 1 positive loops only, 2 negative loops
  only, 3 both. (The cache stores 1 negative-only, 2 positive-only, 3 both; they are swapped
  here.) A strongly connected graph always has a loop, so class 0 is empty by construction.
  SPL / SNL are the lengths of the shortest positive / negative simple cycle: 1 is a
  self-loop, 2 a mutual pair. SPL is binned as none, 1, 2, 3+ and SNL as none, 1, 2, 3, 4+.
  Both are computed here from the stored edge lists.
  Brouwer categories of a sample: (A) consistent: indices of the stored fixed points sum to 1
  and every point is hyperbolic; (B) suspect: all points hyperbolic but the sum is not 1,
  which means at least one fixed point is missing or spurious; (C) near-bifurcation: at least
  one stored point has an eigenvalue with real part at the numerical zero.
  Draw phases: 0 broad prior, 1 switch-biased prior, 2 near the fixed-point-count boundary.

Sections
  1. Loading and derived quantities
  2. Coverage figure
  3. Data-properties figure
  4. Numeric summary and comparison figure
"""
import argparse
import itertools
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

N_VALUES = list(range(2, 11))
TC_COLORS = ["#0072BD", "#D95319", "#EDB120", "#7E2F8E"]
PHASE_NAMES = ["Phase 0 (broad)", "Phase 1 (switch)", "Phase 2 (boundary)"]
PHASE_COLORS = ["#0072BD", "#D95319", "#EDB120"]
CATEGORY_NAMES = ["(A) consistent", "(B) suspect: sum != 1", "(C) near-bifurcation"]
STAB_NAMES = ["stable node", "stable spiral", "unstable node", "unstable spiral", "saddle", "non-hyperbolic"]
STAB_COLORS = ["#2E7D32", "#81C784", "#C62828", "#EF9A9A", "#F9A825", "#212121"]
SPLIT_NAMES = ["train", "validation", "test"]
DENSITY_NAMES = ["sparse", "medium", "dense"]
QUOTA_NEG_ONLY = 15.0                    # percent of the topology quota aimed at negative-only graphs
NEEDED = ["g_n", "g_n_edges", "g_tc", "g_pos_fb", "g_neg_fb", "g_dt", "g_split", "g_edge_ptr",
          "e_src", "e_tgt", "e_sgn", "s_graph", "s_xmax", "s_n_fp", "s_cons", "s_bsum", "s_phase",
          "s_ep_ptr", "fp_stab", "fp_res"]


# ---------------------------------------------------------------------------
# 1. Loading and derived quantities
# ---------------------------------------------------------------------------
def load_array(cache_dir, key):
    return np.load(os.path.join(cache_dir, key + ".npy"), mmap_mode="r")


def load_cache(cache_dir):
    c = {k: np.asarray(load_array(cache_dir, k)) for k in NEEDED}
    c["cache_dir"] = cache_dir
    return c


def dense_graphs(c, ids, n):
    """Signed adjacency [len(ids), n, n] of the graphs `ids`, all with n genes."""
    adj = np.zeros((len(ids), n, n), dtype=np.int8)
    ptr, count = c["g_edge_ptr"], c["g_n_edges"]
    for i, g in enumerate(ids):
        lo = int(ptr[g])
        hi = lo + int(count[g])
        adj[i, c["e_src"][lo:hi].astype(np.int64), c["e_tgt"][lo:hi].astype(np.int64)] = c["e_sgn"][lo:hi]
    return adj


def shortest_loops(adj, max_len=4):
    """Shortest positive and negative simple cycle of every graph, up to max_len.

    adj [G, n, n] int8. Every ordered tuple of distinct genes that starts at its smallest gene
    is one cycle; its sign is the product of the edge signs, evaluated for all graphs at once.
    Returns two arrays of length G with the length of the shortest cycle of that sign, or 0
    if there is none of length <= max_len."""
    G, n, _ = adj.shape
    pos = np.zeros(G, dtype=np.int64)
    neg = np.zeros(G, dtype=np.int64)
    for length in range(1, max_len + 1):
        for cycle in itertools.permutations(range(n), length):
            if cycle[0] != min(cycle):
                continue
            sign = np.ones(G, dtype=np.int8)
            for a, b in zip(cycle, cycle[1:] + cycle[:1]):
                sign = sign * adj[:, a, b]
            pos[(sign > 0) & (pos == 0)] = length
            neg[(sign < 0) & (neg == 0)] = length
    return pos, neg


def graph_features(c):
    """Per-graph arrays: n, Thomas class (MATLAB coding), SPL bin, SNL bin, density tier, split.

    SPL bins: 0 none, 1, 2, 3 (3 or longer). SNL bins: 0 none, 1, 2, 3, 4 (4 or longer)."""
    n = c["g_n"].astype(np.int64)
    spl_len = np.zeros(n.shape[0], dtype=np.int64)
    snl_len = np.zeros(n.shape[0], dtype=np.int64)
    for k in N_VALUES:
        ids = np.nonzero(n == k)[0]
        if ids.size:
            spl_len[ids], snl_len[ids] = shortest_loops(dense_graphs(c, ids, k))
    has_pos, has_neg = c["g_pos_fb"].astype(bool), c["g_neg_fb"].astype(bool)
    if ((spl_len > 0) & ~has_pos).any() or ((snl_len > 0) & ~has_neg).any():
        raise RuntimeError("stored loop flags disagree with the edge lists")
    spl = np.where(~has_pos, 0, np.where(spl_len == 1, 1, np.where(spl_len == 2, 2, 3)))
    snl = np.where(~has_neg, 0, np.where(snl_len == 1, 1, np.where(snl_len == 2, 2, np.where(snl_len == 3, 3, 4))))
    tc = c["g_tc"].astype(np.int64)
    tc = np.where(tc == 1, 2, np.where(tc == 2, 1, tc))
    return {"n": n, "tc": tc, "spl": spl, "snl": snl, "dt": c["g_dt"].astype(np.int64),
            "split": c["g_split"].astype(np.int64)}


def sample_features(c, g):
    """Per-sample and per-fixed-point arrays derived from the cache."""
    n_fp = c["s_n_fp"].astype(np.int64)
    n_samples = n_fp.shape[0]
    sid = np.repeat(np.arange(n_samples), n_fp)               # sample of every fixed-point row
    stab = c["fp_stab"].astype(np.int64)
    nonhyp = np.bincount(sid, weights=(stab == 5).astype(np.float64), minlength=n_samples) > 0
    stable = np.bincount(sid, weights=np.isin(stab, (0, 1)).astype(np.float64), minlength=n_samples)
    cons = c["s_cons"].astype(bool)
    return {
        "n_fp": n_fp, "sid": sid, "stab": stab, "res": c["fp_res"].astype(np.float64),
        "n": g["n"][c["s_graph"]], "split": g["split"][c["s_graph"]], "phase": c["s_phase"].astype(np.int64),
        "cons": cons, "bsum": np.rint(c["s_bsum"]).astype(np.int64), "n_stable": stable,
        "category": np.where(cons, 0, np.where(nonhyp, 2, 1)),
    }


def edge_sample(c, s, n_samples, rng):
    """Edge parameters of a random subset of samples: fold-change, sign, Hill coefficient,
    threshold relative to the box bound of the regulating gene, and the sample's phase."""
    pick = np.sort(rng.choice(s["n_fp"].shape[0], size=min(n_samples, s["n_fp"].shape[0]), replace=False))
    ep_f, ep_t, ep_n = (load_array(c["cache_dir"], k) for k in ("ep_F", "ep_T", "ep_nh"))
    out = {k: [] for k in ("F", "sgn", "nh", "trel", "phase")}
    for i in pick:
        g = int(c["s_graph"][i])
        e0, ne, p0 = int(c["g_edge_ptr"][g]), int(c["g_n_edges"][g]), int(c["s_ep_ptr"][i])
        src = c["e_src"][e0:e0 + ne].astype(np.int64)
        out["F"].append(np.asarray(ep_f[p0:p0 + ne], dtype=np.float64))
        out["nh"].append(np.asarray(ep_n[p0:p0 + ne], dtype=np.int64))
        out["sgn"].append(np.asarray(c["e_sgn"][e0:e0 + ne], dtype=np.int64))
        out["trel"].append(np.asarray(ep_t[p0:p0 + ne], dtype=np.float64) / c["s_xmax"][i, src].astype(np.float64))
        out["phase"].append(np.full(ne, c["s_phase"][i], dtype=np.int64))
    return {k: np.concatenate(v) for k, v in out.items()}


# ---------------------------------------------------------------------------
# 2. Coverage figure
# ---------------------------------------------------------------------------
def label_bars(ax, values, fmt="{:,}"):
    for i, v in enumerate(values):
        ax.text(i, v, fmt.format(v), ha="center", va="bottom", fontsize=11)


def heatmap(ax, mat, xlabels, ylabels, title, xlabel, ylabel):
    im = ax.imshow(mat, cmap="hot", aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels(xlabels)
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels(ylabels)
    top = max(int(mat.max()), 1)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:,}", ha="center", va="center",
                    color="white" if mat[i, j] < 0.6 * top else "black", fontsize=12)
    ax.figure.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def cell_matrix(rows, cols, n_rows, n_cols):
    m = np.zeros((n_rows, n_cols), dtype=np.int64)
    np.add.at(m, (rows, cols), 1)
    return m


def coverage_figure(g, s, title, path, dpi):
    plt.rcParams.update({"font.size": 14, "axes.titlesize": 17, "axes.titleweight": "bold", "axes.grid": True,
                         "grid.alpha": 0.3})
    fig, axes = plt.subplots(3, 4, figsize=(23.2, 15.95))
    n_graphs, n_samples = g["n"].shape[0], s["n_fp"].shape[0]

    ax = axes[0, 0]
    counts = np.bincount(g["tc"], minlength=4)
    ax.bar(range(4), counts, color="#3380CC", edgecolor="k")
    label_bars(ax, counts)
    ax.set_xticks(range(4))
    ax.set(xlabel="Thomas class", ylabel="Count", title="Thomas class (all n)")

    ax = axes[0, 1]
    bottom = np.zeros(len(N_VALUES))
    for cls in range(4):
        vals = np.array([(g["tc"][g["n"] == k] == cls).sum() for k in N_VALUES])
        ax.bar(N_VALUES, vals, bottom=bottom, color=TC_COLORS[cls], edgecolor="k", label=f"tc={cls}")
        bottom += vals
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="upper right")
    ax.set(xlabel="n", ylabel="Count", title="Thomas class by n (stacked)")

    ax = axes[0, 2]
    frac = [100.0 * (g["tc"][g["n"] == k] == 2).mean() for k in N_VALUES]
    ax.bar(N_VALUES, frac, color="#CC4D33", edgecolor="k")
    ax.axhline(QUOTA_NEG_ONLY, color="k", ls="--", lw=1)
    ax.text(N_VALUES[-1] + 0.4, QUOTA_NEG_ONLY, f"{QUOTA_NEG_ONLY:.0f}% quota target", ha="right", va="bottom")
    ax.set(xlabel="n", ylabel="tc=2 fraction (%)", title="tc=2 (neg-only) coverage %")

    ax = axes[0, 3]
    counts = np.array([(g["spl"] == k).sum() for k in range(4)])
    ax.bar(range(4), counts, color="#4DB366", edgecolor="k")
    label_bars(ax, counts)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["none", "1", "2", "3+"])
    ax.set(xlabel="spl (shortest pos loop)", ylabel="Count", title="SPL distribution (all n)")

    ax = axes[1, 0]
    counts = np.array([(g["snl"] == k).sum() for k in range(5)])
    ax.bar(range(5), counts, color="#9B4DB3", edgecolor="k")
    label_bars(ax, counts)
    ax.set_xticks(range(5))
    ax.set_xticklabels(["none", "1", "2", "3", "4+"])
    ax.set(xlabel="snl (shortest neg loop)", ylabel="Count", title="SNL distribution (all n)")

    ax = axes[1, 1]
    counts = np.array([(g["dt"] == k).sum() for k in (1, 2, 3)])
    ax.bar(range(3), counts, color="#E69A1A", edgecolor="k")
    label_bars(ax, counts)
    ax.set_xticks(range(3))
    ax.set_xticklabels(DENSITY_NAMES)
    ax.set(xlabel="Density tier", ylabel="Count", title="Density tier (all n)")

    neg_only = g["tc"] == 2
    heatmap(axes[1, 2], cell_matrix(g["snl"][neg_only] - 1, g["dt"][neg_only] - 1, 4, 3),
            ["sparse", "med", "dense"], ["snl=1", "snl=2", "snl=3", "snl=4+"],
            "tc=2 cell counts (snl x density)", "Density tier", "SNL")
    both = g["tc"] == 3
    heatmap(axes[1, 3], cell_matrix(g["spl"][both] - 1, g["snl"][both] - 1, 3, 4),
            ["snl=1", "snl=2", "snl=3", "snl=4+"], ["spl=1", "spl=2", "spl=3+"],
            "tc=3 cell counts (spl x snl, all density)", "SNL", "SPL")

    ax = axes[2, 0]
    achieved = np.array([(neg_only & (g["n"] == k)).sum() for k in N_VALUES])
    target = np.array([QUOTA_NEG_ONLY / 100.0 * (g["n"] == k).sum() for k in N_VALUES])
    ax.bar(N_VALUES, achieved, color="#CC4D33", edgecolor="k", label="achieved")
    ax.plot(N_VALUES, target, "k--", label=f"{QUOTA_NEG_ONLY:.0f}% target")
    ax.legend()
    ax.set(xlabel="n", ylabel="tc=2 count", title=f"tc=2 count vs {QUOTA_NEG_ONLY:.0f}% target")

    ax = axes[2, 1]
    counts = np.bincount(np.minimum(s["n_fp"], 31), minlength=32)
    ax.bar(range(31), counts[:31], width=1.0, color="#7FBFE0", edgecolor="k", linewidth=0.3)
    ax.set_yscale("log")
    ax.text(0.97, 0.95, f"mean {s['n_fp'].mean():.2f}, max {s['n_fp'].max()}\n> 30 fixed points: {counts[31]:,}",
            transform=ax.transAxes, ha="right", va="top")
    ax.set(xlabel="Fixed points found per sample", ylabel="Count (log)",
           title=f"FP count dist. (n={n_samples:,} samples)")

    ax = axes[2, 2]
    clipped = np.clip(s["bsum"], -5, 5) + 5
    share = [100.0 * (s["category"] == k).mean() for k in range(3)]
    for k, color in enumerate(PHASE_COLORS):
        ax.bar(np.arange(-5, 6) + (k - 1) * 0.28, np.bincount(clipped[s["category"] == k], minlength=11),
               width=0.28, color=color, edgecolor="k", linewidth=0.3, label=CATEGORY_NAMES[k])
    ax.axvline(1, color="r", ls="--", lw=2)
    ax.set_yscale("log")
    ax.legend(loc="upper left", fontsize=11)
    ax.set(xlabel="Brouwer index sum", ylabel="Count (log)",
           title=f"Brouwer sum  A={share[0]:.1f}%  B={share[1]:.1f}%  C={share[2]:.1f}%")

    ax = axes[2, 3]
    for ph in range(3):
        counts = np.bincount(np.minimum(s["n_fp"][s["phase"] == ph], 31), minlength=32)[:31]
        ax.bar(np.arange(31) + (ph - 1) * 0.28, counts, width=0.28, color=PHASE_COLORS[ph],
               edgecolor="k", linewidth=0.3, label=PHASE_NAMES[ph])
    ax.set_yscale("log")
    ax.legend()
    ax.set(xlabel="FP count per sample", ylabel="Count (log)", title="FP count by draw phase")

    fig.suptitle(f"Corpus A Coverage Report ({title}):  {n_graphs:,} graphs, {n_samples:,} samples",
                 fontsize=26, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Data-properties figure
# ---------------------------------------------------------------------------
def properties_figure(s, e, title, path, dpi):
    fig, axes = plt.subplots(2, 4, figsize=(23.2, 11.5))
    n_arr, phase, cons = s["n"], s["phase"], s["cons"]

    ax = axes[0, 0]
    bottom = np.zeros(len(N_VALUES))
    for sp, name in zip((1, 2, 3), SPLIT_NAMES):
        vals = np.array([((n_arr == k) & (s["split"] == sp)).sum() for k in N_VALUES])
        ax.bar(N_VALUES, vals, bottom=bottom, edgecolor="k", label=name)
        bottom += vals
    ax.legend()
    ax.set(xlabel="n", ylabel="Samples", title="Samples by n and split (split by graph)")

    ax = axes[0, 1]
    for ph in range(3):
        y = [100.0 * cons[(n_arr == k) & (phase == ph)].mean() if ((n_arr == k) & (phase == ph)).any() else np.nan
             for k in N_VALUES]
        ax.plot(N_VALUES, y, "o-", color=PHASE_COLORS[ph], label=PHASE_NAMES[ph])
    ax.legend()
    ax.set(xlabel="n", ylabel="Brouwer-consistent samples (%)", title="Index-sum consistency by n and phase")

    ax = axes[0, 2]
    fp_phase = phase[s["sid"]]
    bottom = np.zeros(3)
    for cls in range(6):
        vals = np.array([100.0 * (s["stab"][fp_phase == ph] == cls).mean() for ph in range(3)])
        ax.bar(range(3), vals, bottom=bottom, color=STAB_COLORS[cls], edgecolor="k", label=STAB_NAMES[cls])
        bottom += vals
    ax.set_xticks(range(3))
    ax.set_xticklabels(["broad", "switch", "boundary"])
    ax.legend(loc="lower right", fontsize=10)
    ax.set(ylabel="Fixed points (%)", title="Stability classes by phase")

    ax = axes[0, 3]
    mean_fp = [s["n_fp"][n_arr == k].mean() for k in N_VALUES]
    multi = [100.0 * (s["n_stable"][n_arr == k] >= 2).mean() for k in N_VALUES]
    ax.bar(N_VALUES, mean_fp, color="#7FBFE0", edgecolor="k")
    ax.set(xlabel="n", ylabel="Mean fixed points per sample", title="Multiplicity by n")
    twin = ax.twinx()
    twin.plot(N_VALUES, multi, "o-", color="#C62828")
    twin.set_ylabel("Samples with >= 2 stable points (%)", color="#C62828")
    twin.grid(False)

    ax = axes[1, 0]
    log_res = np.log10(np.maximum(s["res"], 1e-18))
    ax.hist(log_res, bins=100, color="#7FBFE0", edgecolor="k", linewidth=0.2)
    ax.axvline(-9, color="r", ls="--", label="acceptance 1e-9")
    ax.set_yscale("log")
    ax.legend()
    ax.set(xlabel="log10 relative ODE residual |f| / (|P R| + K x)",
           title=f"Label quality: median {np.median(s['res']):.1e}, max {s['res'].max():.1e}")

    ax = axes[1, 1]
    for sign, name, color in ((1, "activation (F > 1)", "#2E7D32"), (-1, "repression (F < 1)", "#C62828")):
        ax.hist(np.log10(e["F"][e["sgn"] == sign]), bins=80, density=True, alpha=0.6, color=color, label=name)
    ax.legend()
    ax.set(xlabel="log10 F", ylabel="Density", title="Fold-change prior")

    ax = axes[1, 2]
    for ph in range(3):
        counts = np.bincount(e["nh"][e["phase"] == ph], minlength=6)[1:6]
        ax.bar(np.arange(1, 6) + (ph - 1) * 0.28, 100.0 * counts / max(counts.sum(), 1), width=0.28,
               color=PHASE_COLORS[ph], edgecolor="k", label=PHASE_NAMES[ph])
    ax.legend()
    ax.set(xlabel="Hill coefficient n_H", ylabel="Edges (%)", title="Cooperativity by phase")

    ax = axes[1, 3]
    lo, hi = np.percentile(np.log10(e["trel"]), [0.5, 99.5])
    for ph in range(3):
        ax.hist(np.log10(e["trel"][e["phase"] == ph]), bins=80, range=(lo, hi), density=True, alpha=0.55,
                color=PHASE_COLORS[ph], label=PHASE_NAMES[ph])
    ax.legend()
    ax.set(xlabel="log10 (T / xmax of the regulating gene)", ylabel="Density",
           title="Threshold relative to expression range")

    fig.suptitle(f"Corpus A data properties ({title})", fontsize=24, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Summary and comparison
# ---------------------------------------------------------------------------
def summary_lines(c, g, s, title):
    n_graphs, n_samples = g["n"].shape[0], s["n_fp"].shape[0]
    pct = np.percentile(s["n_fp"], [50, 90, 99, 99.9])
    lines = [f"Corpus A summary ({title})", f"cache: {c['cache_dir']}",
             f"graphs {n_graphs:,}   samples {n_samples:,}   fixed points {s['n_fp'].sum():,}",
             "Thomas coding: 1 positive-only, 2 negative-only, 3 both (MATLAB convention)", "",
             "per n:  graphs | tc=1 tc=2 tc=3 | sparse medium dense | samples  mean_fp  consistent%"]
    for k in N_VALUES:
        gm, sm = g["n"] == k, s["n"] == k
        if not gm.any():
            continue
        tcs = [int((g["tc"][gm] == t).sum()) for t in (1, 2, 3)]
        dts = [int((g["dt"][gm] == t).sum()) for t in (1, 2, 3)]
        lines.append(f"  n={k:2d}: {int(gm.sum()):6d} | {tcs[0]:4d} {tcs[1]:4d} {tcs[2]:5d} | {dts[0]:6d} {dts[1]:6d} {dts[2]:5d}"
                     f" | {int(sm.sum()):8d} {s['n_fp'][sm].mean():8.2f} {100 * s['cons'][sm].mean():10.1f}")
    lines += ["", f"fixed points per sample: mean {s['n_fp'].mean():.2f}, percentiles 50/90/99/99.9 = "
              f"{pct[0]:.0f}/{pct[1]:.0f}/{pct[2]:.0f}/{pct[3]:.0f}, max {s['n_fp'].max()}",
              f"samples with exactly 1 fixed point {100 * (s['n_fp'] == 1).mean():.1f}%, with >= 3 "
              f"{100 * (s['n_fp'] >= 3).mean():.1f}%, with >= 2 stable {100 * (s['n_stable'] >= 2).mean():.1f}%",
              "Brouwer categories: " + "  ".join(f"{CATEGORY_NAMES[k]} {100 * (s['category'] == k).mean():.2f}%"
                                                  for k in range(3)),
              "", "per phase:  samples  mean_fp  consistent%  >=2 stable%"]
    for ph in range(3):
        m = s["phase"] == ph
        if m.any():
            lines.append(f"  {PHASE_NAMES[ph]:<20s} {int(m.sum()):8d} {s['n_fp'][m].mean():8.2f} "
                         f"{100 * s['cons'][m].mean():11.1f} {100 * (s['n_stable'][m] >= 2).mean():11.1f}")
    lines += ["", "splits (by graph): " + "  ".join(
        f"{SPLIT_NAMES[k - 1]} {int((g['split'] == k).sum()):,} graphs / {int((s['split'] == k).sum()):,} samples"
        for k in (1, 2, 3)),
        f"stored residual |f|/(|P R|+K x): median {np.median(s['res']):.2e}, p99 {np.percentile(s['res'], 99):.2e}, "
        f"max {s['res'].max():.2e}"]
    return lines


def comparison_figure(matlab_png, python_png, path):
    fig, axes = plt.subplots(2, 1, figsize=(23.2, 31.9))
    for ax, image, name in zip(axes, (plt.imread(matlab_png), plt.imread(python_png)),
                               ("MATLAB generator (Rev 2)", "Python generator")):
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(name, fontsize=30, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="corpusA/cache")
    parser.add_argument("--out_dir", default="")
    parser.add_argument("--title", default="Python generator")
    parser.add_argument("--matlab_png", default="", help="MATLAB coverage report to place above ours")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--edge_samples", type=int, default=20000, help="samples used for edge-level panels")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.normpath(args.cache_dir)), "analysis")
    os.makedirs(out_dir, exist_ok=True)
    c = load_cache(args.cache_dir)
    g = graph_features(c)
    s = sample_features(c, g)
    e = edge_sample(c, s, args.edge_samples, np.random.default_rng(args.seed))

    lines = summary_lines(c, g, s, args.title)
    print("\n".join(lines))
    with open(os.path.join(out_dir, "summary.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    coverage_path = os.path.join(out_dir, "coverage_report.png")
    coverage_figure(g, s, args.title, coverage_path, args.dpi)
    properties_figure(s, e, args.title, os.path.join(out_dir, "data_properties.png"), args.dpi)
    if args.matlab_png:
        comparison_figure(args.matlab_png, coverage_path, os.path.join(out_dir, "comparison.png"))
    print(f"wrote figures and summary to {out_dir}")


if __name__ == "__main__":
    main()

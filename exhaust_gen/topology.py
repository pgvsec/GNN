"""Signed strongly connected digraphs for Corpus A.

Sections
  1. Constants and graph predicates (strong connectivity, feedback loops, canonical key)
  2. Construction: exhaustive for n <= 3, ear-decomposition sampling for n >= 4
  3. Stratified selection with quotas and train / validation / test splits
  4. Storage as flat arrays and dense adjacency reconstruction

Definitions
  A gene regulation graph has an edge i -> j with sign +1 (activation) or -1 (repression).
  Self-loops are allowed, multi-edges are not. Every SCC of a GRN is strongly connected, so
  only strongly connected graphs are generated.
  Thomas class (tc): 1 negative loops only, 2 positive loops only, 3 both kinds. A strongly
  connected graph on at least two genes always contains a loop, so class 0 cannot occur.
  Density tier (dt): mean in-degree m / n, 1 sparse (<= 1.5), 2 medium (<= 3.5),
  3 dense (above 3.5, capped at 5.0).
  Edge order: edges are stored in row-major order of (source, target). That is the order
  produced by boolean masking of a dense [source, target] array, and the generator, the cache
  and the dataset all rely on it.

MODIFIED (exhaust_gen):
  - Added enumerate_exhaustive_chunked for n=4 (43M candidates, memory-safe chunking)
  - build_graphs and build_and_save accept exhaust_n (default 4) to extend exhaustive
    enumeration beyond n=3
"""
import hashlib

import numpy as np

N_MIN, N_MAX = 2, 10
TC_NEG, TC_POS, TC_BOTH = 1, 2, 3
CLASS_FRACS = {TC_NEG: 0.15, TC_POS: 0.25, TC_BOTH: 0.60}
DENSITY_FRACS = {1: 0.40, 2: 0.35, 3: 0.25}
P_SELF_LOOP = 0.25            # chance that an extra edge is a self-loop
P_REPRESSION = 0.40           # chance that an unconstrained edge is repressive
SPLIT_FRACS = (0.70, 0.15, 0.15)
DFS_STEP_CAP = 200000         # candidates whose loop search exceeds this are discarded


# ---------------------------------------------------------------------------
# 1. Predicates
# ---------------------------------------------------------------------------
class _Done(Exception):
    pass


class _Cap(Exception):
    pass


def strongly_connected(adj):
    """adj: [M, n, n] bool. Reachability by repeated squaring of (A or I)."""
    n = adj.shape[-1]
    reach = (adj | np.eye(n, dtype=bool)).astype(np.float32)
    for _ in range(int(np.ceil(np.log2(max(n, 2)))) + 1):
        reach = (np.matmul(reach, reach) > 0).astype(np.float32)
    return reach.reshape(reach.shape[0], -1).min(axis=1) > 0


def feedback_loops(sign, step_cap=DFS_STEP_CAP):
    """Whether the graph has a positive and a negative simple cycle.

    Depth-first enumeration of elementary cycles, each rooted at its smallest node, with the
    sign accumulated along the path. Stops as soon as both kinds are found. Closed walks are
    not used: a positive walk can be two negative cycles joined at a node, which is not a
    positive feedback loop. Returns (has_pos, has_neg), or None if the step cap is exceeded."""
    n = sign.shape[0]
    out = [[(j, int(sign[i, j])) for j in range(n) if sign[i, j] != 0] for i in range(n)]
    found = [False, False]
    steps = [0]
    on_path = [False] * n

    def dfs(start, u, s):
        for v, e in out[u]:
            steps[0] += 1
            if steps[0] > step_cap:
                raise _Cap
            ns = s * e
            if v == start:
                found[0 if ns > 0 else 1] = True
                if found[0] and found[1]:
                    raise _Done
            elif v > start and not on_path[v]:
                on_path[v] = True
                dfs(start, v, ns)
                on_path[v] = False

    try:
        for start in range(n):
            on_path[start] = True
            dfs(start, start, 1)
            on_path[start] = False
    except _Done:
        pass
    except _Cap:
        return None
    return found[0], found[1]


def loop_class(has_pos, has_neg):
    if has_pos and has_neg:
        return TC_BOTH
    return TC_POS if has_pos else TC_NEG


def canonical_key(sign):
    """Isomorphism-invariant 64-bit key of a signed digraph (Weisfeiler-Lehman refinement).

    Isomorphic graphs always receive the same key, so splitting by key can never leak a graph
    across train / validation / test. Non-isomorphic graphs that refinement cannot tell apart
    share a key and one of them is dropped, which only costs a rare topology."""
    n = sign.shape[0]
    ins = [[(j, int(sign[j, i])) for j in range(n) if j != i and sign[j, i] != 0] for i in range(n)]
    outs = [[(j, int(sign[i, j])) for j in range(n) if j != i and sign[i, j] != 0] for i in range(n)]
    colors = [(int(sign[i, i]), len(ins[i]), len(outs[i]),
               sum(e > 0 for _, e in ins[i]), sum(e > 0 for _, e in outs[i])) for i in range(n)]
    for _ in range(n):
        colors = [hash((colors[i],
                        tuple(sorted((colors[j], e) for j, e in ins[i])),
                        tuple(sorted((colors[j], e) for j, e in outs[i]))))
                  for i in range(n)]
    digest = hashlib.blake2b(repr((n, sorted(colors))).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def tier_of(m, n):
    d = m / n
    return 1 if d <= 1.5 else (2 if d <= 3.5 else 3)


def edge_range(n, tier):
    """Edge counts m that fall in a density tier, bounded by n * n slots and mean in-degree 5."""
    lo, hi = {1: (n, int(1.5 * n)),
              2: (int(1.5 * n) + 1, int(3.5 * n)),
              3: (int(3.5 * n) + 1, min(n * n, int(5.0 * n)))}[tier]
    return lo, min(hi, n * n)


def make_graph(sign, loops, key):
    n = sign.shape[0]
    m = int(np.count_nonzero(sign))
    return {"n": n, "sign": sign, "tc": loop_class(*loops), "dt": tier_of(m, n),
            "pos": bool(loops[0]), "neg": bool(loops[1]), "key": key, "split": 0}


# ---------------------------------------------------------------------------
# 2. Construction
# ---------------------------------------------------------------------------
def enumerate_exhaustive(n, seen):
    """Every strongly connected signed digraph on n <= 3 genes, one per isomorphism class."""
    slots = n * n
    codes = np.arange(3 ** slots, dtype=np.int64)
    digits = (codes[:, None] // (3 ** np.arange(slots, dtype=np.int64))[None, :]) % 3
    sign = np.where(digits == 1, 1, np.where(digits == 2, -1, 0)).astype(np.int8).reshape(-1, n, n)
    sign = sign[strongly_connected(sign != 0)]
    out = []
    for s in sign:
        key = canonical_key(s)
        if key in seen:
            continue
        seen.add(key)
        loops = feedback_loops(s)
        if loops is None:
            continue
        out.append(make_graph(s, loops, key))
    return out


def enumerate_exhaustive_chunked(n, seen, chunk_size=2_000_000):
    """All non-isomorphic SC signed digraphs on n genes, using chunked iteration.

    For n <= 3 delegates to enumerate_exhaustive. For n=4 (3^16 = 43M candidates)
    the full array does not fit in RAM at once, so candidates are processed in
    chunks of chunk_size rows. For n >= 5 this is infeasible (847B+ candidates)
    and the function raises ValueError.
    """
    if n <= 3:
        return enumerate_exhaustive(n, seen)
    slots = n * n
    total = 3 ** slots
    if total > 50_000_000_000:
        raise ValueError(
            f"enumerate_exhaustive_chunked: n={n} has {total:,} candidates — "
            "full exhaustive enumeration is infeasible. Use sample_graphs instead."
        )
    exponents = (3 ** np.arange(slots, dtype=np.int64))
    out = []
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        codes = np.arange(start, end, dtype=np.int64)
        digits = ((codes[:, None] // exponents[None, :]) % 3).astype(np.int8)
        sign = np.where(digits == 1, np.int8(1), np.where(digits == 2, np.int8(-1), np.int8(0)))
        sign = sign.reshape(-1, n, n)
        sign = sign[strongly_connected(sign != 0)]
        for s in sign:
            key = canonical_key(s)
            if key in seen:
                continue
            seen.add(key)
            loops = feedback_loops(s)
            if loops is None:
                continue
            out.append(make_graph(s, loops, key))
    return out


def random_sc_edges(n, m, p_self, rng):
    """A strongly connected digraph with n nodes and exactly m distinct edges.

    Built by ear decomposition, which reaches every strongly connected digraph: the first ear
    is a cycle through node 0, each later ear is a path between two existing nodes through new
    nodes, and the remaining edges are chords between existing nodes. With n - 1 new nodes
    spread over c ears and E = m - (n - 1) ears in total, the edge count is exactly m.
    Returns a list of (source, target) after a random relabeling, or None if no chord fits."""
    n_ears = m - (n - 1)
    if n_ears < 1:
        return None
    c = int(rng.integers(1, min(n_ears, n - 1) + 1))
    cuts = np.sort(rng.choice(np.arange(1, n - 1), size=c - 1, replace=False)) if c > 1 else []
    sizes = np.diff(np.concatenate([[0], cuts, [n - 1]])).astype(int)
    edges, used = set(), 1
    for k in sizes:
        u = int(rng.integers(0, used))
        v = u if used == 1 else int(rng.integers(0, used))
        path = [u] + list(range(used, used + int(k))) + [v]
        edges.update(zip(path[:-1], path[1:]))
        used += int(k)
    for _ in range(n_ears - c):
        for _attempt in range(60):
            u, v = int(rng.integers(0, n)), int(rng.integers(0, n))
            if u == v and rng.random() >= p_self:
                continue
            if (u, v) not in edges:
                edges.add((u, v))
                break
        else:
            return None
    perm = rng.permutation(n)
    return [(int(perm[a]), int(perm[b])) for a, b in edges]


def signed_matrix(n, edges, balanced, p_rep, rng):
    """Dense sign matrix [source, target].

    balanced=True gives every gene a hidden sign s_i and every edge the sign s_i * s_j, so
    every cycle is positive (no negative feedback). balanced=False draws signs independently."""
    sign = np.zeros((n, n), dtype=np.int8)
    if balanced:
        node = rng.choice(np.array([-1, 1], dtype=np.int8), size=n)
        for a, b in edges:
            sign[a, b] = node[a] * node[b]
    else:
        neg = rng.random(len(edges)) < p_rep
        for (a, b), r in zip(edges, neg):
            sign[a, b] = -1 if r else 1
    return sign


def sample_cell(n, tier, tc, want, seen, rng, attempts):
    """Up to `want` new graphs of one (density tier, Thomas class) cell."""
    lo, hi = edge_range(n, tier)
    out = []
    if lo > hi or want <= 0:
        return out
    for _ in range(attempts * want):
        if len(out) >= want:
            break
        edges = random_sc_edges(n, int(rng.integers(lo, hi + 1)), P_SELF_LOOP, rng)
        if edges is None:
            continue
        sign = signed_matrix(n, edges, tc == TC_POS, P_REPRESSION, rng)
        assert strongly_connected((sign != 0)[None])[0], "ear construction must be strongly connected"
        loops = (True, False) if tc == TC_POS else feedback_loops(sign)
        if loops is None or loop_class(*loops) != tc:
            continue
        key = canonical_key(sign)
        if key in seen:
            continue
        seen.add(key)
        out.append(make_graph(sign, loops, key))
    return out


# ---------------------------------------------------------------------------
# 3. Stratified selection
# ---------------------------------------------------------------------------
def sample_graphs(n, quota, seen, rng):
    """Quota split over (Thomas class x density tier) cells; shortfalls, mostly in the
    negative-only cells where such graphs are rare, are filled with unconstrained graphs."""
    out = []
    for tc, cf in CLASS_FRACS.items():
        for tier, df in DENSITY_FRACS.items():
            want = int(round(quota * cf * df))
            out += sample_cell(n, tier, tc, want, seen, rng, attempts=300 if tc == TC_NEG else 80)
    deficit = quota - len(out)
    if deficit > 0:
        tiers = rng.choice(list(DENSITY_FRACS), size=deficit, p=list(DENSITY_FRACS.values()))
        for tier in tiers:
            out += sample_cell(n, int(tier), TC_BOTH, 1, seen, rng, attempts=200)
    return out


def assign_splits(graphs, rng):
    """Split by graph within each n: 1 train, 2 validation, 3 test."""
    by_n = {}
    for i, g in enumerate(graphs):
        by_n.setdefault(g["n"], []).append(i)
    for idx in by_n.values():
        idx = [int(i) for i in rng.permutation(idx)]
        n_va = int(round(SPLIT_FRACS[1] * len(idx)))
        n_tr = int(round(SPLIT_FRACS[0] * len(idx)))
        if len(idx) >= 3:
            n_va = max(n_va, 1)
            n_tr = min(n_tr, len(idx) - n_va - 1)
        for j, i in enumerate(idx):
            graphs[i]["split"] = 1 if j < n_tr else (2 if j < n_tr + n_va else 3)


def build_graphs(topos_per_n, seed, exhaust_n=3, n_max=N_MAX):
    """topos_per_n holds the quota for n = 4 .. n_max; n = 2..exhaust_n are enumerated exhaustively.

    n_max (default N_MAX=10) caps which gene counts are generated. Set n_max=6 to skip
    n=7-10 entirely and redirect that computation time to smaller gene counts.
    exhaust_n=3 reproduces the original behavior; exhaust_n=4 adds chunked enumeration
    for n=4 (~10-30 min, produces the full non-isomorphic set).
    """
    rng = np.random.default_rng(seed)
    graphs = []
    for n in range(N_MIN, min(n_max, N_MAX) + 1):
        seen = set()
        if n <= exhaust_n:
            found = enumerate_exhaustive_chunked(n, seen)
        else:
            found = sample_graphs(n, topos_per_n[n - 4], seen, rng)
        graphs.extend(found)
        print(f"  n={n:2d}: {len(found):6d} graphs", flush=True)
    assign_splits(graphs, rng)
    return graphs


def report_graphs(graphs):
    print(f"graphs: {len(graphs)}")
    print("per n:  graphs |  neg-only  pos-only  both |  sparse  medium  dense | train/val/test")
    for n in range(N_MIN, N_MAX + 1):
        sub = [g for g in graphs if g["n"] == n]
        if not sub:
            continue
        tc = [sum(g["tc"] == c for g in sub) for c in (TC_NEG, TC_POS, TC_BOTH)]
        dt = [sum(g["dt"] == c for g in sub) for c in (1, 2, 3)]
        sp = [sum(g["split"] == c for g in sub) for c in (1, 2, 3)]
        print(f"  n={n:2d}: {len(sub):6d} | {tc[0]:8d} {tc[1]:8d} {tc[2]:5d} | "
              f"{dt[0]:6d} {dt[1]:7d} {dt[2]:6d} | {sp[0]}/{sp[1]}/{sp[2]}")


# ---------------------------------------------------------------------------
# 4. Storage
# ---------------------------------------------------------------------------
def save_topologies(graphs, path):
    ptr = [0]
    src, tgt, sgn = [], [], []
    for g in graphs:
        a, b = np.nonzero(g["sign"])                    # row-major (source, target) order
        src.append(a)
        tgt.append(b)
        sgn.append(g["sign"][a, b])
        ptr.append(ptr[-1] + a.size)
    np.savez(path,
             t_n=np.array([g["n"] for g in graphs], dtype=np.uint8),
             t_edge_ptr=np.array(ptr, dtype=np.int64),
             t_src=np.concatenate(src).astype(np.uint8),
             t_tgt=np.concatenate(tgt).astype(np.uint8),
             t_sgn=np.concatenate(sgn).astype(np.int8),
             t_tc=np.array([g["tc"] for g in graphs], dtype=np.int8),
             t_dt=np.array([g["dt"] for g in graphs], dtype=np.uint8),
             t_pos=np.array([g["pos"] for g in graphs], dtype=bool),
             t_neg=np.array([g["neg"] for g in graphs], dtype=bool),
             t_split=np.array([g["split"] for g in graphs], dtype=np.uint8),
             t_key=np.array([g["key"] for g in graphs], dtype=np.uint64))


def load_topologies(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def dense_adjacency(topo, graph_ids, n):
    """Signed adjacency [len(graph_ids), n, n] int8, indexed [source, target]."""
    adj = np.zeros((len(graph_ids), n, n), dtype=np.int8)
    ptr = topo["t_edge_ptr"]
    for i, g in enumerate(graph_ids):
        lo, hi = int(ptr[g]), int(ptr[g + 1])
        adj[i, topo["t_src"][lo:hi].astype(np.int64), topo["t_tgt"][lo:hi].astype(np.int64)] = \
            topo["t_sgn"][lo:hi]
    return adj


def validate_graphs(graphs):
    """Re-check every invariant the generator relies on; raises on the first violation.

    Strong connectivity is re-tested in batch, loop flags are recomputed from the sign matrix,
    keys must be unique within each n (this is what keeps isomorphic graphs out of different
    splits), tiers must match the edge count, and sampled positive-only graphs must be
    balanced, that is, admit gene signs s with sign[i, j] = s_i * s_j."""
    for n in range(N_MIN, N_MAX + 1):
        sub = [g for g in graphs if g["n"] == n]
        if not sub:
            continue
        stack = np.stack([g["sign"] for g in sub])
        if not strongly_connected(stack != 0).all():
            raise RuntimeError(f"n={n}: a graph is not strongly connected")
        keys = [g["key"] for g in sub]
        if len(set(keys)) != len(keys):
            raise RuntimeError(f"n={n}: duplicate canonical keys")
        for g in sub:
            if g["tc"] == TC_POS and n >= 4:
                # sampled positive-only graphs are balanced; balance proves that no cycle is
                # negative without enumerating the (possibly very many) cycles of a dense graph
                if not _balanced(g["sign"]) or not g["pos"] or g["neg"]:
                    raise RuntimeError(f"n={n}: positive-only graph is not balanced")
            else:
                loops = feedback_loops(g["sign"])
                if loops is None or (g["pos"], g["neg"]) != loops or g["tc"] != loop_class(*loops):
                    raise RuntimeError(f"n={n}: stored loop flags disagree with the sign matrix")
            if g["dt"] != tier_of(int(np.count_nonzero(g["sign"])), n):
                raise RuntimeError(f"n={n}: density tier disagrees with the edge count")
            if g["split"] not in (1, 2, 3):
                raise RuntimeError(f"n={n}: split not assigned")


def _balanced(sign):
    """True if gene signs exist with sign[i, j] = s_i * s_j on every edge (two-coloring of the
    underlying undirected graph)."""
    n = sign.shape[0]
    color = [0] * n
    for root in range(n):
        if color[root]:
            continue
        color[root] = 1
        stack = [root]
        while stack:
            u = stack.pop()
            for v in range(n):
                e = int(sign[u, v]) if sign[u, v] != 0 else int(sign[v, u])
                if e == 0 or v == u:
                    continue
                want = color[u] * e
                if color[v] == 0:
                    color[v] = want
                    stack.append(v)
                elif color[v] != want:
                    return False
    return True


def describe_saved(path):
    """Counts of a saved topology file by n, Thomas class, density tier and split."""
    t = load_topologies(path)
    print(f"{path}: {t['t_n'].size} graphs, {t['t_src'].size} edges")
    print("  n | graphs | neg-only pos-only both | sparse medium dense | train val test | mean edges")
    for n in range(N_MIN, N_MAX + 1):
        m = t["t_n"] == n
        if not m.any():
            continue
        tc, dt, sp = t["t_tc"][m], t["t_dt"][m], t["t_split"][m]
        n_e = np.diff(t["t_edge_ptr"])[m]
        print(f" {n:2d} | {int(m.sum()):6d} | {int((tc == 1).sum()):8d} {int((tc == 2).sum()):8d} "
              f"{int((tc == 3).sum()):4d} | {int((dt == 1).sum()):6d} {int((dt == 2).sum()):6d} "
              f"{int((dt == 3).sum()):5d} | {int((sp == 1).sum()):5d} {int((sp == 2).sum()):3d} "
              f"{int((sp == 3).sum()):4d} | {n_e.mean():.2f}")


def build_and_save(topos_per_n, seed, path, exhaust_n=3, n_max=N_MAX):
    print("building topologies", flush=True)
    graphs = build_graphs(topos_per_n, seed, exhaust_n=exhaust_n, n_max=n_max)
    validate_graphs(graphs)
    report_graphs(graphs)
    save_topologies(graphs, path)
    print(f"wrote {path}", flush=True)

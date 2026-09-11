"""
Network metrics on weighted directed adjacency matrices (W[i, j] = i -> j weight):
Fagiolo clustering, inverse-weight betweenness and efficiency, algebraic connectivity,
footprint entropy, vulnerability, GSCC size, block-normalised two-layer supra-adjacency,
PageRank versatility, pre/post matrix correlation, rate-based density. Three separate
normalisations: clustering scales by max(w), density uses per-minute rates, supra blocks sum to one.
Inputs: numpy adjacency matrices (rows = source, cols = target). Outputs: scalars or per-node vectors.
Run: library only, imported by code/pipeline/04b_window_convergence.py and 06_metrics.py.
"""

import networkx as nx
import numpy as np


def _to_digraph_inv_w(W: np.ndarray) -> nx.DiGraph:
    """DiGraph with dist = 1/w on each edge; raw weights as distances would invert the path ranking."""
    G = nx.DiGraph()
    n = W.shape[0]
    G.add_nodes_from(range(n))
    for i, j in zip(*np.nonzero(W)):
        G.add_edge(int(i), int(j), weight=float(W[i, j]), dist=1.0 / float(W[i, j]))
    return G


def fagiolo_clustering(W: np.ndarray) -> np.ndarray:
    """Fagiolo (2007) directed weighted clustering per node.
    C_i = [W^(1/3) + (W^T)^(1/3)]^3_ii / (2 [d_tot (d_tot - 1) - 2 d_bi]), W scaled by max(W);
    binary degrees in the denominator; zero denominator -> 0."""
    n = W.shape[0]
    if W.max() == 0:
        return np.zeros(n)
    assert np.isfinite(W).all(), "fagiolo_clustering: non-finite input"
    A = (W > 0).astype(float)
    d_tot = A.sum(axis=0) + A.sum(axis=1)  # in + out
    d_bi = (A * A.T).sum(axis=1)  # bidirectional edges
    denom = 2.0 * (d_tot * (d_tot - 1) - 2 * d_bi)
    What = np.cbrt(W / W.max())
    B = What + What.T
    # macOS Accelerate emits a spurious RuntimeWarning for matmul >= 24 dims on finite input
    with np.errstate(all="ignore"):
        numer = np.diagonal(np.linalg.matrix_power(B, 3))
        c = np.where(denom > 0, numer / denom, 0.0)
    return c


def fagiolo_clustering_binary(W: np.ndarray) -> np.ndarray:
    """Binary Fagiolo clustering, reference for validation."""
    A = (W > 0).astype(float)
    d_tot = A.sum(axis=0) + A.sum(axis=1)
    d_bi = (A * A.T).sum(axis=1)
    denom = d_tot * (d_tot - 1) - 2 * d_bi
    B = A + A.T
    t = np.diagonal(np.linalg.matrix_power(B, 3)) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom > 0, t / denom, 0.0)


def betweenness_inv_w(W: np.ndarray, normalized: bool = False) -> np.ndarray:
    """Directed betweenness with distance 1/w; raw path counts unless normalized."""
    G = _to_digraph_inv_w(W)
    bc = nx.betweenness_centrality(G, weight="dist", normalized=normalized)
    return np.array([bc[i] for i in range(W.shape[0])])


def efficiency_inv_w(W: np.ndarray) -> float:
    """Global efficiency with distance 1/w: sum_{i != j} (1/d_ij) / (N (N - 1)); unreachable pairs count 0."""
    G = _to_digraph_inv_w(W)
    n = W.shape[0]
    if n < 2:
        return 0.0
    total = 0.0
    for src, lengths in nx.all_pairs_dijkstra_path_length(G, weight="dist"):
        for dst, d in lengths.items():
            if dst != src and d > 0:
                total += 1.0 / d
    return total / (n * (n - 1))


def lambda2_sym(W: np.ndarray) -> float:
    """Algebraic connectivity: second-smallest eigenvalue of L = D - M, M = (W + W^T) / 2,
    on the subgraph of active nodes (strength > 0). Symmetrised, unlike the out-strength
    Laplacian of Buldu et al. (2019). Fixed-node layers (24 zones) nearly always contain
    isolated nodes, so the full-set value is identically 0; 0 on the active subgraph is
    real fragmentation."""
    M = (W + W.T) / 2.0
    s = M.sum(axis=1)
    active = s > 0
    if active.sum() < 2:
        return 0.0
    M = M[np.ix_(active, active)]
    L = np.diag(M.sum(axis=1)) - M
    eig = np.sort(np.linalg.eigvalsh(L))
    lam2 = float(eig[1])
    return lam2 if abs(lam2) > 1e-12 else 0.0  # clamp numerical noise


def footprint_entropy(w: np.ndarray) -> float:
    """H(i) = -sum_k p_k ln p_k, p_k = w_k / sum(w), w = zone coupling vector of player i; all-zero -> 0."""
    w = np.asarray(w, dtype=float)
    s = w.sum()
    if s <= 0:
        return 0.0
    p = w[w > 0] / s
    return float(-(p * np.log(p)).sum())


def vulnerability(W: np.ndarray, node: int) -> float:
    """v_i = (C_{-i} - C) / C with C = mean Fagiolo clustering (Guo et al. 2022, Eq. 4); C = 0 -> nan."""
    c_full = float(np.mean(fagiolo_clustering(W)))
    if c_full == 0:
        return float("nan")
    keep = [j for j in range(W.shape[0]) if j != node]
    W_minus = W[np.ix_(keep, keep)]
    c_minus = float(np.mean(fagiolo_clustering(W_minus))) if W_minus.shape[0] else 0.0
    return (c_minus - c_full) / c_full


def gscc_size(W: np.ndarray) -> int:
    """Size of the largest strongly connected component; weak connectivity would overstate robustness."""
    G = nx.DiGraph()
    n = W.shape[0]
    G.add_nodes_from(range(n))
    G.add_edges_from((int(i), int(j)) for i, j in zip(*np.nonzero(W)))
    if n == 0:
        return 0
    return max(len(c) for c in nx.strongly_connected_components(G))


def build_supra(A_P: np.ndarray, A_Z: np.ndarray, C: np.ndarray,
                omega: float = 1.0) -> np.ndarray:
    """S = [[A_P, omega C], [omega C^T, A_Z]], each block divided by its total weight.
    Serves the cross-layer versatility only."""
    n_p, n_z = A_P.shape[0], A_Z.shape[0]
    assert C.shape == (n_p, n_z), "C must be (N_p, N_z)"
    def norm(M):
        s = M.sum()
        return M / s if s > 0 else M
    S = np.zeros((n_p + n_z, n_p + n_z))
    S[:n_p, :n_p] = norm(A_P)
    S[n_p:, n_p:] = norm(A_Z)
    S[:n_p, n_p:] = omega * norm(C)
    S[n_p:, :n_p] = omega * norm(C).T
    return S


def pagerank_versatility(S: np.ndarray, r: float = 0.85,
                         tol: float = 1e-12, max_iter: int = 1000) -> np.ndarray:
    """PageRank versatility with teleportation r: row-normalised S, power iteration,
    uniform dangling rows. Returns an (N_p + N_z) vector summing to 1."""
    n = S.shape[0]
    rowsum = S.sum(axis=1, keepdims=True)
    T = np.divide(S, rowsum, out=np.full_like(S, 1.0 / n), where=rowsum > 0)
    v = np.full(n, 1.0 / n)
    # macOS Accelerate emits a spurious RuntimeWarning for matmul >= 24 dims;
    # checked against networkx pagerank to 1e-13 at the production size (38 nodes)
    with np.errstate(all="ignore"):
        for _ in range(max_iter):
            v_new = r * (v @ T) + (1 - r) / n
            if np.abs(v_new - v).sum() < tol:
                return v_new
            v = v_new
    raise RuntimeError("pagerank_versatility did not converge")


def matrix_correlation(W1: np.ndarray, W2: np.ndarray) -> float:
    """Pearson correlation of the two matrices vectorised without the diagonal
    (Garrido et al. 2020; Xiong et al. 2025); nan if either side has zero variance."""
    assert W1.shape == W2.shape
    mask = ~np.eye(W1.shape[0], dtype=bool)
    a, b = W1[mask].astype(float), W2[mask].astype(float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def density_rate(count_matrix: np.ndarray, minutes: float) -> float:
    """Weighted density with w_ij = count / minutes: sum(w) / (N (N - 1)), passes per ordered
    pair per minute. minutes=None (50-pass windows) falls back to raw counts."""
    n = count_matrix.shape[0]
    if n < 2:
        return 0.0
    w = count_matrix / minutes if minutes else count_matrix.astype(float)
    np_pairs = n * (n - 1)
    return float(w.sum() / np_pairs)

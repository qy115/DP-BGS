import os
import json
import time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh
from itertools import combinations

def graph_sizes(graph):
    
    n_u = graph.get("num_U", len(graph["U"]))
    n_v = graph.get("num_V", len(graph["V"]))
    return int(n_u), int(n_v)


def degree_sequences(graph):
    n_u, n_v = graph_sizes(graph)

    if "degree_U" in graph and "degree_V" in graph:
        du = np.asarray(graph["degree_U"], dtype=np.int64)
        dv = np.asarray(graph["degree_V"], dtype=np.int64)
        if len(du) == n_u and len(dv) == n_v:
            return du, dv

    du = np.zeros(n_u, dtype=np.int64)
    dv = np.zeros(n_v, dtype=np.int64)
    for u, v in graph["edges"]:
        if 0 <= u < n_u and 0 <= v < n_v:
            du[u] += 1
            dv[v] += 1
    return du, dv


def edge_count(graph):
    return len(graph["edges"])


def graph_density(graph):
    n_u, n_v = graph_sizes(graph)
    possible_edges = n_u * n_v
    return edge_count(graph) / possible_edges if possible_edges else 0.0


def relative_error(original, synthetic):
    if original == 0:
        return 0.0 if synthetic == 0 else 1.0
    return abs(synthetic - original) / abs(original)


def evaluate_one_metric(name, original_value, synthetic_value):
    result = {
        "original": original_value,
        "synthetic": synthetic_value,
        "relative_error": relative_error(original_value, synthetic_value),
    }
    print(
        f"{name}: original={original_value}, "
        f"synthetic={synthetic_value}, "
        f"relative_error={result['relative_error']}"
    )
    return result

def degree_distribution_similarity(d1, d2):
    d1 = np.asarray(d1, dtype=np.int64)
    d2 = np.asarray(d2, dtype=np.int64)
    max_degree = max(
        int(d1.max()) if d1.size else 0,
        int(d2.max()) if d2.size else 0,
    )

    hist1 = np.bincount(d1, minlength=max_degree + 1).astype(float)
    hist2 = np.bincount(d2, minlength=max_degree + 1).astype(float)

    sum1, sum2 = hist1.sum(), hist2.sum()
    if sum1:
        hist1 /= sum1
    if sum2:
        hist2 /= sum2

    # Two empty distributions are treated as identical.
    if sum1 == 0 and sum2 == 0:
        return 1.0
    return float(1.0 - 0.5 * np.abs(hist1 - hist2).sum())


def degree_distribution_kl(d1, d2, smoothing=1e-12):
 
    d1 = np.asarray(d1, dtype=np.int64)
    d2 = np.asarray(d2, dtype=np.int64)
    max_degree = max(
        int(d1.max()) if d1.size else 0,
        int(d2.max()) if d2.size else 0,
    )

    p = np.bincount(d1, minlength=max_degree + 1).astype(float)
    q = np.bincount(d2, minlength=max_degree + 1).astype(float)

    p += smoothing
    q += smoothing
    p /= p.sum()
    q /= q.sum()

    return float(np.sum(p * np.log(p / q)))



def spectral_similarity(g1, g2, k=20):
   
    def spectrum(graph):
        n_u, n_v = graph_sizes(graph)
        n = n_u + n_v
        if n == 0:
            return np.array([], dtype=float)

        rows = []
        cols = []
        for u, v in graph["edges"]:
            if 0 <= u < n_u and 0 <= v < n_v:
                rows.extend((u, n_u + v))
                cols.extend((n_u + v, u))

        if not rows:
            return np.zeros(min(k, n), dtype=float)

        adjacency = coo_matrix(
            (np.ones(len(rows), dtype=float), (rows, cols)),
            shape=(n, n),
        ).tocsr()

        # eigsh requires k < n. For small graphs, use a dense eigensolver.
        if n <= 2 or min(k, n - 1) >= n - 1:
            values = np.linalg.eigvalsh(adjacency.toarray())
            values = values[np.argsort(np.abs(values))[::-1][:min(k, n)]]
        else:
            values = eigsh(
                adjacency,
                k=min(k, n - 2),
                which="LM",
                return_eigenvectors=False,
            )
        return np.sort(values)

    s1 = spectrum(g1)
    s2 = spectrum(g2)

  
    length = max(len(s1), len(s2))
    s1 = np.pad(s1, (0, length - len(s1)))
    s2 = np.pad(s2, (0, length - len(s2)))

    return float(1.0 - np.linalg.norm(s1 - s2) / max(np.linalg.norm(s1), 1.0))


def common_neighbor_mean(graph):
  
    n_u, n_v = graph_sizes(graph)
    neighbors = [set() for _ in range(n_u)]

    for u, v in graph["edges"]:
        if 0 <= u < n_u and 0 <= v < n_v:
            neighbors[u].add(v)

    pair_count = n_u * (n_u - 1) // 2
    if pair_count == 0:
        return 0.0

    common_sum = 0
    
    for u1, u2 in combinations(range(n_u), 2):
        common_sum += len(neighbors[u1] & neighbors[u2])

    return float(common_sum / pair_count)


def butterfly_count(graph):
   
    n_u, n_v = graph_sizes(graph)
    neighbors = [set() for _ in range(n_u)]

    for u, v in graph["edges"]:
        if 0 <= u < n_u and 0 <= v < n_v:
            neighbors[u].add(v)

    count = 0
    for u1, u2 in combinations(range(n_u), 2):
        common = len(neighbors[u1] & neighbors[u2])
        count += common * (common - 1) // 2
    return int(count)


def save_json(data, path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


ORIGINAL_CACHE_KEYS = {
    "edge_num",
    "density",
    "U_degree",
    "V_degree",
    "butterfly",
    "common_neighbor_mean",
}


def compute_original_metrics(graph, cal_butterfly=True, cal_neighbor=True):
    start = time.time()
    du, dv = degree_sequences(graph)

    metrics = {
        "edge_num": edge_count(graph),
        "density": graph_density(graph),
        "U_degree": du.tolist(),
        "V_degree": dv.tolist(),
        "butterfly": butterfly_count(graph) if cal_butterfly else None,
        "common_neighbor_mean": common_neighbor_mean(graph) if cal_neighbor else None,
    }

    print(
        "Original metrics computed in "
        f"{time.time() - start:.2f}s "
    )
    return metrics



def evaluate_all(
    original_graph,
    synthetic_graph,
    cache_path="evaluation/cache/orig_metrics.json",
    cal_butterfly=True,
    cal_neighbor=True,
):
    """
    Metrics:
      - edge_num
      - density
      - U_degree_similarity / V_degree_similarity
      - U_degree_kl / V_degree_kl
      - spectral
      - common_neighbor
      - butterfly
    """
    if os.path.exists(cache_path):
        try:
            cached = load_json(cache_path)
            cache_valid = ORIGINAL_CACHE_KEYS.issubset(cached.keys())
            cache_valid = cache_valid and (
                not cal_butterfly or cached.get("butterfly") is not None
            )
            cache_valid = cache_valid and (
                not cal_neighbor or cached.get("common_neighbor_mean") is not None
            )
            if cache_valid:
                original_metrics = cached
                print("Loaded cached original metrics.")
            else:
                print("Original metric cache is outdated; recomputing.")
                original_metrics = compute_original_metrics(
                    original_graph, cal_butterfly, cal_neighbor
                )
                save_json(original_metrics, cache_path)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            print("Original metric cache is invalid; recomputing.")
            original_metrics = compute_original_metrics(
                original_graph, cal_butterfly, cal_neighbor
            )
            save_json(original_metrics, cache_path)
    else:
        original_metrics = compute_original_metrics(
            original_graph, cal_butterfly, cal_neighbor
        )
        save_json(original_metrics, cache_path)
        print("Original metrics saved.")

    results = {}

    print("\nEvaluating synthetic graph...")
    start = time.time()

    # Edge count and density
    results["edge_num"] = evaluate_one_metric(
        "Edge number",
        original_metrics["edge_num"],
        edge_count(synthetic_graph),
    )
    results["density"] = evaluate_one_metric(
        "Density",
        original_metrics["density"],
        graph_density(synthetic_graph),
    )

    # U/V degree similarities and KL divergences
    original_du = np.asarray(original_metrics["U_degree"], dtype=np.int64)
    original_dv = np.asarray(original_metrics["V_degree"], dtype=np.int64)
    synthetic_du, synthetic_dv = degree_sequences(synthetic_graph)

    results["U_degree_similarity"] = degree_distribution_similarity(
        original_du, synthetic_du
    )
    results["V_degree_similarity"] = degree_distribution_similarity(
        original_dv, synthetic_dv
    )
    results["U_degree_kl"] = degree_distribution_kl(original_du, synthetic_du)
    results["V_degree_kl"] = degree_distribution_kl(original_dv, synthetic_dv)
    print(
        "U/V degree similarity:",
        results["U_degree_similarity"],
        results["V_degree_similarity"],
    )
    print(
        "U/V degree-distribution KL:",
        results["U_degree_kl"],
        results["V_degree_kl"],
    )

    # Spectral similarity
    spectral = spectral_similarity(original_graph, synthetic_graph)
    results["spectral"] = {"similarity": spectral}
    print(f"Spectral similarity: {spectral}")

    # Butterfly count
    if cal_butterfly:
        t1 = time.time()
        synthetic_butterfly = butterfly_count(synthetic_graph)
        results["butterfly"] = evaluate_one_metric(
            "Butterfly",
            original_metrics["butterfly"],
            synthetic_butterfly,
        )
        print(f'Butterfly counting time:{time.time()-t1:.3f}s')

    if cal_neighbor:
        t1 = time.time()
        synthetic_cn = common_neighbor_mean(synthetic_graph)
        results["common_neighbor"] = evaluate_one_metric(
            "Common neighbor mean",
            original_metrics["common_neighbor_mean"],
            synthetic_cn,
        )
        print(f'Common Neighbor time:{time.time()-t1:.3f}s')

    

    print(f"\nEvaluation completed in {time.time() - start:.2f}s.")
    return results

"""
MUSS: Multilevel Subset Selection for Relevance and Diversity
Jointly selects items that are high quality and high diversity.

Usage (RAG / candidate retrieval)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Input
  x: dataset, 2D array (n items, d features)
  q: quality scores, 1D array (n items)

Steps
  0. Partition the data
     partition_clustering / partition_clustering_by_faiss / partition_random
  1. Select partitions using MMR
     select_partitions_mmr(q, partition_results, m, lambda_part)
  2. Finalize partitions (when clustering was done on a sample)
     get_full_partitions(x, partition_results, selected_cluster_idx)
  3. MMR on partitioned data
     mmr_on_partitions(x, q, k, top_q, lambda_param, partitions)

High-level entry points
  muss(...)                -- Algorithm 2; also used for RAG when partition_results is provided
  pre_process_for_muss(...)-- preprocessing: normalise embeddings and build partitions once per dataset
"""
import logging
import random
import time

import faiss
import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment
from sklearn import preprocessing
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import pairwise_distances
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)
logging.basicConfig(encoding="utf-8", format="%(name)s: %(message)s", level=logging.DEBUG)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MMR_DIVERSITY_TYPE_SUM = "sum"
MMR_DIVERSITY_TYPE_MIN = "min"

PACKING_ALGO_KMEANS = "packing_k_means"
PACKING_ALGO_BALANCED = "packing_balanced"
PACKING_ALGO_BALANCED_FAST = "packing_balanced_fast"
PACKING_ALGO_RANDOM = "packing_random"


# ---------------------------------------------------------------------------
# Partitioning
# ---------------------------------------------------------------------------

def partition_clustering(
        x: np.ndarray,
        n_partitions: int,
        batch_size: int = 1024,
        random_state: int = 321) -> Dict:
    """Partition data using MiniBatchKMeans clustering."""
    kmeans = MiniBatchKMeans(n_clusters=n_partitions, batch_size=batch_size, random_state=random_state)
    kmeans.fit(x)

    partitions = []
    radii = []
    centers = kmeans.cluster_centers_
    non_empty = np.full(n_partitions, True)

    for c in range(n_partitions):
        indices_in_cluster = np.where(kmeans.labels_ == c)[0]
        if indices_in_cluster.size:
            partitions.append(indices_in_cluster)
            x_c = x[indices_in_cluster, :]
            mu_c = centers[c, :].reshape(1, -1)
            d = pairwise_distances(x_c, mu_c, metric="euclidean")
            radii.append(np.max(d))
        else:
            non_empty[c] = False

    if len(partitions) < n_partitions:
        logger.warning(f"empty clusters: {np.where(~non_empty)[0]}")
        centers = centers[non_empty]

    return {
        "partitions": partitions,
        "radii": radii,
        "centers": centers,
    }


def partition_clustering_by_faiss(x: np.ndarray, n_partitions: int, random_state: int = 321) -> Dict:
    """Partition data using FAISS k-means clustering."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    kmeans = faiss.Kmeans(x.shape[1], n_partitions, niter=50, verbose=False, nredo=4, seed=random_state)
    kmeans.train(x)

    _, cluster_assignment = kmeans.index.search(x, 1)

    clusters: List[List[int]] = [[] for _ in range(n_partitions)]
    for idx, cluster_id in enumerate(cluster_assignment.ravel()):
        clusters[cluster_id].append(idx)

    non_empty = [len(c) > 0 for c in clusters]
    partitions = [np.asarray(c) for c, ne in zip(clusters, non_empty) if ne]
    centers = kmeans.centroids[non_empty]

    return {
        "partitions": partitions,
        "centers": centers,
    }


def partition_random(x: np.ndarray, n_partitions: int, random_state: int = 321) -> Dict:
    """Partition data randomly into equally sized groups."""
    rng = np.random.default_rng(random_state)
    cluster_size = x.shape[0] // n_partitions + 1
    partition = np.repeat(np.arange(n_partitions), cluster_size)
    rng.shuffle(partition)
    partition = partition[:x.shape[0]]

    partitions = []
    centers = np.zeros((n_partitions, x.shape[1]))
    for c in range(n_partitions):
        idx_in_cluster_c = np.where(partition == c)[0]
        partitions.append(idx_in_cluster_c)
        centers[c, :] = np.mean(x[idx_in_cluster_c, :], axis=0)

    return {"partitions": partitions, "centers": centers}


# ---------------------------------------------------------------------------
# Core greedy selection (MMR)
# ---------------------------------------------------------------------------

def greedy_selection(
        x: np.ndarray,
        q: np.ndarray,
        k: int,
        weight_q: float,
        weight_d: float = None,
        diversity_type: str = MMR_DIVERSITY_TYPE_SUM,
) -> List:
    """
    Greedy MMR-style selection balancing quality and diversity.

    :param x: 2D array (n items, d features)
    :param q: quality scores, 1D array (n items)
    :param k: number of items to select
    :param weight_q: quality weight (lambda in MMR)
    :param weight_d: diversity weight; defaults to 1 - weight_q
    :param diversity_type: "sum" or "min"
    :return: list of selected item indices
    """
    if weight_d is None:
        weight_d = 1 - weight_q
    if not (0 <= weight_q <= 1):
        raise ValueError(f"invalid weight_q: {weight_q}, should be within [0, 1]")
    if not (0 <= weight_d <= 1):
        raise ValueError(f"invalid weight_d: {weight_d}, should be within [0, 1]")
    if x.shape[0] != q.shape[0]:
        raise ValueError(f"inconsistent shapes: x: {x.shape}, q: {q.shape}")
    if x.shape[0] == 0:
        raise ValueError("empty data")
    if x.shape[0] < k:
        k = x.shape[0]

    t = np.zeros_like(q)
    selected_docs: List[int] = []

    for i in range(k):
        if selected_docs:
            s = x[selected_docs, :]
            d_xs = pairwise_distances(x, s, metric="euclidean")
            if diversity_type == MMR_DIVERSITY_TYPE_SUM:
                diversity = np.sum(d_xs, axis=1) / (i + 1)
            elif diversity_type == MMR_DIVERSITY_TYPE_MIN:
                diversity = np.min(d_xs, axis=1)
            else:
                raise ValueError(f"invalid diversity type: {diversity_type}")
            mmr_scores = t + weight_q * q + weight_d * diversity
        else:
            mmr_scores = q

        selected_idx = int(np.argmax(mmr_scores))
        t[selected_idx] = -np.inf
        selected_docs.append(selected_idx)

    return selected_docs


def _mmr_wrapper(
        partition_id: int,
        x: np.ndarray,
        q: np.ndarray,
        k: int,
        lambda_param: float,
        diversity_type: str,
) -> Tuple:
    """Helper for parallel greedy_selection calls."""
    return partition_id, greedy_selection(x, q, k, weight_q=lambda_param, diversity_type=diversity_type)


# ---------------------------------------------------------------------------
# Partition-level operations
# ---------------------------------------------------------------------------

def select_partitions_mmr(
        q: np.ndarray,
        partition_results: Dict,
        m: int,
        lambda_part: float = 0.9,
        diversity_type: str = MMR_DIVERSITY_TYPE_SUM,
) -> List:
    """Select m partitions using MMR on cluster centroids and median quality scores."""
    partitions = partition_results["partitions"]
    centers = partition_results["centers"]
    cluster_scores = np.array([np.median(q[p]) for p in partitions])
    return greedy_selection(centers, cluster_scores, m, weight_q=lambda_part, diversity_type=diversity_type)


def select_within_partitions(
        x: np.ndarray,
        q: np.ndarray,
        partitions: List,
        k: int,
        lambda_within: float = 0.9,
        n_jobs: int = 10,
        add_max_q_to_union: bool = True,
        diversity_type: str = MMR_DIVERSITY_TYPE_SUM,
) -> Set:
    """Run MMR within each partition in parallel and union the results."""
    results = Parallel(n_jobs=n_jobs)(
        delayed(_mmr_wrapper)(i, x[cluster_items, :], q[cluster_items], k, lambda_within, diversity_type)
        for i, cluster_items in enumerate(partitions)
    )

    collected_indices: Set[int] = set()
    for i, selected_items in results:
        collected_indices |= set(partitions[i][selected_items])

    if add_max_q_to_union:
        collected_indices.add(int(q.argmax()))

    return collected_indices


def get_full_partitions(
        x: np.ndarray,
        partition_results: Dict,
        selected_cluster_idx: List,
        by_radius: bool = False,
) -> List:
    """
    For each selected partition, find the closest points to the partition centroid.
    Use by_radius=True to select all points within the estimated cluster radius,
    or by_radius=False (default) to select the top <desired_size> closest points.
    """
    centers = partition_results["centers"]
    partitions = []

    if by_radius:
        assert "radii" in partition_results, "cannot finalize by radius without pre-computed radii"
        radii = partition_results["radii"]
        for c in selected_cluster_idx:
            mu = centers[c, :].reshape(1, -1)
            distances = pairwise_distances(x, mu, metric="euclidean").ravel()
            partitions.append(np.where(distances <= radii[c])[0])
    else:
        desired_size = x.shape[0] // centers.shape[0] + 1
        for c in selected_cluster_idx:
            mu = centers[c, :].reshape(1, -1)
            distances = pairwise_distances(x, mu, metric="euclidean").ravel()
            partitions.append(np.argpartition(distances, desired_size)[:desired_size])

    return partitions


def mmr_on_partitions(
        x: np.ndarray,
        q: np.ndarray,
        k: int,
        top_q: Set,
        lambda_param: float,
        partitions: List,
        n_jobs: int = 10,
        k_within: Optional[int] = None,
        add_max_q_to_union: bool = True,
        diversity_type: str = MMR_DIVERSITY_TYPE_SUM,
) -> Tuple:
    """
    Full MUSS inner pipeline:
      1. MMR within each partition (parallel)
      2. Union with top-quality items
      3. Final MMR pass on the union

    :return: (list of selected indices, diagnostics dict)
    """
    if k_within is None:
        k_within = k

    t0 = time.time()
    collected_indices = select_within_partitions(
        x, q, partitions, k_within, lambda_param, n_jobs, add_max_q_to_union, diversity_type
    )
    time_within = time.time() - t0

    collected_indices |= top_q
    collected_list = list(collected_indices)

    t0 = time.time()
    final_selection = greedy_selection(
        x[collected_list, :], q[collected_list], k,
        weight_q=lambda_param, weight_d=1 - lambda_param, diversity_type=diversity_type,
    )
    final_result = [collected_list[i] for i in final_selection]
    time_union = time.time() - t0

    diagnostics = {
        "time_within": time_within,
        "time_union": time_union,
        "size_union": len(collected_list),
    }
    return final_result, diagnostics


# ---------------------------------------------------------------------------
# Shared core
# ---------------------------------------------------------------------------

def _run_muss_core(
        x: np.ndarray,
        q: np.ndarray,
        partition_results: Dict,
        k: int,
        m: int,
        lambda_clusters: float,
        lambda_points: float,
        k_within: Optional[int] = None,
        n_jobs: int = 10,
        add_max_q_to_union: bool = True,
        diversity_type: str = MMR_DIVERSITY_TYPE_SUM,
) -> Tuple:
    """
    Shared core used by muss() for both candidate retrieval and RAG.
    Selects m partitions, runs within-partition MMR, then a final MMR pass.
    Pass lambda_clusters < 0 to select partitions randomly instead of by MMR.
    """
    n_avail = len(partition_results["partitions"])
    if n_avail == m:
        selected_cluster_idx = list(range(m))
    elif lambda_clusters < 0:
        selected_cluster_idx = random.sample(range(n_avail), k=m)
    else:
        selected_cluster_idx = select_partitions_mmr(
            q, partition_results, m, lambda_clusters, diversity_type
        )

    partitions = [partition_results["partitions"][c] for c in selected_cluster_idx]
    top_q = set(np.argpartition(q, -k)[-k:].tolist())

    return mmr_on_partitions(
        x, q,
        k=k,
        top_q=top_q,
        lambda_param=lambda_points,
        partitions=partitions,
        n_jobs=n_jobs,
        k_within=k_within,
        add_max_q_to_union=add_max_q_to_union,
        diversity_type=diversity_type,
    )


# ---------------------------------------------------------------------------
# High-level entry points
# ---------------------------------------------------------------------------

def muss(
        x: np.ndarray,
        q: np.ndarray,
        k: int,
        k_within: Optional[int] = None,
        m: int = 10,
        n_partitions: Optional[int] = None,
        lamb: float = 0.5,
        lamb_c: float = 0.5,
        n_jobs: int = 10,
        diversity_type: str = MMR_DIVERSITY_TYPE_SUM,
        random_state: int = 321,
        partition_results: Optional[Dict] = None,
        add_max_q_to_union: bool = True,
) -> Tuple:
    """
    MUSS Algorithm 2 from the paper.

    :param x: dataset, 2D array (n items, d features)
    :param q: quality scores, 1D array (n items)
    :param k: number of items to select
    :param k_within: items to select within each cluster (defaults to k)
    :param m: number of partitions to select
    :param n_partitions: total partitions for clustering — required when partition_results is None
    :param lamb: item-level quality/diversity trade-off
    :param lamb_c: cluster-level quality/diversity trade-off
    :param n_jobs: parallel jobs for within-partition MMR
    :param diversity_type: "sum" or "min"
    :param random_state: random seed for FAISS clustering
    :param partition_results: pre-computed partition dict; if provided, clustering is skipped
    :param add_max_q_to_union: always include the highest-quality item in the union
    :return: (list of selected indices, diagnostics dict)
    """
    if partition_results is None:
        if n_partitions is None:
            raise ValueError("n_partitions is required when partition_results is not provided")
        t0 = time.time()
        partition_results = partition_clustering_by_faiss(x, n_partitions=n_partitions, random_state=random_state)
        clustering_time = round(time.time() - t0, 2)
    else:
        clustering_time = 0.0

    selected_idx, diagnostics = _run_muss_core(
        x, q, partition_results,
        k=k, m=m,
        lambda_clusters=lamb_c,
        lambda_points=lamb,
        k_within=k_within,
        n_jobs=n_jobs,
        diversity_type=diversity_type,
        add_max_q_to_union=add_max_q_to_union,
    )
    diagnostics["clustering_time"] = clustering_time
    return selected_idx, diagnostics


def pre_process_for_muss(data_info: Dict, n_clusters: int, algorithm: str) -> Dict:
    """
    Preprocessing pipeline for MUSS RAG retrieval.
    Normalises corpus embeddings and pre-computes partitions (done once per dataset).
    """
    text_embeddings = data_info["text_embeddings"]
    x = preprocessing.normalize(np.array([emb for _, emb in text_embeddings]))

    if n_clusters == 1:
        partition_results = {
            "partitions": [np.arange(x.shape[0])],
            "centers": np.mean(x, axis=0).reshape(1, -1),
        }
    elif algorithm == PACKING_ALGO_KMEANS:
        partition_results = partition_clustering(x, n_clusters)
    elif algorithm == PACKING_ALGO_BALANCED:
        partition_results = partition_clustering(x, n_clusters)
        n_clusters_actual = len(partition_results["partitions"])
        centers = partition_results["centers"]
        cluster_size = x.shape[0] // n_clusters_actual + 1
        repeated = np.repeat(centers, cluster_size, axis=0)
        cluster_assignment = np.array(linear_sum_assignment(pairwise_distances(x, repeated, metric="euclidean"))[1]) // cluster_size
        partitions = []
        centers = np.zeros((n_clusters_actual, x.shape[1]))
        for c in range(n_clusters_actual):
            idx = np.where(cluster_assignment == c)[0]
            partitions.append(idx)
            centers[c, :] = np.mean(x[idx, :], axis=0)
        partition_results["partitions"] = partitions
        partition_results["centers"] = centers
    elif algorithm == PACKING_ALGO_BALANCED_FAST:
        partition_results = partition_clustering(x, n_clusters)
        partitions = get_full_partitions(x, partition_results, list(range(n_clusters)))
        centers = np.zeros((len(partitions), x.shape[1]))
        for c, idx in enumerate(partitions):
            centers[c, :] = np.mean(x[idx, :], axis=0)
        partition_results["partitions"] = partitions
        partition_results["centers"] = centers
    elif algorithm == PACKING_ALGO_RANDOM:
        partition_results = partition_random(x, n_clusters)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    centers = partition_results["centers"]
    max_radius = max(
        np.max(pairwise_distances(x[idx, :], centers[c, :].reshape(1, -1), metric="euclidean"))
        for c, idx in enumerate(partition_results["partitions"])
    )
    max_size = max(idx.shape[0] for idx in partition_results["partitions"])

    stats = {"n_clusters": n_clusters, "max_radius": max_radius, "max_size": max_size}
    return {"x": x, "partition_results": partition_results, "stats": stats}

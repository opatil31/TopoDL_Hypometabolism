"""Python implementation of the Topological K-Means originally written in R from: https://github.com/TopoKmeans/TopoKmeans/tree/master

Dependencies
------------
The only non-standard dependency is ripser for persistent homology. Install
it with:
pip install ripser
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from ripser import ripser


@dataclass
class TopoKMeansResult:
    labels: np.ndarray
    persistence: np.ndarray
    distance_matrix: np.ndarray
    medoids: np.ndarray
    iterations: int


def _rbf_kernel_sum(vec_a: np.ndarray, vec_b: np.ndarray, sigma: float) -> float:
    diff = vec_a[:, None] - vec_b[None, :]
    return float(np.exp(-sigma * diff * diff).sum())


def _compute_neighbors(
    data: np.ndarray, n_knn: int, dist_matrix: bool, preserve_ordering: bool
) -> tuple[List[np.ndarray], float]:
    n_samples = data.shape[0]
    k = max(1, min(n_knn, n_samples))

    if dist_matrix:
        sorted_indices = np.argsort(data, axis=1)
        neighbor_indices = [row[:k] for row in sorted_indices]
        sorted_distances = np.sort(data, axis=1)
        max_scale = float(sorted_distances[:, k - 1].max())
    elif preserve_ordering:
        neighbor_indices = []
        half_window = k // 2
        for i in range(n_samples):
            start = max(0, i - half_window)
            stop = min(n_samples, i + half_window + 1)
            neighbor_indices.append(np.arange(start, stop))
        max_scale = float(data.max())
    else:
        diffs = data[:, None, :] - data[None, :, :]
        distances = np.linalg.norm(diffs, axis=2)
        sorted_indices = np.argsort(distances, axis=1)
        neighbor_indices = [row[:k] for row in sorted_indices]
        gathered = np.take_along_axis(distances, sorted_indices[:, k - 1 : k], axis=1)
        max_scale = float(gathered.max())

    return neighbor_indices, max_scale


def _build_diagram(
    subset: np.ndarray, *, distance_matrix: bool, max_dimension: int, max_scale: float
) -> np.ndarray:
    thresh = max(max_scale, 1e-12)
    dgms = ripser(
        subset,
        maxdim=max_dimension,
        thresh=thresh,
        distance_matrix=distance_matrix,
    )["dgms"]

    rows: List[List[float]] = []
    for dim, diagram in enumerate(dgms):
        if diagram.size == 0:
            continue
        births_deaths = np.asarray(diagram)
        deaths = births_deaths[:, 1]
        deaths = np.where(np.isinf(deaths), max_scale, deaths)
        for birth, death in zip(births_deaths[:, 0], deaths):
            rows.append([float(dim), float(birth), float(death)])
    if not rows:
        return np.zeros((0, 3), dtype=float)
    return np.vstack(rows)


def _k_medoids(distance: np.ndarray, n_clusters: int, random_state: Optional[int]) -> tuple[np.ndarray, np.ndarray, int]:

    rng = np.random.default_rng(random_state)
    n_samples = distance.shape[0]
    medoids = rng.choice(n_samples, size=n_clusters, replace=False)
    labels = np.argmin(distance[:, medoids], axis=1)

    for iteration in range(1, 301):
        new_medoids = medoids.copy()
        for cluster_id in range(n_clusters):
            cluster_points = np.where(labels == cluster_id)[0]
            if cluster_points.size == 0:
                continue
            intra = distance[np.ix_(cluster_points, cluster_points)]
            costs = intra.sum(axis=1)
            best_idx = cluster_points[np.argmin(costs)]
            new_medoids[cluster_id] = best_idx

        labels = np.argmin(distance[:, new_medoids], axis=1)
        if np.array_equal(new_medoids, medoids):
            return labels, medoids, iteration
        medoids = new_medoids

    return labels, medoids, 300


def topo_kmeans(
    data: np.ndarray,
    n_knn: int,
    n_clust: int = 2,
    power: int = 5,
    sigma: float = 0.05,
    dist_matrix: bool = False,
    preserve_ordering: bool = False,
    null_dim: bool = False,
    first_dim: bool = False,
    random_state: Optional[int] = None,
) -> TopoKMeansResult:

    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError("data must be a 2D array or distance matrix.")

    n_samples = data.shape[0]
    if n_knn < 1:
        raise ValueError("n_knn must be at least 1.")
    if n_clust < 1:
        raise ValueError("n_clust must be at least 1.")
    if dist_matrix and data.shape[0] != data.shape[1]:
        raise ValueError("When dist_matrix=True, data must be square.")

    neighbor_indices, max_scale = _compute_neighbors(
        data, n_knn, dist_matrix=dist_matrix, preserve_ordering=preserve_ordering
    )

    diagrams: List[np.ndarray] = []
    dim0_counts: List[int] = []
    dim1_counts: List[int] = []
    dim_total: List[int] = []

    max_dimension = 0 if (dist_matrix or preserve_ordering) else 1

    for i in range(n_samples):
        indices = neighbor_indices[i]
        if dist_matrix:
            neighborhood = data[np.ix_(indices, indices)]
            diag = _build_diagram(
                neighborhood,
                distance_matrix=True,
                max_dimension=max_dimension,
                max_scale=max_scale,
            )
        else:
            neighborhood = data[indices]
            diag = _build_diagram(
                neighborhood,
                distance_matrix=False,
                max_dimension=max_dimension,
                max_scale=max_scale,
            )

        diagrams.append(diag)
        dim_total.append(len(diag))
        dim0_counts.append(int((diag[:, 0] == 0).sum()) if diag.size else 0)
        dim1_counts.append(int((diag[:, 0] == 1).sum()) if diag.size else 0)

    max_dim0 = max(dim0_counts) if dim0_counts else 0
    max_dim1 = max(dim1_counts) if dim1_counts else 0
    max_dim_total = max(dim_total) if dim_total else 0

    perst_val_0 = np.zeros((max_dim0, n_samples))
    perst_val_1 = np.zeros((max_dim1, n_samples))
    perst_val = np.zeros((max_dim_total, n_samples))

    for idx, diag in enumerate(diagrams):
        if diag.size == 0:
            continue
        births = diag[:, 1]
        deaths = diag[:, 2]
        persistence_all = deaths - births
        perst_val[: persistence_all.size, idx] = persistence_all

        mask0 = diag[:, 0] == 0
        if mask0.any():
            pers0 = persistence_all[mask0]
            perst_val_0[: pers0.size, idx] = pers0

        mask1 = diag[:, 0] == 1
        if mask1.any():
            pers1 = persistence_all[mask1]
            perst_val_1[: pers1.size, idx] = pers1

    if null_dim:
        persistence = perst_val_0
    elif first_dim:
        persistence = perst_val_1
    else:
        persistence = perst_val

    if persistence.size == 0:
        distance = np.zeros((n_samples, n_samples))
    else:
        n_features = persistence.shape[0]
        k_sums = np.zeros(n_samples)
        for i in range(n_samples):
            vec_i = persistence[:, i]
            k_sums[i] = _rbf_kernel_sum(vec_i, vec_i, sigma=sigma)

        distance = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            vec_i = persistence[:, i]
            for j in range(i, n_samples):
                if i == j:
                    continue
                vec_j = persistence[:, j]
                cross = _rbf_kernel_sum(vec_i, vec_j, sigma=sigma)
                d = (k_sums[i] + k_sums[j] - 2 * cross) ** power
                distance[i, j] = distance[j, i] = d

    labels, medoids, iterations = _k_medoids(
        distance, n_clusters=n_clust, random_state=random_state
    )

    return TopoKMeansResult(
        labels=labels,
        persistence=persistence,
        distance_matrix=distance,
        medoids=medoids,
        iterations=iterations,
    )

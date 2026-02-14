import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import radius_neighbors_graph, NearestNeighbors
from scipy.sparse.csgraph import connected_components

# dbscan clustering, epsilon is the neighborhood radius
def cluster_dbscan(gdf, eps=None, min_samples=1):
    coords = np.column_stack((gdf.geometry.x, gdf.geometry.y))

    if eps is None:
        eps, _ = _estimate_clustering_distance(gdf, min_samples=min_samples)

    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(coords)

    return labels

# graph based clustering, radius is connection distance for nodes
def cluster_graph_radius(gdf, radius=None, min_samples=1):
    coords = np.column_stack((gdf.geometry.x, gdf.geometry.y))

    if radius is None:
        _, radius = _estimate_clustering_distance(gdf, min_samples=min_samples)

    A = radius_neighbors_graph(coords, radius=radius, mode='connectivity', include_self=False)
    n_components, labels = connected_components(A)

    return labels

def _estimate_clustering_distance(gdf, min_samples=3, eps_multiplier=2.5):
    """
    Internal function to estimate spatial scale for clustering.
    Returns eps and radius.
    """

    coords = np.column_stack((gdf.geometry.x, gdf.geometry.y))

    nbrs = NearestNeighbors(n_neighbors=max(min_samples, 2))
    nbrs.fit(coords)
    distances, _ = nbrs.kneighbors(coords)

    # 1st nearest neighbor (excluding self)
    nn_distances = distances[:, 1]
    median_nn = np.median(nn_distances)

    # k-distance for knee detection
    k_distances = np.sort(distances[:, -1])

    x = np.arange(len(k_distances))
    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (k_distances - k_distances.min()) / (k_distances.max() - k_distances.min())

    line = np.column_stack((x_norm, x_norm))
    curve = np.column_stack((x_norm, y_norm))

    distances_to_line = np.linalg.norm(curve - line, axis=1)
    knee_index = np.argmax(distances_to_line)
    knee_eps = k_distances[knee_index]

    eps_statistical = eps_multiplier * median_nn

    eps = np.mean([eps_statistical, knee_eps])
    radius = eps_statistical  # graph radius slightly more conservative

    return eps, radius
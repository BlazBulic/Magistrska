import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import radius_neighbors_graph, NearestNeighbors
from scipy.sparse.csgraph import connected_components
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from sklearn.decomposition import PCA

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

# heleper class for quick merging clusters
class _UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}
        self.rank = {x: 0 for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1

# function for merging clusters that should realy be one
def merge_clusters_by_contact(labels, xy, h, merge_pair_xy=6.0, merge_pair_h=np.inf):
    """
    Merge clusters if ANY inter-cluster point pair is close in XY (<= merge_pair_xy)
    and height-compatible (<= merge_pair_h). Works well for line fragments.
    """
    labels = labels.copy()
    valid = np.unique(labels[labels != -1])
    if len(valid) <= 1:
        return labels

    uf = _UnionFind(valid.tolist())

    # neighbor search on all points (fast)
    nn = NearestNeighbors(radius=merge_pair_xy).fit(xy)
    neigh = nn.radius_neighbors(xy, return_distance=False)

    for i, js in enumerate(neigh):
        li = labels[i]
        if li == -1:
            continue
        hi = h[i]
        for j in js:
            if i == j:
                continue
            lj = labels[j]
            if lj == -1 or lj == li:
                continue

            if abs(hi - h[j]) <= merge_pair_h:
                uf.union(li, lj)

    # apply merges
    root_map = {}
    new_labels = labels.copy()
    for cid in valid:
        root = uf.find(cid)
        root_map[cid] = root
    for i in range(len(new_labels)):
        if new_labels[i] != -1:
            new_labels[i] = root_map[new_labels[i]]

    return new_labels

# main cluster function
def multi_step_clustering_plot(
    gdf,
    height_col,
    #  spatial 
    xy_multiplier_coarse=3.5,          # bigger = fewer coarse patches
    xy_multiplier_split=3,           # bigger = fewer splits inside patch
    #  height split 
    height_mode="neighbor_percentile", # "neighbor_percentile" or "std"
    height_percentile=80,              # higher = looser height constraint
    height_multiplier=2.0,             # used if height_mode="std"
    #  line detection 
    detect_lines=True,
    line_ratio_thresh=0.9,
    line_xy_boost=2,            # multiplies radius_xy_split for line patches
    ignore_height_for_lines=True, # if True, don't use height constraint in line patches,
    #  merging 
    merge_pair_xy=12,   # try 8–15
    merge_pair_h=np.inf,  # or e.g. 10.0 if you want height consistency
    #  plotting 
    plot=True,
    label_clusters=True,
    save=False,
    save_path=r"C:\Users\blazb\Desktop\Magistrska\Figures\Working_process\figure_multi_step.png",
    dpi=300
):
    """
    Multi-step hierarchical clustering with height-aware splitting + post-merge.

    Returns
    -------
    final_labels : np.ndarray of int
    """

    if height_col not in gdf.columns:
        raise KeyError(f"height_col='{height_col}' not in columns: {list(gdf.columns)}")

    xy = np.column_stack((gdf.geometry.x.to_numpy(), gdf.geometry.y.to_numpy()))
    h = gdf[height_col].to_numpy()

    # Rough clusters, only looking at 2D distance, using graphs
    nn2 = NearestNeighbors(n_neighbors=2).fit(xy)
    nn_dist = nn2.kneighbors(xy)[0][:, 1]
    radius_xy_coarse = xy_multiplier_coarse * float(np.median(nn_dist))

    nbrs_r = NearestNeighbors(radius=radius_xy_coarse).fit(xy)
    A_global = nbrs_r.radius_neighbors_graph(xy, mode="connectivity")
    spatial_labels = connected_components(A_global)[1]

    #  within-patch height-aware split (dual threshold graph)
    final_labels = np.full(len(gdf), -1, dtype=int)
    global_id = 0

    for sp in np.unique(spatial_labels):
        idx = np.where(spatial_labels == sp)[0]

        if len(idx) == 1:
            final_labels[idx] = global_id
            global_id += 1
            continue

        xy_sub = xy[idx]
        h_sub = h[idx]

        # patch-specific XY radius
        nn_patch = NearestNeighbors(n_neighbors=2).fit(xy_sub).kneighbors(xy_sub)[0][:, 1]
        radius_xy_split = xy_multiplier_split * float(np.median(nn_patch))

        # detect if the whole patch is line-like 
        is_line_patch = False
        if len(xy_sub) >= 6:  # need enough points for stable PCA
            pca_patch = PCA(n_components=2).fit(xy_sub)
            ratio_patch = float(pca_patch.explained_variance_ratio_[0])
            is_line_patch = ratio_patch >= line_ratio_thresh

        # If line patch: enlarge XY radius to bridge gaps
        if is_line_patch:
            radius_xy_split *= line_xy_boost

        #  patch-specific height threshold 
        if is_line_patch and ignore_height_for_lines:
            radius_h = np.inf  # effectively disables height constraint for line patches
        else:
            if height_mode == "std":
                radius_h = height_multiplier * float(np.std(h_sub))
            elif height_mode == "neighbor_percentile":
                nnr = NearestNeighbors(radius=radius_xy_split).fit(xy_sub)
                neigh = nnr.radius_neighbors(xy_sub, return_distance=False)

                diffs = []
                for i, js in enumerate(neigh):
                    for j in js:
                        if i != j:
                            diffs.append(abs(h_sub[i] - h_sub[j]))

                radius_h = float(np.percentile(diffs, height_percentile)) if diffs else float(np.std(h_sub)) * 2.0
            else:
                raise ValueError("height_mode must be 'neighbor_percentile' or 'std'")

        # build adjacency with dual threshold
        n = len(idx)
        A = lil_matrix((n, n), dtype=np.int8)
        nnr = NearestNeighbors(radius=radius_xy_split).fit(xy_sub)
        neigh = nnr.radius_neighbors(xy_sub, return_distance=False)

        for i, js in enumerate(neigh):
            hi = h_sub[i]
            for j in js:
                if i == j:
                    continue
                if abs(hi - h_sub[j]) <= radius_h:
                    A[i, j] = 1
                    A[j, i] = 1

        sub_labels = connected_components(A.tocsr())[1]

        # assign global ids, optional line detection
        for sub in np.unique(sub_labels):
            local = np.where(sub_labels == sub)[0]
            sub_idx = idx[local]

            if detect_lines and len(sub_idx) >= 4:
                pts = xy[sub_idx]
                ratio = float(PCA(n_components=2).fit(pts).explained_variance_ratio_[0])
                # note: this does NOT split; it just allows you to tag later if desired
                _ = ratio  # kept if you later want to store a "is_line" flag

            final_labels[sub_idx] = global_id
            global_id += 1


    labels = final_labels.copy()
    uniq, counts = np.unique(labels, return_counts=True)

    # Merge pass: merge clusters whose centroids are close in XY and height
    labels = merge_clusters_by_contact(
    labels,
    xy=xy,
    h=h,
    merge_pair_xy=merge_pair_xy,
    merge_pair_h=merge_pair_h
    )

    # relabel to 0..K-1 (keep -1 as -1)
    relabeled = labels.copy()
    valid = sorted([cid for cid in np.unique(relabeled) if cid != -1])
    mapping = {cid: i for i, cid in enumerate(valid)}
    for cid, new in mapping.items():
        relabeled[relabeled == cid] = new

    # Plot
    if plot:
        uniq = np.unique(relabeled)
        label_to_color_idx = {lbl: i for i, lbl in enumerate(uniq)}
        color_idx = np.array([label_to_color_idx[l] for l in relabeled], dtype=int)

        cmap = plt.get_cmap("tab20", len(uniq))
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(xy[:, 0], xy[:, 1], c=color_idx, cmap=cmap, s=15)

        if label_clusters:
            for lbl in uniq:
                if lbl == -1:
                    continue
                pts = xy[relabeled == lbl]
                cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
                ax.text(cx, cy, str(lbl), fontsize=8, ha="center", va="center", weight="bold", color="black")

        ax.set_title(
            f"Multi-step clustering | coarse={xy_multiplier_coarse:.2f}, split={xy_multiplier_split:.2f}, "
            f"h_mode={height_mode}, p={height_percentile}"
        )
        ax.set_axis_off()
        plt.tight_layout()

        if save:
            plt.savefig(save_path, dpi=dpi)
        else:
            plt.show()

    return relabeled
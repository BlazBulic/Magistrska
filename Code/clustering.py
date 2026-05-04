import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import radius_neighbors_graph, NearestNeighbors
from scipy.sparse.csgraph import connected_components
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from shapely.geometry import MultiPoint
from shapely import concave_hull as shapely_concave_hull
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score, completeness_score, v_measure_score

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

####
# This part is the main clustering pipeline
####

# function for merging clusters that should realy be one
def merge_clusters_by_contact(labels, xy, h, cluster_is_line, line_xy_factor = 1, merge_pair_xy=6.0, merge_pair_h=np.inf):
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
    nn = NearestNeighbors(radius=merge_pair_xy * line_xy_factor).fit(xy)
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

            pair_xy = merge_pair_xy
            if cluster_is_line.get(li, False) and cluster_is_line.get(lj, False):
                pair_xy = merge_pair_xy * line_xy_factor

            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dxy = (dx * dx + dy * dy) ** 0.5

            if dxy <= pair_xy and abs(hi - h[j]) <= merge_pair_h:
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

    # 1. stage : Rough clusters, only looking at 2D distance, using graphs
    nn2 = NearestNeighbors(n_neighbors=2).fit(xy)
    nn_dist = nn2.kneighbors(xy)[0][:, 1]
    radius_xy_coarse = xy_multiplier_coarse * float(np.median(nn_dist))

    nbrs_r = NearestNeighbors(radius=radius_xy_coarse).fit(xy)
    A_global = nbrs_r.radius_neighbors_graph(xy, mode="connectivity")
    spatial_labels = connected_components(A_global)[1]

    # 2. stage: Within-patch height-aware split (dual threshold graph)
    final_labels = np.full(len(gdf), -1, dtype=int)
    global_id = 0
    cluster_is_line = {}

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
            cluster_is_line[global_id] = is_line_patch
            global_id += 1


    labels = final_labels.copy()
    uniq, counts = np.unique(labels, return_counts=True)

    # Merge pass: merge clusters whose centroids are close in XY and height
    labels = merge_clusters_by_contact(
    labels,
    xy=xy,
    h=h,
    cluster_is_line=cluster_is_line,
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

####
# This part is for additional sensible splitting of clusters
####

def _try_split_cluster_dbscan_xy(xy_sub: np.ndarray, min_child_size: int = 15, min_child_fraction: float = 0.08, eps_factor: float = 0.75, min_samples: int = 4, max_meaningful_clusters: int = 3, debug: bool = False) -> tuple[np.ndarray | None, dict]:
    """
    Try to split one existing cluster using local DBSCAN in XY only.

    Returns
    -------
    split_labels : np.ndarray or None
        Child labels 0..k-1 if accepted, else None
    info : dict
    """
    n = len(xy_sub)
    info = {"n_points": int(n)}

    if n < max(2 * min_child_size, 8):
        info["reason"] = "cluster_too_small"
        if debug:
            print("[DBSCAN DEBUG]", info)
        return None, info

    nn = NearestNeighbors(n_neighbors=2).fit(xy_sub)
    nn_dist = nn.kneighbors(xy_sub)[0][:, 1]
    median_nn = float(np.median(nn_dist))

    info["median_nn"] = median_nn
    info["nn_min"] = float(np.min(nn_dist))
    info["nn_max"] = float(np.max(nn_dist))
    info["nn_mean"] = float(np.mean(nn_dist))

    if not np.isfinite(median_nn) or median_nn <= 0:
        info["reason"] = "invalid_median_nn"
        if debug:
            print("[DBSCAN DEBUG]", info)
        return None, info

    eps = eps_factor * median_nn
    info["eps_factor"] = eps_factor
    info["eps"] = eps
    info["min_samples"] = min_samples

    raw_labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(xy_sub)

    raw_unique, raw_counts_arr = np.unique(raw_labels, return_counts=True)
    raw_counts = {int(cid): int(cnt) for cid, cnt in zip(raw_unique, raw_counts_arr)}
    info["raw_counts"] = raw_counts
    info["noise_count"] = int(raw_counts.get(-1, 0))
    info["noise_fraction"] = float(raw_counts.get(-1, 0) / n)

    # identify meaningful DBSCAN clusters
    raw_ids = [cid for cid in raw_unique if cid != -1]
    if len(raw_ids) == 0:
        info["reason"] = "no_clusters_found"
        if debug:
            print("[DBSCAN DEBUG]", info)
        return None, info

    meaningful_ids = [int(cid) for cid in raw_ids if raw_counts[int(cid)] >= min_child_size and raw_counts[int(cid)] / n >= min_child_fraction]
    info["meaningful_ids"] = meaningful_ids
    info["meaningful_counts"] = {cid: raw_counts[cid] for cid in meaningful_ids}
    info["min_child_size"] = min_child_size
    info["min_child_fraction"] = min_child_fraction

    if len(meaningful_ids) < 2:
        info["reason"] = "fewer_than_2_meaningful_clusters"
        if debug:
            print("[DBSCAN DEBUG]", info)
        return None, info

    if len(meaningful_ids) > max_meaningful_clusters:
        meaningful_ids = sorted(meaningful_ids, key=lambda cid: raw_counts[cid], reverse=True)[:max_meaningful_clusters]
        info["trimmed_meaningful_ids"] = meaningful_ids

    # remap meaningful clusters to 0..k-1
    split_labels = np.full(n, -1, dtype=int)
    mapping = {cid: new_id for new_id, cid in enumerate(meaningful_ids)}
    info["mapping"] = mapping

    for cid, new_id in mapping.items():
        split_labels[raw_labels == cid] = new_id

    # assign all leftover points (noise + tiny DBSCAN clusters) to nearest meaningful centroid
    kept_labels = np.unique(split_labels[split_labels != -1])
    centroids = np.array([xy_sub[split_labels == lab].mean(axis=0) for lab in kept_labels])

    leftover_idx = np.where(split_labels == -1)[0]
    info["leftover_count_before_attach"] = int(len(leftover_idx))

    for i in leftover_idx:
        d = np.linalg.norm(centroids - xy_sub[i], axis=1)
        split_labels[i] = int(np.argmin(d))

    final_unique, final_counts_arr = np.unique(split_labels, return_counts=True)
    final_counts = {int(cid): int(cnt) for cid, cnt in zip(final_unique, final_counts_arr)}
    info["final_counts"] = final_counts
    info["n_meaningful_clusters"] = int(len(final_unique))
    info["reason"] = "accepted"

    if debug:
        print("[DBSCAN DEBUG]", info)

    return split_labels, info

def split_end_clusters_dbscan(gdf: gpd.GeoDataFrame, labels: np.ndarray | list, min_cluster_size: int = 40, min_child_size: int = 15, min_child_fraction: float = 0.08, eps_factor: float = 0.75, min_samples: int = 4, max_meaningful_clusters: int = 3, max_iter: int = 1, plot: bool = False, label_clusters: bool = True, save: bool = False, save_path: str = r"C:\Users\blazb\Desktop\Magistrska\Figures\Working_process\figure_dbscan_split.png", dpi: int = 300, verbose: bool = False, debug: bool = False, debug_cluster_id: int | None = None) -> tuple[np.ndarray, list[dict]]:
    """
    Post-process final clusters and split some of them using local DBSCAN in XY.
    """
    labels = np.asarray(labels).copy()
    if len(labels) != len(gdf):
        raise ValueError("labels length must match gdf length")

    xy = np.column_stack((gdf.geometry.x.to_numpy(), gdf.geometry.y.to_numpy()))
    new_labels = labels.copy()
    split_log = []

    for _ in range(max_iter):
        changed = False
        valid = [cid for cid in np.unique(new_labels) if cid != -1]
        next_label = max(valid) + 1 if len(valid) > 0 else 0

        for cid in valid:
            idx = np.where(new_labels == cid)[0]
            n = len(idx)

            if n < min_cluster_size:
                continue

            xy_sub = xy[idx]
            do_debug = debug and (debug_cluster_id is None or int(cid) == int(debug_cluster_id))

            if do_debug:
                print(f"\n[DBSCAN DEBUG] inspecting parent cluster {cid}, size={n}")

            split_labels, info = _try_split_cluster_dbscan_xy(
                xy_sub=xy_sub,
                min_child_size=min_child_size,
                min_child_fraction=min_child_fraction,
                eps_factor=eps_factor,
                min_samples=min_samples,
                max_meaningful_clusters=max_meaningful_clusters,
                debug=do_debug
            )

            if split_labels is None:
                if do_debug:
                    print(f"[DBSCAN DEBUG] cluster {cid} NOT split, reason={info.get('reason')}")
                continue

            child_ids = np.unique(split_labels)
            child_sizes = {lab: int(np.sum(split_labels == lab)) for lab in child_ids}
            ordered = sorted(child_ids, key=lambda lab: child_sizes[lab], reverse=True)

            keep_lab = ordered[0]
            keep_idx = idx[split_labels == keep_lab]
            new_labels[keep_idx] = cid

            assigned_children = []
            for lab in ordered[1:]:
                child_idx = idx[split_labels == lab]
                new_labels[child_idx] = next_label
                assigned_children.append((int(lab), int(next_label), int(len(child_idx))))
                next_label += 1

            log_entry = {"parent_cluster": int(cid), "parent_size": int(n), "new_children": assigned_children, **info}
            split_log.append(log_entry)

            if verbose or do_debug:
                print("[DBSCAN DEBUG] accepted split:", log_entry)

            changed = True

        if not changed:
            break

    relabeled = new_labels.copy()
    valid = sorted([cid for cid in np.unique(relabeled) if cid != -1])
    mapping = {cid: i for i, cid in enumerate(valid)}
    for cid, new in mapping.items():
        relabeled[relabeled == cid] = new

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

        ax.set_title(f"DBSCAN refinement | eps_factor={eps_factor:.2f}, min_samples={min_samples}, min_child_size={min_child_size}")
        ax.set_axis_off()
        plt.tight_layout()

        if save:
            plt.savefig(save_path, dpi=dpi)
        else:
            plt.show()

    return relabeled, split_log

####
# This part is for evaluating the clustering
####


def _build_cluster_polygons(trees_gdf, cluster_col="cluster_id", geom_method="buffer_union", point_buffer=0.5, concavity=0.3):
    """
    Build one polygon geometry per cluster from tree points.

    Parameters
    ----------
    trees_gdf : GeoDataFrame
    cluster_col : str
    geom_method : str
        "convex_hull", "concave_hull", or "buffer_union"
    point_buffer : float
        Used only if geom_method == "buffer_union"
    concavity : float
        Used only if geom_method == "concave_hull". Lower values usually hug the
        points more tightly, while higher values approach a convex hull.

    Returns
    -------
    GeoDataFrame
    """
    rows = []

    for cid, sub in trees_gdf.groupby(cluster_col):
        pts = list(sub.geometry)

        if len(pts) == 1:
            geom = pts[0].buffer(point_buffer)
        else:
            if geom_method == "convex_hull":
                geom = MultiPoint(pts).convex_hull
            elif geom_method == "concave_hull":
                if len(pts) < 4:
                    geom = MultiPoint(pts).convex_hull
                else:
                    try:
                        geom = shapely_concave_hull(MultiPoint(pts), ratio=concavity)
                        if geom is None or geom.is_empty:
                            geom = MultiPoint(pts).convex_hull
                    except Exception:
                        geom = MultiPoint(pts).convex_hull
            elif geom_method == "buffer_union":
                geom = sub.geometry.buffer(point_buffer).union_all()
            else:
                raise ValueError("geom_method must be 'convex_hull', 'concave_hull', or 'buffer_union'")

        rows.append({cluster_col: cid, "geometry": geom, "n_points": len(sub)})

    return gpd.GeoDataFrame(rows, geometry="geometry", crs=trees_gdf.crs)

def evaluate_clustering_against_polygons(trees_gdf, cluster_labels, polygons_gdf, polygon_id_col=None, cluster_geom_method="convex_hull", point_buffer=0.5, concavity=0.3, join_predicate="within"):
    """
    Evaluate clustering against ground-truth vegetation polygons.

    Parameters
    ----------
    trees_gdf : GeoDataFrame
        Tree points.
    cluster_labels : array-like
        Predicted cluster label for each tree (same order as trees_gdf).
    polygons_gdf : GeoDataFrame
        Ground-truth vegetation polygons.
    polygon_id_col : str or None
        Column identifying ground-truth polygons. If None, uses polygon index.
    cluster_geom_method : str
        "convex_hull", "concave_hull", or "buffer_union"
    point_buffer : float
        Buffer radius for singleton clusters or buffer-based cluster geometry.
    concavity : float
        Concave-hull ratio used when cluster_geom_method == "concave_hull".
    join_predicate : str
        Usually "within" or "intersects".

    Returns
    -------
    results : dict
    """
    if len(trees_gdf) != len(cluster_labels):
        raise ValueError("cluster_labels must have the same length as trees_gdf")

    trees = trees_gdf.copy()
    trees["cluster_id"] = np.asarray(cluster_labels)

    polys = polygons_gdf.copy()
    if polygon_id_col is None:
        polys["gt_polygon_id"] = np.arange(len(polys))
        polygon_id_col = "gt_polygon_id"
    else:
        if polygon_id_col not in polys.columns:
            raise KeyError(f"polygon_id_col='{"OPIS"}' not found in polygons_gdf")

    # -----------------------------
    # 1) Assign each tree to a GT polygon
    # -----------------------------
    joined = gpd.sjoin(trees, polys[[polygon_id_col, "geometry"]], how="left", predicate=join_predicate)

    # keep only trees that got matched to a polygon
    matched = joined.dropna(subset=[polygon_id_col]).copy()
    matched[polygon_id_col] = matched[polygon_id_col].astype(str)
    matched["cluster_id"] = matched["cluster_id"].astype(str)

    if len(matched) == 0:
        raise ValueError("No tree points matched the ground-truth polygons")

    y_true = matched[polygon_id_col].to_numpy()
    y_pred = matched["cluster_id"].to_numpy()

    # -----------------------------
    # 2) Point-level clustering metrics
    # -----------------------------
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    homo = homogeneity_score(y_true, y_pred)
    comp = completeness_score(y_true, y_pred)
    v_measure = v_measure_score(y_true, y_pred)

    contingency = pd.crosstab(matched["cluster_id"], matched[polygon_id_col])

    # purity: for each predicted cluster, dominant GT polygon fraction
    purity = contingency.max(axis=1).sum() / contingency.to_numpy().sum()

    # inverse purity / coverage-like clustering score
    inverse_purity = contingency.max(axis=0).sum() / contingency.to_numpy().sum()

    # -----------------------------
    # 3) Build cluster polygons
    # -----------------------------
    cluster_polys = _build_cluster_polygons(
        trees_gdf=trees,
        cluster_col="cluster_id",
        geom_method=cluster_geom_method,
        point_buffer=point_buffer,
        concavity=concavity
    )

    # -----------------------------
    # 4) Geometry-level comparison
    # For each cluster polygon, find best matching GT polygon by IoU
    # -----------------------------
    geom_rows = []

    for _, crow in cluster_polys.iterrows():
        cgeom = crow.geometry
        cid = crow["cluster_id"]
        c_area = cgeom.area

        best_iou = 0.0
        best_precision = 0.0
        best_recall = 0.0
        best_gt = None

        for _, prow in polys.iterrows():
            pgeom = prow.geometry
            pid = prow[polygon_id_col]

            inter = cgeom.intersection(pgeom).area
            if inter == 0:
                continue

            union = cgeom.union(pgeom).area
            iou = inter / union if union > 0 else 0.0
            precision = inter / c_area if c_area > 0 else 0.0
            recall = inter / pgeom.area if pgeom.area > 0 else 0.0

            if iou > best_iou:
                best_iou = iou
                best_precision = precision
                best_recall = recall
                best_gt = pid

        geom_rows.append({
            "cluster_id": cid,
            "matched_gt_polygon": best_gt,
            "cluster_area": c_area,
            "best_iou": best_iou,
            "best_precision": best_precision,
            "best_recall": best_recall
        })

    geom_df = pd.DataFrame(geom_rows)

    # weighted by number of trees in cluster
    cluster_sizes = trees.groupby("cluster_id").size().rename("cluster_size").reset_index()
    geom_df = geom_df.merge(cluster_sizes, on="cluster_id", how="left")

    weighted_mean_iou = np.average(geom_df["best_iou"], weights=geom_df["cluster_size"])
    weighted_mean_precision = np.average(geom_df["best_precision"], weights=geom_df["cluster_size"])
    weighted_mean_recall = np.average(geom_df["best_recall"], weights=geom_df["cluster_size"])

    # -----------------------------
    # 5) Polygon-level "best captured by a cluster"
    # -----------------------------
    polygon_rows = []

    for _, prow in polys.iterrows():
        pid = prow[polygon_id_col]
        pgeom = prow.geometry
        p_area = pgeom.area

        best_iou = 0.0
        best_precision = 0.0
        best_recall = 0.0
        best_cluster = None

        for _, crow in cluster_polys.iterrows():
            cid = crow["cluster_id"]
            cgeom = crow.geometry
            c_area = cgeom.area

            inter = cgeom.intersection(pgeom).area
            if inter == 0:
                continue

            union = cgeom.union(pgeom).area
            iou = inter / union if union > 0 else 0.0
            precision = inter / c_area if c_area > 0 else 0.0
            recall = inter / p_area if p_area > 0 else 0.0

            if iou > best_iou:
                best_iou = iou
                best_precision = precision
                best_recall = recall
                best_cluster = cid

        polygon_rows.append({
            "gt_polygon_id": pid,
            "matched_cluster": best_cluster,
            "polygon_area": p_area,
            "best_iou": best_iou,
            "best_precision": best_precision,
            "best_recall": best_recall
        })

    polygon_df = pd.DataFrame(polygon_rows)

    mean_polygon_iou = polygon_df["best_iou"].mean()
    mean_polygon_recall = polygon_df["best_recall"].mean()

    return {
        "n_trees_total": len(trees),
        "n_trees_matched_to_polygons": len(matched),
        "n_clusters": trees["cluster_id"].nunique(),
        "n_gt_polygons": len(polys),

        # point-level metrics
        "ARI": ari,
        "NMI": nmi,
        "homogeneity": homo,
        "completeness": comp,
        "v_measure": v_measure,
        "purity": purity,
        "inverse_purity": inverse_purity,

        # geometry-level cluster->polygon
        "weighted_mean_cluster_best_IoU": weighted_mean_iou,
        "weighted_mean_cluster_best_precision": weighted_mean_precision,
        "weighted_mean_cluster_best_recall": weighted_mean_recall,

        # geometry-level polygon->cluster
        "mean_polygon_best_IoU": mean_polygon_iou,
        "mean_polygon_best_recall": mean_polygon_recall,

        # useful detailed outputs
        "tree_assignments": matched,
        "contingency_table": contingency,
        "cluster_polygon_scores": geom_df,
        "polygon_capture_scores": polygon_df,
        "cluster_polygons": cluster_polys
    }

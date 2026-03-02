import subprocess
import laspy
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from shapely.geometry import Point, MultiPoint
from shapely.ops import unary_union

def segment_trees_r(input_laz, output_laz, r_script_path):
    # Convert paths to strings and fix slashes for R
    input_laz = str(input_laz).replace("\\", "/")
    output_laz = str(output_laz).replace("\\", "/")
    r_script_path = str(r_script_path).replace("\\", "/")
    #Rscript_exe = r"C:\Program Files\R\R-4.5.2\Rscript.exe"

    # Run R script
    subprocess.run(
        ["Rscript", r_script_path, input_laz, output_laz],
        check=True
    )

    return Path(output_laz)


def filter_trees_in_polygons(trees_gdf, polygons_gdf):
    """
    Keep only trees (points) that are inside the given polygons.

    Parameters
    ----------
    trees_gdf : GeoDataFrame
        Aggregated trees (1 point per tree), must have geometry and treeID
    polygons_gdf : GeoDataFrame
        Polygons to filter against

    Returns
    -------
    GeoDataFrame
        Subset of trees_gdf where each tree is within at least one polygon
    """

    # Ensure both have the same CRS
    if trees_gdf.crs != polygons_gdf.crs:
        trees_gdf = trees_gdf.to_crs(polygons_gdf.crs)

    # Spatial join: keep only points within polygons
    intersecting_trees = gpd.sjoin(
        trees_gdf,
        polygons_gdf,
        how="inner",
        predicate="intersects"
    )

    # Drop extra columns added by sjoin (like index_right)
    intersecting_trees = intersecting_trees[trees_gdf.columns]
    print(f"Kept {len(intersecting_trees)} trees inside polygons")
    return intersecting_trees


def compute_water_distance_features(forest_shp_path, water_shp_path, target_crs="EPSG:3794", output_path=None, search_buffer=100):
    """
    Fast computation of three water distance features using spatial index:
    1) edge_dist_water
    2) centroid_to_water_poly_dist

    search_buffer: maximum distance (meters) to consider nearby water polygons
    """

    forest = gpd.read_file(forest_shp_path).to_crs(target_crs)
    water = gpd.read_file(water_shp_path).to_crs(target_crs)

    # Build spatial index for water polygons
    water_sindex = water.sindex

    # Prepare output columns
    forest["edge_dist_water"] = float("nan")
    forest["centroid_to_water_poly_dist"] = float("nan")

    # Precompute water centroids
    water_centroids = water.geometry.centroid

    for idx, fpoly in forest.geometry.items():
        if(idx % 5000 == 0): print(idx/len(forest.geometry))
        # Search bounding box: expand polygon by buffer
        bounds = fpoly.bounds
        buffered_box = (
            bounds[0]-search_buffer, bounds[1]-search_buffer,
            bounds[2]+search_buffer, bounds[3]+search_buffer
        )
        candidate_idx = list(water_sindex.intersection(buffered_box))
        if not candidate_idx:
            continue
        candidates = water.iloc[candidate_idx]

        # 1️⃣ edge distance
        forest.at[idx, "edge_dist_water"] = fpoly.distance(candidates.unary_union)

        # 2️⃣ centroid → nearest water polygon
        centroid = fpoly.centroid
        forest.at[idx, "centroid_to_water_poly_dist"] = min(candidates.distance(centroid))

    if output_path:
        forest.to_file(output_path)

    return forest

# computing pca of clusters
def _pca_2d_eigenvalues(xy: np.ndarray):
    if xy is None or len(xy) < 2:
        return np.nan, np.nan
    mu = xy.mean(axis=0, keepdims=True)
    X = xy - mu
    C = (X.T @ X) / max(len(xy) - 1, 1)
    w = np.linalg.eigvalsh(C)  # ascending
    lam2, lam1 = float(w[0]), float(w[1])
    return lam1, lam2

# computing nearest neigbors stats of clusters
def _nearest_neighbor_stats(xy: np.ndarray):
    n = len(xy)
    if n < 2:
        return np.nan, np.nan
    d2 = np.sum((xy[:, None, :] - xy[None, :, :]) ** 2, axis=2)
    np.fill_diagonal(d2, np.inf)
    nn = np.sqrt(np.min(d2, axis=1))
    return float(np.mean(nn)), float(np.std(nn))

# main function for building the dataset
def build_cluster_dataset_from_labels(
    trees_gdf: gpd.GeoDataFrame,
    labels: np.ndarray | list,              # cluster id per tree row (same order as trees_gdf)
    polygons_gdf: gpd.GeoDataFrame,         # annotated polygons with target + dist-to-water
    target_col: str = "class",
    dist_to_water_col: str = "dist_water",
    height_col: str | None = "Z",
    crown_diam_col: str | None = None,
    cluster_col_out: str = "cluster_label",

    # cluster geometry & density
    cluster_geom: str = "hull",             # "hull" or "centroid"
    density_mode: str = "hull_area",        # "hull_area" or "bbox_area"
    min_cluster_size: int = 1,

    # target assignment from polygons
    join_predicate: str = "within",         # for centroid-in-polygon
    fallback_nearest_polygon: bool = True,  # if centroid not inside any polygon
) -> gpd.GeoDataFrame:
    """
    One row per cluster dataset.

    - trees_gdf: segmented trees points (no cluster column required)
    - labels: array-like of cluster ids aligned with trees_gdf rows
    - polygons_gdf: annotated polygons (must contain target_col and dist_to_water_col)

    Output: GeoDataFrame with cluster-level features + target + dist_to_water + geometry.
    """

    if target_col not in polygons_gdf.columns:
        raise ValueError(f"polygons_gdf missing target column '{target_col}'")
    if dist_to_water_col not in polygons_gdf.columns:
        raise ValueError(f"polygons_gdf missing distance-to-water column '{dist_to_water_col}'")

    if trees_gdf.crs is None or polygons_gdf.crs is None:
        raise ValueError("Both trees_gdf and polygons_gdf must have a CRS set.")
    if trees_gdf.crs != polygons_gdf.crs:
        polygons_gdf = polygons_gdf.to_crs(trees_gdf.crs)

    labels = np.asarray(labels)
    if len(labels) != len(trees_gdf):
        raise ValueError(f"labels length ({len(labels)}) must match trees_gdf length ({len(trees_gdf)})")

    # attach cluster labels to trees
    trees = trees_gdf.copy()
    trees[cluster_col_out] = labels

    # XY for speed
    trees["_x"] = trees.geometry.x.to_numpy()
    trees["_y"] = trees.geometry.y.to_numpy()

    #  compute cluster features 
    feats_list = []
    geoms = []

    for cid, grp in trees.groupby(cluster_col_out, sort=False):
        if pd.isna(cid):
            continue
        n = len(grp)
        if n < min_cluster_size:
            continue

        xy = grp[["_x", "_y"]].to_numpy()
        mp = MultiPoint(list(zip(xy[:, 0], xy[:, 1])))

        if cluster_geom == "hull":
            geom = mp.convex_hull
        elif cluster_geom == "centroid":
            geom = mp.centroid
        else:
            raise ValueError("cluster_geom must be 'hull' or 'centroid'")

        minx, miny, maxx, maxy = mp.bounds
        bbox_area = max((maxx - minx) * (maxy - miny), 0.0)
        hull_area = float(geom.area) if geom.geom_type != "Point" else 0.0

        if density_mode == "hull_area":
            denom = hull_area if hull_area > 0 else (bbox_area if bbox_area > 0 else np.nan)
        elif density_mode == "bbox_area":
            denom = bbox_area if bbox_area > 0 else (hull_area if hull_area > 0 else np.nan)
        else:
            raise ValueError("density_mode must be 'hull_area' or 'bbox_area'")

        density = float(n / denom) if denom and not np.isnan(denom) else np.nan

        lam1, lam2 = _pca_2d_eigenvalues(xy)
        pca_ratio = float(lam1 / lam2) if (lam2 and not np.isnan(lam1) and not np.isnan(lam2) and lam2 > 0) else np.nan
        nn_mean, nn_std = _nearest_neighbor_stats(xy)

        feats = {
            "cluster_id": cid,
            "n_trees": int(n),
            "density": density,
            "bbox_area": float(bbox_area),
            "hull_area": float(hull_area),
            "pca_lam1": lam1,
            "pca_lam2": lam2,
            "pca_ratio": pca_ratio,
            "nn_mean": nn_mean,
            "nn_std": nn_std,
        }

        if height_col is not None and height_col in grp.columns:
            h = grp[height_col].to_numpy()
            feats.update({
                "height_mean": float(np.nanmean(h)),
                "height_std": float(np.nanstd(h)),
                "height_min": float(np.nanmin(h)),
                "height_max": float(np.nanmax(h)),
            })

        if crown_diam_col is not None and crown_diam_col in grp.columns:
            cd = grp[crown_diam_col].to_numpy()
            feats.update({
                "crown_diam_mean": float(np.nanmean(cd)),
                "crown_diam_std": float(np.nanstd(cd)),
            })

        feats_list.append(feats)
        geoms.append(geom)

    clusters = gpd.GeoDataFrame(feats_list, geometry=geoms, crs=trees_gdf.crs)
    if clusters.empty:
        return clusters

    #  assign target class from annotated polygons 
    # Use centroid for within test (even if geometry is hull)
    cent = clusters.copy()
    cent["geometry"] = cent.geometry.centroid

    joined = gpd.sjoin(
        cent,
        polygons_gdf[[target_col, dist_to_water_col, "geometry"]],
        how="left",
        predicate=join_predicate,
    ).drop(columns=["index_right"], errors="ignore")

    # If overlapping polygons can match multiple times, keep the first
    # Deduplicate by cluster row index:
    joined = joined[~joined.index.duplicated(keep="first")]

    clusters[target_col] = joined[target_col].to_numpy()
    clusters[dist_to_water_col] = joined[dist_to_water_col].to_numpy()

    if fallback_nearest_polygon:
        miss = clusters[target_col].isna()
        if miss.any():
            nn = gpd.sjoin_nearest(
                cent.loc[miss, ["geometry"]],
                polygons_gdf[[target_col, dist_to_water_col, "geometry"]],
                how="left",
                distance_col="dist_to_poly",
            ).drop(columns=["index_right"], errors="ignore")
            nn = nn[~nn.index.duplicated(keep="first")]

            clusters.loc[miss, target_col] = nn[target_col].to_numpy()
            clusters.loc[miss, dist_to_water_col] = nn[dist_to_water_col].to_numpy()
            clusters.loc[miss, "dist_to_poly"] = nn["dist_to_poly"].to_numpy()

    output_path = r"C:\Users\blazb\Desktop\Magistrska\Data\Working_data\cluster_dataset.gpkg"
    clusters.to_file(output_path, layer="clusters", driver="GPKG")

    return clusters


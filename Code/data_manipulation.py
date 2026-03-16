import subprocess
import laspy
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
from pathlib import Path
from shapely.geometry import Point, MultiPoint
from shapely.ops import unary_union
from scipy.spatial import cKDTree
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterstats import zonal_stats
from scipy.ndimage import distance_transform_edt

def segment_trees_r(input_folder, output_folder, r_script_path):
    # Convert paths to strings and fix slashes for R
    input_input_folderlaz = str(input_folder).replace("\\", "/")
    output_folder = str(output_folder).replace("\\", "/")
    r_script_path = str(r_script_path).replace("\\", "/")
    #Rscript_exe = r"C:\Program Files\R\R-4.5.2\Rscript.exe"

    # Run R script
    subprocess.run(
        ["Rscript", r_script_path, input_folder, output_folder],
        check=True
    )

    return Path(output_folder)

def remove_duplicate_trees(gdf, distance=1.0):
    coords = np.column_stack((gdf.geometry.x, gdf.geometry.y))
    tree = cKDTree(coords)
    pairs = tree.query_pairs(distance)

    drop = set(j for i, j in pairs)
    return gdf.drop(gdf.index[list(drop)])


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

def make_cluster_geometry(grp, xy, method="concave_hull", concavity=0.3, alpha=0.8, buffer_radius=3.0, buffer_radius_col=None, smooth_radius=0.5):
    """
    Build cluster geometry from tree points.
    """
    if len(xy) == 0:
        return None

    mp = MultiPoint(list(zip(xy[:, 0], xy[:, 1])))

    if len(xy) == 1 or method == "centroid":
        return mp.centroid

    if method == "convex_hull":
        return mp.convex_hull

    elif method == "concave_hull":
        if len(xy) < 4:
            return mp.convex_hull
        try:
            geom = mp.concave_hull(ratio=concavity)
            if geom is None or geom.is_empty:
                return mp.convex_hull
            return geom
        except Exception:
            return mp.convex_hull

    elif method == "buffer_union":
        if buffer_radius_col is not None and buffer_radius_col in grp.columns:
            radii = grp[buffer_radius_col].fillna(buffer_radius).to_numpy()
        else:
            radii = np.full(len(grp), buffer_radius, dtype=float)

        discs = [geom.buffer(r) for geom, r in zip(grp.geometry, radii)]
        geom = unary_union(discs)

        if smooth_radius > 0:
            geom = geom.buffer(smooth_radius).buffer(-smooth_radius)

        if geom is None or geom.is_empty:
            return mp.convex_hull
        return geom

    else:
        raise ValueError("cluster_geom must be one of: 'convex_hull', 'concave_hull', 'alpha_shape', 'buffer_union', 'centroid'")


# main function for building the dataset
def build_cluster_dataset_from_labels(trees_gdf: gpd.GeoDataFrame, labels: np.ndarray | list, polygons_gdf: gpd.GeoDataFrame, target_col: str = "class", dist_to_water_col: str = "dist_water", height_col: str | None = "Z", crown_diam_col: str | None = None, cluster_col_out: str = "cluster_label", cluster_geom: str = "concave_hull", density_mode: str = "geom_area", min_cluster_size: int = 1, concavity: float = 0.3, alpha_shape_value: float = 0.8, buffer_radius: float = 2.0, buffer_radius_col: str | None = None, smooth_radius: float = 0.0, join_predicate: str = "within", fallback_nearest_polygon: bool = True, save_path: str | None = r"C:\Users\blazb\Desktop\Magistrska\Data\Working_data\cluster_dataset.gpkg", save_layer: str = "clusters") -> gpd.GeoDataFrame:
    """
    One row per cluster dataset.

    - trees_gdf: segmented tree points
    - labels: cluster id per tree row, aligned with trees_gdf
    - polygons_gdf: annotated polygons with target_col and dist_to_water_col

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

    trees = trees_gdf.copy()
    trees[cluster_col_out] = labels
    trees["_x"] = trees.geometry.x.to_numpy()
    trees["_y"] = trees.geometry.y.to_numpy()

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

        geom = make_cluster_geometry(grp=grp, xy=xy, method=cluster_geom, concavity=concavity, alpha=alpha_shape_value, buffer_radius=buffer_radius, buffer_radius_col=buffer_radius_col, smooth_radius=smooth_radius)
        if geom is None:
            continue

        minx, miny, maxx, maxy = mp.bounds
        bbox_area = max((maxx - minx) * (maxy - miny), 0.0)
        hull_area = float(mp.convex_hull.area) if len(xy) >= 3 else 0.0
        geom_area = float(geom.area) if geom.geom_type != "Point" else 0.0

        if density_mode == "geom_area":
            denom = geom_area if geom_area > 0 else (hull_area if hull_area > 0 else bbox_area if bbox_area > 0 else np.nan)
        elif density_mode == "hull_area":
            denom = hull_area if hull_area > 0 else (geom_area if geom_area > 0 else bbox_area if bbox_area > 0 else np.nan)
        elif density_mode == "bbox_area":
            denom = bbox_area if bbox_area > 0 else (geom_area if geom_area > 0 else hull_area if hull_area > 0 else np.nan)
        else:
            raise ValueError("density_mode must be 'geom_area', 'hull_area', or 'bbox_area'")

        density = float(n / denom) if denom and not np.isnan(denom) else np.nan
        lam1, lam2 = _pca_2d_eigenvalues(xy)
        pca_ratio = float(lam1 / lam2) if (lam2 and not np.isnan(lam1) and not np.isnan(lam2) and lam2 > 0) else np.nan
        nn_mean, nn_std = _nearest_neighbor_stats(xy)

        feats = {"cluster_id": cid, "n_trees": int(n), "density": density, "bbox_area": float(bbox_area), "hull_area": float(hull_area), "geom_area": float(geom_area), "geom_type_used": cluster_geom, "pca_lam1": lam1, "pca_lam2": lam2, "pca_ratio": pca_ratio, "nn_mean": nn_mean, "nn_std": nn_std}

        if height_col is not None and height_col in grp.columns:
            h = grp[height_col].to_numpy()
            feats.update({"height_mean": float(np.nanmean(h)), "height_std": float(np.nanstd(h)), "height_min": float(np.nanmin(h)), "height_max": float(np.nanmax(h))})

        if crown_diam_col is not None and crown_diam_col in grp.columns:
            cd = grp[crown_diam_col].to_numpy()
            feats.update({"crown_diam_mean": float(np.nanmean(cd)), "crown_diam_std": float(np.nanstd(cd))})

        geoms.append(geom)
        feats_list.append(feats)

    clusters = gpd.GeoDataFrame(feats_list, geometry=geoms, crs=trees_gdf.crs)
    if clusters.empty:
        return clusters

    cent = clusters.copy()
    cent["geometry"] = cent.geometry.centroid

    joined = gpd.sjoin(cent, polygons_gdf[[target_col, dist_to_water_col, "geometry"]], how="left", predicate=join_predicate).drop(columns=["index_right"], errors="ignore")
    joined = joined[~joined.index.duplicated(keep="first")]

    clusters[target_col] = joined[target_col].reindex(clusters.index).to_numpy()
    clusters[dist_to_water_col] = joined[dist_to_water_col].reindex(clusters.index).to_numpy()

    if fallback_nearest_polygon:
        miss = clusters[target_col].isna()
        if miss.any():
            nn = gpd.sjoin_nearest(cent.loc[miss, ["geometry"]], polygons_gdf[[target_col, dist_to_water_col, "geometry"]], how="left", distance_col="dist_to_poly").drop(columns=["index_right"], errors="ignore")
            nn = nn[~nn.index.duplicated(keep="first")]
            clusters.loc[miss, target_col] = nn[target_col].reindex(clusters.loc[miss].index).to_numpy()
            clusters.loc[miss, dist_to_water_col] = nn[dist_to_water_col].reindex(clusters.loc[miss].index).to_numpy()
            clusters.loc[miss, "dist_to_poly"] = nn["dist_to_poly"].reindex(clusters.loc[miss].index).to_numpy()

    if save_path is not None:
        clusters.to_file(save_path, layer=save_layer, driver="GPKG")

    return clusters

def compute_min_raster_distance_to_water(vegetation_shp_path, water_shp_path, target_crs="EPSG:3794", resolution=2.0, buffer_distance=200.0, output_path=None, all_touched=True, progress_every=1000):
    """
    Compute raster-based minimum distance from each vegetation polygon to water polygons.

    For each vegetation polygon:
    - create a local raster window around the polygon
    - rasterize the vegetation polygon
    - rasterize nearby water polygons
    - compute distance-to-water raster
    - assign the minimum distance over vegetation cells

    Parameters
    ----------
    vegetation_shp_path : str
        Path to vegetation polygons shapefile.
    water_shp_path : str
        Path to water polygons shapefile.
    target_crs : str
        Projected CRS in meters.
    resolution : float
        Raster cell size in meters.
    buffer_distance : float
        Extra margin around each vegetation polygon for local computation.
    output_path : str | None
        Optional output shapefile / geopackage path.
    all_touched : bool
        Rasterization mode.
    progress_every : int
        Print progress every N polygons.

    Returns
    -------
    vegetation : GeoDataFrame
        Original vegetation polygons with added column:
        - min_dist_water_r
    """

    vegetation = gpd.read_file(vegetation_shp_path)
    water = gpd.read_file(water_shp_path)

    if vegetation.crs is None:
        raise ValueError("Vegetation shapefile has no CRS.")
    if water.crs is None:
        raise ValueError("Water shapefile has no CRS.")

    vegetation = vegetation.to_crs(target_crs)
    water = water.to_crs(target_crs)

    vegetation["min_dist_water_r"] = np.nan

    water_sindex = water.sindex
    n = len(vegetation)

    for i, (idx, veg_geom) in enumerate(vegetation.geometry.items()):
        if progress_every and i % progress_every == 0:
            print(f"{i}/{n} ({100.0 * i / n:.1f}%)")

        if veg_geom is None or veg_geom.is_empty:
            continue

        minx, miny, maxx, maxy = veg_geom.bounds
        minx -= buffer_distance
        miny -= buffer_distance
        maxx += buffer_distance
        maxy += buffer_distance

        width = int(np.ceil((maxx - minx) / resolution))
        height = int(np.ceil((maxy - miny) / resolution))

        if width <= 0 or height <= 0:
            continue

        transform = from_origin(minx, maxy, resolution, resolution)

        candidate_idx = list(water_sindex.intersection((minx, miny, maxx, maxy)))
        if not candidate_idx:
            continue

        local_water = water.iloc[candidate_idx]
        local_water = local_water[local_water.geometry.intersects(veg_geom.buffer(buffer_distance))]
        if local_water.empty:
            continue

        veg_raster = rasterize(
            [(veg_geom, 1)],
            out_shape=(height, width),
            transform=transform,
            fill=0,
            all_touched=all_touched,
            dtype="uint8"
        )

        water_raster = rasterize(
            [(geom, 1) for geom in local_water.geometry if geom is not None and not geom.is_empty],
            out_shape=(height, width),
            transform=transform,
            fill=0,
            all_touched=all_touched,
            dtype="uint8"
        )

        if veg_raster.max() == 0:
            continue

        if water_raster.max() == 0:
            continue

        dist_raster = distance_transform_edt(water_raster == 0) * resolution
        veg_distances = dist_raster[veg_raster == 1]

        if veg_distances.size == 0:
            continue

        vegetation.at[idx, "min_dist_water_r"] = float(np.min(veg_distances))

    if output_path is not None:
        vegetation.to_file(output_path)

    return vegetation
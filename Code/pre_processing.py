import geopandas as gpd
import pandas as pd
import laspy
import numpy as np
import laspy
from shapely.geometry import box
from pathlib import Path


# returns a geopandas object of the ortophoto polygons
def read_ortophoto(path):
    polygons = gpd.read_file(path)
    #print(f"Loaded {len(polygons)} polygons, CRS: {polygons.crs}")

    return polygons

# returns a laspy object of LIDAR points and a list of those points
def read_lidar_data(path):
    las = laspy.read(path)
    #print(f"Point cloud contains {len(las.x)} points")
    #print(f"Point format: {las.header.point_format}")
    dims = list(las.point_format.dimension_names)

    return las, dims

# takes laspy object and returns a GeoDataFrame of LIDAR points
def las_to_gdf(las, crs="EPSG:3794"):
    data = {
        dim: las[dim]
        for dim in las.point_format.dimension_names
    }
    df = pd.DataFrame(data)
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(las.x, las.y),
        crs=crs
    )

    return gdf

# takes the laspy object of points and the ortophoto polygons and returns those that are in intersection with eachother
def get_intersecting_structures(las, polygons):
    # compute the intersecting polygons
    pc_bbox = box(las.x.min(), las.y.min(), las.x.max(), las.y.max())
    intersecting_polygons = polygons[polygons.intersects(pc_bbox)]
    #print(f"{len(intersecting_polygons)} polygons intersect the point cloud tile")

    # compute intersecting points and keep only the relevant vegetation points
    points_gdf = las_to_gdf(las, polygons.crs)
    vegetation_gdf = points_gdf[points_gdf["classification"].isin([3,4,5])]
    # intersection not needed yet
    intersecting_points = gpd.sjoin(
        vegetation_gdf,
        intersecting_polygons,
        predicate="within",
        how="inner"
    )

    return vegetation_gdf, intersecting_polygons, intersecting_points

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


def get_intersecting_structures_from_trees(trees_gdf, polygons):
    pc_bbox = box(trees_gdf.geometry.x.min(), trees_gdf.geometry.y.min(), trees_gdf.geometry.x.max(), trees_gdf.geometry.y.max())
    intersecting_polygons = polygons[polygons.intersects(pc_bbox)].copy()

    intersecting_points = gpd.sjoin(trees_gdf, intersecting_polygons, predicate="within", how="inner")
    relevant_trees_gdf = trees_gdf.loc[intersecting_points.index.unique()].copy()

    return relevant_trees_gdf, intersecting_polygons, intersecting_points


def read_lidar_folder(folder_path):
    folder = Path(folder_path)
    lidar_files = sorted(list(folder.glob("*.laz")) + list(folder.glob("*.las")))

    if len(lidar_files) == 0:
        raise FileNotFoundError(f"No .laz or .las files found in: {folder_path}")

    las_list = []
    for file_path in lidar_files:
        las = laspy.read(file_path)
        las_list.append(las)

    merged_las = merge_las_files(las_list)
    dims = list(merged_las.point_format.dimension_names)

    return merged_las, dims

def merge_las_files(las_list):
    if len(las_list) == 0:
        raise ValueError("las_list is empty")

    base = las_list[0]
    header = base.header.copy()

    all_x = np.concatenate([las.x for las in las_list])
    all_y = np.concatenate([las.y for las in las_list])
    all_z = np.concatenate([las.z for las in las_list])

    merged = laspy.create(point_format=base.header.point_format, file_version=base.header.version)
    merged.header.scales = base.header.scales
    merged.header.offsets = base.header.offsets

    merged.x = all_x
    merged.y = all_y
    merged.z = all_z

    common_dims = set(base.point_format.dimension_names)

    for dim in common_dims:
        if dim in {"X", "Y", "Z"}:
            continue
        try:
            merged[dim] = np.concatenate([las[dim] for las in las_list])
        except Exception:
            pass

    return merged

def save_las(las, output_path):
    las.write(output_path)
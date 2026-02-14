import geopandas as gpd
import pandas as pd
import laspy
from shapely.geometry import box


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

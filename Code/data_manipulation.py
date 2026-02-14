import subprocess
import laspy
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path

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


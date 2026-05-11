import geopandas as gpd
import pandas as pd

def gpkg_to_csv(gpkg_path, csv_path, layer=None, include_geometry=False, geometry_format="wkt"):
    gdf = gpd.read_file(gpkg_path, layer=layer)

    if include_geometry:
        if geometry_format == "wkt":
            gdf["geometry"] = gdf.geometry.to_wkt()
        elif geometry_format == "xy":
            gdf["x"] = gdf.geometry.centroid.x
            gdf["y"] = gdf.geometry.centroid.y
            gdf = gdf.drop(columns="geometry")
        else:
            raise ValueError("geometry_format must be 'wkt' or 'xy'")
    else:
        gdf = gdf.drop(columns="geometry", errors="ignore")

    gdf.to_csv(csv_path, index=False)
    print(f"Saved CSV to: {csv_path}")


if __name__ == "__main__":
    gpkg_path = r"C:\Users\blazb\Desktop\Magistrska\Data\Working_data\SAT_merged.gpkg"
    csv_path = r"C:\Users\blazb\Desktop\Magistrska\Data\Working_data\SAT_merged.csv"
    layer = None

    gpkg_to_csv(gpkg_path, csv_path, layer=layer, include_geometry=False)
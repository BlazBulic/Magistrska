from pre_processing import *
from data_manipulation import *
from plotting import *
from clustering import *

def main():

    LIDAR_data_path = r"C:\Users\blazb\Desktop\Magistrska\Data\GT_LIDAR\GKOT_586_165.laz"
    GT_ortofoto_path = r"C:\Users\blazb\Desktop\Magistrska\Data\GT_Ortofoto\EKRZ_20260107.shp"
    
    polygons = read_ortophoto(GT_ortofoto_path)
    las, dims = read_lidar_data(LIDAR_data_path)

    vegetation_gdf, intersecting_polygons, intersecting_points = get_intersecting_structures(las, polygons)

    plot_lidar_and_polygons(vegetation_gdf, intersecting_polygons, save=True)

    output_laz = r"C:\Users\blazb\Desktop\Magistrska\Data\Working_data\trees_segmented.laz"
    r_script_path = r"C:\Users\blazb\Desktop\Magistrska\Code\segment_trees.R"
    segmented_trees_path = segment_trees_r(LIDAR_data_path, output_laz, r_script_path)
    #segmented_trees_path = r"C:\Users\blazb\Desktop\Magistrska\Data\Working_data\trees_segmented.laz"

    # get the single trees gdf
    segmented_trees_las, _ = read_lidar_data(segmented_trees_path)
    segmented_trees_gdf = las_to_gdf(segmented_trees_las)

    plot_trees_simple(segmented_trees_gdf, save=True)
    plot_trees_and_polygons(segmented_trees_gdf, intersecting_points, save=True)

    # keep the relevant trees inside the anotated polygons
    relevant_trees_gdf = filter_trees_in_polygons(segmented_trees_gdf, intersecting_polygons)

    # clustering
    dbscan_labels = cluster_dbscan(relevant_trees_gdf)
    graph_labels = cluster_graph_radius(relevant_trees_gdf)
    plot_clusterings(relevant_trees_gdf, save=True, DBSCAN=dbscan_labels, GraphBased=graph_labels)


if __name__ == '__main__':
    main()
from pathlib import Path
from pre_processing import *
from data_manipulation import *
from plotting import *
from clustering import *

def main():

    #LIDAR_data_path = r"C:\Users\blazb\Desktop\Magistrska\Data\GT_LIDAR\GKOT_586_165.laz"
    block_number = 1
    size_of_block = 4
    LIDAR_folder_path = rf"C:\Users\blazb\Desktop\Magistrska\Data\GT_LIDAR\block{block_number}_{size_of_block}by{size_of_block}"
    segmented_output_folder = rf"C:\Users\blazb\Desktop\Magistrska\Data\Working_data\trees_segmented_catalog_block{block_number}_{size_of_block}by{size_of_block}"
    # polygons with added water distance
    GT_ortofoto_path = r"Data/Working_data/vegetation_with_water_distances/vegetation_with_water_distances.shp"

    polygons = read_ortophoto(GT_ortofoto_path)
    #las, dims = read_lidar_folder(LIDAR_folder_path)

    merged_laz_path = r"C:\Users\blazb\Desktop\Magistrska\Data\Working_data\merged_input.laz"
    #save_las(las, merged_laz_path)  

    output_laz = fr"C:\Users\blazb\Desktop\Magistrska\Data\Working_data\trees_segmented_block{block_number}_{size_of_block}by{size_of_block}"
    r_script_path = r"C:\Users\blazb\Desktop\Magistrska\Code\segment_trees_catalog.R"
    segmented_trees_path = segment_trees_r(LIDAR_folder_path, segmented_output_folder, r_script_path)
    #segmented_trees_path = rf"C:\Users\blazb\Desktop\Magistrska\Data\Working_data\trees_segmented_block{block_number}_{size_of_block}by{size_of_block}.laz"

    print("segmented the trees")

    # get the single trees gdf
    segmented_trees_las, _ = read_lidar_folder(segmented_trees_path)
    segmented_trees_las.write(rf"C:\Users\blazb\Desktop\Magistrska\Data\Working_data\trees_segmented_block{block_number}_{size_of_block}by{size_of_block}.laz")
    segmented_trees_gdf = las_to_gdf(segmented_trees_las)
    segmented_trees_gdf = remove_duplicate_trees(segmented_trees_gdf)

    print("got individial trees")

    #plot_trees_simple(segmented_trees_gdf, save=True)
    #plot_trees_and_polygons(segmented_trees_gdf, intersecting_points, save=True)

    vegetation_gdf, intersecting_polygons, intersecting_points = get_intersecting_structures_from_trees(segmented_trees_gdf, polygons)
    #plot_lidar_and_polygons(vegetation_gdf, intersecting_polygons, save=True)

    # keep the relevant trees inside the anotated polygons
    relevant_trees_gdf = filter_trees_in_polygons(segmented_trees_gdf, intersecting_polygons)
    #print(relevant_trees_gdf.columns)

    # multi step clustering
    labels = multi_step_clustering_plot(relevant_trees_gdf,height_col="Z",height_mode="neighbor_percentile",plot=True,save=True)
    #print(intersecting_points.columns)

    print("clustered the trees")

    #consturct the dataset
    dataset_path = rf"C:\Users\blazb\Desktop\Magistrska\Data\Working_data\cluster_dataset_block{block_number}_{size_of_block}by{size_of_block}.gpkg"
    dataset = build_cluster_dataset_from_labels(trees_gdf=relevant_trees_gdf,labels=labels,polygons_gdf=intersecting_polygons,target_col="OPIS",dist_to_water_col="edge_dist_",height_col="Z",cluster_geom="buffer_union",density_mode="geom_area", save_path=dataset_path)
    #print(dataset.columns)


if __name__ == '__main__':
    main()
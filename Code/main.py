from pathlib import Path
from pre_processing import *
from data_manipulation import *
from plotting import *
from clustering import *

RECOMPUTE = True  # True = run R, False = reuse saved file

def aggregate_sat_folder_to_trees(sat_output_folder: str, polygons: gpd.GeoDataFrame, instance_col: str = "PredInstance", height_col: str = "Z", min_overlap_ratio: float = 0.1, min_points_inside: int = 1, min_points_per_tree: int = 10, crs_epsg: int = 3794) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    sat_output_folder = Path(sat_output_folder)

    laz_files = sorted([
        p for p in sat_output_folder.glob("*.laz")
        if p.name.lower() != "merged.laz"
    ])

    if len(laz_files) == 0:
        raise FileNotFoundError(f"No .laz files found in {sat_output_folder}")

    all_trees = []
    all_intersecting_points = []
    all_intersecting_polygons = []

    global_offset = 0

    for file_idx, laz_path in enumerate(laz_files):
        print(f"Processing SAT file {file_idx + 1}/{len(laz_files)}: {laz_path.name}")

        sat_las, _ = read_lidar_data(str(laz_path))

        relevant_sat_points_gdf, intersecting_polygons, intersecting_points = get_intersecting_structures_sat(
            sat_las,
            polygons,
            instance_col=instance_col,
            min_overlap_ratio=min_overlap_ratio,
            min_points_inside=min_points_inside,
            crs_epsg=crs_epsg
        )

        if len(relevant_sat_points_gdf) == 0:
            print("  No relevant SAT points in this file.")
            continue

        trees_gdf = aggregate_sat_points_to_trees(
            relevant_sat_points_gdf,
            instance_col=instance_col,
            height_col=height_col,
            min_points_per_tree=min_points_per_tree
        )

        if len(trees_gdf) == 0:
            print("  No trees after aggregation in this file.")
            continue

        trees_gdf["local_treeID"] = trees_gdf["treeID"]
        trees_gdf["source_file"] = laz_path.name

        trees_gdf["treeID"] = trees_gdf["treeID"].astype(int) + global_offset
        global_offset = int(trees_gdf["treeID"].max()) + 1

        all_trees.append(trees_gdf)
        all_intersecting_points.append(intersecting_points)
        all_intersecting_polygons.append(intersecting_polygons)

        print(f"  Trees kept from file: {len(trees_gdf)}")

    if len(all_trees) == 0:
        empty = gpd.GeoDataFrame(columns=["treeID", "geometry"], geometry="geometry", crs=f"EPSG:{crs_epsg}")
        return empty, polygons.iloc[0:0].copy(), polygons.iloc[0:0].copy()

    trees_all = gpd.GeoDataFrame(pd.concat(all_trees, ignore_index=True), geometry="geometry", crs=all_trees[0].crs)

    intersecting_points_all = gpd.GeoDataFrame(pd.concat(all_intersecting_points, ignore_index=True), geometry="geometry", crs=all_intersecting_points[0].crs) if len(all_intersecting_points) > 0 else gpd.GeoDataFrame(geometry="geometry", crs=trees_all.crs)
    intersecting_polygons_all = gpd.GeoDataFrame(pd.concat(all_intersecting_polygons, ignore_index=True), geometry="geometry", crs=all_intersecting_polygons[0].crs).drop_duplicates() if len(all_intersecting_polygons) > 0 else polygons.iloc[0:0].copy()

    print("Total aggregated SAT trees:", len(trees_all))

    return trees_all, intersecting_polygons_all, intersecting_points_all

def R_version():
    block_number = 1
    size_of_block = 4

    lidar_folder_path = rf"C:\Users\blazb\Desktop\Magistrska\Data\GT_LIDAR\block{block_number}_{size_of_block}by{size_of_block}"
    segmented_output_folder = rf"C:\Users\blazb\Desktop\Magistrska\Data\Working_data\trees_segmented_catalog_block{block_number}_{size_of_block}by{size_of_block}_EXPERIMENTAL"
    merged_segmented_laz_path = rf"C:\Users\blazb\Desktop\Magistrska\Data\Working_data\trees_segmented_block{block_number}_{size_of_block}by{size_of_block}_EXPERIMENTAL.laz"

    gt_ortofoto_path = r"Data/Working_data/vegetation_with_water_distances/vegetation_with_water_distances.shp"
    r_script_path = r"C:\Users\blazb\Desktop\Magistrska\Code\segment_trees_catalog.R"

    polygons = read_ortophoto(gt_ortofoto_path, "OPIS", "Vodna prvina")

    # -------------------------
    # STEP 1: get segmented trees
    # -------------------------
    if RECOMPUTE:
        # run R segmentation
        segment_trees_r(lidar_folder_path, segmented_output_folder, r_script_path)
        print("segmented the trees (R)")

        # merge chunk outputs
        segmented_trees_las, _ = read_lidar_folder(segmented_output_folder)

        # save merged file
        segmented_trees_las.write(merged_segmented_laz_path)
        print("saved merged .laz")

    else:
        # just load existing merged file
        segmented_trees_las, _ = read_lidar_data(merged_segmented_laz_path)
        print("loaded merged .laz")

    # -------------------------
    # STEP 2: convert + clean
    # -------------------------
    segmented_trees_gdf = las_to_gdf(segmented_trees_las)
    segmented_trees_gdf = remove_duplicate_trees(segmented_trees_gdf)

    print("got individual trees")

    #plot_trees_simple(segmented_trees_gdf, save=True)
    #plot_trees_and_polygons(segmented_trees_gdf, intersecting_points, save=True)

    # -------------------------
    # STEP 3: spatial filtering
    # -------------------------
    relevant_trees_gdf, intersecting_polygons, intersecting_points = get_intersecting_structures_from_trees(segmented_trees_gdf, polygons)

    #print(f"Kept {relevant_trees_gdf.length} relevant trees")

    # -------------------------
    # STEP 4: clustering
    # -------------------------
    labels = multi_step_clustering_plot(
        relevant_trees_gdf,
        height_col="Z",
        height_mode="neighbor_percentile",
        merge_pair_xy=10,
        plot=False,
        save=False
    )

    #cluster_id_to_export = 36
    #cluster_csv_path = rf"C:\Users\blazb\Desktop\Magistrska\Data\Working_data\cluster_{cluster_id_to_export}_points.csv"
    #save_cluster_to_csv(relevant_trees_gdf, labels, cluster_id_to_export, cluster_csv_path)

    """ labels_refined, split_log = split_end_clusters_dbscan(
    relevant_trees_gdf,
    labels,
    min_cluster_size=500,
    min_child_size=50,
    min_child_fraction=0.08,
    eps_factor=3,
    min_samples=5,
    max_meaningful_clusters=2,
    max_iter=1,
    plot=False,
    verbose=False,
    debug=False,
    debug_cluster_id=36
    ) """


    # -------------------------
    # STEP 5: dataset
    # -------------------------
    dataset_path = rf"C:\Users\blazb\Desktop\Magistrska\Data\Working_data\cluster_dataset_block{block_number}_{size_of_block}by{size_of_block}_10_1_EXP.gpkg"

    dataset = build_cluster_dataset_from_labels(
        trees_gdf=relevant_trees_gdf,
        labels=labels,
        #labels=labels_refined,
        polygons_gdf=intersecting_polygons,
        target_col="OPIS",
        dist_to_water_col="edge_dist_",
        height_col="Z",
        cluster_geom="buffer_union",
        density_mode="geom_area",
        save_path=dataset_path
    )


    # -------------------------
    # STEP 6: evaluation
    # -------------------------
    results = evaluate_clustering_against_polygons(
        trees_gdf=relevant_trees_gdf,
        cluster_labels=labels,
        polygons_gdf=intersecting_polygons,
        polygon_id_col="EKRZ_PID",
        cluster_geom_method="buffer_union",
        point_buffer=0.5
    )

    print("ARI:", results["ARI"])
    print("Purity:", results["purity"])
    print("Completeness:", results["completeness"])

def SAT_version() :
    block_number = 1
    size_of_block = 4

    # original input block, only kept for reference
    lidar_folder_path = rf"C:\Users\blazb\Desktop\Magistrska\Data\GT_LIDAR\block{block_number}_{size_of_block}by{size_of_block}"

    # SegmentAnyTree output
    sat_output_folder = rf"C:\Users\blazb\Desktop\Magistrska\Data\Working_data\segmentanytree_out\block1_3by3_preprocessed025_out"
    merged_segmented_laz_path = rf"C:\Users\blazb\Desktop\Magistrska\Data\Working_data\SAT_merged.laz"

    gt_ortofoto_path = r"C:\Users\blazb\Desktop\Magistrska\Data\Working_data\vegetation_with_water_distances\vegetation_with_water_distances.shp"

    polygons = read_ortophoto(gt_ortofoto_path, "OPIS", "Vodna prvina")

    # -------------------------
    # STEP 1: get segmented trees from SegmentAnyTree
    # -------------------------
    """ if RECOMPUTE:
        # assumes SAT already wrote one or more .las/.laz files into sat_output_folder
        segmented_trees_las, _ = read_lidar_folder(sat_output_folder)

        segmented_trees_las.write(merged_segmented_laz_path)
        print("loaded SegmentAnyTree output and saved merged .laz")

    else:
        segmented_trees_las, _ = read_lidar_data(merged_segmented_laz_path)
        print("loaded merged SegmentAnyTree .laz")

    # -------------------------
    # STEP 2: convert + spacial filter
    # -------------------------

    relevant_sat_points_gdf, intersecting_polygons, intersecting_points = get_intersecting_structures_sat(segmented_trees_las, polygons, instance_col="PredInstance", min_overlap_ratio=0.1, min_points_inside=1, crs_epsg=3794)

    relevant_trees_gdf = aggregate_sat_points_to_trees(relevant_sat_points_gdf, instance_col="PredInstance", height_col="Z", min_points_per_tree=10)

    print("Relevant trees after SAT point-overlap filtering:", len(relevant_trees_gdf)) """

    relevant_trees_gdf, intersecting_polygons, intersecting_points = aggregate_sat_folder_to_trees(
    sat_output_folder=sat_output_folder,
    polygons=polygons,
    instance_col="PredInstance",
    height_col="Z",
    min_overlap_ratio=0.1,
    min_points_inside=1,
    min_points_per_tree=10,
    crs_epsg=3794
)

    print("Relevant trees after SAT point-overlap filtering:", len(relevant_trees_gdf))

    # -------------------------
    # STEP 4: clustering
    # -------------------------
    labels = multi_step_clustering_plot(
        relevant_trees_gdf,
        height_col="Z",
        height_mode="neighbor_percentile",
        merge_pair_xy=10,
        plot=False,
        save=False
    )

    # -------------------------
    # STEP 5: dataset
    # -------------------------
    dataset_path = rf"C:\Users\blazb\Desktop\Magistrska\Data\Working_data\SAT_merged.gpkg"
    #dataset_path = rf"C:\Users\blazb\Desktop\Magistrska\Data\Working_data\cluster_dataset_SAT_GKOT_585_165.gpkg"

    dataset = build_cluster_dataset_from_labels(
        trees_gdf=relevant_trees_gdf,
        labels=labels,
        polygons_gdf=intersecting_polygons,
        target_col="OPIS",
        dist_to_water_col="edge_dist_",
        height_col="height_p90",
        cluster_geom="buffer_union",
        density_mode="geom_area",
        save_path=dataset_path
    )

    # -------------------------
    # STEP 6: evaluation
    # -------------------------
    results = evaluate_clustering_against_polygons(
        trees_gdf=relevant_trees_gdf,
        cluster_labels=labels,
        polygons_gdf=intersecting_polygons,
        polygon_id_col="EKRZ_PID",
        cluster_geom_method="buffer_union",
        point_buffer=0.5
    )

    print("ARI:", results["ARI"])
    print("Purity:", results["purity"])
    print("Completeness:", results["completeness"])

    return dataset, results

def main():
    SAT_version()

if __name__ == '__main__':
    main()
library(lidR)
library(terra)
library(dplyr)
library(sf)

args <- commandArgs(trailingOnly = TRUE)
input_folder <- args[1]
output_folder <- args[2]

print(input_folder)
print(output_folder)

dir.create(output_folder, recursive = TRUE, showWarnings = FALSE)

ctg <- readLAScatalog(input_folder)

opt_chunk_size(ctg) <- 1000
opt_chunk_buffer(ctg) <- 20
opt_wall_to_wall(ctg) <- TRUE

process_chunk <- function(chunk, output_folder) {
  tryCatch({
    las <- readLAS(chunk)
    if (is.empty(las)) return(NULL)

    las <- normalize_height(las, knnidw())
    #las_veg <- filter_poi(las, Classification %in% c(3, 4, 5) & NumberOfReturns > 1)
    las_veg <- filter_poi(las, Classification %in% c(3, 4, 5))
    if (is.empty(las_veg)) return(NULL)

    chm <- rasterize_canopy(las_veg, res = 1, algorithm = p2r(subcircle = 0.2))
    if (is.null(chm)) return(NULL)
    if (nrow(chm) == 0 || ncol(chm) == 0) return(NULL)

    if (nrow(chm) >= 3 && ncol(chm) >= 3) {
      #chm <- focal(chm, w = matrix(1, 5, 5), fun = mean, na.policy = "omit")
      chm <- focal(chm, w = matrix(1, 3, 3), fun = mean, na.policy = "omit")
    }

    #ttops <- locate_trees(chm, lmf(ws = function(h) { 1.2 + 0.08 * h }, hmin = 1), uniqueness = "bitmerge")
    ttops <- locate_trees(chm, lmf(ws = function(h) { 1.0 + 0.06 * h }, hmin = 1), uniqueness = "bitmerge")
    if (is.null(ttops) || nrow(ttops) == 0) return(NULL)

    las_trees <- segment_trees(las_veg, dalponte2016(chm, ttops, th_tree = 1))
    if (is.empty(las_trees)) return(NULL)

    df <- as.data.frame(las_trees@data)
    if (!("treeID" %in% names(df))) return(NULL)

    df <- df[!is.na(df$treeID), ]
    if (nrow(df) == 0) return(NULL)

    tree_points <- df %>%
      group_by(treeID) %>%
      summarise(X = mean(X), Y = mean(Y), Z = mean(Z), .groups = "drop")

    if (nrow(tree_points) == 0) return(NULL)

    bb <- sf::st_bbox(chunk)
    tree_points <- tree_points %>%
    filter(X >= bb["xmin"], X <= bb["xmax"], Y >= bb["ymin"], Y <= bb["ymax"])

    if (nrow(tree_points) == 0) return(NULL)

    las_centroids <- LAS(tree_points)

    out_path <- tempfile(pattern = "centroids_", tmpdir = output_folder, fileext = ".laz")

    message("Writing: ", out_path)
    message("Trees in chunk: ", nrow(tree_points))

    writeLAS(las_centroids, out_path)
    return(1L)
  }, error = function(e) {
    message("Chunk failed: ", conditionMessage(e))
    return(NULL)
  })
}

opts <- list(need_buffer = TRUE, automerge = FALSE, drop_null = TRUE)
result <- catalog_apply(ctg, process_chunk, output_folder = output_folder, .options = opts)

print(result)
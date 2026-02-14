library(lidR)
library(terra)
library(dplyr)
library(sf)

# for using this code in python program
args <- commandArgs(trailingOnly = TRUE)
input_las  <- args[1]
output_las <- args[2]
print(input_las)
print(output_las)

# read lidar data
las <- readLAS(input_las)
stopifnot(!is.empty(las))

# classify ground (don't need, because lidar already has ground classified?)
#las <- classify_ground(las, csf())

# normalize heights and keep only the points classified as vegetation
las <- normalize_height(las, knnidw())
las_veg <- filter_poi(las, Classification %in% c(3,4,5) & NumberOfReturns > 1)

# get segmentation
chm <- rasterize_canopy(las_veg, res = 0.5, p2r(subcircle = 0.2))
chm <- focal(chm, w = matrix(5,5,3), fun = mean, na.policy = "omit")

ttops <- locate_trees(
  chm,
  lmf(ws = function(h) { 2 + 0.07 * h }, hmin = 6)
)

las_trees <- segment_trees(
  las_veg,
  dalponte2016(chm, ttops, th_tree = 6)
)

# Write ONLY the segmented LAS
#writeLAS(las_trees, output_las)


df <- as.data.frame(las_trees@data)

# getting 1 point per tree
tree_points <- df %>%
  group_by(treeID) %>%
  #filter(n() > 100) %>%
  summarise(
    X = mean(X),
    Y = mean(Y),
    Z = max(Z),
    .groups = "drop"
  )

las_centroids <- LAS(tree_points)

writeLAS(las_centroids, output_las)
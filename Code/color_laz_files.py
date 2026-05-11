
import laspy
import numpy as np

las = laspy.read(r"C:\Users\blazb\Desktop\Magistrska\Data\Working_data\segmentanytree_out\GKOT_585_165_preprocessed_out.laz")

ids = np.array(las.PredInstance)  # change if needed

unique_ids = np.unique(ids)
np.random.seed(0)

colors_map = {
    uid: np.random.randint(0, 65535, 3)
    for uid in unique_ids
}

rgb = np.array([colors_map[i] for i in ids], dtype=np.uint16)

# --- IMPORTANT PART ---
# Create new LAS with RGB-compatible format
header = las.header
header.point_format = laspy.PointFormat(7)  # format with RGB

new_las = laspy.LasData(header)

# copy existing dimensions
for dim in las.point_format.dimension_names:
    if dim in new_las.point_format.dimension_names:
        new_las[dim] = las[dim]

# assign colors
new_las.red = rgb[:, 0]
new_las.green = rgb[:, 1]
new_las.blue = rgb[:, 2]

new_las.write(r"C:\Users\blazb\Desktop\Magistrska\Data\Working_data\segmentanytree_out\GKOT_585_165_preprocessed_out_colored_by_id.laz")
import argparse
import numpy as np
import laspy
from scipy.spatial import cKDTree


def normalize_height(las, k=8):
    ground_mask = las.classification == 2

    if ground_mask.sum() == 0:
        raise ValueError("No ground points found: Classification == 2")

    all_xy = np.column_stack((las.x, las.y))
    ground_xy = np.column_stack((las.x[ground_mask], las.y[ground_mask]))
    ground_z = las.z[ground_mask]

    tree = cKDTree(ground_xy)
    distances, indices = tree.query(all_xy, k=min(k, len(ground_z)))

    if indices.ndim == 1:
        ground_estimate = ground_z[indices]
    else:
        weights = 1.0 / np.maximum(distances, 1e-6)
        ground_estimate = np.sum(weights * ground_z[indices], axis=1) / np.sum(weights, axis=1)

    return las.z - ground_estimate


def random_subsample(indices, keep_fraction, seed=42):
    if keep_fraction >= 1.0:
        return indices

    rng = np.random.default_rng(seed)
    keep_count = int(len(indices) * keep_fraction)

    if keep_count <= 0:
        raise ValueError("Subsampling removed all points.")

    return np.sort(rng.choice(indices, size=keep_count, replace=False))


def preprocess_laz(input_path, output_path, min_height, keep_fraction, seed):
    print(f"Reading: {input_path}")
    las = laspy.read(input_path)

    print(f"Original points: {len(las.points):,}")

    print("Normalizing height using ground points...")
    normalized_z = normalize_height(las)

    veg_mask = np.isin(las.classification, [3, 4, 5])
    height_mask = normalized_z >= min_height

    mask = veg_mask & height_mask
    selected_indices = np.where(mask)[0]

    print(f"After vegetation + height filtering: {len(selected_indices):,}")

    selected_indices = random_subsample(selected_indices, keep_fraction, seed)

    print(f"After random subsampling: {len(selected_indices):,}")

    out_las = laspy.LasData(las.header)
    out_las.points = las.points[selected_indices].copy()

    print("Replacing Z with normalized height...")

    z_scale = out_las.header.scales[2]
    z_offset = out_las.header.offsets[2]

    out_las.Z = np.round((normalized_z[selected_indices] - z_offset) / z_scale).astype(np.int32)

    print(f"Writing: {output_path}")
    out_las.write(output_path)

    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Preprocess LAZ for SegmentAnyTree.")
    parser.add_argument("input", help="Input .laz/.las file")
    parser.add_argument("output", help="Output .laz/.las file")
    parser.add_argument("--min-height", type=float, default=1.0, help="Minimum normalized height to keep")
    parser.add_argument("--keep-fraction", type=float, default=0.5, help="Fraction of points to keep after filtering")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    if not (0 < args.keep_fraction <= 1):
        raise ValueError("--keep-fraction must be in (0, 1].")

    preprocess_laz(args.input, args.output, args.min_height, args.keep_fraction, args.seed)


if __name__ == "__main__":
    main()
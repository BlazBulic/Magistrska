import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
from shapely.geometry import MultiPoint, Polygon, MultiPolygon, GeometryCollection
from shapely.ops import unary_union


def plot_lidar_and_polygons(gdf, intersecting_polygons, save=False, save_path=r"C:\Users\blazb\Desktop\Magistrska\Figures\Working_process\figure_"):
    classification_dict = {
    "0":"neklasificirano",
    "1":"neklasificirano oz. tocke, \nki jih ni mogoce klasificirati v druge razrede ",
    "2":"tla, ki vkljucuje tudi tocke pod mostovi in viadukti",
    "3":"nizka vegetacija (< 3 m)",
    "4":"srednja vegetacija (3 - 10 m)",
    "5":"visoka vegetacija (> 10 m)",
    "6":"stavbe (strehe in fasade)",
    "7":"nizke tocke (uvozi v garaze, stopnice, jaski)",
    "8":"rezervirano",
    "9":"voda",
    "17":"mostovi in viadukti",
    "18":"sum",
    "21":"sneg"
    }   

    # only classes that are actually present
    classes = np.sort(gdf["classification"].unique())
    cmap = cm.get_cmap("tab20", len(classes))

    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot sampled LiDAR points
    gdf.plot(ax=ax, markersize=1, column="classification", categorical=True, cmap=cmap, alpha=0.5, label='LiDAR points (sampled)')

    intersecting_polygons.plot(ax=ax,facecolor="none",edgecolor="red",linewidth=1.5,zorder=5)

    # build custom legend
    handles = []
    for i, cls in enumerate(classes):
        label = f"{cls}: {classification_dict[str(cls)]}"
        handles.append(
            mpatches.Patch(color=cmap(i), label=label)
        )

    ax.legend(
        handles=handles,
        title="LAS klasifikacija",
        loc="upper right",
        fontsize=9,
        title_fontsize=10
    )

    # Label polygons at their centroids
    for idx, row in intersecting_polygons.iterrows():
        centroid = row.geometry.centroid
        # Use a unique ID or name column if available
        if 'opis' in row:
            label_text = str(row['opis'])
        elif 'OPIS' in row:
            label_text = str(row['OPIS'])
        else:
            label_text = str(idx)  # fallback to index
        ax.text(centroid.x, centroid.y, label_text, fontsize=8, color='darkred')



    # Add title and legend
    plt.title("LiDAR points with intersecting polygons and labels", fontsize=14)
    plt.xlabel("X")
    plt.ylabel("Y")
    if save:
        plt.savefig(save_path + "all_vegetation")
    else:
        plt.show()


def plot_trees_simple(trees_gdf, save=False, save_path=r"C:\Users\blazb\Desktop\Magistrska\Figures\Working_process\figure_"):

    fig, ax = plt.subplots(figsize=(10,10))
    trees_gdf.plot(ax=ax, markersize=5, color='blue', alpha=0.6, label='Tree centroids')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Aggregated Trees (1 point per tree)")
    if save:
        plt.savefig(save_path + "segmented_trees")
    else:
        plt.show()


def plot_trees_and_polygons(trees_gdf, polygons_gdf, save=False, save_path=r"C:\Users\blazb\Desktop\Magistrska\Figures\Working_process\figure_"):
    fig, ax = plt.subplots(figsize=(10,10))

    polygons_gdf.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=1, alpha=0.7, label='Polygons')
    trees_gdf.plot(ax=ax, markersize=5, color='blue', alpha=0.6, label='Tree centroids')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Trees and Polygons Alignment Check")
    ax.legend()
    if save:
        plt.savefig(save_path + "segmented_trees_and_polygons")
    else:
        plt.show()


def plot_clusterings(gdf, save=False, save_path=r"C:\Users\blazb\Desktop\Magistrska\Figures\Working_process\figure_", **clusterings):
    n = len(clusterings)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 6), squeeze=False)

    n = len(clusterings)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 6), squeeze=False)

    for ax, (name, labels) in zip(axes[0], clusterings.items()):
        temp = gdf.copy()
        temp["cluster"] = labels

        unique_labels = np.unique(labels)
        n_labels = len(unique_labels)

        # Use a categorical colormap
        cmap = plt.get_cmap("tab20")  # can switch to "tab20b", "tab20c", etc.
        color_dict = {lbl: cmap(i % 20) for i, lbl in enumerate(unique_labels)}

        colors = temp["cluster"].map(color_dict)

        # Plot points with assigned colors
        temp.plot(
            color=colors,
            ax=ax,
            markersize=10
        )

        # Annotate cluster labels at centroids
        for lbl in unique_labels:
            if lbl == -1:  # optional: skip noise
                continue
            cluster_points = temp[temp["cluster"] == lbl].geometry
            x_mean = np.mean([pt.x for pt in cluster_points])
            y_mean = np.mean([pt.y for pt in cluster_points])
            ax.text(x_mean, y_mean, str(lbl), color="black", fontsize=8,
                    ha="center", va="center", weight="bold")

        ax.set_title(name)
        ax.set_axis_off()


    plt.tight_layout()
    if save:
        plt.savefig(save_path + "clusters")
    else:
        plt.show()
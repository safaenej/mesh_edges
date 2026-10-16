"""
================================================================================
Visualisation : le même maillage numéroté par sommets puis par arêtes
================================================================================

Produit mesh_plot.png, qui montre côte à côte les deux descriptions du
maillage : à gauche les indices des sommets, à droite les indices des arêtes
placés au milieu de chaque segment.

Usage :
    python mesh_plot.py
================================================================================
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

from mesh_edges import nodes_to_edges


def main(n_points=60, seed=0):
    rng = np.random.default_rng(seed)
    points = rng.random((n_points, 2))
    triangles = Delaunay(points).simplices

    edge_elements, unique_edges = nodes_to_edges(triangles)

    print(f"{len(triangles)} triangles, {len(unique_edges)} arêtes uniques")
    print("\nTrois premiers triangles, par sommets puis par arêtes :")
    for i in range(3):
        print(f"  triangle {i} : sommets {triangles[i]}  →  arêtes {edge_elements[i]}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    axes[0].triplot(points[:, 0], points[:, 1], triangles, lw=0.8)
    axes[0].plot(points[:, 0], points[:, 1], "o", ms=4)
    for i, (x, y) in enumerate(points):
        axes[0].text(x, y, str(i), fontsize=8, color="tab:blue")
    axes[0].set_title("Par sommets")

    axes[1].triplot(points[:, 0], points[:, 1], triangles, lw=0.8)
    axes[1].plot(points[:, 0], points[:, 1], "o", ms=4)
    for k, (i, j) in enumerate(unique_edges):
        mx = (points[i, 0] + points[j, 0]) / 2
        my = (points[i, 1] + points[j, 1]) / 2
        axes[1].text(mx, my, str(k), fontsize=7, color="tab:red", ha="center")
    axes[1].set_title("Par arêtes")

    for ax in axes:
        ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig("mesh_plot.png", dpi=150, bbox_inches="tight")
    print("\nFigure écrite dans mesh_plot.png")


if __name__ == "__main__":
    main()

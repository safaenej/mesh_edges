"""
================================================================================
Comparaison des performances des trois implémentations
================================================================================

Produit :
  - benchmark.png : temps de calcul en fonction de la taille du maillage
  - un tableau récapitulatif dans le terminal

Usage :
    python benchmark.py
================================================================================
"""

import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

from mesh_edges import nodes_to_edges, nodes_to_edges_set, nodes_to_edges_naive


SIZES_ALL = [100, 300, 1_000, 3_000]              # la version naïve s'arrête là
SIZES_FAST = [100, 300, 1_000, 3_000, 10_000, 30_000, 100_000]


def make_mesh(n_points, seed=0):
    rng = np.random.default_rng(seed)
    points = rng.random((n_points, 2))
    return Delaunay(points).simplices


def timeit(fn, triangles, repeats=3):
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn(triangles)
        best = min(best, time.perf_counter() - start)
    return best


def main():
    results = {"naive": [], "set": [], "vectorised": []}

    print(f"{'points':>8} {'triangles':>10} {'naive (s)':>12} "
          f"{'set (s)':>10} {'vect. (s)':>11} {'vs naive':>10} {'vs set':>8}")
    print("-" * 74)

    for n in SIZES_FAST:
        triangles = make_mesh(n)

        t_vec = timeit(nodes_to_edges, triangles)
        t_set = timeit(nodes_to_edges_set, triangles)
        t_naive = timeit(nodes_to_edges_naive, triangles, repeats=1) if n in SIZES_ALL else None

        results["vectorised"].append(t_vec)
        results["set"].append(t_set)
        results["naive"].append(t_naive)

        naive_str = f"{t_naive:12.4f}" if t_naive is not None else f"{'—':>12}"
        g_naive = f"{t_naive / t_vec:9.0f}×" if t_naive is not None else f"{'—':>10}"
        g_set = f"{t_set / t_vec:6.1f}×"
        print(f"{n:8d} {len(triangles):10d} {naive_str} "
              f"{t_set:10.4f} {t_vec:11.4f} {g_naive:>10} {g_set:>8}")

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(7, 4.5))

    naive_x = [n for n, t in zip(SIZES_FAST, results["naive"]) if t is not None]
    naive_y = [t for t in results["naive"] if t is not None]

    ax.loglog(naive_x, naive_y, "o-", label="liste Python, O(n²)")
    ax.loglog(SIZES_FAST, results["set"], "s-", label="ensemble Python, O(n)")
    ax.loglog(SIZES_FAST, results["vectorised"], "^-", label="NumPy vectorisé")

    ax.set_xlabel("nombre de points du maillage")
    ax.set_ylabel("temps de calcul (s)")
    ax.set_title("Conversion sommets → arêtes")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("benchmark.png", dpi=150)
    print("\nFigure écrite dans benchmark.png")


if __name__ == "__main__":
    main()

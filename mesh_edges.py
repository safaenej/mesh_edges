"""
================================================================================
Conversion d'un maillage triangulaire : des sommets vers les arêtes
================================================================================

Un maillage triangulaire est habituellement décrit par un tableau (N, 3)
contenant, pour chaque triangle, les indices de ses trois sommets.

Certaines méthodes numériques — éléments finis de type Raviart-Thomas ou
Nédélec, volumes finis, parcours de graphe — ont besoin de la description
duale : pour chaque triangle, les indices de ses trois arêtes.

Ce module effectue cette conversion. La difficulté n'est pas la formule,
qui est immédiate, mais le temps de calcul : chaque arête interne est
partagée par deux triangles, et il faut donc identifier les doublons sans
parcourir le maillage élément par élément.

Trois implémentations sont fournies, de la plus naïve à la plus rapide,
afin de rendre l'écart de performance mesurable.
================================================================================
"""

import numpy as np


# ==============================================================================
# Version 1 — liste Python et test d'appartenance
# ==============================================================================

def nodes_to_edges_naive(triangles):
    """
    Approche directe : on construit la liste des arêtes et on vérifie
    l'absence de doublon avant chaque ajout.

    Le test `edge not in edges` parcourt toute la liste à chaque fois,
    ce qui donne une complexité quadratique en nombre d'arêtes.
    Inutilisable au-delà de quelques milliers de triangles.
    """
    edges = []
    for triangle in triangles:
        for a, b in ((0, 1), (1, 2), (0, 2)):
            edge = sorted([triangle[a], triangle[b]])
            if edge not in edges:
                edges.append(edge)
    return np.array(edges)


# ==============================================================================
# Version 2 — ensemble Python
# ==============================================================================

def nodes_to_edges_set(triangles):
    """
    Même logique, mais l'ensemble (`set`) assure l'unicité en temps constant.

    La complexité devient linéaire. Il reste cependant une boucle Python
    sur les triangles, dont le coût par itération est élevé.
    """
    edges = set()
    for triangle in triangles:
        for a, b in ((0, 1), (1, 2), (0, 2)):
            edges.add(tuple(sorted([triangle[a], triangle[b]])))
    return np.array(sorted(edges))


# ==============================================================================
# Version 3 — entièrement vectorisée
# ==============================================================================

def nodes_to_edges(triangles):
    """
    Conversion vectorisée, sans aucune boucle Python.

    Paramètre
    ---------
    triangles : ndarray de forme (N, 3)
        Indices des sommets de chaque triangle.

    Retourne
    --------
    edge_elements : ndarray de forme (N, 3)
        Indices des arêtes de chaque triangle.
    unique_edges : ndarray de forme (M, 2)
        Table des arêtes, chaque ligne donnant ses deux sommets, triés.

    Principe
    --------
    1. On extrait les trois arêtes de chaque triangle, ce qui donne 3N arêtes
       avec doublons.
    2. On trie chaque paire de sommets, pour que l'arête (4, 2) et l'arête
       (2, 4) deviennent identiques. Le tri se fait selon `axis=1`, donc à
       l'intérieur de chaque paire : un tri selon `axis=0` mélangerait les
       arêtes entre elles.
    3. `np.unique` avec `return_inverse` renvoie à la fois la table des arêtes
       uniques et, pour chacune des 3N arêtes de départ, l'indice de l'arête
       unique correspondante.
    4. Ce tableau `inverse` est remis en forme. Il est ordonné en trois blocs
       successifs — toutes les premières arêtes, puis toutes les deuxièmes,
       puis toutes les troisièmes. On le remet donc en forme en (3, N), puis
       on transpose pour obtenir (N, 3), c'est-à-dire les trois arêtes d'un
       même triangle sur une même ligne. Sans cette transposition, chaque
       ligne contiendrait la même arête de trois triangles différents.
    """
    triangles = np.asarray(triangles)

    e0 = triangles[:, [0, 1]]
    e1 = triangles[:, [1, 2]]
    e2 = triangles[:, [0, 2]]

    all_edges = np.vstack([e0, e1, e2])          # (3N, 2)
    all_edges = np.sort(all_edges, axis=1)       # tri à l'intérieur de chaque paire

    unique_edges, inverse = np.unique(all_edges, axis=0, return_inverse=True)

    n_triangles = triangles.shape[0]
    edge_elements = inverse.reshape(3, n_triangles).T

    return edge_elements, unique_edges

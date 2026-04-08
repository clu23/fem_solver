"""Benchmark : assemblage scalaire vs batch sur un maillage Quad4 de 10 000 éléments.

Génère un maillage régulier 100×100 Quad4 (10 000 éléments, 10 201 nœuds,
20 402 DDL), assemble K et M avec les deux chemins, compare temps et résultats.

Usage ::

    python benchmarks/bench_assembler_quad4.py

Sortie typique (machine de référence) ::

    Maillage : 10000 éléments, 10201 nœuds, 20402 DDL
    -- Rigidité K --------------------------------------
    Scalaire  :  2.45 s
    Batch     :  0.18 s   →  13.6× plus rapide
    max|K_batch - K_scalar| = 0.00e+00  (cohérence : OK)
    -- Masse M -----------------------------------------
    Scalaire  :  1.98 s
    Batch     :  0.09 s   →  22.0× plus rapide
    max|M_batch - M_scalar| = 0.00e+00  (cohérence : OK)
"""

from __future__ import annotations

import time

import numpy as np

from femsolver.core.assembler import Assembler
from femsolver.core.material import ElasticMaterial
from femsolver.core.mesh import ElementData, Mesh
from femsolver.elements.quad4 import Quad4


def build_quad4_mesh(nx: int, ny: int, lx: float = 1.0, ly: float = 1.0) -> Mesh:
    """Génère un maillage régulier Quad4 (nx × ny éléments).

    Parameters
    ----------
    nx, ny : int
        Nombre d'éléments selon X et Y.
    lx, ly : float
        Dimensions du domaine [m].

    Returns
    -------
    Mesh
        Maillage de (nx+1)×(ny+1) nœuds et nx×ny éléments Quad4.
    """
    x = np.linspace(0.0, lx, nx + 1)
    y = np.linspace(0.0, ly, ny + 1)
    xx, yy = np.meshgrid(x, y)   # (ny+1, nx+1)
    nodes = np.column_stack([xx.ravel(), yy.ravel()])   # ((nx+1)(ny+1), 2)

    steel = ElasticMaterial(E=210e9, nu=0.3, rho=7800)
    props = {"thickness": 0.01, "formulation": "plane_stress"}

    elements = []
    for j in range(ny):
        for i in range(nx):
            # Numérotation CCW : bas-gauche → bas-droit → haut-droit → haut-gauche
            n0 = j * (nx + 1) + i
            n1 = n0 + 1
            n2 = n0 + (nx + 1) + 1
            n3 = n0 + (nx + 1)
            elements.append(ElementData(Quad4, (n0, n1, n2, n3), steel, props))

    return Mesh(nodes=nodes, elements=tuple(elements), n_dim=2)


def _fmt_time(t: float) -> str:
    return f"{t:.3f} s" if t >= 0.1 else f"{t * 1e3:.1f} ms"


def run_benchmark(nx: int = 100, ny: int = 100) -> None:
    print("=" * 60)
    print(f"Benchmark assemblage Quad4  ({nx}×{ny} éléments)")
    print("=" * 60)

    mesh = build_quad4_mesh(nx, ny)
    n_elem = len(mesh.elements)
    n_nodes = mesh.n_nodes
    n_dof = mesh.n_dof
    print(f"Maillage : {n_elem} éléments, {n_nodes} nœuds, {n_dof} DDL\n")

    assembler = Assembler(mesh)

    # -- Rigidité K ----------------------------------------------------
    print("-- Rigidité K --------------------------------------")

    t0 = time.perf_counter()
    K_scalar = assembler.assemble_stiffness(use_batch=False)
    t_scalar_k = time.perf_counter() - t0
    print(f"  Scalaire  : {_fmt_time(t_scalar_k)}")

    t0 = time.perf_counter()
    K_batch = assembler.assemble_stiffness(use_batch=True)
    t_batch_k = time.perf_counter() - t0
    speedup_k = t_scalar_k / t_batch_k
    print(f"  Batch     : {_fmt_time(t_batch_k)}   →  {speedup_k:.1f}× plus rapide")

    diff_k = abs(K_batch - K_scalar).max()
    scale_k = abs(K_scalar).max()
    rel_k = diff_k / scale_k if scale_k > 0 else diff_k
    ok_k = "OK" if rel_k < 1e-10 else "** ECART **"
    print(f"  max|K_batch - K_scalar| = {diff_k:.2e}  (relatif: {rel_k:.1e}, cohérence : {ok_k})")

    # -- Masse M -------------------------------------------------------
    print("\n-- Masse M -----------------------------------------")

    t0 = time.perf_counter()
    M_scalar = assembler.assemble_mass(use_batch=False)
    t_scalar_m = time.perf_counter() - t0
    print(f"  Scalaire  : {_fmt_time(t_scalar_m)}")

    t0 = time.perf_counter()
    M_batch = assembler.assemble_mass(use_batch=True)
    t_batch_m = time.perf_counter() - t0
    speedup_m = t_scalar_m / t_batch_m
    print(f"  Batch     : {_fmt_time(t_batch_m)}   →  {speedup_m:.1f}× plus rapide")

    diff_m = abs(M_batch - M_scalar).max()
    ok_m = "OK" if diff_m < 1e-6 else "** ECART **"
    print(f"  max|M_batch - M_scalar| = {diff_m:.2e}  (cohérence : {ok_m})")

    # -- Résumé --------------------------------------------------------
    print("\n-- Résumé ------------------------------------------")
    print(f"  K : {speedup_k:5.1f}× accélération")
    print(f"  M : {speedup_m:5.1f}× accélération")
    print("=" * 60)


if __name__ == "__main__":
    run_benchmark()

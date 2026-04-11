"""Benchmark : assemblage scalaire vs batch sur 5 types d'éléments.

Compare les temps d'assemblage K et M en mode scalaire et batch pour :
- Tri3  (CST — 3 nœuds, 2D)
- Quad4 (bilinéaire — 4 nœuds, 2D)
- Tri6  (LST — 6 nœuds, 2D)
- Tetra4 (linéaire — 4 nœuds, 3D)
- Hexa8  (trilinéaire — 8 nœuds, 3D)
- Tetra10 (quadratique — 10 nœuds, 3D)

Tailles de maillage choisies pour obtenir ~5 000–10 000 éléments par type.

Usage ::

    .venv/bin/python3 benchmarks/bench_assembler_all.py

Sortie typique ::

    ============================================================
    Benchmark assemblage  —  6 types d'éléments
    ============================================================

    [Tri3]   50×50 tri  →  5000 éléments, 2652 nœuds, 5304 DDL
    -- K ---   Scalaire :  0.32 s  |  Batch :  15.2 ms  →  21.0×
    -- M ---   Scalaire :  0.25 s  |  Batch :   6.1 ms  →  41.0×

    [Quad4]  100×50 quad  →  5000 éléments, 5151 nœuds, 10302 DDL
    -- K ---   Scalaire :  1.22 s  |  Batch :  55.3 ms  →  22.1×
    -- M ---   Scalaire :  0.98 s  |  Batch :  22.4 ms  →  43.8×
    ...
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from femsolver.core.assembler import Assembler
from femsolver.core.material import ElasticMaterial
from femsolver.core.mesh import ElementData, Mesh
from femsolver.elements.hexa8 import Hexa8
from femsolver.elements.quad4 import Quad4
from femsolver.elements.tetra4 import Tetra4
from femsolver.elements.tetra10 import Tetra10
from femsolver.elements.tri3 import Tri3
from femsolver.elements.tri6 import Tri6


# ---------------------------------------------------------------------------
# Constructeurs de maillages réguliers
# ---------------------------------------------------------------------------


def build_tri3_mesh(nx: int, ny: int) -> Mesh:
    """Grille régulière de triangles Tri3 (2 par carré → 2*nx*ny éléments)."""
    x = np.linspace(0.0, 1.0, nx + 1)
    y = np.linspace(0.0, 1.0, ny + 1)
    xx, yy = np.meshgrid(x, y)
    nodes = np.column_stack([xx.ravel(), yy.ravel()])

    steel = ElasticMaterial(E=210e9, nu=0.3, rho=7800)
    props = {"thickness": 0.01, "formulation": "plane_stress"}
    elements = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            n1 = n0 + 1
            n2 = n0 + (nx + 1)
            n3 = n2 + 1
            elements.append(ElementData(Tri3, (n0, n1, n3), steel, props))
            elements.append(ElementData(Tri3, (n0, n3, n2), steel, props))
    return Mesh(nodes=nodes, elements=tuple(elements), n_dim=2)


def build_quad4_mesh(nx: int, ny: int) -> Mesh:
    """Grille régulière de Quad4 (nx*ny éléments)."""
    x = np.linspace(0.0, 1.0, nx + 1)
    y = np.linspace(0.0, 1.0, ny + 1)
    xx, yy = np.meshgrid(x, y)
    nodes = np.column_stack([xx.ravel(), yy.ravel()])

    steel = ElasticMaterial(E=210e9, nu=0.3, rho=7800)
    props = {"thickness": 0.01, "formulation": "plane_stress"}
    elements = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            n1 = n0 + 1
            n2 = n0 + (nx + 1) + 1
            n3 = n0 + (nx + 1)
            elements.append(ElementData(Quad4, (n0, n1, n2, n3), steel, props))
    return Mesh(nodes=nodes, elements=tuple(elements), n_dim=2)


def build_tri6_mesh(nx: int, ny: int) -> Mesh:
    """Grille de Tri6 (2 par carré) avec nœuds milieux générés automatiquement."""
    # Grille fine : sommets + milieux d'arêtes
    # On utilise un maillage structuré : chaque carré → 2 Tri6
    # Nœuds sur une grille 2× plus fine pour les milieux
    fx, fy = 2 * nx + 1, 2 * ny + 1
    x = np.linspace(0.0, 1.0, fx)
    y = np.linspace(0.0, 1.0, fy)
    xx, yy = np.meshgrid(x, y)
    nodes = np.column_stack([xx.ravel(), yy.ravel()])

    def idx(i, j):
        return j * fx + i

    steel = ElasticMaterial(E=210e9, nu=0.3, rho=7800)
    props = {"thickness": 0.01, "formulation": "plane_stress"}
    elements = []
    for j in range(ny):
        for i in range(nx):
            # Coins du carré
            c00 = idx(2 * i,     2 * j)
            c20 = idx(2 * i + 2, 2 * j)
            c02 = idx(2 * i,     2 * j + 2)
            c22 = idx(2 * i + 2, 2 * j + 2)
            # Milieux
            m10 = idx(2 * i + 1, 2 * j)
            m01 = idx(2 * i,     2 * j + 1)
            m11 = idx(2 * i + 1, 2 * j + 1)
            m21 = idx(2 * i + 2, 2 * j + 1)
            m12 = idx(2 * i + 1, 2 * j + 2)
            # Triangle inférieur : c00, c20, c22 ; milieux : m10, m21, m11
            elements.append(ElementData(Tri6, (c00, c20, c22, m10, m21, m11), steel, props))
            # Triangle supérieur : c00, c22, c02 ; milieux : m11, m12, m01
            elements.append(ElementData(Tri6, (c00, c22, c02, m11, m12, m01), steel, props))
    return Mesh(nodes=nodes, elements=tuple(elements), n_dim=2)


def build_tetra4_mesh(nx: int, ny: int, nz: int) -> Mesh:
    """Grille 3D de Tetra4 (6 tétraèdres par voxel)."""
    x = np.linspace(0.0, 1.0, nx + 1)
    y = np.linspace(0.0, 1.0, ny + 1)
    z = np.linspace(0.0, 1.0, nz + 1)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    nodes = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    def idx(i, j, k):
        return i * (ny + 1) * (nz + 1) + j * (nz + 1) + k

    steel = ElasticMaterial(E=210e9, nu=0.3, rho=7800)
    elements = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # 8 coins du voxel
                v = [
                    idx(i,   j,   k),   idx(i+1, j,   k),
                    idx(i,   j+1, k),   idx(i+1, j+1, k),
                    idx(i,   j,   k+1), idx(i+1, j,   k+1),
                    idx(i,   j+1, k+1), idx(i+1, j+1, k+1),
                ]
                # Décomposition en 6 tétraèdres (schéma standard)
                tets = [
                    (v[0], v[1], v[3], v[7]),
                    (v[0], v[3], v[2], v[7]),
                    (v[0], v[2], v[6], v[7]),
                    (v[0], v[6], v[4], v[7]),
                    (v[0], v[4], v[5], v[7]),
                    (v[0], v[5], v[1], v[7]),
                ]
                for tet in tets:
                    elements.append(ElementData(Tetra4, tet, steel, {}))
    return Mesh(nodes=nodes, elements=tuple(elements), n_dim=3)


def build_hexa8_mesh(nx: int, ny: int, nz: int) -> Mesh:
    """Grille 3D de Hexa8 (nx*ny*nz éléments)."""
    x = np.linspace(0.0, 1.0, nx + 1)
    y = np.linspace(0.0, 1.0, ny + 1)
    z = np.linspace(0.0, 1.0, nz + 1)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    nodes = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    def idx(i, j, k):
        return i * (ny + 1) * (nz + 1) + j * (nz + 1) + k

    steel = ElasticMaterial(E=210e9, nu=0.3, rho=7800)
    elements = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                n = [
                    idx(i,   j,   k),   idx(i+1, j,   k),
                    idx(i+1, j+1, k),   idx(i,   j+1, k),
                    idx(i,   j,   k+1), idx(i+1, j,   k+1),
                    idx(i+1, j+1, k+1), idx(i,   j+1, k+1),
                ]
                elements.append(ElementData(Hexa8, tuple(n), steel, {}))
    return Mesh(nodes=nodes, elements=tuple(elements), n_dim=3)


def build_tetra10_mesh(nx: int, ny: int, nz: int) -> Mesh:
    """Grille 3D de Tetra10 (6 par voxel, nœuds milieux générés)."""
    # On part d'une grille 2× plus fine pour avoir les milieux d'arêtes
    fx, fy, fz = 2 * nx + 1, 2 * ny + 1, 2 * nz + 1
    x = np.linspace(0.0, 1.0, fx)
    y = np.linspace(0.0, 1.0, fy)
    z = np.linspace(0.0, 1.0, fz)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    nodes = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    def idx(i, j, k):
        return i * fy * fz + j * fz + k

    steel = ElasticMaterial(E=210e9, nu=0.3, rho=7800)
    elements = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # Coins à pas double
                ci, cj, ck = 2 * i, 2 * j, 2 * k
                v = [
                    idx(ci,   cj,   ck),    idx(ci+2, cj,   ck),
                    idx(ci,   cj+2, ck),    idx(ci+2, cj+2, ck),
                    idx(ci,   cj,   ck+2),  idx(ci+2, cj,   ck+2),
                    idx(ci,   cj+2, ck+2),  idx(ci+2, cj+2, ck+2),
                ]
                # 6 tétraèdres par voxel (mêmes connectivités coins que Tetra4)
                tet_corners = [
                    (v[0], v[1], v[3], v[7]),
                    (v[0], v[3], v[2], v[7]),
                    (v[0], v[2], v[6], v[7]),
                    (v[0], v[6], v[4], v[7]),
                    (v[0], v[4], v[5], v[7]),
                    (v[0], v[5], v[1], v[7]),
                ]
                for c0, c1, c2, c3 in tet_corners:
                    p = np.array([c0, c1, c2, c3])
                    # Retrouver indices grille fine (i,j,k) pour chaque coin
                    def grid_ijk(flat_idx):
                        ii = flat_idx // (fy * fz)
                        rem = flat_idx % (fy * fz)
                        jj = rem // fz
                        kk = rem % fz
                        return ii, jj, kk

                    def mid_idx(a, b):
                        ia, ja, ka = grid_ijk(a)
                        ib, jb, kb = grid_ijk(b)
                        return idx((ia+ib)//2, (ja+jb)//2, (ka+kb)//2)

                    m01 = mid_idx(c0, c1)
                    m12 = mid_idx(c1, c2)
                    m02 = mid_idx(c0, c2)
                    m03 = mid_idx(c0, c3)
                    m13 = mid_idx(c1, c3)
                    m23 = mid_idx(c2, c3)
                    node_ids = (c0, c1, c2, c3, m01, m12, m02, m03, m13, m23)
                    elements.append(ElementData(Tetra10, node_ids, steel, {}))
    return Mesh(nodes=nodes, elements=tuple(elements), n_dim=3)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _fmt(t: float) -> str:
    if t >= 1.0:
        return f"{t:.2f} s"
    if t >= 0.1:
        return f"{t * 1e3:.0f} ms"
    return f"{t * 1e3:.1f} ms"


def run_one(label: str, mesh: Mesh, check: bool = True) -> None:
    assembler = Assembler(mesh)
    n_e = len(mesh.elements)
    n_n = mesh.n_nodes
    n_d = mesh.n_dof
    print(f"\n[{label}]  {n_e} elements, {n_n} noeuds, {n_d} DDL")

    # --- K ---
    t0 = time.perf_counter()
    K_sc = assembler.assemble_stiffness(use_batch=False)
    t_sc_k = time.perf_counter() - t0

    t0 = time.perf_counter()
    K_bt = assembler.assemble_stiffness(use_batch=True)
    t_bt_k = time.perf_counter() - t0

    spd_k = t_sc_k / t_bt_k if t_bt_k > 0 else float('inf')

    if check:
        diff_k = abs(K_bt - K_sc).max()
        rel_k = diff_k / (abs(K_sc).max() or 1.0)
        ok_k = "OK" if rel_k < 1e-10 else f"** ECART {rel_k:.1e} **"
    else:
        ok_k = "(skip)"

    print(f"  K  Scalaire : {_fmt(t_sc_k)}  |  Batch : {_fmt(t_bt_k)}  "
          f"->  {spd_k:5.1f}x   [{ok_k}]")

    # --- M ---
    t0 = time.perf_counter()
    M_sc = assembler.assemble_mass(use_batch=False)
    t_sc_m = time.perf_counter() - t0

    t0 = time.perf_counter()
    M_bt = assembler.assemble_mass(use_batch=True)
    t_bt_m = time.perf_counter() - t0

    spd_m = t_sc_m / t_bt_m if t_bt_m > 0 else float('inf')

    if check:
        diff_m = abs(M_bt - M_sc).max()
        rel_m = diff_m / (abs(M_sc).max() or 1.0)
        ok_m = "OK" if rel_m < 1e-10 else f"** ECART {rel_m:.1e} **"
    else:
        ok_m = "(skip)"

    print(f"  M  Scalaire : {_fmt(t_sc_m)}  |  Batch : {_fmt(t_bt_m)}  "
          f"->  {spd_m:5.1f}x   [{ok_m}]")


def run_all() -> None:
    print("=" * 62)
    print("Benchmark assemblage -- 6 types d'elements")
    print("=" * 62)

    # 2D -- ~5000 elements chacun
    run_one("Tri3   50x50",   build_tri3_mesh(50, 50))        # 5 000 elements
    run_one("Quad4  100x50",  build_quad4_mesh(100, 50))      # 5 000 elements
    run_one("Tri6   35x35",   build_tri6_mesh(35, 35))        # 2 450 elements

    # 3D -- ~2000-5000 elements chacun (3D couts plus cher)
    run_one("Tetra4  10x10x10", build_tetra4_mesh(10, 10, 10))   # 6 000 elements
    run_one("Hexa8   20x20x10", build_hexa8_mesh(20, 20, 10))    # 4 000 elements
    run_one("Tetra10  7x7x7",   build_tetra10_mesh(7, 7, 7))     # 2 058 elements

    print()
    print("=" * 62)


if __name__ == "__main__":
    run_all()

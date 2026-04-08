"""Exemple : treillis Warren 2D à 6 barres (pont).

Géométrie
---------
     2 --------- 3
    /|           |\\
   / |           | \\
  /  |           |  \\
 0 --+-- 1 ------+-- 4  ...  (à compléter selon le span)

Treillis Warren simplifié à 5 nœuds et 7 barres :

     2       3
    /  \\   /  \\
   /    \\ /    \\
  0------1------4

Nœuds
-----
    0 : (0,  0)  — appui encastrement gauche (ux=uy=0)
    1 : (2,  0)  — nœud bas milieu
    2 : (1,  1)  — nœud haut gauche
    3 : (3,  1)  — nœud haut droit
    4 : (4,  0)  — appui rouleau droit (uy=0)

Barres
------
    0-1, 1-4        (membrure inférieure)
    0-2, 2-3, 3-4   (membrure supérieure)
    1-2, 1-3        (montants diagonaux)

Chargement
----------
Force de 50 kN vers le bas appliquée au nœud central 1 et au nœud 2.
"""

from __future__ import annotations

import numpy as np

import logging

from femsolver.core.assembler import Assembler
from femsolver.core.boundary import apply_dirichlet
from femsolver.core.diagnostics import run_diagnostics
from femsolver.core.material import ElasticMaterial
from femsolver.core.mesh import BoundaryConditions, ElementData, Mesh
from femsolver.core.solver import StaticSolver
from femsolver.elements.bar2d import Bar2D
from femsolver.postprocess.plotter2d import plot_truss

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main() -> None:
    # ------------------------------------------------------------------
    # Matériau et section
    # ------------------------------------------------------------------
    steel = ElasticMaterial(E=210e9, nu=0.3, rho=7800)
    area = 5e-4  # m² (50 cm²)
    props = {"area": area}

    # ------------------------------------------------------------------
    # Nœuds
    # ------------------------------------------------------------------
    nodes = np.array([
        [0.0, 0.0],  # 0
        [2.0, 0.0],  # 1
        [1.0, 1.0],  # 2
        [3.0, 1.0],  # 3
        [4.0, 0.0],  # 4
    ])

    # ------------------------------------------------------------------
    # Connectivité
    # ------------------------------------------------------------------
    connectivity = [
        (0, 1),  # membrure inf. gauche
        (1, 4),  # membrure inf. droite
        (0, 2),  # diagonale gauche
        (2, 3),  # membrure sup.
        (3, 4),  # diagonale droite
        (1, 2),  # montant gauche
        (1, 3),  # montant droit
    ]
    elements = tuple(
        ElementData(Bar2D, conn, steel, props) for conn in connectivity
    )

    mesh = Mesh(nodes=nodes, elements=elements, n_dim=2)

    # ------------------------------------------------------------------
    # Conditions aux limites
    # ------------------------------------------------------------------
    F_ext = -50_000.0  # N (50 kN vers le bas)
    bc = BoundaryConditions(
        dirichlet={
            0: {0: 0.0, 1: 0.0},  # encastrement gauche
            4: {1: 0.0},           # rouleau droit (bloque uy)
        },
        neumann={
            1: {1: F_ext},  # charge au nœud bas milieu
            2: {1: F_ext},  # charge au nœud haut gauche
        },
    )

    # ------------------------------------------------------------------
    # Assemblage → résolution
    # ------------------------------------------------------------------
    assembler = Assembler(mesh)
    K = assembler.assemble_stiffness()
    M = assembler.assemble_mass()
    F_vec = assembler.assemble_forces(bc)
    K_bc, F_bc = apply_dirichlet(K, F_vec, mesh, bc)
    u = StaticSolver().solve(K_bc, F_bc)

    # ------------------------------------------------------------------
    # Post-traitement : efforts normaux
    # ------------------------------------------------------------------
    bar = Bar2D()
    axial_forces = []
    for elem_data in mesh.elements:
        nids = list(elem_data.node_ids)
        node_coords = mesh.node_coords(elem_data.node_ids)
        dofs = mesh.global_dofs(elem_data.node_ids)
        u_e = u[dofs]
        N = bar.axial_force(steel, node_coords, area, u_e)
        axial_forces.append(N)

    # ------------------------------------------------------------------
    # Affichage des résultats
    # ------------------------------------------------------------------
    print("=" * 55)
    print("Treillis Warren 2D — Résultats")
    print("=" * 55)
    print(f"\nDéplacements nodaux [mm] :")
    for i in range(mesh.n_nodes):
        ux = u[2 * i] * 1e3
        uy = u[2 * i + 1] * 1e3
        print(f"  Nœud {i} : ux = {ux:+.4f} mm,  uy = {uy:+.4f} mm")

    print(f"\nEfforts normaux dans les barres [kN] :")
    bar_names = [
        "0-1 (mem. inf. G)",
        "1-4 (mem. inf. D)",
        "0-2 (diag. G)",
        "2-3 (mem. sup.)",
        "3-4 (diag. D)",
        "1-2 (montant G)",
        "1-3 (montant D)",
    ]
    for name, N in zip(bar_names, axial_forces):
        status = "TRACTION" if N > 0 else "COMPRESSION"
        print(f"  {name:<22} : {N/1e3:+8.2f} kN  ({status})")

    # ------------------------------------------------------------------
    # Diagnostics : masse, réactions, bilan d'équilibre
    # ------------------------------------------------------------------
    run_diagnostics(mesh, K, u, F_vec, bc, M=M)

    # ------------------------------------------------------------------
    # Visualisation (amplification × 1000 pour voir la déformée)
    # ------------------------------------------------------------------
    plot_truss(mesh, u=u, axial_forces=axial_forces, nodal_forces=bc.neumann, show=True)


if __name__ == "__main__":
    main()

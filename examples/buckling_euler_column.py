"""Flambage linéaire d'une colonne d'Euler — charge critique de Timoshenko.

Ce script illustre l'analyse de flambage complète :

1. Construction du maillage (colonne Beam2D le long de l'axe y)
2. Analyse statique sous charge de référence P_ref = 1 N
3. Assemblage de la matrice de rigidité géométrique K_g
4. Résolution du problème aux valeurs propres (K + λ K_g)φ = 0
5. Comparaison avec la formule analytique d'Euler

Résultats attendus
------------------
Colonne pince-pincée :   P_cr = π²EI/L²    ≈ 17 272 N  (acier 10×10 mm, L=1 m)
Colonne encastrée-libre : P_cr = π²EI/(4L²) ≈  4 318 N

Usage
-----
    .venv/bin/python3 examples/buckling_euler_column.py
"""

from __future__ import annotations

import numpy as np

from femsolver.core.assembler import Assembler
from femsolver.core.boundary import apply_dirichlet
from femsolver.core.material import ElasticMaterial
from femsolver.core.mesh import BoundaryConditions, ElementData, Mesh
from femsolver.core.solver import BucklingSolver, StaticSolver
from femsolver.elements.beam2d import Beam2D


def build_pinpin_column(
    E: float, A: float, I: float, L: float, n_elem: int, mat: ElasticMaterial
) -> tuple[Mesh, BoundaryConditions]:
    """Colonne verticale pince-pincée sous compression unitaire.

    Orientation : axe y (vertical).
    Chargement  : -1 N en y au nœud supérieur (compression).
    """
    L_e = L / n_elem
    nodes = np.array([[0.0, i * L_e] for i in range(n_elem + 1)])
    props = {"area": A, "inertia": I}
    elements = tuple(
        ElementData(Beam2D, (i, i + 1), mat, props) for i in range(n_elem)
    )
    mesh = Mesh(nodes=nodes, elements=elements, n_dim=2, dof_per_node=3)

    bc = BoundaryConditions(
        dirichlet={
            0: {0: 0.0, 1: 0.0},    # nœud 0 (bas) : ux=0, uy=0
            n_elem: {0: 0.0},         # nœud top   : ux=0 (rouleau)
        },
        neumann={
            n_elem: {1: -1.0},        # 1 N de compression en -y
        },
    )
    return mesh, bc


def build_cantilever_column(
    E: float, A: float, I: float, L: float, n_elem: int, mat: ElasticMaterial
) -> tuple[Mesh, BoundaryConditions]:
    """Colonne verticale encastrée à la base, libre au sommet."""
    L_e = L / n_elem
    nodes = np.array([[0.0, i * L_e] for i in range(n_elem + 1)])
    props = {"area": A, "inertia": I}
    elements = tuple(
        ElementData(Beam2D, (i, i + 1), mat, props) for i in range(n_elem)
    )
    mesh = Mesh(nodes=nodes, elements=elements, n_dim=2, dof_per_node=3)

    bc = BoundaryConditions(
        dirichlet={
            0: {0: 0.0, 1: 0.0, 2: 0.0},   # encastrement complet
        },
        neumann={
            n_elem: {1: -1.0},
        },
    )
    return mesh, bc


def run_buckling(
    mesh: Mesh, bc: BoundaryConditions, n_modes: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """Exécute l'analyse de flambage et retourne (lambda_cr, phi)."""
    assembler = Assembler(mesh)
    K = assembler.assemble_stiffness()
    F = assembler.assemble_forces(bc)

    ds = apply_dirichlet(K, F, mesh, bc)
    u = StaticSolver().solve(*ds)

    K_g = assembler.assemble_geometric_stiffness(u)
    K_g_free = ds.reduce(K_g)

    lambda_cr, phi_free = BucklingSolver().solve(ds.K_free, K_g_free, n_modes=n_modes)
    phi = ds.recover_modes(phi_free)
    return lambda_cr, phi


def main() -> None:
    # -----------------------------------------------------------------------
    # Paramètres géométriques et matériau
    # -----------------------------------------------------------------------
    E = 210e9   # Pa — acier
    nu = 0.3
    rho = 7800.0
    b = h = 0.01   # m — section 10 × 10 mm
    A = b * h
    I = b * h**3 / 12
    L = 1.0     # m

    mat = ElasticMaterial(E=E, nu=nu, rho=rho)

    n_elem = 20  # éléments (convergence atteinte)

    # -----------------------------------------------------------------------
    # Formules analytiques d'Euler
    # -----------------------------------------------------------------------
    P_cr_pinpin     = np.pi**2 * E * I / L**2
    P_cr_cantilever = np.pi**2 * E * I / (4.0 * L**2)

    print("=" * 60)
    print("  ANALYSE DE FLAMBAGE LINÉAIRE — COLONNE D'EULER")
    print("=" * 60)
    print(f"\n  Matériau : E = {E/1e9:.0f} GPa, ν = {nu}")
    print(f"  Section  : {b*1e3:.0f} × {h*1e3:.0f} mm² (A = {A*1e4:.2f} cm²)")
    print(f"  Inertie  : I = {I*1e9:.2f} mm⁴")
    print(f"  Longueur : L = {L:.1f} m")
    print(f"  Maillage : {n_elem} éléments Beam2D")

    # -----------------------------------------------------------------------
    # Cas 1 : colonne pince-pincée
    # -----------------------------------------------------------------------
    mesh_pp, bc_pp = build_pinpin_column(E, A, I, L, n_elem, mat)
    lambda_pp, phi_pp = run_buckling(mesh_pp, bc_pp, n_modes=3)

    print("\n" + "-" * 60)
    print("  Cas 1 : colonne PINCE-PINCÉE")
    print(f"  Analytique : P_cr = π²EI/L²  = {P_cr_pinpin/1e3:.2f} kN")
    print(f"\n  {'Mode':>4}  {'P_cr FEM (kN)':>14}  {'P_cr analytique (kN)':>20}  {'Erreur':>8}")
    print(f"  {'-'*4}  {'-'*14}  {'-'*20}  {'-'*8}")
    for n, lam in enumerate(lambda_pp, start=1):
        p_fem  = float(lam) * 1e-3       # kN (P_ref = 1 N → P_cr = lambda)
        p_ana  = n**2 * P_cr_pinpin * 1e-3
        err    = abs(p_fem - p_ana) / p_ana * 100
        print(f"  {n:>4}  {p_fem:>14.3f}  {p_ana:>20.3f}  {err:>7.3f}%")

    # -----------------------------------------------------------------------
    # Cas 2 : colonne encastrée-libre
    # -----------------------------------------------------------------------
    mesh_cl, bc_cl = build_cantilever_column(E, A, I, L, n_elem, mat)
    lambda_cl, phi_cl = run_buckling(mesh_cl, bc_cl, n_modes=3)

    print("\n" + "-" * 60)
    print("  Cas 2 : colonne ENCASTRÉE-LIBRE (console)")
    print(f"  Analytique : P_cr = π²EI/(4L²) = {P_cr_cantilever/1e3:.2f} kN")
    print(f"\n  {'Mode':>4}  {'P_cr FEM (kN)':>14}  {'L_eff / L analytique':>21}")
    print(f"  {'-'*4}  {'-'*14}  {'-'*21}")
    # Modes k=(1,3,5,...) pour la console : P_n = (2n-1)²π²EI/(4L²)
    for n, lam in enumerate(lambda_cl, start=1):
        p_fem = float(lam) * 1e-3
        k = 2 * n - 1
        p_ana = k**2 * P_cr_cantilever * 1e-3
        err   = abs(p_fem - p_ana) / p_ana * 100
        print(f"  {n:>4}  {p_fem:>14.3f}  {'('+str(k)+'ème mode encastré)':>21}  →  {p_ana:.3f} kN  err={err:.3f}%")

    print("\n  ✓ Analyse de flambage terminée.")
    print("=" * 60)


if __name__ == "__main__":
    main()

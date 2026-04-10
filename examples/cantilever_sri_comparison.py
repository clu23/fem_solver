"""Comparaison intégration complète vs intégration réduite sélective (SRI).

Poutre console en flexion — Quad4, maillage grossier
=====================================================

Ce script montre quantitativement le phénomène de shear locking sur un Quad4
et comment la SRI (Selective Reduced Integration) l'élimine.

Géométrie
---------
    ┌─────────────────────────────────────────┐  ← y = H
    │  [0]   [1]   [2]   [3]   [4]           │
    │ elem0  elem1  elem2  elem3              │ ← P↓ (extrémité libre)
    │                                          │
    └─────────────────────────────────────────┘  ← y = 0
    x=0                                  x=L
    (encastrement)

    L = 1.0 m, H = 0.1 m, t = 0.01 m
    Acier : E = 210 GPa, ν = 0.3
    Charge : P = 1000 N vers -y à x=L

Solution analytique (Euler-Bernoulli)
--------------------------------------
    I = t·H³/12        moment quadratique
    δ = PL³/(3EI)      flèche à l'extrémité libre

Phénomène de shear locking
---------------------------
Le Quad4 bilinéaire avec intégration 2×2 complète génère des *contraintes
de cisaillement parasites* (γxy ≠ 0 de manière artificielle) en flexion.
Cela rend l'élément trop rigide : il "refuse de plier" correctement.

Principe de la SRI
------------------
On décompose la matrice de comportement D en deux parties :

    D = D_dil + D_dev

    D_dil (εxx, εyy) : intégrée en 2×2 Gauss → réponse membranaire complète
    D_dev (γxy = G)  : intégrée en 1 point (ξ=η=0) → élimine le cisaillement parasite

Pourquoi 1 point pour le cisaillement ?
   En flexion pure d'un élément rectangulaire, γxy = 0 au centre par symétrie.
   Évaluer G·γxy en ce seul point donne une contribution nulle → pas de
   rigidité parasite → l'élément peut se plier librement.

Pourquoi 2×2 pour la partie dilatation ?
   Les termes D_dil détectent les modes de déformation membranaires (εxx, εyy).
   Si on les sous-intégrait aussi, des modes zéro-énergie (hourglass) apparaîtraient.
   La 2×2 assure le rang complet de K_e et bloque ces modes parasites.

Usage
-----
    python examples/cantilever_sri_comparison.py
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import lil_matrix

from femsolver.core.assembler import Assembler
from femsolver.core.boundary import apply_dirichlet
from femsolver.core.material import ElasticMaterial
from femsolver.core.mesh import BoundaryConditions, ElementData, Mesh
from femsolver.core.solver import StaticSolver
from femsolver.elements.quad4 import Quad4


# ---------------------------------------------------------------------------
# Paramètres du problème
# ---------------------------------------------------------------------------

E   = 210e9    # Module d'Young acier [Pa]
NU  = 0.3      # Coefficient de Poisson
RHO = 7800.0   # Densité [kg/m³]
L   = 1.0      # Longueur [m]
H   = 0.1      # Hauteur (hauteur de la section) [m]
T   = 0.01     # Épaisseur (plan stress) [m]
P   = 1000.0   # Charge à l'extrémité [N]


def analytical_tip_deflection() -> float:
    """Flèche analytique Euler-Bernoulli : δ = PL³/(3EI)."""
    I = T * H**3 / 12.0
    return P * L**3 / (3.0 * E * I)


def build_and_solve(n_x: int, integration: str) -> dict:
    """Construit un maillage n_x × 1 Quad4 et résout.

    Parameters
    ----------
    n_x : int
        Nombre d'éléments en longueur.
    integration : str
        ``"full"`` ou ``"sri"``.

    Returns
    -------
    dict avec les clés :
        - ``v_tip`` : déplacement vertical (< 0) au nœud de l'extrémité libre
        - ``n_dof`` : nombre total de DDL
        - ``n_elem`` : nombre d'éléments
    """
    mat = ElasticMaterial(E=E, nu=NU, rho=RHO)
    props = {"thickness": T, "formulation": "plane_stress"}
    n_per_row = n_x + 1

    # Nœuds : rangée basse (y=0) puis rangée haute (y=H)
    xs = np.linspace(0.0, L, n_per_row)
    nodes = np.array([[x, y] for y in [0.0, H] for x in xs])

    # Connectivité (sens trigonométrique)
    elements = []
    for i in range(n_x):
        n0, n1 = i, i + 1
        n2, n3 = n_per_row + i + 1, n_per_row + i
        elements.append(ElementData(Quad4, (n0, n1, n2, n3), mat, props))

    mesh = Mesh(nodes=nodes, elements=tuple(elements), n_dim=2)

    # Conditions aux limites
    dirichlet = {
        0:         {0: 0.0, 1: 0.0},   # nœud bas-gauche (y=0, x=0)
        n_per_row: {0: 0.0, 1: 0.0},   # nœud haut-gauche (y=H, x=0)
    }
    neumann = {
        n_x:             {1: -P / 2.0},   # nœud bas-droit (y=0, x=L)
        n_per_row + n_x: {1: -P / 2.0},   # nœud haut-droit (y=H, x=L)
    }
    bc = BoundaryConditions(dirichlet=dirichlet, neumann=neumann)

    # Assemblage selon la méthode d'intégration
    n_dof = mesh.n_dof
    K_glob = lil_matrix((n_dof, n_dof))
    elem_instance = Quad4()

    for elem_data in mesh.elements:
        nc = mesh.node_coords(elem_data.node_ids)
        if integration == "sri":
            K_e = elem_instance.stiffness_matrix_sri(mat, nc, props)
        else:
            K_e = elem_instance.stiffness_matrix(mat, nc, props)
        dofs = list(mesh.global_dofs(elem_data.node_ids))
        for ii, di in enumerate(dofs):
            for jj, dj in enumerate(dofs):
                K_glob[di, dj] += K_e[ii, jj]

    K_csr = K_glob.tocsr()
    F = Assembler(mesh).assemble_forces(bc)
    K_bc, F_bc = apply_dirichlet(K_csr, F, mesh, bc)
    u = StaticSolver().solve(K_bc, F_bc)

    v_tip = float(u[2 * n_x + 1])   # uy du nœud extrême bas
    return {"v_tip": v_tip, "n_dof": n_dof, "n_elem": n_x}


def print_table(results: list[dict]) -> None:
    """Affiche un tableau formaté des résultats."""
    delta = analytical_tip_deflection()

    header = (
        f"\n{'Maillage':>10}  {'n_dof':>6}  "
        f"{'Full 2×2 [mm]':>14}  {'SRI [mm]':>12}  "
        f"{'Analytique [mm]':>15}  "
        f"{'Err Full':>9}  {'Err SRI':>8}"
    )
    print(header)
    print("─" * len(header))

    for r in results:
        nx = r["n_elem"]
        n_dof = r["n_dof"]
        v_full_mm = abs(r["v_full"]) * 1e3
        v_sri_mm  = abs(r["v_sri"])  * 1e3
        delta_mm  = delta * 1e3
        err_full = (abs(r["v_full"]) - delta) / delta * 100
        err_sri  = (abs(r["v_sri"])  - delta) / delta * 100

        mesh_label = f"{nx}×1"
        print(
            f"{mesh_label:>10}  {n_dof:>6d}  "
            f"{v_full_mm:>14.4f}  {v_sri_mm:>12.4f}  "
            f"{delta_mm:>15.4f}  "
            f"{err_full:>+9.1f}%  {err_sri:>+8.1f}%"
        )


def main() -> None:
    """Lance la comparaison sur plusieurs maillages et affiche les résultats."""
    print("=" * 70)
    print("  Poutre console — Shear locking Quad4 : Full vs SRI")
    print("=" * 70)
    print(f"  Géométrie : L={L} m, H={H} m, t={T} m")
    print(f"  Matériau  : E={E/1e9:.0f} GPa, ν={NU}")
    print(f"  Charge    : P={P:.0f} N (sens -y à x=L)")
    print(f"  Solution  : δ = PL³/(3EI) = {analytical_tip_deflection()*1e3:.4f} mm")
    print()

    meshes = [1, 2, 4, 8, 16]
    results = []
    for nx in meshes:
        r_full = build_and_solve(nx, "full")
        r_sri  = build_and_solve(nx, "sri")
        results.append({
            "n_elem": nx,
            "n_dof":  r_full["n_dof"],
            "v_full": r_full["v_tip"],
            "v_sri":  r_sri["v_tip"],
        })

    print_table(results)

    # Résumé du gain
    delta = analytical_tip_deflection()
    r4 = next(r for r in results if r["n_elem"] == 4)
    gain = (abs(r4["v_sri"]) - abs(r4["v_full"])) / abs(r4["v_full"]) * 100

    print()
    print("─" * 70)
    print(f"  Maillage 4×1 (grossier) :")
    print(f"    Full 2×2 : {abs(r4['v_full'])*1e3:.4f} mm  "
          f"(ratio = {abs(r4['v_full'])/delta:.3f} × analytique → LOCKING)")
    print(f"    SRI      : {abs(r4['v_sri'])*1e3:.4f} mm  "
          f"(ratio = {abs(r4['v_sri'])/delta:.3f} × analytique → bien corrigé)")
    print(f"    Gain SRI : +{gain:.1f}% par rapport à Full")
    print()
    print("  Interprétation")
    print("  ──────────────")
    print("  • Full 2×2 : les points de Gauss hors-centre 'voient' un γxy ≠ 0")
    print("    même en flexion pure → rigidité parasite → sous-estimation de δ")
    print()
    print("  • SRI : γxy évalué seulement en (ξ=0,η=0) où il est nul par")
    print("    symétrie pour un rectangle en flexion pure → pas de raideur")
    print("    parasite → flèche correcte dès un maillage grossier")
    print()
    print("  • Pas de hourglass : D_dil intégré en 2×2 assure rang(K_e) = 5")
    print("    (= 8 DDL − 3 modes rigides), identique à l'intégration complète")
    print("=" * 70)


if __name__ == "__main__":
    main()

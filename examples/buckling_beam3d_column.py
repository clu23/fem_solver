"""Flambage linéaire 3D — colonne Beam3D et portique.

Illustre deux configurations :

1. **Colonne isolée 3D** (pince-pincée et encastrée-libre)
   Comparaison avec la formule d'Euler : P_cr = π²EI/L²
   Section non-carrée (20 × 10 mm) → deux modes distincts (axe faible / axe fort).

2. **Portique 3D** (deux colonnes, une traverse)
   Analyse de flambage du mode de déversement latéral sous charge verticale.
   Résultats comparés à la formule approchée du portique en portique de Lewis.

Usage
-----
    .venv/bin/python3 examples/buckling_beam3d_column.py
"""

from __future__ import annotations

import numpy as np

from femsolver.core.assembler import Assembler
from femsolver.core.boundary import apply_dirichlet
from femsolver.core.material import ElasticMaterial
from femsolver.core.mesh import BoundaryConditions, ElementData, Mesh
from femsolver.core.sections import RectangularSection
from femsolver.core.solver import BucklingSolver, StaticSolver
from femsolver.elements.beam3d import Beam3D


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_buckling(mesh: Mesh, bc: BoundaryConditions, n_modes: int = 3):
    """Workflow complet : static → K_g → eigsh → modes de flambage."""
    assembler = Assembler(mesh)
    K = assembler.assemble_stiffness()
    F = assembler.assemble_forces(bc)
    ds = apply_dirichlet(K, F, mesh, bc)
    u  = StaticSolver().solve(*ds)
    K_g      = assembler.assemble_geometric_stiffness(u)
    K_g_free = ds.reduce(K_g)
    lambda_cr, phi_free = BucklingSolver().solve(ds.K_free, K_g_free, n_modes=n_modes)
    phi = ds.recover_modes(phi_free)
    return lambda_cr, phi


def build_column_z(
    mat: ElasticMaterial,
    sec: RectangularSection,
    L: float,
    n_elem: int,
    bc_type: str = "pinpin",
) -> tuple[Mesh, BoundaryConditions]:
    """Colonne verticale (axe z) avec conditions aux limites choisies.

    Parameters
    ----------
    bc_type : "pinpin" | "cantilever"
    """
    Le = L / n_elem
    nodes = np.array([[0.0, 0.0, i * Le] for i in range(n_elem + 1)])
    props = {"section": sec}
    elements = tuple(
        ElementData(Beam3D, (i, i + 1), mat, props) for i in range(n_elem)
    )
    mesh = Mesh(nodes=nodes, elements=elements, n_dim=3, dof_per_node=6)

    if bc_type == "pinpin":
        bc = BoundaryConditions(
            dirichlet={
                0:      {0: 0.0, 1: 0.0, 2: 0.0},   # pied : appui 3D
                n_elem: {0: 0.0, 1: 0.0},             # sommet : rouleau
            },
            neumann={n_elem: {2: -1.0}},
        )
    elif bc_type == "cantilever":
        bc = BoundaryConditions(
            dirichlet={
                0: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0},
            },
            neumann={n_elem: {2: -1.0}},
        )
    else:
        raise ValueError(f"bc_type inconnu : {bc_type!r}")

    return mesh, bc


# ---------------------------------------------------------------------------
# Cas 1 : Colonne 3D seule
# ---------------------------------------------------------------------------

def case_column_3d(E: float, nu: float, L: float, n_elem: int) -> None:
    """Colonne avec section rectangulaire non-carrée — deux plans de flambage."""
    mat = ElasticMaterial(E=E, nu=nu, rho=7800)

    # Section 20 × 10 mm : Iz (flexion forte) = 4 × Iy (flexion faible)
    b = 0.010   # largeur  (z) → Iy = h·b³/12  (axe faible)
    h = 0.020   # hauteur  (y) → Iz = b·h³/12  (axe fort)
    sec = RectangularSection(width=b, height=h)

    Iy = sec.Iy   # flexion faible — premier mode
    Iz = sec.Iz   # flexion forte — deuxième mode

    P_weak   = np.pi**2 * E * Iy / L**2
    P_strong = np.pi**2 * E * Iz / L**2

    print("\n" + "=" * 65)
    print("  CAS 1 : COLONNE 3D — section 20 × 10 mm, L = 1 m")
    print("=" * 65)
    print(f"  Iy = {Iy*1e9:.2f} mm⁴ (axe faible) — P_cr_faible  = {P_weak:.2f} N")
    print(f"  Iz = {Iz*1e9:.2f} mm⁴ (axe fort)   — P_cr_fort    = {P_strong:.2f} N")

    for bc_type, label, k in [("pinpin",    "PINCE-PINCÉE",   1.0),
                                ("cantilever","ENCASTRÉE-LIBRE", 2.0)]:
        mesh, bc = build_column_z(mat, sec, L, n_elem, bc_type)
        lambda_cr, _ = run_buckling(mesh, bc, n_modes=2)

        coeff = 1.0 / k**2   # 1 pour pp, 0.25 pour console
        P1_ana = coeff * P_weak
        P2_ana = coeff * P_strong

        print(f"\n  {label}")
        print(f"  {'Mode':>4}  {'Axe':>8}  {'P_cr FEM (N)':>14}  {'Analytique (N)':>16}  {'Erreur':>8}")
        print(f"  {'----':>4}  {'--------':>8}  {'------------':>14}  {'----------':>16}  {'------':>8}")

        for i, (p_fem, p_ana, axe) in enumerate(
            zip(lambda_cr[:2], [P1_ana, P2_ana], ["faible", "fort"])
        ):
            err = abs(float(p_fem) - p_ana) / p_ana * 100
            print(f"  {i+1:>4}  {axe:>8}  {float(p_fem):>14.3f}  {p_ana:>16.3f}  {err:>7.3f}%")


# ---------------------------------------------------------------------------
# Cas 2 : Convergence Beam3D → Beam2D
# ---------------------------------------------------------------------------

def case_convergence_to_beam2d(E: float, nu: float, L: float, n_elem: int) -> None:
    """Vérifie que Beam3D reproduit exactement le résultat Beam2D pour un cas plan."""
    from femsolver.elements.beam2d import Beam2D

    mat = ElasticMaterial(E=E, nu=nu, rho=7800)
    b = h = 0.010   # section carrée → Iy = Iz
    sec = RectangularSection(width=b, height=h)
    I = sec.Iy   # = Iz

    P_cr_ana = np.pi**2 * E * I / L**2

    # Beam3D
    mesh_3d, bc_3d = build_column_z(mat, sec, L, n_elem, "pinpin")
    lam_3d, _ = run_buckling(mesh_3d, bc_3d, n_modes=1)

    # Beam2D
    Le = L / n_elem
    nodes_2d = np.array([[0.0, i * Le] for i in range(n_elem + 1)])
    props_2d = {"area": sec.area, "inertia": I}
    from femsolver.core.mesh import BoundaryConditions, ElementData, Mesh
    elements_2d = tuple(
        ElementData(Beam2D, (i, i + 1), mat, props_2d) for i in range(n_elem)
    )
    mesh_2d = Mesh(nodes=nodes_2d, elements=elements_2d, n_dim=2, dof_per_node=3)
    bc_2d = BoundaryConditions(
        dirichlet={0: {0: 0.0, 1: 0.0}, n_elem: {0: 0.0}},
        neumann={n_elem: {1: -1.0}},
    )
    assembler_2d = Assembler(mesh_2d)
    K_2d = assembler_2d.assemble_stiffness()
    F_2d = assembler_2d.assemble_forces(bc_2d)
    ds_2d = apply_dirichlet(K_2d, F_2d, mesh_2d, bc_2d)
    u_2d  = StaticSolver().solve(*ds_2d)
    Kg_2d = assembler_2d.assemble_geometric_stiffness(u_2d)
    lam_2d, _ = BucklingSolver().solve(ds_2d.K_free, ds_2d.reduce(Kg_2d), n_modes=1)

    print("\n" + "=" * 65)
    print("  CAS 2 : CONVERGENCE Beam3D → Beam2D (section carrée 10 mm)")
    print("=" * 65)
    print(f"  P_cr analytique : {P_cr_ana:.4f} N")
    print(f"  P_cr Beam2D     : {float(lam_2d[0]):.4f} N  "
          f"(err = {abs(float(lam_2d[0])-P_cr_ana)/P_cr_ana*100:.4f}%)")
    print(f"  P_cr Beam3D     : {float(lam_3d[0]):.4f} N  "
          f"(err = {abs(float(lam_3d[0])-P_cr_ana)/P_cr_ana*100:.4f}%)")
    diff_rel = abs(float(lam_3d[0]) - float(lam_2d[0])) / float(lam_2d[0]) * 100
    # Beam3D est Timoshenko (K inclut la correction de cisaillement Φ ≈ 0.02 %)
    # Beam2D est Euler-Bernoulli → écart attendu ~ Φ/2 < 0.05 %
    print(f"  Écart 3D vs 2D  : {diff_rel:.4f}%  (attendu < 0.05 % — effet Timoshenko)")
    ok = diff_rel < 0.05
    print(f"  {'✓ Convergence validée' if ok else '✗ ÉCHEC convergence'}")


# ---------------------------------------------------------------------------
# Cas 3 : Portique 3D (deux colonnes + traverse)
# ---------------------------------------------------------------------------

def case_portal_frame_3d(E: float, nu: float) -> None:
    """Portique 3D symétrique — flambage sous charge verticale.

    Géométrie :
      - Deux colonnes de hauteur H = 3 m encastrées à la base
      - Traverse de portée W = 4 m sur le dessus des colonnes
      - Charge verticale P_ref = 1 N au sommet de chaque colonne

    Le premier mode de flambage est le déversement latéral (sidesway).
    Pour un portique à base encastrée et traverse rigide, l'approximation
    de la charge critique est (formule Salmon & Johnson) :

        P_cr ≈ π²EI_col / (kH)²   avec k ≈ 1.2  (portique avec traverse)

    On observe aussi la dégénérescence des deux premiers modes (symétrie
    gauche-droite du portique).
    """
    H = 3.0   # m — hauteur des colonnes
    W = 4.0   # m — portée de la traverse
    mat = ElasticMaterial(E=E, nu=nu, rho=7800)

    # Section des colonnes : IPE 200 simplifié (rectangle 200 × 10 mm)
    b_col, h_col = 0.010, 0.200
    sec_col = RectangularSection(width=b_col, height=h_col)

    # Section de la traverse : IPE 120 simplifié (rectangle 120 × 8 mm)
    b_tr, h_tr = 0.008, 0.120
    sec_tr = RectangularSection(width=b_tr, height=h_tr)

    n_col = 10   # éléments par colonne
    n_tr  = 10   # éléments pour la traverse

    # ── Nœuds ────────────────────────────────────────────────────────────────
    # Colonne gauche : nœuds 0..n_col  (x=-W/2, y=0, z=0..H)
    # Colonne droite : nœuds n_col+1..2*n_col+1 (x=+W/2, y=0, z=0..H)
    # Traverse       : nœuds 2*n_col+2..2*n_col+2+n_tr  (x=-W/2..W/2, y=0, z=H)
    #   (nœuds extrêmes déjà dans les colonnes → partage)

    col_nodes_L = np.array([[-W/2, 0.0, i * H / n_col] for i in range(n_col + 1)])
    col_nodes_R = np.array([[ W/2, 0.0, i * H / n_col] for i in range(n_col + 1)])
    # Traverse (sans nœud de jonction gauche et droit — partagés)
    tr_inner = np.array([
        [-W/2 + (j+1) * W / n_tr, 0.0, H]
        for j in range(n_tr - 1)
    ])

    nodes_all = np.vstack([col_nodes_L, col_nodes_R, tr_inner])
    n_L = n_col + 1   # nombre nœuds colonne gauche
    n_R = n_col + 1

    idx_top_L  = n_col             # sommet colonne gauche
    idx_bot_R  = n_L               # pied colonne droite
    idx_top_R  = n_L + n_col       # sommet colonne droite
    idx_tr_s   = n_L + n_R         # 1er nœud intérieur traverse

    # Éléments colonnes
    elems_col_L = [
        ElementData(Beam3D, (i, i+1), mat, {"section": sec_col})
        for i in range(n_col)
    ]
    elems_col_R = [
        ElementData(Beam3D, (n_L+i, n_L+i+1), mat, {"section": sec_col})
        for i in range(n_col)
    ]

    # Éléments traverse (sommet_L → intérieur → sommet_R)
    tr_seq = [idx_top_L] + list(range(idx_tr_s, idx_tr_s + n_tr - 1)) + [idx_top_R]
    elems_tr = [
        ElementData(Beam3D, (tr_seq[j], tr_seq[j+1]), mat, {"section": sec_tr})
        for j in range(n_tr)
    ]

    elements = tuple(elems_col_L + elems_col_R + elems_tr)
    mesh = Mesh(nodes=nodes_all, elements=elements, n_dim=3, dof_per_node=6)

    # ── Conditions aux limites ────────────────────────────────────────────────
    # Encastrement aux deux pieds
    fixed = {k: 0.0 for k in range(6)}
    bc = BoundaryConditions(
        dirichlet={
            0:       fixed,           # pied colonne gauche
            idx_bot_R: fixed,         # pied colonne droite
        },
        neumann={
            idx_top_L: {2: -1.0},    # charge P_ref=1N en z, colonne gauche
            idx_top_R: {2: -1.0},    # charge P_ref=1N en z, colonne droite
        },
    )

    lambda_cr, _ = run_buckling(mesh, bc, n_modes=3)

    # Formule approchée : P_cr ≈ π²EI_col/(kH)²  k≈1.2
    I_col = sec_col.Iy   # axe faible (flexion latérale)
    k_eff = 1.2
    P_cr_approx = np.pi**2 * E * I_col / (k_eff * H)**2

    print("\n" + "=" * 65)
    print("  CAS 3 : PORTIQUE 3D — flambage latéral (sidesway)")
    print("=" * 65)
    print(f"  Colonnes : {h_col*1e3:.0f} × {b_col*1e3:.0f} mm, H = {H} m")
    print(f"  Traverse : {h_tr*1e3:.0f} × {b_tr*1e3:.0f} mm, W = {W} m")
    print(f"  Chargement : P_ref = 1 N / colonne")
    print(f"\n  Approximation de flambage latéral (k = {k_eff}) :")
    print(f"    P_cr ≈ π²EI_col/(kH)² = {P_cr_approx:.2f} N  (ordre de grandeur)")
    print(f"\n  {'Mode':>4}  {'P_cr FEM (N)':>14}")
    print(f"  {'----':>4}  {'------------':>14}")
    for i, lam in enumerate(lambda_cr):
        print(f"  {i+1:>4}  {float(lam):>14.2f}")

    # Vérification qualitative : les deux premiers modes doivent être proches
    # (flambage symétrique et antisymétrique presque dégénérés pour portique symétrique)
    ratio_1_2 = float(lambda_cr[1]) / float(lambda_cr[0])
    print(f"\n  Ratio mode2/mode1 = {ratio_1_2:.3f}  "
          f"({'≈ 1 : dégénérés' if abs(ratio_1_2-1.0) < 0.1 else 'modes distincts'})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    E   = 210e9    # Pa — acier
    nu  = 0.3
    L   = 1.0      # m — longueur colonne
    n   = 20       # éléments par colonne

    print("=" * 65)
    print("  FLAMBAGE LINÉAIRE 3D — Beam3D (Timoshenko)")
    print("=" * 65)

    case_column_3d(E, nu, L, n)
    case_convergence_to_beam2d(E, nu, L, n)
    case_portal_frame_3d(E, nu)

    print("\n  ✓ Analyse de flambage 3D terminée.")
    print("=" * 65)


if __name__ == "__main__":
    main()

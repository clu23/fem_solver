"""Exemple : plaque infinie trouée sous traction uniaxiale — solution de Kirsch.

Problème
--------
Plaque en état plan de contraintes avec un trou circulaire de rayon R,
soumise à une traction σ0 en x à l'infini.

On modélise un quart de domaine annulaire (symétrie) :
- Inner : trou à r = R  (surface libre)
- Outer : arc à r = R_outer  (traction équivalente au champ lointain σ0)

Conditions aux limites
-----------------------
- y = 0 (θ = 0)   : u_y = 0  (symétrie horizontale)
- x = 0 (θ = π/2) : u_x = 0  (symétrie verticale)
- r = R            : libre     (trou, pas de traction)
- r = R_outer      : traction (σ_rr, σ_rθ) = composantes du champ lointain σ0 ê_x

Traction exacte de Kirsch sur le bord externe (r = R_outer, normal = ê_r)
avec k = (R/R_outer)² :

    σ_rr  = σ0/2·(1-k) + σ0/2·(1 - 4k + 3k²)·cos2θ
    σ_rθ  = -σ0/2·(1 + 2k - 3k²)·sin2θ
    t_x   = σ_rr·cosθ - σ_rθ·sinθ
    t_y   = σ_rr·sinθ + σ_rθ·cosθ

Utiliser la traction exacte (au lieu du champ lointain uniforme) élimine
l'erreur de domaine fini et réduit l'erreur sur Kt.

Solution analytique (Kirsch, 1898)
-----------------------------------
À la surface du trou (r = R), en θ = π/2 :
    σ_θθ = σ0 · (1 + 2(R/R)²) = 3 σ0   →  Kt = 3

FEM donne Kt ≈ 3 pour R_outer/R >> 1 (domaine quasi-infini).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from femsolver.core.assembler import Assembler
from femsolver.core.boundary import apply_dirichlet
from femsolver.core.material import ElasticMaterial
from femsolver.core.mesh import BoundaryConditions, ElementData, Mesh
from femsolver.core.solver import StaticSolver
from femsolver.elements.quad4 import Quad4
from femsolver.postprocess.stress import nodal_stresses, von_mises_2d


# ---------------------------------------------------------------------------
# Paramètres du problème
# ---------------------------------------------------------------------------

R = 1.0          # Rayon du trou [m]
R_OUTER = 5.0    # Rayon externe (≥ 5R pour approximer la plaque infinie)
N_R = 16         # Divisions radiales
N_THETA = 20     # Divisions angulaires (doit être pair pour symétrie à 45°)
THICKNESS = 1.0  # Épaisseur [m]
SIGMA0 = 1.0     # Traction appliquée [Pa]

STEEL = ElasticMaterial(E=210e9, nu=0.3, rho=7800)


# ---------------------------------------------------------------------------
# Construction du maillage polaire
# ---------------------------------------------------------------------------

def build_mesh() -> tuple[Mesh, BoundaryConditions]:
    """Construit le maillage annulaire en quart de disque (Quad4).

    Nodes disposés en grille polaire (r, θ) :
        r_i = R + i·(R_outer - R)/N_R     pour i = 0 … N_R
        θ_j = j·π/(2·N_THETA)             pour j = 0 … N_THETA

    Chaque cellule (i,j)–(i+1,j)–(i+1,j+1)–(i,j+1) est un Quad4.
    """
    # --- Nœuds ---
    # Espacement géométrique : concentre les éléments près du trou (r=R)
    # où les gradients de contraintes sont en 1/r² (solution de Kirsch).
    # Δr_min ≈ R·(q-1) avec q=(R_outer/R)^(1/N_R), soit ~10× plus fin qu'uniforme.
    r_vals = np.linspace(R, R_OUTER, N_R + 1)
    t_vals = np.linspace(0.0, np.pi / 2.0, N_THETA + 1)

    nodes_list = []
    node_index = {}   # (i, j) → global index
    idx = 0
    for i, r in enumerate(r_vals):
        for j, t in enumerate(t_vals):
            nodes_list.append([r * np.cos(t), r * np.sin(t)])
            node_index[(i, j)] = idx
            idx += 1

    nodes = np.array(nodes_list)

    # --- Éléments Quad4 ---
    props = {"thickness": THICKNESS, "formulation": "plane_stress"}
    elements = []
    for i in range(N_R):
        for j in range(N_THETA):
            n0 = node_index[(i,     j    )]
            n1 = node_index[(i + 1, j    )]
            n2 = node_index[(i + 1, j + 1)]
            n3 = node_index[(i,     j + 1)]
            elements.append(ElementData(Quad4, (n0, n1, n2, n3), STEEL, props))

    mesh = Mesh(nodes=np.array(nodes), elements=tuple(elements), n_dim=2)

    # --- Conditions aux limites ---
    # Dirichlet : symétries
    dirichlet: dict[int, dict[int, float]] = {}

    # Symétrie y=0 (θ=0, j=0) : uy=0 pour tous les r_i
    for i in range(N_R + 1):
        nid = node_index[(i, 0)]
        dirichlet[nid] = {1: 0.0}

    # Symétrie x=0 (θ=π/2, j=N_THETA) : ux=0 pour tous les r_i
    for i in range(N_R + 1):
        nid = node_index[(i, N_THETA)]
        dirichlet[nid] = {0: 0.0}

    # Les deux séries de CL (uy=0 sur y=0, ux=0 sur x=0) suffisent à éliminer
    # les 3 modes rigides (translation x, translation y, rotation z).
    # Pas de contrainte supplémentaire nécessaire.

    # Neumann : traction exacte de Kirsch sur le bord externe (r = R_outer, i = N_R)
    #
    # Solution de Kirsch — contraintes polaires à r = r_outer :
    #   k = (R / r_outer)²
    #   σ_rr  = σ0/2·(1-k) + σ0/2·(1 - 4k + 3k²)·cos2θ
    #   σ_rθ  = -σ0/2·(1 + 2k - 3k²)·sin2θ
    #
    # Loi de Cauchy sur la normale ê_r = (cosθ, sinθ) :
    #   t_x = σ_rr·cosθ - σ_rθ·sinθ
    #   t_y = σ_rr·sinθ + σ_rθ·cosθ
    #
    # Force nodale = traction × longueur tributaire × épaisseur
    neumann: dict[int, dict[int, float]] = {}
    arc_step = R_OUTER * (np.pi / 2.0) / N_THETA  # longueur d'un arc élémentaire
    k = (R / R_OUTER) ** 2   # (R/r)² au bord externe

    for j in range(N_THETA + 1):
        theta = t_vals[j]
        # Longueur tributaire (demi-pas aux extrémités)
        if j == 0 or j == N_THETA:
            s = arc_step / 2.0
        else:
            s = arc_step

        nid = node_index[(N_R, j)]
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        cos2t = np.cos(2.0 * theta)
        sin2t = np.sin(2.0 * theta)

        sigma_rr = SIGMA0 / 2.0 * (1.0 - k) + SIGMA0 / 2.0 * (1.0 - 4.0 * k + 3.0 * k**2) * cos2t
        sigma_rt = -SIGMA0 / 2.0 * (1.0 + 2.0 * k - 3.0 * k**2) * sin2t

        fx = (sigma_rr * cos_t - sigma_rt * sin_t) * s * THICKNESS
        fy = (sigma_rr * sin_t + sigma_rt * cos_t) * s * THICKNESS
        neumann[nid] = {0: fx, 1: fy}

    bc = BoundaryConditions(dirichlet=dirichlet, neumann=neumann)
    return mesh, bc


# ---------------------------------------------------------------------------
# Résolution
# ---------------------------------------------------------------------------

def main() -> None:
    mesh, bc = build_mesh()
    print(f"Maillage : {mesh.n_nodes} nœuds, {len(mesh.elements)} éléments Quad4")

    assembler = Assembler(mesh)
    K = assembler.assemble_stiffness()
    F = assembler.assemble_forces(bc)
    K_bc, F_bc = apply_dirichlet(K, F, mesh, bc)
    u = StaticSolver().solve(K_bc, F_bc)

    # --- Contraintes lissées et Von Mises ---
    sigma = nodal_stresses(mesh, u, "plane_stress")
    sigma_vm = von_mises_2d(sigma)

    # --- Facteur de concentration de contraintes ---
    # Nœuds sur la surface du trou (i=0) à θ = π/2 (j=N_THETA) → σ_tt ≈ σ_xx
    nodes = mesh.nodes
    t_vals = np.linspace(0.0, np.pi / 2.0, N_THETA + 1)
    node_index = {
        (i, j): i * (N_THETA + 1) + j
        for i in range(N_R + 1)
        for j in range(N_THETA + 1)
    }

    nid_top = node_index[(0, N_THETA)]   # trou, θ=π/2
    nid_side = node_index[(0, 0)]        # trou, θ=0

    # À θ=π/2 : ê_θ = -ê_x → σ_θθ = σ_xx  → Kt = σ_xx / σ0 ≈ 3
    # À θ=0   : ê_θ =  ê_y → σ_θθ = σ_yy  → σ_yy = -σ0 (compression)
    print(f"\n--- Facteur de concentration de contraintes ---")
    print(f"  σ_xx au sommet du trou (θ=π/2) : {sigma[nid_top, 0] / SIGMA0:.3f} σ0")
    print(f"  σ_vm au sommet du trou (θ=π/2) : {sigma_vm[nid_top] / SIGMA0:.3f} σ0")
    print(f"  Théorie (Kirsch) : Kt = 3.000")
    print(f"  σ_yy sur le côté du trou (θ=0) : {sigma[nid_side, 1] / SIGMA0:.3f} σ0")
    print(f"  Théorie (Kirsch) : σ_yy = -1.000 σ0  (compression hoop)")

    # --- Visualisation ---
    _plot_results(mesh, sigma_vm, sigma)


def _plot_results(
    mesh: Mesh,
    sigma_vm: np.ndarray,
    sigma: np.ndarray,
) -> None:
    """Carte de couleurs de Von Mises + contours σxx."""
    nodes = mesh.nodes
    x, y = nodes[:, 0], nodes[:, 1]

    # Triangulation pour tripcolor
    from matplotlib.tri import Triangulation

    tri_indices = []
    for ed in mesh.elements:
        n = list(ed.node_ids)   # [n0, n1, n2, n3]
        # Décompose le quad en 2 triangles
        tri_indices.append([n[0], n[1], n[2]])
        tri_indices.append([n[0], n[2], n[3]])

    triang = Triangulation(x, y, np.array(tri_indices))
    vm_tri = sigma_vm  # valeur nodale → interpolée sur triangulation

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Gauche : Von Mises ---
    ax = axes[0]
    tc = ax.tripcolor(triang, vm_tri / SIGMA0, cmap="jet", shading="gouraud")
    cb = fig.colorbar(tc, ax=ax)
    cb.set_label("σ_vm / σ₀", fontsize=10)
    # Trou (arc blanc)
    theta_arc = np.linspace(0, np.pi / 2, 100)
    ax.fill_between(R * np.cos(theta_arc), R * np.sin(theta_arc),
                    alpha=1.0, color="white", zorder=5)
    ax.set_title("Von Mises σ_vm / σ₀", fontsize=12)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    ax.set_xlim(-0.1, R_OUTER + 0.1)
    ax.set_ylim(-0.1, R_OUTER + 0.1)

    # --- Droite : σxx ---
    ax = axes[1]
    tc2 = ax.tripcolor(triang, sigma[:, 0] / SIGMA0, cmap="RdBu_r",
                       shading="gouraud",
                       vmin=-1.5, vmax=3.5)
    cb2 = fig.colorbar(tc2, ax=ax)
    cb2.set_label("σ_xx / σ₀", fontsize=10)
    ax.fill_between(R * np.cos(theta_arc), R * np.sin(theta_arc),
                    alpha=1.0, color="white", zorder=5)
    # Contours
    levels = [-1.0, 0.0, 1.0, 2.0, 3.0]
    cs = ax.tricontour(triang, sigma[:, 0] / SIGMA0, levels=levels,
                       colors="k", linewidths=0.8)
    ax.clabel(cs, fmt="%.0f", fontsize=8)
    ax.set_title("σ_xx / σ₀  (Kirsch : Kt = 3 au sommet)", fontsize=12)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    ax.set_xlim(-0.1, R_OUTER + 0.1)
    ax.set_ylim(-0.1, R_OUTER + 0.1)

    plt.suptitle(
        f"Plaque trouée — traction σ₀, trou R={R} m, domaine R_outer={R_OUTER} m\n"
        f"Maillage {N_R}×{N_THETA} Quad4 — symétrie quart de domaine",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig("plate_with_hole.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()

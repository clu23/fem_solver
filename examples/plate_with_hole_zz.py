"""Plaque trouée — carte d'erreur ZZ (Zienkiewicz–Zhu).

Montre que l'estimateur d'erreur ZZ identifie correctement la zone à fort
gradient autour du trou, là où le maillage uniforme est insuffisant.

Physique (solution de Kirsch)
------------------------------
Contraintes polaires à r = R sur un trou dans une plaque infinie :
    σ_θθ(r=R, θ=π/2) = 3·σ₀  →  concentration Kt = 3
    σ_rr(r=R) = σ_rθ(r=R) = 0  (surface libre)

Le champ varie en R²/r² : gradient fort à r ≈ R, champ quasi-uniforme à r >> R.
→ L'estimateur ZZ doit concentrer l'erreur près du trou.

Figures produites
-----------------
1. Von Mises σ_vm / σ₀ (champ lissé SNA)
2. Indicateur d'erreur η_e élémentaire (SPR)
3. Indicateur normalisé η_e / η_max (en %)
4. Profil radial de η_e vs r (confirme la décroissance en 1/r²)
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.tri import Triangulation

from femsolver.core.assembler import Assembler
from femsolver.core.boundary import apply_dirichlet
from femsolver.core.material import ElasticMaterial
from femsolver.core.mesh import BoundaryConditions, ElementData, Mesh
from femsolver.core.solver import StaticSolver
from femsolver.elements.quad4 import Quad4
from femsolver.postprocess.error_estimator import zz_error_estimate
from femsolver.postprocess.stress import nodal_stresses, von_mises_2d


# ---------------------------------------------------------------------------
# Paramètres
# ---------------------------------------------------------------------------

R = 1.0
R_OUTER = 5.0
N_R = 16
N_THETA = 20
THICKNESS = 1.0
SIGMA0 = 1.0
STEEL = ElasticMaterial(E=210e9, nu=0.3, rho=7800)


# ---------------------------------------------------------------------------
# Maillage polaire (identique à plate_with_hole.py)
# ---------------------------------------------------------------------------

def build_mesh() -> tuple[Mesh, BoundaryConditions, dict]:
    r_vals = np.linspace(R, R_OUTER, N_R + 1)
    t_vals = np.linspace(0.0, np.pi / 2.0, N_THETA + 1)

    nodes_list = []
    node_index = {}
    idx = 0
    for i, r in enumerate(r_vals):
        for j, t in enumerate(t_vals):
            nodes_list.append([r * np.cos(t), r * np.sin(t)])
            node_index[(i, j)] = idx
            idx += 1

    nodes = np.array(nodes_list)
    props = {"thickness": THICKNESS, "formulation": "plane_stress"}
    elements = []
    elem_center_r = []

    for i in range(N_R):
        for j in range(N_THETA):
            n0 = node_index[(i,     j    )]
            n1 = node_index[(i + 1, j    )]
            n2 = node_index[(i + 1, j + 1)]
            n3 = node_index[(i,     j + 1)]
            elements.append(ElementData(Quad4, (n0, n1, n2, n3), STEEL, props))
            # Rayon au centroïde de l'élément
            r_mid = 0.5 * (r_vals[i] + r_vals[i + 1])
            elem_center_r.append(r_mid)

    mesh = Mesh(nodes=np.array(nodes), elements=tuple(elements), n_dim=2)

    dirichlet: dict[int, dict[int, float]] = {}
    for i in range(N_R + 1):
        dirichlet[node_index[(i, 0)]] = {1: 0.0}
        dirichlet[node_index[(i, N_THETA)]] = {0: 0.0}

    arc_step = R_OUTER * (np.pi / 2.0) / N_THETA
    k = (R / R_OUTER) ** 2
    neumann: dict[int, dict[int, float]] = {}
    for j in range(N_THETA + 1):
        theta = t_vals[j]
        s = arc_step / 2.0 if j in (0, N_THETA) else arc_step
        nid = node_index[(N_R, j)]
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        cos2t, sin2t = np.cos(2.0 * theta), np.sin(2.0 * theta)
        sigma_rr = SIGMA0/2*(1-k) + SIGMA0/2*(1 - 4*k + 3*k**2)*cos2t
        sigma_rt = -SIGMA0/2*(1 + 2*k - 3*k**2)*sin2t
        neumann[nid] = {
            0: (sigma_rr*cos_t - sigma_rt*sin_t) * s * THICKNESS,
            1: (sigma_rr*sin_t + sigma_rt*cos_t) * s * THICKNESS,
        }

    bc = BoundaryConditions(dirichlet=dirichlet, neumann=neumann)
    extra = {"elem_center_r": np.array(elem_center_r), "node_index": node_index}
    return mesh, bc, extra


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _make_triangulation(mesh: Mesh) -> Triangulation:
    """Triangulation Matplotlib à partir d'un maillage Quad4."""
    x, y = mesh.nodes[:, 0], mesh.nodes[:, 1]
    tri_idx = []
    for ed in mesh.elements:
        n = list(ed.node_ids)
        tri_idx.append([n[0], n[1], n[2]])
        tri_idx.append([n[0], n[2], n[3]])
    return Triangulation(x, y, np.array(tri_idx))


def _hole_patch(ax: plt.Axes) -> None:
    """Masque le trou (arc blanc)."""
    theta_arc = np.linspace(0, np.pi / 2, 200)
    ax.fill_between(
        R * np.cos(theta_arc), R * np.sin(theta_arc),
        alpha=1.0, color="white", zorder=5,
    )


def _elem_quad_polygons(mesh: Mesh) -> list[np.ndarray]:
    """Polygones des Quad4 pour PolyCollection."""
    polys = []
    for ed in mesh.elements:
        coords = mesh.nodes[list(ed.node_ids), :]   # (4, 2)
        polys.append(coords)
    return polys


def plot_results(
    mesh: Mesh,
    sigma_vm: np.ndarray,
    result_spr,
    result_sna,
    elem_center_r: np.ndarray,
) -> None:
    """Quatre sous-figures : Von Mises, η_e SPR, η_e SNA, profil radial."""
    triang = _make_triangulation(mesh)
    polys = _elem_quad_polygons(mesh)

    fig, axes = plt.subplots(2, 2, figsize=(16, 13))

    theta_arc = np.linspace(0, np.pi / 2, 200)

    # ---- 1. Von Mises (champ lissé) ----------------------------------------
    ax = axes[0, 0]
    tc = ax.tripcolor(triang, sigma_vm / SIGMA0, cmap="jet", shading="gouraud")
    cb = fig.colorbar(tc, ax=ax)
    cb.set_label("σ_vm / σ₀", fontsize=9)
    ax.fill_between(R * np.cos(theta_arc), R * np.sin(theta_arc),
                    alpha=1.0, color="white", zorder=5)
    ax.set_title("Von Mises σ_vm / σ₀\n(champ FEM lissé SNA)", fontsize=10)
    ax.set_aspect("equal"); ax.set_xlim(-0.1, R_OUTER + 0.1)
    ax.set_ylim(-0.1, R_OUTER + 0.1)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")

    # ---- 2. Indicateur η_e SPR (par élément — PolyCollection) ---------------
    ax = axes[0, 1]
    eta_e_spr = result_spr.eta_e
    eta_max = eta_e_spr.max()

    pc = PolyCollection(polys, array=eta_e_spr / (eta_max + 1e-30),
                        cmap="hot_r", clim=(0, 1), linewidth=0)
    ax.add_collection(pc)
    cb2 = fig.colorbar(pc, ax=ax)
    cb2.set_label("η_e / η_max", fontsize=9)
    ax.fill_between(R * np.cos(theta_arc), R * np.sin(theta_arc),
                    alpha=1.0, color="white", zorder=5)
    ax.set_title(
        f"Indicateur d'erreur ZZ-SPR η_e\n"
        f"η = {result_spr.eta:.3e}   e_rel = {result_spr.relative_error:.1%}",
        fontsize=10,
    )
    ax.set_xlim(-0.1, R_OUTER + 0.1); ax.set_ylim(-0.1, R_OUTER + 0.1)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")

    # ---- 3. Indicateur η_e SNA -----------------------------------------------
    ax = axes[1, 0]
    eta_e_sna = result_sna.eta_e
    eta_max_sna = eta_e_sna.max()

    pc2 = PolyCollection(polys, array=eta_e_sna / (eta_max_sna + 1e-30),
                         cmap="hot_r", clim=(0, 1), linewidth=0)
    ax.add_collection(pc2)
    cb3 = fig.colorbar(pc2, ax=ax)
    cb3.set_label("η_e / η_max", fontsize=9)
    ax.fill_between(R * np.cos(theta_arc), R * np.sin(theta_arc),
                    alpha=1.0, color="white", zorder=5)
    ax.set_title(
        f"Indicateur d'erreur ZZ-SNA η_e\n"
        f"η = {result_sna.eta:.3e}   e_rel = {result_sna.relative_error:.1%}",
        fontsize=10,
    )
    ax.set_xlim(-0.1, R_OUTER + 0.1); ax.set_ylim(-0.1, R_OUTER + 0.1)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")

    # ---- 4. Profil radial de η_e (moyenne par anneau radial) ----------------
    ax = axes[1, 1]

    # Moyenne de η_e par couche radiale (N_R couches de N_THETA éléments)
    eta_by_layer_spr = eta_e_spr.reshape(N_R, N_THETA).mean(axis=1)
    eta_by_layer_sna = eta_e_sna.reshape(N_R, N_THETA).mean(axis=1)
    r_layers = 0.5 * (np.linspace(R, R_OUTER, N_R + 1)[:-1]
                      + np.linspace(R, R_OUTER, N_R + 1)[1:])

    ax.semilogy(r_layers, eta_by_layer_spr, 'o-', color='C1',
                label="ZZ-SPR", linewidth=1.8, markersize=5)
    ax.semilogy(r_layers, eta_by_layer_sna, 's--', color='C0',
                label="ZZ-SNA", linewidth=1.5, markersize=5)

    # Référence théorique en 1/r² (normalisée)
    r_ref = np.linspace(R, R_OUTER, 200)
    scale = eta_by_layer_spr[0] * r_layers[0] ** 2
    ax.semilogy(r_ref, scale / r_ref ** 2, 'k:', linewidth=1.2,
                label="~ 1/r² (Kirsch théorie)", alpha=0.7)

    ax.axvline(R, color='gray', linestyle=':', linewidth=0.8)
    ax.text(R + 0.05, ax.get_ylim()[0] * 2 if ax.get_ylim()[0] > 0 else 1e-4,
            'r = R\n(trou)', fontsize=7, color='gray', va='bottom')

    ax.set_xlabel("r [m] (rayon centroïde de la couche)", fontsize=9)
    ax.set_ylabel("η_e moyen [Pa·√m]", fontsize=9)
    ax.set_title("Profil radial de l'indicateur d'erreur\n"
                 "(décroissance ~ 1/r² attendue)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.3)

    plt.suptitle(
        f"Plaque trouée — Estimateur ZZ   |   Maillage {N_R}×{N_THETA} Quad4\n"
        f"R_trou = {R} m, R_outer = {R_OUTER} m, σ₀ = {SIGMA0} Pa\n"
        f"L'erreur se concentre à r ≈ R (gradient de Kirsch en 1/r²)",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig("plate_with_hole_zz.png", dpi=150, bbox_inches="tight")
    print("Figure sauvegardée : plate_with_hole_zz.png")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    mesh, bc, extra = build_mesh()
    n_nodes = mesh.n_nodes
    n_elem = len(mesh.elements)
    print(f"Maillage : {n_nodes} nœuds, {n_elem} éléments Quad4")

    # Résolution
    assembler = Assembler(mesh)
    K = assembler.assemble_stiffness()
    F = assembler.assemble_forces(bc)
    K_bc, F_bc = apply_dirichlet(K, F, mesh, bc)
    u = StaticSolver().solve(K_bc, F_bc)

    # Contraintes FEM (SNA — pour Von Mises)
    sigma = nodal_stresses(mesh, u, "plane_stress")
    sigma_vm = von_mises_2d(sigma)

    # Vérification Kt
    node_index = extra["node_index"]
    nid_top = node_index[(0, N_THETA)]
    print(f"\n--- Vérification Kt ---")
    print(f"  σ_xx au sommet (θ=π/2) : {sigma[nid_top, 0] / SIGMA0:.3f} σ₀  (théorie : 3.000)")

    # Estimateurs ZZ
    print("\n--- Estimateur ZZ-SPR ---")
    result_spr = zz_error_estimate(mesh, u, formulation="plane_stress", method="spr")
    print(f"  η        = {result_spr.eta:.4e} Pa·√m")
    print(f"  ‖σ_h‖   = {result_spr.norm_sigma_h:.4e} Pa·√m")
    print(f"  e_rel    = {result_spr.relative_error:.2%}")

    print("\n--- Estimateur ZZ-SNA ---")
    result_sna = zz_error_estimate(mesh, u, formulation="plane_stress", method="sna")
    print(f"  η        = {result_sna.eta:.4e} Pa·√m")
    print(f"  ‖σ_h‖   = {result_sna.norm_sigma_h:.4e} Pa·√m")
    print(f"  e_rel    = {result_sna.relative_error:.2%}")

    # Éléments les plus erronés
    top5 = np.argsort(result_spr.eta_e)[-5:][::-1]
    print("\n--- Top 5 éléments les plus erronés (SPR) ---")
    elem_center_r = extra["elem_center_r"]
    for rank, e_idx in enumerate(top5):
        print(f"  #{rank+1} : elem {e_idx:4d}  r_centre = {elem_center_r[e_idx]:.2f} m  "
              f"η_e = {result_spr.eta_e[e_idx]:.3e}")

    # Confirmation : η moyen couche interne vs externe
    r_inner_mask = elem_center_r < R + (R_OUTER - R) / N_R * 2
    r_outer_mask = elem_center_r > R_OUTER - (R_OUTER - R) / N_R * 2
    eta_inner = result_spr.eta_e[r_inner_mask].mean()
    eta_outer = result_spr.eta_e[r_outer_mask].mean()
    print(f"\n  η moyen couche intérieure (r ≈ {R:.0f} m) : {eta_inner:.3e}")
    print(f"  η moyen couche extérieure (r ≈ {R_OUTER:.0f} m) : {eta_outer:.3e}")
    print(f"  Rapport intérieur/extérieur : {eta_inner/eta_outer:.1f}× — "
          f"l'erreur se concentre bien autour du trou")

    plot_results(mesh, sigma_vm, result_spr, result_sna, elem_center_r)


if __name__ == "__main__":
    main()

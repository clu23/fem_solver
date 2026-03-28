"""Validation modale 2D : poutre console Quad4 vs. Euler–Bernoulli.

Problème
--------
Poutre acier encastrée–libre (console) :
    L = 1.0 m,  h = 0.1 m (élancement L/h = 10)
    E = 210 GPa,  ν = 0.3,  ρ = 7800 kg/m³

On compare les fréquences de flexion obtenues par un maillage Quad4 2D
aux valeurs analytiques d'Euler–Bernoulli :

    ω_n = (β_n L)² / L²  ·  √(EI / ρA)
    f_n = ω_n / (2π)

    β_n L :  1.8751  4.6941  7.8548  10.9955  (racines de cos·cosh = −1)

Comment les modes de flexion émergent d'un maillage 2D
-------------------------------------------------------
Les éléments Quad4 n'ont pas de DDL de rotation — chaque nœud ne
possède que (u_x, u_y). La courbure apparaît *implicitement* :

  1. Flexion = gradient de déformation ε_xx sur la hauteur.
     Les nœuds de la fibre supérieure s'allongent (ε_xx > 0) et ceux
     de la fibre inférieure se raccourcissent (ε_xx < 0).

  2. La rigidité EI émerge de l'intégration ∫ y² dA sur la section :
     elle est capturée dès qu'il y a au moins 2 rangées de nœuds sur h.

  3. La masse inertielle de rotation (ρI en formulation poutre) est
     capturée par le léger décalage u_y entre les nœuds du haut et
     du bas — ce décalage encode la courbure.

En résumé : c'est la *géométrie du maillage* (variation de ux sur y)
qui encode le comportement en flexion, sans aucun DDL explicite de θ.

Notes sur la précision
----------------------
Le Quad4 standard (intégration 2×2 complète) souffre de **verrouillage
en cisaillement** (shear locking) pour les éléments à grand rapport
longueur/hauteur. Les fréquences de flexion sont surestimées ; l'erreur
décroît quand on raffine le maillage (plus d'éléments sur la hauteur).
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from femsolver.core.material import ElasticMaterial
from femsolver.core.mesh import BoundaryConditions, ElementData, Mesh
from femsolver.dynamics.modal import run_modal
from femsolver.elements.quad4 import Quad4


# ---------------------------------------------------------------------------
# Paramètres du problème
# ---------------------------------------------------------------------------

L = 1.0          # longueur [m]
H = 0.1          # hauteur de la section [m]
T = 0.01         # épaisseur hors-plan [m]  (se simplifie dans ω² = K/M)
E = 210e9        # module de Young [Pa]
NU = 0.3         # coefficient de Poisson
RHO = 7800.0     # masse volumique [kg/m³]

MATERIAL = ElasticMaterial(E=E, nu=NU, rho=RHO)

# Racines de l'équation transcendante : cos(βL)·cosh(βL) = −1
# (poutre console, encastrée-libre)
BETA_L = np.array([1.87510407, 4.69409113, 7.85475744, 10.99554073])
N_BENDING = len(BETA_L)


# ---------------------------------------------------------------------------
# Fréquences analytiques Euler–Bernoulli
# ---------------------------------------------------------------------------

def euler_bernoulli_freqs() -> np.ndarray:
    """Fréquences de flexion d'une console (Hz).

    Returns
    -------
    f_analytical : np.ndarray, shape (N_BENDING,)
        f_n = (β_n·L)² / (2π·L²) · √(EI / ρA)
    """
    I = T * H**3 / 12.0          # moment quadratique [m⁴]
    A = T * H                     # section [m²]
    # Épaisseur T se simplifie → f_n indépendant de T
    omega_n = BETA_L**2 / L**2 * np.sqrt(E * I / (RHO * A))
    return omega_n / (2.0 * np.pi)


# ---------------------------------------------------------------------------
# Maillage structuré Quad4
# ---------------------------------------------------------------------------

def make_cantilever_mesh(nx: int, ny: int) -> tuple[Mesh, BoundaryConditions]:
    """Maillage rectangulaire nx × ny éléments Quad4.

    Nœuds : grille régulière sur [0, L] × [0, H].
    Numérotation globale : k = j*(nx+1) + i  (i = col x, j = rangée y).
    Encastrement : tous les nœuds sur le bord x = 0.

    Parameters
    ----------
    nx : int
        Éléments dans la direction longitudinale x.
    ny : int
        Éléments dans la direction transverse y.

    Returns
    -------
    mesh : Mesh
    bc   : BoundaryConditions
    """
    xs = np.linspace(0.0, L, nx + 1)
    ys = np.linspace(0.0, H, ny + 1)
    x_grid, y_grid = np.meshgrid(xs, ys)          # shape (ny+1, nx+1)
    nodes = np.column_stack([x_grid.ravel(), y_grid.ravel()])

    def nid(i: int, j: int) -> int:
        """Indice global du nœud à la colonne i, rangée j."""
        return j * (nx + 1) + i

    # Éléments Quad4 sens trigonométrique : SW → SE → NE → NW
    elements = []
    for j in range(ny):
        for i in range(nx):
            node_ids = (nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1))
            elements.append(
                ElementData(
                    etype=Quad4,
                    node_ids=node_ids,
                    material=MATERIAL,
                    properties={"thickness": T, "formulation": "plane_stress"},
                )
            )

    mesh = Mesh(nodes=nodes, elements=tuple(elements), n_dim=2)

    # Encastrement : u_x = u_y = 0 sur le bord gauche (i = 0)
    dirichlet: dict[int, dict[int, float]] = {}
    for j in range(ny + 1):
        dirichlet[nid(0, j)] = {0: 0.0, 1: 0.0}

    bc = BoundaryConditions(dirichlet=dirichlet, neumann={})
    return mesh, bc


# ---------------------------------------------------------------------------
# Identification des modes de flexion
# ---------------------------------------------------------------------------

def identify_bending_modes(modes: np.ndarray, n_wanted: int) -> list[int]:
    """Indices des n_wanted premiers modes à dominante transverse (u_y).

    Un mode est classé « flexion » si son énergie transverse dépasse
    son énergie axiale : Σ u_y² > Σ u_x².

    Parameters
    ----------
    modes : np.ndarray, shape (n_dof, n_computed)
        Vecteurs propres M-normalisés, colonnes classées par fréquence.
    n_wanted : int
        Nombre de modes de flexion à retourner.

    Returns
    -------
    indices : list[int]
        Indices de colonnes dans ``modes``.
    """
    ux = modes[0::2, :]    # DOF pairs   (u_x de chaque nœud)
    uy = modes[1::2, :]    # DOF impairs (u_y de chaque nœud)
    energy_x = np.sum(ux**2, axis=0)
    energy_y = np.sum(uy**2, axis=0)
    bending_mask = energy_y > energy_x
    bending_indices = list(np.where(bending_mask)[0])
    return bending_indices[:n_wanted]


# ---------------------------------------------------------------------------
# Étude de convergence
# ---------------------------------------------------------------------------

MESHES: list[tuple[str, int, int]] = [
    ("4×1",   4,  1),
    ("8×2",   8,  2),
    ("20×4", 20,  4),
    ("40×8", 40,  8),
]


def run_convergence(
    f_analytical: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Lance l'analyse modale pour chaque maillage et renvoie les résultats.

    Returns
    -------
    results : dict  label → (f_fem, error_pct)
    """
    results: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for label, nx, ny in MESHES:
        mesh, bc = make_cantilever_mesh(nx, ny)
        # Ne demander que le nécessaire : 3× N_BENDING pour avoir des marge
        n_req = min(3 * N_BENDING, mesh.n_dof // 2 - 1)
        result = run_modal(mesh, bc, n_modes=n_req, use_lumped=False)

        bend_idx = identify_bending_modes(result.modes, N_BENDING)
        n_found = len(bend_idx)
        f_fem = result.freqs[bend_idx]

        error_pct = 100.0 * (f_fem - f_analytical[:n_found]) / f_analytical[:n_found]
        results[label] = (f_fem, error_pct)

    return results


# ---------------------------------------------------------------------------
# Affichage des résultats
# ---------------------------------------------------------------------------

def print_table(
    f_analytical: np.ndarray,
    results: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    """Tableau comparatif fréquences FEM vs. Euler–Bernoulli."""
    col_w = 15
    header = f"{'Mode':>4}  {'f_EB [Hz]':>{col_w}}"
    for label, *_ in MESHES:
        header += f"  {label + ' [Hz]':>{col_w}} {'err%':>6}"
    print(header)
    print("─" * len(header))

    for k in range(N_BENDING):
        row = f"  {k+1:2d}  {f_analytical[k]:>{col_w}.2f}"
        for label, *_ in MESHES:
            f_fem, err = results[label]
            if k < len(f_fem):
                sign = "+" if err[k] >= 0 else ""
                row += f"  {f_fem[k]:>{col_w}.2f} {sign}{err[k]:>5.1f}%"
            else:
                row += f"  {'—':>{col_w}}  {'—':>6}"
        print(row)
    print()


# ---------------------------------------------------------------------------
# Tracé
# ---------------------------------------------------------------------------

def plot_results(
    f_analytical: np.ndarray,
    results: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    """Convergence + déformées modales sur le maillage fin."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Panneau gauche : convergence des erreurs ──────────────────────────
    ax1 = axes[0]
    mesh_labels = [lab for lab, *_ in MESHES]
    x_ticks = range(len(mesh_labels))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    for k, color in enumerate(colors):
        errs = []
        xs_plot = []
        for xi, label in enumerate(mesh_labels):
            f_fem, err = results[label]
            if k < len(err):
                errs.append(err[k])
                xs_plot.append(xi)
        ax1.plot(xs_plot, errs, "o-", color=color, lw=1.8, label=f"Mode {k+1}")

    ax1.axhline(0.0, color="k", lw=0.8, ls="--", label="Valeur exacte")
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(mesh_labels)
    ax1.set_xlabel("Maillage nx × ny")
    ax1.set_ylabel("Erreur relative [%]")
    ax1.set_title(
        "Convergence des fréquences de flexion\n"
        "Quad4 (masse consistante) vs. Euler–Bernoulli"
    )
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ── Panneau droit : déformées modales (maillage 40×8) ────────────────
    ax2 = axes[1]
    nx_fine, ny_fine = 40, 8
    mesh_fine, bc_fine = make_cantilever_mesh(nx_fine, ny_fine)
    n_req = min(3 * N_BENDING, mesh_fine.n_dof // 2 - 1)
    result_fine = run_modal(mesh_fine, bc_fine, n_modes=n_req, use_lumped=False)
    bend_idx_fine = identify_bending_modes(result_fine.modes, 3)

    nodes = mesh_fine.nodes
    # Sélectionner les nœuds de la fibre médiane (y ≈ H/2)
    mid_mask = np.abs(nodes[:, 1] - H / 2.0) < 1e-10
    x_mid = nodes[mid_mask, 0]
    sort_order = np.argsort(x_mid)
    x_mid = x_mid[sort_order]

    mode_colors = ["tab:blue", "tab:orange", "tab:green"]
    for color, mode_k, k in zip(mode_colors, bend_idx_fine, range(3)):
        phi = result_fine.modes[:, mode_k]
        uy_all = phi[1::2]                  # DDF u_y de chaque nœud
        uy_mid = uy_all[mid_mask][sort_order]
        uy_mid /= np.max(np.abs(uy_mid))    # normalisation
        f_this = result_fine.freqs[mode_k]
        f_eb = f_analytical[k]
        err = 100.0 * (f_this - f_eb) / f_eb
        ax2.plot(
            x_mid, uy_mid, color=color, lw=2.0,
            label=f"Mode {k+1}  FEM {f_this:.0f} Hz  (EB {f_eb:.0f} Hz, {err:+.1f}%)",
        )

    ax2.axhline(0.0, color="k", lw=0.5)
    ax2.axvline(0.0, color="gray", lw=0.5, ls=":")
    ax2.set_xlabel("x [m]")
    ax2.set_ylabel("u_y normalisé (fibre médiane)")
    ax2.set_title(
        "Déformées modales de flexion\n"
        f"Maillage {nx_fine}×{ny_fine}, plane stress"
    )
    ax2.legend(fontsize=8, loc="lower left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = "examples/modal_beam_2d.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Figure sauvegardée : {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Exécute la validation et affiche les résultats."""
    f_analytical = euler_bernoulli_freqs()

    print("=" * 70)
    print("Validation modale 2D — Poutre console Quad4 vs. Euler–Bernoulli")
    print("=" * 70)
    print(f"L={L} m,  h={H} m,  E={E:.3g} Pa,  ν={NU},  ρ={RHO} kg/m³")
    print(f"Élancement L/h = {L/H:.0f}\n")

    print("Fréquences analytiques Euler–Bernoulli :")
    for k, f in enumerate(f_analytical):
        betaL = BETA_L[k]
        print(f"  f_{k+1} : β_n L = {betaL:.4f}  →  {f:.2f} Hz")
    print()

    results = run_convergence(f_analytical)

    print("Tableau comparatif (erreur = (f_FEM − f_EB) / f_EB) :\n")
    print_table(f_analytical, results)

    print(
        "Observations :\n"
        "  • Maillages grossiers (4×1, 8×2) : erreur +85→+97 % — verrouillage\n"
        "    en cisaillement sévère (éléments très élancés, intégration 2×2\n"
        "    complète crée une rigidité parasite de cisaillement).\n"
        "  • Maillage 20×4 : erreur < 5 % sur les modes 1–2, mais la surestimation\n"
        "    subsiste : les éléments restent un peu élancés.\n"
        "  • Maillage 40×8 : modes 1–2 < 1 % d'erreur (excellent). Modes 3–4\n"
        "    passent négatifs (-9 à -15 %) : c'est l'effet Timoshenko.\n"
        "    Le modèle 2D continu inclut naturellement la déformation en\n"
        "    cisaillement et l'inertie de rotation, ce qui abaisse les\n"
        "    fréquences élevées en dessous des valeurs Euler–Bernoulli.\n"
        "    Ce n'est pas une erreur FEM : c'est la physique réelle.\n"
    )

    plot_results(f_analytical, results)


if __name__ == "__main__":
    main()

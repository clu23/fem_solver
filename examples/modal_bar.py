"""Exemple : analyse modale d'une barre encastrée–libre.

Problème
--------
Barre en acier de longueur L, encastrée en x=0, libre en x=L.

Solution analytique (modes longitudinaux)
------------------------------------------
    f_n = (2n − 1) · c / (4L),    n = 1, 2, 3, …

    c = √(E/ρ) ≈ 5189 m/s  (acier : E=210 GPa, ρ=7800 kg/m³)

    f_1 ≈ 1297 Hz,  f_2 ≈ 3892 Hz,  f_3 ≈ 6486 Hz, …

Déformée modale analytique
---------------------------
    φ_n(x) = sin((2n−1)·π·x / (2L))

On compare la masse consistante (M_consistante) et la masse condensée
(M_lumped = diagonale) sur le tableau des fréquences et les déformées.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from femsolver.core.material import ElasticMaterial
from femsolver.core.mesh import BoundaryConditions, ElementData, Mesh
from femsolver.dynamics.modal import run_modal
from femsolver.elements.bar2d import Bar2D


# ---------------------------------------------------------------------------
# Paramètres
# ---------------------------------------------------------------------------

E = 210e9      # Module d'Young [Pa]
RHO = 7800.0   # Densité [kg/m³]
A = 1e-4       # Section [m²]
L = 1.0        # Longueur [m]
N_ELEM = 40    # Nombre d'éléments
N_MODES = 5    # Modes à extraire

C = float(np.sqrt(E / RHO))   # vitesse du son ≈ 5189 m/s
MAT = ElasticMaterial(E=E, nu=0.3, rho=RHO)


# ---------------------------------------------------------------------------
# Construction du maillage
# ---------------------------------------------------------------------------

def build_bar() -> tuple[Mesh, BoundaryConditions]:
    """Barre horizontale, encastrée en x=0, libre en x=L."""
    n_nodes = N_ELEM + 1
    nodes = np.column_stack([np.linspace(0.0, L, n_nodes), np.zeros(n_nodes)])
    elements = tuple(
        ElementData(Bar2D, (i, i + 1), MAT, {"area": A})
        for i in range(N_ELEM)
    )
    mesh = Mesh(nodes=nodes, elements=elements, n_dim=2)

    # Dirichlet : encastrement en x=0 + suppression modes transversaux
    dirichlet: dict[int, dict[int, float]] = {i: {1: 0.0} for i in range(n_nodes)}
    dirichlet[0][0] = 0.0   # ux = 0 à gauche

    bc = BoundaryConditions(dirichlet=dirichlet, neumann={})
    return mesh, bc


# ---------------------------------------------------------------------------
# Analyse modale
# ---------------------------------------------------------------------------

def main() -> None:
    mesh, bc = build_bar()
    print(f"Maillage : {mesh.n_nodes} nœuds, {N_ELEM} éléments Bar2D")
    print(f"c = √(E/ρ) = {C:.1f} m/s\n")

    result_c = run_modal(mesh, bc, n_modes=N_MODES, use_lumped=False)
    result_l = run_modal(mesh, bc, n_modes=N_MODES, use_lumped=True)

    # Fréquences analytiques
    f_exact = np.array([(2 * n - 1) * C / (4 * L) for n in range(1, N_MODES + 1)])

    # --- Tableau comparatif ---
    print(f"{'Mode':>4} | {'Analytique':>12} | {'Consistante':>12} | {'Err%':>6} | {'Condensée':>12} | {'Err%':>6}")
    print("-" * 68)
    for n in range(N_MODES):
        fc = result_c.freqs[n]
        fl = result_l.freqs[n]
        fa = f_exact[n]
        print(
            f"  {n+1:2d} | {fa:12.2f} | {fc:12.2f} | {(fc/fa-1)*100:+6.3f} | "
            f"{fl:12.2f} | {(fl/fa-1)*100:+6.3f}"
        )

    # --- Tracé des déformées modales ---
    _plot_modes(mesh, result_c, result_l, f_exact)


def _plot_modes(
    mesh: Mesh,
    result_c,
    result_l,
    f_exact: np.ndarray,
) -> None:
    """Déformées modales FEM vs analytique."""
    x_nodes = mesh.nodes[:, 0]
    x_cont = np.linspace(0.0, L, 300)

    fig, axes = plt.subplots(1, N_MODES, figsize=(3 * N_MODES, 4), sharey=True)

    colors_c = "#1f77b4"   # bleu — masse consistante
    colors_l = "#ff7f0e"   # orange — masse condensée
    color_a  = "#2ca02c"   # vert — analytique

    for n, ax in enumerate(axes):
        # Déformée analytique
        phi_exact = np.sin((2 * n + 1) * np.pi * x_cont / (2 * L))
        ax.plot(phi_exact, x_cont, color=color_a, lw=1.5,
                label="Analytique" if n == 0 else "")

        # Déformée FEM consistante (DDL ux = indices pairs)
        phi_c = result_c.modes[0::2, n]   # ux DOFs : 0, 2, 4, ...
        # Normaliser le signe pour comparer avec l'analytique
        if phi_c[1] * np.sin(np.pi * x_nodes[1] / (2 * L)) < 0:
            phi_c = -phi_c
        phi_c /= np.max(np.abs(phi_c))
        ax.plot(phi_c, x_nodes, "o-", color=colors_c, ms=3, lw=1,
                label="Consistante" if n == 0 else "")

        # Déformée FEM condensée
        phi_l = result_l.modes[0::2, n]
        if phi_l[1] * np.sin(np.pi * x_nodes[1] / (2 * L)) < 0:
            phi_l = -phi_l
        phi_l /= np.max(np.abs(phi_l))
        ax.plot(phi_l, x_nodes, "s--", color=colors_l, ms=3, lw=1,
                label="Condensée" if n == 0 else "")

        ax.set_title(
            f"Mode {n+1}\n"
            f"$f_{{exact}}$ = {f_exact[n]:.0f} Hz\n"
            f"$f_{{consist}}$ = {result_c.freqs[n]:.0f} Hz",
            fontsize=9,
        )
        ax.set_xlabel("φ(x) [normalisé]", fontsize=8)
        ax.axvline(0, color="gray", lw=0.5, ls=":")
        ax.set_xlim(-1.3, 1.3)

    axes[0].set_ylabel("x [m]", fontsize=9)
    axes[0].legend(fontsize=8, loc="upper left")

    plt.suptitle(
        f"Modes longitudinaux — barre encastrée–libre\n"
        f"L={L} m, E={E/1e9:.0f} GPa, ρ={RHO:.0f} kg/m³, "
        f"c={C:.0f} m/s, {N_ELEM} éléments Bar2D",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig("modal_bar.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nFigure sauvegardée : modal_bar.png")


if __name__ == "__main__":
    main()

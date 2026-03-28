"""Validation modale 3D : poutre console Hexa8 vs Euler–Bernoulli.

Problème
--------
Poutre acier encastrée–libre, section carrée :
    L = 1.0 m,  w = h = 0.05 m   (élancement L/h = 20)
    E = 210 GPa,  ν = 0.3,  ρ = 7800 kg/m³

Fréquences analytiques d'Euler–Bernoulli
-----------------------------------------
    f_n = λ_n² / (2π) · √(EI / ρA) / L²

    λ_1 = 1.87510407  →  f_1 ≈  41.91 Hz
    λ_2 = 4.69409113  →  f_2 ≈ 262.64 Hz
    λ_3 = 7.85475744  →  f_3 ≈ 735.45 Hz

Verrouillage en cisaillement (shear locking)
--------------------------------------------
L'Hexa8 avec intégration complète 2×2×2 de Gauss est sujet au **verrouillage
en cisaillement** (shear locking) lorsqu'il modélise une flexion. Ce phénomène
survient parce que l'élément tente de capturer la courbure de la déformée avec
des fonctions de forme trilinéaires : le champ de cisaillement parasite génère
une rigidité fictive qui surestime les fréquences propres.

Comportement observé (L=1m, w=h=0.05m, ny=nz=2 fixés) :

  nx |  nelem  |  f₁_FEM  |  err₁%  |  f₂_FEM  |  err₂%
  ---|---------|----------|---------|----------|-------
   5 |     20  |  112.4   | +168%   |  727     | +177%
  10 |     40  |   67.5   |  +61%   |  424     |  +61%
  20 |     80  |   50.2   |  +20%   |  312     |  +19%
  40 |    160  |   45.4   |   +8%   |  284     |   +8%
  80 |    320  |   43.6   |   +4%   |  274     |   +4%

L'erreur converge vers zéro quand nx → ∞ (convergence en h), mais à la façon
caractéristique du verrouillage : lente, pas en O(h²).

Remèdes au verrouillage (hors scope de cet exemple)
----------------------------------------------------
- **Reduced integration sélective** : intégration réduite 1×2×2 pour les
  termes de cisaillement transverse (comme l'élément S4R de ABAQUS).
- **Mode incompatibles (Q8I)** : enrichissement des fonctions de forme.
- **Éléments d'ordre supérieur** : Hexa20, Hexa27 (quadratiques).
- **Éléments coque ou poutre** si la géométrie s'y prête.

Démarrage
---------
    python examples/modal_beam_3d_hexa8.py
"""

from __future__ import annotations

import logging
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from femsolver.core.material import ElasticMaterial
from femsolver.core.mesh import BoundaryConditions, ElementData, Mesh
from femsolver.dynamics.modal import run_modal
from femsolver.elements.hexa8 import Hexa8

logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# Paramètres
# ---------------------------------------------------------------------------

E   = 210e9
NU  = 0.3
RHO = 7800.0
L   = 1.0       # m
W   = 0.05      # m
H   = 0.05      # m

MATERIAL = ElasticMaterial(E=E, nu=NU, rho=RHO)

_I      = W * H**3 / 12.0
_EI     = E * _I
_RHO_A  = RHO * W * H
_BETA   = np.array([1.87510407, 4.69409113, 7.85475744])
FREQS_EB = _BETA**2 / (2.0 * np.pi) * np.sqrt(_EI / _RHO_A) / L**2

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Constructeur de maillage
# ---------------------------------------------------------------------------

def build_cantilever(nx: int, ny: int = 2, nz: int = 2) -> tuple[Mesh, BoundaryConditions]:
    """Poutre console Hexa8 encastrée en x=0, libre en x=L.

    Parameters
    ----------
    nx, ny, nz : int
        Nombre d'éléments dans chaque direction.

    Returns
    -------
    mesh : Mesh
    bc : BoundaryConditions
    """
    n_node_y = ny + 1
    n_node_z = nz + 1

    def nid(ix: int, iy: int, iz: int) -> int:
        return ix * n_node_y * n_node_z + iy * n_node_z + iz

    nodes_list = [
        [ix / nx * L, iy / ny * W, iz / nz * H]
        for ix in range(nx + 1)
        for iy in range(ny + 1)
        for iz in range(nz + 1)
    ]
    nodes = np.array(nodes_list)

    elements = tuple(
        ElementData(
            Hexa8,
            (nid(ix,   iy,   iz),   nid(ix+1, iy,   iz),
             nid(ix+1, iy+1, iz),   nid(ix,   iy+1, iz),
             nid(ix,   iy,   iz+1), nid(ix+1, iy,   iz+1),
             nid(ix+1, iy+1, iz+1), nid(ix,   iy+1, iz+1)),
            MATERIAL, {},
        )
        for ix in range(nx)
        for iy in range(ny)
        for iz in range(nz)
    )

    dirichlet = {
        nid(0, iy, iz): {0: 0.0, 1: 0.0, 2: 0.0}
        for iy in range(ny + 1)
        for iz in range(nz + 1)
    }

    return Mesh(nodes=nodes, elements=elements, n_dim=3), BoundaryConditions(dirichlet=dirichlet, neumann={})


# ---------------------------------------------------------------------------
# Convergence en maillage
# ---------------------------------------------------------------------------

def run_convergence_study(nx_list: list[int]) -> list[dict]:
    """Calcule f1 et f2 pour chaque valeur de nx.

    Parameters
    ----------
    nx_list : list[int]
        Liste des nombres d'éléments longitudinaux.

    Returns
    -------
    rows : list[dict]
        Chaque dictionnaire contient nx, n_elem, n_dof, f1_fem, f2_fem,
        err1_pct, err2_pct.
    """
    rows = []
    f1_eb, f2_eb = FREQS_EB[0], FREQS_EB[1]
    for nx in nx_list:
        mesh, bc = build_cantilever(nx, ny=2, nz=2)
        result = run_modal(mesh, bc, n_modes=4)
        # Dégénérescence : modes 1&2 flexion plan xOy/xOz, on prend f[0]
        f1 = result.freqs[0]
        f3 = result.freqs[2]   # 2e harmonique (modes 3&4 dégénérés)
        rows.append({
            "nx": nx,
            "n_elem": nx * 4,
            "n_dof": mesh.n_dof,
            "f1_fem": f1,
            "f2_fem": f3,
            "err1_pct": (f1 - f1_eb) / f1_eb * 100.0,
            "err2_pct": (f3 - f2_eb) / f2_eb * 100.0,
        })
        print(
            f"  nx={nx:3d} | {nx*4:4d} éléments | {mesh.n_dof:5d} DDL "
            f"| f1={f1:7.2f} Hz ({rows[-1]['err1_pct']:+.1f}%) "
            f"| f2={f3:7.2f} Hz ({rows[-1]['err2_pct']:+.1f}%)"
        )
    return rows


# ---------------------------------------------------------------------------
# Visualisation modes propres (PyVista, optionnel)
# ---------------------------------------------------------------------------

def plot_mode_shapes(mesh: Mesh, result, n_modes: int = 4) -> None:
    """Sauvegarde des captures PNG de chaque mode propre."""
    try:
        from femsolver.postprocess.plotter3d import plot_deformed_3d
    except ImportError:
        print("PyVista non disponible — visualisation des modes ignorée.")
        return

    for k in range(min(n_modes, result.n_modes)):
        phi_k = result.modes[:, k]
        fname = os.path.join(OUT_DIR, f"modal_beam_3d_mode{k+1}.png")
        plot_deformed_3d(
            mesh,
            phi_k,
            title=f"Mode {k+1} — f={result.freqs[k]:.2f} Hz",
            show=False,
            screenshot=fname,
            off_screen=True,
        )
        print(f"  Mode {k+1} sauvegardé : {fname}")


# ---------------------------------------------------------------------------
# Graphique convergence
# ---------------------------------------------------------------------------

def plot_convergence(rows: list[dict]) -> None:
    """Courbe de convergence de l'erreur relative sur f₁ en fonction de nx."""
    nx_arr    = np.array([r["nx"]       for r in rows])
    err1_arr  = np.array([r["err1_pct"] for r in rows])
    err2_arr  = np.array([r["err2_pct"] for r in rows])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(nx_arr, err1_arr, "o-", label=f"f₁ (réf. {FREQS_EB[0]:.1f} Hz)", color="steelblue")
    ax.plot(nx_arr, err2_arr, "s--", label=f"f₂ (réf. {FREQS_EB[1]:.1f} Hz)", color="tomato")
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("nx (éléments longitudinaux)")
    ax.set_ylabel("Erreur relative [%]")
    ax.set_title("Convergence modale Hexa8 — Verrouillage en cisaillement")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, nx_arr[-1] + 2)

    fname = os.path.join(OUT_DIR, "modal_beam_3d_convergence.png")
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Graphique de convergence sauvegardé : {fname}")


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 65)
    print(" Validation modale 3D : poutre console Hexa8")
    print("=" * 65)
    print(f"\nGéométrie : L={L} m, w=h={W} m (section carrée)")
    print(f"Matériau  : E={E/1e9:.0f} GPa, ν={NU}, ρ={RHO} kg/m³")
    print(f"\nRéférence Euler–Bernoulli :")
    for k, (beta, f) in enumerate(zip(_BETA, FREQS_EB), 1):
        print(f"  f{k} = {f:.2f} Hz  (λ{k} = {beta:.5f})")

    # --- Étude de convergence ---
    print("\n--- Étude de convergence (ny=nz=2 fixés) ---")
    print(f"  {'nx':>4} | {'n_elem':>6} | {'n_dof':>6} | {'f1_FEM':>9} | {'err1%':>7} | {'f2_FEM':>9} | {'err2%':>7}")
    print("  " + "-" * 62)

    nx_list = [5, 10, 20, 40]
    rows = run_convergence_study(nx_list)

    # --- Tableau récapitulatif ---
    print("\n  Récapitulatif :")
    print(f"  {'nx':>4} | {'n_elem':>6} | {'n_dof':>6} | "
          f"{'f1_FEM':>9} | {'f1_EB':>9} | {'err1%':>7} | "
          f"{'f2_FEM':>9} | {'f2_EB':>9} | {'err2%':>7}")
    print("  " + "-" * 85)
    for r in rows:
        print(
            f"  {r['nx']:>4} | {r['n_elem']:>6} | {r['n_dof']:>6} | "
            f"  {r['f1_fem']:>7.2f} | {FREQS_EB[0]:>7.2f}   | {r['err1_pct']:>+6.1f}% | "
            f"  {r['f2_fem']:>7.2f} | {FREQS_EB[1]:>7.2f}   | {r['err2_pct']:>+6.1f}%"
        )

    # --- Graphique convergence ---
    plot_convergence(rows)

    # --- Dégénérescence modale (nx=20) ---
    print("\n--- Dégénérescence modale (nx=20, section carrée) ---")
    mesh20, bc20 = build_cantilever(nx=20, ny=2, nz=2)
    res20 = run_modal(mesh20, bc20, n_modes=6)
    print(f"  6 premières fréquences [Hz] : {res20.freqs}")
    gap12 = abs(res20.freqs[1] - res20.freqs[0]) / res20.freqs[0] * 100
    gap34 = abs(res20.freqs[3] - res20.freqs[2]) / res20.freqs[2] * 100
    print(f"  Écart modes 1–2 : {gap12:.4f}%  (attendu < 0.1%)")
    print(f"  Écart modes 3–4 : {gap34:.4f}%  (attendu < 0.1%)")

    # --- Visualisation modes propres ---
    print("\n--- Visualisation des 4 premiers modes (PyVista) ---")
    plot_mode_shapes(mesh20, res20, n_modes=4)

    # --- Message de synthèse ---
    print("\n" + "=" * 65)
    print(" Synthèse")
    print("=" * 65)
    print("\nL'Hexa8 avec intégration 2×2×2 surestime les fréquences de")
    print("flexion en raison du verrouillage en cisaillement. L'erreur")
    print("converge vers 0 quand nx → ∞ mais la convergence est lente :")
    err_fine = rows[-1]["err1_pct"]
    print(f"  nx=40 (160 éléments) : erreur = {err_fine:+.1f}% sur f₁")
    print("\nPour une précision < 5 %, deux options :")
    print("  1. Maillage très fin (nx ≳ 80)")
    print("  2. Élément amélioré (Hexa20, intégration réduite sélective)")
    print()


if __name__ == "__main__":
    main()

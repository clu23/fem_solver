"""Réponse aléatoire d'un oscillateur 1-DDL — validation Miles et RMS.

Ce script illustre l'analyse de réponse aléatoire complète :

1. Construction du modèle SDOF (barre Bar2D, ressort-masse équivalent)
2. Excitation en force aléatoire bruit blanc → RMS déplacement (σ_u)
3. Excitation à la base (base excitation) → RMS déplacement relatif + accélération absolue
4. Comparaison avec les formules analytiques exactes et l'équation de Miles

Formules analytiques (bruit blanc G₀, masse effective m_eff)
-------------------------------------------------------------
Excitation en force G_F [N²/Hz] :
    σ²_u = G_F / (8 ξ ωₙ³ m_eff²)

Excitation à la base G₀ [(m/s²)²/Hz] :
    σ²_u_rel = G₀ / (8 ξ ωₙ³)   [déplacement relatif]
    σ_a ≈ √(π fₙ G₀ / (4ξ))      [accélération absolue — Miles]

Usage
-----
    .venv/bin/python3 examples/random_response_sdof.py
"""

from __future__ import annotations

import numpy as np

from femsolver.core.assembler import Assembler
from femsolver.core.boundary import apply_dirichlet
from femsolver.core.material import ElasticMaterial
from femsolver.core.mesh import BoundaryConditions, ElementData, Mesh
from femsolver.dynamics.damping import HystereticDamping
from femsolver.dynamics.modal import run_modal
from femsolver.dynamics.damping import ModalDampingModel
from femsolver.dynamics.rayleigh import RayleighDamping
from femsolver.dynamics.random_response import (
    miles_equation,
    psd_white,
    run_random_base,
    run_random_force,
)
from femsolver.elements.bar2d import Bar2D


# ---------------------------------------------------------------------------
# Construction du modèle SDOF
# ---------------------------------------------------------------------------


def build_sdof(
    m_total: float, k: float, zeta: float
) -> tuple[Mesh, BoundaryConditions, dict]:
    """Barre Bar2D verticale comme oscillateur 1-DDL.

    Nœud 0 : encastrement complet.
    Nœud 1 : libre en uy (axial de la barre).

    Retourne le maillage, les CL, et un dictionnaire de paramètres effectifs
    (k_eff, m_eff, omega_n, fn, rayleigh).
    """
    E = 210e9    # Pa — acier
    L = 1.0      # m
    area = k * L / E
    rho = m_total / (area * L)

    nodes = np.array([[0.0, 0.0], [0.0, L]])
    mat = ElasticMaterial(E=E, nu=0.3, rho=rho)
    props = {"area": area}
    elements = (ElementData(Bar2D, (0, 1), mat, props),)
    mesh = Mesh(nodes=nodes, elements=elements, n_dim=2, dof_per_node=2)
    bc = BoundaryConditions(
        dirichlet={0: {0: 0.0, 1: 0.0}, 1: {0: 0.0}},
        neumann={},
    )

    # Paramètres effectifs depuis les matrices assemblées
    assembler = Assembler(mesh)
    K = assembler.assemble_stiffness()
    M = assembler.assemble_mass()
    F_dummy = np.zeros(mesh.n_dof)
    ds = apply_dirichlet(K, F_dummy, mesh, bc)
    k_eff = float(ds.K_free.toarray()[0, 0])
    m_eff = float(ds.reduce_mass(M).toarray()[0, 0])
    omega_n = np.sqrt(k_eff / m_eff)
    fn = omega_n / (2.0 * np.pi)
    free_dof = int(ds.free_dofs[0])

    # Rayleigh β-seul : ξ = β ωₙ/2
    rayleigh = RayleighDamping(alpha=0.0, beta=2.0 * zeta / omega_n)

    params = {
        "k_eff": k_eff, "m_eff": m_eff,
        "omega_n": omega_n, "fn": fn,
        "zeta": zeta, "rayleigh": rayleigh,
        "free_dof": free_dof,
    }
    return mesh, bc, params


# ---------------------------------------------------------------------------
# Cas 1 — Excitation en force
# ---------------------------------------------------------------------------


def case_force(mesh, bc, params, freqs, G_F, damping_models):
    """Calcule la RMS de déplacement pour plusieurs modèles d'amortissement."""
    omega_n = params["omega_n"]; m_eff = params["m_eff"]; zeta = params["zeta"]
    sigma_ana = np.sqrt(G_F / (8.0 * zeta * omega_n ** 3 * m_eff ** 2))
    free_dof = params["free_dof"]

    F_dir = np.zeros(mesh.n_dof)
    F_dir[free_dof] = 1.0
    G_input = psd_white(G_F, freqs)

    print(f"\n  Analytique : σ_u = {sigma_ana*1e6:.4f} µm")
    print(f"  {'Modèle':>20}  {'σ_u [µm]':>10}  {'Erreur':>8}")
    print(f"  {'-'*20}  {'-'*10}  {'-'*8}")
    for name, damping in damping_models:
        result = run_random_force(mesh, bc, F_dir, G_input, freqs, damping)
        rms = result.rms_u[free_dof] * 1e6
        err = abs(rms - sigma_ana * 1e6) / (sigma_ana * 1e6) * 100
        print(f"  {name:>20}  {rms:>10.4f}  {err:>7.2f}%")


# ---------------------------------------------------------------------------
# Cas 2 — Base excitation
# ---------------------------------------------------------------------------


def case_base(mesh, bc, params, freqs, G0, damping):
    """Calcule la RMS relative et absolue pour la base excitation."""
    omega_n = params["omega_n"]; fn = params["fn"]; zeta = params["zeta"]
    free_dof = params["free_dof"]

    sigma_u_ana = np.sqrt(G0 / (8.0 * zeta * omega_n ** 3))
    sigma_a_miles = miles_equation(fn, zeta, G0)
    G_input = psd_white(G0, freqs)

    result = run_random_base(mesh, bc, direction=1, G_input=G_input,
                              freqs=freqs, damping=damping)
    rms_u = result.rms_u[free_dof]
    rms_a = result.rms_a[free_dof]

    err_u = abs(rms_u - sigma_u_ana) / sigma_u_ana * 100
    err_a = abs(rms_a - sigma_a_miles) / sigma_a_miles * 100

    print(f"\n  Déplacement relatif :")
    print(f"    σ_u_rel numérique : {rms_u*1000:.4f} mm")
    print(f"    σ_u_rel analytique: {sigma_u_ana*1000:.4f} mm   (erreur = {err_u:.2f}%)")

    print(f"\n  Accélération absolue :")
    print(f"    σ_a numérique     : {rms_a:.4f} m/s²")
    print(f"    Miles σ_a         : {sigma_a_miles:.4f} m/s²   (erreur = {err_a:.2f}%)")
    print(f"    PSD peak G_a      : {result.G_a[free_dof, :].max():.4f} (m/s²)²/Hz")
    print(f"    G₀/4ξ²            : {G0/(4*zeta**2):.4f} (m/s²)²/Hz")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # Paramètres SDOF
    m_total = 1.0    # kg  (masse totale barre; m_eff = m/3 consistent mass)
    k = 1000.0       # N/m
    zeta = 0.05      # 5% amortissement

    mesh, bc, params = build_sdof(m_total, k, zeta)
    fn = params["fn"]; omega_n = params["omega_n"]; m_eff = params["m_eff"]

    print("=" * 65)
    print("  RÉPONSE ALÉATOIRE — OSCILLATEUR 1-DDL (SDOF)")
    print("=" * 65)
    print(f"\n  Paramètres FEM :")
    print(f"    m_total = {m_total:.1f} kg   m_eff = {m_eff:.4f} kg  (consistent mass, m/3)")
    print(f"    k_eff   = {params['k_eff']:.1f} N/m")
    print(f"    ωₙ      = {omega_n:.3f} rad/s   fₙ = {fn:.3f} Hz")
    print(f"    ξ       = {zeta:.0%}")

    # Grille de fréquences centrée sur la résonance ±20×fₙ
    freqs = np.linspace(fn / 20.0, fn * 20.0, 4000)

    # Modèles d'amortissement comparés
    rayleigh = params["rayleigh"]
    hysteretic = HystereticDamping(eta=2.0 * zeta)

    # -----------------------------------------------------------------------
    # Cas 1 : excitation en force
    # -----------------------------------------------------------------------
    G_F = 1.0   # N²/Hz (bruit blanc en force)
    print(f"\n{'─'*65}")
    print(f"  Cas 1 : excitation en FORCE  (G_F = {G_F:.1f} N²/Hz)")
    print(f"  Plage : [{freqs[0]:.2f}, {freqs[-1]:.1f}] Hz  —  {len(freqs)} points")
    case_force(mesh, bc, params, freqs, G_F, [
        ("Rayleigh (β seul)", rayleigh),
        ("Hystérétique η=2ξ", hysteretic),
    ])

    # -----------------------------------------------------------------------
    # Cas 2 : base excitation
    # -----------------------------------------------------------------------
    G0 = 0.01   # (m/s²)²/Hz (niveau sol typique)
    print(f"\n{'─'*65}")
    print(f"  Cas 2 : BASE EXCITATION  (G₀ = {G0:.3f} (m/s²)²/Hz)")
    print(f"  Plage : [{freqs[0]:.2f}, {freqs[-1]:.1f}] Hz  —  {len(freqs)} points")
    case_base(mesh, bc, params, freqs, G0, rayleigh)

    # -----------------------------------------------------------------------
    # Résumé Miles
    # -----------------------------------------------------------------------
    print(f"\n{'─'*65}")
    sigma_miles = miles_equation(fn, zeta, G0)
    print(f"  Équation de Miles :  σ_a = √(π fₙ G₀ / (4ξ))")
    print(f"    = √(π × {fn:.3f} × {G0} / (4 × {zeta}))")
    print(f"    = {sigma_miles:.4f} m/s²  ≡  {sigma_miles/9.81*1000:.2f} mg")
    print(f"\n  ✓ Analyse aléatoire terminée.")
    print("=" * 65)


if __name__ == "__main__":
    main()

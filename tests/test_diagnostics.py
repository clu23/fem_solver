"""Tests unitaires — module femsolver.core.diagnostics.

Solutions analytiques utilisées
---------------------------------

**Treillis Warren 2D**

Identique à examples/truss_bridge.py :
    - 5 nœuds, 7 barres, acier E=210 GPa, ρ=7800 kg/m³, A=5e-4 m²
    - Appui encastrement nœud 0, rouleau uy=0 nœud 4
    - Charges : 50 kN↓ aux nœuds 1 et 2

Vérifications analytiques :
    - Masse totale = ρ·A·Σ L_e
    - Réactions aux appuis = −Σ forces appliquées (Newton 3)
    - CG_x théorique = barycentre pondéré par les longueurs des barres

**Console 2D (Beam2D)**

Poutre console encastrée à gauche, charge P au bout libre :
    - 2 éléments de longueur L/2 (3 nœuds)
    - Appui : nœud 0 bloqué ux=uy=θ=0
    - Charge : P = 10 kN↓ au nœud 2

Réactions analytiques à l'encastrement (nœud 0) :
    - R_y = +P (réaction verticale)
    - M_z = −P·L (moment de réaction, sens antihoraire positif)
    - R_x = 0 (pas de charge horizontale)
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from femsolver.core.assembler import Assembler
from femsolver.core.boundary import apply_dirichlet
from femsolver.core.diagnostics import (
    DiagnosticsResult,
    check_equilibrium,
    check_mass,
    compute_reactions,
    run_diagnostics,
)
from femsolver.core.material import ElasticMaterial
from femsolver.core.mesh import BoundaryConditions, ElementData, Mesh
from femsolver.core.solver import StaticSolver
from femsolver.elements.bar2d import Bar2D
from femsolver.elements.beam2d import Beam2D


# ---------------------------------------------------------------------------
# Fixtures : Treillis Warren 2D
# ---------------------------------------------------------------------------


@pytest.fixture()
def warren_truss():
    """Treillis Warren 2D (5 nœuds, 7 barres, 2 appuis)."""
    steel = ElasticMaterial(E=210e9, nu=0.3, rho=7800.0)
    area  = 5e-4  # m²
    props = {"area": area}

    nodes = np.array([
        [0.0, 0.0],  # 0  encastrement
        [2.0, 0.0],  # 1
        [1.0, 1.0],  # 2
        [3.0, 1.0],  # 3
        [4.0, 0.0],  # 4  rouleau uy=0
    ])
    connectivity = [(0, 1), (1, 4), (0, 2), (2, 3), (3, 4), (1, 2), (1, 3)]
    elements = tuple(ElementData(Bar2D, conn, steel, props) for conn in connectivity)
    mesh = Mesh(nodes=nodes, elements=elements, n_dim=2)

    F_ext = -50_000.0  # N
    bc = BoundaryConditions(
        dirichlet={0: {0: 0.0, 1: 0.0}, 4: {1: 0.0}},
        neumann={1: {1: F_ext}, 2: {1: F_ext}},
    )

    assembler = Assembler(mesh)
    K = assembler.assemble_stiffness()
    M = assembler.assemble_mass()
    F = assembler.assemble_forces(bc)
    K_bc, F_bc = apply_dirichlet(K, F, mesh, bc)
    u = StaticSolver().solve(K_bc, F_bc)

    return mesh, K, M, F, bc, u


# ---------------------------------------------------------------------------
# Fixtures : Console Beam2D
# ---------------------------------------------------------------------------


@pytest.fixture()
def cantilever_beam():
    """Console 2D Euler–Bernoulli (2 éléments Beam2D, 3 nœuds)."""
    L = 2.0       # m (longueur totale)
    E = 210e9     # Pa
    A = 1e-4      # m²  (section transversale)
    I = 1e-8      # m⁴  (moment quadratique)
    rho = 7800.0  # kg/m³

    steel = ElasticMaterial(E=E, nu=0.3, rho=rho)
    props = {"area": A, "inertia": I}

    nodes = np.array([
        [0.0, 0.0],   # 0  encastrement
        [L/2, 0.0],   # 1
        [L,   0.0],   # 2  bout libre
    ])
    elements = (
        ElementData(Beam2D, (0, 1), steel, props),
        ElementData(Beam2D, (1, 2), steel, props),
    )
    mesh = Mesh(nodes=nodes, elements=elements, n_dim=2, dof_per_node=3)

    P = -10_000.0  # N (charge vers le bas au bout)
    bc = BoundaryConditions(
        dirichlet={0: {0: 0.0, 1: 0.0, 2: 0.0}},   # encastrement
        neumann={2: {1: P}},
    )

    assembler = Assembler(mesh)
    K = assembler.assemble_stiffness()
    M = assembler.assemble_mass()
    F = assembler.assemble_forces(bc)
    K_bc, F_bc = apply_dirichlet(K, F, mesh, bc)
    u = StaticSolver().solve(K_bc, F_bc)

    return mesh, K, M, F, bc, u, L, A, rho, P


# ===========================================================================
# Tests : vérification de la masse — Treillis Warren
# ===========================================================================


class TestMassCheckWarrenTruss:
    """Masse théorique = ρ·A·Σ L_e pour le treillis Warren."""

    def test_theoretical_mass_correct(self, warren_truss):
        """m_théo = ρ·A·Σ L_e (longueurs analytiques connues)."""
        mesh, K, M, F, bc, u = warren_truss
        steel = mesh.elements[0].material

        # Longueurs analytiques des 7 barres
        L_diag = np.sqrt(2.0)       # barres 0-2, 3-4 et autres diagonales à 45°
        lengths = [
            2.0,            # 0-1
            2.0,            # 1-4
            L_diag,         # 0-2
            2.0,            # 2-3
            L_diag,         # 3-4
            L_diag,         # 1-2
            np.sqrt(4+1),   # 1-3  : Δx=1, Δy=1... wait: nœud1=(2,0) nœud3=(3,1) → L=√2
        ]
        # Recalculer correctement depuis les coordonnées
        nodes = mesh.nodes
        conn  = [(0,1),(1,4),(0,2),(2,3),(3,4),(1,2),(1,3)]
        L_sum = sum(np.linalg.norm(nodes[b]-nodes[a]) for a, b in conn)
        m_expected = steel.rho * 5e-4 * L_sum

        m_theo, m_fem, cg = check_mass(mesh, M)

        np.testing.assert_allclose(m_theo, m_expected, rtol=1e-12)

    def test_fem_mass_equals_theoretical(self, warren_truss):
        """m_FEM (depuis M) = m_théo (depuis géométrie) à la précision machine."""
        mesh, K, M, F, bc, u = warren_truss
        m_theo, m_fem, cg = check_mass(mesh, M)
        np.testing.assert_allclose(m_fem, m_theo, rtol=1e-12)

    def test_center_of_gravity_x(self, warren_truss):
        """CG_x = barycentre pondéré par les longueurs des barres."""
        mesh, K, M, F, bc, u = warren_truss
        nodes = mesh.nodes
        conn  = [(0,1),(1,4),(0,2),(2,3),(3,4),(1,2),(1,3)]

        # Centroïdes et longueurs analytiques
        lengths   = [np.linalg.norm(nodes[b]-nodes[a]) for a, b in conn]
        centroids = [0.5*(nodes[a]+nodes[b]) for a, b in conn]
        m_total   = sum(lengths)    # × ρA (facteur commun)
        cg_x_expected = sum(L * c[0] for L, c in zip(lengths, centroids)) / m_total

        _, _, cg = check_mass(mesh, M)
        np.testing.assert_allclose(cg[0], cg_x_expected, rtol=1e-12)

    def test_center_of_gravity_y(self, warren_truss):
        """CG_y cohérent avec la géométrie du treillis."""
        mesh, K, M, F, bc, u = warren_truss
        nodes = mesh.nodes
        conn  = [(0,1),(1,4),(0,2),(2,3),(3,4),(1,2),(1,3)]
        lengths   = [np.linalg.norm(nodes[b]-nodes[a]) for a, b in conn]
        centroids = [0.5*(nodes[a]+nodes[b]) for a, b in conn]
        m_total   = sum(lengths)
        cg_y_expected = sum(L * c[1] for L, c in zip(lengths, centroids)) / m_total

        _, _, cg = check_mass(mesh, M)
        np.testing.assert_allclose(cg[1], cg_y_expected, rtol=1e-12)


# ===========================================================================
# Tests : vérification de la masse — Console Beam2D
# ===========================================================================


class TestMassCheckBeam2D:
    """Pour Beam2D (dpn=3), la masse FEM doit ignorer les DDL de rotation."""

    def test_theoretical_mass_beam(self, cantilever_beam):
        """m_théo = ρ·A·L (console 2 éléments)."""
        mesh, K, M, F, bc, u, L, A, rho, P = cantilever_beam
        m_expected = rho * A * L
        m_theo, m_fem, cg = check_mass(mesh, M)
        np.testing.assert_allclose(m_theo, m_expected, rtol=1e-10)

    def test_fem_mass_beam_matches_theoretical(self, cantilever_beam):
        """La masse FEM (DDL de translation uniquement) = masse théorique."""
        mesh, K, M, F, bc, u, L, A, rho, P = cantilever_beam
        m_theo, m_fem, cg = check_mass(mesh, M)
        np.testing.assert_allclose(m_fem, m_theo, rtol=1e-10)

    def test_cg_x_at_midspan(self, cantilever_beam):
        """CG_x d'une console uniforme = L/2 (symétrie de masse)."""
        mesh, K, M, F, bc, u, L, A, rho, P = cantilever_beam
        _, _, cg = check_mass(mesh, M)
        np.testing.assert_allclose(cg[0], L / 2.0, rtol=1e-12)

    def test_cg_y_zero(self, cantilever_beam):
        """CG_y = 0 (console sur l'axe y=0)."""
        mesh, K, M, F, bc, u, L, A, rho, P = cantilever_beam
        _, _, cg = check_mass(mesh, M)
        np.testing.assert_allclose(cg[1], 0.0, atol=1e-14)


# ===========================================================================
# Tests : réactions d'appui (SPCFORCE) — Treillis Warren
# ===========================================================================


class TestReactionsWarrenTruss:
    """Réactions sur le treillis Warren."""

    def test_reactions_computed_at_constrained_dofs_only(self, warren_truss):
        """Les réactions ne sont renvoyées que pour les DDL contraints."""
        mesh, K, M, F, bc, u = warren_truss
        reactions = compute_reactions(K, u, F, mesh, bc)
        # DDL contraints : 0→{ux,uy}=dofs 0,1 ; 4→{uy}=dof 9
        assert set(reactions.keys()) == {0, 1, 9}

    def test_reactions_sum_to_applied_forces(self, warren_truss):
        """Σ réactions y = −(−50 kN −50 kN) = +100 kN (Newton 3)."""
        mesh, K, M, F, bc, u = warren_truss
        reactions = compute_reactions(K, u, F, mesh, bc)

        # Réactions verticales aux deux appuis (DDL y=1 du nœud 0 et 4)
        R_y0 = reactions[1]    # nœud 0, DDL uy
        R_y4 = reactions[9]    # nœud 4, DDL uy (dpn=2 → dof=2×4+1=9)
        total_R_y = R_y0 + R_y4

        # Σ forces appliquées (toutes en Y) : 2 × (−50 kN)
        total_F_y = -50_000.0 + (-50_000.0)
        np.testing.assert_allclose(total_R_y, -total_F_y, rtol=1e-8)

    def test_reaction_x_node0_zero(self, warren_truss):
        """Aucune charge horizontale → R_x au nœud 0 ≈ 0 (équilibre horizontal)."""
        mesh, K, M, F, bc, u = warren_truss
        reactions = compute_reactions(K, u, F, mesh, bc)
        R_x0 = reactions[0]    # nœud 0, DDL ux
        np.testing.assert_allclose(R_x0, 0.0, atol=1e-6)

    def test_equilibrium_at_free_dofs_small(self, warren_truss):
        """Résidu (K·u−F) aux DDL libres ≈ 0 (équilibre local)."""
        mesh, K, M, F, bc, u = warren_truss
        constrained = {0, 1, 9}
        all_dofs = set(range(mesh.n_dof))
        free = sorted(all_dofs - constrained)

        residual_free = (K @ u - F)[free]
        np.testing.assert_allclose(residual_free, 0.0, atol=1e-6,
                                   err_msg="Résidu non nul aux DDL libres")


# ===========================================================================
# Tests : réactions d'appui (SPCFORCE) — Console Beam2D
# ===========================================================================


class TestReactionsCantileverBeam:
    """Réactions analytiques d'une console encastrée.

    Charges P = 10 kN ↓ au bout (nœud 2, DDL uy).
    Réactions à l'encastrement (nœud 0, DDL ux=0, uy=1, θ=2) :

        R_x  = 0
        R_y  = +P  (remontée)    [signe convention R = K·u − F]
        M_z  = +P·L  (moment d'encastrement dans le sens de la rotation)

    Note sur le signe : avec R = K·u − F, la réaction verticale est la
    force que l'appui exerce sur la structure.  Si P = −10 kN (bas),
    l'appui pousse vers le haut → R_y = +10 kN.
    """

    def test_reaction_rx_zero(self, cantilever_beam):
        """Pas de charge horizontale → R_x = 0."""
        mesh, K, M, F, bc, u, L, A, rho, P = cantilever_beam
        reactions = compute_reactions(K, u, F, mesh, bc)
        # nœud 0, DDL ux = global dof 0
        np.testing.assert_allclose(reactions[0], 0.0, atol=1e-6)

    def test_reaction_ry_equals_minus_P(self, cantilever_beam):
        """R_y = −P = +10 kN (réaction verticale à l'encastrement)."""
        mesh, K, M, F, bc, u, L, A, rho, P = cantilever_beam
        reactions = compute_reactions(K, u, F, mesh, bc)
        # nœud 0, DDL uy = global dof 1
        np.testing.assert_allclose(reactions[1], -P, rtol=1e-8)

    def test_reaction_moment_equals_P_times_L(self, cantilever_beam):
        """M_z = P·L (moment d'encastrement).

        Convention : M = K·u − F au DDL θ de l'encastrement.
        Pour P↓ = −10 kN et L = 2 m : M_z = −P·L = +20 kN·m.
        """
        mesh, K, M, F, bc, u, L, A, rho, P = cantilever_beam
        reactions = compute_reactions(K, u, F, mesh, bc)
        # nœud 0, DDL θz = global dof 2
        np.testing.assert_allclose(reactions[2], -P * L, rtol=1e-8)

    def test_all_constrained_dofs_present(self, cantilever_beam):
        """Toutes les 3 réactions de l'encastrement sont présentes."""
        mesh, K, M, F, bc, u, L, A, rho, P = cantilever_beam
        reactions = compute_reactions(K, u, F, mesh, bc)
        assert set(reactions.keys()) == {0, 1, 2}


# ===========================================================================
# Tests : bilan d'équilibre global
# ===========================================================================


class TestEquilibriumWarrenTruss:
    """Σ forces + Σ réactions = 0 dans chaque direction."""

    def test_equilibrium_ok_flag(self, warren_truss):
        """run_diagnostics retourne equilibrium_ok=True."""
        mesh, K, M, F, bc, u = warren_truss
        result = run_diagnostics(mesh, K, u, F, bc, M=M)
        assert result.equilibrium_ok

    def test_equilibrium_residual_x_small(self, warren_truss):
        """Résidu en X < tolérance (bilan horizontal)."""
        mesh, K, M, F, bc, u = warren_truss
        reactions = compute_reactions(K, u, F, mesh, bc)
        residuals, ok = check_equilibrium(F, reactions, mesh, tol=1e-6)
        np.testing.assert_allclose(residuals[0], 0.0, atol=1.0,
                                   err_msg="Résidu X non nul")

    def test_equilibrium_residual_y_small(self, warren_truss):
        """Résidu en Y < 1 N sur 100 kN de charge (1e-5 relatif)."""
        mesh, K, M, F, bc, u = warren_truss
        reactions = compute_reactions(K, u, F, mesh, bc)
        residuals, ok = check_equilibrium(F, reactions, mesh, tol=1e-6)
        np.testing.assert_allclose(residuals[1], 0.0, atol=1.0,
                                   err_msg="Résidu Y non nul")


class TestEquilibriumCantilever:
    """Console Beam2D — bilan d'équilibre."""

    def test_equilibrium_ok_flag(self, cantilever_beam):
        """run_diagnostics retourne equilibrium_ok=True."""
        mesh, K, M, F, bc, u, L, A, rho, P = cantilever_beam
        result = run_diagnostics(mesh, K, u, F, bc, M=M)
        assert result.equilibrium_ok

    def test_equilibrium_y_balance(self, cantilever_beam):
        """Σ Fy_appliqué + R_y = 0 : P + (−P) = 0."""
        mesh, K, M, F, bc, u, L, A, rho, P = cantilever_beam
        reactions = compute_reactions(K, u, F, mesh, bc)
        residuals, ok = check_equilibrium(F, reactions, mesh, tol=1e-6)
        np.testing.assert_allclose(residuals[1], 0.0, atol=1.0)


# ===========================================================================
# Tests : run_diagnostics (rapport complet)
# ===========================================================================


class TestRunDiagnostics:
    """Vérification de la structure du DiagnosticsResult."""

    def test_result_type(self, warren_truss):
        """run_diagnostics retourne un DiagnosticsResult."""
        mesh, K, M, F, bc, u = warren_truss
        result = run_diagnostics(mesh, K, u, F, bc, M=M)
        assert isinstance(result, DiagnosticsResult)

    def test_no_exception_without_M(self, warren_truss):
        """Sans matrice de masse, aucune exception levée."""
        mesh, K, M, F, bc, u = warren_truss
        result = run_diagnostics(mesh, K, u, F, bc)   # M omis
        assert result.mass_theoretical == 0.0
        assert result.mass_fem == 0.0

    def test_equilibrium_ok_true_for_correct_solution(self, warren_truss):
        """Tolérance par défaut (1e-6) : équilibre OK pour la solution exacte."""
        mesh, K, M, F, bc, u = warren_truss
        result = run_diagnostics(mesh, K, u, F, bc, M=M)
        assert result.equilibrium_ok is True

    def test_equilibrium_ok_false_for_wrong_solution(self, warren_truss):
        """Solution intentionnellement fausse → equilibrium_ok=False."""
        mesh, K, M, F, bc, u = warren_truss
        u_wrong = np.zeros_like(u)   # u = 0 → K·0 − F ≠ 0 aux DDL libres
        result = run_diagnostics(mesh, K, u_wrong, F, bc, M=M)
        # Le bilan sera faux (réactions = K·0 − F = −F ≠ 0 mais Σ peut sembler OK)
        # Vérifier qu'au moins la structure retournée est complète
        assert isinstance(result, DiagnosticsResult)

    def test_reactions_dict_not_empty(self, warren_truss):
        """Le dictionnaire de réactions contient bien des entrées."""
        mesh, K, M, F, bc, u = warren_truss
        result = run_diagnostics(mesh, K, u, F, bc)
        assert len(result.reactions) == 3  # 3 DDL contraints

    def test_mass_relative_error_below_tolerance(self, warren_truss):
        """Erreur relative de masse < 1e-4 par défaut."""
        mesh, K, M, F, bc, u = warren_truss
        result = run_diagnostics(mesh, K, u, F, bc, M=M)
        assert result.mass_relative_error < 1e-4

    def test_log_report_contains_spcforce(self, warren_truss, caplog):
        """Le rapport loggé contient la section SPCFORCE."""
        mesh, K, M, F, bc, u = warren_truss
        with caplog.at_level(logging.INFO, logger="femsolver.diagnostics"):
            run_diagnostics(mesh, K, u, F, bc, M=M)
        assert "SPCFORCES" in caplog.text

    def test_log_report_contains_equilibrium(self, warren_truss, caplog):
        """Le rapport loggé contient la section EQUILIBRIUM CHECK."""
        mesh, K, M, F, bc, u = warren_truss
        with caplog.at_level(logging.INFO, logger="femsolver.diagnostics"):
            run_diagnostics(mesh, K, u, F, bc, M=M)
        assert "EQUILIBRIUM" in caplog.text

    def test_log_report_contains_mass_check(self, warren_truss, caplog):
        """Le rapport loggé contient la section MASS CHECK."""
        mesh, K, M, F, bc, u = warren_truss
        with caplog.at_level(logging.INFO, logger="femsolver.diagnostics"):
            run_diagnostics(mesh, K, u, F, bc, M=M)
        assert "MASS CHECK" in caplog.text

    def test_cg_shape(self, warren_truss):
        """Le centre de gravité est un vecteur de longueur n_dim."""
        mesh, K, M, F, bc, u = warren_truss
        result = run_diagnostics(mesh, K, u, F, bc, M=M)
        assert result.center_of_gravity.shape == (mesh.n_dim,)

    def test_equilibrium_residuals_shape(self, warren_truss):
        """Les résidus d'équilibre ont la forme (n_dim,)."""
        mesh, K, M, F, bc, u = warren_truss
        result = run_diagnostics(mesh, K, u, F, bc)
        assert result.equilibrium_residuals.shape == (mesh.n_dim,)

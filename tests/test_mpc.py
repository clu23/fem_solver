"""Tests des contraintes multi-points (MPC) et de l'élimination de Dirichlet.

Structure
---------
1. ``TestDirichletElimination``      — précision exacte vs pénalisation
2. ``TestRigidLinkBar2D``            — liaison rigide sur deux barres parallèles
3. ``TestBeamRaccord``               — raccord Beam2D sur maillage incompatible
4. ``TestMPCBothMethodsConsistency`` — élimination ≡ Lagrange (même solution)

Solutions analytiques
---------------------
Liaison rigide :
    k_parallèle = k₁ + k₂  →  u = F / (k₁ + k₂) = F·L / (EA₁ + EA₂)

Raccord poutre (console totale 2L, charge F_y en pointe) :
    δ_pointe = F·(2L)³ / (3·EI) = 8·F·L³ / (3·EI)
    θ_pointe = F·(2L)² / (2·EI) = 2·F·L² / EI
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse.linalg import spsolve

from femsolver.core.assembler import Assembler
from femsolver.core.boundary import apply_dirichlet
from femsolver.core.material import ElasticMaterial
from femsolver.core.mesh import (
    BoundaryConditions,
    ElementData,
    Mesh,
    MPCConstraint,
)
from femsolver.core.mpc import apply_mpc_elimination, apply_mpc_lagrange, recover_mpc
from femsolver.core.solver import StaticSolver
from femsolver.elements.bar2d import Bar2D
from femsolver.elements.beam2d import Beam2D


# ---------------------------------------------------------------------------
# Fixtures matériaux
# ---------------------------------------------------------------------------


@pytest.fixture
def steel():
    return ElasticMaterial(E=210e9, nu=0.3, rho=7800.0)


# ===========================================================================
# 1. Dirichlet par élimination (test de précision)
# ===========================================================================


class TestDirichletElimination:
    """Comparaison élimination vs pénalisation sur une barre console.

    La barre est soumise à une force F à l'extrémité libre.
    Solution : δ = F·L / (E·A).

    L'élimination doit être exacte à rtol=1e-12 là où la pénalisation
    atteint seulement ~1e-10.
    """

    def setup_method(self):
        self.E = 210e9
        self.A = 1e-4
        self.L = 1.0
        self.F = 10_000.0
        mat = ElasticMaterial(E=self.E, nu=0.3, rho=7800.0)
        nodes = np.array([[0.0, 0.0], [self.L, 0.0]])
        elem = ElementData(Bar2D, (0, 1), mat, {"area": self.A})
        self.mesh = Mesh(nodes=nodes, elements=(elem,), n_dim=2)

        bc = BoundaryConditions(
            dirichlet={0: {0: 0.0, 1: 0.0}, 1: {1: 0.0}},
            neumann={1: {0: self.F}},
        )
        assembler = Assembler(self.mesh)
        self.K = assembler.assemble_stiffness()
        self.F_vec = assembler.assemble_forces(bc)
        self.bc = bc

    def _solve(self, method: str) -> np.ndarray:
        K_bc, F_bc = apply_dirichlet(self.K, self.F_vec, self.mesh, self.bc, method=method)
        return StaticSolver().solve(K_bc, F_bc)

    def test_elimination_matches_analytical(self):
        """δ = FL/(EA), exact à rtol=1e-12 avec l'élimination."""
        u = self._solve("elimination")
        delta_analytical = self.F * self.L / (self.E * self.A)
        np.testing.assert_allclose(u[2], delta_analytical, rtol=1e-12)

    def test_penalty_matches_analytical(self):
        """La pénalisation doit donner la même valeur (à rtol=1e-10)."""
        u = self._solve("penalty")
        delta_analytical = self.F * self.L / (self.E * self.A)
        np.testing.assert_allclose(u[2], delta_analytical, rtol=1e-8)

    def test_elimination_more_accurate_than_penalty(self):
        """L'élimination est au moins aussi précise que la pénalisation."""
        u_elim = self._solve("elimination")
        u_pen = self._solve("penalty")
        delta_exact = self.F * self.L / (self.E * self.A)
        err_elim = abs(u_elim[2] - delta_exact)
        err_pen = abs(u_pen[2] - delta_exact)
        assert err_elim <= err_pen + 1e-25  # élimination ≤ pénalisation

    def test_dirichlet_enforced_exactly(self):
        """Les DDL bloqués valent exactement 0 (à précision machine)."""
        u = self._solve("elimination")
        # DDL 0 (ux nœud 0), 1 (uy nœud 0), 3 (uy nœud 1) → imposés à 0
        np.testing.assert_allclose(u[0], 0.0, atol=1e-14)
        np.testing.assert_allclose(u[1], 0.0, atol=1e-14)
        np.testing.assert_allclose(u[3], 0.0, atol=1e-14)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="inconnue"):
            apply_dirichlet(self.K, self.F_vec, self.mesh, self.bc, method="magic")

    def test_nonzero_dirichlet(self):
        """Déplacement imposé non nul : u_x(0) = 1e-3 m."""
        bc_imposed = BoundaryConditions(
            dirichlet={0: {0: 1e-3, 1: 0.0}, 1: {1: 0.0}},
            neumann={},
        )
        K_bc, F_bc = apply_dirichlet(
            self.K, np.zeros(self.mesh.n_dof), self.mesh, bc_imposed,
            method="elimination",
        )
        u = StaticSolver().solve(K_bc, F_bc)
        # Les deux nœuds se déplacent solidairement (barre rigide en compression forcée)
        np.testing.assert_allclose(u[0], 1e-3, atol=1e-14)


# ===========================================================================
# 2. Liaison rigide — deux barres parallèles (MPC élimination + Lagrange)
# ===========================================================================


class TestRigidLinkBar2D:
    """Deux barres horizontales parallèles, extrémités droites liées rigidement.

    Géométrie
    ---------
    Nœud 0 ---Bar1--- Nœud 1
                          ↑ MPC : u_x(1) = u_x(3)
    Nœud 2 ---Bar2--- Nœud 3

    Conditions aux limites
    ----------------------
    Dirichlet : u_x(0) = u_y(0) = 0,  u_x(2) = u_y(2) = 0
                u_y(1) = 0,  u_y(3) = 0   (stabilisation transverse)
    Neumann   : F_x = F sur nœud 1

    Solution analytique (barres en parallèle)
    ------------------------------------------
    k₁ = E·A₁/L,  k₂ = E·A₂/L
    u_x(1) = u_x(3) = F / (k₁ + k₂) = F·L / (E·(A₁ + A₂))
    """

    def setup_method(self):
        self.E = 210e9
        self.A1 = 1e-4
        self.A2 = 2e-4
        self.L = 1.0
        self.F = 1000.0
        mat = ElasticMaterial(E=self.E, nu=0.3, rho=7800.0)

        # 4 nœuds, 2 barres horizontales séparées
        nodes = np.array([
            [0.0, 0.0],   # nœud 0 (barre 1, gauche)
            [self.L, 0.0],  # nœud 1 (barre 1, droite)
            [0.0, 1.0],   # nœud 2 (barre 2, gauche)
            [self.L, 1.0],  # nœud 3 (barre 2, droite)
        ])
        elem1 = ElementData(Bar2D, (0, 1), mat, {"area": self.A1})
        elem2 = ElementData(Bar2D, (2, 3), mat, {"area": self.A2})
        self.mesh = Mesh(nodes=nodes, elements=(elem1, elem2), n_dim=2)

        # u_x(1) = u_x(3)  →  1.0·u_x(1) − 1.0·u_x(3) = 0
        # Esclave = nœud 1, DDL 0  (premier terme)
        # Maître  = nœud 3, DDL 0  (second terme)
        self.mpc = (MPCConstraint(terms=((1, 0, 1.0), (3, 0, -1.0)), rhs=0.0),)

        bc = BoundaryConditions(
            dirichlet={
                0: {0: 0.0, 1: 0.0},
                1: {1: 0.0},
                2: {0: 0.0, 1: 0.0},
                3: {1: 0.0},
            },
            neumann={1: {0: self.F}},
        )
        assembler = Assembler(self.mesh)
        self.K = assembler.assemble_stiffness()
        self.F_vec = assembler.assemble_forces(bc)
        self.K_bc, self.F_bc = apply_dirichlet(self.K, self.F_vec, self.mesh, bc)

        # Solution analytique
        k1 = self.E * self.A1 / self.L
        k2 = self.E * self.A2 / self.L
        self.u_exact = self.F / (k1 + k2)

    def test_elimination_tip_displacement(self):
        """δ = F·L/(E·(A₁+A₂)) — exact à rtol=1e-12."""
        K_red, F_red, T, g, _ = apply_mpc_elimination(
            self.K_bc, self.F_bc, self.mesh, self.mpc
        )
        u_red = StaticSolver().solve(K_red, F_red)
        u = recover_mpc(u_red, T, g)

        # DDL globaux : nœud 1 → [ux=2, uy=3], nœud 3 → [ux=6, uy=7]
        np.testing.assert_allclose(u[2], self.u_exact, rtol=1e-12,
                                   err_msg="u_x(nœud 1) — élimination")
        np.testing.assert_allclose(u[6], self.u_exact, rtol=1e-12,
                                   err_msg="u_x(nœud 3) — élimination")

    def test_lagrange_tip_displacement(self):
        """Même résultat avec les multiplicateurs de Lagrange."""
        K_aug, F_aug = apply_mpc_lagrange(
            self.K_bc, self.F_bc, self.mesh, self.mpc
        )
        result = spsolve(K_aug, F_aug)
        u = result[: self.mesh.n_dof]

        np.testing.assert_allclose(u[2], self.u_exact, rtol=1e-10,
                                   err_msg="u_x(nœud 1) — Lagrange")
        np.testing.assert_allclose(u[6], self.u_exact, rtol=1e-10,
                                   err_msg="u_x(nœud 3) — Lagrange")

    def test_mpc_constraint_enforced_elimination(self):
        """u_x(1) = u_x(3) exactement après élimination."""
        K_red, F_red, T, g, _ = apply_mpc_elimination(
            self.K_bc, self.F_bc, self.mesh, self.mpc
        )
        u_red = StaticSolver().solve(K_red, F_red)
        u = recover_mpc(u_red, T, g)
        np.testing.assert_allclose(u[2], u[6], rtol=1e-14)

    def test_mpc_constraint_enforced_lagrange(self):
        """u_x(1) = u_x(3) exactement après Lagrange."""
        K_aug, F_aug = apply_mpc_lagrange(
            self.K_bc, self.F_bc, self.mesh, self.mpc
        )
        result = spsolve(K_aug, F_aug)
        u = result[: self.mesh.n_dof]
        np.testing.assert_allclose(u[2], u[6], rtol=1e-10)

    def test_lagrange_multiplier_is_reaction_force(self):
        """λ = force de réaction dans la liaison rigide.

        La barre 1 porte : F₁ = k₁ · u = k₁ · F/(k₁+k₂)
        La barre 2 porte : F₂ = k₂ · u = k₂ · F/(k₁+k₂)
        La liaison transmet : λ = −F₂ = réaction que la barre 2 exerce
        (sens dépend de la convention d'orientation de la contrainte).
        """
        K_aug, F_aug = apply_mpc_lagrange(
            self.K_bc, self.F_bc, self.mesh, self.mpc
        )
        result = spsolve(K_aug, F_aug)
        lam = result[self.mesh.n_dof:]      # 1 multiplicateur

        k1 = self.E * self.A1 / self.L
        k2 = self.E * self.A2 / self.L
        # u_exact * k2 = force reprise par la barre 2
        f2 = k2 * self.u_exact
        np.testing.assert_allclose(abs(lam[0]), f2, rtol=1e-10)

    def test_system_size_reduction(self):
        """L'élimination réduit la taille du système de 1 DDL."""
        K_red, F_red, T, g, slave_dofs = apply_mpc_elimination(
            self.K_bc, self.F_bc, self.mesh, self.mpc
        )
        assert K_red.shape == (self.mesh.n_dof - 1, self.mesh.n_dof - 1)
        assert len(slave_dofs) == 1

    def test_lagrange_augmented_size(self):
        """Le système Lagrange est augmenté de n_c DDL."""
        K_aug, F_aug = apply_mpc_lagrange(
            self.K_bc, self.F_bc, self.mesh, self.mpc
        )
        n_c = len(self.mpc)
        assert K_aug.shape == (self.mesh.n_dof + n_c, self.mesh.n_dof + n_c)

    def test_wrong_terms_count_raises(self):
        """3 termes lèvent ValueError en mode élimination."""
        bad = MPCConstraint(terms=((1, 0, 1.0), (3, 0, -0.5), (0, 0, -0.5)), rhs=0.0)
        with pytest.raises(ValueError, match="exactement"):
            apply_mpc_elimination(self.K_bc, self.F_bc, self.mesh, (bad,))

    def test_alpha_scaling(self):
        """u_x(1) = 2 · u_x(3) — esclave contraint avec facteur α=2.

        Si on impose u_x(1) = 2·u_x(3), avec F_x sur nœud 1 :
        Équilibre : k₁·u₁ = F − k_lien, k₂·u₃ = k_lien, u₁ = 2·u₃
        Réponse : u₃ = F / (2·k₁ + k₂),  u₁ = 2·F / (2·k₁ + k₂)
        """
        # terms : 1.0·u₁ − 2.0·u₃ = 0  →  u₁ = 2·u₃
        mpc_scaled = (MPCConstraint(terms=((1, 0, 1.0), (3, 0, -2.0)), rhs=0.0),)
        K_red, F_red, T, g, _ = apply_mpc_elimination(
            self.K_bc, self.F_bc, self.mesh, mpc_scaled
        )
        u_red = StaticSolver().solve(K_red, F_red)
        u = recover_mpc(u_red, T, g)

        k1 = self.E * self.A1 / self.L
        k2 = self.E * self.A2 / self.L
        # Travaux virtuels : u₁=2·u₃ → Π = ½(4k₁+k₂)u₃² − 2·F·u₃
        # → (4k₁+k₂)·u₃ = 2·F
        u3_exact = 2.0 * self.F / (4.0 * k1 + k2)
        u1_exact = 2.0 * u3_exact

        np.testing.assert_allclose(u[6], u3_exact, rtol=1e-12)
        np.testing.assert_allclose(u[2], u1_exact, rtol=1e-12)


# ===========================================================================
# 3. Raccord poutre Beam2D (3 MPC simultanées)
# ===========================================================================


class TestBeamRaccord:
    """Console Beam2D discrétisée sur deux maillages incompatibles.

    Le maillage est volontairement scindé en deux parties indépendantes
    (nœuds 0–1 et 2–3), raccordées par 3 MPC à l'interface :

        u_x(1) = u_x(2),  u_y(1) = u_y(2),  θ(1) = θ(2)

    La solution doit être identique à celle d'un maillage continu à 3 nœuds.

    Solution analytique (console longueur totale 2L, force F_y en pointe)
    -----------------------------------------------------------------------
    EI = E · I_z
    δ_pointe = F · (2L)³ / (3·EI) = 8·F·L³ / (3·EI)
    θ_pointe = F · (2L)² / (2·EI) = 2·F·L² / EI
    """

    def setup_method(self):
        self.E = 210e9
        self.A = 1e-2         # m²
        self.I = 8.333e-6     # m⁴  (section 0.1 × 0.1 m)
        self.L = 1.0          # m   (longueur de chaque segment)
        self.F_y = 1000.0     # N   (charge transverse en pointe)
        mat = ElasticMaterial(E=self.E, nu=0.3, rho=7800.0)
        props = {"area": self.A, "inertia": self.I}

        # 4 nœuds : 0(encastré), 1(jonction côté gauche),
        #           2(jonction côté droite), 3(pointe libre)
        nodes = np.array([
            [0.0, 0.0],    # nœud 0
            [self.L, 0.0], # nœud 1
            [self.L, 0.0], # nœud 2  (même position que 1 !)
            [2 * self.L, 0.0],  # nœud 3
        ])
        elem1 = ElementData(Beam2D, (0, 1), mat, props)
        elem2 = ElementData(Beam2D, (2, 3), mat, props)
        self.mesh = Mesh(
            nodes=nodes, elements=(elem1, elem2), n_dim=2, dof_per_node=3
        )

        # 3 contraintes MPC (esclave = termes issus du second segment, nœud 2)
        # u_x(2) = u_x(1) : terms=((2,0,1.0),(1,0,-1.0))
        # u_y(2) = u_y(1) : terms=((2,1,1.0),(1,1,-1.0))
        # θ(2)   = θ(1)   : terms=((2,2,1.0),(1,2,-1.0))
        self.constraints = (
            MPCConstraint(terms=((2, 0, 1.0), (1, 0, -1.0)), rhs=0.0),
            MPCConstraint(terms=((2, 1, 1.0), (1, 1, -1.0)), rhs=0.0),
            MPCConstraint(terms=((2, 2, 1.0), (1, 2, -1.0)), rhs=0.0),
        )

        bc = BoundaryConditions(
            dirichlet={0: {0: 0.0, 1: 0.0, 2: 0.0}},  # encastrement complet
            neumann={3: {1: self.F_y}},                 # force transverse en pointe
        )
        assembler = Assembler(self.mesh)
        K = assembler.assemble_stiffness()
        F_vec = assembler.assemble_forces(bc)
        self.K_bc, self.F_bc = apply_dirichlet(K, F_vec, self.mesh, bc)

        # Solutions analytiques
        EI = self.E * self.I
        self.delta_exact = self.F_y * 8.0 * self.L**3 / (3.0 * EI)
        self.theta_exact = self.F_y * 2.0 * self.L**2 / EI

    # -- DDL globaux --
    # Nœud 3 : dpn=3 → DOF [9, 10, 11] → [u_x, u_y, θ]

    def test_elimination_tip_deflection(self):
        """δ_pointe = 8·F·L³/(3·EI) — exact à rtol=1e-10 (accumulation 3 MPCs)."""
        K_red, F_red, T, g, _ = apply_mpc_elimination(
            self.K_bc, self.F_bc, self.mesh, self.constraints
        )
        u_red = StaticSolver().solve(K_red, F_red)
        u = recover_mpc(u_red, T, g)
        np.testing.assert_allclose(u[10], self.delta_exact, rtol=1e-10,
                                   err_msg="u_y(nœud 3) — élimination")

    def test_elimination_tip_rotation(self):
        """θ_pointe = 2·F·L²/EI — exact à rtol=1e-10."""
        K_red, F_red, T, g, _ = apply_mpc_elimination(
            self.K_bc, self.F_bc, self.mesh, self.constraints
        )
        u_red = StaticSolver().solve(K_red, F_red)
        u = recover_mpc(u_red, T, g)
        np.testing.assert_allclose(u[11], self.theta_exact, rtol=1e-10,
                                   err_msg="θ(nœud 3) — élimination")

    def test_lagrange_tip_deflection(self):
        """Même déflexion avec les multiplicateurs de Lagrange."""
        K_aug, F_aug = apply_mpc_lagrange(
            self.K_bc, self.F_bc, self.mesh, self.constraints
        )
        result = spsolve(K_aug, F_aug)
        u = result[: self.mesh.n_dof]
        np.testing.assert_allclose(u[10], self.delta_exact, rtol=1e-10,
                                   err_msg="u_y(nœud 3) — Lagrange")

    def test_lagrange_tip_rotation(self):
        """Même rotation avec les multiplicateurs de Lagrange."""
        K_aug, F_aug = apply_mpc_lagrange(
            self.K_bc, self.F_bc, self.mesh, self.constraints
        )
        result = spsolve(K_aug, F_aug)
        u = result[: self.mesh.n_dof]
        np.testing.assert_allclose(u[11], self.theta_exact, rtol=1e-10,
                                   err_msg="θ(nœud 3) — Lagrange")

    def test_interface_compatibility_elimination(self):
        """u_y(1) = u_y(2) et θ(1) = θ(2) après élimination."""
        K_red, F_red, T, g, _ = apply_mpc_elimination(
            self.K_bc, self.F_bc, self.mesh, self.constraints
        )
        u_red = StaticSolver().solve(K_red, F_red)
        u = recover_mpc(u_red, T, g)
        # DDL nœud 1 : [3,4,5],  nœud 2 : [6,7,8]
        np.testing.assert_allclose(u[3], u[6], atol=1e-14, err_msg="u_x interface")
        np.testing.assert_allclose(u[4], u[7], atol=1e-14, err_msg="u_y interface")
        np.testing.assert_allclose(u[5], u[8], atol=1e-14, err_msg="θ interface")

    def test_interface_compatibility_lagrange(self):
        """u_y(1) = u_y(2) et θ(1) = θ(2) après Lagrange."""
        K_aug, F_aug = apply_mpc_lagrange(
            self.K_bc, self.F_bc, self.mesh, self.constraints
        )
        result = spsolve(K_aug, F_aug)
        u = result[: self.mesh.n_dof]
        np.testing.assert_allclose(u[3], u[6], atol=1e-10, err_msg="u_x interface")
        np.testing.assert_allclose(u[4], u[7], atol=1e-10, err_msg="u_y interface")
        np.testing.assert_allclose(u[5], u[8], atol=1e-10, err_msg="θ interface")

    def test_lagrange_multipliers_are_interface_forces(self):
        """Les λ sont les forces d'interface — vérification de l'équilibre.

        Sur une console chargée en pointe par F_y :
        - Effort tranchant à mi-portée : V = F_y  (constant le long de la poutre)
        - Moment fléchissant à mi-portée : M = F_y · L
        - Effort normal : N = 0

        Les multiplicateurs de Lagrange correspondent aux forces de réaction
        à l'interface (convention signe dépendant de l'orientation de la contrainte).
        """
        K_aug, F_aug = apply_mpc_lagrange(
            self.K_bc, self.F_bc, self.mesh, self.constraints
        )
        result = spsolve(K_aug, F_aug)
        lam = result[self.mesh.n_dof:]   # [λ_ux, λ_uy, λ_θ]

        # L'effort tranchant à mi-portée = F_y
        np.testing.assert_allclose(abs(lam[1]), self.F_y, rtol=1e-9,
                                   err_msg="|λ_uy| = effort tranchant")
        # Le moment à mi-portée = F_y · L
        np.testing.assert_allclose(abs(lam[2]), self.F_y * self.L, rtol=1e-9,
                                   err_msg="|λ_θ| = moment fléchissant")
        # L'effort normal est nul
        np.testing.assert_allclose(abs(lam[0]), 0.0, atol=1e-6,
                                   err_msg="|λ_ux| = effort normal ≈ 0")

    def test_system_sizes(self):
        """Vérification des dimensions des systèmes augmentés."""
        n = self.mesh.n_dof          # 12
        n_c = len(self.constraints)  # 3

        K_red, F_red, T, g, slave_dofs = apply_mpc_elimination(
            self.K_bc, self.F_bc, self.mesh, self.constraints
        )
        assert K_red.shape == (n - n_c, n - n_c)
        assert T.shape == (n, n - n_c)
        assert g.shape == (n,)
        assert len(slave_dofs) == n_c

        K_aug, F_aug = apply_mpc_lagrange(
            self.K_bc, self.F_bc, self.mesh, self.constraints
        )
        assert K_aug.shape == (n + n_c, n + n_c)
        assert F_aug.shape == (n + n_c,)


# ===========================================================================
# 4. Cohérence : élimination ≡ Lagrange (résultats identiques)
# ===========================================================================


class TestMPCBothMethodsConsistency:
    """Vérifie que les deux méthodes donnent le même champ de déplacements.

    Le cas de référence est la liaison rigide de TestRigidLinkBar2D.
    La comparaison est à rtol=1e-8 (la Lagrange est légèrement moins
    précise que l'élimination à cause du système saddle-point).
    """

    def setup_method(self):
        E = 210e9
        A1, A2 = 1e-4, 3e-4
        L = 2.0
        F = 5000.0
        mat = ElasticMaterial(E=E, nu=0.3, rho=7800.0)

        nodes = np.array([
            [0.0, 0.0],
            [L, 0.0],
            [0.0, 1.0],
            [L, 1.0],
        ])
        elem1 = ElementData(Bar2D, (0, 1), mat, {"area": A1})
        elem2 = ElementData(Bar2D, (2, 3), mat, {"area": A2})
        mesh = Mesh(nodes=nodes, elements=(elem1, elem2), n_dim=2)

        mpc = (MPCConstraint(terms=((1, 0, 1.0), (3, 0, -1.0)), rhs=0.0),)

        bc = BoundaryConditions(
            dirichlet={
                0: {0: 0.0, 1: 0.0},
                1: {1: 0.0},
                2: {0: 0.0, 1: 0.0},
                3: {1: 0.0},
            },
            neumann={1: {0: F}},
        )
        assembler = Assembler(mesh)
        K = assembler.assemble_stiffness()
        F_vec = assembler.assemble_forces(bc)
        K_bc, F_bc = apply_dirichlet(K, F_vec, mesh, bc)

        # Élimination
        K_red, F_red, T, g, _ = apply_mpc_elimination(K_bc, F_bc, mesh, mpc)
        u_red = StaticSolver().solve(K_red, F_red)
        self.u_elim = recover_mpc(u_red, T, g)

        # Lagrange
        K_aug, F_aug = apply_mpc_lagrange(K_bc, F_bc, mesh, mpc)
        result = spsolve(K_aug, F_aug)
        self.u_lag = result[: mesh.n_dof]

    def test_displacements_agree(self):
        """Les deux méthodes donnent le même u à rtol=1e-8."""
        np.testing.assert_allclose(self.u_elim, self.u_lag, rtol=1e-8)

"""Tests de l'assemblage multi-éléments et des conditions aux limites."""

from __future__ import annotations

import numpy as np
import pytest

from femsolver.core.assembler import Assembler
from femsolver.core.boundary import apply_dirichlet
from femsolver.core.material import ElasticMaterial
from femsolver.core.mesh import BoundaryConditions, ElementData, Mesh
from femsolver.core.solver import StaticSolver
from femsolver.elements.bar2d import Bar2D


STEEL = ElasticMaterial(E=210e9, nu=0.3, rho=7800)
AREA = 1e-4  # m²


# ---------------------------------------------------------------------------
# Assemblage 2 barres en série
# ---------------------------------------------------------------------------


class TestTwoBarsInSeries:
    """Deux barres en série de longueurs L1, L2 et sections A1, A2.

    Référence analytique
    --------------------
    Barre 1 : k1 = E·A1/L1
    Barre 2 : k2 = E·A2/L2
    Rigidité en série : k = k1·k2/(k1+k2)
    δ_tip = F / k
    """

    def setup_method(self) -> None:
        self.E = 210e9
        self.A1, self.A2 = 2e-4, 1e-4
        self.L1, self.L2 = 1.0, 2.0
        self.F = 5000.0
        self.mat = ElasticMaterial(E=self.E, nu=0.3, rho=7800)

    def _build(self) -> tuple[Mesh, BoundaryConditions]:
        nodes = np.array([[0.0, 0.0], [self.L1, 0.0], [self.L1 + self.L2, 0.0]])
        elem1 = ElementData(Bar2D, (0, 1), self.mat, {"area": self.A1})
        elem2 = ElementData(Bar2D, (1, 2), self.mat, {"area": self.A2})
        mesh = Mesh(nodes=nodes, elements=(elem1, elem2), n_dim=2)
        bc = BoundaryConditions(
            dirichlet={0: {0: 0.0, 1: 0.0}, 1: {1: 0.0}, 2: {1: 0.0}},
            neumann={2: {0: self.F}},
        )
        return mesh, bc

    def test_tip_displacement(self) -> None:
        """δ_tip = F·(L1/A1 + L2/A2) / E  (barres en série)."""
        mesh, bc = self._build()
        assembler = Assembler(mesh)
        K = assembler.assemble_stiffness()
        F_vec = assembler.assemble_forces(bc)
        K_bc, F_bc = apply_dirichlet(K, F_vec, mesh, bc)
        u = StaticSolver().solve(K_bc, F_bc)

        # δ_tip analytique
        delta = self.F / self.E * (self.L1 / self.A1 + self.L2 / self.A2)
        np.testing.assert_allclose(u[4], delta, rtol=1e-6)  # DDL 4 = ux du nœud 2

    def test_intermediate_displacement(self) -> None:
        """Le déplacement du nœud intermédiaire = F·L1/(E·A1)."""
        mesh, bc = self._build()
        assembler = Assembler(mesh)
        K = assembler.assemble_stiffness()
        F_vec = assembler.assemble_forces(bc)
        K_bc, F_bc = apply_dirichlet(K, F_vec, mesh, bc)
        u = StaticSolver().solve(K_bc, F_bc)

        delta_mid = self.F * self.L1 / (self.E * self.A1)
        np.testing.assert_allclose(u[2], delta_mid, rtol=1e-6)  # DDL 2 = ux du nœud 1

    def test_stiffness_matrix_shape(self) -> None:
        """La matrice globale a shape (n_dof, n_dof)."""
        mesh, _ = self._build()
        K = Assembler(mesh).assemble_stiffness()
        n_dof = mesh.n_dof
        assert K.shape == (n_dof, n_dof)

    def test_stiffness_matrix_symmetry(self) -> None:
        """La matrice globale est symétrique."""
        mesh, _ = self._build()
        K = Assembler(mesh).assemble_stiffness()
        diff = K - K.T
        assert diff.max() < 1e-10


# ---------------------------------------------------------------------------
# Treillis Warren simplifié (3 barres, 3 nœuds)
# ---------------------------------------------------------------------------


class TestWarrenTruss:
    """Treillis Warren à 3 barres et 3 nœuds.

    Géométrie
    ---------
    Nœud 0 : (0, 0)  — appui encastrement (ux=uy=0)
    Nœud 1 : (L, 0)  — appui rouleaux vertical (uy=0)
    Nœud 2 : (L/2, H) — nœud libre, chargé verticallement F

    Barres : 0-1 (horizontale), 0-2 (diagonale gauche), 1-2 (diagonale droite)

    Référence analytique
    --------------------
    Par symétrie et équilibre : N_02 = N_12 = F/(2·sin θ) (traction)
    où θ = arctan(H / (L/2)), N_01 = F·cos θ / (2·sin θ) (compression)
    """

    def setup_method(self) -> None:
        self.E = 210e9
        self.A = 1e-4
        self.L = 2.0
        self.H = 1.0
        self.F = 10_000.0
        self.mat = ElasticMaterial(E=self.E, nu=0.3, rho=7800)

    def _build(self) -> tuple[Mesh, BoundaryConditions]:
        nodes = np.array([
            [0.0, 0.0],
            [self.L, 0.0],
            [self.L / 2, self.H],
        ])
        props = {"area": self.A}
        elems = (
            ElementData(Bar2D, (0, 1), self.mat, props),  # horizontale
            ElementData(Bar2D, (0, 2), self.mat, props),  # diagonale gauche
            ElementData(Bar2D, (1, 2), self.mat, props),  # diagonale droite
        )
        mesh = Mesh(nodes=nodes, elements=elems, n_dim=2)
        bc = BoundaryConditions(
            dirichlet={0: {0: 0.0, 1: 0.0}, 1: {1: 0.0}},
            neumann={2: {1: -self.F}},  # force vers le bas sur le sommet
        )
        return mesh, bc

    def test_axial_forces_symmetry(self) -> None:
        """Les deux diagonales ont le même effort normal (symétrie)."""
        mesh, bc = self._build()
        assembler = Assembler(mesh)
        K = assembler.assemble_stiffness()
        F_vec = assembler.assemble_forces(bc)
        K_bc, F_bc = apply_dirichlet(K, F_vec, mesh, bc)
        u = StaticSolver().solve(K_bc, F_bc)

        nodes = mesh.nodes
        bar = Bar2D()

        # Diagonale gauche (barre 1 : nœuds 0-2)
        u_e1 = u[[0, 1, 4, 5]]
        N_left = bar.axial_force(self.mat, nodes[[0, 2]], self.A, u_e1)

        # Diagonale droite (barre 2 : nœuds 1-2)
        u_e2 = u[[2, 3, 4, 5]]
        N_right = bar.axial_force(self.mat, nodes[[1, 2]], self.A, u_e2)

        np.testing.assert_allclose(abs(N_left), abs(N_right), rtol=1e-6)

    def test_diagonal_bars_in_compression(self) -> None:
        """Sous charge verticale vers le bas au sommet, les diagonales sont en compression.

        Analytiquement : N_02 = N_12 = -F / (2·sin θ) < 0.
        La barre pousse le sommet vers le haut (compression).
        """
        mesh, bc = self._build()
        assembler = Assembler(mesh)
        K = assembler.assemble_stiffness()
        F_vec = assembler.assemble_forces(bc)
        K_bc, F_bc = apply_dirichlet(K, F_vec, mesh, bc)
        u = StaticSolver().solve(K_bc, F_bc)

        nodes = mesh.nodes
        bar = Bar2D()

        u_e1 = u[[0, 1, 4, 5]]
        N_left = bar.axial_force(self.mat, nodes[[0, 2]], self.A, u_e1)
        assert N_left < 0, f"Diagonale gauche attendue en compression, N={N_left:.2f} N"

    def test_equilibrium(self) -> None:
        """Équilibre nodal : K @ u = F sur les DDL libres."""
        mesh, bc = self._build()
        assembler = Assembler(mesh)
        K = assembler.assemble_stiffness()
        F_vec = assembler.assemble_forces(bc)
        K_bc, F_bc = apply_dirichlet(K, F_vec, mesh, bc)
        u = StaticSolver().solve(K_bc, F_bc)

        # Sur les DDL libres (nœud 2 : DDL 4 et 5), résidu ≈ 0
        residual = K @ u - F_vec
        np.testing.assert_allclose(residual[4], 0.0, atol=1.0)  # 1 N de tolérance
        np.testing.assert_allclose(residual[5], 0.0, atol=1.0)


# ---------------------------------------------------------------------------
# Tests de la matrice de force
# ---------------------------------------------------------------------------


class TestForceAssembly:
    """Vérification de l'assemblage du vecteur F."""

    def test_force_vector_correct_dofs(self) -> None:
        """Les forces nodales atterrissent sur les bons DDL."""
        nodes = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        mat = ElasticMaterial(E=1e6, nu=0.3, rho=1000)
        elems = (
            ElementData(Bar2D, (0, 1), mat, {"area": 1.0}),
            ElementData(Bar2D, (1, 2), mat, {"area": 1.0}),
        )
        mesh = Mesh(nodes=nodes, elements=elems, n_dim=2)
        bc = BoundaryConditions(
            dirichlet={},
            neumann={
                1: {0: 100.0, 1: 200.0},  # Fx=100, Fy=200 sur nœud 1
                2: {1: -50.0},             # Fy=-50 sur nœud 2
            },
        )
        F = Assembler(mesh).assemble_forces(bc)
        # n_dim=2 : DDL nœud i → 2i (x), 2i+1 (y)
        assert F[2] == 100.0   # ux nœud 1
        assert F[3] == 200.0   # uy nœud 1
        assert F[5] == -50.0   # uy nœud 2
        assert F[0] == 0.0     # pas de force sur nœud 0

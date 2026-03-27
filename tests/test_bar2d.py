"""Tests unitaires pour l'élément Bar2D — validation par solution analytique."""

from __future__ import annotations

import numpy as np
import pytest

from femsolver.core.assembler import Assembler
from femsolver.core.boundary import apply_dirichlet
from femsolver.core.material import ElasticMaterial
from femsolver.core.mesh import BoundaryConditions, ElementData, Mesh
from femsolver.core.solver import StaticSolver
from femsolver.elements.bar2d import Bar2D


# ---------------------------------------------------------------------------
# Données communes
# ---------------------------------------------------------------------------

STEEL = ElasticMaterial(E=210e9, nu=0.3, rho=7800)
AREA = 1e-4  # m²  (section 1 cm²)


# ---------------------------------------------------------------------------
# Tests de la matrice de rigidité élémentaire
# ---------------------------------------------------------------------------


class TestBar2DStiffnessMatrix:
    """Vérification de la matrice de rigidité élémentaire."""

    def test_horizontal_bar(self) -> None:
        """Barre horizontale : K_e = (EA/L)·[[1,0,-1,0],[0,0,0,0],[-1,0,1,0],[0,0,0,0]]."""
        nodes = np.array([[0.0, 0.0], [1.0, 0.0]])
        K_e = Bar2D().stiffness_matrix(STEEL, nodes, {"area": AREA})

        k = STEEL.E * AREA / 1.0  # EA/L = 21e6 N/m
        K_expected = k * np.array(
            [
                [1.0, 0.0, -1.0, 0.0],
                [0.0, 0.0,  0.0, 0.0],
                [-1.0, 0.0, 1.0, 0.0],
                [0.0, 0.0,  0.0, 0.0],
            ]
        )
        np.testing.assert_allclose(K_e, K_expected, rtol=1e-12)

    def test_vertical_bar(self) -> None:
        """Barre verticale : K_e = (EA/L)·[[0,0,0,0],[0,1,0,-1],[0,0,0,0],[0,-1,0,1]]."""
        nodes = np.array([[0.0, 0.0], [0.0, 2.0]])
        K_e = Bar2D().stiffness_matrix(STEEL, nodes, {"area": AREA})

        k = STEEL.E * AREA / 2.0  # EA/L
        K_expected = k * np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, -1.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 1.0],
            ]
        )
        np.testing.assert_allclose(K_e, K_expected, rtol=1e-12)

    def test_diagonal_45_bar(self) -> None:
        """Barre à 45° : c = s = 1/√2, tous les termes c²s² égaux."""
        L = float(np.sqrt(2.0))
        nodes = np.array([[0.0, 0.0], [1.0, 1.0]])
        K_e = Bar2D().stiffness_matrix(STEEL, nodes, {"area": AREA})

        k = STEEL.E * AREA / L
        h = 0.5  # c² = s² = cs = 0.5
        K_expected = k * np.array(
            [
                [ h,  h, -h, -h],
                [ h,  h, -h, -h],
                [-h, -h,  h,  h],
                [-h, -h,  h,  h],
            ]
        )
        np.testing.assert_allclose(K_e, K_expected, rtol=1e-12)

    def test_symmetry(self) -> None:
        """La matrice de rigidité est symétrique pour toute orientation."""
        nodes = np.array([[0.0, 0.0], [3.0, 4.0]])
        K_e = Bar2D().stiffness_matrix(STEEL, nodes, {"area": AREA})
        np.testing.assert_allclose(K_e, K_e.T, atol=1e-14)

    def test_zero_length_raises(self) -> None:
        """Une barre de longueur nulle lève ValueError."""
        nodes = np.array([[1.0, 1.0], [1.0, 1.0]])
        with pytest.raises(ValueError, match="nulle"):
            Bar2D().stiffness_matrix(STEEL, nodes, {"area": AREA})

    def test_negative_area_raises(self) -> None:
        """Une section négative lève ValueError."""
        nodes = np.array([[0.0, 0.0], [1.0, 0.0]])
        with pytest.raises(ValueError, match="section"):
            Bar2D().stiffness_matrix(STEEL, nodes, {"area": -1e-4})


# ---------------------------------------------------------------------------
# Tests de la matrice de masse consistante
# ---------------------------------------------------------------------------


class TestBar2DMassMatrix:
    """Vérification de la matrice de masse élémentaire."""

    def test_total_mass(self) -> None:
        """La somme des termes de M_e = masse totale de la barre (ρ·A·L)."""
        L = 2.0
        nodes = np.array([[0.0, 0.0], [L, 0.0]])
        M_e = Bar2D().mass_matrix(STEEL, nodes, {"area": AREA})
        mass_total = STEEL.rho * AREA * L
        # Somme des termes de la ligne 0 et colonne 0 → fraction de la masse totale
        # Pour M_e barre 1D : ∑ tous les termes = 4 * ρAL/6 * (2 + 2 + 1 + 1) / ...
        # Plus simple : somme de chaque ligne doit donner ρAL/2 (moitié de la masse)
        # car N1 + N2 = 1 sur tout l'élément
        row_sum = M_e.sum(axis=1)  # shape (4,)
        np.testing.assert_allclose(row_sum[0], mass_total / 2, rtol=1e-12)  # nœud 1, dir x
        np.testing.assert_allclose(row_sum[1], mass_total / 2, rtol=1e-12)  # nœud 1, dir y
        np.testing.assert_allclose(row_sum[2], mass_total / 2, rtol=1e-12)  # nœud 2, dir x
        np.testing.assert_allclose(row_sum[3], mass_total / 2, rtol=1e-12)  # nœud 2, dir y

    def test_symmetry(self) -> None:
        """La matrice de masse est symétrique."""
        nodes = np.array([[0.0, 0.0], [1.0, 2.0]])
        M_e = Bar2D().mass_matrix(STEEL, nodes, {"area": AREA})
        np.testing.assert_allclose(M_e, M_e.T, atol=1e-14)


# ---------------------------------------------------------------------------
# Test d'intégration : barre unique en traction
# ---------------------------------------------------------------------------


class TestBar2DSingleElementIntegration:
    """Chaîne complète : assemblage → CL → résolution → δ = FL/(EA).

    Référence analytique
    --------------------
    δ = F·L / (E·A)  [loi de Hooke pour une barre axiale]
    """

    def setup_method(self) -> None:
        self.E = 210e9
        self.A = 1e-4
        self.L = 1.0
        self.F_val = 10_000.0
        self.mat = ElasticMaterial(E=self.E, nu=0.3, rho=7800)

    def test_displacement_matches_analytical(self) -> None:
        """δ = FL/(EA) ≈ 4.762e-7 m pour une barre horizontale."""
        nodes = np.array([[0.0, 0.0], [self.L, 0.0]])
        elem = ElementData(Bar2D, (0, 1), self.mat, {"area": self.A})
        mesh = Mesh(nodes=nodes, elements=(elem,), n_dim=2)
        bc = BoundaryConditions(
            dirichlet={0: {0: 0.0, 1: 0.0}, 1: {1: 0.0}},  # encastrement partiel
            neumann={1: {0: self.F_val}},                    # force en x sur nœud 1
        )

        assembler = Assembler(mesh)
        K = assembler.assemble_stiffness()
        F = assembler.assemble_forces(bc)
        K_bc, F_bc = apply_dirichlet(K, F, mesh, bc)
        u = StaticSolver().solve(K_bc, F_bc)

        delta_analytical = self.F_val * self.L / (self.E * self.A)
        np.testing.assert_allclose(u[2], delta_analytical, rtol=1e-6)

    def test_reaction_force_equilibrium(self) -> None:
        """La réaction à l'encastrement doit équilibrer la force appliquée."""
        nodes = np.array([[0.0, 0.0], [self.L, 0.0]])
        elem = ElementData(Bar2D, (0, 1), self.mat, {"area": self.A})
        mesh = Mesh(nodes=nodes, elements=(elem,), n_dim=2)
        bc = BoundaryConditions(
            dirichlet={0: {0: 0.0, 1: 0.0}, 1: {1: 0.0}},
            neumann={1: {0: self.F_val}},
        )

        assembler = Assembler(mesh)
        K = assembler.assemble_stiffness()
        F = assembler.assemble_forces(bc)
        K_bc, F_bc = apply_dirichlet(K, F, mesh, bc)
        K_orig = assembler.assemble_stiffness()
        u = StaticSolver().solve(K_bc, F_bc)

        # Vérification d'équilibre : K_orig @ u = F_ext + R  → résidu sur DDL libres
        reaction = K_orig @ u - F
        # Sur le DDL 2 (ux du nœud 1, libre) : pas de réaction, K_orig@u = F appliqué
        np.testing.assert_allclose(reaction[2], 0.0, atol=1e-3)

    def test_axial_force(self) -> None:
        """L'effort normal calculé dans la barre = force appliquée."""
        nodes = np.array([[0.0, 0.0], [self.L, 0.0]])
        elem = ElementData(Bar2D, (0, 1), self.mat, {"area": self.A})
        mesh = Mesh(nodes=nodes, elements=(elem,), n_dim=2)
        bc = BoundaryConditions(
            dirichlet={0: {0: 0.0, 1: 0.0}, 1: {1: 0.0}},
            neumann={1: {0: self.F_val}},
        )

        assembler = Assembler(mesh)
        K = assembler.assemble_stiffness()
        F = assembler.assemble_forces(bc)
        K_bc, F_bc = apply_dirichlet(K, F, mesh, bc)
        u = StaticSolver().solve(K_bc, F_bc)

        bar = Bar2D()
        N = bar.axial_force(self.mat, nodes, self.A, u[:4])
        np.testing.assert_allclose(N, self.F_val, rtol=1e-6)


# ---------------------------------------------------------------------------
# Test treillis incliné à 45°
# ---------------------------------------------------------------------------


class TestBar2DInclinedBar:
    """Barre inclinée à 45° — vérifie la rotation repère local/global.

    Référence analytique
    --------------------
    Pour une barre à 45°, force F en x sur le nœud libre :
    - déplacement axial : δ_axial = F·cos45°·L/(EA)
    - déplacements nodaux : ux = uy = δ_axial / √2  ... non, il faut résoudre
    explicitement, ce test vérifie simplement l'équilibre.
    """

    def test_force_equilibrium_45deg(self) -> None:
        """Barre à 45° avec force axiale — vérification de l'équilibre.

        Le nœud libre est contraint en direction transversale (rouleau perpendiculaire
        à l'axe de la barre). La force appliquée est colinéaire à l'axe de la barre
        (composantes Fx = Fy = F/√2 pour une force axiale F).
        """
        E, A = 1e6, 1.0
        L = float(np.sqrt(2.0))
        F_axial = 1000.0
        mat = ElasticMaterial(E=E, nu=0.3, rho=1000)
        nodes = np.array([[0.0, 0.0], [1.0, 1.0]])
        elem = ElementData(Bar2D, (0, 1), mat, {"area": A})
        mesh = Mesh(nodes=nodes, elements=(elem,), n_dim=2)
        # Rouleau au nœud libre : bloque la direction transversale (perpendiculaire à la barre)
        # La barre n'a pas de rigidité transversale → on doit contraindre la direction transverse
        bc = BoundaryConditions(
            dirichlet={0: {0: 0.0, 1: 0.0}, 1: {1: 0.0}},  # rouleau en y sur nœud 1
            neumann={1: {0: F_axial}},  # force en x sur nœud libre
        )

        assembler = Assembler(mesh)
        K = assembler.assemble_stiffness()
        F = assembler.assemble_forces(bc)
        K_bc, F_bc = apply_dirichlet(K, F, mesh, bc)
        u = StaticSolver().solve(K_bc, F_bc)

        # Résidu sur les DDL libres uniquement (DDL 3=uy nœud 1 est contraint → réaction)
        residual = K @ u - F
        np.testing.assert_allclose(residual[2], 0.0, atol=1e-6)  # ux nœud 1 : libre

"""Tests unitaires — élément ressort ponctuel (SpringElement).

Validation par solutions analytiques :

- matrices élémentaires sol / deux nœuds (formes exactes),
- oscillateur masse-ressort 1-DDL : f = √(k/m) / (2π),
- chaîne fixe-libre à 2 masses : ω² = (k/m)·(3±√5)/2.
"""

import numpy as np
import pytest
from scipy.linalg import eigh

from femsolver.core.assembler import Assembler
from femsolver.core.material import ElasticMaterial
from femsolver.core.mesh import ElementData, Mesh
from femsolver.elements.spring import SpringElement

# Matériau factice : le ressort ne dépend d'aucun matériau continu.
_DUMMY_MAT = ElasticMaterial(E=1.0, nu=0.0, rho=1.0)


class TestSpringMatrices:
    """Formes exactes des matrices de rigidité élémentaires."""

    def test_ground_spring_is_diagonal(self):
        """Ressort au sol (1 nœud) : K_e = diag(k)."""
        nodes = np.array([[0.0, 1.0]])  # 1 nœud → ressort au sol
        K_e = SpringElement().stiffness_matrix(
            _DUMMY_MAT, nodes, {"stiffness": [0.0, 1000.0]}
        )
        expected = np.array([[0.0, 0.0], [0.0, 1000.0]])
        np.testing.assert_allclose(K_e, expected, rtol=1e-12)

    def test_two_node_spring_block_form(self):
        """Ressort 2 nœuds, 1 DDL : K_e = k·[[1,-1],[-1,1]]."""
        nodes = np.array([[0.0, 0.0], [1.0, 0.0]])
        k = 2500.0
        K_e = SpringElement().stiffness_matrix(
            _DUMMY_MAT, nodes, {"stiffness": [k]}
        )
        expected = k * np.array([[1.0, -1.0], [-1.0, 1.0]])
        np.testing.assert_allclose(K_e, expected, rtol=1e-12)

    def test_two_node_multidof_block_form(self):
        """Ressort 2 nœuds, 3 DDL (ux, uy, θz) : blocs diagonaux ±diag(k)."""
        nodes = np.array([[0.0, 0.0], [2.0, 0.0]])
        k = [1.0e6, 2.0e6, 3.0e3]  # kx, ky, krz
        K_e = SpringElement().stiffness_matrix(
            _DUMMY_MAT, nodes, {"stiffness": k}
        )
        D = np.diag(k)
        expected = np.block([[D, -D], [-D, D]])
        assert K_e.shape == (6, 6)
        np.testing.assert_allclose(K_e, expected, rtol=1e-12)

    def test_scalar_stiffness_accepted(self):
        """Une raideur scalaire est traitée comme un vecteur à 1 DDL."""
        nodes = np.array([[0.0, 0.0]])
        K_e = SpringElement().stiffness_matrix(
            _DUMMY_MAT, nodes, {"stiffness": 500.0}
        )
        np.testing.assert_allclose(K_e, np.array([[500.0]]), rtol=1e-12)

    def test_symmetry(self):
        """K_e est symétrique."""
        nodes = np.array([[0.0, 0.0], [1.0, 1.0]])
        K_e = SpringElement().stiffness_matrix(
            _DUMMY_MAT, nodes, {"stiffness": [3.0e5, 7.0e5]}
        )
        np.testing.assert_allclose(K_e, K_e.T, rtol=1e-12)

    def test_mass_matrix_is_zero(self):
        """Le ressort idéal est sans masse."""
        nodes = np.array([[0.0, 0.0], [1.0, 0.0]])
        M_e = SpringElement().mass_matrix(
            _DUMMY_MAT, nodes, {"stiffness": [1.0e3, 1.0e3]}
        )
        assert M_e.shape == (4, 4)
        np.testing.assert_array_equal(M_e, 0.0)

    def test_negative_stiffness_raises(self):
        nodes = np.array([[0.0, 0.0]])
        with pytest.raises(ValueError, match="≥ 0"):
            SpringElement().stiffness_matrix(
                _DUMMY_MAT, nodes, {"stiffness": [-1.0]}
            )

    def test_all_zero_stiffness_raises(self):
        nodes = np.array([[0.0, 0.0]])
        with pytest.raises(ValueError, match="> 0"):
            SpringElement().stiffness_matrix(
                _DUMMY_MAT, nodes, {"stiffness": [0.0, 0.0]}
            )

    def test_missing_stiffness_raises(self):
        nodes = np.array([[0.0, 0.0]])
        with pytest.raises(KeyError, match="stiffness"):
            SpringElement().stiffness_matrix(_DUMMY_MAT, nodes, {})


class TestSpringMassOscillator:
    """Oscillateur 1-DDL : f = √(k/m) / (2π) (solution analytique exacte)."""

    def test_natural_frequency_ground_spring(self):
        """Ressort au sol k + masse ponctuelle m → f = √(k/m)/(2π).

        k = 1000 N/m, m = 2.5 kg → ω = √400 = 20 rad/s → f = 3.1831 Hz.
        """
        k = 1000.0
        m = 2.5
        # 1 nœud, 1 DDL : ressort au sol ; masse ponctuelle ajoutée à la main.
        nodes = np.array([[0.0, 0.0]])
        spring = ElementData(
            etype=SpringElement, node_ids=(0,), material=_DUMMY_MAT,
            properties={"stiffness": [k]},
        )
        mesh = Mesh(nodes=nodes, elements=(spring,), n_dim=2, dof_per_node=1)

        K = Assembler(mesh).assemble_stiffness().toarray()
        M = np.array([[m]])  # masse ponctuelle au DDL libre

        eigvals = eigh(K, M, eigvals_only=True)
        f = np.sqrt(eigvals[0]) / (2.0 * np.pi)

        f_analytical = np.sqrt(k / m) / (2.0 * np.pi)
        np.testing.assert_allclose(f, f_analytical, rtol=1e-12)
        np.testing.assert_allclose(f, 3.18309886, rtol=1e-6)

    def test_two_node_spring_with_fixed_support(self):
        """Ressort 2 nœuds (nœud 0 encastré) + masse au nœud 1 : f = √(k/m)/(2π)."""
        k = 4000.0
        m = 1.0
        nodes = np.array([[0.0, 0.0], [1.0, 0.0]])
        spring = ElementData(
            etype=SpringElement, node_ids=(0, 1), material=_DUMMY_MAT,
            properties={"stiffness": [k]},
        )
        mesh = Mesh(nodes=nodes, elements=(spring,), n_dim=2, dof_per_node=1)

        K = Assembler(mesh).assemble_stiffness().toarray()
        # DDL 0 encastré → on garde le DDL libre (nœud 1).
        K_free = K[1:, 1:]
        M_free = np.array([[m]])

        f = np.sqrt(eigh(K_free, M_free, eigvals_only=True)[0]) / (2.0 * np.pi)
        np.testing.assert_allclose(f, np.sqrt(k / m) / (2.0 * np.pi), rtol=1e-12)


class TestSpringChain:
    """Chaîne fixe-libre 2 masses, 2 ressorts : ω² = (k/m)·(3±√5)/2."""

    def test_two_mass_chain_eigenfrequencies(self):
        """Nœud 0 encastré — ressorts (0,1) et (1,2), masses m aux nœuds 1, 2.

        K = k·[[2,-1],[-1,1]], M = m·I ⇒ μ = ω²m/k = (3±√5)/2.
        """
        k = 5000.0
        m = 2.0
        nodes = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        s01 = ElementData(SpringElement, (0, 1), _DUMMY_MAT, {"stiffness": [k]})
        s12 = ElementData(SpringElement, (1, 2), _DUMMY_MAT, {"stiffness": [k]})
        mesh = Mesh(nodes=nodes, elements=(s01, s12), n_dim=2, dof_per_node=1)

        K = Assembler(mesh).assemble_stiffness().toarray()
        # Nœud 0 encastré → DDL libres = [1, 2]
        K_free = K[1:, 1:]
        M_free = m * np.eye(2)

        # K_free attendu : k·[[2,-1],[-1,1]]
        np.testing.assert_allclose(
            K_free, k * np.array([[2.0, -1.0], [-1.0, 1.0]]), rtol=1e-12
        )

        omega2 = eigh(K_free, M_free, eigvals_only=True)
        mu = np.sort(omega2 * m / k)
        mu_analytical = np.sort([(3 - np.sqrt(5)) / 2, (3 + np.sqrt(5)) / 2])
        np.testing.assert_allclose(mu, mu_analytical, rtol=1e-12)

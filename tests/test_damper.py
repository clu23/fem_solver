"""Tests unitaires — élément amortisseur ponctuel (DamperElement).

Validation par solutions analytiques :

- matrices élémentaires sol / deux nœuds (formes exactes),
- nullité des contributions rigidité et masse,
- oscillateur amorti 1-DDL : ζ = c/(2√(km)), ω_d = ω_n·√(1−ζ²).

L'oscillateur amorti est résolu par le problème aux valeurs propres d'état
(state-space) : les valeurs propres complexes λ = −ζω_n ± i·ω_d valident à la
fois la matrice de raideur (ressort) et la matrice d'amortissement (amortisseur).
"""

import numpy as np
import pytest

from femsolver.core.assembler import Assembler
from femsolver.core.material import ElasticMaterial
from femsolver.core.mesh import ElementData, Mesh
from femsolver.elements.damper import DamperElement
from femsolver.elements.spring import SpringElement

_DUMMY_MAT = ElasticMaterial(E=1.0, nu=0.0, rho=1.0)


class TestDamperMatrices:
    """Formes exactes des matrices d'amortissement élémentaires."""

    def test_ground_damper_is_diagonal(self):
        """Amortisseur au sol (1 nœud) : C_e = diag(c)."""
        nodes = np.array([[0.0, 1.0]])
        C_e = DamperElement().damping_matrix(
            _DUMMY_MAT, nodes, {"damping": [0.0, 50.0]}
        )
        np.testing.assert_allclose(
            C_e, np.array([[0.0, 0.0], [0.0, 50.0]]), rtol=1e-12
        )

    def test_two_node_damper_block_form(self):
        """Amortisseur 2 nœuds, 1 DDL : C_e = c·[[1,-1],[-1,1]]."""
        nodes = np.array([[0.0, 0.0], [1.0, 0.0]])
        c = 12.0
        C_e = DamperElement().damping_matrix(_DUMMY_MAT, nodes, {"damping": [c]})
        np.testing.assert_allclose(
            C_e, c * np.array([[1.0, -1.0], [-1.0, 1.0]]), rtol=1e-12
        )

    def test_two_node_multidof_block_form(self):
        """Amortisseur 2 nœuds, 3 DDL : blocs ±diag(c)."""
        nodes = np.array([[0.0, 0.0], [1.0, 0.0]])
        c = [10.0, 20.0, 5.0]
        C_e = DamperElement().damping_matrix(_DUMMY_MAT, nodes, {"damping": c})
        D = np.diag(c)
        np.testing.assert_allclose(C_e, np.block([[D, -D], [-D, D]]), rtol=1e-12)

    def test_stiffness_and_mass_are_zero(self):
        """L'amortisseur n'apporte ni raideur ni masse."""
        nodes = np.array([[0.0, 0.0], [1.0, 0.0]])
        props = {"damping": [10.0, 10.0]}
        K_e = DamperElement().stiffness_matrix(_DUMMY_MAT, nodes, props)
        M_e = DamperElement().mass_matrix(_DUMMY_MAT, nodes, props)
        assert K_e.shape == (4, 4) and M_e.shape == (4, 4)
        np.testing.assert_array_equal(K_e, 0.0)
        np.testing.assert_array_equal(M_e, 0.0)

    def test_negative_damping_raises(self):
        nodes = np.array([[0.0, 0.0]])
        with pytest.raises(ValueError, match="≥ 0"):
            DamperElement().damping_matrix(_DUMMY_MAT, nodes, {"damping": [-1.0]})

    def test_missing_damping_raises(self):
        nodes = np.array([[0.0, 0.0]])
        with pytest.raises(KeyError, match="damping"):
            DamperElement().damping_matrix(_DUMMY_MAT, nodes, {})


class TestDamperAssembly:
    """Assemblage de la matrice C globale via Assembler.assemble_damping."""

    def test_assemble_damping_collects_dampers(self):
        """C globale = contributions des amortisseurs ; ressorts ignorés."""
        nodes = np.array([[0.0, 0.0], [1.0, 0.0]])
        spring = ElementData(SpringElement, (0, 1), _DUMMY_MAT, {"stiffness": [1.0e4]})
        damper = ElementData(DamperElement, (0, 1), _DUMMY_MAT, {"damping": [25.0]})
        mesh = Mesh(nodes=nodes, elements=(spring, damper), n_dim=2, dof_per_node=1)

        C = Assembler(mesh).assemble_damping().toarray()
        np.testing.assert_allclose(
            C, 25.0 * np.array([[1.0, -1.0], [-1.0, 1.0]]), rtol=1e-12
        )

    def test_assemble_damping_empty_without_dampers(self):
        """Sans amortisseur, C est entièrement nulle (même taille que K)."""
        nodes = np.array([[0.0, 0.0], [1.0, 0.0]])
        spring = ElementData(SpringElement, (0, 1), _DUMMY_MAT, {"stiffness": [1.0e4]})
        mesh = Mesh(nodes=nodes, elements=(spring,), n_dim=2, dof_per_node=1)

        C = Assembler(mesh).assemble_damping()
        assert C.shape == (2, 2)
        assert C.nnz == 0


class TestDampedOscillator:
    """Oscillateur amorti 1-DDL : ζ = c/(2√(km)), ω_d = ω_n·√(1−ζ²)."""

    def test_damped_eigenvalues(self):
        """m=2, k=200, c=2 → ω_n=10, ζ=0.05, ω_d=ω_n√(1−ζ²).

        Système : ressort au sol k + amortisseur au sol c + masse m (1 DDL).
        Les valeurs propres de la matrice d'état A = [[0,1],[−k/m,−c/m]] sont
        λ = −ζω_n ± i·ω_d.
        """
        m, k, c = 2.0, 200.0, 2.0
        nodes = np.array([[0.0, 0.0]])
        spring = ElementData(SpringElement, (0,), _DUMMY_MAT, {"stiffness": [k]})
        damper = ElementData(DamperElement, (0,), _DUMMY_MAT, {"damping": [c]})
        mesh = Mesh(nodes=nodes, elements=(spring, damper), n_dim=2, dof_per_node=1)

        asm = Assembler(mesh)
        K = asm.assemble_stiffness().toarray()
        C = asm.assemble_damping().toarray()
        M = np.array([[m]])

        # Problème d'état : A = [[0, I], [-M⁻¹K, -M⁻¹C]]
        Minv = np.linalg.inv(M)
        A = np.block([
            [np.zeros((1, 1)), np.eye(1)],
            [-Minv @ K, -Minv @ C],
        ])
        lam = np.linalg.eigvals(A)
        lam_complex = lam[np.argmax(np.abs(lam.imag))]  # une des racines complexes

        omega_n = np.sqrt(k / m)
        zeta = c / (2.0 * np.sqrt(k * m))
        omega_d = omega_n * np.sqrt(1.0 - zeta**2)

        np.testing.assert_allclose(abs(lam_complex.imag), omega_d, rtol=1e-10)
        np.testing.assert_allclose(-lam_complex.real, zeta * omega_n, rtol=1e-10)
        # Cohérence des chiffres : ζ = 0.05, ω_d ≈ 9.98749 rad/s
        np.testing.assert_allclose(zeta, 0.05, rtol=1e-12)
        np.testing.assert_allclose(omega_d, 9.98749217, rtol=1e-6)

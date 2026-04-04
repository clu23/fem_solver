"""Tests unitaires — amortissement de Rayleigh.

Solutions analytiques utilisées
--------------------------------

1. **Calibration par Cramer**

   Pour ω₁=10, ω₂=100, ξ₁=ξ₂=0.05 :

       α = 2·ξ·ω₁·ω₂/(ω₁+ω₂) = 2×0.05×10×100/110 ≈ 0.9091
       β = 2·ξ/(ω₁+ω₂)        = 2×0.05/110       ≈ 9.091×10⁻⁴

   Vérification : ξ(ω₁) = α/(2ω₁) + β·ω₁/2 = 0.9091/20 + 9.091e-4×5 = 0.05 ✓
                  ξ(ω₂) = α/(2ω₂) + β·ω₂/2 = 0.9091/200 + 9.091e-4×50 = 0.05 ✓

2. **Orthogonalité modale**

   Pour des modes M-orthonormés (φᵀMφ = I) :

       φₙᵀ C φₙ = α·1 + β·ωₙ² = 2·ξₙ·ωₙ

   Si α=0, β=β₀ : φₙᵀ C φₙ = β₀·ωₙ²
   Si β=0, α=α₀ : φₙᵀ C φₙ = α₀

3. **Masse consistante barre 1 DDL**

   Barre 1 élément : BC nœud 0 → u₁=0.
   M_free = ρAL × 2/6 = ρAL/3 (terme [1,1] de la masse consistante 2×2)
   K_free = EA/L
   ωₙ = √(K_free/M_free) = √(3E/(ρL²))
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.sparse import eye as speye

from femsolver.core.assembler import Assembler
from femsolver.core.boundary import apply_dirichlet
from femsolver.core.material import ElasticMaterial
from femsolver.core.mesh import BoundaryConditions, ElementData, Mesh
from femsolver.core.solver import ModalSolver
from femsolver.dynamics.rayleigh import (
    RayleighDamping,
    build_damping_matrix,
    rayleigh_from_modes,
)
from femsolver.elements.bar2d import Bar2D
from femsolver.elements.beam2d import Beam2D


# ---------------------------------------------------------------------------
# Matériau et données communes
# ---------------------------------------------------------------------------

E   = 210e9   # Pa
NU  = 0.3
RHO = 7800.0  # kg/m³
A   = 1e-4    # m²
h   = 0.01    # m   (hauteur section carrée 1 cm)
I   = h**4 / 12
L   = 1.0     # m

MAT      = ElasticMaterial(E=E, nu=NU, rho=RHO)
PROPS_BAR  = {"area": A}
PROPS_BEAM = {"area": A, "inertia": I}


def _bar_system() -> tuple:
    """Console barre 1 élément : 1 DDL libre (ux₂)."""
    nodes    = np.array([[0.0, 0.0], [L, 0.0]])
    elements = (ElementData(Bar2D, (0, 1), MAT, PROPS_BAR),)
    bc = BoundaryConditions(dirichlet={0: {0: 0.0}}, neumann={})
    mesh = Mesh(nodes=nodes, elements=elements, n_dim=2, dof_per_node=2)
    asm  = Assembler(mesh)
    K    = asm.assemble_stiffness()
    M    = asm.assemble_mass()
    ds   = apply_dirichlet(K, np.zeros(mesh.n_dof), mesh, bc)
    K_f  = ds.K_free
    M_f  = ds.reduce_mass(M)
    return K_f, M_f, ds.free_dofs


def _beam_system(n_elem: int = 4) -> tuple:
    """Console poutre n_elem éléments : retourne (K_free, M_free, free_dofs, omega_n)."""
    n_nodes = n_elem + 1
    nodes   = np.column_stack([np.linspace(0.0, L, n_nodes), np.zeros(n_nodes)])
    elems   = tuple(ElementData(Beam2D, (i, i + 1), MAT, PROPS_BEAM) for i in range(n_elem))
    bc = BoundaryConditions(dirichlet={0: {0: 0.0, 1: 0.0, 2: 0.0}}, neumann={})
    mesh = Mesh(nodes=nodes, elements=elems, n_dim=2, dof_per_node=3)
    asm  = Assembler(mesh)
    K    = asm.assemble_stiffness()
    M    = asm.assemble_mass()
    ds   = apply_dirichlet(K, np.zeros(mesh.n_dof), mesh, bc)
    K_f  = ds.K_free
    M_f  = ds.reduce_mass(M)
    freqs, modes = ModalSolver().solve(K_f, M_f, n_modes=3)
    omega = freqs * 2.0 * np.pi
    return K_f, M_f, ds.free_dofs, omega, modes


# ===========================================================================
# 1. RayleighDamping — dataclass et ξ(ω)
# ===========================================================================

class TestRayleighDampingDataclass:
    """Tests sur RayleighDamping et modal_damping_ratio."""

    def test_valid_creation(self) -> None:
        d = RayleighDamping(alpha=2.0, beta=0.001)
        assert d.alpha == 2.0
        assert d.beta == 0.001

    def test_alpha_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            RayleighDamping(alpha=-1.0, beta=0.0)

    def test_beta_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="beta"):
            RayleighDamping(alpha=0.0, beta=-1e-3)

    def test_zero_coefficients(self) -> None:
        """α=β=0 : ξ=0 (système non amorti)."""
        d = RayleighDamping(alpha=0.0, beta=0.0)
        omega = np.array([10.0, 100.0, 1000.0])
        np.testing.assert_allclose(d.modal_damping_ratio(omega), 0.0)

    def test_alpha_only(self) -> None:
        """β=0 : ξ(ω) = α/(2ω)."""
        alpha = 5.0
        d = RayleighDamping(alpha=alpha, beta=0.0)
        omega = np.array([10.0, 50.0, 100.0])
        expected = alpha / (2.0 * omega)
        np.testing.assert_allclose(d.modal_damping_ratio(omega), expected, rtol=1e-14)

    def test_beta_only(self) -> None:
        """α=0 : ξ(ω) = β·ω/2."""
        beta = 0.001
        d = RayleighDamping(alpha=0.0, beta=beta)
        omega = np.array([10.0, 50.0, 100.0])
        expected = beta * omega / 2.0
        np.testing.assert_allclose(d.modal_damping_ratio(omega), expected, rtol=1e-14)

    def test_zeta_minimum_between_targets(self) -> None:
        """ξ atteint son minimum entre ω₁ et ω₂ (dérivée nulle).

        dξ/dω = −α/(2ω²) + β/2 = 0  ⟹  ω* = √(α/β)
        Le ξ minimum vaut ξ* = β·ω*/2 + α/(2ω*) = √(αβ).
        """
        omega1, omega2, zeta = 10.0, 100.0, 0.05
        d = rayleigh_from_modes(omega1, omega2, zeta, zeta)

        omega_star = math.sqrt(d.alpha / d.beta)
        zeta_star  = math.sqrt(d.alpha * d.beta)   # = √(α·β)

        assert omega1 < omega_star < omega2
        np.testing.assert_allclose(
            d.modal_damping_ratio(np.array([omega_star])),
            np.array([zeta_star]),
            rtol=1e-12,
        )


# ===========================================================================
# 2. Calibration rayleigh_from_modes
# ===========================================================================

class TestRayleighFromModes:
    """Formule de calibration et cas particuliers."""

    def test_equal_zeta_recovers_formula(self) -> None:
        """α = 2ξω₁ω₂/(ω₁+ω₂),  β = 2ξ/(ω₁+ω₂)."""
        omega1, omega2, zeta = 10.0, 100.0, 0.05
        d = rayleigh_from_modes(omega1, omega2, zeta, zeta)
        alpha_expected = 2.0 * zeta * omega1 * omega2 / (omega1 + omega2)
        beta_expected  = 2.0 * zeta / (omega1 + omega2)
        np.testing.assert_allclose(d.alpha, alpha_expected, rtol=1e-12)
        np.testing.assert_allclose(d.beta,  beta_expected,  rtol=1e-12)

    def test_equal_zeta_roundtrip(self) -> None:
        """ξ(ω₁) = ξ₁ et ξ(ω₂) = ξ₂ après calibration."""
        omega1, omega2, zeta = 20.0, 200.0, 0.03
        d = rayleigh_from_modes(omega1, omega2, zeta, zeta)
        xi = d.modal_damping_ratio(np.array([omega1, omega2]))
        np.testing.assert_allclose(xi, [zeta, zeta], rtol=1e-12)

    def test_different_zeta_roundtrip(self) -> None:
        """Taux d'amortissement distincts récupérés exactement."""
        omega1, omega2 = 10.0, 150.0
        zeta1, zeta2   = 0.02, 0.08
        d = rayleigh_from_modes(omega1, omega2, zeta1, zeta2)
        xi = d.modal_damping_ratio(np.array([omega1, omega2]))
        np.testing.assert_allclose(xi[0], zeta1, rtol=1e-12)
        np.testing.assert_allclose(xi[1], zeta2, rtol=1e-12)

    def test_omega1_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="omega1"):
            rayleigh_from_modes(0.0, 100.0, 0.05, 0.05)

    def test_omega2_le_omega1_raises(self) -> None:
        with pytest.raises(ValueError, match="omega2"):
            rayleigh_from_modes(100.0, 10.0, 0.05, 0.05)

    def test_omega2_equal_omega1_raises(self) -> None:
        with pytest.raises(ValueError, match="omega2"):
            rayleigh_from_modes(50.0, 50.0, 0.05, 0.05)

    def test_negative_alpha_raises(self) -> None:
        """Des taux très asymétriques peuvent produire α < 0 → ValueError."""
        # zeta2 >> zeta1 et omega2 >> omega1 : α peut devenir négatif
        with pytest.raises(ValueError):
            rayleigh_from_modes(10.0, 100.0, zeta1=0.001, zeta2=0.50)

    def test_cramer_formula_explicit(self) -> None:
        """Vérifie la formule de Cramer terme à terme (ξ₁ ≠ ξ₂)."""
        omega1, omega2 = 10.0, 100.0
        zeta1, zeta2   = 0.03, 0.07
        d = rayleigh_from_modes(omega1, omega2, zeta1, zeta2)

        denom  = omega2**2 - omega1**2
        alpha_expected = 2 * omega1 * omega2 * (zeta1 * omega2 - zeta2 * omega1) / denom
        beta_expected  = 2 * (zeta2 * omega2 - zeta1 * omega1) / denom

        np.testing.assert_allclose(d.alpha, alpha_expected, rtol=1e-12)
        np.testing.assert_allclose(d.beta,  beta_expected,  rtol=1e-12)


# ===========================================================================
# 3. build_damping_matrix — propriétés de C
# ===========================================================================

class TestBuildDampingMatrix:
    """Propriétés algébriques de C = αM + βK."""

    def setup_method(self) -> None:
        self.K_f, self.M_f, self.free = _bar_system()

    def test_shape(self) -> None:
        d = RayleighDamping(alpha=2.0, beta=0.001)
        C = build_damping_matrix(d, self.M_f, self.K_f)
        assert C.shape == self.K_f.shape

    def test_symmetric(self) -> None:
        d = RayleighDamping(alpha=2.0, beta=0.001)
        C = build_damping_matrix(d, self.M_f, self.K_f)
        diff = (C - C.T).toarray()
        np.testing.assert_allclose(diff, 0.0, atol=1e-20)

    def test_alpha_only(self) -> None:
        """β=0 : C = αM."""
        alpha = 3.7
        d = RayleighDamping(alpha=alpha, beta=0.0)
        C = build_damping_matrix(d, self.M_f, self.K_f)
        expected = (alpha * self.M_f).toarray()
        np.testing.assert_allclose(C.toarray(), expected, rtol=1e-14)

    def test_beta_only(self) -> None:
        """α=0 : C = βK."""
        beta = 5e-4
        d = RayleighDamping(alpha=0.0, beta=beta)
        C = build_damping_matrix(d, self.M_f, self.K_f)
        expected = (beta * self.K_f).toarray()
        np.testing.assert_allclose(C.toarray(), expected, rtol=1e-14)

    def test_incompatible_shapes_raises(self) -> None:
        d = RayleighDamping(alpha=1.0, beta=0.0)
        # K_f est 3×3 (bar 1 elem avec 1 BC scalaire → 3 DDL libres)
        # On crée une matrice de taille différente pour déclencher l'erreur
        M_wrong = speye(5, format="csr")   # 5×5 ≠ 3×3
        with pytest.raises(ValueError, match="shape"):
            build_damping_matrix(d, M_wrong, self.K_f)

    def test_positive_diagonal(self) -> None:
        """Tous les termes diagonaux de C doivent être ≥ 0."""
        d = RayleighDamping(alpha=2.0, beta=0.001)
        C = build_damping_matrix(d, self.M_f, self.K_f)
        assert (C.diagonal() >= 0.0).all()


# ===========================================================================
# 4. Orthogonalité modale de C
# ===========================================================================

class TestRayleighModalOrthogonality:
    """φᵀ C φ = diag(2ξₙωₙ) dans la base modale.

    Vérification fondamentale : Rayleigh garantit que C est diagonale dans
    la base modale des modes propres.  C'est l'hypothèse clé qui permet
    de traiter chaque mode comme un SDOF indépendant.
    """

    def setup_method(self) -> None:
        K_f, M_f, free, omega, modes = _beam_system(n_elem=4)
        self.K_f   = K_f
        self.M_f   = M_f
        self.omega = omega       # pulsations propres des 3 premiers modes [rad/s]
        self.modes = modes       # (n_free, 3), M_f-normalisés

    def _check_modal_damping(
        self,
        damping: RayleighDamping,
        tol: float = 1e-8,
    ) -> None:
        """Vérifie φᵀ·C·φ ≈ diag(2ξₙωₙ) et que les termes hors diagonale ≈ 0."""
        C = build_damping_matrix(damping, self.M_f, self.K_f)
        phi = self.modes          # (n_free, n_modes), M-orthonormés

        C_modal = phi.T @ C.toarray() @ phi   # doit être diagonale

        # Termes diagonaux attendus : 2·ξₙ·ωₙ
        zeta = damping.modal_damping_ratio(self.omega)
        expected_diag = 2.0 * zeta * self.omega
        np.testing.assert_allclose(
            np.diag(C_modal), expected_diag, rtol=tol,
            err_msg="Termes diagonaux de C_modal incorrects",
        )

        # Termes hors diagonale doivent être ≈ 0
        off_diag = C_modal - np.diag(np.diag(C_modal))
        assert np.abs(off_diag).max() < tol * np.abs(expected_diag).max(), (
            f"Termes hors diagonale non nuls : {np.abs(off_diag).max():.2e}"
        )

    def test_alpha_only(self) -> None:
        """β=0 : C = αM → φᵀCφ = α·I (car M-orthonormé : φᵀMφ = I)."""
        d = RayleighDamping(alpha=5.0, beta=0.0)
        self._check_modal_damping(d)

    def test_beta_only(self) -> None:
        """α=0 : C = βK → φᵀCφ = β·diag(ωₙ²)."""
        d = RayleighDamping(alpha=0.0, beta=1e-3)
        self._check_modal_damping(d)

    def test_full_rayleigh(self) -> None:
        """Rayleigh complet calibré sur les modes 1 et 2."""
        omega1, omega2 = self.omega[0], self.omega[1]
        d = rayleigh_from_modes(omega1, omega2, zeta1=0.02, zeta2=0.05)
        self._check_modal_damping(d)

    def test_equal_zeta_calibration(self) -> None:
        """Calibration sur ω₁ et ω₂ avec ξ=5% : termes modaux exacts."""
        omega1, omega2 = self.omega[0], self.omega[2]
        zeta_target = 0.05
        d = rayleigh_from_modes(omega1, omega2, zeta_target, zeta_target)
        # Les modes aux fréquences de calibration doivent avoir ξ = 5%
        xi_check = d.modal_damping_ratio(np.array([omega1, omega2]))
        np.testing.assert_allclose(xi_check, zeta_target, rtol=1e-12)
        self._check_modal_damping(d)

    def test_phi_transposed_m_phi_is_identity(self) -> None:
        """Pré-condition : modes bien M-orthonormés (fourni par ModalSolver)."""
        phi = self.modes
        M_modal = phi.T @ self.M_f.toarray() @ phi
        np.testing.assert_allclose(M_modal, np.eye(phi.shape[1]), atol=1e-10)

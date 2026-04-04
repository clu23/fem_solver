"""Tests — amortissement hystérétique et modal constant.

Solutions analytiques utilisées
--------------------------------

**SDOF (oscillateur 1 DDL)**

Paramètres :

    K_SDOF = EA/L           (rigidité du DDL libre ux)
    M_SDOF = ρAL/3          (masse consistante du DDL libre ux)
    ωₙ = √(K_SDOF/M_SDOF)

**1. Amortissement hystérétique — limite statique**

Pour Ω → 0 :

    Z → K(1 + iη)    →    H → 1/(K(1+iη))

    |H(0)| = 1 / (K √(1+η²))
    arg(H(0)) = −arctan(η)             ← ≠ 0° contrairement au visqueux

**2. Amortissement hystérétique — résonance**

Pour Ω = ωₙ, K = ωₙ²M dans la base modale :

    Z = K(1+iη) − ωₙ²M = K − K + iηK = iηK

    |H(ωₙ)| = 1 / (ηK)    avec η = 2ξ  →  1 / (2ξK)

**3. Superposition modale — SDOF**

Mode M-normalisé : φ₁ = 1/√M_SDOF  (φ₁²·M = 1).
Facteur de participation : Γ₁ = φ₁·F̂.

Pour Ω → 0 :
    denom → ωₙ²    →    H = φ₁·Γ₁/ωₙ² = (1/M_SDOF)/ωₙ² = 1/K_SDOF  ✓

Pour Ω = ωₙ :
    denom = 2iξωₙ²    →    |H| = (1/M_SDOF)/(2ξωₙ²) = 1/(2ξK)  ✓

**4. Équivalence des trois modèles à la résonance**

Pour le même ξ avec η = 2ξ :

    Rayleigh α (α=2ξωₙ) : Z = i·2ξK  →  |H| = 1/(2ξK)
    Hystérétique (η=2ξ) : Z = iηK   = i·2ξK  →  |H| = 1/(2ξK)
    Modal ξ₁ = ξ        : Z_modal = 2iξωₙ²   →  |H| = 1/(2ξK)

**5. Différence hors résonance (Rayleigh vs hystérétique)**

À haute fréquence Ω ≫ ωₙ, Rayleigh α pur (β=0) :
    ξ_eff(Ω) = α/(2Ω) → 0  (amortissement qui disparaît)

Hystérétique : ξ_eff = η/2 = constante (indépendant de Ω).
→ |H|_hysteretic < |H|_rayleigh pour Ω ≫ ωₙ.

**6. build_C_physical — orthogonalité modale**

Pour les modes M-normalisés φₙ :
    φₘᵀ · C · φₙ = 2ξₙωₙ · δₘₙ
"""

from __future__ import annotations

import cmath
import math

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from femsolver.core.assembler import Assembler
from femsolver.core.boundary import apply_dirichlet
from femsolver.core.material import ElasticMaterial
from femsolver.core.mesh import BoundaryConditions, ElementData, Mesh
from femsolver.core.solver import ModalSolver
from femsolver.dynamics.damping import HystereticDamping, ModalDampingModel
from femsolver.dynamics.harmonic import (
    run_harmonic,
    solve_harmonic,
    solve_harmonic_hysteretic,
    solve_harmonic_modal,
)
from femsolver.dynamics.modal import ModalResult, run_modal
from femsolver.dynamics.rayleigh import RayleighDamping
from femsolver.elements.bar2d import Bar2D
from femsolver.elements.beam2d import Beam2D


# ---------------------------------------------------------------------------
# Paramètres communs
# ---------------------------------------------------------------------------

E   = 210e9
NU  = 0.3
RHO = 7800.0
A   = 1e-4    # m²
L   = 1.0     # m
MAT = ElasticMaterial(E=E, nu=NU, rho=RHO)

#: k = EA/L  (rigidité SDOF scalaire)
K_SDOF = E * A / L
#: m = ρAL/3  (masse consistante du DDL ux libre, terme [1,1])
M_SDOF = RHO * A * L / 3.0
#: ωₙ = √(k/m)
OMEGA_N = math.sqrt(K_SDOF / M_SDOF)
#: fₙ = ωₙ / (2π)
F_N = OMEGA_N / (2.0 * math.pi)


def _make_1x1() -> tuple:
    """Matrices 1×1 pour le SDOF analytique (DDL ux libre)."""
    K1 = csr_matrix(np.array([[K_SDOF]]))
    M1 = csr_matrix(np.array([[M_SDOF]]))
    F1 = np.array([1.0])
    return K1, M1, F1


def _make_cantilever(n_elem: int = 4) -> tuple[Mesh, BoundaryConditions, int]:
    """Console poutre Euler-Bernoulli.  Retourne (mesh, bc, dof_tip_uy)."""
    n_nodes = n_elem + 1
    nodes   = np.column_stack([np.linspace(0.0, L, n_nodes), np.zeros(n_nodes)])
    h       = 0.01    # m, section carrée
    Iz      = h**4 / 12
    elems   = tuple(
        ElementData(Beam2D, (i, i + 1), MAT, {"area": A, "inertia": Iz})
        for i in range(n_elem)
    )
    bc = BoundaryConditions(dirichlet={0: {0: 0.0, 1: 0.0, 2: 0.0}}, neumann={})
    mesh = Mesh(nodes=nodes, elements=elems, n_dim=2, dof_per_node=3)
    dof_tip_uy = 3 * n_elem + 1
    return mesh, bc, dof_tip_uy


# ===========================================================================
# 1. HystereticDamping — dataclass
# ===========================================================================

class TestHystereticDamping:
    """Validation du dataclass HystereticDamping."""

    def test_valid_creation(self) -> None:
        d = HystereticDamping(eta=0.10)
        assert d.eta == 0.10

    def test_zero_eta(self) -> None:
        """η=0 est valide (système non amorti)."""
        d = HystereticDamping(eta=0.0)
        assert d.eta == 0.0

    def test_negative_eta_raises(self) -> None:
        with pytest.raises(ValueError, match="eta"):
            HystereticDamping(eta=-0.01)

    def test_equivalent_zeta(self) -> None:
        """η = 2ξ  →  ξ = η/2."""
        d = HystereticDamping(eta=0.10)
        np.testing.assert_allclose(d.equivalent_zeta(), 0.05, rtol=1e-14)

    def test_equivalent_zeta_zero(self) -> None:
        d = HystereticDamping(eta=0.0)
        assert d.equivalent_zeta() == 0.0


# ===========================================================================
# 2. ModalDampingModel — dataclass et factory
# ===========================================================================

class TestModalDampingModel:
    """Validation du dataclass ModalDampingModel."""

    def setup_method(self) -> None:
        mesh, bc, _ = _make_cantilever(n_elem=4)
        self.modal_result = run_modal(mesh, bc, n_modes=3)
        self.mesh = mesh
        self.bc   = bc

    # ------------------------------------------------------------------
    # Instanciation
    # ------------------------------------------------------------------

    def test_from_modal_result_uniform(self) -> None:
        """Taux uniforme appliqué à tous les modes."""
        d = ModalDampingModel.from_modal_result(self.modal_result, zeta_n=0.05)
        np.testing.assert_allclose(d.zeta_n, [0.05, 0.05, 0.05])
        np.testing.assert_allclose(d.omega_n, self.modal_result.omega)
        assert d.phi.shape == self.modal_result.modes.shape

    def test_from_modal_result_array(self) -> None:
        """Taux différents par mode."""
        zeta = [0.02, 0.03, 0.04]
        d = ModalDampingModel.from_modal_result(self.modal_result, zeta_n=zeta)
        np.testing.assert_allclose(d.zeta_n, zeta)

    def test_mismatched_shapes_raises(self) -> None:
        """omega_n et zeta_n doivent avoir la même longueur."""
        with pytest.raises(ValueError, match="même longueur"):
            ModalDampingModel(
                omega_n=np.array([10.0, 20.0]),
                zeta_n=np.array([0.05]),
                phi=np.zeros((6, 2)),
            )

    def test_phi_wrong_columns_raises(self) -> None:
        with pytest.raises(ValueError, match="phi"):
            ModalDampingModel(
                omega_n=np.array([10.0]),
                zeta_n=np.array([0.05]),
                phi=np.zeros((6, 3)),   # 3 colonnes ≠ 1 mode
            )

    def test_negative_omega_raises(self) -> None:
        with pytest.raises(ValueError, match="pulsations"):
            ModalDampingModel(
                omega_n=np.array([-10.0]),
                zeta_n=np.array([0.05]),
                phi=np.zeros((4, 1)),
            )

    def test_negative_zeta_raises(self) -> None:
        with pytest.raises(ValueError, match="amortissement"):
            ModalDampingModel(
                omega_n=np.array([10.0]),
                zeta_n=np.array([-0.01]),
                phi=np.zeros((4, 1)),
            )

    # ------------------------------------------------------------------
    # build_C_physical — propriétés algébriques
    # ------------------------------------------------------------------

    def test_build_C_physical_shape(self) -> None:
        """C a la même forme que M."""
        asm = Assembler(self.mesh)
        M   = asm.assemble_mass()
        d   = ModalDampingModel.from_modal_result(self.modal_result, zeta_n=0.02)
        C   = d.build_C_physical(M)
        assert C.shape == M.shape

    def test_build_C_physical_symmetric(self) -> None:
        """C doit être symétrique (C = Cᵀ), à la précision float64."""
        asm = Assembler(self.mesh)
        M   = asm.assemble_mass()
        d   = ModalDampingModel.from_modal_result(self.modal_result, zeta_n=0.02)
        C   = d.build_C_physical(M)
        diff = (C - C.T).toarray()
        # Bruit float64 ≈ 2.2e-16 × |C|_max ; atol doit couvrir cette échelle
        np.testing.assert_allclose(diff, 0.0, atol=1e-14)

    def test_build_C_physical_modal_orthogonality(self) -> None:
        """φₘᵀ·C·φₙ = 2ξₙωₙ·δₘₙ dans la base modale M-normalisée.

        C'est la condition exacte garantissant que C diagonalise dans la base
        des modes propres.
        """
        asm = Assembler(self.mesh)
        M   = asm.assemble_mass()
        ds  = apply_dirichlet(
            asm.assemble_stiffness(), np.zeros(self.mesh.n_dof), self.mesh, self.bc
        )
        M_free = ds.reduce_mass(M)
        free   = ds.free_dofs

        # Modes M_free-normalisés extraits des DDL libres
        modal = self.modal_result
        phi_free = modal.modes[free, :]   # (n_free, n_modes)

        zeta = np.array([0.01, 0.03, 0.05])
        d    = ModalDampingModel.from_modal_result(modal, zeta_n=zeta)
        C    = d.build_C_physical(M)
        C_free = C[free, :][:, free].toarray()

        C_modal = phi_free.T @ C_free @ phi_free   # (n_modes, n_modes)

        # Termes diagonaux attendus : 2ξₙωₙ
        expected_diag = 2.0 * zeta * modal.omega
        np.testing.assert_allclose(
            np.diag(C_modal), expected_diag, rtol=1e-8,
            err_msg="Termes diagonaux de C_modal incorrects",
        )

        # Hors-diagonale ≈ 0
        off = C_modal - np.diag(np.diag(C_modal))
        assert np.abs(off).max() < 1e-8 * np.abs(expected_diag).max(), (
            f"Termes hors-diagonale non nuls : {np.abs(off).max():.2e}"
        )

    def test_build_C_physical_incompatible_M_raises(self) -> None:
        d = ModalDampingModel.from_modal_result(self.modal_result, zeta_n=0.02)
        M_wrong = csr_matrix(np.eye(3))   # mauvaise taille
        with pytest.raises(ValueError, match="incompatible"):
            d.build_C_physical(M_wrong)


# ===========================================================================
# 3. solve_harmonic_hysteretic — propriétés du SDOF
# ===========================================================================

class TestSolveHarmonicHysteretic:
    """FRF hystérétique sur un SDOF analytique 1×1."""

    def setup_method(self) -> None:
        self.K1, self.M1, self.F1 = _make_1x1()
        self.eta  = 0.10    # η = 2ξ = 2×5%
        self.zeta = 0.05

    # ------------------------------------------------------------------
    # Forme de la sortie
    # ------------------------------------------------------------------

    def test_shape(self) -> None:
        freqs = np.linspace(0.1 * F_N, 3.0 * F_N, 50)
        U = solve_harmonic_hysteretic(self.K1, self.M1, self.eta, self.F1, freqs)
        assert U.shape == (1, 50)
        assert U.dtype == complex

    # ------------------------------------------------------------------
    # Limite statique : Ω → 0
    # ------------------------------------------------------------------

    def test_static_limit_amplitude(self) -> None:
        """|H(0)| = 1 / (K·√(1+η²))."""
        freqs = np.array([1e-6 * F_N])
        U = solve_harmonic_hysteretic(self.K1, self.M1, self.eta, self.F1, freqs)
        expected_amp = 1.0 / (K_SDOF * math.sqrt(1.0 + self.eta**2))
        np.testing.assert_allclose(abs(U[0, 0]), expected_amp, rtol=1e-6)

    def test_static_limit_phase_nonzero(self) -> None:
        """Phase à Ω→0 : arg(H) = −arctan(η) ≠ 0° (spécificité hystérétique).

        Contrairement au modèle visqueux où Im(H(0)) = 0, l'amortissement
        hystérétique maintient une phase −arctan(η) même à fréquence nulle.
        """
        freqs = np.array([1e-8 * F_N])
        U = solve_harmonic_hysteretic(self.K1, self.M1, self.eta, self.F1, freqs)
        expected_phase = -math.degrees(math.atan(self.eta))  # ≈ −5.71°
        actual_phase   = math.degrees(cmath.phase(U[0, 0]))
        np.testing.assert_allclose(actual_phase, expected_phase, atol=0.01)

    def test_zero_eta_real_response(self) -> None:
        """η=0 : H est réel (pas de déphasage)."""
        freqs = np.linspace(0.1 * F_N, 0.9 * F_N, 20)
        U = solve_harmonic_hysteretic(self.K1, self.M1, 0.0, self.F1, freqs)
        np.testing.assert_allclose(U.imag, 0.0, atol=1e-20)

    # ------------------------------------------------------------------
    # Résonance : Ω = ωₙ
    # ------------------------------------------------------------------

    def test_resonance_amplitude(self) -> None:
        """|H(ωₙ)| = 1/(ηK) = 1/(2ξK)."""
        freqs = np.array([F_N])
        U = solve_harmonic_hysteretic(self.K1, self.M1, self.eta, self.F1, freqs)
        expected = 1.0 / (self.eta * K_SDOF)
        np.testing.assert_allclose(abs(U[0, 0]), expected, rtol=1e-10)

    def test_resonance_phase_minus90(self) -> None:
        """Phase à la résonance ≈ −90° (même que visqueux)."""
        freqs = np.array([F_N])
        U = solve_harmonic_hysteretic(self.K1, self.M1, self.eta, self.F1, freqs)
        phase_deg = math.degrees(cmath.phase(U[0, 0]))
        np.testing.assert_allclose(phase_deg, -90.0, atol=0.5)

    # ------------------------------------------------------------------
    # Haute fréquence
    # ------------------------------------------------------------------

    def test_high_frequency_decay(self) -> None:
        """|H| → 0 pour Ω ≫ ωₙ."""
        freqs = np.array([100.0 * F_N])
        U = solve_harmonic_hysteretic(self.K1, self.M1, self.eta, self.F1, freqs)
        H_static = 1.0 / K_SDOF
        assert abs(U[0, 0]) < H_static * 1e-3


# ===========================================================================
# 4. solve_harmonic_modal — propriétés du SDOF
# ===========================================================================

class TestSolveHarmonicModal:
    """Superposition modale sur un SDOF analytique (1 mode, 1 DDL libre)."""

    def setup_method(self) -> None:
        # Mode M-normalisé : φ = 1/√M_SDOF  (scalaire → tableau (1,1))
        self.phi_free = np.array([[1.0 / math.sqrt(M_SDOF)]])
        self.omega_n  = np.array([OMEGA_N])
        self.zeta_n   = np.array([0.05])
        self.F1       = np.array([1.0])

    # ------------------------------------------------------------------
    # Forme
    # ------------------------------------------------------------------

    def test_shape(self) -> None:
        freqs = np.linspace(0.1 * F_N, 3.0 * F_N, 50)
        U = solve_harmonic_modal(self.omega_n, self.zeta_n, self.phi_free,
                                  self.F1, freqs)
        assert U.shape == (1, 50)
        assert U.dtype == complex

    # ------------------------------------------------------------------
    # Limite statique : Ω → 0
    # ------------------------------------------------------------------

    def test_static_limit(self) -> None:
        """|H(0)| = 1/K_SDOF (limite statique, SDOF avec 1 mode = base complète)."""
        freqs = np.array([1e-8 * F_N])
        U = solve_harmonic_modal(self.omega_n, self.zeta_n, self.phi_free,
                                  self.F1, freqs)
        np.testing.assert_allclose(abs(U[0, 0]), 1.0 / K_SDOF, rtol=1e-6)

    def test_static_limit_phase_zero(self) -> None:
        """À Ω→0 : phase ≈ 0° (amortissement visqueux, comme Rayleigh)."""
        freqs = np.array([1e-8 * F_N])
        U = solve_harmonic_modal(self.omega_n, self.zeta_n, self.phi_free,
                                  self.F1, freqs)
        phase_deg = math.degrees(cmath.phase(U[0, 0]))
        np.testing.assert_allclose(phase_deg, 0.0, atol=0.01)

    # ------------------------------------------------------------------
    # Résonance : Ω = ωₙ
    # ------------------------------------------------------------------

    def test_resonance_amplitude(self) -> None:
        """|H(ωₙ)| = 1/(2ξK) via superposition modale."""
        freqs = np.array([F_N])
        U = solve_harmonic_modal(self.omega_n, self.zeta_n, self.phi_free,
                                  self.F1, freqs)
        expected = 1.0 / (2.0 * self.zeta_n[0] * K_SDOF)
        np.testing.assert_allclose(abs(U[0, 0]), expected, rtol=1e-10)

    def test_resonance_phase_minus90(self) -> None:
        """Phase à la résonance ≈ −90°."""
        freqs = np.array([F_N])
        U = solve_harmonic_modal(self.omega_n, self.zeta_n, self.phi_free,
                                  self.F1, freqs)
        phase_deg = math.degrees(cmath.phase(U[0, 0]))
        np.testing.assert_allclose(phase_deg, -90.0, atol=0.5)

    def test_matches_rayleigh_alpha_sdof(self) -> None:
        """Modal et Rayleigh α donnent la même FRF pour le SDOF (base complète).

        Pour un SDOF avec 1 seul mode, les deux formulations sont exactement
        équivalentes sur tout le spectre (pas de troncature modale).
        """
        K1 = csr_matrix(np.array([[K_SDOF]]))
        M1 = csr_matrix(np.array([[M_SDOF]]))
        F1 = np.array([1.0])
        alpha  = 2.0 * self.zeta_n[0] * OMEGA_N
        c_sdof = alpha * M_SDOF
        C1     = csr_matrix(np.array([[c_sdof]]))

        freqs = np.linspace(0.01 * F_N, 3.0 * F_N, 200)
        U_rayleigh = solve_harmonic(K1, M1, C1, F1, freqs)[0, :]
        U_modal    = solve_harmonic_modal(
            self.omega_n, self.zeta_n, self.phi_free, F1, freqs
        )[0, :]
        np.testing.assert_allclose(U_modal, U_rayleigh, rtol=1e-10)


# ===========================================================================
# 5. Équivalence à la résonance — comparaison des trois modèles
# ===========================================================================

class TestResonanceEquivalence:
    """Amplitude identique pour les trois modèles avec le même ξ.

    Test fondamental : pour ξ = 5% et η = 2ξ = 10%, l'amplitude à la
    résonance doit valoir 1/(2ξK) pour Rayleigh, hystérétique et modal.
    """

    def setup_method(self) -> None:
        self.K1, self.M1, self.F1 = _make_1x1()
        self.freqs    = np.array([F_N])    # exactement à la résonance
        self.zeta     = 0.05
        self.expected = 1.0 / (2.0 * self.zeta * K_SDOF)

    def test_rayleigh_alpha_resonance(self) -> None:
        """Rayleigh α pur (β=0) : |H(ωₙ)| = 1/(2ξK)."""
        alpha = 2.0 * self.zeta * OMEGA_N
        C1    = csr_matrix(np.array([[alpha * M_SDOF]]))
        U = solve_harmonic(self.K1, self.M1, C1, self.F1, self.freqs)
        np.testing.assert_allclose(abs(U[0, 0]), self.expected, rtol=1e-10)

    def test_hysteretic_resonance(self) -> None:
        """Hystérétique η=2ξ : |H(ωₙ)| = 1/(2ξK)."""
        eta = 2.0 * self.zeta
        U = solve_harmonic_hysteretic(self.K1, self.M1, eta, self.F1, self.freqs)
        np.testing.assert_allclose(abs(U[0, 0]), self.expected, rtol=1e-10)

    def test_modal_resonance(self) -> None:
        """Modal ξ₁=ξ : |H(ωₙ)| = 1/(2ξK)."""
        phi_free = np.array([[1.0 / math.sqrt(M_SDOF)]])
        omega_n  = np.array([OMEGA_N])
        zeta_n   = np.array([self.zeta])
        U = solve_harmonic_modal(omega_n, zeta_n, phi_free, self.F1, self.freqs)
        np.testing.assert_allclose(abs(U[0, 0]), self.expected, rtol=1e-10)

    def test_all_three_agree(self) -> None:
        """Les trois modèles avec le même ξ donnent la même amplitude (rtol=1e-8)."""
        alpha    = 2.0 * self.zeta * OMEGA_N
        C1       = csr_matrix(np.array([[alpha * M_SDOF]]))
        H_ray    = abs(solve_harmonic(self.K1, self.M1, C1, self.F1, self.freqs)[0, 0])

        H_hys    = abs(solve_harmonic_hysteretic(
            self.K1, self.M1, 2.0 * self.zeta, self.F1, self.freqs
        )[0, 0])

        phi_free = np.array([[1.0 / math.sqrt(M_SDOF)]])
        H_mod    = abs(solve_harmonic_modal(
            np.array([OMEGA_N]), np.array([self.zeta]),
            phi_free, self.F1, self.freqs,
        )[0, 0])

        np.testing.assert_allclose(H_hys, H_ray, rtol=1e-8,
                                   err_msg="Hystérétique ≠ Rayleigh à la résonance")
        np.testing.assert_allclose(H_mod, H_ray, rtol=1e-8,
                                   err_msg="Modal ≠ Rayleigh à la résonance")

    def test_different_zeta_values(self) -> None:
        """L'équivalence tient pour plusieurs valeurs de ξ."""
        for zeta in [0.01, 0.02, 0.05, 0.10]:
            expected = 1.0 / (2.0 * zeta * K_SDOF)

            alpha = 2.0 * zeta * OMEGA_N
            C1    = csr_matrix(np.array([[alpha * M_SDOF]]))
            H_ray = abs(solve_harmonic(self.K1, self.M1, C1, self.F1, self.freqs)[0, 0])

            H_hys = abs(solve_harmonic_hysteretic(
                self.K1, self.M1, 2.0 * zeta, self.F1, self.freqs
            )[0, 0])

            phi_free = np.array([[1.0 / math.sqrt(M_SDOF)]])
            H_mod    = abs(solve_harmonic_modal(
                np.array([OMEGA_N]), np.array([zeta]),
                phi_free, self.F1, self.freqs,
            )[0, 0])

            np.testing.assert_allclose(H_ray, expected, rtol=1e-10,
                                       err_msg=f"Rayleigh, ξ={zeta}")
            np.testing.assert_allclose(H_hys, expected, rtol=1e-10,
                                       err_msg=f"Hystérétique, ξ={zeta}")
            np.testing.assert_allclose(H_mod, expected, rtol=1e-10,
                                       err_msg=f"Modal, ξ={zeta}")


# ===========================================================================
# 6. Différences hors résonance (caractère fréquence-indépendant vs visqueux)
# ===========================================================================

class TestFrequencyDependenceDifferences:
    """L'amortissement hystérétique est indépendant de Ω, Rayleigh ne l'est pas.

    Ces tests documentent le comportement physique différent loin de la résonance.
    """

    def setup_method(self) -> None:
        self.K1, self.M1, self.F1 = _make_1x1()

    def test_rayleigh_alpha_more_damping_at_high_frequency(self) -> None:
        """À haute fréquence Ω ≫ ωₙ, Rayleigh α amortit plus que l'hystérétique.

        Rayleigh α : force amortissante dans Z = iΩ·αM.  Elle croît avec Ω,
        donc |Im(Z_ray)| = αΩM >> ηK = |Im(Z_hys)| pour Ω → ∞.
        → |H|_rayleigh < |H|_hysteretic loin de la résonance.

        Calcul à Ω = 20ωₙ, ξ = 5% :
            |Im(Z_ray)| = αΩM = 2ξωₙ·20ωₙ·M = 40ξK
            |Im(Z_hys)| = ηK  = 2ξK
        Le terme Rayleigh est 20× plus grand → amplitude plus faible.
        """
        zeta = 0.05
        alpha = 2.0 * zeta * OMEGA_N
        C1    = csr_matrix(np.array([[alpha * M_SDOF]]))

        freqs_high = np.array([20.0 * F_N])   # Ω = 20·ωₙ

        H_ray = abs(solve_harmonic(
            self.K1, self.M1, C1, self.F1, freqs_high
        )[0, 0])
        H_hys = abs(solve_harmonic_hysteretic(
            self.K1, self.M1, 2.0 * zeta, self.F1, freqs_high
        )[0, 0])

        # Rayleigh α fournit plus d'amortissement → amplitude plus faible
        assert H_ray < H_hys, (
            f"À 20·fₙ, H_ray={H_ray:.3e} devrait être < H_hys={H_hys:.3e}"
        )

    def test_hysteretic_phase_independent_of_frequency(self) -> None:
        """La phase hystérétique (hors résonance) est quasi-constante.

        Pour Ω ≪ ωₙ, la phase de H_hysteretic ≈ −arctan(η) constante,
        indépendante de Ω.  La phase visqueuse varie avec Ω.
        """
        eta = 0.10
        freqs_low = np.array([0.01 * F_N, 0.05 * F_N, 0.1 * F_N])
        U = solve_harmonic_hysteretic(self.K1, self.M1, eta, self.F1, freqs_low)
        phases = np.degrees(np.angle(U[0, :]))

        # Phase attendue à Ω ≪ ωₙ : −arctan(η) ≈ −5.71°
        expected_phase = -math.degrees(math.atan(eta))
        np.testing.assert_allclose(phases, expected_phase, atol=0.1)


# ===========================================================================
# 7. run_harmonic — intégration avec les nouveaux modèles
# ===========================================================================

class TestRunHarmonicNewDamping:
    """Tests d'intégration de run_harmonic avec HystereticDamping et ModalDampingModel."""

    def setup_method(self) -> None:
        n_elem = 4
        self.mesh, self.bc, self.dof_tip = _make_cantilever(n_elem)
        self.F_hat = np.zeros(self.mesh.n_dof)
        self.F_hat[self.dof_tip] = 1.0

        self.modal_result = run_modal(self.mesh, self.bc, n_modes=3)
        self.omega_n = self.modal_result.omega
        self.freq_n  = self.modal_result.freqs

    def test_hysteretic_result_shape(self) -> None:
        """run_harmonic avec HystereticDamping retourne la bonne forme."""
        freqs = np.linspace(1.0, 3.0 * self.freq_n[0], 50)
        d     = HystereticDamping(eta=0.04)
        r     = run_harmonic(self.mesh, self.bc, self.F_hat, freqs, d)
        assert r.U.shape == (self.mesh.n_dof, 50)

    def test_modal_result_shape(self) -> None:
        """run_harmonic avec ModalDampingModel retourne la bonne forme."""
        freqs = np.linspace(1.0, 3.0 * self.freq_n[0], 50)
        d     = ModalDampingModel.from_modal_result(self.modal_result, zeta_n=0.02)
        r     = run_harmonic(self.mesh, self.bc, self.F_hat, freqs, d)
        assert r.U.shape == (self.mesh.n_dof, 50)

    def test_hysteretic_constrained_dofs_zero(self) -> None:
        """Les DDL bloqués ont U = 0 avec amortissement hystérétique."""
        freqs = np.array([self.freq_n[0]])
        d     = HystereticDamping(eta=0.04)
        r     = run_harmonic(self.mesh, self.bc, self.F_hat, freqs, d)
        np.testing.assert_allclose(r.U[:3, :], 0.0)

    def test_modal_constrained_dofs_zero(self) -> None:
        """Les DDL bloqués ont U = 0 avec amortissement modal."""
        freqs = np.array([self.freq_n[0]])
        d     = ModalDampingModel.from_modal_result(self.modal_result, zeta_n=0.02)
        r     = run_harmonic(self.mesh, self.bc, self.F_hat, freqs, d)
        np.testing.assert_allclose(r.U[:3, :], 0.0)

    def test_resonance_amplitude_equivalent_hysteretic_vs_rayleigh(self) -> None:
        """À la première résonance, hystérétique (η=2ξ) ≈ Rayleigh (ξ).

        Tolérance plus large (1%) car les modes off-resonance contribuent
        différemment dans les deux formulations pour un système multi-DDL.
        """
        from femsolver.dynamics.rayleigh import rayleigh_from_modes
        zeta = 0.02

        # Rayleigh calibré sur modes 1-2 à ξ=2%
        d_ray = rayleigh_from_modes(
            self.omega_n[0], self.omega_n[1], zeta1=zeta, zeta2=zeta
        )
        freqs = np.array([self.freq_n[0]])

        r_ray = run_harmonic(self.mesh, self.bc, self.F_hat, freqs, d_ray)
        r_hys = run_harmonic(self.mesh, self.bc, self.F_hat, freqs,
                              HystereticDamping(eta=2.0 * zeta))

        amp_ray = abs(r_ray.U[self.dof_tip, 0])
        amp_hys = abs(r_hys.U[self.dof_tip, 0])

        # Les contributions off-résonance sont légèrement différentes (≤ 1%)
        np.testing.assert_allclose(amp_hys, amp_ray, rtol=0.01)

    def test_resonance_amplitude_equivalent_modal_vs_rayleigh(self) -> None:
        """À la première résonance, modal ≈ Rayleigh pour ξ uniforme."""
        from femsolver.dynamics.rayleigh import rayleigh_from_modes
        zeta = 0.02

        d_ray   = rayleigh_from_modes(
            self.omega_n[0], self.omega_n[1], zeta1=zeta, zeta2=zeta
        )
        d_modal = ModalDampingModel.from_modal_result(self.modal_result, zeta_n=zeta)
        freqs   = np.array([self.freq_n[0]])

        r_ray   = run_harmonic(self.mesh, self.bc, self.F_hat, freqs, d_ray)
        r_modal = run_harmonic(self.mesh, self.bc, self.F_hat, freqs, d_modal)

        amp_ray   = abs(r_ray.U[self.dof_tip, 0])
        amp_modal = abs(r_modal.U[self.dof_tip, 0])

        np.testing.assert_allclose(amp_modal, amp_ray, rtol=0.01)

    def test_unsupported_damping_type_raises(self) -> None:
        """Un type non supporté lève TypeError."""
        freqs = np.array([1.0])
        with pytest.raises(TypeError, match="non supporté"):
            run_harmonic(self.mesh, self.bc, self.F_hat, freqs, damping="invalid")

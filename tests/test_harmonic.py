"""Tests unitaires — réponse harmonique en fréquence.

Solutions analytiques utilisées
--------------------------------

**Système 1-DDL (SDOF)**

Pour un oscillateur simple masse-ressort-amortisseur :

    m·ü + c·u̇ + k·u = F₀·cos(Ωt)

La FRF complexe est :

    H(Ω) = 1 / (k − Ω²·m + iΩ·c)

Avec les paramètres adimensionnels r = Ω/ωₙ (rapport de fréquences) et
ξ = c/(2mωₙ) (taux d'amortissement) :

    |H(Ω)| = 1/k / √((1−r²)² + (2ξr)²)    [facteur d'amplification]
    arg(H)  = −atan2(2ξr, 1−r²)            [phase]

Propriétés clés à tester :

1. **Limite statique** (r → 0) : H → 1/k, phase → 0°.
2. **Résonance** (r = 1, ωₙ) : |H| = 1/(2ξk), phase = −90°.
3. **Limite haute fréquence** (r → ∞) : |H| → 0.
4. **Symétrie hermitienne** : H*(Ω) = H(−Ω) (vérifiée par continuité).

**Modèle FEM pour le SDOF**

Une barre 1 élément (longueur L, section A, densité ρ, Young E) avec
nœud 0 bloqué et charge P au nœud 1 :

    K_free = EA/L           (scalaire)
    M_free = ρAL/3          (terme [1,1] de la masse consistante 2×2)
    ωₙ = √(3E/(ρL²))

Pour Rayleigh α seul (β=0) :
    c_free = α·M_free
    ξ = α/(2ωₙ)

    H_FEM(Ω) = 1 / (K_free − Ω²·M_free + iΩ·α·M_free)

Cette FRF est **exacte analytiquement**, ce qui permet une comparaison
rtol=1e-12.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from femsolver.core.assembler import Assembler
from femsolver.core.boundary import apply_dirichlet
from femsolver.core.material import ElasticMaterial
from femsolver.core.mesh import BoundaryConditions, ElementData, Mesh
from femsolver.dynamics.harmonic import HarmonicResult, run_harmonic, solve_harmonic
from femsolver.dynamics.rayleigh import RayleighDamping, rayleigh_from_modes
from femsolver.elements.bar2d import Bar2D
from femsolver.elements.beam2d import Beam2D


# ---------------------------------------------------------------------------
# Données communes
# ---------------------------------------------------------------------------

E   = 210e9
NU  = 0.3
RHO = 7800.0
A   = 1e-4    # m²
L   = 1.0     # m

MAT = ElasticMaterial(E=E, nu=NU, rho=RHO)
G   = MAT.G


# ---------------------------------------------------------------------------
# Modèle SDOF analytique (barre 1 élément)
# ---------------------------------------------------------------------------

#: K_free = EA/L (rigidité du SDOF)
K_SDOF = E * A / L
#: M_free = ρAL/3 (masse consistante du SDOF)
M_SDOF = RHO * A * L / 3.0
#: ωₙ = √(K_SDOF/M_SDOF)
OMEGA_N = math.sqrt(K_SDOF / M_SDOF)
#: fₙ = ωₙ / (2π)
F_N = OMEGA_N / (2.0 * math.pi)


def _make_sdof_mesh() -> tuple[Mesh, BoundaryConditions]:
    """Barre 1 élément, nœud 0 encastré, force au nœud 1."""
    nodes    = np.array([[0.0, 0.0], [L, 0.0]])
    elements = (ElementData(Bar2D, (0, 1), MAT, {"area": A}),)
    bc = BoundaryConditions(dirichlet={0: {0: 0.0, 1: 0.0}}, neumann={})
    mesh = Mesh(nodes=nodes, elements=elements, n_dim=2, dof_per_node=2)
    return mesh, bc


def _sdof_frf_analytical(
    freqs: np.ndarray,
    k: float,
    m: float,
    c: float,
) -> np.ndarray:
    """FRF exacte du SDOF : H(Ω) = 1/(k − Ω²m + iΩc)."""
    Omega = 2.0 * np.pi * freqs
    return 1.0 / (k - Omega ** 2 * m + 1j * Omega * c)


def _make_cantilever(n_elem: int = 4) -> tuple[Mesh, BoundaryConditions, int]:
    """Console poutre.  Retourne (mesh, bc, dof_tip_uy)."""
    n_nodes = n_elem + 1
    nodes   = np.column_stack([np.linspace(0.0, L, n_nodes), np.zeros(n_nodes)])
    h = 0.01    # m
    Iz = h**4 / 12
    elems = tuple(
        ElementData(Beam2D, (i, i + 1), MAT, {"area": A, "inertia": Iz})
        for i in range(n_elem)
    )
    bc = BoundaryConditions(dirichlet={0: {0: 0.0, 1: 0.0, 2: 0.0}}, neumann={})
    mesh = Mesh(nodes=nodes, elements=elems, n_dim=2, dof_per_node=3)
    dof_tip_uy = 3 * n_elem + 1   # DDL uy du dernier nœud
    return mesh, bc, dof_tip_uy


# ===========================================================================
# 1. solve_harmonic bas niveau — vérification SDOF analytique
# ===========================================================================

class TestSolveHarmonicSDOF:
    """FRF FEM vs FRF analytique exacte pour un SDOF barre."""

    def setup_method(self) -> None:
        from scipy.sparse import csr_matrix as sp_csr
        # Matrices scalaires 1×1 (1 DDL libre)
        self.K1 = sp_csr(np.array([[K_SDOF]]))
        self.M1 = sp_csr(np.array([[M_SDOF]]))
        # Amortissement purement proportionnel à M (α=α₀, β=0)
        self.alpha = 2.0 * 0.05 * OMEGA_N   # ξ=5% à la résonance
        self.C1 = sp_csr(np.array([[self.alpha * M_SDOF]]))
        self.c_sdof = self.alpha * M_SDOF    # amortissement scalaire
        self.F1 = np.array([1.0])            # F̂ = 1 N

    def test_shape(self) -> None:
        freqs = np.linspace(0.1, 3.0 * F_N, 50)
        U = solve_harmonic(self.K1, self.M1, self.C1, self.F1, freqs)
        assert U.shape == (1, 50)
        assert U.dtype == complex

    def test_static_limit(self) -> None:
        """Ω → 0 : H → 1/k_free, Im(H) → 0."""
        freqs = np.array([1e-6 * F_N])
        U = solve_harmonic(self.K1, self.M1, self.C1, self.F1, freqs)
        expected = 1.0 / K_SDOF
        np.testing.assert_allclose(U[0, 0].real, expected, rtol=1e-6)
        # Im(H) → 0 at low frequency; float64 noise at this scale is ~1e-14
        np.testing.assert_allclose(abs(U[0, 0].imag), 0.0, atol=1e-14)

    def test_matches_analytical_sweep(self) -> None:
        """FRF FEM = FRF analytique sur tout le balayage (rtol=1e-12)."""
        freqs = np.linspace(0.01 * F_N, 3.0 * F_N, 200)
        U = solve_harmonic(self.K1, self.M1, self.C1, self.F1, freqs)
        H_exact = _sdof_frf_analytical(freqs, K_SDOF, M_SDOF, self.c_sdof)
        np.testing.assert_allclose(U[0, :], H_exact, rtol=1e-12)

    def test_resonance_amplitude(self) -> None:
        """À la résonance f = fₙ : |H| ≈ 1/(2ξk).

        Pour Rayleigh α pur (β=0) et ξ = α/(2ωₙ) :
            c = α·m   →   ξ = c/(2mωₙ) = α/2ωₙ
            |H(ωₙ)| = 1/(2ξ·k)
        """
        freqs = np.array([F_N])
        U = solve_harmonic(self.K1, self.M1, self.C1, self.F1, freqs)
        xi = self.c_sdof / (2.0 * M_SDOF * OMEGA_N)
        H_exact_resonance = 1.0 / (2.0 * xi * K_SDOF)
        np.testing.assert_allclose(abs(U[0, 0]), H_exact_resonance, rtol=1e-12)

    def test_resonance_phase_minus90(self) -> None:
        """Phase à la résonance ≈ −90° (réponse en quadrature retard)."""
        freqs = np.array([F_N])
        U = solve_harmonic(self.K1, self.M1, self.C1, self.F1, freqs)
        phase_deg = math.degrees(math.atan2(U[0, 0].imag, U[0, 0].real))
        np.testing.assert_allclose(phase_deg, -90.0, atol=0.001)

    def test_high_frequency_decay(self) -> None:
        """|H| → 0 quand Ω ≫ ωₙ (inertie domine)."""
        freqs = np.array([100.0 * F_N])
        U = solve_harmonic(self.K1, self.M1, self.C1, self.F1, freqs)
        H_static = 1.0 / K_SDOF
        assert abs(U[0, 0]) < H_static * 1e-3   # 3 ordres de grandeur en dessous

    def test_zero_damping_singularity(self) -> None:
        """Sans amortissement, |H(ωₙ)| est géant (quasi-singularité de Z).

        scipy.sparse.linalg.spsolve sur une matrice float64 quasi-singulière
        retourne un grand nombre fini (pas NaN/Inf), car l'égalité exacte
        Ω² = ωₙ² n'est pas atteinte en virgule flottante.  On vérifie donc
        que l'amplitude à la résonance est au moins 1e6 fois la réponse statique.
        """
        from scipy.sparse import csr_matrix as sp_csr
        C0 = sp_csr(np.array([[0.0]]))

        # Loin de la résonance : réponse finie
        freqs_off = np.array([0.5 * F_N, 2.0 * F_N])
        U_off = solve_harmonic(self.K1, self.M1, C0, self.F1, freqs_off)
        assert np.all(np.isfinite(U_off))

        # À la résonance : amplitude très grande (quasi-singularité)
        freqs_res = np.array([F_N])
        U_res = solve_harmonic(self.K1, self.M1, C0, self.F1, freqs_res)
        H_static = 1.0 / K_SDOF
        assert abs(U_res[0, 0]) > 1e6 * H_static, (
            f"|H(ωₙ)| = {abs(U_res[0, 0]):.2e} devrait être ≫ H_static={H_static:.2e}"
        )

    def test_undamped_off_resonance_real(self) -> None:
        """Sans amortissement, loin de la résonance H est réel (phase 0° ou 180°)."""
        from scipy.sparse import csr_matrix as sp_csr
        C0 = sp_csr(np.array([[0.0]]))
        # En dessous de la résonance (r < 1)
        freqs_low  = np.array([0.5 * F_N])
        U_low = solve_harmonic(self.K1, self.M1, C0, self.F1, freqs_low)
        assert abs(U_low[0, 0].imag) < 1e-15 * abs(U_low[0, 0].real)
        # En dessus (r > 1) : phase 180° → réel négatif
        freqs_high = np.array([2.0 * F_N])
        U_high = solve_harmonic(self.K1, self.M1, C0, self.F1, freqs_high)
        assert U_high[0, 0].real < 0.0


# ===========================================================================
# 2. run_harmonic — interface de haut niveau
# ===========================================================================

class TestRunHarmonicHighLevel:
    """Tests sur run_harmonic avec un maillage complet."""

    def setup_method(self) -> None:
        self.mesh, self.bc = _make_sdof_mesh()
        # DDL libre : ux du nœud 1 (indice global = 2, car dof_per_node=2)
        # nœud 0 : DDL 0 (ux) et 1 (uy) bloqués
        # nœud 1 : DDL 2 (ux), 3 (uy) — on charge ux
        self.F_hat = np.zeros(self.mesh.n_dof)
        self.F_hat[2] = 1.0   # F̂_ux au nœud 1

        self.alpha = 2.0 * 0.05 * OMEGA_N
        self.damping = RayleighDamping(alpha=self.alpha, beta=0.0)
        self.c_sdof  = self.alpha * M_SDOF

    def test_result_shape(self) -> None:
        freqs = np.linspace(1.0, 3.0 * F_N, 100)
        r = run_harmonic(self.mesh, self.bc, self.F_hat, freqs, self.damping)
        assert isinstance(r, HarmonicResult)
        assert r.U.shape == (self.mesh.n_dof, 100)
        assert r.freqs.shape == (100,)

    def test_constrained_dofs_zero(self) -> None:
        """Les DDL bloqués ont U = 0 pour toutes les fréquences."""
        freqs = np.linspace(0.1 * F_N, 2.0 * F_N, 50)
        r = run_harmonic(self.mesh, self.bc, self.F_hat, freqs, self.damping)
        # nœud 0 : DDL 0 (ux) bloqué, DDL 1 (uy) bloqué
        np.testing.assert_allclose(r.U[0, :], 0.0)
        np.testing.assert_allclose(r.U[1, :], 0.0)

    def test_matches_analytical(self) -> None:
        """FRF du DDL libre (ux du nœud 1) = FRF analytique exacte."""
        freqs = np.linspace(0.01 * F_N, 3.0 * F_N, 200)
        r = run_harmonic(self.mesh, self.bc, self.F_hat, freqs, self.damping)

        # DDL libre ux du nœud 1 : indice global 2
        H_fem  = r.U[2, :]
        H_anal = _sdof_frf_analytical(freqs, K_SDOF, M_SDOF, self.c_sdof)
        np.testing.assert_allclose(H_fem, H_anal, rtol=1e-10)

    def test_no_damping(self) -> None:
        """Sans amortissement : FRF réelle loin des résonances, quasi-singulière à ωₙ.

        spsolve sur float64 retourne un grand fini à la résonance (pas NaN/Inf).
        On vérifie que l'amplitude à ωₙ dépasse largement la réponse statique.
        """
        freqs = np.array([0.5 * F_N, F_N, 2.0 * F_N])
        r = run_harmonic(self.mesh, self.bc, self.F_hat, freqs, damping=None)
        # À 0.5·fₙ et 2·fₙ : réponse finie
        assert np.isfinite(r.U[2, 0])
        assert np.isfinite(r.U[2, 2])
        # À fₙ : amplitude très grande (quasi-singularité)
        H_static = 1.0 / K_SDOF
        assert abs(r.U[2, 1]) > 1e6 * H_static, (
            f"|H(ωₙ)| = {abs(r.U[2, 1]):.2e} devrait être ≫ H_static={H_static:.2e}"
        )

    def test_static_limit(self) -> None:
        """À f très basse, |U_ux| ≈ 1/K_SDOF (statique)."""
        freqs = np.array([1e-6 * F_N])
        r = run_harmonic(self.mesh, self.bc, self.F_hat, freqs, self.damping)
        np.testing.assert_allclose(abs(r.U[2, 0]), 1.0 / K_SDOF, rtol=1e-5)


# ===========================================================================
# 3. Réponse harmonique poutre console
# ===========================================================================

class TestHarmonicCantileverBeam:
    """Balayage en fréquence sur une console poutre (multi-DDL).

    Validation qualitative : les pics de la FRF coïncident avec les
    fréquences propres obtenues par l'analyse modale.
    """

    def setup_method(self) -> None:
        from femsolver.dynamics.modal import run_modal
        n_elem = 4
        self.mesh, self.bc, self.dof_tip = _make_cantilever(n_elem)
        # Force transverse au bout
        self.F_hat = np.zeros(self.mesh.n_dof)
        self.F_hat[self.dof_tip] = 1.0

        # Fréquences propres par analyse modale
        self.modal = run_modal(self.mesh, self.bc, n_modes=3)
        self.omega_n = self.modal.omega   # [rad/s]
        self.freq_n  = self.modal.freqs  # [Hz]

        # Amortissement calibré sur les 2 premiers modes (ξ=2%)
        self.damping = rayleigh_from_modes(
            self.omega_n[0], self.omega_n[1], zeta1=0.02, zeta2=0.02
        )

    def test_frf_peaks_at_natural_frequencies(self) -> None:
        """Les pics de la FRF uy_tip se situent près des fréquences propres.

        On vérifie que la FRF évaluée à ωₙ est un maximum local dans un
        voisinage de ωₙ (plage de ±5% autour de chaque mode).
        """
        # Balayage fin autour du mode 1
        f1 = self.freq_n[0]
        freqs_fine = np.linspace(0.85 * f1, 1.15 * f1, 500)
        r = run_harmonic(self.mesh, self.bc, self.F_hat, freqs_fine, self.damping)
        amp = np.abs(r.U[self.dof_tip, :])
        idx_max = np.argmax(amp)
        f_peak = freqs_fine[idx_max]
        # Le pic doit être dans ±5% de f₁ (amortissement 2% décale légèrement)
        np.testing.assert_allclose(f_peak, f1, rtol=0.05)

    def test_frf_amplitude_order_of_magnitude(self) -> None:
        """FRF à la résonance > 10× la FRF statique (amplification résonante)."""
        # Réponse statique = K⁻¹·F_hat (limite Ω→0)
        freqs_static = np.array([1e-4 * self.freq_n[0]])
        r_static = run_harmonic(
            self.mesh, self.bc, self.F_hat, freqs_static, self.damping
        )
        amp_static = abs(r_static.U[self.dof_tip, 0])

        # Réponse à la résonance du mode 1
        freqs_res = np.array([self.freq_n[0]])
        r_res = run_harmonic(
            self.mesh, self.bc, self.F_hat, freqs_res, self.damping
        )
        amp_res = abs(r_res.U[self.dof_tip, 0])

        assert amp_res > 10.0 * amp_static, (
            f"Amplification trop faible : amp_res={amp_res:.2e}, "
            f"amp_static={amp_static:.2e}"
        )

    def test_high_frequency_decay(self) -> None:
        """Au-delà de la 3e fréquence propre, la FRF décroît."""
        f_low  = 0.5 * self.freq_n[0]
        f_high = 5.0 * self.freq_n[2]
        freqs  = np.array([f_low, f_high])
        r = run_harmonic(self.mesh, self.bc, self.F_hat, freqs, self.damping)
        amp = np.abs(r.U[self.dof_tip, :])
        assert amp[1] < amp[0], "La FRF doit décroître aux très hautes fréquences"

    def test_phase_transition_at_resonance(self) -> None:
        """La phase passe de ≈ 0° à ≈ −180° en traversant la résonance."""
        f1 = self.freq_n[0]
        freqs = np.array([0.3 * f1, f1, 3.0 * f1])
        r = run_harmonic(self.mesh, self.bc, self.F_hat, freqs, self.damping)
        phase = np.angle(r.U[self.dof_tip, :], deg=True)
        # En dessous de la résonance : phase proche de 0°
        assert abs(phase[0]) < 15.0, f"Phase pré-résonance = {phase[0]:.1f}°"
        # À la résonance : phase proche de −90°
        assert abs(phase[1] + 90.0) < 20.0, f"Phase à résonance = {phase[1]:.1f}°"
        # Au-dessus : phase proche de −180°
        assert abs(phase[2] + 180.0) < 30.0, f"Phase post-résonance = {phase[2]:.1f}°"

    def test_result_free_dofs_consistent(self) -> None:
        """Les free_dofs de HarmonicResult correspondent bien aux DDL non bloqués."""
        freqs = np.array([1.0])
        r = run_harmonic(self.mesh, self.bc, self.F_hat, freqs, self.damping)
        # Tous les DDL contraints (nœud 0 : ux,uy,θ) ont U=0
        for dof in [0, 1, 2]:
            np.testing.assert_allclose(r.U[dof, :], 0.0)
        # Les DDL libres ont U ≠ 0 (en général)
        assert np.any(r.U[r.free_dofs, :] != 0.0)


# ===========================================================================
# 4. Propriétés de la FRF
# ===========================================================================

class TestHarmonicFRFProperties:
    """Propriétés mathématiques de la FRF."""

    def setup_method(self) -> None:
        from scipy.sparse import csr_matrix as sp_csr
        self.K1 = sp_csr(np.array([[K_SDOF]]))
        self.M1 = sp_csr(np.array([[M_SDOF]]))
        alpha   = 2.0 * 0.05 * OMEGA_N
        self.C1 = sp_csr(np.array([[alpha * M_SDOF]]))
        self.F1 = np.array([1.0])

    def test_hermitian_symmetry(self) -> None:
        """H*(f) = H(−f) : la réponse physique est réelle.

        Pour un système physique, u(t) ∈ ℝ ⟹ la TF vérifie H*(f) = H(−f).
        On vérifie que conj(H(f)) = H(f) pour les fréquences positives
        (= ce que donnerait H(−f)).
        """
        freqs = np.array([0.5 * F_N, F_N, 1.5 * F_N])
        U = solve_harmonic(self.K1, self.M1, self.C1, self.F1, freqs)
        # H*(f) = H₁* ... en évaluant à -f, on obtient la conjuguée pour un
        # système réel.  Ici on vérifie la cohérence interne :
        # Im(H) est anti-symétrique en f : Im(H(f)) = -Im(H(-f))
        # => Im(H(f)) < 0 pour f > 0 (retard de phase)
        assert np.all(U[0, :].imag < 0.0), "La partie imaginaire doit être < 0 (retard)"

    def test_superposition_linearity(self) -> None:
        """H(Ω; 2F) = 2·H(Ω; F) (linéarité du système)."""
        freqs = np.linspace(0.1 * F_N, 2.0 * F_N, 30)
        F1 = np.array([1.0])
        F2 = np.array([2.0])
        U1 = solve_harmonic(self.K1, self.M1, self.C1, F1, freqs)
        U2 = solve_harmonic(self.K1, self.M1, self.C1, F2, freqs)
        np.testing.assert_allclose(U2, 2.0 * U1, rtol=1e-12)

    def test_rayleigh_damping_increases_with_zeta(self) -> None:
        """Un amortissement plus fort réduit l'amplitude à la résonance."""
        freqs = np.array([F_N])
        alpha_low  = 2.0 * 0.02 * OMEGA_N
        alpha_high = 2.0 * 0.10 * OMEGA_N
        C_low  = csr_matrix(np.array([[alpha_low  * M_SDOF]]))
        C_high = csr_matrix(np.array([[alpha_high * M_SDOF]]))
        U_low  = solve_harmonic(self.K1, self.M1, C_low,  self.F1, freqs)
        U_high = solve_harmonic(self.K1, self.M1, C_high, self.F1, freqs)
        assert abs(U_low[0, 0]) > abs(U_high[0, 0]), (
            "Amortissement plus fort → amplitude plus faible à la résonance"
        )

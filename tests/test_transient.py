"""Tests unitaires — intégration temporelle Newmark-β.

Solutions analytiques utilisées
---------------------------------

**Oscillateur 1-DDL amorti (SDOF)**

    m·ü + c·u̇ + k·u = F(t)

Paramètres adimensionnels :

    ωₙ = √(k/m)                          pulsation naturelle [rad/s]
    ξ  = c / (2·m·ωₙ)                    taux d'amortissement [-]
    ωd = ωₙ·√(1 − ξ²)                    pulsation amortie [rad/s]

**Vibration libre amortie** (F = 0, u(0) = u₀, v(0) = v₀, 0 < ξ < 1) :

    u(t) = e^(−ξωₙt) · [u₀·cos(ωd·t) + (v₀ + ξωₙ·u₀)/ωd · sin(ωd·t)]

**Réponse à un échelon** (F(t) = F₀·H(t), u(0) = 0, v(0) = 0) :

    u_st = F₀/k                          déplacement statique
    u(t) = u_st · [1 − e^(−ξωₙt) · (cos(ωd·t) + ξ/√(1−ξ²)·sin(ωd·t))]

**Stabilité explicite** (β = 0, différences centrales) :

    Ω_crit = 2  ⟹  Δt_crit = 2/ωₙ

Références
----------
Newmark N.M., ASCE, 1959.
Hughes T.J.R., The Finite Element Method, §9.2–9.3, Prentice-Hall, 1987.
Bathe K.J., Finite Element Procedures, §9.4, 2nd ed., 2014.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from femsolver.dynamics.transient import (
    NEWMARK_CENTRAL_DIFF,
    NEWMARK_FOX_GOODWIN,
    NEWMARK_LINEAR_ACCEL,
    NEWMARK_TRAPEZOIDAL,
    NewmarkBeta,
    solve_newmark,
)


# ---------------------------------------------------------------------------
# Helpers : construction d'un oscillateur 1-DDL
# ---------------------------------------------------------------------------


def _sdof_matrices(m: float, c: float, k: float):
    """Renvoie K_free, M_free, C_free comme (1×1) csr_matrix."""
    K = csr_matrix(np.array([[k]]))
    M = csr_matrix(np.array([[m]]))
    C = csr_matrix(np.array([[c]]))
    return K, M, C


def _free_vibration_analytical(
    t: np.ndarray, u0: float, v0: float, omega_n: float, xi: float
) -> np.ndarray:
    """Solution analytique vibration libre amortie sous-critique (ξ < 1).

    Parameters
    ----------
    t : np.ndarray
        Instants [s].
    u0, v0 : float
        Conditions initiales.
    omega_n : float
        Pulsation naturelle [rad/s].
    xi : float
        Taux d'amortissement [-], 0 < xi < 1.

    Returns
    -------
    np.ndarray
        Déplacement analytique à chaque instant.
    """
    omega_d = omega_n * np.sqrt(1.0 - xi ** 2)
    return np.exp(-xi * omega_n * t) * (
        u0 * np.cos(omega_d * t)
        + (v0 + xi * omega_n * u0) / omega_d * np.sin(omega_d * t)
    )


def _step_load_analytical(
    t: np.ndarray, F0: float, k: float, omega_n: float, xi: float
) -> np.ndarray:
    """Solution analytique réponse à un échelon F₀·H(t) (CI nulles, ξ < 1).

    u(t) = u_st · [1 − e^(−ξωₙt) · (cos(ωd·t) + ξ/√(1−ξ²)·sin(ωd·t))]
    """
    u_st = F0 / k
    omega_d = omega_n * np.sqrt(1.0 - xi ** 2)
    env = np.exp(-xi * omega_n * t)
    return u_st * (
        1.0 - env * (np.cos(omega_d * t) + xi / np.sqrt(1.0 - xi ** 2) * np.sin(omega_d * t))
    )


# ---------------------------------------------------------------------------
# Classe de tests — NewmarkBeta (paramètres)
# ---------------------------------------------------------------------------


class TestNewmarkBetaParams:
    """Vérification de la dataclass NewmarkBeta et ses méthodes."""

    def test_default_is_trapezoidal(self):
        p = NewmarkBeta()
        assert p.gamma == 0.5
        assert p.beta  == 0.25

    def test_trapezoidal_is_unconditionally_stable(self):
        assert NEWMARK_TRAPEZOIDAL.is_unconditionally_stable() is True

    def test_central_diff_is_conditionally_stable(self):
        assert NEWMARK_CENTRAL_DIFF.is_unconditionally_stable() is False

    def test_fox_goodwin_is_conditionally_stable(self):
        assert NEWMARK_FOX_GOODWIN.is_unconditionally_stable() is False

    def test_linear_accel_is_conditionally_stable(self):
        assert NEWMARK_LINEAR_ACCEL.is_unconditionally_stable() is False

    def test_critical_dt_trapezoidal_returns_none(self):
        assert NEWMARK_TRAPEZOIDAL.critical_dt(omega_max=100.0) is None

    def test_critical_dt_central_diff(self):
        """Δt_crit = 2/ωₙ pour les différences centrales (Ω_crit = 2)."""
        omega_n = 10.0
        dt_crit = NEWMARK_CENTRAL_DIFF.critical_dt(omega_n)
        assert dt_crit == pytest.approx(2.0 / omega_n, rel=1e-12)

    def test_critical_dt_fox_goodwin(self):
        """Ω_crit = 2 / √(1 − 4·β) pour β = 1/12."""
        omega_n = 10.0
        beta = 1.0 / 12.0
        expected = 2.0 / (omega_n * np.sqrt(1.0 - 4.0 * beta))
        assert NEWMARK_FOX_GOODWIN.critical_dt(omega_n) == pytest.approx(expected, rel=1e-12)

    def test_invalid_gamma_raises(self):
        with pytest.raises(ValueError, match="gamma"):
            NewmarkBeta(gamma=1.5, beta=0.25)

    def test_invalid_beta_raises(self):
        with pytest.raises(ValueError, match="beta"):
            NewmarkBeta(gamma=0.5, beta=0.6)


# ---------------------------------------------------------------------------
# Oscillateur 1-DDL sans amortissement — conservation de l'énergie
# ---------------------------------------------------------------------------


class TestConservativeOscillator:
    """Sans amortissement et sans force, l'énergie mécanique est constante.

    E = ½·m·v² + ½·k·u²  = constante.

    Le schéma trapèze (γ=½, β=¼) est exactement conservatif.
    """

    def setup_method(self):
        self.m   = 2.0       # kg
        self.k   = 50.0      # N/m
        self.u0  = 0.05      # m (CI non nulle)
        self.v0  = 0.0       # m/s
        self.omega_n = np.sqrt(self.k / self.m)   # ≈ 5 rad/s
        self.T_n = 2.0 * np.pi / self.omega_n

    def test_energy_conservation_trapezoidal(self):
        """Schéma trapèze (β=¼) : énergie constante à la précision machine."""
        K, M, C = _sdof_matrices(self.m, 0.0, self.k)
        n_steps = 2000
        dt = self.T_n / 200           # 200 pas par période

        u_h, v_h, _ = solve_newmark(
            K, M, C, None,
            np.array([self.u0]), np.array([self.v0]),
            dt, n_steps, NEWMARK_TRAPEZOIDAL,
        )

        E = 0.5 * self.k * u_h[0] ** 2 + 0.5 * self.m * v_h[0] ** 2
        np.testing.assert_allclose(E, E[0], rtol=1e-10,
                                   err_msg="Énergie non conservée (trapèze)")

    def test_amplitude_bounded_central_diff_subcritical(self):
        """Différences centrales sous Δt_crit : amplitude reste bornée (pas de divergence).

        Note : l'énergie calculée aux pas entiers oscille légèrement en schéma
        leapfrog implicite (vitesse corrigée au même instant que le déplacement).
        On vérifie donc que l'amplitude max ne dérive pas plutôt que l'énergie exacte.
        """
        K, M, C = _sdof_matrices(self.m, 0.0, self.k)
        dt_crit = NEWMARK_CENTRAL_DIFF.critical_dt(self.omega_n)
        dt = 0.5 * dt_crit        # bien en-dessous du critère
        n_steps = int(5 * self.T_n / dt)

        u_h, _, _ = solve_newmark(
            K, M, C, None,
            np.array([self.u0]), np.array([self.v0]),
            dt, n_steps, NEWMARK_CENTRAL_DIFF,
        )

        # L'amplitude ne doit pas gonfler (pas de divergence ni de dissipation parasite)
        amp_max = np.max(np.abs(u_h[0]))
        assert amp_max <= 1.05 * abs(self.u0), (
            f"Amplitude gonfle : {amp_max:.4f} > 1.05 × u0 = {1.05*self.u0:.4f}"
        )
        assert amp_max >= 0.95 * abs(self.u0), (
            f"Amplitude dissipée : {amp_max:.4f} < 0.95 × u0 = {0.95*self.u0:.4f}"
        )


# ---------------------------------------------------------------------------
# Vibration libre amortie — solution analytique
# ---------------------------------------------------------------------------


class TestFreeDampedVibration:
    """Vibration libre — comparaison avec u(t) = e^(−ξωₙt)·(…)."""

    # Paramètres physiques
    m     = 1.0        # kg
    k     = 100.0      # N/m  ⟹  ωₙ = 10 rad/s
    xi    = 0.05       # 5 % amortissement
    u0    = 0.02       # m
    v0    = 0.0        # m/s

    @classmethod
    def setup_class(cls):
        cls.omega_n = np.sqrt(cls.k / cls.m)         # 10 rad/s
        cls.c       = 2.0 * cls.xi * cls.omega_n * cls.m    # N·s/m
        cls.T_n     = 2.0 * np.pi / cls.omega_n

    def _run(self, params: NewmarkBeta, n_per_period: int = 500, n_periods: int = 5):
        K, M, C = _sdof_matrices(self.m, self.c, self.k)
        dt = self.T_n / n_per_period
        n_steps = n_per_period * n_periods

        u_h, _, _ = solve_newmark(
            K, M, C, None,
            np.array([self.u0]), np.array([self.v0]),
            dt, n_steps, params,
        )

        t = np.arange(n_steps + 1) * dt
        u_ana = _free_vibration_analytical(t, self.u0, self.v0, self.omega_n, self.xi)
        return u_h[0], u_ana, dt

    # -- Schéma trapèze -------------------------------------------------------

    def test_trapezoidal_free_vibration(self):
        """γ=½, β=¼ : erreur absolue < 0.1 % de l'amplitude avec 500 pas/période.

        On utilise atol (pas rtol) car l'erreur relative explose aux passages par zéro.
        """
        u_num, u_ana, _ = self._run(NEWMARK_TRAPEZOIDAL)
        np.testing.assert_allclose(u_num, u_ana, atol=1e-3 * self.u0,
                                   err_msg="Erreur trapèze trop grande")

    def test_trapezoidal_free_vibration_fine_dt(self):
        """Avec 2000 pas/période l'erreur absolue est < 1e-5 × amplitude."""
        u_num, u_ana, _ = self._run(NEWMARK_TRAPEZOIDAL, n_per_period=2000)
        # atol = 1e-5 × u0 = 2e-7 ; erreur attendue O((Δt/T)²) ≈ 1.6e-7
        np.testing.assert_allclose(u_num, u_ana, atol=1e-5 * self.u0,
                                   err_msg="Erreur trapèze (Δt fin) trop grande")

    def test_trapezoidal_order2_convergence(self):
        """Convergence ordre 2 : diviser Δt par 2 divise l'erreur par ~4."""
        def max_err(npp):
            u_num, u_ana, _ = self._run(NEWMARK_TRAPEZOIDAL, n_per_period=npp)
            return np.max(np.abs(u_num - u_ana))

        e1 = max_err(100)
        e2 = max_err(200)
        ratio = e1 / e2
        assert ratio == pytest.approx(4.0, abs=0.5), (
            f"Convergence ordre 2 attendue (ratio ≈ 4), obtenu {ratio:.2f}"
        )

    # -- Différences centrales ------------------------------------------------

    def test_central_diff_free_vibration(self):
        """β=0 : erreur absolue < 0.1 % de l'amplitude avec 500 pas/période."""
        u_num, u_ana, dt = self._run(NEWMARK_CENTRAL_DIFF)
        dt_crit = NEWMARK_CENTRAL_DIFF.critical_dt(self.omega_n)
        assert dt < dt_crit, "Δt dépasse Δt_crit — test invalide"
        np.testing.assert_allclose(u_num, u_ana, atol=1e-3 * self.u0,
                                   err_msg="Erreur diff. centrales trop grande")

    def test_central_diff_instability_above_crit(self):
        """β=0, Δt > Δt_crit : divergence rapide (norme croissante)."""
        K, M, C = _sdof_matrices(self.m, 0.0, self.k)   # sans amortissement
        dt_crit = NEWMARK_CENTRAL_DIFF.critical_dt(self.omega_n)
        dt = 1.5 * dt_crit          # au-delà du critère

        u_h, _, _ = solve_newmark(
            K, M, C, None,
            np.array([self.u0]), np.array([self.v0]),
            dt, n_steps=200, params=NEWMARK_CENTRAL_DIFF,
        )
        # La norme doit diverger (dépasse 10× la CI)
        assert np.max(np.abs(u_h[0])) > 10.0 * abs(self.u0), (
            "Le schéma explicite devrait diverger au-delà de Δt_crit"
        )

    # -- Fox-Goodwin ----------------------------------------------------------

    def test_fox_goodwin_free_vibration(self):
        """β=1/12, sous-critique : erreur absolue < 0.2 % de l'amplitude."""
        u_num, u_ana, dt = self._run(NEWMARK_FOX_GOODWIN)
        dt_crit = NEWMARK_FOX_GOODWIN.critical_dt(self.omega_n)
        assert dt < dt_crit, "Δt dépasse Δt_crit — test invalide"
        np.testing.assert_allclose(u_num, u_ana, atol=2e-3 * self.u0,
                                   err_msg="Erreur Fox-Goodwin trop grande")

    # -- Accélération linéaire ------------------------------------------------

    def test_linear_accel_free_vibration(self):
        """β=1/6, sous-critique : erreur absolue < 0.2 % de l'amplitude."""
        u_num, u_ana, dt = self._run(NEWMARK_LINEAR_ACCEL)
        dt_crit = NEWMARK_LINEAR_ACCEL.critical_dt(self.omega_n)
        assert dt < dt_crit, "Δt dépasse Δt_crit — test invalide"
        np.testing.assert_allclose(u_num, u_ana, atol=2e-3 * self.u0,
                                   err_msg="Erreur accél. linéaire trop grande")


# ---------------------------------------------------------------------------
# Réponse à un échelon de force
# ---------------------------------------------------------------------------


class TestStepLoadResponse:
    """Réponse à un échelon F(t) = F₀·H(t), CI nulles.

    Solution analytique (sous-amorti) :

        u_st = F₀/k
        u(t) = u_st·[1 − e^(−ξωₙt)·(cos(ωd·t) + ξ/√(1−ξ²)·sin(ωd·t))]
    """

    m    = 1.0       # kg
    k    = 100.0     # N/m
    xi   = 0.10      # 10 %
    F0   = 50.0      # N

    @classmethod
    def setup_class(cls):
        cls.omega_n = np.sqrt(cls.k / cls.m)
        cls.c       = 2.0 * cls.xi * cls.omega_n * cls.m
        cls.T_n     = 2.0 * np.pi / cls.omega_n

    def _run_step(self, params: NewmarkBeta, n_per_period: int = 500, n_periods: int = 5):
        K, M, C = _sdof_matrices(self.m, self.c, self.k)
        dt = self.T_n / n_per_period
        n_steps = n_per_period * n_periods
        F_fn = lambda t: np.array([self.F0])

        u_h, _, _ = solve_newmark(
            K, M, C, F_fn,
            np.zeros(1), np.zeros(1),
            dt, n_steps, params,
        )

        t = np.arange(n_steps + 1) * dt
        u_ana = _step_load_analytical(t, self.F0, self.k, self.omega_n, self.xi)
        return u_h[0], u_ana

    def test_trapezoidal_step_load(self):
        """Trapèze : réponse à l'échelon, erreur < 0.5 % (500 pas/période)."""
        u_num, u_ana = self._run_step(NEWMARK_TRAPEZOIDAL)
        np.testing.assert_allclose(u_num, u_ana, rtol=5e-3,
                                   err_msg="Erreur échelon (trapèze) trop grande")

    def test_central_diff_step_load(self):
        """Diff. centrales : réponse à l'échelon, erreur < 0.5 % (500 pas/période)."""
        u_num, u_ana = self._run_step(NEWMARK_CENTRAL_DIFF)
        np.testing.assert_allclose(u_num, u_ana, rtol=5e-3,
                                   err_msg="Erreur échelon (diff. centrales) trop grande")

    def test_step_load_static_limit(self):
        """u_max ≤ 2·u_static (dépassement max pour oscillateur sous-amorti)."""
        u_num, _ = self._run_step(NEWMARK_TRAPEZOIDAL)
        u_st = self.F0 / self.k
        assert np.max(u_num) <= 2.0 * u_st + 1e-10, (
            "Le dépassement dépasse 2·u_static, ce qui est impossible"
        )
        assert np.max(u_num) > u_st, (
            "Le déplacement maximal doit dépasser u_static (résonance transitoire)"
        )

    def test_step_load_steady_state(self):
        """En régime établi (t >> amortissement), u → F₀/k."""
        K, M, C = _sdof_matrices(self.m, self.c, self.k)
        dt = self.T_n / 500
        # Simuler longtemps pour atteindre le régime permanent
        n_steps = int(50 * self.T_n / dt)
        F_fn = lambda t: np.array([self.F0])

        u_h, _, _ = solve_newmark(
            K, M, C, F_fn,
            np.zeros(1), np.zeros(1),
            dt, n_steps, NEWMARK_TRAPEZOIDAL,
        )

        u_steady = np.mean(u_h[0, -100:])   # moyenne sur les 100 derniers pas
        np.testing.assert_allclose(u_steady, self.F0 / self.k, rtol=1e-3,
                                   err_msg="Déplacement statique permanent incorrect")


# ---------------------------------------------------------------------------
# Conditions initiales vitesse non nulle
# ---------------------------------------------------------------------------


class TestNonZeroInitialVelocity:
    """Vérifie la solution analytique pour v₀ ≠ 0."""

    m  = 1.0
    k  = 400.0     # ωₙ = 20 rad/s
    xi = 0.02

    @classmethod
    def setup_class(cls):
        cls.omega_n = np.sqrt(cls.k / cls.m)
        cls.c       = 2.0 * cls.xi * cls.omega_n * cls.m
        cls.T_n     = 2.0 * np.pi / cls.omega_n

    def test_trapezoidal_nonzero_v0(self):
        """CI u₀=0, v₀=1 m/s — trajectoire analytique.

        Amplitude max ≈ v₀/ωd ≈ 0.05 m ; tolérance absolue 5e-6 m.
        """
        u0, v0 = 0.0, 1.0
        K, M, C = _sdof_matrices(self.m, self.c, self.k)
        dt = self.T_n / 1000
        n_steps = 2000
        # Amplitude caractéristique : v0/omega_d
        omega_d = self.omega_n * np.sqrt(1.0 - self.xi ** 2)
        amp = v0 / omega_d

        u_h, _, _ = solve_newmark(
            K, M, C, None,
            np.array([u0]), np.array([v0]),
            dt, n_steps, NEWMARK_TRAPEZOIDAL,
        )

        t = np.arange(n_steps + 1) * dt
        u_ana = _free_vibration_analytical(t, u0, v0, self.omega_n, self.xi)
        np.testing.assert_allclose(u_h[0], u_ana, atol=1e-4 * amp)

    def test_trapezoidal_nonzero_u0_and_v0(self):
        """CI u₀=0.01 m, v₀=0.5 m/s."""
        u0, v0 = 0.01, 0.5
        K, M, C = _sdof_matrices(self.m, self.c, self.k)
        dt = self.T_n / 1000
        n_steps = 2000
        omega_d = self.omega_n * np.sqrt(1.0 - self.xi ** 2)
        amp = np.sqrt(u0 ** 2 + ((v0 + self.xi * self.omega_n * u0) / omega_d) ** 2)

        u_h, _, _ = solve_newmark(
            K, M, C, None,
            np.array([u0]), np.array([v0]),
            dt, n_steps, NEWMARK_TRAPEZOIDAL,
        )

        t = np.arange(n_steps + 1) * dt
        u_ana = _free_vibration_analytical(t, u0, v0, self.omega_n, self.xi)
        np.testing.assert_allclose(u_h[0], u_ana, atol=1e-4 * amp)


# ---------------------------------------------------------------------------
# Accélération initiale — équilibre dynamique au pas 0
# ---------------------------------------------------------------------------


class TestInitialAcceleration:
    """a₀ est déterminée par M·a₀ = F(0) − C·v₀ − K·u₀."""

    def test_initial_accel_no_damping_no_force(self):
        """CI u₀=0.1, v₀=0, F=0, C=0 : a₀ = −(k/m)·u₀."""
        m, k = 2.0, 80.0
        K, M, C = _sdof_matrices(m, 0.0, k)

        _, _, a_h = solve_newmark(
            K, M, C, None,
            np.array([0.1]), np.array([0.0]),
            dt=1e-3, n_steps=1, params=NEWMARK_TRAPEZOIDAL,
        )

        expected_a0 = -(k / m) * 0.1   # = −4.0 m/s²
        np.testing.assert_allclose(a_h[0, 0], expected_a0, rtol=1e-12)

    def test_initial_accel_with_force(self):
        """CI nulles, F₀ = 20 N : a₀ = F₀/m."""
        m, k = 2.0, 80.0
        K, M, C = _sdof_matrices(m, 0.0, k)
        F_fn = lambda t: np.array([20.0])

        _, _, a_h = solve_newmark(
            K, M, C, F_fn,
            np.zeros(1), np.zeros(1),
            dt=1e-3, n_steps=1, params=NEWMARK_TRAPEZOIDAL,
        )

        np.testing.assert_allclose(a_h[0, 0], 20.0 / m, rtol=1e-12)


# ---------------------------------------------------------------------------
# Cohérence entre schémas implicite et explicite
# ---------------------------------------------------------------------------


class TestSchemeConsistency:
    """Trapèze et diff. centrales convergent vers la même solution pour Δt → 0."""

    m  = 1.0
    k  = 100.0
    xi = 0.05

    @classmethod
    def setup_class(cls):
        cls.omega_n = np.sqrt(cls.k / cls.m)
        cls.c       = 2.0 * cls.xi * cls.omega_n * cls.m
        cls.T_n     = 2.0 * np.pi / cls.omega_n

    def test_implicit_explicit_agree_small_dt(self):
        """Pour Δt très petit (Δt_crit/50), trapèze et diff. centrales coïncident."""
        K, M, C = _sdof_matrices(self.m, self.c, self.k)
        dt_crit = NEWMARK_CENTRAL_DIFF.critical_dt(self.omega_n)
        dt = dt_crit / 50.0
        n_steps = int(2 * self.T_n / dt)

        u0, v0 = np.array([0.01]), np.array([0.0])

        u_imp, _, _ = solve_newmark(K, M, C, None, u0, v0, dt, n_steps, NEWMARK_TRAPEZOIDAL)
        u_exp, _, _ = solve_newmark(K, M, C, None, u0, v0, dt, n_steps, NEWMARK_CENTRAL_DIFF)

        # Les deux schémas (ordre 2) convergent vers la même solution ; leur différence
        # est O(Δt²) par pas. Tolérance absolue : 2e-5 m (0.2 % de l'amplitude).
        np.testing.assert_allclose(u_imp[0], u_exp[0], atol=2e-5,
                                   err_msg="Désaccord implicite/explicite pour Δt fin")

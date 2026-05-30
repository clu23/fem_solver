"""Tests unitaires — réponse aléatoire (PSD).

Tous les résultats sont comparés à des solutions analytiques exactes.

Modèle SDOF (1-DDL)
--------------------
Un seul élément Bar2D vertical (y), nœuds 0 (encastré) et 1 (libre en uy).
Paramètres : m_total=1 kg (densité → ρAL=1), k=1000 N/m.

Après assemblage et application des CL, seul uy₁ (DDL global 3) est libre.
Masse effective (consistent mass, nœud libre) : m_eff = ρAL/3 = 1/3 kg.
Rigidité effective : k_eff = EA/L = 1000 N/m.

Fréquence propre FEM : ωₙ = √(k_eff / m_eff) = √3000 ≈ 54.77 rad/s.

Solutions analytiques (bruit blanc G₀, intégrale sur (-∞,+∞)) :

  Excitation en force G_F [N²/Hz] :
    σ²_u = G_F / (8 ξ ωₙ³ m_eff²)

  Excitation à la base G₀ [(m/s²)²/Hz] :
    σ²_u_rel = G₀ / (8 ξ ωₙ³)        (déplacement relatif, indépendant de m)
    Miles :  σ_a ≈ √(π fₙ G₀ / (4ξ))  (accélération absolue, one-sided PSD)
"""

from __future__ import annotations

import numpy as np
import pytest

from femsolver.core.assembler import Assembler
from femsolver.core.boundary import apply_dirichlet
from femsolver.core.material import ElasticMaterial
from femsolver.core.mesh import BoundaryConditions, ElementData, Mesh
from femsolver.dynamics.damping import HystereticDamping, ModalDampingModel
from femsolver.dynamics.rayleigh import RayleighDamping, build_damping_matrix
from femsolver.dynamics.random_response import (
    RandomResult,
    compute_rms,
    influence_vector,
    miles_equation,
    psd_white,
    run_random_base,
    run_random_force,
    solve_random_force,
)
from femsolver.elements.bar2d import Bar2D


# ---------------------------------------------------------------------------
# Helpers — SDOF Bar2D
# ---------------------------------------------------------------------------


def _sdof_mesh_bc(
    m_total: float = 1.0,
    k: float = 1000.0,
) -> tuple[Mesh, BoundaryConditions]:
    """Oscillateur 1-DDL — barre Bar2D verticale.

    Nœud 0 : encastrement complet (ux=0, uy=0).
    Nœud 1 : libre en uy (ux=0 bloqué — pas de rigidité transversale pour Bar2D).

    Propriétés : E=210 GPa, EA/L = k, ρAL = m_total.
    Masse effective au DDL libre : m_eff = m_total/3 (consistent mass).
    Fréquence propre FEM : ωₙ = √(k / m_eff) = √(3k / m_total).
    """
    E = 210e9
    L = 1.0
    area = k * L / E
    rho = m_total / (area * L)

    nodes = np.array([[0.0, 0.0], [0.0, L]])
    mat = ElasticMaterial(E=E, nu=0.3, rho=rho)
    props = {"area": area}
    elements = (ElementData(Bar2D, (0, 1), mat, props),)
    mesh = Mesh(nodes=nodes, elements=elements, n_dim=2, dof_per_node=2)

    bc = BoundaryConditions(
        dirichlet={0: {0: 0.0, 1: 0.0}, 1: {0: 0.0}},
        neumann={},
    )
    return mesh, bc


def _sdof_params(
    m_total: float = 1.0,
    k: float = 1000.0,
    zeta: float = 0.05,
) -> dict:
    """Paramètres effectifs du SDOF FEM (k_eff, m_eff, ωₙ, fₙ, damping)."""
    mesh, bc = _sdof_mesh_bc(m_total, k)
    assembler = Assembler(mesh)
    K = assembler.assemble_stiffness()
    M = assembler.assemble_mass()
    F_dummy = np.zeros(mesh.n_dof)
    ds = apply_dirichlet(K, F_dummy, mesh, bc)

    # 1 seul DDL libre
    k_eff = float(ds.K_free.toarray()[0, 0])
    m_eff = float(ds.reduce_mass(M).toarray()[0, 0])
    omega_n = np.sqrt(k_eff / m_eff)
    fn = omega_n / (2.0 * np.pi)

    # Rayleigh β-seul : ξ = βωₙ/2  →  β = 2ξ/ωₙ
    damping = RayleighDamping(alpha=0.0, beta=2.0 * zeta / omega_n)

    # ModalDampingModel construit manuellement (eigsh ne fonctionne pas pour n_free=1)
    phi_free = np.array([[1.0 / np.sqrt(m_eff)]])   # (1, 1) M-normalisé
    phi_full = np.zeros((mesh.n_dof, 1))
    phi_full[ds.free_dofs[0], 0] = phi_free[0, 0]
    modal_damping = ModalDampingModel(
        omega_n=np.array([omega_n]),
        zeta_n=np.array([zeta]),
        phi=phi_full,
    )

    return {
        "mesh": mesh, "bc": bc,
        "k_eff": k_eff, "m_eff": m_eff,
        "omega_n": omega_n, "fn": fn,
        "zeta": zeta,
        "rayleigh": damping,
        "modal": modal_damping,
        "free_dof": int(ds.free_dofs[0]),   # DDL global du degré libre
    }


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------


class TestUtilities:
    """Tests des fonctions utilitaires."""

    def test_psd_white_shape(self) -> None:
        freqs = np.linspace(1.0, 100.0, 500)
        G = psd_white(2.5, freqs)
        assert G.shape == (500,)
        assert np.allclose(G, 2.5)

    def test_compute_rms_1d(self) -> None:
        """σ = √(G₀ · Δf) pour spectre plat."""
        freqs = np.linspace(0.0, 100.0, 1001)
        G = psd_white(4.0, freqs)
        sigma = compute_rms(G, freqs)
        assert abs(sigma - np.sqrt(4.0 * 100.0)) / np.sqrt(4.0 * 100.0) < 1e-3

    def test_compute_rms_2d(self) -> None:
        """Forme (n_dof,) pour entrée 2D."""
        freqs = np.linspace(1.0, 10.0, 200)
        G = np.ones((5, len(freqs))) * 3.0
        rms = compute_rms(G, freqs)
        assert rms.shape == (5,)
        expected = np.sqrt(3.0 * (freqs[-1] - freqs[0]))
        assert np.allclose(rms, expected, rtol=1e-2)

    def test_miles_equation_positive(self) -> None:
        assert miles_equation(fn=5.0, zeta=0.05, G0=0.01) > 0.0

    def test_miles_equation_formula(self) -> None:
        fn, zeta, G0 = 10.0, 0.02, 0.5
        expected = np.sqrt(np.pi * fn * G0 / (4.0 * zeta))
        assert abs(miles_equation(fn, zeta, G0) - expected) < 1e-12

    def test_miles_proportional_to_sqrt_fn(self) -> None:
        """Miles scale en √fₙ."""
        s1 = miles_equation(fn=4.0, zeta=0.05, G0=1.0)
        s2 = miles_equation(fn=16.0, zeta=0.05, G0=1.0)
        assert abs(s2 / s1 - 2.0) < 1e-10


# ---------------------------------------------------------------------------
# Vecteur d'influence
# ---------------------------------------------------------------------------


class TestInfluenceVector:
    def test_shape(self) -> None:
        mesh, bc = _sdof_mesh_bc()
        r = influence_vector(mesh, direction=1)
        assert r.shape == (mesh.n_dof,)

    def test_y_direction(self) -> None:
        """direction=1 (uy) : DDL uy de chaque nœud = 1, ux = 0."""
        mesh, bc = _sdof_mesh_bc()
        r = influence_vector(mesh, direction=1)
        dpn = mesh.dof_per_node
        for node in range(mesh.n_dof // dpn):
            assert r[node * dpn + 1] == 1.0
            assert r[node * dpn + 0] == 0.0

    def test_x_direction(self) -> None:
        mesh, bc = _sdof_mesh_bc()
        r = influence_vector(mesh, direction=0)
        dpn = mesh.dof_per_node
        for node in range(mesh.n_dof // dpn):
            assert r[node * dpn + 0] == 1.0

    def test_invalid_direction_raises(self) -> None:
        mesh, bc = _sdof_mesh_bc()
        with pytest.raises(ValueError):
            influence_vector(mesh, direction=5)


# ---------------------------------------------------------------------------
# Excitation en force — SDOF
# ---------------------------------------------------------------------------


class TestRandomForceSDOF:
    """Réponse RMS d'un SDOF sous force aléatoire bruit blanc.

    Formule analytique : σ²_u = G_F / (8 ξ ωₙ³ m_eff²)
    """

    def setup_method(self) -> None:
        self.G_F = 1.0    # N²/Hz
        self.p = _sdof_params(m_total=1.0, k=1000.0, zeta=0.05)
        # Plage couvrant ±20×fₙ
        f_min = max(0.1, self.p["fn"] / 20.0)
        f_max = self.p["fn"] * 20.0
        self.freqs = np.linspace(f_min, f_max, 4000)
        self.sigma_ana = np.sqrt(
            self.G_F / (8.0 * self.p["zeta"] * self.p["omega_n"] ** 3 * self.p["m_eff"] ** 2)
        )

    def _F_dir(self) -> np.ndarray:
        F = np.zeros(self.p["mesh"].n_dof)
        F[self.p["free_dof"]] = 1.0
        return F

    def test_rayleigh_damping(self) -> None:
        """Rayleigh β-seul : σ_u numérique ≈ analytique (rtol < 5%)."""
        result = run_random_force(
            self.p["mesh"], self.p["bc"], self._F_dir(),
            psd_white(self.G_F, self.freqs), self.freqs,
            self.p["rayleigh"],
        )
        rms = result.rms_u[self.p["free_dof"]]
        assert abs(rms - self.sigma_ana) / self.sigma_ana < 0.05

    def test_hysteretic_damping(self) -> None:
        """Hystérétique η=2ξ ≡ ξ à la résonance : σ_u ≈ analytique (rtol < 10%)."""
        damping = HystereticDamping(eta=2.0 * self.p["zeta"])
        result = run_random_force(
            self.p["mesh"], self.p["bc"], self._F_dir(),
            psd_white(self.G_F, self.freqs), self.freqs, damping,
        )
        rms = result.rms_u[self.p["free_dof"]]
        assert abs(rms - self.sigma_ana) / self.sigma_ana < 0.10

    def test_modal_damping(self) -> None:
        """Superposition modale manuelle : σ_u ≈ analytique (rtol < 5%)."""
        result = run_random_force(
            self.p["mesh"], self.p["bc"], self._F_dir(),
            psd_white(self.G_F, self.freqs), self.freqs,
            self.p["modal"],
        )
        rms = result.rms_u[self.p["free_dof"]]
        assert abs(rms - self.sigma_ana) / self.sigma_ana < 0.05

    def test_result_shape(self) -> None:
        """G_u (n_dof, n_freqs), rms_u (n_dof), G_a is None."""
        result = run_random_force(
            self.p["mesh"], self.p["bc"], self._F_dir(),
            psd_white(self.G_F, self.freqs), self.freqs,
            self.p["rayleigh"],
        )
        mesh = self.p["mesh"]
        assert result.G_u.shape == (mesh.n_dof, len(self.freqs))
        assert result.rms_u.shape == (mesh.n_dof,)
        assert result.G_a is None
        assert result.rms_a is None

    def test_constrained_dofs_zero(self) -> None:
        """DDL contraints → G_u = 0, rms_u = 0."""
        result = run_random_force(
            self.p["mesh"], self.p["bc"], self._F_dir(),
            psd_white(self.G_F, self.freqs), self.freqs,
            self.p["rayleigh"],
        )
        constrained = [i for i in range(self.p["mesh"].n_dof)
                       if i != self.p["free_dof"]]
        for dof in constrained:
            assert np.allclose(result.G_u[dof, :], 0.0)
            assert result.rms_u[dof] == pytest.approx(0.0)

    def test_psd_non_negative(self) -> None:
        """PSD ≥ 0 partout."""
        result = run_random_force(
            self.p["mesh"], self.p["bc"], self._F_dir(),
            psd_white(self.G_F, self.freqs), self.freqs,
            self.p["rayleigh"],
        )
        assert np.all(result.G_u >= 0.0)

    def test_solve_random_force_low_level(self) -> None:
        """Interface bas niveau → même RMS que run_random_force."""
        mesh, bc = self.p["mesh"], self.p["bc"]
        assembler = Assembler(mesh)
        K = assembler.assemble_stiffness()
        M = assembler.assemble_mass()
        F_dummy = np.zeros(mesh.n_dof)
        ds = apply_dirichlet(K, F_dummy, mesh, bc)
        free = ds.free_dofs
        C = build_damping_matrix(self.p["rayleigh"], M, K)
        C_free = C[free, :][:, free].tocsr()

        F_dir_free = self._F_dir()[free]
        G_input = psd_white(self.G_F, self.freqs)
        G_u_free, rms_free = solve_random_force(
            ds.K_free, ds.reduce_mass(M), C_free, F_dir_free, G_input, self.freqs
        )

        result = run_random_force(
            mesh, bc, self._F_dir(), G_input, self.freqs, self.p["rayleigh"]
        )
        np.testing.assert_allclose(rms_free, result.rms_u[free], rtol=1e-6)

    def test_no_damping_runs(self) -> None:
        """Sans amortissement (C=0) la simulation tourne sans exception."""
        result = run_random_force(
            self.p["mesh"], self.p["bc"], self._F_dir(),
            psd_white(self.G_F, self.freqs), self.freqs, None,
        )
        # RMS finite ou très grande (quasi-singulier à la résonance)
        rms = result.rms_u[self.p["free_dof"]]
        assert np.isfinite(rms) or rms > 1e3


# ---------------------------------------------------------------------------
# Excitation à la base — SDOF
# ---------------------------------------------------------------------------


class TestRandomBaseSDOF:
    """Réponse SDOF sous base excitation bruit blanc.

    Analytique :
        σ²_u_rel = G₀ / (8 ξ ωₙ³)
        σ_a_Miles ≈ √(π fₙ G₀ / (2ξ))   (excédent résonance)
    """

    def setup_method(self) -> None:
        self.G0 = 0.01   # (m/s²)²/Hz
        self.p = _sdof_params(m_total=1.0, k=1000.0, zeta=0.05)
        f_min = max(0.1, self.p["fn"] / 20.0)
        f_max = self.p["fn"] * 20.0
        self.freqs = np.linspace(f_min, f_max, 4000)
        self.sigma_u_rel = np.sqrt(
            self.G0 / (8.0 * self.p["zeta"] * self.p["omega_n"] ** 3)
        )

    def test_relative_displacement_rms(self) -> None:
        """σ_u_rel numérique ≈ analytique (rtol < 5%)."""
        result = run_random_base(
            self.p["mesh"], self.p["bc"], direction=1,
            G_input=psd_white(self.G0, self.freqs),
            freqs=self.freqs, damping=self.p["rayleigh"],
        )
        rms = result.rms_u[self.p["free_dof"]]
        assert abs(rms - self.sigma_u_rel) / self.sigma_u_rel < 0.05

    def test_result_has_acceleration_fields(self) -> None:
        """G_a et rms_a présents avec les bonnes formes."""
        result = run_random_base(
            self.p["mesh"], self.p["bc"], direction=1,
            G_input=psd_white(self.G0, self.freqs),
            freqs=self.freqs, damping=self.p["rayleigh"],
        )
        mesh = self.p["mesh"]
        assert result.G_a is not None
        assert result.rms_a is not None
        assert result.G_a.shape == (mesh.n_dof, len(self.freqs))
        assert result.rms_a.shape == (mesh.n_dof,)

    def test_miles_equation_direct(self) -> None:
        """σ_a numérique ≈ miles_equation directement (rtol < 3%).

        Miles approxime ∫₀^∞|T|²G₀ df ≈ G₀×πfₙ/(4ξ).
        La résonance domine à ~94% du total → erreur d'approximation < 3%.
        """
        result = run_random_base(
            self.p["mesh"], self.p["bc"], direction=1,
            G_input=psd_white(self.G0, self.freqs),
            freqs=self.freqs, damping=self.p["rayleigh"],
        )
        rms_a = result.rms_a[self.p["free_dof"]]
        sigma_miles = miles_equation(self.p["fn"], self.p["zeta"], self.G0)
        assert abs(rms_a - sigma_miles) / sigma_miles < 0.03

    def test_g_a_non_negative(self) -> None:
        """PSD d'accélération ≥ 0 partout."""
        result = run_random_base(
            self.p["mesh"], self.p["bc"], direction=1,
            G_input=psd_white(self.G0, self.freqs),
            freqs=self.freqs, damping=self.p["rayleigh"],
        )
        assert np.all(result.G_a >= 0.0)

    def test_hysteretic_damping_base(self) -> None:
        """Hystérétique η=2ξ : σ_u_rel ≈ analytique (rtol < 10%)."""
        damping = HystereticDamping(eta=2.0 * self.p["zeta"])
        result = run_random_base(
            self.p["mesh"], self.p["bc"], direction=1,
            G_input=psd_white(self.G0, self.freqs),
            freqs=self.freqs, damping=damping,
        )
        rms = result.rms_u[self.p["free_dof"]]
        assert abs(rms - self.sigma_u_rel) / self.sigma_u_rel < 0.10

    def test_modal_damping_base(self) -> None:
        """Superposition modale manuelle : σ_u_rel ≈ analytique (rtol < 5%)."""
        result = run_random_base(
            self.p["mesh"], self.p["bc"], direction=1,
            G_input=psd_white(self.G0, self.freqs),
            freqs=self.freqs, damping=self.p["modal"],
        )
        rms = result.rms_u[self.p["free_dof"]]
        assert abs(rms - self.sigma_u_rel) / self.sigma_u_rel < 0.05

    def test_low_freq_absolute_accel_approaches_input(self) -> None:
        """À très basse fréquence : transmissibilité → 1, G_a → G₀."""
        fn = self.p["fn"]
        freqs_low = np.linspace(0.01, fn / 10.0, 300)
        result = run_random_base(
            self.p["mesh"], self.p["bc"], direction=1,
            G_input=psd_white(self.G0, freqs_low),
            freqs=freqs_low, damping=self.p["rayleigh"],
        )
        mean_G_a = np.mean(result.G_a[self.p["free_dof"], :])
        assert abs(mean_G_a - self.G0) / self.G0 < 0.05

    def test_constrained_dofs_zero_base(self) -> None:
        """DDL contraints → G_u = G_a = 0."""
        result = run_random_base(
            self.p["mesh"], self.p["bc"], direction=1,
            G_input=psd_white(self.G0, self.freqs),
            freqs=self.freqs, damping=self.p["rayleigh"],
        )
        constrained = [i for i in range(self.p["mesh"].n_dof)
                       if i != self.p["free_dof"]]
        for dof in constrained:
            assert np.allclose(result.G_u[dof, :], 0.0)
            assert np.allclose(result.G_a[dof, :], 0.0)

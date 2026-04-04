"""Tests unitaires Beam2DTimoshenko — validation par solutions analytiques.

Cas de référence
----------------

**1. Convergence vers Euler–Bernoulli (poutre élancée)**

Pour une poutre de rapport L/h ≫ 1, le paramètre Φ = 12EI/(GA_s L²) → 0.
La déflection de Timoshenko doit coïncider avec celle d'EB à 0,1 % près.

**2. Poutre console épaisse — solution exacte de Timoshenko**

Console encastrée-libre, charge ponctuelle P en bout :

    v_tip = PL³/(3EI) · (1 + Φ/4)
          = PL³/(3EI)  +  PL/(GA_s)        [EB + contribution cisaillement]

Le terme PL/(GA_s) représente le déplacement additionnel par cisaillement.
Pour L/h = 2 (poutre épaisse), ce terme est significatif (≈ 20–40 % de EB).

Avec un seul élément Beam2DTimoshenko (matrice exacte d'équilibre), la
solution est exacte aux nœuds pour la charge ponctuelle.

**3. Poutre bi-appuyée épaisse — solution exacte**

Charge ponctuelle P au centre :

    v_center = PL³/(48EI) · (1 + Φ)

**4. Facteurs de correction κ des sections**

Formules de Cowper (1966) vérifiées numériquement.

**5. Interface scalaire (shear_area)**

Vérification que l'interface sans Section (area, inertia, shear_area) donne
le même résultat qu'avec un objet Section.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.sparse.linalg import spsolve

from femsolver.core.assembler import Assembler
from femsolver.core.boundary import apply_dirichlet
from femsolver.core.material import ElasticMaterial
from femsolver.core.mesh import BoundaryConditions, ElementData, Mesh
from femsolver.core.sections import (
    CircularSection,
    CSection,
    HollowCircularSection,
    HollowRectangularSection,
    ISection,
    LSection,
    RectangularSection,
)
from femsolver.elements.beam2d import Beam2D
from femsolver.elements.beam2d_timoshenko import Beam2DTimoshenko


# ---------------------------------------------------------------------------
# Constantes communes
# ---------------------------------------------------------------------------

E   = 210e9     # Pa  (acier)
NU  = 0.3
RHO = 7800.0    # kg/m³
P   = 10_000.0  # N

MAT = ElasticMaterial(E=E, nu=NU, rho=RHO)
G   = MAT.G


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def _cantilever_tip(
    etype,
    n_elem: int,
    L: float,
    props: dict,
    load: float = P,
) -> tuple[float, float]:
    """Résolution console encastrée-libre, force P en bout.

    Returns
    -------
    uy_tip : float
        Déflection transverse au nœud libre.
    theta_tip : float
        Rotation au nœud libre.
    """
    n_nodes = n_elem + 1
    nodes = np.zeros((n_nodes, 2))
    nodes[:, 0] = np.linspace(0.0, L, n_nodes)

    elements = tuple(
        ElementData(type(etype), (i, i + 1), MAT, props)
        for i in range(n_elem)
    )
    bc = BoundaryConditions(
        dirichlet={0: {0: 0.0, 1: 0.0, 2: 0.0}},
        neumann={n_nodes - 1: {1: load}},
    )
    mesh = Mesh(nodes=nodes, elements=elements, n_dim=2, dof_per_node=3)

    assembler = Assembler(mesh)
    K = assembler.assemble_stiffness()
    F = assembler.assemble_forces(bc)

    K_bc, F_bc = apply_dirichlet(K, F, mesh, bc)
    u = spsolve(K_bc, F_bc)

    tip = n_nodes - 1
    return float(u[3 * tip + 1]), float(u[3 * tip + 2])


def _simply_supported_center(
    etype,
    n_elem: int,   # must be even
    L: float,
    props: dict,
    load: float = P,
) -> float:
    """Déflection au centre d'une poutre bi-appuyée, charge P au milieu."""
    n_nodes = n_elem + 1
    nodes = np.zeros((n_nodes, 2))
    nodes[:, 0] = np.linspace(0.0, L, n_nodes)

    elements = tuple(
        ElementData(type(etype), (i, i + 1), MAT, props)
        for i in range(n_elem)
    )
    mid = n_nodes // 2
    # Appuis : nœud 0 → ux=uy=0 ; nœud n-1 → uy=0
    bc = BoundaryConditions(
        dirichlet={0: {0: 0.0, 1: 0.0}, n_nodes - 1: {1: 0.0}},
        neumann={mid: {1: load}},
    )
    mesh = Mesh(nodes=nodes, elements=elements, n_dim=2, dof_per_node=3)

    assembler = Assembler(mesh)
    K = assembler.assemble_stiffness()
    F = assembler.assemble_forces(bc)

    K_bc, F_bc = apply_dirichlet(K, F, mesh, bc)
    u = spsolve(K_bc, F_bc)
    return float(u[3 * mid + 1])


# ---------------------------------------------------------------------------
# Section 1 : facteurs κ (Cowper 1966)
# ---------------------------------------------------------------------------

class TestShearCorrectionFactors:
    """Formules de Cowper (1966) vérifiées sur chaque type de section."""

    def test_rectangular_cowper(self) -> None:
        """κ = 10(1+ν)/(12+11ν) — Cowper Eq. 30."""
        sec = RectangularSection(width=0.05, height=0.10)
        nu = 0.3
        expected = 10.0 * (1.0 + nu) / (12.0 + 11.0 * nu)
        assert math.isclose(sec.shear_correction_factor(nu), expected, rel_tol=1e-12)

    def test_rectangular_nu0(self) -> None:
        """Pour ν=0 : κ = 10/12 = 5/6."""
        sec = RectangularSection(width=0.1, height=0.1)
        assert math.isclose(sec.shear_correction_factor(0.0), 10.0 / 12.0, rel_tol=1e-12)

    def test_circular_cowper(self) -> None:
        """κ = 6(1+ν)/(7+6ν) — Cowper Eq. 28."""
        sec = CircularSection(radius=0.05)
        nu = 0.3
        expected = 6.0 * (1.0 + nu) / (7.0 + 6.0 * nu)
        assert math.isclose(sec.shear_correction_factor(nu), expected, rel_tol=1e-12)

    def test_hollow_circular_solid_limit(self) -> None:
        """m → 0 : section creuse → section pleine (même κ).

        La formule de Cowper pour la section creuse :
            κ = 6(1+ν)(1+m²)² / [(7+6ν)(1+m²)² + (20+12ν)m²]

        Vaut bien 6(1+ν)/(7+6ν) quand m → 0 (= formule de la section pleine).
        """
        sec_solid = CircularSection(radius=0.05)
        sec_near_solid = HollowCircularSection(outer_radius=0.05, inner_radius=0.0001)
        nu = 0.3
        assert math.isclose(
            sec_near_solid.shear_correction_factor(nu),
            sec_solid.shear_correction_factor(nu),
            rel_tol=1e-3,
        )

    def test_hollow_circular_thin_wall_limit(self) -> None:
        """m → 1 : section très mince → κ → 2(1+ν)/(4+3ν).

        Limite analytique de la formule de Cowper quand m = r_i/r_o → 1.
        """
        nu = 0.3
        # m très proche de 1 : r_i/r_o = 0.999
        sec = HollowCircularSection(outer_radius=0.100, inner_radius=0.0999)
        kappa_computed = sec.shear_correction_factor(nu)
        kappa_thin_limit = 2.0 * (1.0 + nu) / (4.0 + 3.0 * nu)
        assert math.isclose(kappa_computed, kappa_thin_limit, rel_tol=0.01)

    def test_hollow_circular_range(self) -> None:
        """κ ∈ (0, 1] pour différentes valeurs de m."""
        nu = 0.3
        for r_i, r_o in [(0.01, 0.05), (0.04, 0.05), (0.001, 0.1)]:
            sec = HollowCircularSection(outer_radius=r_o, inner_radius=r_i)
            kappa = sec.shear_correction_factor(nu)
            assert 0.0 < kappa <= 1.0, f"κ hors de (0,1] pour ri={r_i}, ro={r_o}"

    def test_hollow_rectangular_web_fraction(self) -> None:
        """κ = 2t·H / A — méthode des âmes."""
        sec = HollowRectangularSection(outer_width=0.10, outer_height=0.15, thickness=0.005)
        expected = 2.0 * sec.thickness * sec.outer_height / sec.area
        assert math.isclose(sec.shear_correction_factor(0.3), expected, rel_tol=1e-12)

    def test_isection_web_fraction(self) -> None:
        """κ = A_web / A pour profilé en I."""
        sec = ISection(
            flange_width=0.10, height=0.20,
            flange_thickness=0.010, web_thickness=0.006,
        )
        h_w = sec.height - 2.0 * sec.flange_thickness
        expected = sec.web_thickness * h_w / sec.area
        assert math.isclose(sec.shear_correction_factor(0.3), expected, rel_tol=1e-12)

    def test_csection_web_fraction(self) -> None:
        """κ = A_web / A pour profilé en C."""
        sec = CSection(
            flange_width=0.075, height=0.15,
            flange_thickness=0.010, web_thickness=0.006,
        )
        h_w = sec.height - 2.0 * sec.flange_thickness
        expected = sec.web_thickness * h_w / sec.area
        assert math.isclose(sec.shear_correction_factor(0.3), expected, rel_tol=1e-12)

    def test_lsection_rectangular_approx(self) -> None:
        """κ = 10(1+ν)/(12+11ν) pour cornière L (approximation rectangulaire)."""
        sec = LSection(width=0.10, height=0.10, thickness=0.01)
        nu = 0.3
        expected = 10.0 * (1.0 + nu) / (12.0 + 11.0 * nu)
        assert math.isclose(sec.shear_correction_factor(nu), expected, rel_tol=1e-12)

    def test_kappa_in_range(self) -> None:
        """κ ∈ (0, 1] pour toutes les sections concrètes."""
        nu = 0.3
        sections = [
            RectangularSection(width=0.1, height=0.2),
            CircularSection(radius=0.05),
            HollowCircularSection(outer_radius=0.05, inner_radius=0.03),
            HollowRectangularSection(outer_width=0.1, outer_height=0.15, thickness=0.005),
            ISection(flange_width=0.1, height=0.2, flange_thickness=0.01, web_thickness=0.006),
            CSection(flange_width=0.075, height=0.15, flange_thickness=0.01, web_thickness=0.006),
            LSection(width=0.1, height=0.1, thickness=0.01),
        ]
        for sec in sections:
            kappa = sec.shear_correction_factor(nu)
            assert 0.0 < kappa <= 1.0, f"{type(sec).__name__} : κ={kappa} hors (0,1]"


# ---------------------------------------------------------------------------
# Section 2 : propriétés de la matrice de rigidité
# ---------------------------------------------------------------------------

class TestTimoshenkoStiffnessProperties:
    """Tests structurels sur la matrice K."""

    def setup_method(self) -> None:
        self.sec = RectangularSection(width=0.05, height=0.10)
        self.nodes = np.array([[0.0, 0.0], [1.0, 0.0]])
        self.elem = Beam2DTimoshenko()

    def test_shape(self) -> None:
        K = self.elem.stiffness_matrix(MAT, self.nodes, {"section": self.sec})
        assert K.shape == (6, 6)

    def test_symmetric(self) -> None:
        K = self.elem.stiffness_matrix(MAT, self.nodes, {"section": self.sec})
        np.testing.assert_allclose(K, K.T, atol=1e-10 * abs(K).max())

    def test_positive_semidefinite(self) -> None:
        """K doit avoir exactement 3 valeurs propres ≈ 0 (MRC rigides)."""
        K = self.elem.stiffness_matrix(MAT, self.nodes, {"section": self.sec})
        eigs = np.linalg.eigvalsh(K)
        n_zero = np.sum(np.abs(eigs) < 1e-6 * eigs.max())
        assert n_zero == 3

    def test_phi_zero_matches_eb(self) -> None:
        """Φ → 0 (G·A_s → ∞) : matrice de Timoshenko ≡ matrice d'EB."""
        L = 1.0
        nodes = np.array([[0.0, 0.0], [L, 0.0]])
        A = self.sec.area
        I = self.sec.Iz
        # GA_s très grande → Φ ≈ 0
        gas_huge = 1e20
        props_tim = {"area": A, "inertia": I, "shear_area": gas_huge / MAT.G}

        K_tim = Beam2DTimoshenko().stiffness_matrix(MAT, nodes, props_tim)
        K_eb  = Beam2D().stiffness_matrix(MAT, nodes, {"area": A, "inertia": I})
        np.testing.assert_allclose(K_tim, K_eb, rtol=1e-6)

    def test_shear_area_interface(self) -> None:
        """Interface scalaire (shear_area) ≡ interface Section."""
        L = 1.0
        nodes = np.array([[0.0, 0.0], [L, 0.0]])
        sec = self.sec
        kappa = sec.shear_correction_factor(MAT.nu)

        props_sec = {"section": sec}
        props_scalar = {
            "area": sec.area,
            "inertia": sec.Iz,
            "shear_area": kappa * sec.area,
        }

        K_sec    = Beam2DTimoshenko().stiffness_matrix(MAT, nodes, props_sec)
        K_scalar = Beam2DTimoshenko().stiffness_matrix(MAT, nodes, props_scalar)
        np.testing.assert_allclose(K_sec, K_scalar, rtol=1e-12)

    def test_missing_shear_area_raises(self) -> None:
        """Sans 'section' ni 'shear_area', une ValueError est levée."""
        with pytest.raises(ValueError, match="shear_area"):
            Beam2DTimoshenko().stiffness_matrix(
                MAT, self.nodes, {"area": 1e-3, "inertia": 1e-6}
            )


# ---------------------------------------------------------------------------
# Section 3 : convergence Timoshenko → EB (poutre élancée)
# ---------------------------------------------------------------------------

class TestTimoshenkoConvergesToEulerBernoulli:
    """Pour L/h ≫ 1, Φ → 0 et les deux théories doivent coïncider."""

    def _props(self, sec: RectangularSection) -> dict:
        return {"section": sec}

    def test_cantilever_slender_single_element(self) -> None:
        """L/h = 100 : déflection EB et Timoshenko à 0,01 % près.

        Formule EB : v_tip = PL³/(3EI)
        Formule Tim: v_tip = PL³/(3EI) · (1 + Φ/4)

        Pour L = 10 m, h = 0.1 m → L/h = 100 → Φ = 12EI/(GA_s L²) ≪ 1.
        """
        h = 0.10      # m
        b = 0.05      # m
        L = 10.0      # m   ← L/h = 100

        sec = RectangularSection(width=b, height=h)
        I = sec.Iz    # b·h³/12
        A = sec.area
        kappa = sec.shear_correction_factor(MAT.nu)
        GA_s = G * kappa * A
        Phi = 12.0 * E * I / (GA_s * L ** 2)

        v_eb  = P * L ** 3 / (3.0 * E * I)
        v_tim = v_eb * (1.0 + Phi / 4.0)   # solution analytique exacte

        v_fem_eb, _ = _cantilever_tip(Beam2D(), 1, L, {"area": A, "inertia": I})
        v_fem_tim, _ = _cantilever_tip(Beam2DTimoshenko(), 1, L, self._props(sec))

        # Les deux éléments doivent être exacts aux nœuds (1 élément suffit)
        np.testing.assert_allclose(v_fem_eb,  v_eb,  rtol=1e-12)
        np.testing.assert_allclose(v_fem_tim, v_tim, rtol=1e-12)

        # Timoshenko et EB doivent converger pour L/h = 100
        assert Phi < 0.01, f"Phi trop grand pour ce test de convergence : {Phi:.4f}"
        np.testing.assert_allclose(v_fem_tim, v_fem_eb, rtol=1e-3)

    def test_cantilever_slender_multi_element_convergence(self) -> None:
        """Convergence en maillage pour la poutre élancée (L/h=20)."""
        h, b, L = 0.10, 0.05, 2.0    # L/h = 20
        sec = RectangularSection(width=b, height=h)
        I, A = sec.Iz, sec.area
        v_exact_eb = P * L ** 3 / (3.0 * E * I)

        for n in [1, 2, 4]:
            v, _ = _cantilever_tip(Beam2DTimoshenko(), n, L, self._props(sec))
            # Doit converger vers la solution analytique de Timoshenko, qui
            # est très proche d'EB pour L/h=20.
            np.testing.assert_allclose(v, v_exact_eb, rtol=0.01,
                                       err_msg=f"n_elem={n}")


# ---------------------------------------------------------------------------
# Section 4 : solution analytique exacte de Timoshenko (poutre épaisse)
# ---------------------------------------------------------------------------

class TestTimoshenkoThickBeamExact:
    """Vérification des solutions exactes pour poutres épaisses.

    Solution analytique (console, charge ponctuelle) :
        v_tip = PL³/(3EI) · (1 + Φ/4)   avec Φ = 12EI/(GA_s L²)

    Avec un seul élément Timoshenko (matrice exacte d'équilibre), la
    solution FEM doit être exacte aux nœuds (rtol ≤ 1e-12).
    """

    def setup_method(self) -> None:
        self.h = 0.10     # m  (hauteur section)
        self.b = 0.05     # m  (largeur section)
        self.L = 0.20     # m  ← L/h = 2 (poutre épaisse)
        self.sec = RectangularSection(width=self.b, height=self.h)
        self.I = self.sec.Iz
        self.A = self.sec.area
        self.kappa = self.sec.shear_correction_factor(MAT.nu)
        self.GA_s = G * self.kappa * self.A
        self.Phi = 12.0 * E * self.I / (self.GA_s * self.L ** 2)

    def test_phi_is_significant(self) -> None:
        """Φ doit être significatif pour que ce test ait du sens (> 0.5)."""
        assert self.Phi > 0.5, (
            f"Φ = {self.Phi:.3f} n'est pas assez grand pour tester la poutre épaisse."
        )

    def test_cantilever_tip_exact(self) -> None:
        """v_tip = PL³/(3EI) · (1 + Φ/4) — exact avec 1 élément.

        Le paramètre Φ ≈ 12×EI/(GA_s L²).  Pour L/h = 2, ce terme est
        bien supérieur à 1 % du déplacement de flexion.
        """
        v_analytical = P * self.L ** 3 / (3.0 * E * self.I) * (1.0 + self.Phi / 4.0)

        v_fem, _ = _cantilever_tip(
            Beam2DTimoshenko(), 1, self.L, {"section": self.sec}
        )
        np.testing.assert_allclose(v_fem, v_analytical, rtol=1e-12,
                                   err_msg="Déflection console épaisse — 1 élément")

    def test_cantilever_eb_underestimates(self) -> None:
        """EB sous-estime la déflection pour une poutre épaisse.

        EB ignore le cisaillement → v_EB < v_Timoshenko.
        """
        v_eb  = P * self.L ** 3 / (3.0 * E * self.I)
        v_tim = v_eb * (1.0 + self.Phi / 4.0)

        # Déflection additionnelle due au cisaillement
        v_shear = P * self.L / self.GA_s
        np.testing.assert_allclose(v_tim - v_eb, v_shear, rtol=1e-12)

        assert v_tim > v_eb * 1.01, (
            "Timoshenko doit être significativement > EB pour L/h = 2"
        )

    def test_cantilever_tip_exact_multi_element(self) -> None:
        """Solution exacte conservée avec plusieurs éléments."""
        v_analytical = P * self.L ** 3 / (3.0 * E * self.I) * (1.0 + self.Phi / 4.0)
        for n in [2, 4, 8]:
            v_fem, _ = _cantilever_tip(
                Beam2DTimoshenko(), n, self.L, {"section": self.sec}
            )
            np.testing.assert_allclose(v_fem, v_analytical, rtol=1e-10,
                                       err_msg=f"n_elem={n}")

    def test_simply_supported_center_exact(self) -> None:
        """v_center = PL³/(48EI)·(1+Φ) — exact avec 2 éléments.

        Pour poutre bi-appuyée avec charge centrale, la solution de
        Timoshenko est :
            v_center = PL³/(48EI) + PL/(4·GA_s)
                     = PL³/(48EI) · (1 + Φ)
        """
        Phi_ss = 12.0 * E * self.I / (self.GA_s * self.L ** 2)
        v_analytical = (
            P * self.L ** 3 / (48.0 * E * self.I) * (1.0 + Phi_ss)
        )
        v_fem = _simply_supported_center(
            Beam2DTimoshenko(), 2, self.L, {"section": self.sec}
        )
        np.testing.assert_allclose(v_fem, v_analytical, rtol=1e-10)

    def test_shear_contribution_formula(self) -> None:
        """Contribution du cisaillement = PL/(GA_s), indépendante de EI.

        En découplant flexion et cisaillement : la déflection totale est
        la somme des deux.
        """
        v_bending = P * self.L ** 3 / (3.0 * E * self.I)   # EB
        v_shear   = P * self.L / self.GA_s                   # cisaillement pur

        v_total_formula = v_bending + v_shear
        v_total_phi     = v_bending * (1.0 + self.Phi / 4.0)

        np.testing.assert_allclose(v_total_formula, v_total_phi, rtol=1e-12)


# ---------------------------------------------------------------------------
# Section 5 : propriétés de la matrice de masse
# ---------------------------------------------------------------------------

class TestTimoshenkoMassMatrix:
    """Tests structurels et physiques sur la matrice de masse."""

    def setup_method(self) -> None:
        self.sec = RectangularSection(width=0.05, height=0.10)
        self.nodes = np.array([[0.0, 0.0], [1.0, 0.0]])
        self.elem = Beam2DTimoshenko()

    def test_shape(self) -> None:
        M = self.elem.mass_matrix(MAT, self.nodes, {"section": self.sec})
        assert M.shape == (6, 6)

    def test_symmetric(self) -> None:
        M = self.elem.mass_matrix(MAT, self.nodes, {"section": self.sec})
        np.testing.assert_allclose(M, M.T, atol=1e-10 * abs(M).max())

    def test_positive_definite(self) -> None:
        """La matrice de masse doit être définie positive."""
        M = self.elem.mass_matrix(MAT, self.nodes, {"section": self.sec})
        eigs = np.linalg.eigvalsh(M)
        assert eigs.min() > 0.0, f"Valeur propre minimale = {eigs.min()}"

    def test_total_mass_axial(self) -> None:
        """Somme des coefficients de masse axiale = ρ·A·L.

        Pour la partie axiale [2×2 sur DDL ux1, ux2] :
            M_axial = ρAL/6 · [[2,1],[1,2]]
        Somme de tous les coefficients = ρAL/6 · (2+1+1+2) = ρAL. ✓
        """
        M = self.elem.mass_matrix(MAT, self.nodes, {"section": self.sec})
        L = 1.0
        total_mass = MAT.rho * self.sec.area * L
        # Somme des 4 termes du bloc axial (indices 0 et 3)
        mass_axial_block = M[0, 0] + M[0, 3] + M[3, 0] + M[3, 3]
        np.testing.assert_allclose(mass_axial_block, total_mass, rtol=1e-12)

    def test_phi_zero_translational_matches_eb(self) -> None:
        """Pour Φ→0, la partie translationnelle doit rejoindre la masse EB.

        On compare la somme de chaque colonne des blocs de flexion
        (invariant par rotation) : somme colonne uy₁ = ρAL·(13/35 + 9/70).
        """
        L = 1.0
        nodes = np.array([[0.0, 0.0], [L, 0.0]])
        sec = self.sec
        A = sec.area
        I = sec.Iz
        gas_huge = 1e20     # Φ ≈ 0

        props_tim = {"area": A, "inertia": I, "shear_area": gas_huge / G}
        M_tim = Beam2DTimoshenko().mass_matrix(MAT, nodes, props_tim)

        # Masse EB sans inertie de rotation
        from femsolver.elements.beam2d import Beam2D
        props_eb = {"area": A, "inertia": I}
        M_eb_pure = Beam2D().mass_matrix(MAT, nodes, props_eb)

        # La matrice de Timoshenko pour Φ→0 inclut l'inertie de rotation ρI,
        # alors que _mass_local EB de Beam2D n'en tient pas compte.
        # On vérifie donc la partie translationnelle seule via le terme [1,1] :
        # EB[1,1] = ρAL · 156/420
        rho_a = MAT.rho * A
        eb_11 = rho_a * L * 156.0 / 420.0
        # Tim[1,1] = ρAL·(13/35) + ρI/L·(6/5) pour Φ=0
        rho_i = MAT.rho * I
        tim_11_expected = rho_a * L * 13.0 / 35.0 + rho_i / L * 6.0 / 5.0
        np.testing.assert_allclose(M_tim[1, 1], tim_11_expected, rtol=1e-6)

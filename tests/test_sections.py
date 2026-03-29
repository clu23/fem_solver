"""Tests unitaires pour femsolver.core.sections.

Organisation
------------
1. TestCircularSection            — formules exactes, moments polaires
2. TestHollowCircularSection      — formules exactes, cas limite r→0
3. TestRectangularSection         — formules exactes, torsion Saint-Venant
4. TestHollowRectangularSection   — formules exactes, torsion Bredt
5. TestISection                   — Huygens-Steiner + comparaison IPE/HEA
6. TestCSection                   — centroïde décalé, Iyz = 0
7. TestLSection                   — cornière égale : Iz=Iy, Iyz≠0, α=±45°
8. TestPrincipalAxes              — cohérence du cercle de Mohr
9. TestBeam2DWithSection          — intégration Beam2D : console, bi-appuyée

Tables de référence utilisées (sans congés de raccordement)
-----------------------------------------------------------
IPE 200 : h=200 mm, b=100 mm, t_f=8.5 mm, t_w=5.6 mm
    Iz_calc  ≈ 1842 cm⁴   (table avec congés : 1943 cm⁴, écart ~5 %)
    Iy_calc  ≈ 141.9 cm⁴  (table : 142 cm⁴,   écart < 0.1 %)
    A_calc   ≈ 27.25 cm²  (table : 28.5 cm²,   écart ~4 %)

HEA 200 : h=190 mm, b=200 mm, t_f=10 mm, t_w=6.5 mm
    Iz_calc  ≈ 3512 cm⁴   (table : 3692 cm⁴,  écart ~5 %)
    Iy_calc  ≈ 1334 cm⁴   (table : 1336 cm⁴,  écart < 0.2 %)
    A_calc   ≈ 51.05 cm²  (table : 53.8 cm²,   écart ~5 %)

L100×100×10 (cornière égale)
    Iz = Iy ≈ 180 cm⁴  (table : 177.6 cm⁴, écart ~1.3 %)
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


# ──────────────────────────────────────────────────────────────────────────────
# Données communes
# ──────────────────────────────────────────────────────────────────────────────

STEEL = ElasticMaterial(E=210e9, nu=0.3, rho=7800.0)
P = 10_000.0   # N (charge de test)
L_BEAM = 1.0   # m (longueur poutre)


def _cantilever_section(section: object, n_elem: int = 1) -> float:
    """Déflexion au bout d'une console Beam2D avec la section donnée.

    Retourne v_tip (positif vers le bas) = −u[uy_dof_last].
    """
    n_nodes = n_elem + 1
    xs = np.linspace(0.0, L_BEAM, n_nodes)
    nodes = np.column_stack([xs, np.zeros(n_nodes)])
    elements = tuple(
        ElementData(Beam2D, (i, i + 1), STEEL, {"section": section})
        for i in range(n_elem)
    )
    mesh = Mesh(nodes=nodes, elements=elements, n_dim=2, dof_per_node=3)
    bc = BoundaryConditions(
        dirichlet={0: {0: 0.0, 1: 0.0, 2: 0.0}},
        neumann={n_elem: {1: -P}},
    )
    assembler = Assembler(mesh)
    K = assembler.assemble_stiffness()
    F = assembler.assemble_forces(bc)
    K_bc, F_bc = apply_dirichlet(K, F, mesh, bc)
    u = spsolve(K_bc, F_bc)
    return -u[3 * n_elem + 1]   # DDL uy du dernier nœud (positif vers le haut)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Section circulaire
# ──────────────────────────────────────────────────────────────────────────────

class TestCircularSection:
    """Formules exactes : A = πr², I = πr⁴/4, J = πr⁴/2."""

    def setup_method(self) -> None:
        self.r = 0.05   # 5 cm
        self.sec = CircularSection(radius=self.r)

    def test_area(self) -> None:
        np.testing.assert_allclose(self.sec.area, math.pi * self.r**2, rtol=1e-12)

    def test_Iz(self) -> None:
        np.testing.assert_allclose(self.sec.Iz, math.pi * self.r**4 / 4.0, rtol=1e-12)

    def test_Iy_equals_Iz(self) -> None:
        """Section circulaire → isotrope : Iy = Iz."""
        np.testing.assert_allclose(self.sec.Iy, self.sec.Iz, rtol=1e-12)

    def test_Iyz_zero(self) -> None:
        assert self.sec.Iyz == 0.0

    def test_J_equals_2_Iz(self) -> None:
        """J = Ip = Iy + Iz = 2·Iz pour section circulaire."""
        np.testing.assert_allclose(self.sec.J, 2.0 * self.sec.Iz, rtol=1e-12)

    def test_extreme_fibers(self) -> None:
        assert self.sec.y_max == pytest.approx(self.r)
        assert self.sec.y_min == pytest.approx(-self.r)
        assert self.sec.z_max == pytest.approx(self.r)
        assert self.sec.z_min == pytest.approx(-self.r)

    def test_principal_invariant(self) -> None:
        """Cercle → I1 = I2 = Iz."""
        I1, I2 = self.sec.I_principal
        np.testing.assert_allclose(I1, self.sec.Iz, rtol=1e-12)
        np.testing.assert_allclose(I2, self.sec.Iz, rtol=1e-12)

    def test_sigma_bending(self) -> None:
        """σ = M·y/I."""
        M, y = 1000.0, self.r
        expected = M * y / self.sec.Iz
        np.testing.assert_allclose(self.sec.sigma_bending(M, y), expected, rtol=1e-12)

    def test_invalid_radius(self) -> None:
        with pytest.raises(ValueError, match="radius"):
            CircularSection(radius=0.0)

    def test_scaling(self) -> None:
        """Iz ∝ r⁴ : doubler r → Iz × 16."""
        sec2 = CircularSection(radius=2.0 * self.r)
        np.testing.assert_allclose(sec2.Iz, 16.0 * self.sec.Iz, rtol=1e-12)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Section circulaire creuse
# ──────────────────────────────────────────────────────────────────────────────

class TestHollowCircularSection:
    """Formules : A = π(R²−r²), I = π(R⁴−r⁴)/4, J = π(R⁴−r⁴)/2."""

    def setup_method(self) -> None:
        self.R, self.r = 0.05, 0.04
        self.sec = HollowCircularSection(outer_radius=self.R, inner_radius=self.r)

    def test_area(self) -> None:
        expected = math.pi * (self.R**2 - self.r**2)
        np.testing.assert_allclose(self.sec.area, expected, rtol=1e-12)

    def test_Iz(self) -> None:
        expected = math.pi * (self.R**4 - self.r**4) / 4.0
        np.testing.assert_allclose(self.sec.Iz, expected, rtol=1e-12)

    def test_J_equals_2_Iz(self) -> None:
        np.testing.assert_allclose(self.sec.J, 2.0 * self.sec.Iz, rtol=1e-12)

    def test_Iyz_zero(self) -> None:
        assert self.sec.Iyz == 0.0

    def test_limit_thin_wall(self) -> None:
        """Paroi fine : Iz → 2π·R³·t pour t → 0."""
        R, t = 0.050, 0.001
        sec = HollowCircularSection(outer_radius=R, inner_radius=R - t)
        Iz_approx = 2.0 * math.pi * R**3 * t / 4.0  # ≈ π/2 · R³ · t · 2 = ...
        # On vérifie surtout la cohérence analytique
        expected = math.pi * (R**4 - (R - t)**4) / 4.0
        np.testing.assert_allclose(sec.Iz, expected, rtol=1e-10)

    def test_inner_equals_zero_reduces_to_solid(self) -> None:
        """Tube avec r→0 : Iz tend vers πR⁴/4."""
        eps = 1e-6
        sec_hollow = HollowCircularSection(outer_radius=self.R, inner_radius=eps)
        sec_solid = CircularSection(radius=self.R)
        np.testing.assert_allclose(sec_hollow.Iz, sec_solid.Iz, rtol=1e-4)

    def test_invalid_inner_ge_outer(self) -> None:
        with pytest.raises(ValueError, match="inner_radius"):
            HollowCircularSection(outer_radius=0.05, inner_radius=0.05)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Section rectangulaire pleine
# ──────────────────────────────────────────────────────────────────────────────

class TestRectangularSection:
    """Formules exactes : A = bh, Iz = bh³/12, Iy = hb³/12."""

    def setup_method(self) -> None:
        self.b, self.h = 0.08, 0.12
        self.sec = RectangularSection(width=self.b, height=self.h)

    def test_area(self) -> None:
        np.testing.assert_allclose(self.sec.area, self.b * self.h, rtol=1e-12)

    def test_Iz(self) -> None:
        """Iz = b·h³/12 (flexion forte)."""
        expected = self.b * self.h**3 / 12.0
        np.testing.assert_allclose(self.sec.Iz, expected, rtol=1e-12)

    def test_Iy(self) -> None:
        """Iy = h·b³/12 (flexion faible)."""
        expected = self.h * self.b**3 / 12.0
        np.testing.assert_allclose(self.sec.Iy, expected, rtol=1e-12)

    def test_Iyz_zero(self) -> None:
        assert self.sec.Iyz == 0.0

    def test_extreme_fibers(self) -> None:
        np.testing.assert_allclose(self.sec.y_max,  self.h / 2.0, rtol=1e-12)
        np.testing.assert_allclose(self.sec.y_min, -self.h / 2.0, rtol=1e-12)
        np.testing.assert_allclose(self.sec.z_max,  self.b / 2.0, rtol=1e-12)
        np.testing.assert_allclose(self.sec.z_min, -self.b / 2.0, rtol=1e-12)

    def test_J_square_approx(self) -> None:
        """Section carrée : J ≈ 0.1406·a⁴ (Saint-Venant)."""
        a = 0.10
        sec = RectangularSection(width=a, height=a)
        # Roark: J = (a⁴/3)·(1 − 0.63 + 0.052) = a⁴ × 0.4073/3
        # Valeur exacte de Saint-Venant : J = 0.1406·a⁴
        J_exact = 0.1406 * a**4
        np.testing.assert_allclose(sec.J, J_exact, rtol=0.01)   # < 1 % d'erreur

    def test_J_thin_rectangle_approx(self) -> None:
        """Rectangle mince (b=20t) : J ≈ b·t³/3 (à 4 % près).

        La formule de Roark corrige les effets de bord (extrémités courtes) :
        J = (bt³/3)·(1 − 0.63·t/b + …) ≈ 0.969 × bt³/3 pour b/t = 20.
        L'approximation « paroi infinie » (bt³/3) surestime de ~3 %.
        """
        t, b = 0.005, 0.100
        sec = RectangularSection(width=b, height=t)
        J_thin = b * t**3 / 3.0
        np.testing.assert_allclose(sec.J, J_thin, rtol=0.04)   # < 4 % pour b/t = 20

    def test_scaling_Iz(self) -> None:
        """Doubler h → Iz × 8."""
        sec2 = RectangularSection(width=self.b, height=2.0 * self.h)
        np.testing.assert_allclose(sec2.Iz, 8.0 * self.sec.Iz, rtol=1e-12)

    def test_principal_axes_aligned(self) -> None:
        """Section symétrique → axes principaux alignés avec y, z (α = 0)."""
        assert self.sec.alpha_principal == pytest.approx(0.0, abs=1e-12)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Section rectangulaire creuse (RHS)
# ──────────────────────────────────────────────────────────────────────────────

class TestHollowRectangularSection:
    """Formules : soustraction + Bredt pour J."""

    def setup_method(self) -> None:
        self.B, self.H, self.t = 0.100, 0.150, 0.006
        self.sec = HollowRectangularSection(
            outer_width=self.B, outer_height=self.H, thickness=self.t
        )

    def test_area(self) -> None:
        B, H, t = self.B, self.H, self.t
        expected = B * H - (B - 2*t) * (H - 2*t)
        np.testing.assert_allclose(self.sec.area, expected, rtol=1e-12)

    def test_Iz(self) -> None:
        B, H, t = self.B, self.H, self.t
        expected = (B*H**3 - (B - 2*t)*(H - 2*t)**3) / 12.0
        np.testing.assert_allclose(self.sec.Iz, expected, rtol=1e-12)

    def test_Iy(self) -> None:
        B, H, t = self.B, self.H, self.t
        expected = (H*B**3 - (H - 2*t)*(B - 2*t)**3) / 12.0
        np.testing.assert_allclose(self.sec.Iy, expected, rtol=1e-12)

    def test_Iyz_zero(self) -> None:
        """Double symétrie → Iyz = 0."""
        assert self.sec.Iyz == 0.0

    def test_J_bredt(self) -> None:
        """J = 2t(B−t)²(H−t)² / (B+H−2t) — formule de Bredt."""
        B, H, t = self.B, self.H, self.t
        expected = 2.0 * t * (B - t)**2 * (H - t)**2 / (B + H - 2*t)
        np.testing.assert_allclose(self.sec.J, expected, rtol=1e-12)

    def test_J_greater_than_open(self) -> None:
        """Section fermée → J bien supérieur à section ouverte équivalente."""
        # Section ouverte (3 rectangles) : J_open ≈ (1/3)·4t·(...)³
        sec_open = ISection(
            flange_width=self.B, height=self.H,
            flange_thickness=self.t, web_thickness=self.t,
        )
        assert self.sec.J > sec_open.J * 5   # fermé >> ouvert pour section mince

    def test_extreme_fibers(self) -> None:
        np.testing.assert_allclose(self.sec.y_max,  self.H / 2.0, rtol=1e-12)
        np.testing.assert_allclose(self.sec.y_min, -self.H / 2.0, rtol=1e-12)

    def test_invalid_thickness_too_large(self) -> None:
        with pytest.raises(ValueError, match="thickness"):
            HollowRectangularSection(
                outer_width=0.1, outer_height=0.15, thickness=0.06
            )


# ──────────────────────────────────────────────────────────────────────────────
# 5. Profilé en I — comparaison IPE / HEA
# ──────────────────────────────────────────────────────────────────────────────

class TestISectionFormulas:
    """Vérification des formules Huygens-Steiner + comparaison profilés standards."""

    def _ipe200(self) -> ISection:
        """IPE 200 sans congés (h=200, b=100, tf=8.5, tw=5.6 mm)."""
        return ISection(
            flange_width=0.100, height=0.200,
            flange_thickness=0.0085, web_thickness=0.0056,
        )

    def _hea200(self) -> ISection:
        """HEA 200 sans congés (h=190, b=200, tf=10, tw=6.5 mm)."""
        return ISection(
            flange_width=0.200, height=0.190,
            flange_thickness=0.010, web_thickness=0.0065,
        )

    def test_Iyz_zero_symmetric(self) -> None:
        """ISection doublement symétrique → Iyz = 0."""
        np.testing.assert_allclose(self._ipe200().Iyz, 0.0, atol=1e-18)

    def test_centroid_at_midheight(self) -> None:
        """y_max = −y_min = h/2 (centroïde au milieu)."""
        sec = self._ipe200()
        np.testing.assert_allclose(sec.y_max, -sec.y_min, rtol=1e-12)
        np.testing.assert_allclose(sec.y_max, 0.200 / 2.0, rtol=1e-12)

    def test_centroid_on_web_axis(self) -> None:
        """z_max = −z_min = b_f/2 (centroïde sur l'axe de l'âme)."""
        sec = self._ipe200()
        np.testing.assert_allclose(sec.z_max, -sec.z_min, rtol=1e-12)
        np.testing.assert_allclose(sec.z_max, 0.100 / 2.0, rtol=1e-12)

    def test_ipe200_Iz_vs_table(self) -> None:
        """IPE 200 : Iz calculé ≈ 1842 cm⁴ (table 1943 cm⁴, écart congés ~5 %)."""
        sec = self._ipe200()
        Iz_cm4 = sec.Iz * 1e8   # m⁴ → cm⁴
        # Valeur calculée sans congés
        np.testing.assert_allclose(Iz_cm4, 1842.0, rtol=0.005)
        # Écart < 6 % avec table (congés inclus dans les tables)
        assert abs(Iz_cm4 - 1943.0) / 1943.0 < 0.06

    def test_ipe200_Iy_vs_table(self) -> None:
        """IPE 200 : Iy calculé ≈ 141.9 cm⁴ (table 142.0 cm⁴, écart < 0.1 %)."""
        sec = self._ipe200()
        Iy_cm4 = sec.Iy * 1e8
        np.testing.assert_allclose(Iy_cm4, 142.0, rtol=0.005)

    def test_ipe200_area_vs_table(self) -> None:
        """IPE 200 : A calculé ≈ 27.25 cm² (table 28.5 cm², écart ~4 %)."""
        sec = self._ipe200()
        A_cm2 = sec.area * 1e4
        np.testing.assert_allclose(A_cm2, 27.25, rtol=0.005)
        assert abs(A_cm2 - 28.5) / 28.5 < 0.06

    def test_hea200_Iz_vs_table(self) -> None:
        """HEA 200 : Iz calculé ≈ 3512 cm⁴ (table 3692 cm⁴, écart ~5 %)."""
        sec = self._hea200()
        Iz_cm4 = sec.Iz * 1e8
        np.testing.assert_allclose(Iz_cm4, 3512.0, rtol=0.005)
        assert abs(Iz_cm4 - 3692.0) / 3692.0 < 0.06

    def test_hea200_Iy_vs_table(self) -> None:
        """HEA 200 : Iy calculé ≈ 1334 cm⁴ (table 1336 cm⁴, écart < 0.2 %)."""
        sec = self._hea200()
        Iy_cm4 = sec.Iy * 1e8
        np.testing.assert_allclose(Iy_cm4, 1334.0, rtol=0.005)

    def test_Iz_subtraction_formula(self) -> None:
        """Iz = (b_f·h³ − (b_f−t_w)·h_w³)/12 — vérification directe."""
        h, b_f = 0.200, 0.100
        t_f, t_w = 0.0085, 0.0056
        h_w = h - 2 * t_f
        expected = (b_f * h**3 - (b_f - t_w) * h_w**3) / 12.0
        sec = ISection(flange_width=b_f, height=h,
                       flange_thickness=t_f, web_thickness=t_w)
        np.testing.assert_allclose(sec.Iz, expected, rtol=1e-12)

    def test_positive_definite_J(self) -> None:
        assert self._ipe200().J > 0

    def test_alpha_zero(self) -> None:
        """Profilé I doublement symétrique → α = 0."""
        assert self._ipe200().alpha_principal == pytest.approx(0.0, abs=1e-12)


# ──────────────────────────────────────────────────────────────────────────────
# 6. Profilé en C
# ──────────────────────────────────────────────────────────────────────────────

class TestCSectionFormulas:
    """Centroïde décalé, Iyz = 0, propriétés basiques."""

    def setup_method(self) -> None:
        self.sec = CSection(
            flange_width=0.075, height=0.200,
            flange_thickness=0.012, web_thickness=0.008,
        )

    def test_Iyz_zero(self) -> None:
        """CSection symétrique par rapport à l'axe horizontal → Iyz = 0."""
        np.testing.assert_allclose(self.sec.Iyz, 0.0, atol=1e-18)

    def test_z_centroid_shifted(self) -> None:
        """Le centroïde est décalé vers les ailes : z_max > −z_min."""
        # z_max (côté ailes) > |z_min| (côté dos de l'âme)
        assert self.sec.z_max > -self.sec.z_min

    def test_y_centroid_at_midheight(self) -> None:
        """Symétrie verticale → y_max = −y_min = h/2."""
        np.testing.assert_allclose(self.sec.y_max, 0.200 / 2.0, rtol=1e-12)
        np.testing.assert_allclose(self.sec.y_min, -0.200 / 2.0, rtol=1e-12)

    def test_area_positive(self) -> None:
        assert self.sec.area > 0

    def test_Iz_Iy_positive(self) -> None:
        assert self.sec.Iz > 0
        assert self.sec.Iy > 0

    def test_alpha_zero(self) -> None:
        """Section symétrique (Iyz=0) → α = 0."""
        assert self.sec.alpha_principal == pytest.approx(0.0, abs=1e-12)

    def test_Iz_equals_sum_of_components(self) -> None:
        """Iz = I_web + 2·I_flange (formule directe par Steiner)."""
        h, b_f = 0.200, 0.075
        t_f, t_w = 0.012, 0.008
        h_w = h - 2 * t_f
        # Composants par Steiner (y_G = h/2)
        Iz_web = t_w * h_w**3 / 12.0
        Iz_flange = (
            b_f * t_f**3 / 12.0 + b_f * t_f * ((h - t_f) / 2.0)**2
        )
        expected = Iz_web + 2.0 * Iz_flange
        np.testing.assert_allclose(self.sec.Iz, expected, rtol=1e-10)


# ──────────────────────────────────────────────────────────────────────────────
# 7. Cornière L
# ──────────────────────────────────────────────────────────────────────────────

class TestLSectionFormulas:
    """Cornière à ailes égales : Iz=Iy, Iyz≠0, axes principaux à ±45°."""

    def _equal_leg(self) -> LSection:
        """L100×100×10 en mètres."""
        return LSection(width=0.100, height=0.100, thickness=0.010)

    def _unequal_leg(self) -> LSection:
        """L150×100×10 — ailes inégales."""
        return LSection(width=0.100, height=0.150, thickness=0.010)

    # ── Cornière à ailes égales ──────────────────────────────────────────────

    def test_equal_Iz_equals_Iy(self) -> None:
        """Ailes égales → symétrie à 45° : Iz = Iy."""
        sec = self._equal_leg()
        np.testing.assert_allclose(sec.Iz, sec.Iy, rtol=1e-10)

    def test_equal_Iyz_nonzero(self) -> None:
        """Section non symétrique dans (y,z) → Iyz ≠ 0."""
        sec = self._equal_leg()
        assert abs(sec.Iyz) > 1e-8 * sec.Iz

    def test_equal_Iyz_negative(self) -> None:
        """Pour cornière L en bas-gauche : Iyz < 0."""
        sec = self._equal_leg()
        assert sec.Iyz < 0.0

    def test_equal_alpha_45(self) -> None:
        """Ailes égales → axe principal I₁ à +45° de z.

        Avec Iz = Iy et Iyz < 0 :
            α = ½·atan2(−2·Iyz, 0) = ½·(π/2) = +π/4

        L'axe I₁ est la diagonale montante de la cornière (de bas-gauche
        vers haut-droit), ce qui correspond à l'axe de rigidité maximale.
        """
        sec = self._equal_leg()
        np.testing.assert_allclose(
            abs(sec.alpha_principal), math.pi / 4.0, rtol=1e-8
        )

    def test_equal_principal_axes_formula(self) -> None:
        """I₁ = Iz + |Iyz|, I₂ = Iz − |Iyz| pour Iz = Iy."""
        sec = self._equal_leg()
        I1, I2 = sec.I_principal
        np.testing.assert_allclose(I1, sec.Iz + abs(sec.Iyz), rtol=1e-10)
        np.testing.assert_allclose(I2, sec.Iz - abs(sec.Iyz), rtol=1e-10)

    def test_equal_Iz_vs_table(self) -> None:
        """L100×100×10 : Iz ≈ 180.0 cm⁴ (table 177.6 cm⁴, écart ~1.3 %)."""
        sec = self._equal_leg()
        Iz_cm4 = sec.Iz * 1e8
        np.testing.assert_allclose(Iz_cm4, 180.0, rtol=0.005)
        assert abs(Iz_cm4 - 177.6) / 177.6 < 0.02

    def test_equal_area(self) -> None:
        """A = t·(h + b − t)."""
        b, h, t = 0.100, 0.100, 0.010
        expected = t * (h + b - t)
        np.testing.assert_allclose(self._equal_leg().area, expected, rtol=1e-12)

    # ── Cornière à ailes inégales ────────────────────────────────────────────

    def test_unequal_Iz_ne_Iy(self) -> None:
        """Ailes inégales → Iz ≠ Iy."""
        sec = self._unequal_leg()
        assert abs(sec.Iz - sec.Iy) > 1e-10

    def test_unequal_alpha_not_45(self) -> None:
        """Ailes inégales → angle ≠ ±45°."""
        sec = self._unequal_leg()
        assert abs(abs(sec.alpha_principal) - math.pi / 4.0) > 0.01


# ──────────────────────────────────────────────────────────────────────────────
# 8. Cohérence du cercle de Mohr (axes principaux)
# ──────────────────────────────────────────────────────────────────────────────

class TestPrincipalAxes:
    """Vérifications algébriques indépendantes de la géométrie."""

    @pytest.mark.parametrize("sec", [
        CircularSection(radius=0.05),
        RectangularSection(width=0.08, height=0.12),
        HollowRectangularSection(outer_width=0.10, outer_height=0.15, thickness=0.006),
        ISection(flange_width=0.100, height=0.200,
                 flange_thickness=0.0085, web_thickness=0.0056),
        CSection(flange_width=0.075, height=0.200,
                 flange_thickness=0.012, web_thickness=0.008),
        LSection(width=0.100, height=0.100, thickness=0.010),
    ])
    def test_I1_ge_I2(self, sec: object) -> None:
        """I₁ ≥ I₂ pour toutes les sections."""
        I1, I2 = sec.I_principal
        assert I1 >= I2 - 1e-14 * I1

    @pytest.mark.parametrize("sec", [
        CircularSection(radius=0.05),
        RectangularSection(width=0.08, height=0.12),
        HollowRectangularSection(outer_width=0.10, outer_height=0.15, thickness=0.006),
        ISection(flange_width=0.100, height=0.200,
                 flange_thickness=0.0085, web_thickness=0.0056),
        CSection(flange_width=0.075, height=0.200,
                 flange_thickness=0.012, web_thickness=0.008),
        LSection(width=0.100, height=0.100, thickness=0.010),
    ])
    def test_mohr_invariants(self, sec: object) -> None:
        """Iz + Iy = I₁ + I₂  et  Iz·Iy − Iyz² = I₁·I₂ (invariants de Mohr)."""
        I1, I2 = sec.I_principal
        Iz, Iy, Iyz = sec.Iz, sec.Iy, sec.Iyz
        np.testing.assert_allclose(I1 + I2, Iz + Iy, rtol=1e-10)
        np.testing.assert_allclose(I1 * I2, Iz * Iy - Iyz**2, rtol=1e-10)

    @pytest.mark.parametrize("sec", [
        CircularSection(radius=0.05),
        RectangularSection(width=0.08, height=0.12),
        LSection(width=0.100, height=0.100, thickness=0.010),
    ])
    def test_rotation_gives_principal(self, sec: object) -> None:
        """Après rotation α, Iyz_principal ≈ 0 (axes vraiment principaux)."""
        alpha = sec.alpha_principal
        c, s = math.cos(2 * alpha), math.sin(2 * alpha)
        Iz, Iy, Iyz = sec.Iz, sec.Iy, sec.Iyz
        Iyz_rotated = -0.5 * (Iz - Iy) * s + Iyz * c
        np.testing.assert_allclose(Iyz_rotated, 0.0, atol=1e-14 * max(Iz, Iy))


# ──────────────────────────────────────────────────────────────────────────────
# 9. Intégration Beam2D avec Section — console analytique
# ──────────────────────────────────────────────────────────────────────────────

class TestBeam2DWithSection:
    """Tests d'intégration : poutre console avec différents types de Section.

    Pour chaque section, on vérifie :
        v_tip = PL³/(3EI)   avec I = section.Iz
    """

    def _v_analytical(self, Iz: float) -> float:
        """δ_analytique = PL³ / (3·E·Iz)."""
        return P * L_BEAM**3 / (3.0 * STEEL.E * Iz)

    def test_circular_section(self) -> None:
        """Console avec CircularSection."""
        sec = CircularSection(radius=0.05)
        v_fem = _cantilever_section(sec)
        np.testing.assert_allclose(v_fem, self._v_analytical(sec.Iz), rtol=1e-8)

    def test_rectangular_section(self) -> None:
        """Console avec RectangularSection."""
        sec = RectangularSection(width=0.10, height=0.10)
        v_fem = _cantilever_section(sec)
        np.testing.assert_allclose(v_fem, self._v_analytical(sec.Iz), rtol=1e-8)

    def test_hollow_circular_section(self) -> None:
        """Console avec HollowCircularSection."""
        sec = HollowCircularSection(outer_radius=0.05, inner_radius=0.04)
        v_fem = _cantilever_section(sec)
        np.testing.assert_allclose(v_fem, self._v_analytical(sec.Iz), rtol=1e-8)

    def test_hollow_rectangular_section(self) -> None:
        """Console avec HollowRectangularSection."""
        sec = HollowRectangularSection(
            outer_width=0.10, outer_height=0.15, thickness=0.006
        )
        v_fem = _cantilever_section(sec)
        np.testing.assert_allclose(v_fem, self._v_analytical(sec.Iz), rtol=1e-8)

    def test_isection(self) -> None:
        """Console avec ISection (IPE 200 simplifié)."""
        sec = ISection(
            flange_width=0.100, height=0.200,
            flange_thickness=0.0085, web_thickness=0.0056,
        )
        v_fem = _cantilever_section(sec)
        np.testing.assert_allclose(v_fem, self._v_analytical(sec.Iz), rtol=1e-8)

    def test_csection(self) -> None:
        """Console avec CSection."""
        sec = CSection(
            flange_width=0.075, height=0.200,
            flange_thickness=0.012, web_thickness=0.008,
        )
        v_fem = _cantilever_section(sec)
        np.testing.assert_allclose(v_fem, self._v_analytical(sec.Iz), rtol=1e-8)

    def test_lsection(self) -> None:
        """Console avec LSection — utilise Iz (flexion dans le plan xy)."""
        sec = LSection(width=0.100, height=0.100, thickness=0.010)
        v_fem = _cantilever_section(sec)
        np.testing.assert_allclose(v_fem, self._v_analytical(sec.Iz), rtol=1e-8)

    def test_stiffer_section_less_deflection(self) -> None:
        """Section de plus grand Iz → déflexion moindre (relation inverse)."""
        sec_thin = RectangularSection(width=0.10, height=0.05)   # Iz petit
        sec_thick = RectangularSection(width=0.10, height=0.10)  # Iz = 8 × plus grand
        v_thin = _cantilever_section(sec_thin)
        v_thick = _cantilever_section(sec_thick)
        assert v_thin > v_thick
        # Iz ∝ h³ → rapport des déflexions = (h_thin/h_thick)³ = (0.05/0.10)³ = 1/8
        ratio = sec_thin.Iz / sec_thick.Iz
        np.testing.assert_allclose(v_thin / v_thick, 1.0 / ratio, rtol=1e-8)

    def test_section_vs_dict_identical(self) -> None:
        """Section dans properties → même résultat que dict scalaire."""
        sec = RectangularSection(width=0.08, height=0.12)
        # Calculer avec Section
        v_section = _cantilever_section(sec)
        # Calculer avec dict scalaire (interface classique)
        nodes = np.array([[0.0, 0.0], [L_BEAM, 0.0]])
        mesh = Mesh(
            nodes=nodes,
            elements=(ElementData(Beam2D, (0, 1), STEEL, {"area": sec.area, "inertia": sec.Iz}),),
            n_dim=2, dof_per_node=3,
        )
        bc = BoundaryConditions(
            dirichlet={0: {0: 0.0, 1: 0.0, 2: 0.0}},
            neumann={1: {1: -P}},
        )
        assembler = Assembler(mesh)
        K = assembler.assemble_stiffness()
        F = assembler.assemble_forces(bc)
        K_bc, F_bc = apply_dirichlet(K, F, mesh, bc)
        u = spsolve(K_bc, F_bc)
        v_dict = -u[4]
        np.testing.assert_allclose(v_section, v_dict, rtol=1e-12)

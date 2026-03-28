"""Tests unitaires pour l'élément Tetra4 — validation par solution analytique.

Structure des tests
-------------------
1. TestTetra4BMatrix       : forme, volume, coefficients analytiques du tétraèdre
                             unitaire, erreurs attendues.
2. TestTetra4StiffnessMatrix: forme, symétrie, modes rigides, valeur analytique K[0,0].
3. TestTetra4MassMatrix    : forme, symétrie, masse totale, somme par ligne.
4. TestTetra4StrainStress  : déformations et contraintes sous champs connus.
5. TestTetra4PatchTest     : assemblage + résolution statique sur cube [0,1]³
                             découpé en 5 Tetra4 — comparaison à la solution
                             analytique de traction uniaxiale.

Solutions analytiques de référence
-----------------------------------
Tétraèdre unitaire (nœuds aux sommets (0,0,0),(1,0,0),(0,1,0),(0,0,1)) :
  - Volume V = 1/6
  - J = I₃  →  dN_phys = dN_nat
  - B[:, 0] = [-1, 0, 0, 0, -1, -1] (colonne ux₀)

Traction uniaxiale (E=1, ν=0, σ_xx=1 sur cube [0,1]³) :
  - u_x = x,  u_y = u_z = 0  (solution exacte pour ν=0)
  - σ = [1, 0, 0, 0, 0, 0]  (uniforme dans chaque élément)
"""

from __future__ import annotations

import numpy as np
import pytest

from femsolver.core.assembler import Assembler
from femsolver.core.boundary import apply_dirichlet
from femsolver.core.material import ElasticMaterial
from femsolver.core.mesh import BoundaryConditions, ElementData, Mesh
from femsolver.core.solver import StaticSolver
from femsolver.elements.tetra4 import Tetra4


# ---------------------------------------------------------------------------
# Données communes
# ---------------------------------------------------------------------------

# Matériau simplifié (E=1, ν=0) : découplage des directions, calculs simples
MAT_SIMPLE = ElasticMaterial(E=1.0, nu=0.0, rho=1.0)
MAT_STEEL  = ElasticMaterial(E=210e9, nu=0.3, rho=7800)

# Tétraèdre unitaire : nœuds aux 4 coins du trièdre trirectangle
# J = I₃,  det(J) = 1,  V = 1/6
NODES_UNIT = np.array([
    [0.0, 0.0, 0.0],   # nœud 0
    [1.0, 0.0, 0.0],   # nœud 1
    [0.0, 1.0, 0.0],   # nœud 2
    [0.0, 0.0, 1.0],   # nœud 3
])
PROPS_EMPTY: dict = {}


# ---------------------------------------------------------------------------
# 1. Tests de la matrice B
# ---------------------------------------------------------------------------


class TestTetra4BMatrix:
    """Vérification de la matrice déformation–déplacement B et du volume."""

    def test_shape(self) -> None:
        """B est de shape (6, 12)."""
        B, _ = Tetra4()._strain_displacement_matrix(NODES_UNIT)
        assert B.shape == (6, 12)

    def test_volume_unit_tet(self) -> None:
        """Tétraèdre unitaire : V = det(I) / 6 = 1/6."""
        _, volume = Tetra4()._strain_displacement_matrix(NODES_UNIT)
        np.testing.assert_allclose(volume, 1.0 / 6.0, rtol=1e-12)

    def test_volume_scaled_tet(self) -> None:
        """Tétraèdre avec arêtes de longueur a : V = a³/6."""
        a = 2.5
        nodes = NODES_UNIT * a
        _, volume = Tetra4()._strain_displacement_matrix(nodes)
        np.testing.assert_allclose(volume, a**3 / 6.0, rtol=1e-12)

    def test_b_coefficients_unit_tet(self) -> None:
        """Vérification analytique de B sur le tétraèdre unitaire.

        Pour J = I₃ :
            dN_phys = J⁻¹ · dN_nat = dN_nat
            ∂N0/∂x = -1,  ∂N0/∂y = -1,  ∂N0/∂z = -1
            ∂N1/∂x =  1,  ∂N1/∂y =  0,  ∂N1/∂z =  0
            ∂N2/∂x =  0,  ∂N2/∂y =  1,  ∂N2/∂z =  0
            ∂N3/∂x =  0,  ∂N3/∂y =  0,  ∂N3/∂z =  1

        B attendue (6×12) :
             0  1  2   3  4  5   6  7  8   9 10 11
         εxx[-1, 0, 0,  1, 0, 0,  0, 0, 0,  0, 0, 0]
         εyy[ 0,-1, 0,  0, 0, 0,  0, 1, 0,  0, 0, 0]
         εzz[ 0, 0,-1,  0, 0, 0,  0, 0, 0,  0, 0, 1]
         γyz[ 0,-1,-1,  0, 0, 0,  0, 0, 1,  0, 1, 0]
         γxz[-1, 0,-1,  0, 0, 1,  0, 0, 0,  1, 0, 0]
         γxy[-1,-1, 0,  0, 1, 0,  1, 0, 0,  0, 0, 0]
        """
        B, _ = Tetra4()._strain_displacement_matrix(NODES_UNIT)
        B_expected = np.array([
            # cols: ux0 uy0 uz0  ux1 uy1 uz1  ux2 uy2 uz2  ux3 uy3 uz3
            [-1,  0,  0,   1,  0,  0,   0,  0,  0,   0,  0,  0],  # εxx
            [ 0, -1,  0,   0,  0,  0,   0,  1,  0,   0,  0,  0],  # εyy
            [ 0,  0, -1,   0,  0,  0,   0,  0,  0,   0,  0,  1],  # εzz
            [ 0, -1, -1,   0,  0,  0,   0,  0,  1,   0,  1,  0],  # γyz
            [-1,  0, -1,   0,  0,  1,   0,  0,  0,   1,  0,  0],  # γxz
            [-1, -1,  0,   0,  1,  0,   1,  0,  0,   0,  0,  0],  # γxy
        ], dtype=float)
        np.testing.assert_allclose(B, B_expected, atol=1e-12)

    def test_degenerate_coplanar_raises(self) -> None:
        """4 nœuds coplanaires (V = 0) → ValueError."""
        nodes_flat = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.0],  # dans le même plan z=0
        ])
        with pytest.raises(ValueError):
            Tetra4()._strain_displacement_matrix(nodes_flat)

    def test_negative_volume_raises(self) -> None:
        """Orientation incorrecte (det J < 0) → ValueError."""
        # Inversion des nœuds 1 et 2 retourne det(J) = -1
        nodes_inv = NODES_UNIT[[0, 2, 1, 3], :]
        with pytest.raises(ValueError, match="det"):
            Tetra4()._strain_displacement_matrix(nodes_inv)

    def test_wrong_shape_raises(self) -> None:
        """nodes.shape ≠ (4, 3) → ValueError."""
        with pytest.raises(ValueError, match="shape"):
            Tetra4()._strain_displacement_matrix(np.zeros((3, 3)))


# ---------------------------------------------------------------------------
# 2. Tests de la matrice de rigidité K_e
# ---------------------------------------------------------------------------


class TestTetra4StiffnessMatrix:
    """Vérification de K_e."""

    def test_shape(self) -> None:
        """K_e est de shape (12, 12)."""
        K_e = Tetra4().stiffness_matrix(MAT_SIMPLE, NODES_UNIT, PROPS_EMPTY)
        assert K_e.shape == (12, 12)

    def test_symmetry(self) -> None:
        """K_e est symétrique."""
        K_e = Tetra4().stiffness_matrix(MAT_STEEL, NODES_UNIT, PROPS_EMPTY)
        np.testing.assert_allclose(K_e, K_e.T, atol=1e-6)

    def test_rank_before_bc(self) -> None:
        """K_e est de rang 6 avant CL (12 DDL − 6 modes rigides = 6 modes déformables).

        Un corps solide 3D admet exactement 6 modes rigides :
          3 translations (ux=cst, uy=cst, uz=cst)
          3 rotations (autour de chaque axe)
        """
        K_e = Tetra4().stiffness_matrix(MAT_SIMPLE, NODES_UNIT, PROPS_EMPTY)
        rank = np.linalg.matrix_rank(K_e, tol=1e-10)
        assert rank == 6

    def test_rigid_translation_x(self) -> None:
        """Translation rigide en x → K_e · u = 0 (aucune force interne)."""
        K_e = Tetra4().stiffness_matrix(MAT_SIMPLE, NODES_UNIT, PROPS_EMPTY)
        u_tx = np.zeros(12)
        u_tx[0::3] = 1.0                            # ux = 1 pour tous les nœuds
        f = K_e @ u_tx
        np.testing.assert_allclose(f, 0.0, atol=1e-12)

    def test_rigid_translation_y(self) -> None:
        """Translation rigide en y → K_e · u = 0."""
        K_e = Tetra4().stiffness_matrix(MAT_SIMPLE, NODES_UNIT, PROPS_EMPTY)
        u_ty = np.zeros(12)
        u_ty[1::3] = 1.0
        np.testing.assert_allclose(K_e @ u_ty, 0.0, atol=1e-12)

    def test_rigid_translation_z(self) -> None:
        """Translation rigide en z → K_e · u = 0."""
        K_e = Tetra4().stiffness_matrix(MAT_SIMPLE, NODES_UNIT, PROPS_EMPTY)
        u_tz = np.zeros(12)
        u_tz[2::3] = 1.0
        np.testing.assert_allclose(K_e @ u_tz, 0.0, atol=1e-12)

    def test_k00_analytical_nu0(self) -> None:
        """Valeur analytique K_e[0,0] pour E=1, ν=0, tétraèdre unitaire.

        DOF 0 = ux₀. Colonne 0 de B = [-1, 0, 0, 0, -1, -1]ᵀ.
        Pour E=1, ν=0 : D = diag([1, 1, 1, 0.5, 0.5, 0.5]).

        K_e[0,0] = (Bᵀ D B)_{0,0} · V
                 = (Σᵢ Bᵢ₀² · Dᵢᵢ) · (1/6)
                 = ((-1)²·1 + (-1)²·0.5 + (-1)²·0.5) · (1/6)
                 = (1 + 0.5 + 0.5) / 6 = 2/6 = 1/3
        """
        K_e = Tetra4().stiffness_matrix(MAT_SIMPLE, NODES_UNIT, PROPS_EMPTY)
        np.testing.assert_allclose(K_e[0, 0], 1.0 / 3.0, rtol=1e-12)

    def test_positive_semidefinite(self) -> None:
        """Les valeurs propres de K_e sont ≥ 0 (6 quasi-nulles, 6 positives).

        Les 6 valeurs propres "rigides" ne sont pas exactement nulles : leur
        magnitude dépend de E (ici ~210 GPa) et de l'arrondi flottant.
        La tolérance est relative à la plus grande valeur propre.
        """
        K_e = Tetra4().stiffness_matrix(MAT_STEEL, NODES_UNIT, PROPS_EMPTY)
        eigenvalues = np.linalg.eigvalsh(K_e)
        lam_max = eigenvalues[-1]
        tol = lam_max * 1e-8          # tolérance relative : ~1e-8 × E
        assert np.all(eigenvalues > -tol), (
            f"Valeurs propres négatives significatives : {eigenvalues[eigenvalues < -tol]}"
        )
        # Les 6 dernières valeurs propres représentent les modes déformables
        assert np.sum(eigenvalues > lam_max * 1e-6) == 6

    def test_non_unit_tet_symmetry(self) -> None:
        """K_e reste symétrique pour un tétraèdre quelconque (non unitaire)."""
        nodes = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.5, 0.1],
            [0.3, 1.8, 0.2],
            [0.4, 0.3, 1.5],
        ])
        K_e = Tetra4().stiffness_matrix(MAT_STEEL, nodes, PROPS_EMPTY)
        np.testing.assert_allclose(K_e, K_e.T, atol=1e-4)


# ---------------------------------------------------------------------------
# 3. Tests de la matrice de masse
# ---------------------------------------------------------------------------


class TestTetra4MassMatrix:
    """Vérification de M_e."""

    def test_shape(self) -> None:
        """M_e est de shape (12, 12)."""
        M_e = Tetra4().mass_matrix(MAT_SIMPLE, NODES_UNIT, PROPS_EMPTY)
        assert M_e.shape == (12, 12)

    def test_symmetry(self) -> None:
        """M_e est symétrique."""
        M_e = Tetra4().mass_matrix(MAT_STEEL, NODES_UNIT, PROPS_EMPTY)
        np.testing.assert_allclose(M_e, M_e.T, atol=1e-14)

    def test_total_mass_via_subblock(self) -> None:
        """Somme de la sous-matrice ux-ux = ρV (masse totale de l'élément).

        En notation scalaire (une direction), la matrice 4×4 est :
            m_scalar = ρV/20 · [[2,1,1,1],[1,2,1,1],[1,1,2,1],[1,1,1,2]]

        Σ m_scalar = ρV/20 · (4·2 + 12·1) = ρV/20 · 20 = ρV.
        """
        rho = MAT_STEEL.rho
        V = 1.0 / 6.0
        M_e = Tetra4().mass_matrix(MAT_STEEL, NODES_UNIT, PROPS_EMPTY)
        # Sous-matrice ux-ux (indices 0,3,6,9)
        ux_idx = [0, 3, 6, 9]
        M_ux = M_e[np.ix_(ux_idx, ux_idx)]
        np.testing.assert_allclose(M_ux.sum(), rho * V, rtol=1e-12)

    def test_row_sum_per_dof(self) -> None:
        """Chaque ligne de M_e somme à ρV/4.

        Avec la partition de l'unité ΣNi = 1, la masse inertielle
        de chaque DDL (somme de ligne = masse condensée) vaut ρV/4.
        """
        rho = MAT_STEEL.rho
        V = 1.0 / 6.0
        M_e = Tetra4().mass_matrix(MAT_STEEL, NODES_UNIT, PROPS_EMPTY)
        row_sums = M_e.sum(axis=1)
        np.testing.assert_allclose(row_sums, rho * V / 4.0, rtol=1e-12)

    def test_diagonal_value(self) -> None:
        """Chaque terme diagonal vaut 2ρV/20 = ρV/10."""
        rho = MAT_SIMPLE.rho   # = 1
        V = 1.0 / 6.0
        M_e = Tetra4().mass_matrix(MAT_SIMPLE, NODES_UNIT, PROPS_EMPTY)
        np.testing.assert_allclose(
            np.diag(M_e), rho * V / 10.0, rtol=1e-12
        )

    def test_off_diagonal_block_value(self) -> None:
        """Bloc hors-diagonale M_e[0:3, 3:6] = (ρV/20)·I₃."""
        rho = MAT_SIMPLE.rho
        V = 1.0 / 6.0
        M_e = Tetra4().mass_matrix(MAT_SIMPLE, NODES_UNIT, PROPS_EMPTY)
        expected_block = (rho * V / 20.0) * np.eye(3)
        np.testing.assert_allclose(M_e[0:3, 3:6], expected_block, rtol=1e-12)

    def test_scaled_tet_mass(self) -> None:
        """Tétraèdre d'arête a : masse totale = ρ · a³/6."""
        rho = 2000.0
        a = 0.3
        mat = ElasticMaterial(E=1e9, nu=0.3, rho=rho)
        nodes = NODES_UNIT * a
        M_e = Tetra4().mass_matrix(mat, nodes, PROPS_EMPTY)
        expected_mass = rho * (a**3 / 6.0)
        M_ux = M_e[np.ix_([0, 3, 6, 9], [0, 3, 6, 9])]
        np.testing.assert_allclose(M_ux.sum(), expected_mass, rtol=1e-12)


# ---------------------------------------------------------------------------
# 4. Tests déformations et contraintes
# ---------------------------------------------------------------------------


class TestTetra4StrainStress:
    """Vérification de strain() et stress() sous champs analytiques."""

    def test_strain_uniaxial_x(self) -> None:
        """Traction uniaxiale u_x = ε₀·x → ε = [ε₀, 0, 0, 0, 0, 0].

        Déplacements appliqués aux nœuds du tétraèdre unitaire :
            nœud 0 (x=0) : u = (0, 0, 0)
            nœud 1 (x=1) : u = (ε₀, 0, 0)
            nœud 2 (x=0) : u = (0, 0, 0)
            nœud 3 (x=0) : u = (0, 0, 0)

        Résultat attendu : B · u_e = [ε₀, 0, 0, 0, 0, 0].
        """
        eps0 = 1e-3
        u_e = np.zeros(12)
        u_e[3] = eps0  # ux du nœud 1 (col 3)
        epsilon = Tetra4().strain(NODES_UNIT, u_e)
        expected = np.array([eps0, 0.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(epsilon, expected, atol=1e-14)

    def test_strain_uniaxial_poisson(self) -> None:
        """Traction σ_xx = σ₀ (E, ν) → ε = [σ₀/E, -ν σ₀/E, -ν σ₀/E, 0, 0, 0].

        Déplacements uniaxiaux avec effet Poisson :
            ux = (σ₀/E) · x,  uy = −(ν σ₀/E) · y,  uz = −(ν σ₀/E) · z

        Sur le tétraèdre unitaire :
            nœud 0 (0,0,0) : u = (0, 0, 0)
            nœud 1 (1,0,0) : u = (σ₀/E, 0, 0)
            nœud 2 (0,1,0) : u = (0, -ν σ₀/E, 0)
            nœud 3 (0,0,1) : u = (0, 0, -ν σ₀/E)
        """
        E, nu = 210e9, 0.3
        sigma0 = 1e6
        eps_axial  =  sigma0 / E
        eps_lateral = -nu * sigma0 / E

        u_e = np.zeros(12)
        # nœud 1 (x=1) : ux = σ0/E
        u_e[3] = eps_axial
        # nœud 2 (y=1) : uy = -ν σ0/E
        u_e[7] = eps_lateral
        # nœud 3 (z=1) : uz = -ν σ0/E
        u_e[11] = eps_lateral

        mat = ElasticMaterial(E=E, nu=nu, rho=7800)
        epsilon = Tetra4().strain(NODES_UNIT, u_e)
        expected = np.array([eps_axial, eps_lateral, eps_lateral, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(epsilon, expected, rtol=1e-12)

    def test_stress_uniaxial_recovers_applied(self) -> None:
        """σ = D · ε → σxx = E·ε₀ = σ₀, autres composantes ≈ 0 (ν=0).

        Pour ν=0 : D = diag([1, 1, 1, 0.5, 0.5, 0.5]) (avec E=1).
        Avec ε = [ε₀, 0, …, 0] : σ = [ε₀, 0, …, 0] = [σ₀, 0, …, 0].
        """
        eps0 = 2e-3
        u_e = np.zeros(12)
        u_e[3] = eps0   # ux du nœud 1
        sigma = Tetra4().stress(MAT_SIMPLE, NODES_UNIT, u_e)
        np.testing.assert_allclose(sigma[0], eps0, rtol=1e-12)
        np.testing.assert_allclose(sigma[1:], 0.0, atol=1e-14)

    def test_stress_uniaxial_full_poisson(self) -> None:
        """Traction pure σxx = σ₀ avec ν≠0 → seul σxx ≠ 0 (contrainte uniaxiale).

        En appliquant les déplacements analytiques (εxx=σ₀/E, εyy=εzz=-ν σ₀/E),
        la contrainte recalculée vaut exactement [σ₀, 0, 0, 0, 0, 0].
        """
        E, nu = 210e9, 0.3
        sigma0 = 5e6
        eps_axial   = sigma0 / E
        eps_lateral = -nu * sigma0 / E

        u_e = np.zeros(12)
        u_e[3]  = eps_axial    # ux nœud 1
        u_e[7]  = eps_lateral  # uy nœud 2
        u_e[11] = eps_lateral  # uz nœud 3

        mat = ElasticMaterial(E=E, nu=nu, rho=7800)
        sigma = Tetra4().stress(mat, NODES_UNIT, u_e)
        np.testing.assert_allclose(sigma[0], sigma0, rtol=1e-10)
        np.testing.assert_allclose(sigma[1], 0.0, atol=1.0)   # σyy ≈ 0 (tolérance absolue)
        np.testing.assert_allclose(sigma[2], 0.0, atol=1.0)   # σzz ≈ 0
        np.testing.assert_allclose(sigma[3:], 0.0, atol=1.0)  # τ ≈ 0

    def test_strain_shear_xz(self) -> None:
        """Cisaillement ux = γ₀·z → γxz = γ₀, autres composantes = 0.

        Déplacements : ux_i = γ₀ · z_i, uy_i = 0, uz_i = 0.
            nœud 0 (z=0) : u = (0, 0, 0)
            nœud 1 (z=0) : u = (0, 0, 0)
            nœud 2 (z=0) : u = (0, 0, 0)
            nœud 3 (z=1) : u = (γ₀, 0, 0)
        """
        gamma0 = 5e-4
        u_e = np.zeros(12)
        u_e[9] = gamma0   # ux du nœud 3 (z=1), colonne 3×3+0=9
        epsilon = Tetra4().strain(NODES_UNIT, u_e)
        np.testing.assert_allclose(epsilon[0], 0.0, atol=1e-14)  # εxx
        np.testing.assert_allclose(epsilon[1], 0.0, atol=1e-14)  # εyy
        np.testing.assert_allclose(epsilon[2], 0.0, atol=1e-14)  # εzz
        np.testing.assert_allclose(epsilon[3], 0.0, atol=1e-14)  # γyz
        np.testing.assert_allclose(epsilon[4], gamma0, rtol=1e-12)  # γxz
        np.testing.assert_allclose(epsilon[5], 0.0, atol=1e-14)  # γxy


# ---------------------------------------------------------------------------
# 5. Patch test 3D — assemblage + résolution statique
# ---------------------------------------------------------------------------


class TestTetra4PatchTest:
    """Patch test 3D : traction uniaxiale σxx = 1 sur cube [0,1]³.

    Décomposition de Sommerville (5 Tetra4, tous det > 0) :
        T1 (0,1,2,4) : det J = 1,  V = 1/6
        T2 (1,3,2,7) : det J = 1,  V = 1/6
        T3 (1,4,5,7) : det J = 1,  V = 1/6
        T4 (2,6,4,7) : det J = 1,  V = 1/6
        T5 (1,2,4,7) : det J = 2,  V = 1/3

    Numérotation des nœuds (binaire : bit2=z, bit1=y, bit0=x) :
        0=(0,0,0)  1=(1,0,0)  2=(0,1,0)  3=(1,1,0)
        4=(0,0,1)  5=(1,0,1)  6=(0,1,1)  7=(1,1,1)

    Solution analytique (E=1, ν=0, σ₀=1) :
        ux = x,  uy = 0,  uz = 0

    Forces nodales consistantes sur la face x=1 (2 triangles : (1,3,7) et (1,5,7)) :
        F_nœud1 = σ₀·A_face/3 + σ₀·A_face/3 = 1/3  (nœud commun)
        F_nœud3 = σ₀·A_face/3                = 1/6
        F_nœud5 = σ₀·A_face/3                = 1/6
        F_nœud7 = σ₀·A_face/3 + σ₀·A_face/3 = 1/3  (nœud commun)
    """

    def setup_method(self) -> None:
        self.E    = 1.0
        self.nu   = 0.0
        self.mat  = ElasticMaterial(E=self.E, nu=self.nu, rho=1.0)

    def _build_mesh(self) -> tuple[Mesh, BoundaryConditions]:
        """Cube [0,1]³ découpé en 5 Tetra4 (Sommerville Type 1)."""
        nodes = np.array([
            [0., 0., 0.],  # 0
            [1., 0., 0.],  # 1
            [0., 1., 0.],  # 2
            [1., 1., 0.],  # 3
            [0., 0., 1.],  # 4
            [1., 0., 1.],  # 5
            [0., 1., 1.],  # 6
            [1., 1., 1.],  # 7
        ])
        connectivity = [
            (0, 1, 2, 4),   # T1 — det = 1
            (1, 3, 2, 7),   # T2 — det = 1
            (1, 4, 5, 7),   # T3 — det = 1
            (2, 6, 4, 7),   # T4 — det = 1
            (1, 2, 4, 7),   # T5 — det = 2
        ]
        elements = tuple(
            ElementData(Tetra4, nc, self.mat, {})
            for nc in connectivity
        )
        mesh = Mesh(nodes=nodes, elements=elements, n_dim=3)

        # Conditions de Dirichlet — 6 contraintes = 6 modes rigides 3D supprimés :
        #   • ux=0 aux 4 nœuds de la face x=0 : translations-x, rotations-y/z
        #   • uy=0 au nœud 0 : translation-y
        #   • uy=0 au nœud 4 (z=1) : rotation-x (u_rot_x = ω×r, composante y ∝ z)
        #   • uz=0 au nœud 0 : translation-z (rotation-y déjà bloquée par ux)
        dirichlet: dict[int, dict[int, float]] = {
            0: {0: 0.0, 1: 0.0, 2: 0.0},  # origine : blocage complet
            2: {0: 0.0},                    # (0,1,0) : ux=0
            4: {0: 0.0, 1: 0.0},            # (0,0,1) : ux=uy=0
            6: {0: 0.0},                    # (0,1,1) : ux=0
        }
        # Forces nodales consistantes : σxx=1, face x=1 composée de 2 triangles
        # Triangle (1,3,7) : aire=0.5  →  F = σ₀·0.5/3 = 1/6 par nœud
        # Triangle (1,5,7) : aire=0.5  →  F = σ₀·0.5/3 = 1/6 par nœud
        neumann: dict[int, dict[int, float]] = {
            1: {0: 1.0 / 3.0},   # nœud 1 dans les 2 triangles : 1/6 + 1/6 = 1/3
            3: {0: 1.0 / 6.0},   # nœud 3 dans triangle (1,3,7) seulement
            5: {0: 1.0 / 6.0},   # nœud 5 dans triangle (1,5,7) seulement
            7: {0: 1.0 / 3.0},   # nœud 7 dans les 2 triangles : 1/3
        }
        bc = BoundaryConditions(dirichlet=dirichlet, neumann=neumann)
        return mesh, bc

    def _solve(self) -> tuple[np.ndarray, Mesh]:
        """Assemble et résout le système."""
        mesh, bc = self._build_mesh()
        assembler = Assembler(mesh)
        K = assembler.assemble_stiffness()
        F = assembler.assemble_forces(bc)
        K_bc, F_bc = apply_dirichlet(K, F, mesh, bc)
        u = StaticSolver().solve(K_bc, F_bc)
        return u, mesh

    def test_displacement_matches_analytical(self) -> None:
        """ux = x sur tous les nœuds, uy = uz = 0 (solution exacte pour ν=0).

        Le Tetra4 est à déformation constante (linéaire complet) :
        tout champ de déplacement linéaire est reproduit exactement.
        Tolérance atol = 1e-10 (précision de la pénalisation).
        """
        u, mesh = self._solve()
        for i, (x, y, z) in enumerate(mesh.nodes):
            np.testing.assert_allclose(
                u[3 * i    ], x,   atol=1e-10,
                err_msg=f"ux nœud {i} (x={x})"
            )
            np.testing.assert_allclose(
                u[3 * i + 1], 0.0, atol=1e-10,
                err_msg=f"uy nœud {i}"
            )
            np.testing.assert_allclose(
                u[3 * i + 2], 0.0, atol=1e-10,
                err_msg=f"uz nœud {i}"
            )

    def test_strain_constant_unit_in_all_elements(self) -> None:
        """εxx = 1/E = 1 dans chaque élément, autres composantes = 0."""
        u, mesh = self._solve()
        tet = Tetra4()
        for k, elem_data in enumerate(mesh.elements):
            coords = mesh.node_coords(elem_data.node_ids)
            dofs   = mesh.global_dofs(elem_data.node_ids)
            u_e    = u[dofs]
            eps    = tet.strain(coords, u_e)
            np.testing.assert_allclose(
                eps[0], 1.0, atol=1e-10,
                err_msg=f"εxx élément {k}"
            )
            np.testing.assert_allclose(
                eps[1:], 0.0, atol=1e-10,
                err_msg=f"autres composantes ε élément {k}"
            )

    def test_stress_uniform_in_all_elements(self) -> None:
        """σxx = 1 = σ₀ dans chaque élément, autres contraintes = 0."""
        u, mesh = self._solve()
        tet = Tetra4()
        for k, elem_data in enumerate(mesh.elements):
            coords = mesh.node_coords(elem_data.node_ids)
            dofs   = mesh.global_dofs(elem_data.node_ids)
            u_e    = u[dofs]
            sigma  = tet.stress(self.mat, coords, u_e)
            np.testing.assert_allclose(
                sigma[0], 1.0, atol=1e-10,
                err_msg=f"σxx élément {k}"
            )
            np.testing.assert_allclose(
                sigma[1:], 0.0, atol=1e-10,
                err_msg=f"autres contraintes élément {k}"
            )

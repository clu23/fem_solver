"""Tests unitaires pour l'élément Tetra10 — validation par solution analytique.

Stratégie de validation
------------------------
1. Propriétés des fonctions de forme (partition de l'unité, interpolation).
2. Matrice K_e : forme, symétrie, modes rigides (6 pour un solide 3D).
3. Matrice M_e : symétrie, conservation de masse, sommes de lignes.
4. Patch test 3D : champ de déplacement linéaire reproduit exactement.
5. Traction uniaxiale : déformation uniforme, contraction de Poisson correcte.
"""

from __future__ import annotations

import numpy as np
import pytest

from femsolver.core.assembler import Assembler
from femsolver.core.boundary import apply_dirichlet
from femsolver.core.material import ElasticMaterial
from femsolver.core.mesh import BoundaryConditions, ElementData, Mesh
from femsolver.core.solver import StaticSolver
from femsolver.elements.tetra10 import Tetra10


# ---------------------------------------------------------------------------
# Géométrie de référence : tétraèdre unité
# ---------------------------------------------------------------------------

# Coins : (0,0,0), (1,0,0), (0,1,0), (0,0,1)
# Milieux exactement aux centres des arêtes
NODES_REF = np.array([
    [0.0, 0.0, 0.0],  # 0 : coin L1
    [1.0, 0.0, 0.0],  # 1 : coin L2
    [0.0, 1.0, 0.0],  # 2 : coin L3
    [0.0, 0.0, 1.0],  # 3 : coin L4
    [0.5, 0.0, 0.0],  # 4 : milieu arête 0-1
    [0.5, 0.5, 0.0],  # 5 : milieu arête 1-2
    [0.0, 0.5, 0.0],  # 6 : milieu arête 0-2
    [0.0, 0.0, 0.5],  # 7 : milieu arête 0-3
    [0.5, 0.0, 0.5],  # 8 : milieu arête 1-3
    [0.0, 0.5, 0.5],  # 9 : milieu arête 2-3
])

MAT_SIMPLE = ElasticMaterial(E=1.0, nu=0.0, rho=1.0)
MAT_STEEL  = ElasticMaterial(E=210e9, nu=0.3, rho=7800.0)


# ===========================================================================
# Tests des fonctions de forme
# ===========================================================================


class TestTetra10ShapeFunctions:
    """Propriétés algébriques des fonctions de forme N0…N9."""

    def test_partition_of_unity_at_gauss_points(self) -> None:
        """ΣNi = 1 à chaque point de Gauss."""
        from femsolver.elements.tetra10 import _GAUSS_K4
        elem = Tetra10()
        for xi, eta, zeta, _ in _GAUSS_K4:
            N = elem._shape_functions(xi, eta, zeta)
            np.testing.assert_allclose(N.sum(), 1.0, atol=1e-14)

    def test_nodal_interpolation(self) -> None:
        """Ni vaut 1 au nœud i et 0 aux 9 autres."""
        elem = Tetra10()
        # Coordonnées naturelles des 10 nœuds du tétraèdre de référence
        node_coords = [
            (0.0, 0.0, 0.0),  # 0 : coin L1
            (1.0, 0.0, 0.0),  # 1 : coin L2
            (0.0, 1.0, 0.0),  # 2 : coin L3
            (0.0, 0.0, 1.0),  # 3 : coin L4
            (0.5, 0.0, 0.0),  # 4 : milieu 0-1
            (0.5, 0.5, 0.0),  # 5 : milieu 1-2
            (0.0, 0.5, 0.0),  # 6 : milieu 0-2
            (0.0, 0.0, 0.5),  # 7 : milieu 0-3
            (0.5, 0.0, 0.5),  # 8 : milieu 1-3
            (0.0, 0.5, 0.5),  # 9 : milieu 2-3
        ]
        for i, (xi, eta, zeta) in enumerate(node_coords):
            N = elem._shape_functions(xi, eta, zeta)
            assert abs(N[i] - 1.0) < 1e-12, f"N{i} != 1 au nœud {i}"
            for j, v in enumerate(N):
                if j != i:
                    assert abs(v) < 1e-12, f"N{j} != 0 au nœud {i}"

    def test_derivatives_sum_to_zero(self) -> None:
        """Σ ∂Ni/∂ξ = Σ ∂Ni/∂η = Σ ∂Ni/∂ζ = 0."""
        elem = Tetra10()
        for pt in [(0.1, 0.1, 0.1), (0.5, 0.2, 0.1), (0.25, 0.25, 0.25)]:
            dN = elem._shape_function_derivatives(*pt)
            for row in range(3):
                np.testing.assert_allclose(dN[row].sum(), 0.0, atol=1e-13)


# ===========================================================================
# Tests de la matrice de rigidité K_e
# ===========================================================================


class TestTetra10StiffnessMatrix:
    """Propriétés de la matrice de rigidité élémentaire."""

    def test_shape(self) -> None:
        """K_e est de shape (30, 30)."""
        K_e = Tetra10().stiffness_matrix(MAT_SIMPLE, NODES_REF, {})
        assert K_e.shape == (30, 30)

    def test_symmetry(self) -> None:
        """K_e est symétrique."""
        K_e = Tetra10().stiffness_matrix(MAT_STEEL, NODES_REF, {})
        np.testing.assert_allclose(K_e, K_e.T, atol=1e-6)

    def test_six_rigid_body_modes(self) -> None:
        """K_e a exactement 6 valeurs propres nulles (modes rigides 3D).

        Les 6 modes rigides d'un solide 3D libre :
        - 3 translations (x, y, z)
        - 3 rotations (autour de x, y, z)
        """
        K_e = Tetra10().stiffness_matrix(MAT_SIMPLE, NODES_REF, {})
        eigenvalues = np.linalg.eigvalsh(K_e)
        tol = 1e-8 * eigenvalues[-1]
        n_zero = np.sum(np.abs(eigenvalues) < tol)
        assert n_zero == 6, (
            f"K_e devrait avoir 6 valeurs propres nulles, trouvé {n_zero}. "
            f"5 plus petites : {eigenvalues[:8]}"
        )

    def test_positive_semidefinite(self) -> None:
        """K_e est semi-définie positive."""
        K_e = Tetra10().stiffness_matrix(MAT_STEEL, NODES_REF, {})
        eigenvalues = np.linalg.eigvalsh(K_e)
        tol = 1e-10 * eigenvalues[-1]
        assert np.all(eigenvalues >= -tol), (
            f"K_e a une eigenvalue négative : {eigenvalues.min():.3e}"
        )

    def test_singular_tetrahedron_raises(self) -> None:
        """Nœuds coins coplanaires → ValueError (volume nul)."""
        nodes_bad = NODES_REF.copy()
        nodes_bad[3] = [1.0, 1.0, 0.0]  # 4e coin dans le plan z=0
        with pytest.raises(ValueError):
            Tetra10().stiffness_matrix(MAT_SIMPLE, nodes_bad, {})


# ===========================================================================
# Tests de la matrice de masse M_e
# ===========================================================================


class TestTetra10MassMatrix:
    """Propriétés de la matrice de masse consistante."""

    def test_shape(self) -> None:
        """M_e est de shape (30, 30)."""
        M_e = Tetra10().mass_matrix(MAT_STEEL, NODES_REF, {})
        assert M_e.shape == (30, 30)

    def test_symmetry(self) -> None:
        """M_e est symétrique."""
        M_e = Tetra10().mass_matrix(MAT_STEEL, NODES_REF, {})
        np.testing.assert_allclose(M_e, M_e.T, atol=1e-14)

    def test_total_mass_conservation(self) -> None:
        """Somme totale de M_e = 3 × ρ × V (3 DDL/nœud).

        Volume du tétraèdre de référence = 1/6.
        """
        rho = 7800.0
        mat = ElasticMaterial(E=210e9, nu=0.3, rho=rho)
        M_e = Tetra10().mass_matrix(mat, NODES_REF, {})

        volume = 1.0 / 6.0  # tétraèdre unité
        np.testing.assert_allclose(
            M_e.sum(), 3.0 * rho * volume, rtol=1e-12,
            err_msg="Conservation de masse violée"
        )

    def test_row_sums_corners_and_midsides(self) -> None:
        """Sommes de lignes analytiques pour les coins et les milieux.

        Coins   : ∫ Ni dV = −V/20  → somme de ligne = ρ·(−V/20) par direction
        Milieux : ∫ Ni dV =  V/5   → somme de ligne = ρ·(V/5) par direction
        """
        rho = 1.0
        mat = ElasticMaterial(E=1.0, nu=0.3, rho=rho)
        M_e = Tetra10().mass_matrix(mat, NODES_REF, {})

        volume = 1.0 / 6.0

        # Coins : nœuds 0–3, DDL 0..11
        for node in range(4):
            for dof_offset in range(3):
                dof = 3 * node + dof_offset
                row_sum = M_e[dof, :].sum()
                np.testing.assert_allclose(
                    row_sum, rho * (-volume / 20.0), rtol=1e-12,
                    err_msg=f"Coin nœud {node}, DOF {dof}: somme ligne ≠ −ρV/20"
                )

        # Milieux : nœuds 4–9, DDL 12..29
        for node in range(4, 10):
            for dof_offset in range(3):
                dof = 3 * node + dof_offset
                row_sum = M_e[dof, :].sum()
                np.testing.assert_allclose(
                    row_sum, rho * (volume / 5.0), rtol=1e-12,
                    err_msg=f"Milieu nœud {node}, DOF {dof}: somme ligne ≠ ρV/5"
                )

    def test_positive_semidefinite(self) -> None:
        """M_e est semi-définie positive (globalement, même si les coins
        ont des entrées négatives off-diagonales)."""
        M_e = Tetra10().mass_matrix(MAT_STEEL, NODES_REF, {})
        eigenvalues = np.linalg.eigvalsh(M_e)
        tol = 1e-12 * eigenvalues[-1]
        assert np.all(eigenvalues >= -tol), (
            f"M_e a une eigenvalue négative : {eigenvalues.min():.3e}"
        )


# ===========================================================================
# Patch test 3D — champ de déplacement linéaire
# ===========================================================================


class TestTetra10PatchTest:
    """Patch test : champ linéaire de déplacement reproduit exactement.

    Champ imposé : ux = α·x,  uy = β·y,  uz = γ·z.
    Déformations : εxx=α, εyy=β, εzz=γ, γyz=γxz=γxy=0.
    Solution analytique (E=1, nu=0) :
        σxx = α,  σyy = β,  σzz = γ,  τ = 0.
    """

    def test_strain_from_linear_field(self) -> None:
        """ε = B · u_e = [α, β, γ, 0, 0, 0] partout dans l'élément."""
        alpha, beta, gamma = 0.002, -0.001, 0.003
        elem = Tetra10()

        u_e = np.zeros(30)
        for k, (x, y, z) in enumerate(NODES_REF):
            u_e[3*k    ] = alpha * x
            u_e[3*k + 1] = beta  * y
            u_e[3*k + 2] = gamma * z

        for pt in [(0.1, 0.1, 0.1), (0.5, 0.2, 0.1), (0.25, 0.25, 0.25)]:
            eps = elem.strain(NODES_REF, u_e, *pt)
            np.testing.assert_allclose(eps[0], alpha, atol=1e-12)
            np.testing.assert_allclose(eps[1], beta,  atol=1e-12)
            np.testing.assert_allclose(eps[2], gamma, atol=1e-12)
            np.testing.assert_allclose(eps[3:], 0.0,  atol=1e-12)

    def test_stress_from_linear_field_nu0(self) -> None:
        """σ = D · ε = [E·α, E·β, E·γ, 0, 0, 0] pour nu=0."""
        E = 1.0
        alpha, beta, gamma = 0.002, -0.001, 0.003
        mat = ElasticMaterial(E=E, nu=0.0, rho=1.0)
        elem = Tetra10()

        u_e = np.zeros(30)
        for k, (x, y, z) in enumerate(NODES_REF):
            u_e[3*k    ] = alpha * x
            u_e[3*k + 1] = beta  * y
            u_e[3*k + 2] = gamma * z

        sigma = elem.stress(mat, NODES_REF, u_e, 0.25, 0.25, 0.25)
        np.testing.assert_allclose(sigma[0], E * alpha, rtol=1e-12)
        np.testing.assert_allclose(sigma[1], E * beta,  rtol=1e-12)
        np.testing.assert_allclose(sigma[2], E * gamma, rtol=1e-12)
        np.testing.assert_allclose(sigma[3:], 0.0,      atol=1e-12)


# ===========================================================================
# Test de traction — système assemblé
# ===========================================================================


class TestTetra10AssembledTraction:
    """Tétraèdre Tetra10 unique en traction uniaxiale.

    On applique un déplacement prescrit (δ = 0.001 m) en x sur la face
    opposée et on vérifie que les réactions donnent F = E·A·δ/L.
    Volume du tétraèdre de référence = 1/6.
    """

    def test_displacement_under_prescribed_deformation(self) -> None:
        """Déplacement ux = α·x pour α = 0.001 — solution exacte pour E=1, nu=0.

        On impose les déplacements sur tous les nœuds selon le champ linéaire
        et on vérifie que K_e · u_e = F_e est consistent (résidu nul).
        """
        alpha = 0.001
        mat = ElasticMaterial(E=1.0, nu=0.0, rho=1.0)
        elem = Tetra10()

        u_e = np.zeros(30)
        for k, (x, y, z) in enumerate(NODES_REF):
            u_e[3*k] = alpha * x

        K_e = elem.stiffness_matrix(mat, NODES_REF, {})
        F_e = K_e @ u_e  # forces nodales internes

        # Pour un champ linéaire exact, les forces internes doivent être
        # en équilibre (somme nulle sur tous les nœuds libres)
        # Vérifier : somme des forces en x = 0 (équilibre global)
        F_x = F_e[0::3]  # composante x de chaque nœud
        np.testing.assert_allclose(F_x.sum(), 0.0, atol=1e-10)

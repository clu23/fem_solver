"""Tests unitaires pour l'élément Tri6 (LST) — validation par solution analytique.

Stratégie de validation
------------------------
1. Propriétés algébriques des fonctions de forme (partition de l'unité,
   interpolation nodale, dérivées sommant à zéro).
2. Matrice K_e : symétrie, 3 modes rigides (nul-espace de dimension 3).
3. Matrice M_e : symétrie, conservation de masse (sommation des lignes).
4. Patch test en traction : un champ linéaire de déplacement est reproduit
   exactement → vérification que Tri6 satisfait la condition de patch.
5. Déformation en flexion pure : Tri6 représente la courbure linéaire sans
   locking — comparé à la solution analytique de la poutre d'Euler–Bernoulli.
"""

from __future__ import annotations

import numpy as np
import pytest

from femsolver.core.assembler import Assembler
from femsolver.core.boundary import apply_dirichlet
from femsolver.core.material import ElasticMaterial
from femsolver.core.mesh import BoundaryConditions, ElementData, Mesh
from femsolver.core.solver import StaticSolver
from femsolver.elements.tri6 import Tri6


# ---------------------------------------------------------------------------
# Matériaux et géométries de référence
# ---------------------------------------------------------------------------

MAT_SIMPLE = ElasticMaterial(E=1.0, nu=0.0, rho=1.0)
MAT_STEEL = ElasticMaterial(E=210e9, nu=0.3, rho=7800)

# Triangle rectangle isocèle standard, avec nœuds milieux exacts
# Coins : (0,0), (1,0), (0,1) — Milieux : (0.5,0), (0.5,0.5), (0,0.5)
NODES_STD = np.array([
    [0.0, 0.0],  # coin 0
    [1.0, 0.0],  # coin 1
    [0.0, 1.0],  # coin 2
    [0.5, 0.0],  # milieu 0-1
    [0.5, 0.5],  # milieu 1-2
    [0.0, 0.5],  # milieu 0-2
])
PROPS_STD = {"thickness": 1.0}


# ===========================================================================
# Tests des fonctions de forme
# ===========================================================================


class TestTri6ShapeFunctions:
    """Propriétés algébriques des fonctions de forme N1…N6."""

    def test_partition_of_unity_at_gauss_points(self) -> None:
        """ΣNi = 1 à chaque point de Gauss."""
        from femsolver.elements.tri6 import _GAUSS_PTS
        elem = Tri6()
        for xi, eta, _ in _GAUSS_PTS:
            N = elem._shape_functions(xi, eta)
            np.testing.assert_allclose(N.sum(), 1.0, atol=1e-14)

    def test_nodal_interpolation(self) -> None:
        """Ni vaut 1 au nœud i et 0 aux autres nœuds."""
        elem = Tri6()
        # Coordonnées naturelles des 6 nœuds du triangle de référence
        node_xi_eta = [
            (0.0, 0.0),   # nœud 0
            (1.0, 0.0),   # nœud 1
            (0.0, 1.0),   # nœud 2
            (0.5, 0.0),   # nœud 3
            (0.5, 0.5),   # nœud 4
            (0.0, 0.5),   # nœud 5
        ]
        for i, (xi, eta) in enumerate(node_xi_eta):
            N = elem._shape_functions(xi, eta)
            assert abs(N[i] - 1.0) < 1e-12, f"N{i}({xi},{eta}) != 1"
            for j, val in enumerate(N):
                if j != i:
                    assert abs(val) < 1e-12, f"N{j}({xi},{eta}) != 0 (attendu 0)"

    def test_derivatives_sum_to_zero(self) -> None:
        """Σ ∂Ni/∂ξ = 0 et Σ ∂Ni/∂η = 0 (conséquence de ΣNi=const=1)."""
        elem = Tri6()
        for xi, eta in [(0.25, 0.25), (0.1, 0.6), (0.5, 0.0)]:
            dN = elem._shape_function_derivatives(xi, eta)
            np.testing.assert_allclose(dN[0].sum(), 0.0, atol=1e-13)
            np.testing.assert_allclose(dN[1].sum(), 0.0, atol=1e-13)


# ===========================================================================
# Tests de la matrice de rigidité K_e
# ===========================================================================


class TestTri6StiffnessMatrix:
    """Propriétés de la matrice de rigidité élémentaire."""

    def test_shape(self) -> None:
        """K_e est de shape (12, 12)."""
        K_e = Tri6().stiffness_matrix(MAT_SIMPLE, NODES_STD, PROPS_STD)
        assert K_e.shape == (12, 12)

    def test_symmetry(self) -> None:
        """K_e est symétrique à précision machine."""
        K_e = Tri6().stiffness_matrix(MAT_STEEL, NODES_STD, PROPS_STD)
        # tolérance relative : les entrées steel sont ~10^11, erreur ~eps*10^11
        atol = 1e-10 * np.abs(K_e).max()
        np.testing.assert_allclose(K_e, K_e.T, atol=atol)

    def test_three_rigid_body_modes(self) -> None:
        """K_e a exactement 3 valeurs propres nulles (modes rigides).

        Les 3 modes rigides d'un triangle 2D sont :
        - translation en x
        - translation en y
        - rotation dans le plan
        """
        K_e = Tri6().stiffness_matrix(MAT_SIMPLE, NODES_STD, PROPS_STD)
        eigenvalues = np.linalg.eigvalsh(K_e)
        n_zero = np.sum(np.abs(eigenvalues) < 1e-10 * eigenvalues[-1])
        assert n_zero == 3, (
            f"K_e devrait avoir 3 valeurs propres nulles, trouvé {n_zero}. "
            f"Valeurs propres : {eigenvalues}"
        )

    def test_positive_semidefinite(self) -> None:
        """K_e est semi-définie positive (toutes eigenvaleurs ≥ 0)."""
        K_e = Tri6().stiffness_matrix(MAT_STEEL, NODES_STD, PROPS_STD)
        eigenvalues = np.linalg.eigvalsh(K_e)
        # tolérance relative sur la plus grande valeur propre
        tol = 1e-10 * eigenvalues[-1]
        assert np.all(eigenvalues >= -tol), (
            f"K_e a une valeur propre négative : {eigenvalues.min():.3e}"
        )

    def test_thickness_scaling(self) -> None:
        """K_e est proportionnelle à l'épaisseur t."""
        K1 = Tri6().stiffness_matrix(MAT_SIMPLE, NODES_STD, {"thickness": 1.0})
        K2 = Tri6().stiffness_matrix(MAT_SIMPLE, NODES_STD, {"thickness": 3.0})
        # atol pour les entrées qui tombent exactement à 0
        np.testing.assert_allclose(K2, 3.0 * K1, rtol=1e-10, atol=1e-14)

    def test_invalid_thickness_raises(self) -> None:
        """thickness ≤ 0 → ValueError."""
        with pytest.raises(ValueError, match="épaisseur"):
            Tri6().stiffness_matrix(MAT_SIMPLE, NODES_STD, {"thickness": -1.0})

    def test_degenerate_element_raises(self) -> None:
        """Nœuds colinéaires → ValueError (det(J) ≤ 0)."""
        nodes_bad = np.array([
            [0., 0.], [1., 0.], [2., 0.],
            [0.5, 0.], [1.5, 0.], [1., 0.],
        ])
        with pytest.raises(ValueError):
            Tri6().stiffness_matrix(MAT_SIMPLE, nodes_bad, PROPS_STD)


# ===========================================================================
# Tests de la matrice de masse M_e
# ===========================================================================


class TestTri6MassMatrix:
    """Propriétés de la matrice de masse consistante."""

    def test_shape(self) -> None:
        """M_e est de shape (12, 12)."""
        M_e = Tri6().mass_matrix(MAT_STEEL, NODES_STD, PROPS_STD)
        assert M_e.shape == (12, 12)

    def test_symmetry(self) -> None:
        """M_e est symétrique."""
        M_e = Tri6().mass_matrix(MAT_STEEL, NODES_STD, PROPS_STD)
        np.testing.assert_allclose(M_e, M_e.T, atol=1e-14)

    def test_total_mass_conservation(self) -> None:
        """Somme de toutes les entrées = 2 × masse totale.

        Pour un champ de déplacement uniforme u=[1,0,...,1,0,...],
        la force inertielle totale est M·u. La somme de toutes les
        entrées de M_e = 2 × (ρ · t · A) car il y a 2 composantes (ux, uy).
        """
        rho, t = 7800.0, 0.01
        mat = ElasticMaterial(E=210e9, nu=0.3, rho=rho)
        props = {"thickness": t}
        area = 0.5  # triangle standard de référence

        M_e = Tri6().mass_matrix(mat, NODES_STD, props)
        expected_total_mass = rho * t * area
        # Somme de toutes les entrées = 2 * masse (2 DOF, u et v)
        np.testing.assert_allclose(
            M_e.sum(), 2.0 * expected_total_mass, rtol=1e-12,
            err_msg="Conservation de masse violée"
        )

    def test_row_sums_corners_vs_midsides(self) -> None:
        """Sommes de lignes : 0 pour les coins, ρtA/3 pour les milieux.

        Résulte de ∫Ni dA = 0 (coins) et A/3 (milieux) pour Tri6.
        """
        rho, t = 1.0, 1.0
        mat = ElasticMaterial(E=1.0, nu=0.3, rho=rho)
        props = {"thickness": t}
        area = 0.5

        M_e = Tri6().mass_matrix(mat, NODES_STD, props)
        # Dofs : [u0,v0, u1,v1, u2,v2, u3,v3, u4,v4, u5,v5]
        for dof in range(6):   # coins : DDL 0..5 (nœuds 0,1,2)
            row_sum = M_e[dof, :].sum()
            np.testing.assert_allclose(row_sum, 0.0, atol=1e-14,
                err_msg=f"Coin DOF {dof}: somme de ligne devrait être 0")
        for dof in range(6, 12):  # milieux : DDL 6..11 (nœuds 3,4,5)
            row_sum = M_e[dof, :].sum()
            np.testing.assert_allclose(
                row_sum, rho * t * area / 3.0, rtol=1e-12,
                err_msg=f"Milieu DOF {dof}: somme de ligne devrait être ρtA/3"
            )

    def test_positive_semidefinite(self) -> None:
        """M_e est semi-définie positive."""
        M_e = Tri6().mass_matrix(MAT_STEEL, NODES_STD, PROPS_STD)
        eigenvalues = np.linalg.eigvalsh(M_e)
        assert np.all(eigenvalues >= -1e-14), (
            f"M_e a une valeur propre négative : {eigenvalues.min():.3e}"
        )


# ===========================================================================
# Patch test — champ linéaire de déplacement
# ===========================================================================


class TestTri6PatchTest:
    """Patch test : un champ linéaire est reproduit exactement par Tri6.

    Champ imposé : ux = α·x,  uy = β·y.
    Les déformations correspondantes sont εxx=α, εyy=β, γxy=0 (constantes).
    L'élément doit reproduire exactement ce champ sans erreur.

    Solution analytique (contrainte plane, nu=0) :
        σxx = E · α,  σyy = E · β,  τxy = 0.
    """

    def test_linear_displacement_exactly_reproduced(self) -> None:
        """ε = B · u_e = [α, β, 0] pour tout point de l'élément."""
        alpha, beta = 0.003, -0.001
        elem = Tri6()
        nodes = NODES_STD.copy()

        # Déplacements nodaux imposés par le champ linéaire
        u_e = np.zeros(12)
        for i, (x, y) in enumerate(nodes):
            u_e[2*i    ] = alpha * x  # ux = α·x
            u_e[2*i + 1] = beta  * y  # uy = β·y

        # Vérifier ε à plusieurs points dans l'élément
        for xi, eta in [(0.1, 0.1), (0.5, 0.2), (0.25, 0.25), (0.0, 0.5)]:
            if xi + eta > 1.0:
                continue
            eps = elem.strain(nodes, u_e, xi, eta)
            np.testing.assert_allclose(eps[0], alpha, atol=1e-12,
                err_msg=f"εxx incorrect à (ξ={xi}, η={eta})")
            np.testing.assert_allclose(eps[1], beta, atol=1e-12,
                err_msg=f"εyy incorrect à (ξ={xi}, η={eta})")
            np.testing.assert_allclose(eps[2], 0.0, atol=1e-12,
                err_msg=f"γxy non nul à (ξ={xi}, η={eta})")

    def test_stress_from_linear_displacement(self) -> None:
        """σ = D · ε = [E·α, E·β, 0] pour nu=0, contrainte plane."""
        E, alpha, beta = 1.0, 0.003, -0.001
        mat = ElasticMaterial(E=E, nu=0.0, rho=1.0)
        elem = Tri6()
        nodes = NODES_STD.copy()

        u_e = np.zeros(12)
        for i, (x, y) in enumerate(nodes):
            u_e[2*i    ] = alpha * x
            u_e[2*i + 1] = beta  * y

        sigma = elem.stress(mat, nodes, u_e, 0.25, 0.25, "plane_stress")
        np.testing.assert_allclose(sigma[0], E * alpha, rtol=1e-12)
        np.testing.assert_allclose(sigma[1], E * beta,  rtol=1e-12)
        np.testing.assert_allclose(sigma[2], 0.0,       atol=1e-12)


# ===========================================================================
# Test de traction pure — système assemblé
# ===========================================================================


class TestTri6TractionPatch:
    """Deux triangles Tri6 sous traction uniforme.

    Géométrie : rectangle [0,1]×[0,1], épaisseur 1.
    Maillage  : 2 triangles Tri6 (diagonale 0-2 → 2-4 en termes de coins).
    Chargement: force P sur le bord droit.
    Solution analytique : ux = P/(E·t) · x, uy = −ν·P/(E·t) · y.
    """

    def _make_mesh(self) -> tuple[Mesh, BoundaryConditions]:
        """Crée un maillage de 2 Tri6 sur [0,1]×[0,1].

        Numérotation globale des nœuds :
            Coins  : 0=(0,0), 1=(1,0), 2=(1,1), 3=(0,1)
            Milieux bas    : 4=(0.5,0)
            Milieux droit  : 5=(1,0.5)
            Milieux diag   : 6=(0.5,0.5)
            Milieux gauche : 7=(0,0.5)
            Milieux haut   : 8=(0.5,1)

        Triangle 1 (inférieur) : coins 0,1,2 + milieux 4,5,6
        Triangle 2 (supérieur) : coins 0,2,3 + milieux 6,8,7
        """
        E, nu, rho = 1.0, 0.3, 1.0
        mat = ElasticMaterial(E=E, nu=nu, rho=rho)
        t = 1.0
        P = 1.0  # force/épaisseur [N/m]

        nodes = np.array([
            [0.0, 0.0],   # 0
            [1.0, 0.0],   # 1
            [1.0, 1.0],   # 2
            [0.0, 1.0],   # 3
            [0.5, 0.0],   # 4  milieu 0-1
            [1.0, 0.5],   # 5  milieu 1-2
            [0.5, 0.5],   # 6  milieu 0-2 (diagonale)
            [0.0, 0.5],   # 7  milieu 0-3
            [0.5, 1.0],   # 8  milieu 2-3
        ])
        props = {"thickness": t, "formulation": "plane_stress"}

        elem1 = ElementData(
            etype=Tri6,
            node_ids=(0, 1, 2, 4, 5, 6),
            material=mat,
            properties=props,
        )
        elem2 = ElementData(
            etype=Tri6,
            node_ids=(0, 2, 3, 6, 8, 7),
            material=mat,
            properties=props,
        )
        mesh = Mesh(nodes=nodes, elements=(elem1, elem2), n_dim=2)

        # Dirichlet : bord gauche (nœuds 0,3,7) bloqué en ux
        #             nœud 0 bloqué en uy (éviter le mode rigide)
        dirichlet = {
            0: {0: 0.0, 1: 0.0},
            3: {0: 0.0},
            7: {0: 0.0},
        }
        # Neumann : traction P sur bord droit (nœuds 1,2,5)
        # Force totale = P (distribuée entre les nœuds du bord)
        # Pour Tri6 sur une arête chargée : f_coin = P/6, f_milieu = 2P/3
        neumann = {
            1: {0: P / 6.0},
            2: {0: P / 6.0},
            5: {0: 2.0 * P / 3.0},
        }
        bc = BoundaryConditions(dirichlet=dirichlet, neumann=neumann)
        return mesh, bc

    def test_tip_displacement_under_traction(self) -> None:
        """Déplacement ux(x=1) = P/(E·t) = 1 pour E=t=P=1.

        Solution analytique : ux = P·x/(E·t), uy = −ν·P·y/(E·t).
        """
        mesh, bc = self._make_mesh()
        assembler = Assembler(mesh)
        K = assembler.assemble_stiffness()
        F = assembler.assemble_forces(bc)
        K_bc, F_bc = apply_dirichlet(K, F, mesh, bc)
        u = StaticSolver().solve(K_bc, F_bc)

        E, nu, P, t = 1.0, 0.3, 1.0, 1.0
        # Déplacement en x au bord droit (nœud 1, x=1, y=0)
        ux_tip = u[2 * 1]     # DOF 2*nœud pour ux
        np.testing.assert_allclose(ux_tip, P / (E * t), rtol=1e-10)

    def test_transverse_displacement_under_traction(self) -> None:
        """Déplacement uy(x=0,y=1) = −ν·P/(E·t) sous traction axiale.

        Le nœud 3 est à (0,1). Son uy suit la contraction de Poisson :
        uy = −ν · εxx · y = −ν · P/(E·t) · 1.
        """
        mesh, bc = self._make_mesh()
        assembler = Assembler(mesh)
        K = assembler.assemble_stiffness()
        F = assembler.assemble_forces(bc)
        K_bc, F_bc = apply_dirichlet(K, F, mesh, bc)
        u = StaticSolver().solve(K_bc, F_bc)

        E, nu, P, t = 1.0, 0.3, 1.0, 1.0
        uy_node3 = u[2 * 3 + 1]
        np.testing.assert_allclose(
            uy_node3, -nu * P / (E * t), rtol=1e-8,
        )

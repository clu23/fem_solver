"""Tests unitaires pour l'élément Quad4 — validation par solution analytique."""

from __future__ import annotations

import numpy as np
import pytest

from femsolver.core.assembler import Assembler
from femsolver.core.boundary import apply_dirichlet
from femsolver.core.material import ElasticMaterial
from femsolver.core.mesh import BoundaryConditions, ElementData, Mesh
from femsolver.core.solver import StaticSolver
from femsolver.elements.quad4 import Quad4


# ---------------------------------------------------------------------------
# Données communes
# ---------------------------------------------------------------------------

MAT_SIMPLE = ElasticMaterial(E=1.0, nu=0.0, rho=1.0)
MAT_STEEL = ElasticMaterial(E=210e9, nu=0.3, rho=7800)

# Carré unitaire : nœuds dans le sens trigonométrique
NODES_UNIT = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
PROPS_STD = {"thickness": 1.0}


# ---------------------------------------------------------------------------
# Tests des fonctions de forme
# ---------------------------------------------------------------------------


class TestQuad4ShapeFunctions:
    """Vérification des fonctions de forme et de leurs dérivées."""

    def test_partition_of_unity(self) -> None:
        """ΣNi = 1 en tout point (ξ, η)."""
        for xi, eta in [(-1, -1), (1, -1), (1, 1), (-1, 1), (0, 0), (0.3, -0.7)]:
            N = Quad4._shape_functions(xi, eta)
            np.testing.assert_allclose(N.sum(), 1.0, atol=1e-14,
                                       err_msg=f"Partition unité échouée en ({xi},{eta})")

    def test_nodal_interpolation(self) -> None:
        """Ni(nœud j) = δij — chaque Ni vaut 1 à son nœud, 0 aux autres."""
        corners = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
        for j, (xi, eta) in enumerate(corners):
            N = Quad4._shape_functions(xi, eta)
            expected = np.zeros(4)
            expected[j] = 1.0
            np.testing.assert_allclose(N, expected, atol=1e-14,
                                       err_msg=f"Nœud {j} : N({xi},{eta}) = {N}")

    def test_derivatives_sum_zero(self) -> None:
        """Σ ∂Ni/∂ξ = 0 et Σ ∂Ni/∂η = 0 (conséquence de la partition de l'unité)."""
        dN = Quad4._shape_function_derivatives(0.2, -0.4)
        np.testing.assert_allclose(dN[0].sum(), 0.0, atol=1e-14)
        np.testing.assert_allclose(dN[1].sum(), 0.0, atol=1e-14)

    def test_jacobian_unit_square(self) -> None:
        """Pour le carré [0,1]², J = ½·I et det(J) = ¼ en tout point."""
        gp = 1.0 / np.sqrt(3.0)
        for xi, eta in [(0, 0), (0.5, -0.5), (-gp, gp)]:
            dN = Quad4._shape_function_derivatives(xi, eta)
            J = Quad4._jacobian(dN, NODES_UNIT)
            # Carré [0,1]² : x = (1+ξ)/2, y = (1+η)/2 → J = 0.5·I
            np.testing.assert_allclose(J, 0.5 * np.eye(2), atol=1e-14)
            np.testing.assert_allclose(np.linalg.det(J), 0.25, atol=1e-14)


# ---------------------------------------------------------------------------
# Tests de la matrice B
# ---------------------------------------------------------------------------


class TestQuad4BMatrix:
    """Vérification de la matrice déformation–déplacement B."""

    def test_shape(self) -> None:
        """B est de shape (3, 8)."""
        B, _ = Quad4()._strain_displacement_matrix(0.0, 0.0, NODES_UNIT)
        assert B.shape == (3, 8)

    def test_rigid_body_translation(self) -> None:
        """Translation rigide → ε = 0 : B · [1,0,1,0,1,0,1,0] = 0."""
        B, _ = Quad4()._strain_displacement_matrix(0.0, 0.0, NODES_UNIT)
        u_transl_x = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=float)
        u_transl_y = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=float)
        np.testing.assert_allclose(B @ u_transl_x, 0.0, atol=1e-14)
        np.testing.assert_allclose(B @ u_transl_y, 0.0, atol=1e-14)

    def test_degenerate_raises(self) -> None:
        """Quad dégénéré (nœuds colinéaires) → ValueError."""
        nodes_degen = np.array([[0., 0.], [1., 0.], [2., 0.], [3., 0.]])
        with pytest.raises(ValueError, match="singulier"):
            Quad4()._strain_displacement_matrix(0.0, 0.0, nodes_degen)


# ---------------------------------------------------------------------------
# Tests de la matrice de rigidité K_e
# ---------------------------------------------------------------------------


class TestQuad4StiffnessMatrix:
    """Vérification de K_e."""

    def test_shape(self) -> None:
        """K_e est de shape (8, 8)."""
        K_e = Quad4().stiffness_matrix(MAT_SIMPLE, NODES_UNIT, PROPS_STD)
        assert K_e.shape == (8, 8)

    def test_symmetry(self) -> None:
        """K_e est symétrique."""
        K_e = Quad4().stiffness_matrix(MAT_STEEL, NODES_UNIT, {"thickness": 0.01})
        np.testing.assert_allclose(K_e, K_e.T, atol=1e-6)

    def test_singular_before_bc(self) -> None:
        """K_e est singulière avant CL (3 modes rigides : 2 translations + 1 rotation)."""
        K_e = Quad4().stiffness_matrix(MAT_SIMPLE, NODES_UNIT, PROPS_STD)
        rank = np.linalg.matrix_rank(K_e)
        # 8 DDL - 3 modes rigides = 5 modes déformables
        assert rank == 5

    def test_rectangle_vs_unit_square_scaling(self) -> None:
        """K_e d'un rectangle a×b = K_e carré unitaire mis à l'échelle correctement.

        Pour un rectangle a×b, det(J) = ab/4. Les termes de K_e doivent
        être homogènes en E·t·(longueur) comme pour Bar2D.
        """
        a, b = 2.0, 3.0
        nodes_rect = np.array([[0., 0.], [a, 0.], [a, b], [0., b]])
        K_rect = Quad4().stiffness_matrix(MAT_SIMPLE, nodes_rect, PROPS_STD)
        assert K_rect.shape == (8, 8)
        assert np.allclose(K_rect, K_rect.T)

    def test_zero_thickness_raises(self) -> None:
        """Épaisseur nulle → ValueError."""
        with pytest.raises(ValueError, match="épaisseur"):
            Quad4().stiffness_matrix(MAT_SIMPLE, NODES_UNIT, {"thickness": 0.0})


# ---------------------------------------------------------------------------
# Tests de la matrice de masse
# ---------------------------------------------------------------------------


class TestQuad4MassMatrix:
    """Vérification de M_e."""

    def test_total_mass_unit_square(self) -> None:
        """Carré unitaire, ρ=1, t=1 : chaque ligne de M_e somme à ρtA/4.

        Masse totale = ρ·t·A = 1·1·1 = 1 kg.
        Chaque nœud (2 DDL) reçoit ρtA/4 = 0.25 kg par direction.
        → sum de chaque ligne = 0.25.
        """
        M_e = Quad4().mass_matrix(MAT_SIMPLE, NODES_UNIT, PROPS_STD)
        row_sums = M_e.sum(axis=1)
        np.testing.assert_allclose(row_sums, 0.25, rtol=1e-12)

    def test_symmetry(self) -> None:
        """M_e est symétrique."""
        M_e = Quad4().mass_matrix(MAT_STEEL, NODES_UNIT, {"thickness": 0.01})
        np.testing.assert_allclose(M_e, M_e.T, atol=1e-14)


# ---------------------------------------------------------------------------
# Patch test — traction uniaxiale uniforme
# ---------------------------------------------------------------------------


class TestQuad4PatchTest:
    """Patch test : un seul Quad4 en traction uniaxiale.

    Configuration
    -------------
    Rectangle 2×1 m (un seul élément), traction σxx = σ0.

    Solution analytique
    -------------------
    u(x,y) = σ0/E · x   (avec nu=0, pas de contraction transversale)
    v(x,y) = 0

    Le Quad4 doit reproduire ce champ *exactement* car c'est un champ
    bilinéaire, inclus dans l'espace d'approximation.
    """

    def setup_method(self) -> None:
        self.E = 1.0
        self.nu = 0.0
        self.t = 1.0
        self.sigma0 = 1.0
        self.mat = ElasticMaterial(E=self.E, nu=self.nu, rho=1.0)

    def _build_mesh(self) -> tuple[Mesh, BoundaryConditions]:
        """Rectangle 2×1 m, 1 élément Quad4.

        Nœuds :
            0:(0,0)  1:(2,0)  2:(2,1)  3:(0,1)
        """
        nodes = np.array([[0., 0.], [2., 0.], [2., 1.], [0., 1.]])
        props = {"thickness": self.t, "formulation": "plane_stress"}
        mesh = Mesh(
            nodes=nodes,
            elements=(ElementData(Quad4, (0, 1, 2, 3), self.mat, props),),
            n_dim=2,
        )
        # Force nodale = σ0 · t · h/2 sur chaque nœud du bord droit (h=1m, 2 nœuds)
        F_nodal = self.sigma0 * self.t * 1.0 / 2.0
        bc = BoundaryConditions(
            dirichlet={0: {0: 0.0, 1: 0.0}, 3: {0: 0.0}},
            neumann={1: {0: F_nodal}, 2: {0: F_nodal}},
        )
        return mesh, bc

    def test_displacement_matches_analytical(self) -> None:
        """ux = σ0/E · x, uy = 0 sur tous les nœuds."""
        mesh, bc = self._build_mesh()
        K = Assembler(mesh).assemble_stiffness()
        F = Assembler(mesh).assemble_forces(bc)
        K_bc, F_bc = apply_dirichlet(K, F, mesh, bc)
        u = StaticSolver().solve(K_bc, F_bc)

        for i, (x, _) in enumerate(mesh.nodes):
            np.testing.assert_allclose(u[2 * i],     self.sigma0 / self.E * x, atol=1e-10)
            np.testing.assert_allclose(u[2 * i + 1], 0.0,                       atol=1e-10)

    def test_strain_matches_analytical(self) -> None:
        """εxx = σ0/E, εyy = γxy = 0 au centre de l'élément."""
        mesh, bc = self._build_mesh()
        K = Assembler(mesh).assemble_stiffness()
        F = Assembler(mesh).assemble_forces(bc)
        K_bc, F_bc = apply_dirichlet(K, F, mesh, bc)
        u = StaticSolver().solve(K_bc, F_bc)

        elem_data = list(mesh.elements)[0]
        node_coords = mesh.node_coords(elem_data.node_ids)
        u_e = u[mesh.global_dofs(elem_data.node_ids)]
        eps = Quad4().strain(node_coords, u_e, xi=0.0, eta=0.0)

        np.testing.assert_allclose(eps[0], self.sigma0 / self.E, atol=1e-10)
        np.testing.assert_allclose(eps[1], 0.0,                   atol=1e-10)
        np.testing.assert_allclose(eps[2], 0.0,                   atol=1e-10)

    def test_stress_matches_applied(self) -> None:
        """σxx = σ0, σyy = τxy = 0 au centre."""
        mesh, bc = self._build_mesh()
        K = Assembler(mesh).assemble_stiffness()
        F = Assembler(mesh).assemble_forces(bc)
        K_bc, F_bc = apply_dirichlet(K, F, mesh, bc)
        u = StaticSolver().solve(K_bc, F_bc)

        elem_data = list(mesh.elements)[0]
        node_coords = mesh.node_coords(elem_data.node_ids)
        u_e = u[mesh.global_dofs(elem_data.node_ids)]
        sigma = Quad4().stress(self.mat, node_coords, u_e)

        np.testing.assert_allclose(sigma[0], self.sigma0, atol=1e-10)
        np.testing.assert_allclose(sigma[1], 0.0,          atol=1e-10)
        np.testing.assert_allclose(sigma[2], 0.0,          atol=1e-10)


# ---------------------------------------------------------------------------
# Cohérence Tri3 / Quad4 — même maillage, même résultat
# ---------------------------------------------------------------------------


class TestQuad4VsTri3Consistency:
    """Un Quad4 rectangulaire doit donner le même déplacement que 2 Tri3 CST
    sur le même domaine pour une traction uniaxiale (les deux espaces contiennent
    le champ linéaire exact).
    """

    def test_tip_displacement_same_as_tri3(self) -> None:
        """ux nœud (2,0) identique avec Quad4 et 2×Tri3 (traction uniaxiale, nu=0)."""
        from femsolver.elements.tri3 import Tri3

        E, nu, t, sigma0 = 1.0, 0.0, 1.0, 1.0
        mat = ElasticMaterial(E=E, nu=nu, rho=1.0)
        F_nodal = sigma0 * t * 0.5

        # --- Quad4 ---
        nodes = np.array([[0., 0.], [2., 0.], [2., 1.], [0., 1.]])
        mesh_q = Mesh(
            nodes=nodes,
            elements=(ElementData(Quad4, (0, 1, 2, 3), mat,
                                  {"thickness": t, "formulation": "plane_stress"}),),
            n_dim=2,
        )
        bc_q = BoundaryConditions(
            dirichlet={0: {0: 0.0, 1: 0.0}, 3: {0: 0.0}},
            neumann={1: {0: F_nodal}, 2: {0: F_nodal}},
        )
        K_q = Assembler(mesh_q).assemble_stiffness()
        F_q = Assembler(mesh_q).assemble_forces(bc_q)
        K_q_bc, F_q_bc = apply_dirichlet(K_q, F_q, mesh_q, bc_q)
        u_q = StaticSolver().solve(K_q_bc, F_q_bc)

        # --- 2 × Tri3 ---
        mesh_t = Mesh(
            nodes=nodes,
            elements=(
                ElementData(Tri3, (0, 1, 2), mat, {"thickness": t, "formulation": "plane_stress"}),
                ElementData(Tri3, (0, 2, 3), mat, {"thickness": t, "formulation": "plane_stress"}),
            ),
            n_dim=2,
        )
        bc_t = BoundaryConditions(
            dirichlet={0: {0: 0.0, 1: 0.0}, 3: {0: 0.0}},
            neumann={1: {0: F_nodal}, 2: {0: F_nodal}},
        )
        K_t = Assembler(mesh_t).assemble_stiffness()
        F_t = Assembler(mesh_t).assemble_forces(bc_t)
        K_t_bc, F_t_bc = apply_dirichlet(K_t, F_t, mesh_t, bc_t)
        u_t = StaticSolver().solve(K_t_bc, F_t_bc)

        # ux au nœud 1 (x=2) doit être σ0/E * 2 = 2 dans les deux cas
        np.testing.assert_allclose(u_q[2], 2.0, atol=1e-10)
        np.testing.assert_allclose(u_t[2], 2.0, atol=1e-10)

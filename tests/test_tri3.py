"""Tests unitaires pour l'élément Tri3 (CST) — validation par solution analytique."""

from __future__ import annotations

import numpy as np
import pytest

from femsolver.core.assembler import Assembler
from femsolver.core.boundary import apply_dirichlet
from femsolver.core.material import ElasticMaterial
from femsolver.core.mesh import BoundaryConditions, ElementData, Mesh
from femsolver.core.solver import StaticSolver
from femsolver.elements.tri3 import Tri3


# ---------------------------------------------------------------------------
# Matériau de référence
# ---------------------------------------------------------------------------

# Matériau simplifié (E=1, nu=0) pour vérifications analytiques faciles
MAT_SIMPLE = ElasticMaterial(E=1.0, nu=0.0, rho=1.0)
MAT_STEEL = ElasticMaterial(E=210e9, nu=0.3, rho=7800)

# Triangle rectangle isocèle standard : nœuds (0,0), (1,0), (0,1)
NODES_STD = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
PROPS_STD = {"thickness": 1.0}


# ---------------------------------------------------------------------------
# Tests de la matrice B
# ---------------------------------------------------------------------------


class TestTri3BMatrix:
    """Vérification de la matrice déformation–déplacement B."""

    def test_shape(self) -> None:
        """B est de shape (3, 6)."""
        B, _ = Tri3()._strain_displacement_matrix(NODES_STD)
        assert B.shape == (3, 6)

    def test_area_standard_triangle(self) -> None:
        """Triangle (0,0)-(1,0)-(0,1) : aire = 0.5 m²."""
        _, area = Tri3()._strain_displacement_matrix(NODES_STD)
        np.testing.assert_allclose(area, 0.5, rtol=1e-12)

    def test_b_coefficients_standard(self) -> None:
        """Vérification analytique des coefficients b, c pour le triangle standard.

        b1 = y2-y3 = 0-1 = -1,  b2 = y3-y1 = 1-0 = 1,  b3 = y1-y2 = 0-0 = 0
        c1 = x3-x2 = 0-1 = -1,  c2 = x1-x3 = 0-0 = 0,  c3 = x2-x1 = 1-0 = 1

        B = 1/(2·0.5) · [[b1,0,b2,0,b3,0],[0,c1,0,c2,0,c3],[c1,b1,c2,b2,c3,b3]]
          = [[-1,0,1,0,0,0],[0,-1,0,0,0,1],[-1,-1,0,1,1,0]]
        """
        B, _ = Tri3()._strain_displacement_matrix(NODES_STD)
        B_expected = np.array(
            [
                [-1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0, 0.0, 1.0],
                [-1.0, -1.0, 0.0, 1.0, 1.0, 0.0],
            ]
        )
        np.testing.assert_allclose(B, B_expected, atol=1e-12)

    def test_clockwise_nodes_same_result(self) -> None:
        """L'inversion du sens de numérotation ne change pas B (correction automatique)."""
        nodes_ccw = NODES_STD.copy()
        nodes_cw = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]])  # sens horaire
        B_ccw, area_ccw = Tri3()._strain_displacement_matrix(nodes_ccw)
        B_cw, area_cw = Tri3()._strain_displacement_matrix(nodes_cw)
        # L'aire doit être positive dans les deux cas
        assert area_ccw > 0
        assert area_cw > 0
        np.testing.assert_allclose(area_ccw, area_cw, rtol=1e-12)

    def test_degenerate_raises(self) -> None:
        """Nœuds colinéaires → ValueError."""
        nodes_degen = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        with pytest.raises(ValueError, match="Aire nulle"):
            Tri3()._strain_displacement_matrix(nodes_degen)


# ---------------------------------------------------------------------------
# Tests de la matrice de rigidité K_e
# ---------------------------------------------------------------------------


class TestTri3StiffnessMatrix:
    """Vérification de K_e."""

    def test_shape(self) -> None:
        """K_e est de shape (6, 6)."""
        K_e = Tri3().stiffness_matrix(MAT_SIMPLE, NODES_STD, PROPS_STD)
        assert K_e.shape == (6, 6)

    def test_symmetry(self) -> None:
        """K_e est symétrique."""
        K_e = Tri3().stiffness_matrix(MAT_STEEL, NODES_STD, {"thickness": 0.01})
        np.testing.assert_allclose(K_e, K_e.T, atol=1e-6)

    def test_singular_before_bc(self) -> None:
        """K_e est singulière avant application des CL (3 modes rigides)."""
        K_e = Tri3().stiffness_matrix(MAT_SIMPLE, NODES_STD, PROPS_STD)
        rank = np.linalg.matrix_rank(K_e)
        # 6 DDL - 3 modes rigides = 3 modes déformables
        assert rank == 3

    def test_nu0_analytical(self) -> None:
        """Pour nu=0, E=1, t=1 sur triangle standard : K_e[0,0] vérifié analytiquement.

        Avec nu=0, D = [[1,0,0],[0,1,0],[0,0,0.5]].
        B[col 0] = [-1, 0, -1]  → Bᵀ D B · (0.5·1) terme (0,0) :
        K_e[0,0] = (b1²·D11 + c1²·D33) * A*t / (2A)²
                 = (1·1 + 1·0.5) * 0.5 * 1 / 1 = 0.75
        """
        K_e = Tri3().stiffness_matrix(MAT_SIMPLE, NODES_STD, PROPS_STD)
        # K_e = Bᵀ D B * A*t, avec A=0.5, t=1
        # B col 0 = [-1, 0, -1], D diag = [1, 1, 0.5]
        # K_e[0,0] = ((-1)²*1 + (-1)²*0.5) * 0.5 = 1.5 * 0.5 = 0.75
        np.testing.assert_allclose(K_e[0, 0], 0.75, rtol=1e-12)

    def test_zero_thickness_raises(self) -> None:
        """Épaisseur nulle → ValueError."""
        with pytest.raises(ValueError, match="épaisseur"):
            Tri3().stiffness_matrix(MAT_SIMPLE, NODES_STD, {"thickness": 0.0})


# ---------------------------------------------------------------------------
# Tests de la matrice de masse
# ---------------------------------------------------------------------------


class TestTri3MassMatrix:
    """Vérification de M_e."""

    def test_total_mass(self) -> None:
        """Chaque ligne de M_e somme à ρtA/3 (masse nodale consistante).

        Masse totale = ρ · t · A = 7800 · 0.01 · 1.0 = 78 kg.
        La partition de l'unité (ΣNi = 1) garantit ∫ Ni dA = A/3,
        donc chaque ligne de M_e somme à ρt·(A/3) = masse_totale / 3.
        sum(M_e) = 6 lignes × ρtA/3 = 2·ρtA  (2 directions par nœud).
        """
        mat = ElasticMaterial(E=210e9, nu=0.3, rho=7800)
        t = 0.01
        nodes = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 1.0]])
        area = 0.5 * 2.0 * 1.0
        M_e = Tri3().mass_matrix(mat, nodes, {"thickness": t})
        mass_total = mat.rho * t * area
        row_sums = M_e.sum(axis=1)
        np.testing.assert_allclose(row_sums, mass_total / 3.0, rtol=1e-12)

    def test_symmetry(self) -> None:
        """M_e est symétrique."""
        M_e = Tri3().mass_matrix(MAT_STEEL, NODES_STD, {"thickness": 0.01})
        np.testing.assert_allclose(M_e, M_e.T, atol=1e-14)


# ---------------------------------------------------------------------------
# Patch test — traction uniaxiale uniforme
# ---------------------------------------------------------------------------


class TestTri3PatchTest:
    """Patch test : un champ de déplacement linéaire doit être reproduit exactement.

    Configuration
    -------------
    Rectangle 2×1 m découpé en 2 triangles, traction σxx = σ0 uniforme.

    Solution analytique
    -------------------
    u(x,y) = σ0/E · x   (extension en x)
    v(x,y) = -nu·σ0/E · y  (contraction en y, Poisson)

    Force nodale équivalente sur le bord droit (x=2) :
    F = σ0 · t · h_elem  (répartie sur les nœuds du bord)

    Le patch test est validé si le déplacement calculé correspond à la
    solution analytique à la précision machine (rtol=1e-10).
    """

    def setup_method(self) -> None:
        self.E = 1.0
        self.nu = 0.0   # nu=0 simplifie : v=0 partout
        self.t = 1.0
        self.sigma0 = 1.0
        self.mat = ElasticMaterial(E=self.E, nu=self.nu, rho=1.0)

    def _build_mesh(self) -> tuple[Mesh, BoundaryConditions]:
        """Rectangle 2×1 m, 4 nœuds, 2 triangles.

        Nœuds :
            0:(0,0)  1:(2,0)  2:(2,1)  3:(0,1)

        Triangles :
            T1 : 0-1-2  (bas-droit)
            T2 : 0-2-3  (haut-gauche)
        """
        nodes = np.array([
            [0.0, 0.0],  # 0
            [2.0, 0.0],  # 1
            [2.0, 1.0],  # 2
            [0.0, 1.0],  # 3
        ])
        props = {"thickness": self.t, "formulation": "plane_stress"}
        elems = (
            ElementData(Tri3, (0, 1, 2), self.mat, props),
            ElementData(Tri3, (0, 2, 3), self.mat, props),
        )
        mesh = Mesh(nodes=nodes, elements=elems, n_dim=2)

        # Conditions aux limites
        # Dirichlet : bord gauche (x=0) bloqué en x ; nœud 0 bloqué en y
        # Neumann : traction σ0 sur bord droit (x=2), répartie sur nœuds 1 et 2
        # Force nodale = σ0 * t * h/2  (h=1m, 2 nœuds sur le bord)
        F_nodal = self.sigma0 * self.t * 1.0 / 2.0  # 0.5 N

        bc = BoundaryConditions(
            dirichlet={0: {0: 0.0, 1: 0.0}, 3: {0: 0.0}},
            neumann={1: {0: F_nodal}, 2: {0: F_nodal}},
        )
        return mesh, bc

    def test_displacement_matches_analytical(self) -> None:
        """u_x = σ0/E · x sur tous les nœuds (nu=0 → u_y = 0)."""
        mesh, bc = self._build_mesh()
        assembler = Assembler(mesh)
        K = assembler.assemble_stiffness()
        F = assembler.assemble_forces(bc)
        K_bc, F_bc = apply_dirichlet(K, F, mesh, bc)
        u = StaticSolver().solve(K_bc, F_bc)

        # Solution analytique : ux = σ0/E * x, uy = 0
        for i, (x, _) in enumerate(mesh.nodes):
            ux_analytical = self.sigma0 / self.E * x
            np.testing.assert_allclose(
                u[2 * i], ux_analytical, atol=1e-10,
                err_msg=f"ux nœud {i} (x={x})"
            )
            np.testing.assert_allclose(
                u[2 * i + 1], 0.0, atol=1e-10,
                err_msg=f"uy nœud {i}"
            )

    def test_strain_constant_in_element(self) -> None:
        """εxx = σ0/E dans chaque élément (CST → déformations constantes)."""
        mesh, bc = self._build_mesh()
        assembler = Assembler(mesh)
        K = assembler.assemble_stiffness()
        F = assembler.assemble_forces(bc)
        K_bc, F_bc = apply_dirichlet(K, F, mesh, bc)
        u = StaticSolver().solve(K_bc, F_bc)

        tri = Tri3()
        for elem_data in mesh.elements:
            node_coords = mesh.node_coords(elem_data.node_ids)
            dofs = mesh.global_dofs(elem_data.node_ids)
            u_e = u[dofs]
            eps = tri.strain(node_coords, u_e)
            np.testing.assert_allclose(
                eps[0], self.sigma0 / self.E, atol=1e-10,
                err_msg="εxx doit valoir σ0/E"
            )
            np.testing.assert_allclose(eps[1], 0.0, atol=1e-10)  # εyy = 0
            np.testing.assert_allclose(eps[2], 0.0, atol=1e-10)  # γxy = 0

    def test_stress_matches_applied(self) -> None:
        """σxx = σ0, σyy = τxy = 0 dans chaque élément."""
        mesh, bc = self._build_mesh()
        assembler = Assembler(mesh)
        K = assembler.assemble_stiffness()
        F = assembler.assemble_forces(bc)
        K_bc, F_bc = apply_dirichlet(K, F, mesh, bc)
        u = StaticSolver().solve(K_bc, F_bc)

        tri = Tri3()
        for elem_data in mesh.elements:
            node_coords = mesh.node_coords(elem_data.node_ids)
            dofs = mesh.global_dofs(elem_data.node_ids)
            u_e = u[dofs]
            sigma = tri.stress(self.mat, node_coords, u_e, "plane_stress")
            np.testing.assert_allclose(sigma[0], self.sigma0, atol=1e-10)
            np.testing.assert_allclose(sigma[1], 0.0, atol=1e-10)
            np.testing.assert_allclose(sigma[2], 0.0, atol=1e-10)

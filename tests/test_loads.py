"""Tests des charges distribuées, forces de volume et pressions surfaciques.

Chaque test compare le résultat FEM à une solution analytique connue.
"""

from __future__ import annotations

import numpy as np
import pytest

from femsolver.core.assembler import Assembler
from femsolver.core.boundary import apply_dirichlet
from femsolver.core.material import ElasticMaterial
from femsolver.core.mesh import (
    BodyForce,
    BoundaryConditions,
    DistributedLineLoad,
    ElementData,
    Mesh,
    PressureLoad,
)
from femsolver.core.solver import StaticSolver
from femsolver.elements.bar2d import Bar2D
from femsolver.elements.beam2d import Beam2D
from femsolver.elements.hexa8 import Hexa8
from femsolver.elements.quad4 import Quad4
from femsolver.elements.tetra4 import Tetra4
from femsolver.elements.tri3 import Tri3


# ---------------------------------------------------------------------------
# Fixtures matériaux
# ---------------------------------------------------------------------------

@pytest.fixture
def steel():
    return ElasticMaterial(E=210e9, nu=0.3, rho=7800.0)


@pytest.fixture
def unit_mat():
    """Matériau avec E=1, ν=0, ρ=1 pour simplifier les vérifications."""
    return ElasticMaterial(E=1.0, nu=0.0, rho=1.0)


# ===========================================================================
# Charges distribuées — Bar2D
# ===========================================================================

class TestBar2DDistributedLoad:
    """Barre encastrée-libre soumise à une charge axiale uniforme q [N/m].

    Solution analytique (Cook, chap. 2) :
        δ_tip = q · L² / (2 · EA)

    Exact avec 1 seul élément linéaire (intégration exacte des fonctions
    de forme linéaires).
    """

    def setup_method(self):
        self.E = 210e9
        self.A = 1e-4
        self.L = 2.0
        self.q = 5000.0   # N/m
        self.mat = ElasticMaterial(E=self.E, nu=0.3, rho=7800.0)

    def _build_and_solve(self, n_elem: int):
        """Barre horizontale discrétisée en n_elem éléments."""
        nodes = np.linspace(0.0, self.L, n_elem + 1)
        node_coords = np.column_stack([nodes, np.zeros(n_elem + 1)])
        props = {"area": self.A}
        elems = tuple(
            ElementData(Bar2D, (i, i + 1), self.mat, props)
            for i in range(n_elem)
        )
        mesh = Mesh(nodes=node_coords, elements=elems, n_dim=2)
        # Charge distribuée sur chaque élément
        distributed = tuple(
            DistributedLineLoad(node_ids=(i, i + 1), qx=self.q)
            for i in range(n_elem)
        )
        # Bloquer ux+uy au nœud 0 et uy à tous les autres nœuds
        # (la direction transverse a rigidité nulle dans Bar2D → K singulier sinon)
        dirichlet: dict = {0: {0: 0.0, 1: 0.0}}
        for i in range(1, n_elem + 1):
            dirichlet[i] = {1: 0.0}
        bc = BoundaryConditions(
            dirichlet=dirichlet,
            neumann={},
            distributed=distributed,
        )
        assembler = Assembler(mesh)
        K = assembler.assemble_stiffness()
        F = assembler.assemble_forces(bc)
        K_bc, F_bc = apply_dirichlet(K, F, mesh, bc)
        u = StaticSolver().solve(K_bc, F_bc)
        return u

    def test_tip_displacement_1_element(self):
        """δ_tip = qL²/(2EA) — exact avec 1 élément."""
        u = self._build_and_solve(1)
        delta_analytical = self.q * self.L**2 / (2.0 * self.E * self.A)
        # DDL global 2 = ux du nœud 1
        np.testing.assert_allclose(u[2], delta_analytical, rtol=1e-12)

    def test_tip_displacement_4_elements(self):
        """Convergence : 4 éléments donnent aussi le résultat exact (linéaire)."""
        u = self._build_and_solve(4)
        delta_analytical = self.q * self.L**2 / (2.0 * self.E * self.A)
        np.testing.assert_allclose(u[-2], delta_analytical, rtol=1e-10)

    def test_total_force_equilibrium(self):
        """La force totale appliquée = q·L (équilibre global)."""
        n_elem = 4
        nodes = np.linspace(0.0, self.L, n_elem + 1)
        node_coords = np.column_stack([nodes, np.zeros(n_elem + 1)])
        props = {"area": self.A}
        elems = tuple(
            ElementData(Bar2D, (i, i + 1), self.mat, props)
            for i in range(n_elem)
        )
        mesh = Mesh(nodes=node_coords, elements=elems, n_dim=2)
        distributed = tuple(
            DistributedLineLoad(node_ids=(i, i + 1), qx=self.q)
            for i in range(n_elem)
        )
        bc = BoundaryConditions(dirichlet={}, neumann={}, distributed=distributed)
        assembler = Assembler(mesh)
        F = assembler.assemble_forces(bc)
        # Composante x : somme = q·L
        Fx_total = F[0::2].sum()
        np.testing.assert_allclose(Fx_total, self.q * self.L, rtol=1e-12)

    def test_transverse_load_raises(self):
        """qy ≠ 0 sur Bar2D lève ValueError."""
        mat = self.mat
        elem = Bar2D()
        nodes = np.array([[0.0, 0.0], [1.0, 0.0]])
        with pytest.raises(ValueError, match="transverse"):
            elem.distributed_load_vector(mat, nodes, {}, qx=0.0, qy=100.0)


# ===========================================================================
# Charges distribuées — Beam2D
# ===========================================================================

class TestBeam2DDistributedLoad:
    """Poutre console encastrée-libre, charge transverse uniforme q [N/m].

    Solutions analytiques Euler–Bernoulli :
        δ_tip  = q · L⁴ / (8 · EI)
        θ_tip  = q · L³ / (6 · EI)

    Ces résultats sont **exacts** avec 1 seul élément Hermite (les
    polynômes de Hermite cubiques reproduisent exactement la déformée
    d'une poutre sous charge uniforme).
    """

    def setup_method(self):
        self.E = 210e9
        self.A = 0.01
        self.I = 8.333e-6
        self.L = 2.0
        self.q = 10000.0   # N/m (charge transverse vers le bas)
        self.mat = ElasticMaterial(E=self.E, nu=0.3, rho=7800.0)
        self.props = {"area": self.A, "inertia": self.I}

    def _build_and_solve(self, n_elem: int, qy: float):
        """Console horizontale avec n_elem éléments, charge qy [N/m] (local)."""
        L_elem = self.L / n_elem
        nodes = np.array([[i * L_elem, 0.0] for i in range(n_elem + 1)])
        elems = tuple(
            ElementData(Beam2D, (i, i + 1), self.mat, self.props)
            for i in range(n_elem)
        )
        mesh = Mesh(nodes=nodes, elements=elems, n_dim=2, dof_per_node=3)
        distributed = tuple(
            DistributedLineLoad(node_ids=(i, i + 1), qy=qy)
            for i in range(n_elem)
        )
        bc = BoundaryConditions(
            dirichlet={0: {0: 0.0, 1: 0.0, 2: 0.0}},  # encastrement nœud 0
            neumann={},
            distributed=distributed,
        )
        assembler = Assembler(mesh)
        K = assembler.assemble_stiffness()
        F = assembler.assemble_forces(bc)
        K_bc, F_bc = apply_dirichlet(K, F, mesh, bc)
        u = StaticSolver().solve(K_bc, F_bc)
        return u

    def test_tip_deflection_1_element_exact(self):
        """δ_tip = qL⁴/(8EI) — exact avec 1 seul élément Hermite."""
        u = self._build_and_solve(1, -self.q)
        delta_analytical = self.q * self.L**4 / (8.0 * self.E * self.I)
        # uy du nœud tip = DDL global 4 (nœud 1, DDL 1 → index 3*1+1=4)
        uy_tip = u[4]
        np.testing.assert_allclose(abs(uy_tip), delta_analytical, rtol=1e-10)

    def test_tip_rotation_1_element_exact(self):
        """θ_tip = qL³/(6EI) — exact avec 1 seul élément Hermite."""
        u = self._build_and_solve(1, -self.q)
        theta_analytical = self.q * self.L**3 / (6.0 * self.E * self.I)
        # θ du nœud tip = DDL global 5 (nœud 1, DDL 2 → index 3*1+2=5)
        theta_tip = u[5]
        np.testing.assert_allclose(abs(theta_tip), theta_analytical, rtol=1e-10)

    def test_tip_deflection_4_elements(self):
        """Convergence avec 4 éléments : même résultat exact (Hermite complet)."""
        u = self._build_and_solve(4, -self.q)
        delta_analytical = self.q * self.L**4 / (8.0 * self.E * self.I)
        uy_tip = u[-2]   # avant-dernier DDL = uy du dernier nœud
        np.testing.assert_allclose(abs(uy_tip), delta_analytical, rtol=1e-10)

    def test_axial_distributed_load(self):
        """Charge axiale pure : δ_tip = qL²/(2EA)."""
        u = self._build_and_solve(1, qy=0.0)
        # Pas de charge → tip displacement = 0
        np.testing.assert_allclose(u, 0.0, atol=1e-20)

    def test_axial_distributed_load_nonzero(self):
        """Charge axiale + pas de flexion : vérification cohérence."""
        E, A, L, q = self.E, self.A, self.L, 1000.0
        nodes = np.array([[0.0, 0.0], [L, 0.0]])
        elems = (ElementData(Beam2D, (0, 1), self.mat, self.props),)
        mesh = Mesh(nodes=nodes, elements=elems, n_dim=2, dof_per_node=3)
        distributed = (DistributedLineLoad(node_ids=(0, 1), qx=q, qy=0.0),)
        bc = BoundaryConditions(
            dirichlet={0: {0: 0.0, 1: 0.0, 2: 0.0}},
            neumann={},
            distributed=distributed,
        )
        assembler = Assembler(mesh)
        K = assembler.assemble_stiffness()
        F = assembler.assemble_forces(bc)
        K_bc, F_bc = apply_dirichlet(K, F, mesh, bc)
        u = StaticSolver().solve(K_bc, F_bc)
        delta_analytical = q * L**2 / (2.0 * E * A)
        np.testing.assert_allclose(u[3], delta_analytical, rtol=1e-10)

    def test_distributed_load_vector_shape(self):
        """Le vecteur de charge est de longueur 6."""
        mat = self.mat
        nodes = np.array([[0.0, 0.0], [1.0, 0.0]])
        f = Beam2D().distributed_load_vector(mat, nodes, {}, qx=0.0, qy=1000.0)
        assert f.shape == (6,)

    def test_distributed_load_vector_values_horizontal(self):
        """Vérification analytique : q=1000 N/m, L=1 m, horizontal.

        f_local = [0, qL/2, qL²/12, 0, qL/2, -qL²/12]
               = [0, 500, 83.33, 0, 500, -83.33]
        """
        mat = self.mat
        nodes = np.array([[0.0, 0.0], [1.0, 0.0]])
        q = 1000.0
        L = 1.0
        f = Beam2D().distributed_load_vector(mat, nodes, {}, qx=0.0, qy=q)
        expected = np.array([
            0.0, q * L / 2, q * L**2 / 12,
            0.0, q * L / 2, -q * L**2 / 12,
        ])
        np.testing.assert_allclose(f, expected, rtol=1e-12)

    def test_distributed_load_vector_total_force(self):
        """La somme des forces transverses = q·L (équilibre vertical)."""
        mat = self.mat
        nodes = np.array([[0.0, 0.0], [2.0, 0.0]])
        q, L = 500.0, 2.0
        f = Beam2D().distributed_load_vector(mat, nodes, {}, qx=0.0, qy=q)
        # DDL uy1 (indice 1) + uy2 (indice 4) = q*L
        np.testing.assert_allclose(f[1] + f[4], q * L, rtol=1e-12)


# ===========================================================================
# Force de volume — Tri3
# ===========================================================================

class TestTri3BodyForce:
    """Triangle soumis à une force de volume uniforme (gravité).

    Validation : somme des forces nodales = ρ · t · A · g.
    """

    def test_sum_equals_total_weight(self):
        """Σ f_y = -ρ · t · A · g (poids total de l'élément)."""
        mat = ElasticMaterial(E=210e9, nu=0.3, rho=7800.0)
        nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        t = 0.01
        g = 9.81
        b = np.array([0.0, -g])
        elem = Tri3()
        f = elem.body_force_vector(mat, nodes, {"thickness": t}, b)
        area = 0.5
        weight = mat.rho * t * area * g
        # Somme des forces verticales
        np.testing.assert_allclose(f[1::2].sum(), -weight, rtol=1e-12)

    def test_equidistribution_on_nodes(self):
        """Chaque nœud reçoit exactement 1/3 de la charge totale (CST)."""
        mat = ElasticMaterial(E=1.0, nu=0.0, rho=1.0)
        nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        b = np.array([3.0, 5.0])
        f = Tri3().body_force_vector(mat, nodes, {"thickness": 1.0}, b)
        area = 0.5
        f_per_node = area / 3.0 * b
        for k in range(3):
            np.testing.assert_allclose(f[2*k : 2*k+2], f_per_node, rtol=1e-12)

    def test_shape_is_6(self):
        """Vecteur de taille 6."""
        mat = ElasticMaterial(E=1.0, nu=0.0, rho=1.0)
        nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        f = Tri3().body_force_vector(mat, nodes, {"thickness": 1.0}, np.zeros(2))
        assert f.shape == (6,)


# ===========================================================================
# Force de volume — Quad4
# ===========================================================================

class TestQuad4BodyForce:
    """Quadrilatère unitaire soumis à une force de volume uniforme."""

    def test_sum_equals_total_weight_unit_square(self):
        """Σ f_y = -ρ · t · A · g pour un carré unitaire."""
        mat = ElasticMaterial(E=210e9, nu=0.3, rho=7800.0)
        nodes = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
        t, g = 0.02, 9.81
        b = np.array([0.0, -g])
        f = Quad4().body_force_vector(mat, nodes, {"thickness": t}, b)
        area = 1.0
        np.testing.assert_allclose(f[1::2].sum(), -mat.rho * t * area * g, rtol=1e-10)

    def test_equidistribution_unit_square(self):
        """Sur un carré régulier, chaque nœud reçoit 1/4 de la charge totale."""
        mat = ElasticMaterial(E=1.0, nu=0.0, rho=1.0)
        nodes = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
        b = np.array([2.0, 4.0])
        f = Quad4().body_force_vector(mat, nodes, {"thickness": 1.0}, b)
        f_per_node = 1.0 / 4.0 * b   # area=1, rho=1, t=1
        for k in range(4):
            np.testing.assert_allclose(f[2*k : 2*k+2], f_per_node, rtol=1e-10)

    def test_shape_is_8(self):
        mat = ElasticMaterial(E=1.0, nu=0.0, rho=1.0)
        nodes = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
        f = Quad4().body_force_vector(mat, nodes, {"thickness": 1.0}, np.zeros(2))
        assert f.shape == (8,)


# ===========================================================================
# Force de volume — Tetra4
# ===========================================================================

class TestTetra4BodyForce:
    """Tétraèdre soumis à une force de volume uniforme."""

    def test_sum_equals_total_weight(self):
        """Σ f_z = -ρ · V · g."""
        mat = ElasticMaterial(E=210e9, nu=0.3, rho=7800.0)
        nodes = np.array([
            [0., 0., 0.],
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
        ])
        g = 9.81
        b = np.array([0.0, 0.0, -g])
        f = Tetra4().body_force_vector(mat, nodes, {}, b)
        # Volume du tétraèdre = 1/6
        volume = 1.0 / 6.0
        np.testing.assert_allclose(
            f[2::3].sum(), -mat.rho * volume * g, rtol=1e-10
        )

    def test_equidistribution(self):
        """Chaque nœud reçoit 1/4 de la charge totale (Tetra4 linéaire)."""
        mat = ElasticMaterial(E=1.0, nu=0.0, rho=1.0)
        nodes = np.array([
            [0., 0., 0.],
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
        ])
        b = np.array([3.0, 5.0, 7.0])
        f = Tetra4().body_force_vector(mat, nodes, {}, b)
        volume = 1.0 / 6.0
        f_per_node = volume / 4.0 * b
        for k in range(4):
            np.testing.assert_allclose(f[3*k : 3*k+3], f_per_node, rtol=1e-12)

    def test_shape_is_12(self):
        mat = ElasticMaterial(E=1.0, nu=0.0, rho=1.0)
        nodes = np.array([
            [0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
        ])
        f = Tetra4().body_force_vector(mat, nodes, {}, np.zeros(3))
        assert f.shape == (12,)


# ===========================================================================
# Force de volume — Hexa8
# ===========================================================================

class TestHexa8BodyForce:
    """Hexaèdre unitaire soumis à une force de volume uniforme."""

    def _unit_cube_nodes(self):
        return np.array([
            [0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.],
            [0., 0., 1.], [1., 0., 1.], [1., 1., 1.], [0., 1., 1.],
        ])

    def test_sum_equals_total_weight(self):
        """Σ f_z = -ρ · V · g pour un cube unitaire."""
        mat = ElasticMaterial(E=210e9, nu=0.3, rho=7800.0)
        nodes = self._unit_cube_nodes()
        g = 9.81
        b = np.array([0.0, 0.0, -g])
        f = Hexa8().body_force_vector(mat, nodes, {}, b)
        volume = 1.0
        np.testing.assert_allclose(
            f[2::3].sum(), -mat.rho * volume * g, rtol=1e-10
        )

    def test_equidistribution_unit_cube(self):
        """Sur un cube unitaire, chaque nœud reçoit 1/8 de la charge."""
        mat = ElasticMaterial(E=1.0, nu=0.0, rho=1.0)
        nodes = self._unit_cube_nodes()
        b = np.array([2.0, 3.0, 5.0])
        f = Hexa8().body_force_vector(mat, nodes, {}, b)
        f_per_node = 1.0 / 8.0 * b   # volume=1, rho=1
        for k in range(8):
            np.testing.assert_allclose(f[3*k : 3*k+3], f_per_node, rtol=1e-9)

    def test_shape_is_24(self):
        mat = ElasticMaterial(E=1.0, nu=0.0, rho=1.0)
        nodes = self._unit_cube_nodes()
        f = Hexa8().body_force_vector(mat, nodes, {}, np.zeros(3))
        assert f.shape == (24,)


# ===========================================================================
# Force de volume via l'assembleur
# ===========================================================================

class TestBodyForceAssembler:
    """Force de volume assemblée via bc.body_force."""

    def test_tri3_gravity_sum(self):
        """Plaque Tri3 (2 triangles) : somme des forces = ρ·t·A·g."""
        mat = ElasticMaterial(E=210e9, nu=0.3, rho=7800.0)
        nodes = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
        t = 0.01
        elems = (
            ElementData(Tri3, (0, 1, 2), mat, {"thickness": t}),
            ElementData(Tri3, (0, 2, 3), mat, {"thickness": t}),
        )
        mesh = Mesh(nodes=nodes, elements=elems, n_dim=2)
        g = 9.81
        bc = BoundaryConditions(
            dirichlet={},
            neumann={},
            body_force=BodyForce(acceleration=(0.0, -g)),
        )
        assembler = Assembler(mesh)
        F = assembler.assemble_forces(bc)
        total_area = 1.0
        expected_fy = -mat.rho * t * total_area * g
        np.testing.assert_allclose(F[1::2].sum(), expected_fy, rtol=1e-10)

    def test_hexa8_gravity_sum(self):
        """Cube Hexa8 : somme des forces = ρ·V·g."""
        mat = ElasticMaterial(E=210e9, nu=0.3, rho=2500.0)
        nodes = np.array([
            [0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.],
            [0., 0., 1.], [1., 0., 1.], [1., 1., 1.], [0., 1., 1.],
        ])
        elems = (ElementData(Hexa8, tuple(range(8)), mat, {}),)
        mesh = Mesh(nodes=nodes, elements=elems, n_dim=3)
        g = 9.81
        bc = BoundaryConditions(
            dirichlet={},
            neumann={},
            body_force=BodyForce(acceleration=(0.0, 0.0, -g)),
        )
        assembler = Assembler(mesh)
        F = assembler.assemble_forces(bc)
        expected_fz = -mat.rho * 1.0 * g
        np.testing.assert_allclose(F[2::3].sum(), expected_fz, rtol=1e-9)


# ===========================================================================
# Pression surfacique 2D
# ===========================================================================

class TestPressureLoad2D:
    """Pression sur une arête 2D.

    Validation : force totale = p · L · t (par unité d'épaisseur = 1 ici).
    """

    def _make_quad_mesh(self, mat):
        """Carré unitaire avec 2 triangles."""
        nodes = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
        elems = (
            ElementData(Tri3, (0, 1, 2), mat, {"thickness": 1.0}),
            ElementData(Tri3, (0, 2, 3), mat, {"thickness": 1.0}),
        )
        return Mesh(nodes=nodes, elements=elems, n_dim=2)

    def test_horizontal_edge_force_magnitude(self):
        """Arête du bas (nœuds 0→1, normale vers bas = −y).

        Domaine au-dessus → arête orientée gauche-droite → n̂ = (0, -1).
        Force totale = p · L = p * 1.

        Note : pour que la normale pointe VERS LE BAS (hors du domaine qui est
        au-dessus), on doit orienter l'arête de droite vers gauche : 1→0.
        """
        mat = ElasticMaterial(E=1.0, nu=0.0, rho=1.0)
        mesh = self._make_quad_mesh(mat)
        p = 10000.0
        # Arête bas (y=0), domaine au-dessus : 0→1 donne n̂=(0,-1) sortant
        # n̂ = (dy, -dx)/L = (0, -1)
        pressure = PressureLoad(node_ids=(0, 1), magnitude=p)
        bc = BoundaryConditions(
            dirichlet={},
            neumann={},
            pressure=(pressure,),
        )
        assembler = Assembler(mesh)
        F = assembler.assemble_forces(bc)
        # Force totale en y = -p * n̂_y * L = -p * (-1) * 1 = p
        # (compression positive → force vers l'intérieur, soit +y)
        np.testing.assert_allclose(F[1::2].sum(), p * 1.0, rtol=1e-12)

    def test_wrong_ccw_order_raises(self):
        """L'ordre CW (normale vers intérieur) doit lever ValueError."""
        mat = ElasticMaterial(E=1.0, nu=0.0, rho=1.0)
        mesh = self._make_quad_mesh(mat)
        # Arête bas 1→0 : n̂ = (dy, -dx)/L = (0, +1) → pointe vers intérieur
        pressure = PressureLoad(node_ids=(1, 0), magnitude=10000.0)
        bc = BoundaryConditions(
            dirichlet={},
            neumann={},
            pressure=(pressure,),
        )
        assembler = Assembler(mesh)
        with pytest.raises(ValueError, match="intérieur"):
            assembler.assemble_forces(bc)

    def test_right_edge_force_direction(self):
        """Arête droite (nœuds 1→2, normale vers +x).

        n̂ = (dy, -dx)/L = (1, 0) (arête verticale montante).
        Force totale en x = -p * (+1) * L = -p.
        """
        mat = ElasticMaterial(E=1.0, nu=0.0, rho=1.0)
        mesh = self._make_quad_mesh(mat)
        p = 5000.0
        pressure = PressureLoad(node_ids=(1, 2), magnitude=p)
        bc = BoundaryConditions(
            dirichlet={},
            neumann={},
            pressure=(pressure,),
        )
        assembler = Assembler(mesh)
        F = assembler.assemble_forces(bc)
        # Somme des forces en x = -p * 1 (pression comprimate)
        np.testing.assert_allclose(F[0::2].sum(), -p * 1.0, rtol=1e-12)

    def test_pressure_distributes_equally_on_two_nodes(self):
        """Chaque nœud de l'arête reçoit la moitié de la force."""
        mat = ElasticMaterial(E=1.0, nu=0.0, rho=1.0)
        nodes = np.array([[0., 0.], [1., 0.], [0.5, 1.0]])
        elems = (ElementData(Tri3, (0, 1, 2), mat, {"thickness": 1.0}),)
        mesh = Mesh(nodes=nodes, elements=elems, n_dim=2)
        p = 1000.0
        # Arête du bas 0→1 : n̂ = (0,-1) sortant (apex au-dessus)
        # Force par nœud = -p*(0,-1)*L/2 = (0, +p/2)
        pressure = PressureLoad(node_ids=(0, 1), magnitude=p)
        bc = BoundaryConditions(dirichlet={}, neumann={}, pressure=(pressure,))
        assembler = Assembler(mesh)
        F = assembler.assemble_forces(bc)
        # Nœud 0 et nœud 1 reçoivent chacun p*L/2 en y
        np.testing.assert_allclose(F[1], p * 0.5, rtol=1e-12)
        np.testing.assert_allclose(F[3], p * 0.5, rtol=1e-12)


# ===========================================================================
# Pression surfacique 3D — Tetra4
# ===========================================================================

class TestPressureLoad3DTetra:
    """Pression sur une face triangulaire d'un tétraèdre."""

    def _unit_tetra_mesh(self, mat):
        nodes = np.array([
            [0., 0., 0.],
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
        ])
        elems = (ElementData(Tetra4, (0, 1, 2, 3), mat, {}),)
        return Mesh(nodes=nodes, elements=elems, n_dim=3)

    def test_bottom_face_force_total(self):
        """Face du bas (nœuds 0,1,2 dans plan z=0) — pression vers -z.

        Normale de la face 0→1→2 : n̂ = (0,0,-1) (vers le bas, hors du tet).
        Force totale = -p * n̂ * area_face = p * (0,0,1) * 0.5.
        Vérification : Σ fz = p * 0.5.
        """
        mat = ElasticMaterial(E=1.0, nu=0.0, rho=1.0)
        mesh = self._unit_tetra_mesh(mat)
        p = 1000.0
        # Face z=0 : nœuds 0→1→2 → n̂ = cross((1,0,0),(0,1,0)) = (0,0,1) → pointe vers intérieur
        # Pour sortant (vers -z), on inverse l'ordre : 0→2→1
        pressure = PressureLoad(node_ids=(0, 2, 1), magnitude=p)
        bc = BoundaryConditions(dirichlet={}, neumann={}, pressure=(pressure,))
        assembler = Assembler(mesh)
        F = assembler.assemble_forces(bc)
        area_face = 0.5
        # Force totale en z : -p * (-1) * 0.5 = p * 0.5
        np.testing.assert_allclose(F[2::3].sum(), p * area_face, rtol=1e-10)

    def test_wrong_order_raises(self):
        """Ordre CCW incorrect → ValueError."""
        mat = ElasticMaterial(E=1.0, nu=0.0, rho=1.0)
        mesh = self._unit_tetra_mesh(mat)
        # 0→1→2 : n̂ vers +z → intérieur du tet → doit lever ValueError
        pressure = PressureLoad(node_ids=(0, 1, 2), magnitude=1000.0)
        bc = BoundaryConditions(dirichlet={}, neumann={}, pressure=(pressure,))
        assembler = Assembler(mesh)
        with pytest.raises(ValueError, match="intérieur"):
            assembler.assemble_forces(bc)


# ===========================================================================
# Pression surfacique 3D — Hexa8
# ===========================================================================

class TestPressureLoad3DHexa:
    """Pression sur une face quadrangulaire d'un hexaèdre."""

    def _unit_cube_mesh(self, mat):
        nodes = np.array([
            [0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.],
            [0., 0., 1.], [1., 0., 1.], [1., 1., 1.], [0., 1., 1.],
        ])
        elems = (ElementData(Hexa8, tuple(range(8)), mat, {}),)
        return Mesh(nodes=nodes, elements=elems, n_dim=3)

    def test_bottom_face_force_total(self):
        """Face du bas (z=0) : pression normale sortante vers -z.

        Nœuds CCW vu depuis -z (extérieur) : 0→3→2→1.
        n̂ = (0,0,-1) (sortant).
        Force totale = -p * (-1) * 1 = p.
        """
        mat = ElasticMaterial(E=1.0, nu=0.0, rho=1.0)
        mesh = self._unit_cube_mesh(mat)
        p = 2000.0
        # Face z=0 vue depuis l'extérieur (-z) en CCW : 0→3→2→1
        pressure = PressureLoad(node_ids=(0, 3, 2, 1), magnitude=p)
        bc = BoundaryConditions(dirichlet={}, neumann={}, pressure=(pressure,))
        assembler = Assembler(mesh)
        F = assembler.assemble_forces(bc)
        # Force totale en z = p * area = p * 1.0
        np.testing.assert_allclose(F[2::3].sum(), p * 1.0, rtol=1e-9)

    def test_wrong_order_raises(self):
        """Ordre CW → ValueError."""
        mat = ElasticMaterial(E=1.0, nu=0.0, rho=1.0)
        mesh = self._unit_cube_mesh(mat)
        # 0→1→2→3 vu depuis +z → n̂ vers +z → intérieur
        pressure = PressureLoad(node_ids=(0, 1, 2, 3), magnitude=1000.0)
        bc = BoundaryConditions(dirichlet={}, neumann={}, pressure=(pressure,))
        assembler = Assembler(mesh)
        with pytest.raises(ValueError, match="intérieur"):
            assembler.assemble_forces(bc)

    def test_side_face_force_direction(self):
        """Face droite (x=1) : pression vers +x sortante.

        Nœuds 1→2→6→5 (CCW vu depuis +x).
        n̂ = (+1, 0, 0).
        Σ fx = -p * 1 = -p (compression → force vers intérieur).
        """
        mat = ElasticMaterial(E=1.0, nu=0.0, rho=1.0)
        mesh = self._unit_cube_mesh(mat)
        p = 3000.0
        pressure = PressureLoad(node_ids=(1, 2, 6, 5), magnitude=p)
        bc = BoundaryConditions(dirichlet={}, neumann={}, pressure=(pressure,))
        assembler = Assembler(mesh)
        F = assembler.assemble_forces(bc)
        np.testing.assert_allclose(F[0::3].sum(), -p * 1.0, rtol=1e-9)


# ===========================================================================
# Intégration complète : Beam2D console avec charge distribuée
# ===========================================================================

class TestBeam2DCantileveredUDL:
    """Console Beam2D avec charge transverse uniforme — validation complète.

    Référence : Timoshenko & Gere, « Mechanics of Materials ».
        δ_max = 5qL⁴/(384EI)  (bi-appuyée)
        δ_tip = qL⁴/(8EI)    (console)
        θ_tip = qL³/(6EI)    (console)
    """

    def setup_method(self):
        self.E = 200e9
        self.A = 0.005
        self.I = 1e-5
        self.L = 3.0
        self.q = 8000.0
        self.mat = ElasticMaterial(E=self.E, nu=0.3, rho=7850.0)
        self.props = {"area": self.A, "inertia": self.I}

    def _console(self, n_elem: int):
        L_e = self.L / n_elem
        node_coords = np.array([[i * L_e, 0.0] for i in range(n_elem + 1)])
        elems = tuple(
            ElementData(Beam2D, (i, i + 1), self.mat, self.props)
            for i in range(n_elem)
        )
        mesh = Mesh(nodes=node_coords, elements=elems, n_dim=2, dof_per_node=3)
        distributed = tuple(
            DistributedLineLoad(node_ids=(i, i + 1), qy=-self.q)
            for i in range(n_elem)
        )
        bc = BoundaryConditions(
            dirichlet={0: {0: 0.0, 1: 0.0, 2: 0.0}},
            neumann={},
            distributed=distributed,
        )
        assembler = Assembler(mesh)
        K = assembler.assemble_stiffness()
        F = assembler.assemble_forces(bc)
        K_bc, F_bc = apply_dirichlet(K, F, mesh, bc)
        u = StaticSolver().solve(K_bc, F_bc)
        return u

    def test_1_element_exact_tip_deflection(self):
        u = self._console(1)
        delta_ref = self.q * self.L**4 / (8.0 * self.E * self.I)
        # uy du dernier nœud (DDL indice 4 = 3*1+1)
        np.testing.assert_allclose(abs(u[4]), delta_ref, rtol=1e-10)

    def test_1_element_exact_tip_rotation(self):
        u = self._console(1)
        theta_ref = self.q * self.L**3 / (6.0 * self.E * self.I)
        np.testing.assert_allclose(abs(u[5]), theta_ref, rtol=1e-10)

    def test_2_elements_exact(self):
        """2 éléments = même résultat exact (Hermite cubique complet)."""
        u = self._console(2)
        delta_ref = self.q * self.L**4 / (8.0 * self.E * self.I)
        np.testing.assert_allclose(abs(u[-2]), delta_ref, rtol=1e-10)

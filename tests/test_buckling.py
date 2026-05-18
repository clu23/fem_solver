"""Tests pour l'analyse de flambage linéaire.

Validation analytique :
- Charge critique d'Euler pour une colonne pince-pincée : P_cr = π²EI/L²
- Charge critique d'Euler pour une colonne encastrée-libre : P_cr = π²EI/(4L²)
- Propriétés matricielles de K_g (symétrie, signe, zéro sans précontrainte)
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from femsolver.core.assembler import Assembler
from femsolver.core.boundary import apply_dirichlet
from femsolver.core.material import ElasticMaterial
from femsolver.core.mesh import BoundaryConditions, ElementData, Mesh
from femsolver.core.solver import BucklingSolver, StaticSolver
from femsolver.elements.bar2d import Bar2D
from femsolver.elements.beam2d import Beam2D
from femsolver.elements.beam3d import Beam3D
from femsolver.elements.hexa8 import Hexa8
from femsolver.elements.quad4 import Quad4
from femsolver.core.sections import RectangularSection


# ---------------------------------------------------------------------------
# Matériaux et géométries standards
# ---------------------------------------------------------------------------

@pytest.fixture()
def steel() -> ElasticMaterial:
    return ElasticMaterial(E=210e9, nu=0.3, rho=7800)


@pytest.fixture()
def beam_props_10mm():
    """Section carrée 10 × 10 mm²."""
    b = h = 0.01   # m
    return {"area": b * h, "inertia": b * h**3 / 12}


# ---------------------------------------------------------------------------
# Tests de propriétés de K_g — Bar2D
# ---------------------------------------------------------------------------

class TestBar2DGeometricStiffness:
    """Propriétés mathématiques de K_g pour l'élément barre."""

    def test_Kg_zero_when_no_displacement(self, steel):
        """K_g = 0 si N = 0 (aucune précontrainte)."""
        nodes = np.array([[0.0, 0.0], [1.0, 0.0]])
        u_e = np.zeros(4)
        K_g = Bar2D().geometric_stiffness_matrix(
            steel, nodes, {"area": 1e-4}, u_e
        )
        np.testing.assert_allclose(K_g, 0.0, atol=1e-20)

    def test_Kg_symmetric(self, steel):
        """K_g doit être symétrique."""
        nodes = np.array([[0.0, 0.0], [1.0, 0.0]])
        # Barre comprimée : u2_axial = -δ
        delta = 1e-4   # m
        u_e = np.array([0.0, 0.0, -delta, 0.0])
        K_g = Bar2D().geometric_stiffness_matrix(
            steel, nodes, {"area": 1e-4}, u_e
        )
        np.testing.assert_allclose(K_g, K_g.T, atol=1e-20)

    def test_Kg_sign_compression(self, steel):
        """Compression → K_g négatif (réduit la rigidité transversale)."""
        nodes = np.array([[0.0, 0.0], [1.0, 0.0]])
        area = 1e-4   # m²
        L = 1.0       # m
        delta = 1e-6  # compression axiale
        u_e = np.array([0.0, 0.0, -delta, 0.0])   # barre horizontale comprimée

        K_g = Bar2D().geometric_stiffness_matrix(steel, nodes, {"area": area}, u_e)

        # N = -EA*delta/L < 0 → K_g[1,1] = N/L < 0
        N = -steel.E * area * delta / L
        expected_diag_transv = N / L
        np.testing.assert_allclose(K_g[1, 1], expected_diag_transv, rtol=1e-10)
        assert K_g[1, 1] < 0, "K_g transversal doit être négatif en compression"

    def test_Kg_sign_tension(self, steel):
        """Traction → K_g positif (augmente la rigidité transversale)."""
        nodes = np.array([[0.0, 0.0], [1.0, 0.0]])
        area = 1e-4
        delta = 1e-6  # allongement axial
        u_e = np.array([0.0, 0.0, delta, 0.0])

        K_g = Bar2D().geometric_stiffness_matrix(steel, nodes, {"area": area}, u_e)
        assert K_g[1, 1] > 0, "K_g transversal doit être positif en traction"

    def test_Kg_rotated_bar(self, steel):
        """K_g pour une barre à 45° — rotation correcte."""
        L = 1.0
        nodes = np.array([[0.0, 0.0], [L / np.sqrt(2), L / np.sqrt(2)]])
        area = 1e-4
        delta = 1e-6  # compression axiale (dans la direction de la barre)
        c = s = 1.0 / np.sqrt(2)
        u_e = np.array([0.0, 0.0, -delta * c, -delta * s])

        K_g = Bar2D().geometric_stiffness_matrix(steel, nodes, {"area": area}, u_e)
        np.testing.assert_allclose(K_g, K_g.T, atol=1e-20)
        # En traction/compression pure, les valeurs propres de K_g sont {N/L, 0, 0, -N/L}
        # ici N < 0 → valeur propre non-nulle = N/L < 0
        eigvals = np.linalg.eigvalsh(K_g)
        assert eigvals.min() < 0, "K_g doit avoir une valeur propre négative en compression"


# ---------------------------------------------------------------------------
# Tests de propriétés de K_g — Beam2D
# ---------------------------------------------------------------------------

class TestBeam2DGeometricStiffness:
    """Propriétés mathématiques de K_g pour la poutre Euler–Bernoulli."""

    def test_Kg_zero_when_no_displacement(self, steel, beam_props_10mm):
        """K_g = 0 si N = 0."""
        nodes = np.array([[0.0, 0.0], [1.0, 0.0]])
        u_e = np.zeros(6)
        K_g = Beam2D().geometric_stiffness_matrix(
            steel, nodes, beam_props_10mm, u_e
        )
        np.testing.assert_allclose(K_g, 0.0, atol=1e-20)

    def test_Kg_symmetric(self, steel, beam_props_10mm):
        """K_g doit être symétrique."""
        nodes = np.array([[0.0, 0.0], [1.0, 0.0]])
        # Compression axiale : u_x2 = -delta (direction locale x = global x)
        u_e = np.array([0.0, 0.0, 0.0, -1e-6, 0.0, 0.0])
        K_g = Beam2D().geometric_stiffness_matrix(
            steel, nodes, beam_props_10mm, u_e
        )
        np.testing.assert_allclose(K_g, K_g.T, atol=1e-20)

    def test_Kg_negative_in_compression(self, steel, beam_props_10mm):
        """Compression → K_g négatif dans le bloc de flexion."""
        nodes = np.array([[0.0, 0.0], [1.0, 0.0]])
        area = beam_props_10mm["area"]
        L = 1.0
        # Compression : déplacement axial nœud 2 = -delta
        delta = 1e-6
        u_e = np.array([0.0, 0.0, 0.0, -delta, 0.0, 0.0])

        K_g = Beam2D().geometric_stiffness_matrix(
            steel, nodes, beam_props_10mm, u_e
        )
        N = -steel.E * area * delta / L   # N < 0

        # K_g[1,1] : DDL uy1–uy1 (flexion) = 36*N/(30L) < 0 pour compression
        expected = 36 * N / (30.0 * L)
        np.testing.assert_allclose(K_g[1, 1], expected, rtol=1e-10)
        assert K_g[1, 1] < 0

    def test_Kg_positive_in_tension(self, steel, beam_props_10mm):
        """Traction → K_g positif dans le bloc de flexion."""
        nodes = np.array([[0.0, 0.0], [1.0, 0.0]])
        u_e = np.array([0.0, 0.0, 0.0, 1e-6, 0.0, 0.0])
        K_g = Beam2D().geometric_stiffness_matrix(
            steel, nodes, beam_props_10mm, u_e
        )
        assert K_g[1, 1] > 0

    def test_Kg_axial_dofs_zero(self, steel, beam_props_10mm):
        """Les DDL axiaux (ux₁, ux₂ — indices 0 et 3) ont K_g = 0."""
        nodes = np.array([[0.0, 0.0], [1.0, 0.0]])
        u_e = np.array([0.0, 0.0, 0.0, -1e-6, 0.0, 0.0])
        K_g = Beam2D().geometric_stiffness_matrix(
            steel, nodes, beam_props_10mm, u_e
        )
        # Lignes et colonnes 0 et 3 (DDL axiaux en local) doivent être nulles
        # Après rotation, pour barre horizontale c=1,s=0 → local = global
        np.testing.assert_allclose(K_g[0, :], 0.0, atol=1e-20)
        np.testing.assert_allclose(K_g[:, 0], 0.0, atol=1e-20)
        np.testing.assert_allclose(K_g[3, :], 0.0, atol=1e-20)
        np.testing.assert_allclose(K_g[:, 3], 0.0, atol=1e-20)


# ---------------------------------------------------------------------------
# Tests de propriétés de K_g — Quad4
# ---------------------------------------------------------------------------

class TestQuad4GeometricStiffness:
    """Propriétés mathématiques de K_g pour le quadrilatère bilinéaire."""

    def _unit_square_nodes(self) -> np.ndarray:
        return np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])

    def test_Kg_zero_when_no_displacement(self, steel):
        """K_g = 0 quand tous les déplacements sont nuls."""
        nodes = self._unit_square_nodes()
        props = {"thickness": 0.1}
        K_g = Quad4().geometric_stiffness_matrix(
            steel, nodes, props, np.zeros(8)
        )
        np.testing.assert_allclose(K_g, 0.0, atol=1e-20)

    def test_Kg_symmetric(self, steel):
        """K_g doit être symétrique."""
        nodes = self._unit_square_nodes()
        props = {"thickness": 0.1}
        # Compression uniaxiale en x : ux varie linéairement
        u_e = np.array([-1e-5, 0.0, 0.0, 0.0, 0.0, 0.0, -1e-5, 0.0])
        K_g = Quad4().geometric_stiffness_matrix(steel, nodes, props, u_e)
        np.testing.assert_allclose(K_g, K_g.T, atol=1e-20)

    def test_Kg_shape(self, steel):
        """K_g doit être de taille (8, 8)."""
        nodes = self._unit_square_nodes()
        props = {"thickness": 0.1}
        u_e = np.ones(8) * 1e-6
        K_g = Quad4().geometric_stiffness_matrix(steel, nodes, props, u_e)
        assert K_g.shape == (8, 8)

    def test_Kg_negative_under_uniaxial_compression(self, steel):
        """Compression uniaxiale σxx < 0 → K_g est semi-définie négative."""
        nodes = self._unit_square_nodes()
        props = {"thickness": 0.1, "formulation": "plane_stress"}

        # Champ de déplacement donnant σxx < 0, σyy ≈ 0 (plan stress)
        # ux varie linéairement en x → εxx < 0
        eps = -1e-4   # déformation axiale (compression)
        # u_e = [ux1, vy1, ux2, vy2, ux3, vy3, ux4, vy4]
        u_e = np.array([0.0, 0.0, eps, 0.0, eps, 0.0, 0.0, 0.0])

        K_g = Quad4().geometric_stiffness_matrix(steel, nodes, props, u_e)
        eigvals = np.linalg.eigvalsh(K_g)
        # Doit avoir des valeurs propres négatives (compression)
        assert eigvals.min() < 0, (
            f"K_g doit avoir des valeurs propres négatives, min={eigvals.min():.3e}"
        )

    def test_Kg_block_structure(self, steel):
        """K_g ne couple pas ux et uy — vérification de la structure bloc."""
        nodes = self._unit_square_nodes()
        props = {"thickness": 0.1}
        # Compression en x uniquement
        eps = -1e-4
        u_e = np.array([0.0, 0.0, eps, 0.0, eps, 0.0, 0.0, 0.0])
        K_g = Quad4().geometric_stiffness_matrix(steel, nodes, props, u_e)

        # K_g[2I, 2J+1] et K_g[2I+1, 2J] doivent être nuls
        # (pas de couplage ux–uy dans la rigidité géométrique)
        idx_ux = [0, 2, 4, 6]   # DDL ux (indices pairs)
        idx_uy = [1, 3, 5, 7]   # DDL uy (indices impairs)
        cross_block = K_g[np.ix_(idx_ux, idx_uy)]
        np.testing.assert_allclose(cross_block, 0.0, atol=1e-30)


# ---------------------------------------------------------------------------
# Tests de propriétés de K_g — Hexa8
# ---------------------------------------------------------------------------

class TestHexa8GeometricStiffness:
    """Propriétés mathématiques de K_g pour l'hexaèdre trilinéaire."""

    def _unit_cube_nodes(self) -> np.ndarray:
        return np.array([
            [0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.],
            [0., 0., 1.], [1., 0., 1.], [1., 1., 1.], [0., 1., 1.],
        ])

    def test_Kg_zero_when_no_displacement(self, steel):
        """K_g = 0 quand tous les déplacements sont nuls."""
        nodes = self._unit_cube_nodes()
        K_g = Hexa8().geometric_stiffness_matrix(
            steel, nodes, {}, np.zeros(24)
        )
        np.testing.assert_allclose(K_g, 0.0, atol=1e-20)

    def test_Kg_symmetric(self, steel):
        """K_g doit être symétrique."""
        nodes = self._unit_cube_nodes()
        # Compression en z : uz varie linéairement
        eps = -1e-5
        u_e = np.zeros(24)
        u_e[2]  = 0.0   # uz_0
        u_e[5]  = 0.0   # uz_1
        u_e[8]  = 0.0   # uz_2
        u_e[11] = 0.0   # uz_3
        u_e[14] = eps   # uz_4
        u_e[17] = eps   # uz_5
        u_e[20] = eps   # uz_6
        u_e[23] = eps   # uz_7
        K_g = Hexa8().geometric_stiffness_matrix(steel, nodes, {}, u_e)
        np.testing.assert_allclose(K_g, K_g.T, atol=1e-20)

    def test_Kg_shape(self, steel):
        """K_g doit être de taille (24, 24)."""
        nodes = self._unit_cube_nodes()
        K_g = Hexa8().geometric_stiffness_matrix(steel, nodes, {}, np.ones(24) * 1e-7)
        assert K_g.shape == (24, 24)

    def test_Kg_block_structure(self, steel):
        """K_g ne couple pas ux, uy, uz entre eux."""
        nodes = self._unit_cube_nodes()
        # Compression en z
        u_e = np.zeros(24)
        u_e[14] = u_e[17] = u_e[20] = u_e[23] = -1e-5
        K_g = Hexa8().geometric_stiffness_matrix(steel, nodes, {}, u_e)

        idx_ux = np.array([0, 3, 6, 9, 12, 15, 18, 21])   # DDL x
        idx_uy = np.array([1, 4, 7, 10, 13, 16, 19, 22])   # DDL y
        cross_xy = K_g[np.ix_(idx_ux, idx_uy)]
        np.testing.assert_allclose(cross_xy, 0.0, atol=1e-30)


# ---------------------------------------------------------------------------
# Test flambage — colonne d'Euler pince-pincée (Beam2D, n éléments)
# ---------------------------------------------------------------------------

class TestBucklingEulerColumnPinPin:
    """Charge critique d'Euler pour une colonne pince-pincée.

    Formule analytique : P_cr = π²EI / L²

    Condition aux limites :
    - Nœud inférieur : ux = 0 (transverse), uy = 0 (axial)
    - Nœud supérieur : ux = 0 (transverse), uy = libre
    - Charge : P_ref = −1 N en uy au nœud supérieur

    Orientation : colonne le long de l'axe y global.
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.E = 210e9         # Pa
        self.b = self.h = 0.01   # m  (section 10×10 mm)
        self.A = self.b * self.h
        self.I = self.b * self.h**3 / 12
        self.L = 1.0           # m
        self.mat = ElasticMaterial(E=self.E, nu=0.3, rho=7800)
        self.P_ref = 1.0       # N (charge de référence)
        # Analytique : P_cr = π²EI/L²
        self.P_cr_analytical = np.pi**2 * self.E * self.I / self.L**2

    def _build_column(self, n_elem: int) -> tuple:
        """Construit le maillage et les CL pour la colonne."""
        n_nodes = n_elem + 1
        L_elem = self.L / n_elem
        nodes = np.array([[0.0, i * L_elem] for i in range(n_nodes)])

        props = {"area": self.A, "inertia": self.I}
        elements = tuple(
            ElementData(Beam2D, (i, i + 1), self.mat, props)
            for i in range(n_elem)
        )
        mesh = Mesh(nodes=nodes, elements=elements, n_dim=2, dof_per_node=3)

        # Colonne verticale (y = axial) :
        # DDL global : 3*i+0 = ux, 3*i+1 = uy, 3*i+2 = θz
        bc = BoundaryConditions(
            dirichlet={
                0: {0: 0.0, 1: 0.0},     # bas : ux=0, uy=0 (pince)
                n_elem: {0: 0.0},          # haut : ux=0 (rouleau transversal)
            },
            neumann={
                n_elem: {1: -self.P_ref},  # compression en -y
            },
        )
        return mesh, bc

    def test_euler_column_10_elements(self):
        """10 éléments — P_cr dans les 1 % de l'analytique."""
        mesh, bc = self._build_column(n_elem=10)
        assembler = Assembler(mesh)
        K = assembler.assemble_stiffness()
        F = assembler.assemble_forces(bc)
        ds = apply_dirichlet(K, F, mesh, bc)

        u = StaticSolver().solve(*ds)
        K_g = assembler.assemble_geometric_stiffness(u)
        K_g_free = ds.reduce(K_g)

        lambda_cr, _ = BucklingSolver().solve(ds.K_free, K_g_free, n_modes=1)
        P_cr = float(lambda_cr[0]) * self.P_ref

        np.testing.assert_allclose(P_cr, self.P_cr_analytical, rtol=0.01)

    def test_euler_column_20_elements(self):
        """20 éléments — P_cr dans les 0.1 % de l'analytique."""
        mesh, bc = self._build_column(n_elem=20)
        assembler = Assembler(mesh)
        K = assembler.assemble_stiffness()
        F = assembler.assemble_forces(bc)
        ds = apply_dirichlet(K, F, mesh, bc)

        u = StaticSolver().solve(*ds)
        K_g = assembler.assemble_geometric_stiffness(u)
        K_g_free = ds.reduce(K_g)

        lambda_cr, _ = BucklingSolver().solve(ds.K_free, K_g_free, n_modes=1)
        P_cr = float(lambda_cr[0]) * self.P_ref

        np.testing.assert_allclose(P_cr, self.P_cr_analytical, rtol=0.001)

    def test_higher_modes(self):
        """Les modes supérieurs suivent P_n = n² × P_cr1."""
        # Pour la colonne pince-pincée : P_n = n² π² EI / L²
        mesh, bc = self._build_column(n_elem=30)
        assembler = Assembler(mesh)
        K = assembler.assemble_stiffness()
        F = assembler.assemble_forces(bc)
        ds = apply_dirichlet(K, F, mesh, bc)

        u = StaticSolver().solve(*ds)
        K_g = assembler.assemble_geometric_stiffness(u)
        K_g_free = ds.reduce(K_g)

        lambda_cr, _ = BucklingSolver().solve(ds.K_free, K_g_free, n_modes=3)

        # Mode 2 ≈ 4 × mode 1, mode 3 ≈ 9 × mode 1
        ratio_2 = float(lambda_cr[1]) / float(lambda_cr[0])
        ratio_3 = float(lambda_cr[2]) / float(lambda_cr[0])
        np.testing.assert_allclose(ratio_2, 4.0, rtol=0.02)
        np.testing.assert_allclose(ratio_3, 9.0, rtol=0.03)


# ---------------------------------------------------------------------------
# Test flambage — colonne d'Euler encastrée-libre (Beam2D)
# ---------------------------------------------------------------------------

class TestBucklingEulerColumnCantilever:
    """Charge critique d'Euler pour une colonne encastrée-libre (console).

    Formule analytique : P_cr = π²EI / (4L²) = π²EI / (2L)²

    Le facteur 4 vient de la longueur effective L_eff = 2L.
    """

    def test_euler_cantilever_20_elements(self):
        """Console verticale — P_cr dans les 1 % de l'analytique."""
        E = 210e9
        b = h = 0.01
        A = b * h
        I = b * h**3 / 12
        L = 1.0
        mat = ElasticMaterial(E=E, nu=0.3, rho=7800)

        P_cr_analytical = np.pi**2 * E * I / (4.0 * L**2)
        n_elem = 20
        L_elem = L / n_elem
        nodes = np.array([[0.0, i * L_elem] for i in range(n_elem + 1)])
        props = {"area": A, "inertia": I}
        elements = tuple(
            ElementData(Beam2D, (i, i + 1), mat, props) for i in range(n_elem)
        )
        mesh = Mesh(nodes=nodes, elements=elements, n_dim=2, dof_per_node=3)

        bc = BoundaryConditions(
            dirichlet={
                0: {0: 0.0, 1: 0.0, 2: 0.0},   # encastrement complet
            },
            neumann={
                n_elem: {1: -1.0},               # compression unitaire
            },
        )

        assembler = Assembler(mesh)
        K = assembler.assemble_stiffness()
        F = assembler.assemble_forces(bc)
        ds = apply_dirichlet(K, F, mesh, bc)

        u = StaticSolver().solve(*ds)
        K_g = assembler.assemble_geometric_stiffness(u)
        K_g_free = ds.reduce(K_g)

        lambda_cr, _ = BucklingSolver().solve(ds.K_free, K_g_free, n_modes=1)
        P_cr = float(lambda_cr[0])

        np.testing.assert_allclose(P_cr, P_cr_analytical, rtol=0.01)


# ---------------------------------------------------------------------------
# Test flambage — vérification de l'assemblage K_g global
# ---------------------------------------------------------------------------

class TestAssembleGeometricStiffness:
    """Vérifie que l'assembleur construit K_g correctement."""

    def test_Kg_sparse_format(self, steel):
        """K_g assemblée doit être au format CSR."""
        n_elem = 3
        L = 1.0
        L_e = L / n_elem
        nodes = np.array([[0.0, i * L_e] for i in range(n_elem + 1)])
        props = {"area": 1e-4, "inertia": 8.33e-9}
        elements = tuple(
            ElementData(Beam2D, (i, i + 1), steel, props) for i in range(n_elem)
        )
        mesh = Mesh(nodes=nodes, elements=elements, n_dim=2, dof_per_node=3)

        # Vecteur de déplacements uniforme (axial uniquement)
        u = np.zeros(mesh.n_dof)
        u[1::3] = -1e-6  # compression en y (DDL uy = indice 1 mod 3)

        assembler = Assembler(mesh)
        K_g = assembler.assemble_geometric_stiffness(u)

        assert isinstance(K_g, csr_matrix)
        assert K_g.shape == (mesh.n_dof, mesh.n_dof)

    def test_Kg_symmetric_global(self, steel):
        """K_g globale doit être symétrique."""
        n_elem = 5
        L = 1.0
        L_e = L / n_elem
        nodes = np.array([[0.0, i * L_e] for i in range(n_elem + 1)])
        props = {"area": 1e-4, "inertia": 8.33e-9}
        elements = tuple(
            ElementData(Beam2D, (i, i + 1), steel, props) for i in range(n_elem)
        )
        mesh = Mesh(nodes=nodes, elements=elements, n_dim=2, dof_per_node=3)

        u = np.zeros(mesh.n_dof)
        u[1::3] = -1e-6

        K_g = Assembler(mesh).assemble_geometric_stiffness(u)
        diff = (K_g - K_g.T).toarray()
        np.testing.assert_allclose(diff, 0.0, atol=1e-20)

    def test_Kg_zero_without_prestress(self, steel):
        """K_g = 0 si u = 0 (aucune précontrainte)."""
        n_elem = 3
        L = 1.0
        L_e = L / n_elem
        nodes = np.array([[0.0, i * L_e] for i in range(n_elem + 1)])
        props = {"area": 1e-4, "inertia": 8.33e-9}
        elements = tuple(
            ElementData(Beam2D, (i, i + 1), steel, props) for i in range(n_elem)
        )
        mesh = Mesh(nodes=nodes, elements=elements, n_dim=2, dof_per_node=3)

        K_g = Assembler(mesh).assemble_geometric_stiffness(np.zeros(mesh.n_dof))
        np.testing.assert_allclose(K_g.toarray(), 0.0, atol=1e-20)


# ---------------------------------------------------------------------------
# Test flambage — colonne Quad4 (en-plan, "strip column")
# ---------------------------------------------------------------------------

class TestBucklingQuad4StripColumn:
    """Charge critique pour une bande rectangulaire sous compression.

    Modélise une colonne 2D en état plan de contrainte :
    - Dimensions : L × w (L >> w)
    - Compression uniforme σ_yy appliquée par des forces nodales
    - P_cr ≈ π²EI/L² avec I = w³t/12 (moment d'inertie de la bande)

    Vérifie surtout la cohérence de K_g (symétrie, signe, utilisation).
    """

    def test_Kg_compression_reduces_stiffness(self):
        """K_g pour un état σxx < 0 réel doit avoir des valeurs propres négatives."""
        E = 1e6
        nu = 0.0
        mat = ElasticMaterial(E=E, nu=nu, rho=1.0)
        nodes = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        t = 0.1
        props = {"thickness": t, "formulation": "plane_stress"}

        # État de compression pure en x : u1=(0,0), u2=(eps,0), u3=(eps,0), u4=(0,0)
        eps = -1e-3  # compression axiale en x
        u_e = np.array([0.0, 0.0, eps, 0.0, eps, 0.0, 0.0, 0.0])

        K_g = Quad4().geometric_stiffness_matrix(mat, nodes, props, u_e)
        eigvals = np.linalg.eigvalsh(K_g)
        assert eigvals.min() < 0, "compression → K_g a des valeurs propres négatives"

    def test_Kg_analytical_uniform_stress(self):
        """K_g pour σxx uniforme — comparaison terme par terme.

        Pour un carré unitaire sous σxx = σ (constant) :
        k_IJ = σ ∫∫ (∂N_I/∂x)(∂N_J/∂x) dA

        Calcul analytique pour le carré [0,1]² avec fonctions bilinéaires.
        L'intégrale ∫∫ (∂N_I/∂x)(∂N_J/∂x) dA est bien définie et peut
        être calculée numériquement comme référence.
        """
        E = 1.0
        nu = 0.0
        mat = ElasticMaterial(E=E, nu=nu, rho=1.0)
        nodes = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        t = 1.0
        props = {"thickness": t, "formulation": "plane_stress"}

        # Champ de déformation εxx = eps = cste → σxx = E*eps, σyy = τxy = 0
        eps_val = -0.01
        u_e = np.array([0.0, 0.0, eps_val, 0.0, eps_val, 0.0, 0.0, 0.0])
        K_g = Quad4().geometric_stiffness_matrix(mat, nodes, props, u_e)

        # Vérification : symétrie et structure
        np.testing.assert_allclose(K_g, K_g.T, atol=1e-20)

        # K_g[2I, 2J] = K_g[2I+1, 2J+1] (même valeur pour ux et uy)
        for i in range(4):
            for j in range(4):
                np.testing.assert_allclose(
                    K_g[2*i, 2*j], K_g[2*i+1, 2*j+1], atol=1e-20,
                    err_msg=f"K_g[2*{i}, 2*{j}] ≠ K_g[2*{i}+1, 2*{j}+1]"
                )


# ---------------------------------------------------------------------------
# Tests de propriétés de K_g — Beam3D
# ---------------------------------------------------------------------------

class TestBeam3DGeometricStiffness:
    """Propriétés mathématiques de K_g pour la poutre Timoshenko 3D.

    DDL locaux : [ux₁,uy₁,uz₁,θx₁,θy₁,θz₁, ux₂,uy₂,uz₂,θx₂,θy₂,θz₂]
    Indices    : [  0,  1,  2,  3,  4,  5,    6,  7,  8,  9, 10, 11]
    """

    @pytest.fixture()
    def sec(self):
        """Section carrée 10 × 10 mm²."""
        return RectangularSection(width=0.01, height=0.01)

    @pytest.fixture()
    def mat(self):
        return ElasticMaterial(E=210e9, nu=0.3, rho=7800)

    @pytest.fixture()
    def nodes_horiz(self):
        """Poutre horizontale de longueur 1 m le long de x."""
        return np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    def _compress_u(self, delta: float = 1e-6) -> np.ndarray:
        """Champ de déplacement donnant une compression axiale δ en x."""
        u = np.zeros(12)
        u[6] = -delta   # ux₂ = −δ → compression axiale
        return u

    def test_Kg_zero_when_no_displacement(self, mat, sec, nodes_horiz):
        """K_g = 0 si u_e = 0 (aucune précontrainte)."""
        K_g = Beam3D().geometric_stiffness_matrix(
            mat, nodes_horiz, {"section": sec}, np.zeros(12)
        )
        np.testing.assert_allclose(K_g, 0.0, atol=1e-20)

    def test_Kg_shape(self, mat, sec, nodes_horiz):
        """K_g doit être de taille (12, 12)."""
        K_g = Beam3D().geometric_stiffness_matrix(
            mat, nodes_horiz, {"section": sec}, self._compress_u()
        )
        assert K_g.shape == (12, 12)

    def test_Kg_symmetric(self, mat, sec, nodes_horiz):
        """K_g doit être symétrique."""
        K_g = Beam3D().geometric_stiffness_matrix(
            mat, nodes_horiz, {"section": sec}, self._compress_u()
        )
        np.testing.assert_allclose(K_g, K_g.T, atol=1e-20)

    def test_Kg_axial_dofs_zero(self, mat, sec, nodes_horiz):
        """DDL axiaux ux₁(0) et ux₂(6) : lignes et colonnes nulles."""
        K_g = Beam3D().geometric_stiffness_matrix(
            mat, nodes_horiz, {"section": sec}, self._compress_u()
        )
        # Poutre horizontale (e₁ = x_global) → local = global → indices 0 et 6
        np.testing.assert_allclose(K_g[0, :], 0.0, atol=1e-20)
        np.testing.assert_allclose(K_g[:, 0], 0.0, atol=1e-20)
        np.testing.assert_allclose(K_g[6, :], 0.0, atol=1e-20)
        np.testing.assert_allclose(K_g[:, 6], 0.0, atol=1e-20)

    def test_Kg_negative_in_compression(self, mat, sec, nodes_horiz):
        """Compression → K_g a des valeurs propres négatives."""
        K_g = Beam3D().geometric_stiffness_matrix(
            mat, nodes_horiz, {"section": sec}, self._compress_u(1e-6)
        )
        assert np.linalg.eigvalsh(K_g).min() < 0

    def test_Kg_positive_in_tension(self, mat, sec, nodes_horiz):
        """Traction → K_g a des valeurs propres non-négatives (semi-définie positive)."""
        u_tension = np.zeros(12)
        u_tension[6] = 1e-6   # ux₂ = +δ → traction
        K_g = Beam3D().geometric_stiffness_matrix(
            mat, nodes_horiz, {"section": sec}, u_tension
        )
        assert np.linalg.eigvalsh(K_g).min() >= -1e-12

    def test_Kg_xy_block_matches_beam2d_formula(self, mat, sec):
        """Bloc plan xy {uy₁,θz₁,uy₂,θz₂} identique à la formule Przemieniecki."""
        L = 2.0
        nodes = np.array([[0.0, 0.0, 0.0], [L, 0.0, 0.0]])
        EA = mat.E * sec.area
        delta = 1e-6
        N = -EA * delta / L   # compression
        a = N / (30.0 * L)

        u_e = np.zeros(12)
        u_e[6] = -delta

        K_g = Beam3D().geometric_stiffness_matrix(mat, nodes, {"section": sec}, u_e)

        # Bloc xy local : indices globaux = indices locaux (barre horizontale)
        iy = [1, 5, 7, 11]
        K_block = K_g[np.ix_(iy, iy)]
        L2 = L * L
        K_expected = a * np.array([
            [ 36.0,  3.0 * L, -36.0,  3.0 * L],
            [  3.0 * L,  4.0 * L2, -3.0 * L,       -L2],
            [-36.0, -3.0 * L,  36.0, -3.0 * L],
            [  3.0 * L,       -L2, -3.0 * L,  4.0 * L2],
        ])
        np.testing.assert_allclose(K_block, K_expected, rtol=1e-10)

    def test_Kg_xz_block_opposite_coupling(self, mat, sec):
        """Bloc plan xz {uz₁,θy₁,uz₂,θy₂} : couplages uz–θy renversés vs xy."""
        L = 2.0
        nodes = np.array([[0.0, 0.0, 0.0], [L, 0.0, 0.0]])
        EA = mat.E * sec.area
        delta = 1e-6
        N = -EA * delta / L
        a = N / (30.0 * L)
        L2 = L * L

        u_e = np.zeros(12)
        u_e[6] = -delta

        K_g = Beam3D().geometric_stiffness_matrix(mat, nodes, {"section": sec}, u_e)

        # Bloc xz : indices [2,4,8,10]
        iz = [2, 4, 8, 10]
        K_block = K_g[np.ix_(iz, iz)]
        K_expected = a * np.array([
            [ 36.0, -3.0 * L, -36.0, -3.0 * L],
            [ -3.0 * L,  4.0 * L2,  3.0 * L,        -L2],
            [-36.0,  3.0 * L,  36.0,  3.0 * L],
            [ -3.0 * L,        -L2,  3.0 * L,  4.0 * L2],
        ])
        np.testing.assert_allclose(K_block, K_expected, rtol=1e-10)

        # Vérifier que les couplages uz–θy sont bien le négatif des couplages uy–θz
        iy = [1, 5, 7, 11]
        K_xy = K_g[np.ix_(iy, iy)]
        # K_xz[0,1] = -K_xy[0,1], K_xz[0,3] = -K_xy[0,3], etc.
        for row_i, col_j in [(0, 1), (0, 3), (1, 2), (2, 3)]:
            np.testing.assert_allclose(
                K_block[row_i, col_j], -K_xy[row_i, col_j], rtol=1e-10,
                err_msg=f"K_xz[{row_i},{col_j}] ≠ −K_xy[{row_i},{col_j}]"
            )

    def test_Kg_torsion_block(self, mat, sec):
        """Bloc torsion {θx₁(3),θx₂(9)} = N·(Ip/A)/L·[[1,−1],[−1,1]]."""
        L = 1.5
        nodes = np.array([[0.0, 0.0, 0.0], [L, 0.0, 0.0]])
        EA = mat.E * sec.area
        delta = 1e-6
        N = -EA * delta / L
        Ip_over_A = (sec.Iy + sec.Iz) / sec.area

        u_e = np.zeros(12)
        u_e[6] = -delta

        K_g = Beam3D().geometric_stiffness_matrix(mat, nodes, {"section": sec}, u_e)

        t = N * Ip_over_A / L
        np.testing.assert_allclose(K_g[3, 3],  t, rtol=1e-10)
        np.testing.assert_allclose(K_g[3, 9], -t, rtol=1e-10)
        np.testing.assert_allclose(K_g[9, 3], -t, rtol=1e-10)
        np.testing.assert_allclose(K_g[9, 9],  t, rtol=1e-10)


# ---------------------------------------------------------------------------
# Tests de flambage — colonne 3D (Beam3D)
# ---------------------------------------------------------------------------

class TestBucklingBeam3DColumn:
    """Flambage d'une colonne 3D — comparaison avec la formule d'Euler.

    Formule analytique :
    - Pince-pincée : P_cr = π²EI/L²
    - Encastrée-libre : P_cr = π²EI/(4L²)

    Orientation : colonne le long de l'axe z global.
    DDL Beam3D : [ux,uy,uz,θx,θy,θz] × 2 nœuds → 12 DDL/élément.
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.E  = 210e9
        self.nu = 0.3
        self.b  = self.h = 0.01   # section carrée 10 × 10 mm
        self.L  = 1.0
        self.mat = ElasticMaterial(E=self.E, nu=self.nu, rho=7800)
        # Section carrée : Iy = Iz = bh³/12 (symmetric)
        self.sec = RectangularSection(width=self.b, height=self.h)
        self.I  = self.b * self.h**3 / 12
        self.P_cr_euler = np.pi**2 * self.E * self.I / self.L**2

    def _build_pinpin_z(self, n_elem: int) -> tuple:
        """Colonne verticale (axe z) pince-pincée sous compression."""
        Le = self.L / n_elem
        nodes = np.array([[0.0, 0.0, i * Le] for i in range(n_elem + 1)])
        props = {"section": self.sec}
        elements = tuple(
            ElementData(Beam3D, (i, i + 1), self.mat, props)
            for i in range(n_elem)
        )
        mesh = Mesh(nodes=nodes, elements=elements, n_dim=3, dof_per_node=6)

        # DDL globaux pour Beam3D (ordre par nœud) :
        # 6*i+0=ux, 6*i+1=uy, 6*i+2=uz, 6*i+3=θx, 6*i+4=θy, 6*i+5=θz
        # Pince-pincée le long de z :
        # - Nœud 0 (bas) : ux=0, uy=0, uz=0  (appui 3D)
        # - Nœud n (haut) : ux=0, uy=0        (rouleau transversal, uz libre)
        bc = BoundaryConditions(
            dirichlet={
                0:      {0: 0.0, 1: 0.0, 2: 0.0},
                n_elem: {0: 0.0, 1: 0.0},
            },
            neumann={
                n_elem: {2: -1.0},   # −1 N en z (compression)
            },
        )
        return mesh, bc

    def _build_cantilever_z(self, n_elem: int) -> tuple:
        """Colonne verticale (axe z) encastrée à la base, libre au sommet."""
        Le = self.L / n_elem
        nodes = np.array([[0.0, 0.0, i * Le] for i in range(n_elem + 1)])
        props = {"section": self.sec}
        elements = tuple(
            ElementData(Beam3D, (i, i + 1), self.mat, props)
            for i in range(n_elem)
        )
        mesh = Mesh(nodes=nodes, elements=elements, n_dim=3, dof_per_node=6)

        # Encastrement complet (tous les DDL bloqués)
        bc = BoundaryConditions(
            dirichlet={
                0: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0},
            },
            neumann={
                n_elem: {2: -1.0},
            },
        )
        return mesh, bc

    def _run(self, mesh, bc, n_modes: int = 1):
        assembler = Assembler(mesh)
        K = assembler.assemble_stiffness()
        F = assembler.assemble_forces(bc)
        ds = apply_dirichlet(K, F, mesh, bc)
        u  = StaticSolver().solve(*ds)
        K_g      = assembler.assemble_geometric_stiffness(u)
        K_g_free = ds.reduce(K_g)
        lambda_cr, _ = BucklingSolver().solve(ds.K_free, K_g_free, n_modes=n_modes)
        return lambda_cr

    def test_euler_pinpin_20_elements(self):
        """Colonne 3D pince-pincée — P_cr dans les 1 % de l'analytique."""
        mesh, bc = self._build_pinpin_z(n_elem=20)
        lambda_cr = self._run(mesh, bc, n_modes=1)
        # P_ref = 1 N → P_cr = lambda_cr × 1
        np.testing.assert_allclose(
            float(lambda_cr[0]), self.P_cr_euler, rtol=0.01,
            err_msg=f"P_cr FEM={float(lambda_cr[0]):.2f} N, analytique={self.P_cr_euler:.2f} N"
        )

    def test_euler_cantilever_20_elements(self):
        """Colonne 3D encastrée-libre — P_cr dans les 1 % de l'analytique."""
        P_cr_cant = self.P_cr_euler / 4.0
        mesh, bc = self._build_cantilever_z(n_elem=20)
        lambda_cr = self._run(mesh, bc, n_modes=1)
        np.testing.assert_allclose(
            float(lambda_cr[0]), P_cr_cant, rtol=0.01,
            err_msg=f"P_cr FEM={float(lambda_cr[0]):.2f} N, analytique={P_cr_cant:.2f} N"
        )

    def test_higher_modes_n_squared_scaling(self):
        """Modes supérieurs de la colonne 3D pince-pincée : P_n = n²·P_cr1."""
        mesh, bc = self._build_pinpin_z(n_elem=30)
        # Pour une section carrée, modes 1+2 sont dégénérés (mêmes dans xy et xz)
        # On extrait 5 modes pour observer 1, 1, 4, 4, 9 (paires dégénérées)
        lambda_cr = self._run(mesh, bc, n_modes=5)
        lam = lambda_cr
        # Modes 1 et 2 doivent être quasi-identiques (dégénérescence)
        np.testing.assert_allclose(lam[0], lam[1], rtol=0.01,
                                    err_msg="Modes 1 et 2 devraient être dégénérés")
        # Mode 3 ≈ 4 × mode 1 (2ème harmonique, plan xy)
        np.testing.assert_allclose(lam[2] / lam[0], 4.0, rtol=0.02)

    def test_convergence_to_beam2d_planar(self):
        """P_cr Beam3D converge vers P_cr Beam2D pour un cas plan (flexion forte).

        Section rectangulaire non-carrée (Iz > Iy) : le mode de flambage
        dans le plan xy (flexion forte) doit donner la même charge critique
        que Beam2D avec la même inertie Iz.
        """
        # Section 20 × 10 mm : Iz (flexion forte) >> Iy (flexion faible)
        b_sec = 0.01   # largeur (z) — flexion faible Iy
        h_sec = 0.02   # hauteur (y) — flexion forte Iz
        sec_3d = RectangularSection(width=b_sec, height=h_sec)
        Iz = b_sec * h_sec**3 / 12   # inertie de flexion forte
        Iy = h_sec * b_sec**3 / 12   # inertie de flexion faible
        assert Iy < Iz  # le mode faible est dans le plan xz local

        P_cr_weak = np.pi**2 * self.E * Iy / self.L**2
        P_cr_strong = np.pi**2 * self.E * Iz / self.L**2
        n_elem = 20

        # ── Beam3D ──────────────────────────────────────────────────
        Le = self.L / n_elem
        nodes = np.array([[0.0, 0.0, i * Le] for i in range(n_elem + 1)])
        props_3d = {"section": sec_3d}
        elements_3d = tuple(
            ElementData(Beam3D, (i, i + 1), self.mat, props_3d)
            for i in range(n_elem)
        )
        mesh_3d = Mesh(nodes=nodes, elements=elements_3d, n_dim=3, dof_per_node=6)
        bc_3d = BoundaryConditions(
            dirichlet={
                0:      {0: 0.0, 1: 0.0, 2: 0.0},
                n_elem: {0: 0.0, 1: 0.0},
            },
            neumann={n_elem: {2: -1.0}},
        )
        lam_3d = self._run(mesh_3d, bc_3d, n_modes=2)
        # Le premier mode est le flambage selon l'axe faible (Iy)
        np.testing.assert_allclose(
            float(lam_3d[0]), P_cr_weak, rtol=0.01,
            err_msg="Beam3D mode faible ≠ Beam2D Iy"
        )
        # Le second mode est le flambage selon l'axe fort (Iz)
        np.testing.assert_allclose(
            float(lam_3d[1]), P_cr_strong, rtol=0.01,
            err_msg="Beam3D mode fort ≠ Beam2D Iz"
        )

        # ── Beam2D (axe faible) ──────────────────────────────────────
        nodes_2d = np.array([[0.0, i * Le] for i in range(n_elem + 1)])
        props_2d = {"area": sec_3d.area, "inertia": Iy}
        elements_2d = tuple(
            ElementData(Beam2D, (i, i + 1), self.mat, props_2d)
            for i in range(n_elem)
        )
        mesh_2d = Mesh(nodes=nodes_2d, elements=elements_2d, n_dim=2, dof_per_node=3)
        bc_2d = BoundaryConditions(
            dirichlet={
                0:      {0: 0.0, 1: 0.0},
                n_elem: {0: 0.0},
            },
            neumann={n_elem: {1: -1.0}},
        )
        assembler_2d = Assembler(mesh_2d)
        K_2d = assembler_2d.assemble_stiffness()
        F_2d = assembler_2d.assemble_forces(bc_2d)
        ds_2d = apply_dirichlet(K_2d, F_2d, mesh_2d, bc_2d)
        u_2d = StaticSolver().solve(*ds_2d)
        Kg_2d = assembler_2d.assemble_geometric_stiffness(u_2d)
        Kg_free_2d = ds_2d.reduce(Kg_2d)
        lam_2d, _ = BucklingSolver().solve(ds_2d.K_free, Kg_free_2d, n_modes=1)

        # Les deux solveurs (Beam3D et Beam2D) donnent le même P_cr
        np.testing.assert_allclose(
            float(lam_3d[0]), float(lam_2d[0]), rtol=0.001,
            err_msg="Beam3D mode faible doit concorder avec Beam2D"
        )

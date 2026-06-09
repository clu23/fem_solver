"""Tests des diagrammes d'efforts internes M / V / N pour les poutres.

Chaque test compare les efforts internes reconstruits à la solution
analytique de la résistance des matériaux pour une console (poutre
encastrée-libre) de longueur L.

Console + charge ponctuelle F en bout
-------------------------------------
    V(x) = F                 (constant)
    M(x) = F·(L − x)         (linéaire, |M|_encastrement = F·L)

Console + charge répartie q uniforme
------------------------------------
    V(x) = q·(L − x)         (linéaire, |V|_encastrement = q·L)
    M(x) = q·(L − x)²/2      (parabolique, |M|_encastrement = q·L²/2)
"""

import matplotlib

matplotlib.use("Agg")  # backend non interactif pour les tests

import numpy as np
import pytest

from femsolver.core.assembler import Assembler
from femsolver.core.boundary import apply_dirichlet
from femsolver.core.material import ElasticMaterial
from femsolver.core.mesh import (
    BoundaryConditions,
    DistributedLineLoad,
    ElementData,
    Mesh,
)
from femsolver.core.solver import StaticSolver
from femsolver.elements.beam2d import Beam2D
from femsolver.elements.beam3d import Beam3D
from femsolver.postprocess.beam_diagrams import (
    BeamDiagram,
    extract_beam_diagrams,
    plot_beam_diagrams,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

E = 210e9          # Pa
NU = 0.3
RHO = 7800.0       # kg/m³
AREA = 1e-4        # m²
INERTIA = 8.333e-6  # m⁴ (≈ section 0.1×0.1 / 12 ... valeur arbitraire cohérente)
L_TOTAL = 2.0      # m


def _build_cantilever(n_elem: int, *, fy_tip: float = 0.0,
                      qy: float = 0.0, fx_tip: float = 0.0):
    """Construit une console (encastrée à gauche) discrétisée en n_elem poutres.

    Returns
    -------
    mesh, bc, u_full
    """
    mat = ElasticMaterial(E=E, nu=NU, rho=RHO)
    n_nodes = n_elem + 1
    xs = np.linspace(0.0, L_TOTAL, n_nodes)
    nodes = np.column_stack([xs, np.zeros(n_nodes)])

    elements = [
        ElementData(
            etype=Beam2D,
            node_ids=(i, i + 1),
            material=mat,
            properties={"area": AREA, "inertia": INERTIA},
        )
        for i in range(n_elem)
    ]

    # Encastrement total au nœud 0 (ux, uy, θz).
    dirichlet = {0: {0: 0.0, 1: 0.0, 2: 0.0}}
    neumann: dict[int, dict[int, float]] = {}
    tip = n_nodes - 1
    if fy_tip or fx_tip:
        neumann[tip] = {}
        if fy_tip:
            neumann[tip][1] = fy_tip
        if fx_tip:
            neumann[tip][0] = fx_tip

    distributed = []
    if qy:
        distributed = [
            DistributedLineLoad(node_ids=(i, i + 1), qy=qy)
            for i in range(n_elem)
        ]

    mesh = Mesh(nodes=nodes, elements=elements, n_dim=2, dof_per_node=3)
    bc = BoundaryConditions(
        dirichlet=dirichlet,
        neumann=neumann,
        distributed=tuple(distributed),
    )

    assembler = Assembler(mesh)
    K = assembler.assemble_stiffness()
    F = assembler.assemble_forces(bc)
    ds = apply_dirichlet(K, F, mesh, bc)
    u = StaticSolver().solve(ds.K_free, ds.F_free)
    u_full = ds.recover(u)
    return mesh, bc, u_full


# ---------------------------------------------------------------------------
# Charge ponctuelle : V constant, M linéaire
# ---------------------------------------------------------------------------


class TestCantileverPointLoad:
    """Console + charge ponctuelle F en bout."""

    def setup_method(self):
        self.F = -1000.0  # N (vers le bas)
        self.mesh, self.bc, self.u = _build_cantilever(1, fy_tip=self.F)
        self.diag = extract_beam_diagrams(self.mesh, self.bc, self.u)[0]

    def test_shear_is_constant(self):
        """V(x) constant pour une charge ponctuelle (dV/dx = 0)."""
        V = self.diag.forces["V"]
        np.testing.assert_allclose(V, V[0], rtol=1e-10)

    def test_shear_magnitude_equals_load(self):
        """|V| = |F| partout."""
        V = self.diag.forces["V"]
        np.testing.assert_allclose(np.abs(V), abs(self.F), rtol=1e-10)

    def test_moment_is_linear(self):
        """M(x) linéaire → dérivée seconde nulle."""
        M = self.diag.forces["M"]
        x = self.diag.x_local
        d2 = np.diff(M, 2) / np.diff(x)[0] ** 2
        assert np.allclose(d2, 0.0, atol=1e-6 * abs(self.F) * L_TOTAL)

    def test_moment_at_clamp(self):
        """|M|_encastrement = F·L (solution RDM)."""
        M = self.diag.forces["M"]
        np.testing.assert_allclose(abs(M[0]), abs(self.F) * L_TOTAL, rtol=1e-9)

    def test_moment_free_end_is_zero(self):
        """M = 0 à l'extrémité libre."""
        M = self.diag.forces["M"]
        assert abs(M[-1]) < 1e-6 * abs(self.F) * L_TOTAL

    def test_normal_force_zero(self):
        """Aucun effort axial → N = 0."""
        N = self.diag.forces["N"]
        assert np.allclose(N, 0.0, atol=1e-6 * abs(self.F))


# ---------------------------------------------------------------------------
# Charge répartie : V linéaire, M parabolique
# ---------------------------------------------------------------------------


class TestCantileverDistributedLoad:
    """Console + charge répartie q uniforme."""

    def setup_method(self):
        self.q = -5000.0  # N/m (vers le bas)
        # Un seul élément : la formulation Hermite est exacte aux nœuds.
        self.mesh, self.bc, self.u = _build_cantilever(1, qy=self.q)
        self.diag = extract_beam_diagrams(
            self.mesh, self.bc, self.u, n_points=41
        )[0]

    def test_shear_is_linear(self):
        """V(x) linéaire → dérivée seconde nulle, première constante."""
        V = self.diag.forces["V"]
        x = self.diag.x_local
        d1 = np.diff(V) / np.diff(x)
        np.testing.assert_allclose(d1, d1[0], rtol=1e-9)

    def test_shear_at_clamp(self):
        """|V|_encastrement = q·L (effort tranchant = charge totale)."""
        V = self.diag.forces["V"]
        np.testing.assert_allclose(
            abs(V[0]), abs(self.q) * L_TOTAL, rtol=1e-9
        )

    def test_shear_free_end_is_zero(self):
        """V = 0 à l'extrémité libre."""
        V = self.diag.forces["V"]
        assert abs(V[-1]) < 1e-6 * abs(self.q) * L_TOTAL

    def test_moment_is_parabolic(self):
        """M(x) = q·(L−x)²/2 : compare au profil analytique exact."""
        M = self.diag.forces["M"]
        x = self.diag.x_local
        M_analytical = self.q * (L_TOTAL - x) ** 2 / 2.0
        # Tolérance relative à l'amplitude max du moment.
        scale = abs(self.q) * L_TOTAL**2 / 2.0
        np.testing.assert_allclose(np.abs(M), np.abs(M_analytical),
                                   atol=1e-8 * scale)

    def test_moment_at_clamp(self):
        """|M|_encastrement = q·L²/2 (solution RDM)."""
        M = self.diag.forces["M"]
        np.testing.assert_allclose(
            abs(M[0]), abs(self.q) * L_TOTAL**2 / 2.0, rtol=1e-9
        )

    def test_moment_curvature_nonzero(self):
        """Le moment est réellement parabolique (courbure ≠ 0)."""
        M = self.diag.forces["M"]
        x = self.diag.x_local
        d2 = np.diff(M, 2) / np.diff(x)[0] ** 2
        # d²M/dx² = q (constante non nulle).
        np.testing.assert_allclose(d2, self.q, rtol=1e-6)


# ---------------------------------------------------------------------------
# Effort normal
# ---------------------------------------------------------------------------


class TestCantileverAxial:
    """Charge axiale en bout → effort normal constant."""

    def test_normal_force_constant_equals_load(self):
        """N(x) = F_axial constant (traction)."""
        Fx = 3000.0
        mesh, bc, u = _build_cantilever(1, fx_tip=Fx)
        diag = extract_beam_diagrams(mesh, bc, u)[0]
        N = diag.forces["N"]
        np.testing.assert_allclose(N, N[0], rtol=1e-10)
        np.testing.assert_allclose(abs(N[0]), Fx, rtol=1e-9)


# ---------------------------------------------------------------------------
# Continuité multi-éléments
# ---------------------------------------------------------------------------


class TestMultiElementContinuity:
    """Console à 4 éléments : les diagrammes restent corrects par morceaux."""

    def test_moment_continuous_across_elements(self):
        """Le moment au raccord de deux éléments coïncide."""
        q = -2000.0
        mesh, bc, u = _build_cantilever(4, qy=q)
        diagrams = extract_beam_diagrams(mesh, bc, u, n_points=11)
        assert len(diagrams) == 4
        # Fin de l'élément i ≈ début de l'élément i+1.
        for i in range(3):
            M_end = diagrams[i].forces["M"][-1]
            M_start = diagrams[i + 1].forces["M"][0]
            np.testing.assert_allclose(M_end, M_start, atol=1e-6 * abs(q))

    def test_global_moment_matches_analytical(self):
        """Sur 4 éléments, le moment global suit q·(L−s)²/2."""
        q = -2000.0
        mesh, bc, u = _build_cantilever(4, qy=q)
        diagrams = extract_beam_diagrams(mesh, bc, u, n_points=11)
        s_offset = 0.0
        scale = abs(q) * L_TOTAL**2 / 2.0
        for diag in diagrams:
            s = s_offset + diag.x_local
            M_analytical = q * (L_TOTAL - s) ** 2 / 2.0
            np.testing.assert_allclose(
                np.abs(diag.forces["M"]), np.abs(M_analytical),
                atol=1e-7 * scale,
            )
            s_offset += diag.length


# ---------------------------------------------------------------------------
# Poutre 3D
# ---------------------------------------------------------------------------


class TestBeam3DDiagram:
    """Console 3D : effort tranchant constant, moment linéaire."""

    def _build(self, fy_tip: float):
        from femsolver.core.sections import RectangularSection
        mat = ElasticMaterial(E=E, nu=NU, rho=RHO)
        nodes = np.array([[0.0, 0.0, 0.0], [L_TOTAL, 0.0, 0.0]])
        sec = RectangularSection(width=0.05, height=0.05)
        props = {"section": sec}
        elements = [ElementData(etype=Beam3D, node_ids=(0, 1),
                                material=mat, properties=props)]
        mesh = Mesh(nodes=nodes, elements=elements, n_dim=3, dof_per_node=6)
        # Encastrement complet au nœud 0 (6 DDL).
        # Charge globale +y : pour une poutre selon +x, le repère local par
        # défaut donne ŷ_local = ẑ_global et ẑ_local = −ŷ_global, donc une
        # charge globale y sollicite l'effort tranchant local Vz et My.
        dirichlet = {0: {d: 0.0 for d in range(6)}}
        bc = BoundaryConditions(dirichlet=dirichlet,
                                neumann={1: {1: fy_tip}})
        assembler = Assembler(mesh)
        K = assembler.assemble_stiffness()
        F = assembler.assemble_forces(bc)
        ds = apply_dirichlet(K, F, mesh, bc)
        u = StaticSolver().solve(ds.K_free, ds.F_free)
        return mesh, bc, ds.recover(u)

    def test_has_3d_components(self):
        """Le diagramme 3D expose Vz, My et T."""
        mesh, bc, u = self._build(fy_tip=-1500.0)
        diag = extract_beam_diagrams(mesh, bc, u)[0]
        assert set(diag.components) == {"N", "Vy", "Vz", "T", "My", "Mz"}

    def test_transverse_shear_constant(self):
        """Charge ponctuelle transverse → Vz constant = |F|."""
        Fy = -1500.0
        mesh, bc, u = self._build(fy_tip=Fy)
        diag = extract_beam_diagrams(mesh, bc, u)[0]
        Vz = diag.forces["Vz"]
        np.testing.assert_allclose(Vz, Vz[0], rtol=1e-9)
        np.testing.assert_allclose(np.abs(Vz), abs(Fy), rtol=1e-7)

    def test_bending_moment_linear_at_clamp(self):
        """|My|_encastrement = F·L, M = 0 au bout libre."""
        Fy = -1500.0
        mesh, bc, u = self._build(fy_tip=Fy)
        diag = extract_beam_diagrams(mesh, bc, u)[0]
        My = diag.forces["My"]
        np.testing.assert_allclose(abs(My[0]), abs(Fy) * L_TOTAL, rtol=1e-6)
        assert abs(My[-1]) < 1e-5 * abs(Fy) * L_TOTAL


# ---------------------------------------------------------------------------
# Tracé Matplotlib
# ---------------------------------------------------------------------------


class TestPlotting:
    """Le tracé produit une figure sans erreur."""

    def test_plot_returns_figure(self, tmp_path):
        mesh, bc, u = _build_cantilever(2, qy=-3000.0)
        diagrams = extract_beam_diagrams(mesh, bc, u)
        out = tmp_path / "diagrams.png"
        fig = plot_beam_diagrams(
            diagrams, title="Console", show=False, savefig=str(out)
        )
        assert out.exists()
        # 2D → 3 panneaux (N, V, M).
        assert len(fig.axes) == 3

    def test_plot_empty_raises(self):
        with pytest.raises(ValueError, match="poutre"):
            plot_beam_diagrams([], show=False)

    def test_non_beam_mesh_returns_empty(self):
        """Un maillage sans poutre donne une liste vide."""
        from femsolver.elements.bar2d import Bar2D
        mat = ElasticMaterial(E=E, nu=NU, rho=RHO)
        nodes = np.array([[0.0, 0.0], [1.0, 0.0]])
        elements = [ElementData(etype=Bar2D, node_ids=(0, 1),
                                material=mat, properties={"area": AREA})]
        mesh = Mesh(nodes=nodes, elements=elements, n_dim=2)
        bc = BoundaryConditions(dirichlet={0: {0: 0.0, 1: 0.0}}, neumann={})
        u = np.zeros(mesh.n_dof)
        assert extract_beam_diagrams(mesh, bc, u) == []

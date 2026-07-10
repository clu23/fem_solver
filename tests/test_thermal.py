"""Tests du chargement thermomécanique (déformations initiales ΔT).

Solutions analytiques de référence
----------------------------------
**Barre encastrée-encastrée chauffée** — la dilatation est empêchée, donc
ε = 0 et la contrainte est purement thermique :

    σ = -E · α · ΔT      (compression, indépendante de ν en 1D)

**Barre libre chauffée** — la dilatation est libre, donc la contrainte est
nulle et l'allongement vaut :

    δ = α · ΔT · L        (σ = 0)

Ces deux cas sont reproduits **exactement** (rtol 1e-10) par les éléments
isoparamétriques car le champ de déplacement de dilatation libre (u = αΔT·x,
v = αΔT(1+ν)·y) est linéaire, donc représentable sans erreur.
"""

import numpy as np
import pytest

from femsolver.core.assembler import Assembler
from femsolver.core.boundary import apply_dirichlet
from femsolver.core.material import ElasticMaterial
from femsolver.core.mesh import BoundaryConditions, ElementData, Mesh
from femsolver.core.solver import StaticSolver
from femsolver.elements.bar2d import Bar2D
from femsolver.elements.hexa8 import Hexa8
from femsolver.elements.quad4 import Quad4
from femsolver.elements.tetra4 import Tetra4
from femsolver.elements.tri3 import Tri3
from femsolver.io.json_model import load_model, run_from_json

E = 210e9
NU = 0.3
RHO = 7850.0
ALPHA = 1.2e-5
STEEL = ElasticMaterial(E=E, nu=NU, rho=RHO, alpha=ALPHA)


def _solve_thermal(mesh, bc, delta_T):
    """Résout K u = F_th (chargement purement thermique) et retourne u."""
    assembler = Assembler(mesh)
    K = assembler.assemble_stiffness()
    F = assembler.assemble_thermal_forces(delta_T)
    ds = apply_dirichlet(K, F, mesh, bc)
    u = StaticSolver().solve(ds.K_free, ds.F_free)
    return ds.recover(u)


# ===========================================================================
# Matériau
# ===========================================================================

class TestMaterialAlpha:
    """Coefficient de dilatation thermique sur ElasticMaterial."""

    def test_default_alpha_is_zero(self):
        mat = ElasticMaterial(E=E, nu=NU, rho=RHO)
        assert mat.alpha == 0.0

    def test_alpha_stored(self):
        assert STEEL.alpha == ALPHA

    def test_negative_alpha_rejected(self):
        with pytest.raises(ValueError, match="dilatation"):
            ElasticMaterial(E=E, nu=NU, rho=RHO, alpha=-1e-5)


# ===========================================================================
# Quad4 — barre libre et encastrée-encastrée
# ===========================================================================

class TestQuad4FreeExpansion:
    """Carré unité Quad4 libre de se dilater — σ = 0, δ = αΔT·L."""

    def setup_method(self):
        self.dT = 80.0
        nodes = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
        elem = ElementData(Quad4, (0, 1, 2, 3), STEEL, {"thickness": 0.01})
        self.mesh = Mesh(nodes=nodes, elements=(elem,), n_dim=2)
        # Appui isostatique : 0(ux,uy), 1(uy), 3(ux) — dilatation libre.
        bc = BoundaryConditions(
            dirichlet={0: {0: 0.0, 1: 0.0}, 1: {1: 0.0}, 3: {0: 0.0}},
            neumann={},
        )
        self.u = _solve_thermal(self.mesh, bc, self.dT)

    def test_tip_displacement(self):
        """δ = αΔT·L à l'extrémité chargée (nœuds x=1)."""
        delta = ALPHA * self.dT * 1.0
        # Nœud 1 (1,0) : ux = δ
        np.testing.assert_allclose(self.u[2], delta, rtol=1e-10)
        # Nœud 2 (1,1) : ux = δ
        np.testing.assert_allclose(self.u[4], delta, rtol=1e-10)

    def test_transverse_expansion(self):
        """v = αΔT·y à y=1 (dilatation isotrope, libre)."""
        delta = ALPHA * self.dT * 1.0
        np.testing.assert_allclose(self.u[5], delta, rtol=1e-10)  # nœud 2 uy
        np.testing.assert_allclose(self.u[7], delta, rtol=1e-10)  # nœud 3 uy

    def test_zero_stress(self):
        """Contrainte nulle : ε mécanique = ε thermique."""
        elem = Quad4()
        u_e = self.u  # 8 ddl = tout le maillage (1 élément)
        sigma = elem.stress(STEEL, self.mesh.nodes, u_e, delta_T=self.dT)
        np.testing.assert_allclose(sigma, np.zeros(3), atol=1e-3)


class TestQuad4Clamped:
    """Carré unité Quad4 bloqué en x aux deux bords — σxx = -EαΔT."""

    def setup_method(self):
        self.dT = 100.0
        nodes = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
        elem = ElementData(Quad4, (0, 1, 2, 3), STEEL, {"thickness": 0.01})
        self.mesh = Mesh(nodes=nodes, elements=(elem,), n_dim=2)
        # ux = 0 sur les 4 nœuds (bords gauche 0,3 et droit 1,2) → εxx = 0.
        # uy = 0 sur le bord bas (0,1) → ancrage, dilatation y libre.
        bc = BoundaryConditions(
            dirichlet={
                0: {0: 0.0, 1: 0.0}, 1: {0: 0.0, 1: 0.0},
                2: {0: 0.0}, 3: {0: 0.0},
            },
            neumann={},
        )
        self.u = _solve_thermal(self.mesh, bc, self.dT)

    def test_compressive_stress(self):
        """σxx = -E·α·ΔT (compression), σyy ≈ 0, τxy ≈ 0."""
        sigma = Quad4().stress(STEEL, self.mesh.nodes, self.u, delta_T=self.dT)
        expected = -E * ALPHA * self.dT
        np.testing.assert_allclose(sigma[0], expected, rtol=1e-10)
        np.testing.assert_allclose(sigma[1], 0.0, atol=1e-3)
        np.testing.assert_allclose(sigma[2], 0.0, atol=1e-3)

    def test_axial_displacement_zero(self):
        """ux = 0 partout (dilatation axiale totalement empêchée)."""
        np.testing.assert_allclose(self.u[0::2], 0.0, atol=1e-18)

    def test_free_transverse_displacement(self):
        """v = αΔT(1+ν)·y à y=1 (état uniaxial : σyy=0)."""
        expected = ALPHA * self.dT * (1.0 + NU) * 1.0
        np.testing.assert_allclose(self.u[5], expected, rtol=1e-10)  # nœud 2
        np.testing.assert_allclose(self.u[7], expected, rtol=1e-10)  # nœud 3


# ===========================================================================
# Tri3 — barre encastrée-encastrée (deux triangles)
# ===========================================================================

class TestTri3Clamped:
    """Carré unité maillé en 2 Tri3, bloqué en x → σxx = -EαΔT."""

    def setup_method(self):
        self.dT = 100.0
        nodes = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
        props = {"thickness": 0.01}
        elems = (
            ElementData(Tri3, (0, 1, 2), STEEL, props),
            ElementData(Tri3, (0, 2, 3), STEEL, props),
        )
        self.mesh = Mesh(nodes=nodes, elements=elems, n_dim=2)
        bc = BoundaryConditions(
            dirichlet={
                0: {0: 0.0, 1: 0.0}, 1: {0: 0.0, 1: 0.0},
                2: {0: 0.0}, 3: {0: 0.0},
            },
            neumann={},
        )
        self.u = _solve_thermal(self.mesh, bc, self.dT)

    def test_compressive_stress(self):
        """σxx = -E·α·ΔT sur chaque triangle."""
        expected = -E * ALPHA * self.dT
        for nid in [(0, 1, 2), (0, 2, 3)]:
            coords = self.mesh.node_coords(nid)
            dofs = self.mesh.global_dofs(nid)
            sigma = Tri3().stress(STEEL, coords, self.u[dofs], delta_T=self.dT)
            np.testing.assert_allclose(sigma[0], expected, rtol=1e-10)
            np.testing.assert_allclose(sigma[1], 0.0, atol=1e-2)


# ===========================================================================
# Hexa8 — cube libre et bloqué uniaxialement
# ===========================================================================

class TestHexa8FreeExpansion:
    """Cube unité Hexa8 libre — σ = 0, δ = αΔT·L."""

    def setup_method(self):
        self.dT = 60.0
        nodes = np.array([
            [0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.],
            [0., 0., 1.], [1., 0., 1.], [1., 1., 1.], [0., 1., 1.],
        ])
        elem = ElementData(Hexa8, tuple(range(8)), STEEL, {})
        self.mesh = Mesh(nodes=nodes, elements=(elem,), n_dim=3)
        # Appui isostatique 3-2-1 : 0(x,y,z), 1(y,z), 3(z) — dilatation libre.
        bc = BoundaryConditions(
            dirichlet={0: {0: 0.0, 1: 0.0, 2: 0.0}, 1: {1: 0.0, 2: 0.0},
                       3: {2: 0.0}},
            neumann={},
        )
        self.u = _solve_thermal(self.mesh, bc, self.dT)

    def test_tip_displacement(self):
        """ux = αΔT·L au nœud 1 (1,0,0)."""
        delta = ALPHA * self.dT * 1.0
        np.testing.assert_allclose(self.u[3], delta, rtol=1e-10)  # nœud 1 ux

    def test_zero_stress(self):
        sigma = Hexa8().stress(STEEL, self.mesh.nodes, self.u, delta_T=self.dT)
        np.testing.assert_allclose(sigma, np.zeros(6), atol=1e-2)


class TestHexa8Clamped:
    """Cube Hexa8 bloqué en x aux deux faces — σxx = -EαΔT, σyy=σzz≈0."""

    def setup_method(self):
        self.dT = 100.0
        nodes = np.array([
            [0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.],
            [0., 0., 1.], [1., 0., 1.], [1., 1., 1.], [0., 1., 1.],
        ])
        elem = ElementData(Hexa8, tuple(range(8)), STEEL, {})
        self.mesh = Mesh(nodes=nodes, elements=(elem,), n_dim=3)
        # ux=0 sur toutes les faces x (8 nœuds) → εxx=0.
        # uy=0 sur la face y=0 (0,1,4,5), uz=0 sur la face z=0 (0,1,2,3).
        dirichlet = {i: {0: 0.0} for i in range(8)}
        for i in (0, 1, 4, 5):
            dirichlet[i][1] = 0.0
        for i in (0, 1, 2, 3):
            dirichlet[i][2] = 0.0
        bc = BoundaryConditions(dirichlet=dirichlet, neumann={})
        self.u = _solve_thermal(self.mesh, bc, self.dT)

    def test_uniaxial_compressive_stress(self):
        """σxx = -E·α·ΔT ; σyy, σzz ≈ 0 (faces latérales libres)."""
        sigma = Hexa8().stress(STEEL, self.mesh.nodes, self.u, delta_T=self.dT)
        expected = -E * ALPHA * self.dT
        np.testing.assert_allclose(sigma[0], expected, rtol=1e-9)
        np.testing.assert_allclose(sigma[1], 0.0, atol=1e-1)
        np.testing.assert_allclose(sigma[2], 0.0, atol=1e-1)


# ===========================================================================
# Champ ΔT nodal (interpolé) vs uniforme
# ===========================================================================

class TestNodalTemperatureField:
    """ΔT uniforme via un scalaire == ΔT nodal constant via un tableau."""

    def setup_method(self):
        nodes = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
        elem = ElementData(Quad4, (0, 1, 2, 3), STEEL, {"thickness": 0.01})
        self.mesh = Mesh(nodes=nodes, elements=(elem,), n_dim=2)

    def test_uniform_scalar_equals_constant_field(self):
        dT = 50.0
        F_scalar = Assembler(self.mesh).assemble_thermal_forces(dT)
        F_field = Assembler(self.mesh).assemble_thermal_forces(np.full(4, dT))
        np.testing.assert_allclose(F_scalar, F_field, rtol=1e-12)

    def test_zero_temperature_zero_force(self):
        F = Assembler(self.mesh).assemble_thermal_forces(0.0)
        np.testing.assert_allclose(F, 0.0, atol=1e-18)

    def test_nodal_field_wrong_length_raises(self):
        with pytest.raises(ValueError, match="ΔT|nœud"):
            Assembler(self.mesh).assemble_thermal_forces(np.array([1.0, 2.0]))

    def test_linear_gradient_force_balance(self):
        """Un gradient linéaire de ΔT produit des forces auto-équilibrées."""
        # ΔT nodal croissant : les forces internes thermiques s'annulent
        # globalement (Σ F = 0) car il n'y a pas de force extérieure réelle.
        dT_nodes = np.array([0.0, 100.0, 100.0, 0.0])
        F = Assembler(self.mesh).assemble_thermal_forces(dT_nodes)
        np.testing.assert_allclose(F.reshape(-1, 2).sum(axis=0), 0.0, atol=1e-6)


# ===========================================================================
# Éléments non thermiques
# ===========================================================================

class TestUnsupportedElements:
    """Les éléments structuraux (Bar2D…) ne portent pas de charge thermique."""

    def test_bar2d_raises(self):
        nodes = np.array([[0., 0.], [1., 0.]])
        with pytest.raises(NotImplementedError, match="thermique"):
            Bar2D().thermal_force_vector(STEEL, nodes, {"area": 1e-4}, 50.0)

    def test_assembler_skips_non_thermal(self):
        """Un maillage de Bar2D produit un F_th nul (éléments ignorés)."""
        nodes = np.array([[0., 0.], [1., 0.]])
        elem = ElementData(Bar2D, (0, 1), STEEL, {"area": 1e-4})
        mesh = Mesh(nodes=nodes, elements=(elem,), n_dim=2)
        F = Assembler(mesh).assemble_thermal_forces(50.0)
        np.testing.assert_allclose(F, 0.0, atol=1e-18)


# ===========================================================================
# Tetra4
# ===========================================================================

class TestTetra4Clamped:
    """Tétraèdre bloqué en x sur la face x=0 et au sommet x>0 → σxx<0."""

    def test_free_expansion_zero_stress(self):
        """Tétraèdre libre : dilatation isotrope, σ = 0."""
        dT = 70.0
        nodes = np.array([[0., 0., 0.], [1., 0., 0.],
                          [0., 1., 0.], [0., 0., 1.]])
        elem = ElementData(Tetra4, (0, 1, 2, 3), STEEL, {})
        mesh = Mesh(nodes=nodes, elements=(elem,), n_dim=3)
        # Appui isostatique : 0(x,y,z), 1(y,z), 2(z).
        bc = BoundaryConditions(
            dirichlet={0: {0: 0.0, 1: 0.0, 2: 0.0}, 1: {1: 0.0, 2: 0.0},
                       2: {2: 0.0}},
            neumann={},
        )
        u = _solve_thermal(mesh, bc, dT)
        sigma = Tetra4().stress(STEEL, nodes, u, delta_T=dT)
        np.testing.assert_allclose(sigma, np.zeros(6), atol=1e-2)
        # ux = αΔT·L au nœud 1
        np.testing.assert_allclose(u[3], ALPHA * dT, rtol=1e-10)


# ===========================================================================
# Intégration JSON
# ===========================================================================

class TestJsonThermal:
    """Pipeline JSON complet : exemple barre encastrée-encastrée."""

    PATH = "examples/thermal_clamped_bar.json"

    def test_runs_and_axial_displacement_zero(self):
        results = run_from_json(self.PATH, verbose=False)
        u = np.array(results["u"])
        # ux ≈ 0 partout (dilatation empêchée)
        np.testing.assert_allclose(u[0::2], 0.0, atol=1e-12)

    def test_recovered_stress_is_compressive(self):
        """Recalcule σxx depuis u et vérifie σxx = -EαΔT."""
        model = load_model(self.PATH)
        results = run_from_json(self.PATH, verbose=False)
        u = np.array(results["u"])
        dT = model.analysis["thermal"]["delta_T"]
        ed = model.mesh.elements[0]
        coords = model.mesh.node_coords(ed.node_ids)
        dofs = model.mesh.global_dofs(ed.node_ids)
        sigma = Quad4().stress(ed.material, coords, u[dofs], delta_T=dT)
        expected = -ed.material.E * ed.material.alpha * dT
        np.testing.assert_allclose(sigma[0], expected, rtol=1e-9)

    def test_json_nodal_field(self):
        """Champ ΔT par nœud accepté et équivalent au scalaire uniforme."""
        import json
        import tempfile
        from pathlib import Path

        with open(self.PATH) as fh:
            data = json.load(fh)
        n_nodes = len(data["nodes"])
        data["analysis"]["thermal"] = {"delta_T_nodes": [100.0] * n_nodes}
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "model.json"
            p.write_text(json.dumps(data))
            results = run_from_json(p, verbose=False)
        u = np.array(results["u"])
        np.testing.assert_allclose(u[0::2], 0.0, atol=1e-12)

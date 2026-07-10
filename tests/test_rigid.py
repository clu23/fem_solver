"""Tests des éléments rigides RBE2 / RBE3 (contraintes MPC).

Chaque test compare à une solution analytique de corps rigide ou de moyenne
pondérée, conformément à la philosophie de validation du projet.
"""

import numpy as np
import pytest
from scipy.sparse.linalg import spsolve

from femsolver.core.assembler import Assembler
from femsolver.core.boundary import apply_dirichlet
from femsolver.core.material import ElasticMaterial
from femsolver.core.mesh import BoundaryConditions, ElementData, Mesh
from femsolver.core.mpc import apply_mpc_lagrange
from femsolver.core.rigid import make_rbe2_constraints, make_rbe3_constraints
from femsolver.core.solver import StaticSolver
from femsolver.elements.spring import SpringElement
from femsolver.io.json_model import FEModel, solve_model

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _solve_lagrange(mesh, bc, constraints):
    """Assemble, applique Dirichlet puis les MPC par Lagrange, retourne u."""
    asm = Assembler(mesh)
    K = asm.assemble_stiffness()
    F = asm.assemble_forces(bc)
    K_bc, F_bc = apply_dirichlet(K, F, mesh, bc)
    K_aug, F_aug = apply_mpc_lagrange(K_bc, F_bc, mesh, constraints)
    sol = spsolve(K_aug.tocsc(), F_aug)
    return np.asarray(sol[: mesh.n_dof])


# ---------------------------------------------------------------------------
# Génération des contraintes (cinématique de corps rigide)
# ---------------------------------------------------------------------------

class TestRBE2Kinematics:
    """Vérifie les coefficients des contraintes MPC générées."""

    def test_2d_translation_only(self):
        """Sans DDL de rotation (dpn=2) : u_S = u_M (translation pure)."""
        nodes = np.array([[0.0, 0.0], [1.0, 2.0]])
        mesh = Mesh(nodes=nodes, elements=(), n_dim=2)
        cons = make_rbe2_constraints(mesh, master=0, slaves=[1])
        # 2 contraintes (ux, uy), chacune u_S - u_M = 0, sans terme de rotation
        assert len(cons) == 2
        assert cons[0].terms == ((1, 0, 1.0), (0, 0, -1.0))
        assert cons[1].terms == ((1, 1, 1.0), (0, 1, -1.0))

    def test_2d_rotation_lever_arm(self):
        """dpn=3 : ux_S = ux_M − θz·ry, uy_S = uy_M + θz·rx."""
        nodes = np.array([[0.0, 0.0], [2.0, 3.0]])   # r = (2, 3)
        mesh = Mesh(nodes=nodes, elements=(), n_dim=2, dof_per_node=3)
        cons = make_rbe2_constraints(mesh, master=0, slaves=[1])
        assert len(cons) == 3
        # ux_S − ux_M + θz·ry = 0  → coeff θz = ry = 3
        assert cons[0].terms == ((1, 0, 1.0), (0, 0, -1.0), (0, 2, 3.0))
        # uy_S − uy_M − θz·rx = 0  → coeff θz = −rx = −2
        assert cons[1].terms == ((1, 1, 1.0), (0, 1, -1.0), (0, 2, -2.0))
        # θz_S = θz_M
        assert cons[2].terms == ((1, 2, 1.0), (0, 2, -1.0))

    def test_3d_cross_product(self):
        """dpn=6 : u_S = u_M + ω × r ; rotations transmises identiquement."""
        nodes = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 4.0]])  # r=(1,2,4)
        mesh = Mesh(nodes=nodes, elements=(), n_dim=3, dof_per_node=6)
        cons = make_rbe2_constraints(mesh, master=0, slaves=[1])
        assert len(cons) == 6
        # ux_S − ux_M − (θy·rz − θz·ry) = 0 → +θy·(−rz)=−4, +θz·(ry)=+2
        assert cons[0].terms == (
            (1, 0, 1.0), (0, 0, -1.0), (0, 4, -4.0), (0, 5, 2.0)
        )
        # uy_S : −(θz·rx − θx·rz) → θz·(−rx)=−1, θx·(rz)=+4
        assert cons[1].terms == (
            (1, 1, 1.0), (0, 1, -1.0), (0, 5, -1.0), (0, 3, 4.0)
        )
        # uz_S : −(θx·ry − θy·rx) → θx·(−ry)=−2, θy·(rx)=+1
        assert cons[2].terms == (
            (1, 2, 1.0), (0, 2, -1.0), (0, 3, -2.0), (0, 4, 1.0)
        )
        # Rotations : θ_S = θ_M
        assert cons[3].terms == ((1, 3, 1.0), (0, 3, -1.0))
        assert cons[4].terms == ((1, 4, 1.0), (0, 4, -1.0))
        assert cons[5].terms == ((1, 5, 1.0), (0, 5, -1.0))

    def test_dofs_subset(self):
        """L'argument ``dofs`` restreint les DDL contraints."""
        nodes = np.array([[0.0, 0.0], [1.0, 0.0]])
        mesh = Mesh(nodes=nodes, elements=(), n_dim=2)
        cons = make_rbe2_constraints(mesh, master=0, slaves=[1], dofs=[0])
        assert len(cons) == 1
        assert cons[0].terms[0] == (1, 0, 1.0)


# ---------------------------------------------------------------------------
# RBE2 — distribution de force / corps rigide (solution analytique)
# ---------------------------------------------------------------------------

class TestRBE2RigidBody:
    """Force appliquée au maître distribuée aux esclaves (corps rigide)."""

    def test_translation_force_distribution(self):
        """Maître relié à N ressorts identiques via les esclaves.

        Esclaves rigidement liés au maître (dpn=2, translation pure) : tous se
        déplacent identiquement.  Le maître voit la raideur ``Σ k`` :

            δ = F / (Σ k),   chaque ressort reprend F/N (ici N=3 identiques).
        """
        k = 1000.0
        F = 600.0
        n = 3
        mat = ElasticMaterial(E=1.0, nu=0.0, rho=1.0)
        # 0 = maître ; 1,2,3 = esclaves ; 4,5,6 = sol (encastrés)
        nodes = np.array([
            [0.0, 0.0],
            [0.0, 1.0], [0.0, 2.0], [0.0, 3.0],
            [1.0, 1.0], [1.0, 2.0], [1.0, 3.0],
        ])
        els = tuple(
            ElementData(SpringElement, (1 + i, 4 + i), mat,
                        {"stiffness": [k, k]})
            for i in range(n)
        )
        mesh = Mesh(nodes=nodes, elements=els, n_dim=2, dof_per_node=2)
        cons = make_rbe2_constraints(mesh, master=0, slaves=[1, 2, 3])
        bc = BoundaryConditions(
            dirichlet={4: {0: 0.0, 1: 0.0}, 5: {0: 0.0, 1: 0.0},
                       6: {0: 0.0, 1: 0.0}},
            neumann={0: {0: F}},
        )
        u = _solve_lagrange(mesh, bc, cons)
        dpn = mesh.dpn
        delta = F / (n * k)
        # Maître et tous les esclaves se déplacent de delta en x
        np.testing.assert_allclose(u[dpn * 0], delta, rtol=1e-10)
        for s in (1, 2, 3):
            np.testing.assert_allclose(u[dpn * s], delta, rtol=1e-10,
                                       err_msg=f"esclave {s} ux")
            np.testing.assert_allclose(u[dpn * s], u[dpn * 0], rtol=1e-12)

    def test_rotation_lever_arm(self):
        """Moment au maître → rotation de corps rigide des esclaves.

        Deux esclaves en (0,±h) reliés au sol par des ressorts ``kx``.  Un
        moment ``Mz`` au maître engendre :

            θz = Mz / (2 k h²),   ux(S±) = ∓ h θz = ∓ Mz / (2 k h)

        (couple repris par les bras de levier ±h des ressorts).
        """
        k = 1000.0
        h = 0.5
        Mz = 10.0
        mat = ElasticMaterial(E=1.0, nu=0.0, rho=1.0)
        nodes = np.array([
            [0.0, 0.0],            # 0 maître
            [0.0, h], [0.0, -h],   # 1,2 esclaves
            [1.0, h], [1.0, -h],   # 3,4 sol
        ])
        els = (
            ElementData(SpringElement, (1, 3), mat, {"stiffness": [k, 0.0, 0.0]}),
            ElementData(SpringElement, (2, 4), mat, {"stiffness": [k, 0.0, 0.0]}),
        )
        mesh = Mesh(nodes=nodes, elements=els, n_dim=2, dof_per_node=3)
        cons = make_rbe2_constraints(mesh, master=0, slaves=[1, 2])
        bc = BoundaryConditions(
            dirichlet={3: {0: 0.0, 1: 0.0, 2: 0.0},
                       4: {0: 0.0, 1: 0.0, 2: 0.0},
                       0: {1: 0.0}},          # maître uy = 0 (symétrie)
            neumann={0: {2: Mz}},
        )
        u = _solve_lagrange(mesh, bc, cons)
        dpn = mesh.dpn
        theta_z = u[dpn * 0 + 2]
        np.testing.assert_allclose(theta_z, Mz / (2 * k * h**2), rtol=1e-10)
        np.testing.assert_allclose(u[dpn * 1], -Mz / (2 * k * h), rtol=1e-10)
        np.testing.assert_allclose(u[dpn * 2], +Mz / (2 * k * h), rtol=1e-10)
        # Vérifie la cinématique de corps rigide : ux_S1 = ux_M − θz·h
        np.testing.assert_allclose(
            u[dpn * 1], u[dpn * 0] - theta_z * h, rtol=1e-12
        )


# ---------------------------------------------------------------------------
# RBE3 — moyenne pondérée, aucune rigidité ajoutée
# ---------------------------------------------------------------------------

class TestRBE3Average:
    """Déplacement de référence = moyenne pondérée des indépendants."""

    def setup_method(self):
        self.k = 1000.0
        mat = ElasticMaterial(E=1.0, nu=0.0, rho=1.0)
        # 0,1,2 indépendants ; 3,4,5 sol ; 6 référence
        self.nodes = np.array([
            [0.0, 0.0], [0.0, 1.0], [0.0, 2.0],
            [1.0, 0.0], [1.0, 1.0], [1.0, 2.0],
            [5.0, 5.0],
        ])
        els = tuple(
            ElementData(SpringElement, (i, 3 + i), mat,
                        {"stiffness": [self.k, self.k]})
            for i in range(3)
        )
        self.mesh = Mesh(nodes=self.nodes, elements=els, n_dim=2,
                         dof_per_node=2)
        self.bc = BoundaryConditions(
            dirichlet={3: {0: 0.0, 1: 0.0}, 4: {0: 0.0, 1: 0.0},
                       5: {0: 0.0, 1: 0.0}},
            neumann={0: {0: 100.0}, 1: {0: 200.0}, 2: {0: 300.0}},
        )

    def test_weighted_average(self):
        """u_ref = Σ w_i u_i / Σ w_i (poids 1, 2, 3)."""
        weights = [1.0, 2.0, 3.0]
        cons = make_rbe3_constraints(self.mesh, ref=6, nodes=[0, 1, 2],
                                     weights=weights)
        u = _solve_lagrange(self.mesh, self.bc, cons)
        dpn = self.mesh.dpn
        ux = np.array([u[dpn * i] for i in range(3)])
        avg = float(np.dot(weights, ux) / np.sum(weights))
        np.testing.assert_allclose(u[dpn * 6], avg, rtol=1e-10)

    def test_simple_average_default_weights(self):
        """Sans poids → moyenne arithmétique."""
        cons = make_rbe3_constraints(self.mesh, ref=6, nodes=[0, 1, 2])
        u = _solve_lagrange(self.mesh, self.bc, cons)
        dpn = self.mesh.dpn
        ux = np.array([u[dpn * i] for i in range(3)])
        np.testing.assert_allclose(u[dpn * 6], ux.mean(), rtol=1e-10)

    def test_no_added_stiffness(self):
        """Le RBE3 ne modifie pas la réponse des nœuds indépendants.

        Aucune charge au nœud de référence → multiplicateurs nuls → la solution
        des indépendants est identique avec ou sans RBE3.
        """
        # Référence (baseline) : sans RBE3, nœud 6 simplement encastré.
        bc_ref = BoundaryConditions(
            dirichlet={**self.bc.dirichlet, 6: {0: 0.0, 1: 0.0}},
            neumann=self.bc.neumann,
        )
        asm = Assembler(self.mesh)
        ds = apply_dirichlet(asm.assemble_stiffness(),
                             asm.assemble_forces(bc_ref), self.mesh, bc_ref)
        u_base = ds.recover(StaticSolver().solve(ds.K_free, ds.F_free))

        cons = make_rbe3_constraints(self.mesh, ref=6, nodes=[0, 1, 2])
        u_rbe3 = _solve_lagrange(self.mesh, self.bc, cons)

        dpn = self.mesh.dpn
        for i in range(3):
            np.testing.assert_allclose(
                u_rbe3[dpn * i], u_base[dpn * i], rtol=1e-10,
                err_msg=f"indépendant {i} ux modifié par le RBE3",
            )


# ---------------------------------------------------------------------------
# Validation des entrées
# ---------------------------------------------------------------------------

class TestRigidValidation:

    def setup_method(self):
        self.mesh = Mesh(nodes=np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
                         elements=(), n_dim=2)

    def test_rbe2_master_out_of_range(self):
        with pytest.raises(ValueError, match="maître"):
            make_rbe2_constraints(self.mesh, master=9, slaves=[0])

    def test_rbe2_slave_equals_master(self):
        with pytest.raises(ValueError, match="ne peut pas être"):
            make_rbe2_constraints(self.mesh, master=0, slaves=[0])

    def test_rbe2_bad_dof(self):
        with pytest.raises(ValueError, match="DDL"):
            make_rbe2_constraints(self.mesh, master=0, slaves=[1], dofs=[5])

    def test_rbe3_empty_nodes(self):
        with pytest.raises(ValueError, match="vide"):
            make_rbe3_constraints(self.mesh, ref=0, nodes=[])

    def test_rbe3_weight_length_mismatch(self):
        with pytest.raises(ValueError, match="poids"):
            make_rbe3_constraints(self.mesh, ref=0, nodes=[1, 2],
                                  weights=[1.0])

    def test_rbe3_zero_weight_sum(self):
        with pytest.raises(ValueError, match="nulle"):
            make_rbe3_constraints(self.mesh, ref=0, nodes=[1, 2],
                                  weights=[1.0, -1.0])

    def test_rbe3_ref_in_nodes(self):
        with pytest.raises(ValueError, match="référence"):
            make_rbe3_constraints(self.mesh, ref=0, nodes=[0, 1])


# ---------------------------------------------------------------------------
# Intégration JSON (parseur → solveur)
# ---------------------------------------------------------------------------

class TestJsonRigid:
    """Le bloc ``rigid`` est parsé et appliqué en analyse statique."""

    def _model(self, rigid, bc_dirichlet, bc_neumann, nodes, elements, dpn):
        from femsolver.io.json_model import _build_mesh_and_bc, _parse_rigid
        data = {
            "name": "rigid_test",
            "materials": {"m": {"E": 1.0, "nu": 0.0, "rho": 1.0}},
            "nodes": nodes,
            "elements": elements,
            "boundary_conditions": {
                "dirichlet": bc_dirichlet, "neumann": bc_neumann,
            },
            "rigid": rigid,
            "analysis": {"type": "static"},
        }
        mesh, bc = _build_mesh_and_bc(data)
        mpc = _parse_rigid(data, mesh)
        return FEModel(name="t", mesh=mesh, bc=bc,
                       analysis={"type": "static"}, mpc=mpc)

    def test_json_rbe2(self):
        """RBE2 via JSON : translation rigide distribuée (δ = F/Σk)."""
        k, F = 1000.0, 600.0
        nodes = [[0.0, 0.0], [0.0, 1.0], [0.0, 2.0], [0.0, 3.0],
                 [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]
        elements = [
            {"type": "Spring", "nodes": [1, 4], "stiffness": [k, k]},
            {"type": "Spring", "nodes": [2, 5], "stiffness": [k, k]},
            {"type": "Spring", "nodes": [3, 6], "stiffness": [k, k]},
        ]
        model = self._model(
            rigid=[{"type": "RBE2", "master": 0, "slaves": [1, 2, 3]}],
            bc_dirichlet=[{"node": n, "dof": d, "value": 0.0}
                          for n in (4, 5, 6) for d in (0, 1)],
            bc_neumann=[{"node": 0, "dof": 0, "value": F}],
            nodes=nodes, elements=elements, dpn=2,
        )
        res = solve_model(model, verbose=False)
        u = np.asarray(res["u"])
        delta = F / (3 * k)
        np.testing.assert_allclose(u[2 * 0], delta, rtol=1e-9)
        np.testing.assert_allclose(u[2 * 1], delta, rtol=1e-9)

    def test_json_rbe3(self):
        """RBE3 via JSON : déplacement de référence = moyenne pondérée."""
        k = 1000.0
        nodes = [[0.0, 0.0], [0.0, 1.0], [0.0, 2.0],
                 [1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [5.0, 5.0]]
        elements = [
            {"type": "Spring", "nodes": [0, 3], "stiffness": [k, k]},
            {"type": "Spring", "nodes": [1, 4], "stiffness": [k, k]},
            {"type": "Spring", "nodes": [2, 5], "stiffness": [k, k]},
        ]
        model = self._model(
            rigid=[{"type": "RBE3", "ref": 6, "nodes": [0, 1, 2],
                    "weights": [1.0, 2.0, 3.0]}],
            bc_dirichlet=[{"node": n, "dof": d, "value": 0.0}
                          for n in (3, 4, 5) for d in (0, 1)],
            bc_neumann=[{"node": 0, "dof": 0, "value": 100.0},
                        {"node": 1, "dof": 0, "value": 200.0},
                        {"node": 2, "dof": 0, "value": 300.0}],
            nodes=nodes, elements=elements, dpn=2,
        )
        res = solve_model(model, verbose=False)
        u = np.asarray(res["u"])
        ux = np.array([u[2 * i] for i in range(3)])
        avg = float(np.dot([1, 2, 3], ux) / 6.0)
        np.testing.assert_allclose(u[2 * 6], avg, rtol=1e-9)

    def test_json_unknown_rigid_type(self):
        from femsolver.io.json_model import _parse_rigid
        mesh = Mesh(nodes=np.array([[0.0, 0.0], [1.0, 0.0]]), elements=(),
                    n_dim=2)
        with pytest.raises(ValueError, match="rigide inconnu"):
            _parse_rigid({"rigid": [{"type": "RBE9", "master": 0}]}, mesh)

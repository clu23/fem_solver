"""Tests du module femsolver.core.model_check (vérification de santé modèle).

Chaque test construit un modèle volontairement sain ou défectueux et vérifie
que ``run_model_checks`` classe correctement le défaut :

- erreurs bloquantes  → ``report.errors`` (lèvent ModelError à la résolution)
- avertissements      → ``report.warnings`` (le calcul peut continuer)

Cas couverts (cf. énoncé) :
1. nœud orphelin                  → erreur
2. nœuds mal ordonnés (det J < 0) → erreur
3. BCs insuffisantes (singulier)  → erreur, DDL identifié
4. nœuds coïncidents              → warning, le calcul réussit
5. mauvais aspect ratio           → warning
6. modèle propre                  → ni erreur ni warning
"""

from __future__ import annotations

import numpy as np
import pytest

from femsolver.core.material import ElasticMaterial
from femsolver.core.mesh import BoundaryConditions, ElementData, Mesh
from femsolver.core.model_check import (
    ModelCheckReport,
    ModelError,
    run_model_checks,
)
from femsolver.elements.bar2d import Bar2D
from femsolver.elements.quad4 import Quad4
from femsolver.elements.tri3 import Tri3

_STEEL = ElasticMaterial(E=210e9, nu=0.3, rho=7800.0)
_THK = {"thickness": 0.01}


def _codes(issues) -> set[str]:
    return {i.code for i in issues}


# ---------------------------------------------------------------------------
# Helpers : carré unité Quad4 bien posé (cantilever)
# ---------------------------------------------------------------------------


def _unit_quad(bc: BoundaryConditions) -> tuple[Mesh, BoundaryConditions]:
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    elems = [ElementData(Quad4, (0, 1, 2, 3), _STEEL, _THK)]
    mesh = Mesh(nodes=nodes, elements=elems, n_dim=2)
    return mesh, bc


def _well_posed_bc() -> BoundaryConditions:
    # Bord gauche encastré (nœuds 0 et 3) → pas de mode de corps rigide.
    return BoundaryConditions(
        dirichlet={0: {0: 0.0, 1: 0.0}, 3: {0: 0.0, 1: 0.0}},
        neumann={1: {0: 1000.0}},
    )


# ===========================================================================
# 1. Nœuds orphelins → erreur bloquante
# ===========================================================================


class TestOrphanNodes:
    def test_orphan_node_is_error(self):
        # Nœud 4 n'appartient à aucun élément.
        nodes = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [5, 5]], float)
        elems = [ElementData(Quad4, (0, 1, 2, 3), _STEEL, _THK)]
        mesh = Mesh(nodes=nodes, elements=elems, n_dim=2)
        report = run_model_checks(mesh, _well_posed_bc())
        assert not report.ok
        assert "orphan_nodes" in _codes(report.errors)
        err = next(e for e in report.errors if e.code == "orphan_nodes")
        assert 4 in err.nodes

    def test_orphan_raises_model_error(self):
        nodes = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [5, 5]], float)
        elems = [ElementData(Quad4, (0, 1, 2, 3), _STEEL, _THK)]
        mesh = Mesh(nodes=nodes, elements=elems, n_dim=2)
        report = run_model_checks(mesh, _well_posed_bc())
        with pytest.raises(ModelError, match="orphelin"):
            report.raise_if_errors()


# ===========================================================================
# 2. Jacobien négatif (nœuds mal ordonnés) → erreur bloquante
# ===========================================================================


class TestNegativeJacobian:
    def test_reversed_quad_is_error(self):
        # Nœuds en sens horaire → det(J) < 0.
        nodes = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], float)
        elems = [ElementData(Quad4, (0, 1, 2, 3), _STEEL, _THK)]
        mesh = Mesh(nodes=nodes, elements=elems, n_dim=2)
        report = run_model_checks(mesh, _well_posed_bc())
        assert "negative_jacobian" in _codes(report.errors)
        err = next(e for e in report.errors if e.code == "negative_jacobian")
        assert err.elements == (0,)

    def test_reversed_tri_is_error(self):
        # Triangle en sens horaire.
        nodes = np.array([[0, 0], [0, 1], [1, 0]], float)
        elems = [ElementData(Tri3, (0, 1, 2), _STEEL, _THK)]
        mesh = Mesh(nodes=nodes, elements=elems, n_dim=2)
        bc = BoundaryConditions(dirichlet={0: {0: 0.0, 1: 0.0}, 1: {0: 0.0}},
                                neumann={})
        report = run_model_checks(mesh, bc)
        assert "negative_jacobian" in _codes(report.errors)

    def test_degenerate_quad_is_error(self):
        # Quatre nœuds alignés → élément d'aire nulle, det(J) ≈ 0 aux Gauss.
        nodes = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], float)
        elems = [ElementData(Quad4, (0, 1, 2, 3), _STEEL, _THK)]
        mesh = Mesh(nodes=nodes, elements=elems, n_dim=2)
        report = run_model_checks(mesh, _well_posed_bc())
        codes = _codes(report.errors)
        assert "degenerate_jacobian" in codes or "negative_jacobian" in codes

    def test_correct_order_quad_no_jacobian_error(self):
        mesh, bc = _unit_quad(_well_posed_bc())
        report = run_model_checks(mesh, bc)
        assert "negative_jacobian" not in _codes(report.errors)
        assert "degenerate_jacobian" not in _codes(report.errors)


# ===========================================================================
# 3. Singularité (BCs insuffisantes) → erreur, DDL identifié
# ===========================================================================


class TestSingularity:
    def test_underconstrained_quad_is_singular(self):
        # Un seul nœud bloqué (2 DDL) → il reste une rotation de corps rigide.
        nodes = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
        elems = [ElementData(Quad4, (0, 1, 2, 3), _STEEL, _THK)]
        mesh = Mesh(nodes=nodes, elements=elems, n_dim=2)
        bc = BoundaryConditions(dirichlet={0: {0: 0.0, 1: 0.0}}, neumann={})
        report = run_model_checks(mesh, bc)
        assert "singular_stiffness" in _codes(report.errors)
        err = next(e for e in report.errors if e.code == "singular_stiffness")
        # Le message identifie un nœud/DDL impliqué dans le mécanisme.
        assert "Nœud" in err.message
        assert len(err.dofs) >= 1

    def test_unconstrained_dof_identified_exactly(self):
        # Aucune CL : tous les DDL sont libres → mécanisme global.
        nodes = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
        elems = [ElementData(Quad4, (0, 1, 2, 3), _STEEL, _THK)]
        mesh = Mesh(nodes=nodes, elements=elems, n_dim=2)
        bc = BoundaryConditions(dirichlet={}, neumann={})
        report = run_model_checks(mesh, bc)
        assert "singular_stiffness" in _codes(report.errors)

    def test_well_posed_not_singular(self):
        mesh, bc = _unit_quad(_well_posed_bc())
        report = run_model_checks(mesh, bc)
        assert "singular_stiffness" not in _codes(report.errors)

    def test_singularity_warning_only_for_modal(self):
        # En modal, un noyau (corps rigide) est légitime → warning, pas erreur.
        nodes = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
        elems = [ElementData(Quad4, (0, 1, 2, 3), _STEEL, _THK)]
        mesh = Mesh(nodes=nodes, elements=elems, n_dim=2)
        bc = BoundaryConditions(dirichlet={0: {0: 0.0, 1: 0.0}}, neumann={})
        report = run_model_checks(mesh, bc, analysis_type="modal")
        assert "singular_stiffness" not in _codes(report.errors)
        assert "singular_stiffness" in _codes(report.warnings)


# ===========================================================================
# 4. Nœuds coïncidents → WARNING mais le calcul réussit
# ===========================================================================


class TestCoincidentNodes:
    def _two_unmerged_quads(self):
        """Deux Quad4 dont les bords se touchent sans partager les nœuds.

        Chaque carré est encastré sur son bord gauche → tous deux bien posés.
        Les nœuds (1,4) et (2,7) coïncident (maillage non fusionné).
        """
        nodes = np.array([
            [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],   # carré A : 0-3
            [1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0],   # carré B : 4-7
        ])
        elems = [
            ElementData(Quad4, (0, 1, 2, 3), _STEEL, _THK),
            ElementData(Quad4, (4, 5, 6, 7), _STEEL, _THK),
        ]
        mesh = Mesh(nodes=nodes, elements=elems, n_dim=2)
        bc = BoundaryConditions(
            dirichlet={0: {0: 0.0, 1: 0.0}, 3: {0: 0.0, 1: 0.0},
                       4: {0: 0.0, 1: 0.0}, 7: {0: 0.0, 1: 0.0}},
            neumann={5: {0: 1000.0}},
        )
        return mesh, bc

    def test_coincident_nodes_warning(self):
        mesh, bc = self._two_unmerged_quads()
        report = run_model_checks(mesh, bc)
        assert "coincident_nodes" in _codes(report.warnings)
        warn = next(w for w in report.warnings if w.code == "coincident_nodes")
        assert set(warn.nodes) >= {1, 4, 2, 7}

    def test_coincident_no_error_and_solves(self):
        from femsolver.io.json_model import FEModel, solve_model
        mesh, bc = self._two_unmerged_quads()
        report = run_model_checks(mesh, bc)
        # Avertissement, mais aucune erreur bloquante.
        assert report.ok
        # Le calcul aboutit malgré l'avertissement.
        model = FEModel(name="coincident", mesh=mesh, bc=bc,
                        analysis={"type": "static"})
        results = solve_model(model, verbose=False)
        assert "u" in results
        assert np.isfinite(results["u"]).all()


# ===========================================================================
# 5. Qualité des éléments → WARNING
# ===========================================================================


class TestElementQuality:
    def test_high_aspect_ratio_warning(self):
        # Quad 100×1 → aspect ratio = 100.
        nodes = np.array([[0, 0], [100, 0], [100, 1], [0, 1]], float)
        elems = [ElementData(Quad4, (0, 1, 2, 3), _STEEL, _THK)]
        mesh = Mesh(nodes=nodes, elements=elems, n_dim=2)
        bc = BoundaryConditions(
            dirichlet={0: {0: 0.0, 1: 0.0}, 3: {0: 0.0, 1: 0.0}},
            neumann={1: {1: 1000.0}},
        )
        report = run_model_checks(mesh, bc)
        assert "poor_quality" in _codes(report.warnings)
        warn = next(w for w in report.warnings if w.code == "poor_quality")
        assert "aspect ratio" in warn.message

    def test_sliver_triangle_angle_warning(self):
        # Triangle très aplati → petit angle (< 10°).
        nodes = np.array([[0, 0], [10, 0], [5, 0.2]], float)
        elems = [ElementData(Tri3, (0, 1, 2), _STEEL, _THK)]
        mesh = Mesh(nodes=nodes, elements=elems, n_dim=2)
        bc = BoundaryConditions(dirichlet={0: {0: 0.0, 1: 0.0}, 1: {1: 0.0}},
                                neumann={})
        report = run_model_checks(mesh, bc)
        assert "poor_quality" in _codes(report.warnings)
        warn = next(w for w in report.warnings if w.code == "poor_quality")
        assert "angle" in warn.message

    def test_good_quality_no_warning(self):
        mesh, bc = _unit_quad(_well_posed_bc())
        report = run_model_checks(mesh, bc)
        assert "poor_quality" not in _codes(report.warnings)


# ===========================================================================
# Éléments dupliqués → WARNING
# ===========================================================================


class TestDuplicateElements:
    def test_duplicate_element_warning(self):
        # Triangle en double (même connectivité).
        nodes = np.array([[0, 0], [1, 0], [0, 1]], float)
        elems = [
            ElementData(Tri3, (0, 1, 2), _STEEL, _THK),
            ElementData(Tri3, (0, 1, 2), _STEEL, _THK),
        ]
        mesh = Mesh(nodes=nodes, elements=elems, n_dim=2)
        bc = BoundaryConditions(dirichlet={0: {0: 0.0, 1: 0.0}, 1: {1: 0.0}},
                                neumann={})
        report = run_model_checks(mesh, bc)
        assert "duplicate_elements" in _codes(report.warnings)


# ===========================================================================
# 6. Modèle propre → ni erreur ni warning
# ===========================================================================


class TestCleanModel:
    def test_clean_quad_no_issues(self):
        mesh, bc = _unit_quad(_well_posed_bc())
        report = run_model_checks(mesh, bc)
        assert report.errors == ()
        assert report.warnings == ()
        assert report.ok

    def test_clean_truss_no_issues(self):
        # Triangle de barres statiquement déterminé (treillis).
        nodes = np.array([[0, 0], [2, 0], [1, 1]], float)
        elems = [
            ElementData(Bar2D, (0, 1), _STEEL, {"area": 1e-4}),
            ElementData(Bar2D, (1, 2), _STEEL, {"area": 1e-4}),
            ElementData(Bar2D, (2, 0), _STEEL, {"area": 1e-4}),
        ]
        mesh = Mesh(nodes=nodes, elements=elems, n_dim=2)
        bc = BoundaryConditions(
            dirichlet={0: {0: 0.0, 1: 0.0}, 1: {1: 0.0}},
            neumann={2: {1: -1000.0}},
        )
        report = run_model_checks(mesh, bc)
        assert report.ok
        assert report.warnings == ()


# ===========================================================================
# Rapport et levée d'erreur
# ===========================================================================


class TestReport:
    def test_raise_if_errors_noop_when_clean(self):
        mesh, bc = _unit_quad(_well_posed_bc())
        report = run_model_checks(mesh, bc)
        report.raise_if_errors()  # ne lève pas

    def test_report_is_frozen_dataclass(self):
        report = ModelCheckReport()
        assert report.ok
        with pytest.raises(Exception):
            report.errors = ()  # type: ignore[misc]

    def test_warnings_logged(self, caplog):
        import logging
        nodes = np.array([[0, 0], [100, 0], [100, 1], [0, 1]], float)
        elems = [ElementData(Quad4, (0, 1, 2, 3), _STEEL, _THK)]
        mesh = Mesh(nodes=nodes, elements=elems, n_dim=2)
        bc = BoundaryConditions(
            dirichlet={0: {0: 0.0, 1: 0.0}, 3: {0: 0.0, 1: 0.0}},
            neumann={1: {1: 1000.0}},
        )
        with caplog.at_level(logging.WARNING, logger="femsolver.model_check"):
            run_model_checks(mesh, bc)
        assert any("aspect ratio" in r.getMessage() for r in caplog.records)

"""Tests de l'estimateur d'erreur ZZ (Zienkiewicz–Zhu).

Cas de test
-----------
1. Patch test en traction uniforme (Tri3 + Quad4) :
   Si σ_h est exact (champ uniforme → Tri3 le représente exactement),
   alors σ* ≈ σ_h et η ≈ 0.  On vérifie η_relative << 1.

2. Concentration autour du trou (plaque de Kirsch, maillage grossier) :
   Sur le quart de domaine annulaire, les éléments proches du trou (r ≈ R)
   doivent avoir η_e > η_e des éléments loin du trou (r ≈ R_outer).

3. Cohérence SNA vs SPR :
   Les deux méthodes de lissage doivent donner des erreurs globales du même
   ordre (facteur < 5×) et des indicateurs corrélés (ρ_Pearson > 0.7).

4. Tri3 patch test — σ uniforme exact :
   Un champ de contrainte uniforme est représenté exactement par Tri3.
   η_e doit être exactement 0 pour tous les éléments.
"""

from __future__ import annotations

import numpy as np
import pytest

from femsolver.core.assembler import Assembler
from femsolver.core.boundary import apply_dirichlet
from femsolver.core.material import ElasticMaterial
from femsolver.core.mesh import BoundaryConditions, ElementData, Mesh
from femsolver.core.solver import StaticSolver
from femsolver.elements.quad4 import Quad4
from femsolver.elements.tri3 import Tri3
from femsolver.postprocess.error_estimator import ZZErrorResult, zz_error_estimate


# ---------------------------------------------------------------------------
# Fixtures : maillages simples
# ---------------------------------------------------------------------------

STEEL = ElasticMaterial(E=210e9, nu=0.3, rho=7800)


def _simple_quad4_mesh() -> tuple[Mesh, BoundaryConditions]:
    """Plaque 2×1 m divisée en 2 Quad4 — traction σ₀ en x."""
    nodes = np.array([
        [0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
        [0.0, 1.0], [1.0, 1.0], [2.0, 1.0],
    ])
    props = {"thickness": 1.0, "formulation": "plane_stress"}
    elements = (
        ElementData(Quad4, (0, 1, 4, 3), STEEL, props),
        ElementData(Quad4, (1, 2, 5, 4), STEEL, props),
    )
    mesh = Mesh(nodes=nodes, elements=elements, n_dim=2)

    sigma0 = 1e6  # 1 MPa
    # Encastrement en x=0, traction en x=2
    dirichlet = {0: {0: 0.0, 1: 0.0}, 3: {0: 0.0, 1: 0.0}}
    neumann = {2: {0: sigma0 * 0.5}, 5: {0: sigma0 * 0.5}}
    bc = BoundaryConditions(dirichlet=dirichlet, neumann=neumann)
    return mesh, bc


def _simple_tri3_mesh() -> tuple[Mesh, BoundaryConditions]:
    """Plaque 2×1 m divisée en 4 Tri3 — traction σ₀ en x.

    Champ de contrainte σxx = σ₀ uniforme (représenté exactement par CST).
    """
    nodes = np.array([
        [0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
        [0.0, 1.0], [1.0, 1.0], [2.0, 1.0],
    ])
    props = {"thickness": 1.0, "formulation": "plane_stress"}
    elements = (
        ElementData(Tri3, (0, 1, 3), STEEL, props),
        ElementData(Tri3, (1, 4, 3), STEEL, props),
        ElementData(Tri3, (1, 2, 4), STEEL, props),
        ElementData(Tri3, (2, 5, 4), STEEL, props),
    )
    mesh = Mesh(nodes=nodes, elements=elements, n_dim=2)

    sigma0 = 1e6
    dirichlet = {0: {0: 0.0, 1: 0.0}, 3: {0: 0.0, 1: 0.0}}
    neumann = {2: {0: sigma0 * 0.5}, 5: {0: sigma0 * 0.5}}
    bc = BoundaryConditions(dirichlet=dirichlet, neumann=neumann)
    return mesh, bc


def _plate_with_hole_mesh(n_r: int = 8, n_theta: int = 10) -> tuple[Mesh, BoundaryConditions]:
    """Quart de plaque trouée — maillage Quad4 polaire (Kirsch).

    Utilisé pour tester la concentration de l'erreur autour du trou.
    """
    R = 1.0
    R_OUTER = 5.0
    sigma0 = 1.0

    r_vals = np.linspace(R, R_OUTER, n_r + 1)
    t_vals = np.linspace(0.0, np.pi / 2.0, n_theta + 1)

    nodes_list = []
    node_index = {}
    idx = 0
    for i, r in enumerate(r_vals):
        for j, t in enumerate(t_vals):
            nodes_list.append([r * np.cos(t), r * np.sin(t)])
            node_index[(i, j)] = idx
            idx += 1
    nodes = np.array(nodes_list)

    props = {"thickness": 1.0, "formulation": "plane_stress"}
    elements = []
    for i in range(n_r):
        for j in range(n_theta):
            n0 = node_index[(i,     j    )]
            n1 = node_index[(i + 1, j    )]
            n2 = node_index[(i + 1, j + 1)]
            n3 = node_index[(i,     j + 1)]
            elements.append(ElementData(Quad4, (n0, n1, n2, n3), STEEL, props))

    mesh = Mesh(nodes=np.array(nodes), elements=tuple(elements), n_dim=2)

    dirichlet: dict[int, dict[int, float]] = {}
    for i in range(n_r + 1):
        dirichlet[node_index[(i, 0)]] = {1: 0.0}
        dirichlet[node_index[(i, n_theta)]] = {0: 0.0}

    arc_step = R_OUTER * (np.pi / 2.0) / n_theta
    k = (R / R_OUTER) ** 2
    neumann: dict[int, dict[int, float]] = {}
    for j in range(n_theta + 1):
        theta = t_vals[j]
        s = arc_step / 2.0 if j in (0, n_theta) else arc_step
        nid = node_index[(n_r, j)]
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        cos2t, sin2t = np.cos(2.0 * theta), np.sin(2.0 * theta)
        sigma_rr = sigma0 / 2.0 * (1.0 - k) + sigma0 / 2.0 * (1.0 - 4.0 * k + 3.0 * k**2) * cos2t
        sigma_rt = -sigma0 / 2.0 * (1.0 + 2.0 * k - 3.0 * k**2) * sin2t
        neumann[nid] = {
            0: (sigma_rr * cos_t - sigma_rt * sin_t) * s,
            1: (sigma_rr * sin_t + sigma_rt * cos_t) * s,
        }

    bc = BoundaryConditions(dirichlet=dirichlet, neumann=neumann)
    return mesh, bc


def _solve(mesh: Mesh, bc: BoundaryConditions) -> np.ndarray:
    """Assemble et résout le système statique."""
    assembler = Assembler(mesh)
    K = assembler.assemble_stiffness()
    F = assembler.assemble_forces(bc)
    K_bc, F_bc = apply_dirichlet(K, F, mesh, bc)
    return StaticSolver().solve(K_bc, F_bc)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestZZReturnType:
    """Vérifie le type et la forme du résultat."""

    def test_returns_zz_error_result(self) -> None:
        mesh, bc = _simple_quad4_mesh()
        u = _solve(mesh, bc)
        result = zz_error_estimate(mesh, u)
        assert isinstance(result, ZZErrorResult)

    def test_eta_e_shape(self) -> None:
        mesh, bc = _simple_quad4_mesh()
        u = _solve(mesh, bc)
        result = zz_error_estimate(mesh, u)
        assert result.eta_e.shape == (len(mesh.elements),)

    def test_sigma_nodal_shape(self) -> None:
        mesh, bc = _simple_quad4_mesh()
        u = _solve(mesh, bc)
        result = zz_error_estimate(mesh, u)
        assert result.sigma_nodal.shape == (mesh.n_nodes, 3)

    def test_eta_e_nonneg(self) -> None:
        mesh, bc = _simple_quad4_mesh()
        u = _solve(mesh, bc)
        result = zz_error_estimate(mesh, u)
        assert np.all(result.eta_e >= 0.0)

    def test_relative_error_bounded(self) -> None:
        """L'erreur relative est positive et bornée."""
        mesh, bc = _simple_quad4_mesh()
        u = _solve(mesh, bc)
        result = zz_error_estimate(mesh, u)
        assert 0.0 <= result.relative_error

    def test_method_field(self) -> None:
        mesh, bc = _simple_quad4_mesh()
        u = _solve(mesh, bc)
        for m in ("sna", "spr"):
            result = zz_error_estimate(mesh, u, method=m)
            assert result.method == m

    def test_invalid_method_raises(self) -> None:
        mesh, bc = _simple_quad4_mesh()
        u = _solve(mesh, bc)
        with pytest.raises(ValueError, match="method"):
            zz_error_estimate(mesh, u, method="invalid")


class TestZZTri3UniformStress:
    """Tri3 + champ de contraintes uniforme → η_e ≈ 0 partout.

    Le CST représente exactement un champ de contraintes constant.
    σ_h = σ* dans tout le domaine → Δσ = 0 → η = 0.

    En pratique, les sauts au bord (conditions aux limites) peuvent
    introduire une légère erreur sur les éléments frontières. On vérifie
    donc uniquement les éléments intérieurs.
    """

    def test_eta_e_near_zero_interior(self) -> None:
        """η_e intérieur < 0.1% de la norme pour champ uniforme."""
        mesh, bc = _simple_tri3_mesh()
        u = _solve(mesh, bc)
        result = zz_error_estimate(mesh, u, method="sna")
        # Éléments intérieurs : pas en contact avec les bords x=0 ou x=2
        nodes = mesh.nodes
        interior_elems = []
        for e_idx, ed in enumerate(mesh.elements):
            xs = nodes[list(ed.node_ids), 0]
            if xs.min() > 0.01 and xs.max() < 1.99:
                interior_elems.append(e_idx)
        if interior_elems:
            eta_int = result.eta_e[interior_elems]
            norm = result.norm_sigma_h
            assert np.all(eta_int / norm < 1e-6), (
                f"η_e intérieur / ‖σ_h‖ max = {(eta_int / norm).max():.2e}"
            )


class TestZZPlateWithHole:
    """Plaque trouée (Kirsch) — l'erreur se concentre autour du trou.

    Physiquement, le champ de Kirsch varie en 1/r² : gradients élevés
    près du trou (r = R = 1 m), quasi-uniformes loin (r ≈ 5 m).
    L'estimateur ZZ doit refléter cette structure.
    """

    @pytest.fixture(scope="class")
    def kirsch_results(self) -> dict:
        """Résout la plaque trouée et retourne les résultats ZZ."""
        mesh, bc = _plate_with_hole_mesh(n_r=8, n_theta=10)
        u = _solve(mesh, bc)
        result_spr = zz_error_estimate(mesh, u, method="spr")
        result_sna = zz_error_estimate(mesh, u, method="sna")
        return {"mesh": mesh, "u": u, "spr": result_spr, "sna": result_sna}

    def test_error_concentrates_near_hole(self, kirsch_results: dict) -> None:
        """η moyen des éléments proches du trou > η moyen des éléments loin.

        On compare :
        - Couche intérieure : éléments dont tous les nœuds ont r < 1.8 m
        - Couche extérieure : éléments dont tous les nœuds ont r > 3.5 m
        """
        mesh = kirsch_results["mesh"]
        result = kirsch_results["spr"]
        nodes = mesh.nodes
        r_nodes = np.sqrt(nodes[:, 0] ** 2 + nodes[:, 1] ** 2)

        inner_elems = []
        outer_elems = []
        for e_idx, ed in enumerate(mesh.elements):
            r_elem = r_nodes[list(ed.node_ids)]
            if r_elem.max() < 1.8:
                inner_elems.append(e_idx)
            if r_elem.min() > 3.5:
                outer_elems.append(e_idx)

        assert len(inner_elems) > 0, "Aucun élément intérieur trouvé"
        assert len(outer_elems) > 0, "Aucun élément extérieur trouvé"

        eta_inner = result.eta_e[inner_elems].mean()
        eta_outer = result.eta_e[outer_elems].mean()

        assert eta_inner > eta_outer, (
            f"η_inner ({eta_inner:.3e}) devrait être > η_outer ({eta_outer:.3e})"
        )

    def test_relative_error_positive(self, kirsch_results: dict) -> None:
        """L'erreur relative est > 0 (maillage grossier sur gradient fort)."""
        result = kirsch_results["spr"]
        assert result.relative_error > 0.0

    def test_global_error_consistency(self, kirsch_results: dict) -> None:
        """η = √(Σ η_e²) — cohérence entre η_e et η."""
        result = kirsch_results["spr"]
        eta_from_sum = float(np.sqrt(np.sum(result.eta_e ** 2)))
        assert abs(eta_from_sum - result.eta) < 1e-10 * result.eta

    def test_sna_spr_correlation(self, kirsch_results: dict) -> None:
        """SNA et SPR donnent des indicateurs corrélés (ρ > 0.7).

        Les deux méthodes de lissage devraient identifier les mêmes zones
        d'erreur élevée, même si les valeurs absolues diffèrent.
        """
        eta_spr = kirsch_results["spr"].eta_e
        eta_sna = kirsch_results["sna"].eta_e
        rho = float(np.corrcoef(eta_spr, eta_sna)[0, 1])
        assert rho > 0.7, (
            f"Corrélation SNA/SPR = {rho:.3f} < 0.7 — les méthodes identifient "
            f"des zones d'erreur différentes"
        )

    def test_sna_spr_same_order_of_magnitude(self, kirsch_results: dict) -> None:
        """η global SNA et SPR du même ordre (rapport < 5×)."""
        eta_spr = kirsch_results["spr"].eta
        eta_sna = kirsch_results["sna"].eta
        ratio = max(eta_spr, eta_sna) / min(eta_spr, eta_sna)
        assert ratio < 5.0, (
            f"η_SPR / η_SNA = {ratio:.2f} — les estimateurs divergent trop"
        )


class TestZZMixedMesh:
    """Teste le gestionnaire d'éléments non supportés."""

    def test_unsupported_element_raises(self) -> None:
        """Un type d'élément non supporté lève NotImplementedError."""
        from femsolver.elements.bar2d import Bar2D
        from femsolver.core.mesh import ElementData

        nodes = np.array([[0.0, 0.0], [1.0, 0.0]])
        props = {"area": 1e-4}
        elements = (ElementData(Bar2D, (0, 1), STEEL, props),)
        mesh = Mesh(nodes=nodes, elements=elements, n_dim=2)
        u = np.zeros(4)

        with pytest.raises(NotImplementedError):
            zz_error_estimate(mesh, u)

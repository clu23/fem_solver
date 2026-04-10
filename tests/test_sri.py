"""Tests de l'intégration réduite sélective (SRI) pour Quad4 et Hexa8.

Validations
-----------
1. Forme, symétrie, rang de K_e_sri.
2. Pas de modes hourglass : rang(K_sri) == rang(K_full).
3. Patch test traction sur Quad4 SRI : solution identique.
4. Poutre console en flexion Quad4 : K_sri réduit le shear locking.
5. Hexa8 SRI Hughes : rang correct, symétrie, valeurs propres positives.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import lil_matrix

from femsolver.core.assembler import Assembler
from femsolver.core.boundary import apply_dirichlet
from femsolver.core.material import ElasticMaterial
from femsolver.core.mesh import BoundaryConditions, ElementData, Mesh
from femsolver.core.solver import StaticSolver
from femsolver.elements.hexa8 import Hexa8
from femsolver.elements.quad4 import Quad4


# ---------------------------------------------------------------------------
# Données communes
# ---------------------------------------------------------------------------

MAT_STEEL = ElasticMaterial(E=210e9, nu=0.3, rho=7800)
MAT_SIMPLE = ElasticMaterial(E=1.0, nu=0.0, rho=1.0)

NODES_UNIT_2D = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
NODES_UNIT_3D = np.array([
    [0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.],
    [0., 0., 1.], [1., 0., 1.], [1., 1., 1.], [0., 1., 1.],
])
PROPS_STD = {"thickness": 1.0}


# ===========================================================================
# Tests Quad4 SRI — split membrane (2×2) / cisaillement (1 point)
# ===========================================================================


class TestQuad4SRI:
    """Vérifications de stiffness_matrix_sri() pour Quad4.

    Principe : D = D_dil + D_dev
    - D_dil (εxx, εyy, couplage ν) → intégration 2×2 (ordre complet)
    - D_dev (γxy, G = D[2,2])      → 1 point au centre (ξ=η=0)
    """

    # --- Forme et propriétés de base ----------------------------------------

    def test_shape(self) -> None:
        """K_e_sri est de shape (8, 8)."""
        K_sri = Quad4().stiffness_matrix_sri(MAT_SIMPLE, NODES_UNIT_2D, PROPS_STD)
        assert K_sri.shape == (8, 8)

    def test_symmetry(self) -> None:
        """K_e_sri est symétrique."""
        K_sri = Quad4().stiffness_matrix_sri(MAT_STEEL, NODES_UNIT_2D, {"thickness": 0.01})
        np.testing.assert_allclose(K_sri, K_sri.T, atol=1e-6)

    def test_no_hourglass_same_rank_as_full(self) -> None:
        """SRI ne crée pas de modes hourglass : rang(K_sri) == rang(K_full).

        La partie dilatation (2×2) assure le rang pour tous les modes de
        déformation ; la partie cisaillement (1 pt) n'introduit pas de
        mode zéro-énergie supplémentaire.

        Rang attendu : 8 DDL − 3 modes rigides = 5.
        """
        K_full = Quad4().stiffness_matrix(MAT_STEEL, NODES_UNIT_2D, {"thickness": 0.01})
        K_sri  = Quad4().stiffness_matrix_sri(MAT_STEEL, NODES_UNIT_2D, {"thickness": 0.01})
        rank_full = np.linalg.matrix_rank(K_full)
        rank_sri  = np.linalg.matrix_rank(K_sri)
        assert rank_sri == rank_full, (
            f"SRI a introduit des modes hourglass : rang {rank_sri} ≠ {rank_full}"
        )

    def test_sri_differs_from_full_for_nonzero_nu(self) -> None:
        """K_sri ≠ K_full pour ν ≠ 0 (réduction du cisaillement).

        Le split réduit la rigidité en cisaillement intégré (évaluation
        au centre uniquement), ce qui doit produire une matrice différente
        pour un matériau avec ν ≠ 0 (acier).
        """
        K_full = Quad4().stiffness_matrix(MAT_STEEL, NODES_UNIT_2D, {"thickness": 0.01})
        K_sri  = Quad4().stiffness_matrix_sri(MAT_STEEL, NODES_UNIT_2D, {"thickness": 0.01})
        # K_sri doit être différente de K_full (moins de rigidité en cisaillement)
        assert not np.allclose(K_full, K_sri), (
            "K_sri devrait différer de K_full pour ν=0.3"
        )

    def test_sri_softer_in_shear_direction(self) -> None:
        """K_sri a des valeurs propres ≤ K_full en mode cisaillement.

        La SRI réduit la rigidité : max eigenvalue de K_full ≥ max de K_sri.
        """
        K_full = Quad4().stiffness_matrix(MAT_STEEL, NODES_UNIT_2D, {"thickness": 0.01})
        K_sri  = Quad4().stiffness_matrix_sri(MAT_STEEL, NODES_UNIT_2D, {"thickness": 0.01})
        eigs_full = np.linalg.eigvalsh(K_full)
        eigs_sri  = np.linalg.eigvalsh(K_sri)
        # La plus grande valeur propre de K_full ≥ K_sri (K_full est plus rigide)
        assert eigs_full.max() >= eigs_sri.max() * 0.99, (
            "K_full devrait être au moins aussi rigide que K_sri"
        )

    def test_zero_thickness_raises(self) -> None:
        """Épaisseur nulle → ValueError."""
        with pytest.raises(ValueError, match="épaisseur"):
            Quad4().stiffness_matrix_sri(MAT_SIMPLE, NODES_UNIT_2D, {"thickness": 0.0})

    # --- Patch test : traction uniaxiale ------------------------------------

    def test_patch_traction_sri(self) -> None:
        """Patch test traction uniforme σxx = σ0 avec SRI : résultat identique.

        Configuration : rectangle 2×1 m, σxx = 1 Pa, ν = 0.
        Solution : ux(x=2) = σ0/E * 2 = 2 m.
        Avec ν=0 et γxy = 0, la partie SRI (cisaillement) ne contribue pas
        et le résultat est identique à l'intégration complète.
        """
        E, nu, t, sigma0 = 1.0, 0.0, 1.0, 1.0
        mat = ElasticMaterial(E=E, nu=nu, rho=1.0)
        nodes = np.array([[0., 0.], [2., 0.], [2., 1.], [0., 1.]])
        F_nodal = sigma0 * t * 0.5

        mesh = Mesh(
            nodes=nodes,
            elements=(ElementData(Quad4, (0, 1, 2, 3), mat,
                                  {"thickness": t, "formulation": "plane_stress"}),),
            n_dim=2,
        )
        bc = BoundaryConditions(
            dirichlet={0: {0: 0.0, 1: 0.0}, 3: {0: 0.0}},
            neumann={1: {0: F_nodal}, 2: {0: F_nodal}},
        )

        # Assemblage manuel avec K_sri
        from scipy.sparse import csr_matrix as csr
        elem = Quad4()
        elem_data = list(mesh.elements)[0]
        nc = mesh.node_coords(elem_data.node_ids)
        K_sri_dense = elem.stiffness_matrix_sri(mat, nc, elem_data.properties)
        K_sparse = csr(K_sri_dense)
        F = Assembler(mesh).assemble_forces(bc)
        K_bc, F_bc = apply_dirichlet(K_sparse, F, mesh, bc)
        u = StaticSolver().solve(K_bc, F_bc)

        np.testing.assert_allclose(u[2], sigma0 / E * 2.0, atol=1e-10,
            err_msg="Patch test traction SRI : ux(x=2) incorrect")


# ===========================================================================
# Tests Hexa8 SRI — split Hughes (1980) : déviatorique (2×2×2) / volumétrique (1 pt)
# ===========================================================================


class TestHexa8SRI:
    """Vérifications de stiffness_matrix_sri() pour Hexa8.

    Formulation de Hughes (1980) :
    - D_dev = D − K_bulk·m·mᵀ  → intégration 2×2×2 (ordre complet)
    - D_vol = K_bulk·m·mᵀ       → 1 point au centre

    Cible le volumetric locking des matériaux quasi-incompressibles.
    """

    def test_shape(self) -> None:
        """K_e_sri est de shape (24, 24)."""
        K_sri = Hexa8().stiffness_matrix_sri(MAT_SIMPLE, NODES_UNIT_3D, {})
        assert K_sri.shape == (24, 24)

    def test_symmetry(self) -> None:
        """K_e_sri est symétrique."""
        K_sri = Hexa8().stiffness_matrix_sri(MAT_STEEL, NODES_UNIT_3D, {})
        np.testing.assert_allclose(K_sri, K_sri.T, atol=1e-4)

    def test_no_hourglass_same_rank_as_full(self) -> None:
        """SRI Hughes ne crée pas de modes hourglass : rang(K_sri) == rang(K_full).

        K_dev (2×2×2, contient toute la réponse déviatorique + cisaillement) a
        rang 18 à lui seul.  L'ajout K_vol (rang 1) ne peut que conserver
        ou augmenter le rang.

        Rang attendu : 24 DDL − 6 modes rigides = 18.
        """
        K_full = Hexa8().stiffness_matrix(MAT_STEEL, NODES_UNIT_3D, {})
        K_sri  = Hexa8().stiffness_matrix_sri(MAT_STEEL, NODES_UNIT_3D, {})
        rank_full = np.linalg.matrix_rank(K_full)
        rank_sri  = np.linalg.matrix_rank(K_sri)
        assert rank_sri == rank_full, (
            f"SRI Hughes a introduit des modes hourglass : rang {rank_sri} ≠ {rank_full}"
        )

    def test_positive_semidefinite(self) -> None:
        """K_e_sri est semi-définie positive (valeurs propres ≥ 0)."""
        K_sri = Hexa8().stiffness_matrix_sri(MAT_STEEL, NODES_UNIT_3D, {})
        eigs = np.linalg.eigvalsh(K_sri)
        # Les 6 premières doivent être ≈ 0 (modes rigides), les autres > 0
        assert eigs.min() >= -1e-6 * eigs.max(), (
            f"K_e_sri a des valeurs propres négatives : min = {eigs.min():.3e}"
        )

    def test_sri_softer_than_full(self) -> None:
        """K_sri est plus souple que K_full (trace(K_sri) < trace(K_full)).

        La SRI Hughes intègre D_vol à 1 point au lieu de 2×2×2.
        La contribution volumétrique est ainsi sous-évaluée → K_sri est
        globalement moins rigide que K_full.  C'est le comportement attendu :
        la SRI *relâche* la contrainte volumétrique pour les matériaux
        quasi-incompressibles.
        """
        K_full = Hexa8().stiffness_matrix(MAT_STEEL, NODES_UNIT_3D, {})
        K_sri  = Hexa8().stiffness_matrix_sri(MAT_STEEL, NODES_UNIT_3D, {})
        # K_sri doit être plus souple (trace plus faible)
        assert np.trace(K_sri) < np.trace(K_full), (
            "K_sri Hughes devrait être plus souple que K_full "
            f"(trace_sri={np.trace(K_sri):.3e} vs trace_full={np.trace(K_full):.3e})"
        )
        # Mais pas trop souple : ratio > 0.5 (pas de hourglass catastrophique)
        ratio = np.trace(K_sri) / np.trace(K_full)
        assert ratio > 0.5, f"K_sri trop souple (ratio={ratio:.3f}), possible hourglass"

    def test_invalid_shape_raises(self) -> None:
        """nodes.shape ≠ (8,3) → ValueError."""
        bad = np.zeros((6, 3))
        with pytest.raises(ValueError, match="Hexa8"):
            Hexa8().stiffness_matrix_sri(MAT_SIMPLE, bad, {})


# ===========================================================================
# Comparaison poutre console — shear locking Quad4
# ===========================================================================


def _build_cantilever_quad4(
    n_x: int,
    integration: str,
    E: float = 210e9,
    nu: float = 0.3,
    L: float = 1.0,
    H: float = 0.1,
    t: float = 0.01,
    P: float = 1000.0,
) -> float:
    """Construit et résout une poutre console Quad4 n_x × 1 éléments.

    Retourne le déplacement vertical (uy < 0) au nœud de l'extrémité libre.

    Parameters
    ----------
    n_x : int
        Nombre d'éléments le long de la longueur.
    integration : str
        ``"full"`` (2×2 Gauss) ou ``"sri"`` (intégration réduite sélective).

    Notes
    -----
    Maillage (n_x × 1) ::

        n_per_row … 2*(n_per_row)-1   ← rangée haute y=H
        0          …    n_x            ← rangée basse y=0

    Encastrement : nœuds de la colonne x=0.
    Charge : P/2 vers −y sur chaque nœud de la colonne x=L.
    """
    mat = ElasticMaterial(E=E, nu=nu, rho=7800.0)
    props = {"thickness": t, "formulation": "plane_stress"}
    n_per_row = n_x + 1

    xs = np.linspace(0.0, L, n_per_row)
    # Rangée basse (y=0) : indices 0…n_x ; rangée haute (y=H) : indices n_per_row…
    nodes = np.array([[x, y] for y in [0.0, H] for x in xs])

    elements: list[ElementData] = []
    for i in range(n_x):
        n0, n1 = i, i + 1
        n2, n3 = n_per_row + i + 1, n_per_row + i
        elements.append(ElementData(Quad4, (n0, n1, n2, n3), mat, props))

    mesh = Mesh(nodes=nodes, elements=tuple(elements), n_dim=2)

    # CL : encastrement à x=0
    dirichlet = {
        0:         {0: 0.0, 1: 0.0},
        n_per_row: {0: 0.0, 1: 0.0},
    }
    # Charge vers -y répartie sur les 2 nœuds de la colonne droite
    neumann = {
        n_x:             {1: -P / 2.0},
        n_per_row + n_x: {1: -P / 2.0},
    }
    bc = BoundaryConditions(dirichlet=dirichlet, neumann=neumann)

    # Assemblage selon la méthode d'intégration choisie
    n_dof = mesh.n_dof
    K_glob = lil_matrix((n_dof, n_dof))
    elem_instance = Quad4()
    for elem_data in mesh.elements:
        nc = mesh.node_coords(elem_data.node_ids)
        if integration == "sri":
            K_e = elem_instance.stiffness_matrix_sri(mat, nc, props)
        else:
            K_e = elem_instance.stiffness_matrix(mat, nc, props)
        dofs = list(mesh.global_dofs(elem_data.node_ids))
        for ii, di in enumerate(dofs):
            for jj, dj in enumerate(dofs):
                K_glob[di, dj] += K_e[ii, jj]

    K_glob_csr = K_glob.tocsr()
    F = Assembler(mesh).assemble_forces(bc)
    K_bc, F_bc = apply_dirichlet(K_glob_csr, F, mesh, bc)
    u = StaticSolver().solve(K_bc, F_bc)

    # uy du nœud extrême inférieur (nœud n_x, y=0) — négatif pour charge vers -y
    return float(u[2 * n_x + 1])


class TestCantileverShearLocking:
    """Comparaison intégration complète vs SRI — shear locking Quad4.

    Configuration
    -------------
    Poutre acier 1 m × 0.1 m (t = 0.01 m, ν = 0.3).
    Encastrée à x=0, charge ponctuelle P = 1000 N à x=L (vers −y).
    Maillage grossier : 4 éléments Quad4 en longueur, 1 en hauteur.

    Solution analytique Euler-Bernoulli
    ------------------------------------
    I = t·H³/12 = 0.01×0.001/12 ≈ 8.333×10⁻⁷ m⁴
    |δ| = PL³/(3EI) = 1000/(3·210e9·8.333e-7) ≈ 1.905×10⁻³ m

    Le déplacement réel est négatif (vers −y) : v_tip ≈ −1.905×10⁻³ m.
    """

    E = 210e9
    nu = 0.3
    L, H, t = 1.0, 0.1, 0.01
    P = 1000.0
    n_x = 4   # maillage grossier 4×1

    @property
    def delta_analytical(self) -> float:
        """δ = PL³/(3EI) (en valeur absolue, > 0)."""
        I = self.t * self.H ** 3 / 12.0
        return self.P * self.L ** 3 / (3.0 * self.E * I)

    def test_full_integration_underestimates(self) -> None:
        """Full 2×2 sous-estime la flèche (shear locking → trop rigide).

        On attend |v_full| / |δ_analytique| < 0.90 pour un maillage 4×1.
        """
        v_full = _build_cantilever_quad4(
            self.n_x, "full",
            E=self.E, nu=self.nu, L=self.L, H=self.H, t=self.t, P=self.P,
        )
        ratio = abs(v_full) / self.delta_analytical
        assert ratio < 0.90, (
            f"Attendu ratio < 0.90 (shear locking), obtenu {ratio:.3f}\n"
            f"|v_full|={abs(v_full):.4e}, δ_analytique={self.delta_analytical:.4e}"
        )

    def test_sri_closer_to_analytical_than_full(self) -> None:
        """SRI doit être strictement plus proche de l'analytique que Full.

        La SRI réduit la rigidité parasite de cisaillement → flèche plus proche.
        """
        v_full = _build_cantilever_quad4(
            self.n_x, "full",
            E=self.E, nu=self.nu, L=self.L, H=self.H, t=self.t, P=self.P,
        )
        v_sri = _build_cantilever_quad4(
            self.n_x, "sri",
            E=self.E, nu=self.nu, L=self.L, H=self.H, t=self.t, P=self.P,
        )
        delta = self.delta_analytical

        err_full = abs(abs(v_full) - delta) / delta
        err_sri  = abs(abs(v_sri)  - delta) / delta

        assert err_sri < err_full, (
            f"SRI devrait être plus précis que Full :\n"
            f"  Analytique : {delta:.6e} m\n"
            f"  Full 2×2   : {abs(v_full):.6e} m  (erreur {err_full:.1%})\n"
            f"  SRI        : {abs(v_sri):.6e} m  (erreur {err_sri:.1%})"
        )

    def test_sri_matches_analytical_coarse_mesh(self) -> None:
        """SRI sur maillage 4×1 doit être proche de l'analytique (< 15 %).

        La SRI élimine la rigidité parasite de cisaillement : la flèche SRI
        est bien plus proche de Euler-Bernoulli que l'intégration complète.
        """
        v_sri = _build_cantilever_quad4(
            self.n_x, "sri",
            E=self.E, nu=self.nu, L=self.L, H=self.H, t=self.t, P=self.P,
        )
        delta = self.delta_analytical
        err = abs(abs(v_sri) - delta) / delta
        assert err < 0.15, (
            f"Erreur SRI trop grande : {err:.1%}\n"
            f"  |v_sri| = {abs(v_sri):.4e}, δ = {delta:.4e}"
        )

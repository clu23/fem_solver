"""Tests unitaires Beam2D — validation par solutions analytiques.

Cas de référence
----------------

Poutre console (cantilever), charge ponctuelle P en bout :
    v_max  = PL³ / (3EI)      (déflection à l'extrémité libre)
    θ_max  = PL²  / (2EI)     (rotation à l'extrémité libre)
    R_y    = P                 (réaction verticale à l'encastrement)
    M_0    = P·L               (moment de réaction à l'encastrement)

Poutre bi-appuyée (simply supported), charge ponctuelle P au centre :
    v_max = PL³ / (48EI)      (déflection au centre)

La déformée d'Euler-Bernoulli est un polynôme cubique.  Les fonctions de
forme de Hermite sont exactement cubiques → un seul élément Beam2D suffit
pour retrouver la solution exacte avec précision machine (rtol ≈ 1e-12).

Structure des DDL avec dof_per_node=3
--------------------------------------
Nœud k → DDL  3k (ux),  3k+1 (uy),  3k+2 (θz).

La plupart des tests construisent le maillage directement avec Mesh et
BoundaryConditions sans passer par une fonction auxiliaire, pour rester
aussi proches que possible de l'interface réelle.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse.linalg import spsolve

from femsolver.core.assembler import Assembler
from femsolver.core.boundary import apply_dirichlet
from femsolver.core.material import ElasticMaterial
from femsolver.core.mesh import BoundaryConditions, ElementData, Mesh
from femsolver.elements.beam2d import Beam2D


# ---------------------------------------------------------------------------
# Données communes
# ---------------------------------------------------------------------------

E   = 210e9     # Pa (acier)
NU  = 0.3
RHO = 7800.0    # kg/m³
A   = 1e-4      # m² (section)
h   = 0.01      # m  (hauteur section — b=h=1 cm → carré)
I   = h**4 / 12 # m⁴ ≈ 8.333e-9  (I = bh³/12, b=h)
L   = 1.0       # m
P   = 10_000.0  # N

MAT   = ElasticMaterial(E=E, nu=NU, rho=RHO)
PROPS = {"area": A, "inertia": I}


def _build_cantilever(
    n_elem: int,
    load: float = P,
) -> tuple[Mesh, BoundaryConditions]:
    """Poutre console horizontale encastrée en x=0, charge Fy à x=L.

    Nœuds sur l'axe x, espacés de L/n_elem.
    CL Dirichlet : nœud 0 → ux=uy=θz=0.
    CL Neumann   : nœud n_elem → uy = −load (vers le bas).
    """
    n_nodes = n_elem + 1
    xs = np.linspace(0.0, L, n_nodes)
    nodes = np.column_stack([xs, np.zeros(n_nodes)])

    elements = tuple(
        ElementData(Beam2D, (i, i + 1), MAT, PROPS)
        for i in range(n_elem)
    )
    mesh = Mesh(nodes=nodes, elements=elements, n_dim=2, dof_per_node=3)

    dirichlet = {0: {0: 0.0, 1: 0.0, 2: 0.0}}
    neumann   = {n_elem: {1: -load}}
    bc = BoundaryConditions(dirichlet=dirichlet, neumann=neumann)
    return mesh, bc


def _build_simply_supported(n_elem: int, load: float = P) -> tuple[Mesh, BoundaryConditions]:
    """Poutre bi-appuyée, charge ponctuelle au centre.

    n_elem doit être pair (centre exact sur un nœud).
    CL Dirichlet : nœud 0 → ux=uy=0 ; nœud n_elem → uy=0.
    CL Neumann   : nœud central → uy = −load.
    """
    assert n_elem % 2 == 0, "n_elem doit être pair"
    n_nodes = n_elem + 1
    xs = np.linspace(0.0, L, n_nodes)
    nodes = np.column_stack([xs, np.zeros(n_nodes)])

    elements = tuple(
        ElementData(Beam2D, (i, i + 1), MAT, PROPS)
        for i in range(n_elem)
    )
    mesh = Mesh(nodes=nodes, elements=elements, n_dim=2, dof_per_node=3)

    mid = n_elem // 2
    dirichlet = {0: {0: 0.0, 1: 0.0}, n_elem: {1: 0.0}}
    neumann   = {mid: {1: -load}}
    bc = BoundaryConditions(dirichlet=dirichlet, neumann=neumann)
    return mesh, bc


def _solve(mesh: Mesh, bc: BoundaryConditions) -> np.ndarray:
    """Assemble, applique CL, résout K·u = F."""
    assembler = Assembler(mesh)
    K = assembler.assemble_stiffness()
    F = assembler.assemble_forces(bc)
    K_bc, F_bc = apply_dirichlet(K, F, mesh, bc)
    return spsolve(K_bc, F_bc)


# ===========================================================================
# 1. Matrice de rigidité — propriétés algébriques
# ===========================================================================

class TestBeam2DStiffnessMatrix:
    """Propriétés de K_e : forme, symétrie, valeurs pivots."""

    def setup_method(self) -> None:
        self.nodes = np.array([[0.0, 0.0], [L, 0.0]])
        self.elem  = Beam2D()

    def test_shape(self) -> None:
        K = self.elem.stiffness_matrix(MAT, self.nodes, PROPS)
        assert K.shape == (6, 6)

    def test_symmetry(self) -> None:
        K = self.elem.stiffness_matrix(MAT, self.nodes, PROPS)
        np.testing.assert_allclose(K, K.T, atol=1e-10)

    def test_axial_term_K00(self) -> None:
        """K[0,0] = EA/L (rigidité axiale)."""
        K = self.elem.stiffness_matrix(MAT, self.nodes, PROPS)
        np.testing.assert_allclose(K[0, 0], E * A / L, rtol=1e-12)

    def test_axial_coupling_K03(self) -> None:
        """K[0,3] = -EA/L (couplage axial nœud 1–nœud 2)."""
        K = self.elem.stiffness_matrix(MAT, self.nodes, PROPS)
        np.testing.assert_allclose(K[0, 3], -E * A / L, rtol=1e-12)

    def test_bending_K11(self) -> None:
        """K[1,1] = 12EI/L³."""
        K = self.elem.stiffness_matrix(MAT, self.nodes, PROPS)
        np.testing.assert_allclose(K[1, 1], 12.0 * E * I / L**3, rtol=1e-12)

    def test_bending_K12(self) -> None:
        """K[1,2] = 6EI/L² (couplage déflection–rotation)."""
        K = self.elem.stiffness_matrix(MAT, self.nodes, PROPS)
        np.testing.assert_allclose(K[1, 2], 6.0 * E * I / L**2, rtol=1e-12)

    def test_bending_K22(self) -> None:
        """K[2,2] = 4EI/L (rigidité rotationnelle)."""
        K = self.elem.stiffness_matrix(MAT, self.nodes, PROPS)
        np.testing.assert_allclose(K[2, 2], 4.0 * E * I / L, rtol=1e-12)

    def test_bending_K25(self) -> None:
        """K[2,5] = 2EI/L (couplage croisé moments)."""
        K = self.elem.stiffness_matrix(MAT, self.nodes, PROPS)
        np.testing.assert_allclose(K[2, 5], 2.0 * E * I / L, rtol=1e-12)

    def test_axial_bending_decoupled(self) -> None:
        """DDL axiaux et de flexion sont découplés (K_local hors-diagonale nulle)."""
        # Pour une poutre horizontale, le couplage axial/flexion est nul
        K = self.elem.stiffness_matrix(MAT, self.nodes, PROPS)
        # K[0, 1], K[0, 2], K[0, 4], K[0, 5] doivent être nuls
        axial_dofs   = [0, 3]
        bending_dofs = [1, 2, 4, 5]
        for i in axial_dofs:
            for j in bending_dofs:
                assert abs(K[i, j]) < 1e-10, f"K[{i},{j}] = {K[i,j]} ≠ 0"

    def test_singular_before_bc(self) -> None:
        """K sans CL est singulière (modes de corps rigide en 2D : 3 modes)."""
        K = self.elem.stiffness_matrix(MAT, self.nodes, PROPS)
        rank = np.linalg.matrix_rank(K, tol=1e-6)
        assert rank == 3, f"rang attendu = 3, obtenu {rank}"

    def test_inclined_beam_symmetry(self) -> None:
        """K reste symétrique pour une poutre inclinée à 45°."""
        l = np.sqrt(2.0)
        nodes_45 = np.array([[0.0, 0.0], [1.0, 1.0]])
        K = self.elem.stiffness_matrix(MAT, nodes_45, PROPS)
        np.testing.assert_allclose(K, K.T, atol=1e-8)

    def test_length_scaling(self) -> None:
        """K[0,0] proportionnel à 1/L (rigidité axiale)."""
        nodes_2L = np.array([[0.0, 0.0], [2.0 * L, 0.0]])
        K2 = self.elem.stiffness_matrix(MAT, nodes_2L, PROPS)
        K1 = self.elem.stiffness_matrix(MAT, self.nodes, PROPS)
        np.testing.assert_allclose(K2[0, 0], K1[0, 0] / 2.0, rtol=1e-12)


# ===========================================================================
# 2. Matrice de masse — propriétés
# ===========================================================================

class TestBeam2DMassMatrix:
    """Propriétés de M_e : forme, symétrie, masse totale."""

    def setup_method(self) -> None:
        self.nodes = np.array([[0.0, 0.0], [L, 0.0]])
        self.elem  = Beam2D()

    def test_shape(self) -> None:
        M = self.elem.mass_matrix(MAT, self.nodes, PROPS)
        assert M.shape == (6, 6)

    def test_symmetry(self) -> None:
        M = self.elem.mass_matrix(MAT, self.nodes, PROPS)
        np.testing.assert_allclose(M, M.T, atol=1e-10)

    def test_positive_definite(self) -> None:
        """M doit être définie positive (tous valeurs propres > 0)."""
        M = self.elem.mass_matrix(MAT, self.nodes, PROPS)
        eigvals = np.linalg.eigvalsh(M)
        assert np.all(eigvals > 0), f"Valeurs propres non positives : {eigvals}"

    def test_total_mass_ux(self) -> None:
        """Somme de la ligne du DDL ux₁ = ρAL/2 (demi-masse axiale).

        M_axial = (ρAL/6)·[[2,1],[1,2]]. La somme de la ligne ux₁ = 3/6 = ρAL/2.
        """
        M = self.elem.mass_matrix(MAT, self.nodes, PROPS)
        m_total = RHO * A * L
        # Ligne 0 (ux1) : somme = ρAL/2
        np.testing.assert_allclose(M[0, :].sum(), m_total / 2.0, rtol=1e-12)

    def test_total_mass_uy(self) -> None:
        """Somme de la ligne uy₁ : 156 + 22L + 54 − 13L = 210 − (-9L) → ...

        ρAL/420 · (156 + 22L + 54 − 13L) = ρAL/420 · (210 + 9L).
        Pour L=1 : ρAL/420 · 219 ≈ 0.521·ρAL.
        Vérification directe.
        """
        M = self.elem.mass_matrix(MAT, self.nodes, PROPS)
        mb = RHO * A * L / 420.0
        expected = mb * (156.0 + 22.0 * L + 54.0 - 13.0 * L)
        np.testing.assert_allclose(M[1, :].sum(), expected, rtol=1e-12)

    def test_axial_term_M00(self) -> None:
        """M[0,0] = ρAL/3 (terme diagonal axial : 2/6 = 1/3)."""
        M = self.elem.mass_matrix(MAT, self.nodes, PROPS)
        np.testing.assert_allclose(M[0, 0], RHO * A * L / 3.0, rtol=1e-12)

    def test_axial_term_M03(self) -> None:
        """M[0,3] = ρAL/6 (terme croisé axial)."""
        M = self.elem.mass_matrix(MAT, self.nodes, PROPS)
        np.testing.assert_allclose(M[0, 3], RHO * A * L / 6.0, rtol=1e-12)


# ===========================================================================
# 3. Poutre console — solution analytique
# ===========================================================================

class TestCantileverAnalytical:
    """Poutre console encastrée–libre, charge Fy = P en bout.

    Solution exacte (polynomial cubique, capturé par 1 élément Hermite) :
        v_tip = PL³/(3EI)
        θ_tip = PL²/(2EI)
    """

    def test_deflection_single_element(self) -> None:
        """1 élément : déflection exacte à précision machine.

        δ = PL³/(3EI) = 10000 × 1³ / (3 × 210e9 × 8.333e-9)
          = 10000 / 5250 ≈ 1.905e-3 m
        """
        mesh, bc = _build_cantilever(n_elem=1)
        u = _solve(mesh, bc)
        v_tip = -u[4]   # DDL uy au nœud 1 (vers le haut = positif, charge vers le bas)
        v_analytical = P * L**3 / (3.0 * E * I)
        np.testing.assert_allclose(v_tip, v_analytical, rtol=1e-10)

    def test_rotation_single_element(self) -> None:
        """1 élément : rotation exacte à précision machine.

        θ = PL²/(2EI)
        """
        mesh, bc = _build_cantilever(n_elem=1)
        u = _solve(mesh, bc)
        theta_tip = -u[5]   # θz au nœud 1 (sens anti-horaire positif)
        theta_analytical = P * L**2 / (2.0 * E * I)
        np.testing.assert_allclose(theta_tip, theta_analytical, rtol=1e-10)

    def test_deflection_4_elements(self) -> None:
        """4 éléments : même résultat (polynomial cubique = solution exacte)."""
        mesh, bc = _build_cantilever(n_elem=4)
        u = _solve(mesh, bc)
        v_tip = -u[3 * 4 + 1]   # DDL uy du dernier nœud
        v_analytical = P * L**3 / (3.0 * E * I)
        np.testing.assert_allclose(v_tip, v_analytical, rtol=1e-10)

    def test_axial_dof_zero(self) -> None:
        """Aucun chargement axial → ux = 0 partout (découplage)."""
        mesh, bc = _build_cantilever(n_elem=2)
        u = _solve(mesh, bc)
        ux_dofs = [3 * i for i in range(3)]   # ux des 3 nœuds
        np.testing.assert_allclose(u[ux_dofs], 0.0, atol=1e-12)

    def test_clamped_dofs_zero(self) -> None:
        """Les 3 DDL du nœud encastré (0) sont nuls après résolution."""
        mesh, bc = _build_cantilever(n_elem=2)
        u = _solve(mesh, bc)
        np.testing.assert_allclose(u[:3], 0.0, atol=1e-8)

    def test_section_forces_single_element(self) -> None:
        """Efforts internes : V₁ = P, M₁ = PL (à l'encastrement).

        Pour une console chargée en bout :
            - Tranchant constant = P dans toute la poutre
            - Moment à l'encastrement = P·L
        """
        mesh, bc = _build_cantilever(n_elem=1)
        u = _solve(mesh, bc)
        u_e = u  # 6 DDL pour 1 seul élément

        node_coords = mesh.node_coords((0, 1))
        sf = Beam2D().section_forces(MAT, node_coords, PROPS, u_e)

        # Convention : f_local = K_local @ u_local → forces internes nodales
        # Au nœud 1 (encastrement) : réaction = P vers le haut → V1 = +P
        np.testing.assert_allclose(abs(sf["V1"]), P, rtol=1e-8)
        np.testing.assert_allclose(abs(sf["M1"]), P * L, rtol=1e-8)


# ===========================================================================
# 4. Poutre bi-appuyée — solution analytique
# ===========================================================================

class TestSimplySupportedAnalytical:
    """Poutre bi-appuyée, charge P au centre.

    Solution exacte :
        v_center = PL³ / (48EI)

    Avec 2 éléments (nœud central au milieu de la portée), la déformée
    cubique est exactement dans l'espace Hermite → précision machine.
    """

    def test_center_deflection_2_elements(self) -> None:
        """v_center = PL³/(48EI) avec 2 éléments (solution exacte).

        v = 10000 × 1³ / (48 × 210e9 × 8.333e-9) ≈ 5.952e-4 m
        """
        mesh, bc = _build_simply_supported(n_elem=2)
        u = _solve(mesh, bc)
        v_center = -u[4]   # DDL uy du nœud central (nœud 1)
        v_analytical = P * L**3 / (48.0 * E * I)
        np.testing.assert_allclose(v_center, v_analytical, rtol=1e-10)

    def test_center_deflection_4_elements(self) -> None:
        """Même résultat avec 4 éléments (toujours exact pour charge ponctuelle)."""
        mesh, bc = _build_simply_supported(n_elem=4)
        u = _solve(mesh, bc)
        v_center = -u[3 * 2 + 1]   # nœud central = nœud 2
        v_analytical = P * L**3 / (48.0 * E * I)
        np.testing.assert_allclose(v_center, v_analytical, rtol=1e-10)

    def test_symmetry_of_deflection(self) -> None:
        """La déflection est symétrique par rapport au centre.

        uy(x) = uy(L−x) par symétrie de charge et géométrie.
        """
        mesh, bc = _build_simply_supported(n_elem=4)
        u = _solve(mesh, bc)
        # nœuds : 0, 1, 2(center), 3, 4
        uy = np.array([u[3 * k + 1] for k in range(5)])
        np.testing.assert_allclose(uy[1], uy[3], rtol=1e-10)
        np.testing.assert_allclose(uy[0], 0.0, atol=1e-10)
        np.testing.assert_allclose(uy[4], 0.0, atol=1e-10)

    def test_support_rotations_nonzero(self) -> None:
        """Les rotations aux appuis doivent être non nulles (appuis simples ≠ encastrement)."""
        mesh, bc = _build_simply_supported(n_elem=2)
        u = _solve(mesh, bc)
        theta_0 = u[2]   # θz au nœud 0
        theta_2 = u[8]   # θz au nœud 2
        assert abs(theta_0) > 1e-10, "Rotation à l'appui 0 anormalement nulle"
        assert abs(theta_2) > 1e-10, "Rotation à l'appui 2 anormalement nulle"

    def test_support_rotations_equal_opposite(self) -> None:
        """Par symétrie, θ(0) = −θ(L) pour une charge centrale."""
        mesh, bc = _build_simply_supported(n_elem=2)
        u = _solve(mesh, bc)
        theta_0 =  u[2]
        theta_L = -u[8]   # antisymétrique par convention de signe
        np.testing.assert_allclose(abs(theta_0), abs(theta_L), rtol=1e-10)


# ===========================================================================
# 5. Poutre inclinée — rotation de repère
# ===========================================================================

class TestInclinedBeam:
    """Vérifications de la rotation de repère pour une poutre inclinée."""

    def test_vertical_beam_symmetry(self) -> None:
        """K reste symétrique pour une poutre verticale (θ = 90°)."""
        nodes = np.array([[0.0, 0.0], [0.0, L]])
        K = Beam2D().stiffness_matrix(MAT, nodes, PROPS)
        np.testing.assert_allclose(K, K.T, atol=1e-8)

    def test_horizontal_beam_no_coupling(self) -> None:
        """Poutre horizontale : ux et uy découplés dans K_global."""
        nodes = np.array([[0.0, 0.0], [L, 0.0]])
        K = Beam2D().stiffness_matrix(MAT, nodes, PROPS)
        # K[0,1] = K[0,4] = 0
        np.testing.assert_allclose(K[0, 1], 0.0, atol=1e-10)
        np.testing.assert_allclose(K[0, 4], 0.0, atol=1e-10)

    def test_vertical_beam_stiffness_swap(self) -> None:
        """Poutre verticale : K_global[0,0] = rigidité de flexion 12EI/L³.

        Pour une poutre horizontale K[0,0] = EA/L (axial).
        Pour une poutre verticale,  K[0,0] = 12EI/L³ (flexion projetée sur x).
        Les DDL axiaux et de flexion restent découplés (K_local[axial,flex]=0)
        mais la direction de rigidité maximale tourne avec la poutre.
        """
        nodes = np.array([[0.0, 0.0], [0.0, L]])
        K = Beam2D().stiffness_matrix(MAT, nodes, PROPS)
        np.testing.assert_allclose(K[0, 0], 12.0 * E * I / L**3, rtol=1e-10)

    def test_rotation_invariant_stiffness(self) -> None:
        """Trace de K_global invariante par rotation (trace = trace(K_local)).

        tr(T.T K T) = tr(K) car tr(AB) = tr(BA).
        """
        nodes_h = np.array([[0.0, 0.0], [L,   0.0]])
        nodes_v = np.array([[0.0, 0.0], [0.0,  L]])
        K_h = Beam2D().stiffness_matrix(MAT, nodes_h, PROPS)
        K_v = Beam2D().stiffness_matrix(MAT, nodes_v, PROPS)
        np.testing.assert_allclose(np.trace(K_h), np.trace(K_v), rtol=1e-10)


# ===========================================================================
# 6. Propriétés du maillage avec dof_per_node=3
# ===========================================================================

class TestMeshDofPerNode:
    """Vérification que Mesh gère correctement dof_per_node=3."""

    def test_n_dof_3_per_node(self) -> None:
        """n_dof = n_nodes × 3 pour un maillage Beam2D."""
        mesh, _ = _build_cantilever(n_elem=4)
        assert mesh.n_dof == 5 * 3  # 5 nœuds × 3 DDL

    def test_global_dofs_node0(self) -> None:
        """Nœud 0 → DDL [0, 1, 2]."""
        mesh, _ = _build_cantilever(n_elem=2)
        assert mesh.global_dofs((0,)) == [0, 1, 2]

    def test_global_dofs_node2(self) -> None:
        """Nœud 2 → DDL [6, 7, 8]."""
        mesh, _ = _build_cantilever(n_elem=2)
        assert mesh.global_dofs((2,)) == [6, 7, 8]

    def test_global_dofs_two_nodes(self) -> None:
        """Nœuds (0, 1) → DDL [0, 1, 2, 3, 4, 5]."""
        mesh, _ = _build_cantilever(n_elem=2)
        assert mesh.global_dofs((0, 1)) == [0, 1, 2, 3, 4, 5]

    def test_dpn_property(self) -> None:
        """mesh.dpn == 3 quand dof_per_node=3."""
        mesh, _ = _build_cantilever(n_elem=1)
        assert mesh.dpn == 3

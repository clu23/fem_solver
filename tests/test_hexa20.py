"""Tests unitaires pour l'élément Hexa20 (sérendipité 3D) — validation analytique.

Stratégie de validation
------------------------
1. Propriétés des fonctions de forme (partition de l'unité, interpolation nodale,
   dérivées sommant à zéro).
2. Matrice K_e : forme, symétrie, 6 modes rigides (solide 3D).
3. Matrice M_e : symétrie, conservation de masse.
4. Patch test 3D linéaire : champ de déplacement linéaire reproduit exactement.
5. Patch test quadratique : champ du second degré reproduit exactement.
6. Traction uniaxiale assemblée : déformation uniforme + contraction de Poisson.
"""

from __future__ import annotations

import numpy as np
import pytest

from femsolver.core.assembler import Assembler
from femsolver.core.boundary import apply_dirichlet
from femsolver.core.material import ElasticMaterial
from femsolver.core.mesh import BoundaryConditions, ElementData, Mesh
from femsolver.core.solver import StaticSolver
from femsolver.elements.hexa20 import Hexa20, _NODE_COORDS, _GAUSS_POINTS_3X3X3


# ---------------------------------------------------------------------------
# Géométrie de référence : cube [-1,1]³
# ---------------------------------------------------------------------------

# Les 20 nœuds du Hexa20 de référence = coordonnées naturelles
NODES_REF = _NODE_COORDS.copy()   # shape (20, 3)

# Cube unitaire [0,1]³ en coordonnées physiques
NODES_UNIT = np.array([
    # Coins (0–7) : même ordre que Hexa8
    [0., 0., 0.],  # 0
    [1., 0., 0.],  # 1
    [1., 1., 0.],  # 2
    [0., 1., 0.],  # 3
    [0., 0., 1.],  # 4
    [1., 0., 1.],  # 5
    [1., 1., 1.],  # 6
    [0., 1., 1.],  # 7
    # Milieux face ζ=0 (bas)
    [.5, 0., 0.],  # 8  : mid 0-1
    [1., .5, 0.],  # 9  : mid 1-2
    [.5, 1., 0.],  # 10 : mid 2-3
    [0., .5, 0.],  # 11 : mid 3-0
    # Milieux face ζ=1 (haut)
    [.5, 0., 1.],  # 12 : mid 4-5
    [1., .5, 1.],  # 13 : mid 5-6
    [.5, 1., 1.],  # 14 : mid 6-7
    [0., .5, 1.],  # 15 : mid 7-4
    # Milieux arêtes verticales
    [0., 0., .5],  # 16 : mid 0-4
    [1., 0., .5],  # 17 : mid 1-5
    [1., 1., .5],  # 18 : mid 2-6
    [0., 1., .5],  # 19 : mid 3-7
])

MAT_SIMPLE = ElasticMaterial(E=1.0, nu=0.0, rho=1.0)
MAT_STEEL  = ElasticMaterial(E=210e9, nu=0.3, rho=7800.0)


# ===========================================================================
# Tests des fonctions de forme
# ===========================================================================


class TestHexa20ShapeFunctions:
    """Propriétés algébriques des 20 fonctions de forme sérendipité 3D."""

    def test_partition_of_unity(self) -> None:
        """ΣNi = 1 à plusieurs points (ξ,η,ζ), dont les 27 points de Gauss."""
        elem = Hexa20()
        # Points de Gauss + quelques points intérieurs arbitraires
        test_pts = [(xi, eta, zeta, _) for xi, eta, zeta, _ in _GAUSS_POINTS_3X3X3]
        test_pts += [(0, 0, 0, 0), (0.5, -0.3, 0.7, 0), (-0.8, 0.2, -0.5, 0)]
        for xi, eta, zeta, _ in test_pts:
            N = elem._shape_functions(xi, eta, zeta)
            np.testing.assert_allclose(N.sum(), 1.0, atol=1e-13,
                err_msg=f"ΣNi ≠ 1 en ({xi:.2f},{eta:.2f},{zeta:.2f})")

    def test_nodal_interpolation(self) -> None:
        """Ni vaut 1 au nœud i et 0 aux 19 autres nœuds."""
        elem = Hexa20()
        for i in range(20):
            xi_i, eta_i, zeta_i = _NODE_COORDS[i]
            N = elem._shape_functions(xi_i, eta_i, zeta_i)
            assert abs(N[i] - 1.0) < 1e-12, (
                f"N{i}({xi_i},{eta_i},{zeta_i}) = {N[i]:.6g} ≠ 1"
            )
            for j in range(20):
                if j != i:
                    assert abs(N[j]) < 1e-12, (
                        f"N{j}({xi_i},{eta_i},{zeta_i}) = {N[j]:.6g} ≠ 0 "
                        f"(attendu nul, nœud {i})"
                    )

    def test_derivatives_sum_to_zero(self) -> None:
        """Σ ∂Ni/∂ξ = Σ ∂Ni/∂η = Σ ∂Ni/∂ζ = 0 (conséquence de ΣNi=1)."""
        elem = Hexa20()
        for xi, eta, zeta in [(0.1, -0.3, 0.5), (0.0, 0.0, 0.0), (-0.7, 0.4, -0.2)]:
            dN = elem._shape_function_derivatives(xi, eta, zeta)
            np.testing.assert_allclose(dN[0].sum(), 0.0, atol=1e-13)
            np.testing.assert_allclose(dN[1].sum(), 0.0, atol=1e-13)
            np.testing.assert_allclose(dN[2].sum(), 0.0, atol=1e-13)

    def test_quadratic_completeness(self) -> None:
        """Hexa20 interpole exactement tout polynôme quadratique 3D.

        Espace polynomial sérendipité : contient toute la base quadratique
        {1, x, y, z, xy, xz, yz, x², y², z²}.
        f(x,y,z) = 1+2x−y+z+x²−yz+xz (arbitraire, degré 2) est exacte.
        """
        elem = Hexa20()
        # Cube de référence : coords physiques = coords naturelles
        nodes = NODES_REF

        def f(x: float, y: float, z: float) -> float:
            return 1.0 + 2.0*x - y + z + x**2 - y*z + x*z

        f_nodal = np.array([f(x, y, z) for x, y, z in nodes])

        for xi, eta, zeta in [(0.1, 0.2, -0.3), (-0.5, 0.7, 0.1), (0.0, 0.0, 0.0)]:
            N = elem._shape_functions(xi, eta, zeta)
            x_phys = N @ nodes[:, 0]
            y_phys = N @ nodes[:, 1]
            z_phys = N @ nodes[:, 2]
            f_interp = N @ f_nodal
            np.testing.assert_allclose(
                f_interp, f(x_phys, y_phys, z_phys), atol=1e-12,
                err_msg=f"Complétude quadratique 3D échouée en ({xi},{eta},{zeta})"
            )

    def test_gauss_rule_weight_sum(self) -> None:
        """La somme des poids 3×3×3 = 8 = volume du cube [-1,1]³."""
        total = sum(w for _, _, _, w in _GAUSS_POINTS_3X3X3)
        np.testing.assert_allclose(total, 8.0, rtol=1e-14)

    def test_gauss_rule_count(self) -> None:
        """27 points de Gauss exactement."""
        assert len(_GAUSS_POINTS_3X3X3) == 27


# ===========================================================================
# Tests de la matrice de rigidité K_e
# ===========================================================================


class TestHexa20StiffnessMatrix:
    """Propriétés de la matrice de rigidité élémentaire 60×60."""

    def test_shape(self) -> None:
        """K_e est de shape (60, 60)."""
        K_e = Hexa20().stiffness_matrix(MAT_SIMPLE, NODES_UNIT, {})
        assert K_e.shape == (60, 60)

    def test_symmetry(self) -> None:
        """K_e est symétrique à précision machine."""
        K_e = Hexa20().stiffness_matrix(MAT_STEEL, NODES_UNIT, {})
        atol = 1e-9 * np.abs(K_e).max()
        np.testing.assert_allclose(K_e, K_e.T, atol=atol)

    def test_six_rigid_body_modes(self) -> None:
        """K_e a exactement 6 valeurs propres nulles (3 translations + 3 rotations).

        Pour un solide 3D, les modes rigides sont :
        - 3 translations (tx, ty, tz)
        - 3 rotations (Rx, Ry, Rz)
        Soit 6 modes au total → nul-espace de dimension 6.
        """
        K_e = Hexa20().stiffness_matrix(MAT_SIMPLE, NODES_UNIT, {})
        eigenvalues = np.linalg.eigvalsh(K_e)
        n_zero = np.sum(np.abs(eigenvalues) < 1e-9 * eigenvalues[-1])
        assert n_zero == 6, (
            f"K_e devrait avoir 6 valeurs propres nulles, trouvé {n_zero}.\n"
            f"6 plus petites: {eigenvalues[:8]}"
        )

    def test_positive_semidefinite(self) -> None:
        """K_e est semi-définie positive."""
        K_e = Hexa20().stiffness_matrix(MAT_STEEL, NODES_UNIT, {})
        eigenvalues = np.linalg.eigvalsh(K_e)
        tol = 1e-9 * eigenvalues[-1]
        assert np.all(eigenvalues >= -tol), (
            f"K_e a une valeur propre négative : {eigenvalues.min():.3e}"
        )

    def test_no_spurious_modes_27pts(self) -> None:
        """K_e a exactement 54 valeurs propres non-nulles (60 − 6 rigides).

        Avec 3×3×3=27 points de Gauss, on a 27×6=162 contraintes pour
        60−6=54 modes mécaniques → rang correct, pas de mode hourglass.
        """
        K_e = Hexa20().stiffness_matrix(MAT_SIMPLE, NODES_UNIT, {})
        eigenvalues = np.linalg.eigvalsh(K_e)
        n_nonzero = np.sum(np.abs(eigenvalues) > 1e-9 * eigenvalues[-1])
        assert n_nonzero == 54, (
            f"K_e doit avoir 54 modes non-rigides, trouvé {n_nonzero}."
        )

    def test_invalid_nodes_shape_raises(self) -> None:
        """nodes.shape ≠ (20,3) → ValueError."""
        with pytest.raises(ValueError, match="nodes.shape"):
            Hexa20().stiffness_matrix(MAT_SIMPLE, np.zeros((8, 3)), {})


# ===========================================================================
# Tests de la matrice de masse M_e
# ===========================================================================


class TestHexa20MassMatrix:
    """Propriétés de la matrice de masse consistante 60×60."""

    def test_shape(self) -> None:
        """M_e est de shape (60, 60)."""
        M_e = Hexa20().mass_matrix(MAT_STEEL, NODES_UNIT, {})
        assert M_e.shape == (60, 60)

    def test_symmetry(self) -> None:
        """M_e est symétrique."""
        M_e = Hexa20().mass_matrix(MAT_STEEL, NODES_UNIT, {})
        np.testing.assert_allclose(M_e, M_e.T, atol=1e-14)

    def test_total_mass_conservation(self) -> None:
        """Somme de toutes les entrées = 3 × masse totale.

        Pour un champ uniforme u=[1,0,0,1,0,0,...], la force M·u donne la
        masse totale dans chaque direction. La somme de toutes les entrées
        de M = 3 × (ρ · Volume) (3 composantes : ux, uy, uz).
        """
        rho = 2700.0
        mat = ElasticMaterial(E=1.0, nu=0.3, rho=rho)
        volume = 1.0   # cube [0,1]³
        M_e = Hexa20().mass_matrix(mat, NODES_UNIT, {})
        expected = 3.0 * rho * volume
        np.testing.assert_allclose(M_e.sum(), expected, rtol=1e-10)

    def test_positive_semidefinite(self) -> None:
        """M_e est semi-définie positive."""
        M_e = Hexa20().mass_matrix(MAT_STEEL, NODES_UNIT, {})
        eigenvalues = np.linalg.eigvalsh(M_e)
        assert np.all(eigenvalues >= -1e-14)


# ===========================================================================
# Patch test 3D — champs linéaire et quadratique
# ===========================================================================


class TestHexa20PatchTest:
    """Patch tests : Hexa20 reproduit exactement les polynômes 1er et 2e degré.

    Hexa20 est complet jusqu'au degré 2 en 3D → les champs linéaires ET
    quadratiques sont représentés sans erreur d'interpolation.
    """

    def test_linear_displacement_patch(self) -> None:
        """ε = B · u_e reproduit εxx=α, εyy=β, εzz=γ pour champ linéaire.

        Champ : ux=α·x, uy=β·y, uz=γ·z.
        Déformations : εxx=α, εyy=β, εzz=γ, γyz=γxz=γxy=0.
        """
        alpha, beta, gamma_d = 0.002, -0.001, 0.0015
        elem = Hexa20()
        nodes = NODES_UNIT.copy()

        u_e = np.zeros(60)
        for i, (x, y, z) in enumerate(nodes):
            u_e[3 * i    ] = alpha   * x
            u_e[3 * i + 1] = beta    * y
            u_e[3 * i + 2] = gamma_d * z

        for xi, eta, zeta in [(0.0, 0.0, 0.0), (0.5, -0.5, 0.3), (-0.7, 0.2, 0.8)]:
            eps = elem.strain(nodes, u_e, xi, eta, zeta)
            np.testing.assert_allclose(eps[0], alpha,   atol=1e-12, err_msg="εxx")
            np.testing.assert_allclose(eps[1], beta,    atol=1e-12, err_msg="εyy")
            np.testing.assert_allclose(eps[2], gamma_d, atol=1e-12, err_msg="εzz")
            np.testing.assert_allclose(eps[3], 0.0,     atol=1e-12, err_msg="γyz")
            np.testing.assert_allclose(eps[4], 0.0,     atol=1e-12, err_msg="γxz")
            np.testing.assert_allclose(eps[5], 0.0,     atol=1e-12, err_msg="γxy")

    def test_quadratic_displacement_patch(self) -> None:
        """Hexa20 reproduit un champ de déplacement quadratique.

        Champ : ux = a·x², uy = b·y², uz = c·z² (purement quadratique diagonal).
        Déformations attendues :
            εxx = 2a·x,  εyy = 2b·y,  εzz = 2c·z,  γ = 0.
        """
        a, b, c = 0.001, -0.0005, 0.0008
        elem = Hexa20()
        nodes = NODES_UNIT.copy()

        u_e = np.zeros(60)
        for i, (x, y, z) in enumerate(nodes):
            u_e[3 * i    ] = a * x**2
            u_e[3 * i + 1] = b * y**2
            u_e[3 * i + 2] = c * z**2

        for xi, eta, zeta in [(0.0, 0.0, 0.0), (0.5, 0.3, -0.4)]:
            N = elem._shape_functions(xi, eta, zeta)
            x = N @ nodes[:, 0]
            y = N @ nodes[:, 1]
            z = N @ nodes[:, 2]
            eps = elem.strain(nodes, u_e, xi, eta, zeta)
            np.testing.assert_allclose(eps[0], 2.0*a*x, atol=1e-11, err_msg="εxx quad")
            np.testing.assert_allclose(eps[1], 2.0*b*y, atol=1e-11, err_msg="εyy quad")
            np.testing.assert_allclose(eps[2], 2.0*c*z, atol=1e-11, err_msg="εzz quad")

    def test_stress_linear_patch(self) -> None:
        """σ = D · ε correct pour un champ linéaire (nu=0 → pas de couplage)."""
        E = 1e6
        mat = ElasticMaterial(E=E, nu=0.0, rho=1.0)
        alpha = 0.001
        elem = Hexa20()
        nodes = NODES_UNIT.copy()

        u_e = np.zeros(60)
        for i, (x, y, z) in enumerate(nodes):
            u_e[3 * i] = alpha * x   # ux = α·x → εxx = α

        sigma = elem.stress(mat, nodes, u_e)
        np.testing.assert_allclose(sigma[0], E * alpha, rtol=1e-10, err_msg="σxx")
        np.testing.assert_allclose(sigma[1], 0.0, atol=1e-8, err_msg="σyy")
        np.testing.assert_allclose(sigma[2], 0.0, atol=1e-8, err_msg="σzz")


# ===========================================================================
# Traction uniaxiale assemblée — cube unique
# ===========================================================================


class TestHexa20TractionSingleElement:
    """Un seul Hexa20 sous traction uniforme — solution analytique exacte.

    Géométrie : cube [0,1]³.
    Matériau  : E, ν = 0.
    Chargement: traction σxx = P sur la face x=1 (nœuds 1,2,6,5,9,18,13,17).
    Solution  : ux = P·x/E, uy = 0, uz = 0 (car ν=0).

    Comme Hexa20 est complet quadratique et que le champ solution est linéaire,
    la solution FEM est exacte à précision machine.
    """

    E = 1e6
    nu = 0.0   # ν=0 → pas de couplage, solution analytique simple
    rho = 1.0
    P = 1000.0   # pression [Pa] sur face x=1

    def _build_mesh_and_bc(self) -> tuple[Mesh, BoundaryConditions]:
        """Un seul Hexa20 sur [0,1]³.

        BCs :
        - Face x=0 : encastrement complet (ux=uy=uz=0). Pour ν=0, la traction
          n'induit pas de contraction transverse, donc bloquer uy,uz sur x=0
          ne modifie pas la solution.
        - Face x=1 : forces nodales consistantes pour une traction uniforme P.

        Calcul des forces nodales consistantes sur la face x=1 (Quad8 2D) :
        La face est un Quad8 en (η,ζ) ∈ [−1,1]² avec Jacobien = 1/4.
            ∫∫ N_coin   dη dζ = −1/3  (négatif — connu pour les éléments Q8)
            ∫∫ N_milieu dη dζ = +4/3
        Forces physiques (× Jacobien 1/4 × P) :
            F_coin   = P × (−1/3) × (1/4) = −P/12
            F_milieu = P × (+4/3) × (1/4) = +P/3
        Total : 4×(−P/12) + 4×(P/3) = −P/3 + 4P/3 = P ✓
        """
        mat = ElasticMaterial(E=self.E, nu=self.nu, rho=self.rho)
        elem = ElementData(etype=Hexa20, node_ids=tuple(range(20)),
                           material=mat, properties={})
        mesh = Mesh(nodes=NODES_UNIT, elements=(elem,), n_dim=3)

        # Encastrement complet sur la face x=0
        left_nodes = [i for i, (x, y, z) in enumerate(NODES_UNIT) if abs(x) < 1e-9]
        dirichlet: dict = {nid: {0: 0.0, 1: 0.0, 2: 0.0} for nid in left_nodes}

        # Forces consistantes sur la face x=1
        # Coins face x=1 : nœuds 1,2,6,5
        # Milieux d'arête face x=1 : nœuds 9(y=0.5,z=0), 18(y=1,z=0.5),
        #                                    13(y=0.5,z=1), 17(y=0,z=0.5)
        P_face = self.P
        F_corner  = -P_face / 12.0   # forces négatives aux coins (propriété Q8)
        F_midside = +P_face / 3.0    # forces positives aux milieux d'arête

        neumann: dict = {}
        for nid in [1, 2, 6, 5]:
            neumann[nid] = {0: F_corner}
        for nid in [9, 18, 13, 17]:
            neumann[nid] = {0: F_midside}

        bc = BoundaryConditions(dirichlet=dirichlet, neumann=neumann)
        return mesh, bc

    def test_axial_displacement(self) -> None:
        """ux(x=1) = P/E pour une traction uniaxiale (ν=0).

        Avec les forces nodales consistantes et ν=0, Hexa20 reproduit
        exactement le champ linéaire ux = P·x/E à précision machine.
        """
        mesh, bc = self._build_mesh_and_bc()
        assembler = Assembler(mesh)
        K = assembler.assemble_stiffness()
        F = assembler.assemble_forces(bc)
        K_bc, F_bc = apply_dirichlet(K, F, mesh, bc)
        u = StaticSolver().solve(K_bc, F_bc)

        right_nodes = [i for i, (x, y, z) in enumerate(NODES_UNIT) if abs(x - 1.0) < 1e-9]
        ux_right = np.mean([u[3 * nid] for nid in right_nodes])
        ux_analytic = self.P / self.E

        np.testing.assert_allclose(ux_right, ux_analytic, rtol=1e-8,
            err_msg=f"ux(x=1) = {ux_right:.6g}, attendu {ux_analytic:.6g}")

    def test_no_transverse_displacement(self) -> None:
        """uy = uz = 0 partout pour ν=0 sous traction uniaxiale.

        Avec encastrement complet sur x=0 et ν=0, pas de contraction de
        Poisson → tous les déplacements transverses sont nuls à machine-epsilon.
        """
        mesh, bc = self._build_mesh_and_bc()
        assembler = Assembler(mesh)
        K = assembler.assemble_stiffness()
        F = assembler.assemble_forces(bc)
        K_bc, F_bc = apply_dirichlet(K, F, mesh, bc)
        u = StaticSolver().solve(K_bc, F_bc)

        ux_max = max(abs(u[3 * nid]) for nid in range(20))
        for nid in range(20):
            uy = u[3 * nid + 1]
            uz = u[3 * nid + 2]
            np.testing.assert_allclose(uy, 0.0, atol=1e-8 * ux_max,
                err_msg=f"uy≠0 au nœud {nid}")
            np.testing.assert_allclose(uz, 0.0, atol=1e-8 * ux_max,
                err_msg=f"uz≠0 au nœud {nid}")


# ===========================================================================
# Tests batch
# ===========================================================================


class TestHexa20Batch:
    """Cohérence entre stiffness_matrix() et batch_stiffness_matrix()."""

    def test_batch_stiffness_matches_scalar(self) -> None:
        """batch_stiffness_matrix produit les mêmes K_e que la méthode scalaire."""
        mat = ElasticMaterial(E=210e9, nu=0.3, rho=7800)
        D = mat.elasticity_matrix_3d()

        rng = np.random.default_rng(13)
        n_e = 4
        nodes_batch = NODES_UNIT[np.newaxis, :, :].repeat(n_e, axis=0)
        nodes_batch += rng.uniform(-0.03, 0.03, nodes_batch.shape)

        K_batch = Hexa20.batch_stiffness_matrix(nodes_batch, D)

        for e in range(n_e):
            K_scalar = Hexa20().stiffness_matrix(mat, nodes_batch[e], {})
            np.testing.assert_allclose(
                K_batch[e], K_scalar, rtol=1e-11,
                err_msg=f"Batch vs scalar divergent pour l'élément {e}"
            )

    def test_batch_mass_matches_scalar(self) -> None:
        """batch_mass_matrix produit les mêmes M_e que la méthode scalaire."""
        mat = ElasticMaterial(E=1.0, nu=0.3, rho=2700)
        n_e = 3
        rng = np.random.default_rng(99)
        nodes_batch = NODES_UNIT[np.newaxis, :, :].repeat(n_e, axis=0)
        nodes_batch += rng.uniform(-0.02, 0.02, nodes_batch.shape)

        M_batch = Hexa20.batch_mass_matrix(nodes_batch, mat.rho)
        for e in range(n_e):
            M_scalar = Hexa20().mass_matrix(mat, nodes_batch[e], {})
            np.testing.assert_allclose(
                M_batch[e], M_scalar, rtol=1e-11,
                err_msg=f"Batch masse vs scalar divergent pour l'élément {e}"
            )

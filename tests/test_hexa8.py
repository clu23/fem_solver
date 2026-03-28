"""Tests unitaires pour l'élément Hexa8 — validation par solution analytique.

Structure des tests
-------------------
1. TestHexa8ShapeFunctions   : partition de l'unité, interpolation de position,
                               valeurs aux nœuds (Nᵢ = δᵢⱼ).
2. TestHexa8BMatrix          : shape (6×24), symétrie, valeurs au centre du cube
                               unitaire (J = ½ I₃ → dN_phys connus analytiquement).
3. TestHexa8StiffnessMatrix  : shape, symétrie, 6 modes rigides, cohérence avec
                               solution analytique de traction uniaxiale.
4. TestHexa8MassMatrix       : shape, symétrie, conservation de la masse totale.
5. TestHexa8PatchTest        : assemblage + résolution statique sur cube [0,1]³
                               en Hexa8 unique — comparaison à u_z = P/E · z
                               (traction axiale, solution exacte).

Solutions analytiques de référence
-----------------------------------
Cube unitaire [0,1]³ en Hexa8 (nœuds aux coins) :
  - Volume V = 1
  - J(centre) = ½ I₃  → det(J) = 0.125

Traction uniaxiale (E=1, ν=0, σ_zz = σ₀ sur cube [0,1]³) :
  - u_z = σ₀/E · z,  u_x = u_y = 0  (solution exacte car ν=0, linéaire en z)
  - σ = [0, 0, σ₀, 0, 0, 0]  (uniforme dans l'élément)

Masse totale d'un cube a×b×c de densité ρ :
  - m = ρ · a · b · c
"""

from __future__ import annotations

import numpy as np
import pytest

from femsolver.core.assembler import Assembler
from femsolver.core.boundary import apply_dirichlet
from femsolver.core.material import ElasticMaterial
from femsolver.core.mesh import BoundaryConditions, ElementData, Mesh
from femsolver.core.solver import StaticSolver
from femsolver.elements.hexa8 import Hexa8


# ---------------------------------------------------------------------------
# Données communes
# ---------------------------------------------------------------------------

MAT_SIMPLE = ElasticMaterial(E=1.0, nu=0.0, rho=1.0)
MAT_STEEL  = ElasticMaterial(E=210e9, nu=0.3, rho=7800.0)

# Cube unitaire [0,1]³ — numérotation standard Hexa8
#  face ζ=-1 : nœuds 0,1,2,3 (z=0)
#  face ζ=+1 : nœuds 4,5,6,7 (z=1)
NODES_UNIT: np.ndarray = np.array([
    [0.0, 0.0, 0.0],   # 0  bas-avant-gauche
    [1.0, 0.0, 0.0],   # 1  bas-avant-droit
    [1.0, 1.0, 0.0],   # 2  bas-arrière-droit
    [0.0, 1.0, 0.0],   # 3  bas-arrière-gauche
    [0.0, 0.0, 1.0],   # 4  haut-avant-gauche
    [1.0, 0.0, 1.0],   # 5  haut-avant-droit
    [1.0, 1.0, 1.0],   # 6  haut-arrière-droit
    [0.0, 1.0, 1.0],   # 7  haut-arrière-gauche
])

PROPS_EMPTY: dict = {}


# ---------------------------------------------------------------------------
# 1. Fonctions de forme
# ---------------------------------------------------------------------------


class TestHexa8ShapeFunctions:
    """Validation des fonctions de forme trilinéaires Nᵢ(ξ,η,ζ)."""

    def test_partition_of_unity_at_center(self) -> None:
        """Σ Nᵢ(0,0,0) = 1 (partition de l'unité au centre)."""
        N = Hexa8._shape_functions(0.0, 0.0, 0.0)
        np.testing.assert_allclose(N.sum(), 1.0, rtol=1e-15)

    def test_partition_of_unity_at_gauss_points(self) -> None:
        """Σ Nᵢ = 1 à chacun des 8 points de Gauss."""
        gp = 1.0 / np.sqrt(3.0)
        for xi in (-gp, gp):
            for eta in (-gp, gp):
                for zeta in (-gp, gp):
                    N = Hexa8._shape_functions(xi, eta, zeta)
                    np.testing.assert_allclose(
                        N.sum(), 1.0, rtol=1e-15,
                        err_msg=f"Partition de l'unité violée en ({xi},{eta},{zeta})"
                    )

    def test_nodal_interpolation(self) -> None:
        """Nᵢ(ξⱼ, ηⱼ, ζⱼ) = δᵢⱼ (Kronecker delta aux nœuds)."""
        # Coordonnées naturelles des 8 nœuds
        node_nat = np.array([
            (-1., -1., -1.), (1., -1., -1.), (1.,  1., -1.), (-1.,  1., -1.),
            (-1., -1.,  1.), (1., -1.,  1.), (1.,  1.,  1.), (-1.,  1.,  1.),
        ])
        for j, (xi, eta, zeta) in enumerate(node_nat):
            N = Hexa8._shape_functions(xi, eta, zeta)
            np.testing.assert_allclose(
                N[j], 1.0, atol=1e-15, err_msg=f"N[{j}] devrait valoir 1 au nœud {j}"
            )
            other = np.delete(N, j)
            np.testing.assert_allclose(
                other, 0.0, atol=1e-15, err_msg=f"N[i≠{j}] devrait valoir 0 au nœud {j}"
            )

    def test_isoparametric_coordinate_interpolation(self) -> None:
        """x = Σ Nᵢ xᵢ : interpolation des coordonnées physiques du cube [0,1]³."""
        # Au centre (ξ=η=ζ=0), on doit obtenir le barycentre (0.5, 0.5, 0.5)
        N = Hexa8._shape_functions(0.0, 0.0, 0.0)
        x_interp = N @ NODES_UNIT
        np.testing.assert_allclose(x_interp, [0.5, 0.5, 0.5], atol=1e-15)

    def test_shape_function_derivatives_sum_zero(self) -> None:
        """Σᵢ ∂Nᵢ/∂ξ = 0 (car Σ Nᵢ = 1 est constante)."""
        dN = Hexa8._shape_function_derivatives(0.3, -0.2, 0.7)
        np.testing.assert_allclose(dN.sum(axis=1), 0.0, atol=1e-15)


# ---------------------------------------------------------------------------
# 2. Matrice B
# ---------------------------------------------------------------------------


class TestHexa8BMatrix:
    """Vérification de la matrice déformation–déplacement B (6×24)."""

    def test_shape(self) -> None:
        """B est de shape (6, 24)."""
        B, _ = Hexa8()._strain_displacement_matrix(0.0, 0.0, 0.0, NODES_UNIT)
        assert B.shape == (6, 24)

    def test_det_j_unit_cube_center(self) -> None:
        """Au centre du cube [0,1]³, J = ½ I₃ → det(J) = 0.125.

        La transformation ξ = 2x-1 donne ∂x/∂ξ = ½, d'où J = ½ I₃ pour un
        cube unité orienté selon les axes.
        """
        _, det_J = Hexa8()._strain_displacement_matrix(0.0, 0.0, 0.0, NODES_UNIT)
        np.testing.assert_allclose(det_J, 0.125, rtol=1e-12)

    def test_det_j_scaled_cube(self) -> None:
        """Cube a×b×c → det(J) = a·b·c/8 au centre.

        Pour un cube (a,b,c), J = diag(a/2, b/2, c/2) → det = abc/8.
        """
        a, b, c = 2.0, 3.0, 5.0
        nodes = NODES_UNIT * np.array([a, b, c])
        _, det_J = Hexa8()._strain_displacement_matrix(0.0, 0.0, 0.0, nodes)
        np.testing.assert_allclose(det_J, a * b * c / 8.0, rtol=1e-12)

    def test_b_encodes_rigid_body_correctly(self) -> None:
        """B · u_rigid = 0 pour un déplacement de corps rigide.

        Un déplacement de translation pure (u_x=1 constant, reste nul) doit
        donner ε = 0 (pas de déformation).
        """
        B, _ = Hexa8()._strain_displacement_matrix(0.0, 0.0, 0.0, NODES_UNIT)
        # Translation u_x = 1 pour tous les nœuds : u_e[0::3] = 1, reste = 0
        u_rigid = np.zeros(24)
        u_rigid[0::3] = 1.0
        epsilon = B @ u_rigid
        np.testing.assert_allclose(epsilon, 0.0, atol=1e-14)

    def test_b_linear_field_uniaxial_z(self) -> None:
        """B · u_linear donne εzz correct pour u_z = z.

        Pour u_z = z (déformation axiale selon z unitaire) et E=1, ν=0 :
            ε_zz = ∂uz/∂z = 1  (uniquement εzz ≠ 0)

        On construit u_e tel que w_i = z_i (nœuds 0..3 ont z=0, nœuds 4..7 ont z=1).
        """
        u_e = np.zeros(24)
        for i in range(8):
            u_e[3 * i + 2] = NODES_UNIT[i, 2]   # w_i = z_i

        B, _ = Hexa8()._strain_displacement_matrix(0.0, 0.0, 0.0, NODES_UNIT)
        epsilon = B @ u_e
        # εzz = 1, tous les autres composantes = 0
        np.testing.assert_allclose(epsilon[2], 1.0, atol=1e-14, err_msg="εzz doit valoir 1")
        np.testing.assert_allclose(
            epsilon[[0, 1, 3, 4, 5]], 0.0, atol=1e-14,
            err_msg="Autres composantes de déformation doivent être nulles"
        )

    def test_invalid_node_shape(self) -> None:
        """ValueError si nodes.shape ≠ (8, 3)."""
        bad_nodes = np.zeros((6, 3))
        with pytest.raises(ValueError, match="8"):
            Hexa8().stiffness_matrix(MAT_SIMPLE, bad_nodes, {})

    def test_negative_det_j(self) -> None:
        """ValueError si l'ordre des nœuds donne det(J) < 0."""
        nodes_inverted = NODES_UNIT[[1, 0, 2, 3, 5, 4, 6, 7]]  # échange 0↔1 et 4↔5
        with pytest.raises(ValueError, match="[Dd]et|[Oo]rientation"):
            Hexa8().stiffness_matrix(MAT_SIMPLE, nodes_inverted, {})


# ---------------------------------------------------------------------------
# 3. Matrice de rigidité
# ---------------------------------------------------------------------------


class TestHexa8StiffnessMatrix:
    """Vérification de la matrice K_e (24×24)."""

    def test_shape(self) -> None:
        """K_e est de shape (24, 24)."""
        K_e = Hexa8().stiffness_matrix(MAT_STEEL, NODES_UNIT, {})
        assert K_e.shape == (24, 24)

    def test_symmetry(self) -> None:
        """K_e est symétrique : K_e = K_eᵀ."""
        K_e = Hexa8().stiffness_matrix(MAT_STEEL, NODES_UNIT, {})
        np.testing.assert_allclose(K_e, K_e.T, atol=1e-8)

    def test_six_zero_eigenvalues(self) -> None:
        """K_e possède exactement 6 valeurs propres nulles (modes rigides).

        Un hexaèdre 3D libre a 6 modes de corps rigide :
        3 translations (ux, uy, uz) + 3 rotations (Rx, Ry, Rz).
        Les 18 autres valeurs propres sont strictement positives.
        """
        K_e = Hexa8().stiffness_matrix(MAT_STEEL, NODES_UNIT, {})
        eigenvalues = np.linalg.eigvalsh(K_e)
        eigenvalues_sorted = np.sort(eigenvalues)
        # Les 6 premières doivent être proches de 0
        np.testing.assert_allclose(
            eigenvalues_sorted[:6], 0.0, atol=1e-2,
            err_msg="Les 6 modes rigides doivent avoir une fréquence nulle"
        )
        # Les 18 restantes doivent être strictement positives
        assert np.all(eigenvalues_sorted[6:] > 0.0), \
            "Les 18 modes déformables doivent avoir des valeurs propres positives"

    def test_stiffness_scales_with_E(self) -> None:
        """K_e est proportionnel à E : K(2E) = 2 · K(E)."""
        mat1 = ElasticMaterial(E=1.0,   nu=0.3, rho=1.0)
        mat2 = ElasticMaterial(E=2.0,   nu=0.3, rho=1.0)
        K1 = Hexa8().stiffness_matrix(mat1, NODES_UNIT, {})
        K2 = Hexa8().stiffness_matrix(mat2, NODES_UNIT, {})
        np.testing.assert_allclose(K2, 2.0 * K1, rtol=1e-12)

    def test_stiffness_scales_with_volume(self) -> None:
        """K_e est proportionnel au volume pour un cube homothétique.

        Pour un cube de côté a, V = a³ et K ∝ a (pas a³ : K ~ E·A/L).
        Pour un cube unité vs cube de côté 2 (acier, nu=0) :
        - K(2) = K(1) × 1/2  car L double, A quadruple → K ∝ A/L ∝ a
        """
        mat = ElasticMaterial(E=1.0, nu=0.0, rho=1.0)
        nodes2 = NODES_UNIT * 2.0
        K1 = Hexa8().stiffness_matrix(mat, NODES_UNIT, {})
        K2 = Hexa8().stiffness_matrix(mat, nodes2, {})
        # Pour un cube homothétique : K(a) = K(1) × 1  (K ∝ E·a, mais a/a² → a/a² = 1/a)
        # Plus précisément : K ~ ∫ B^T D B dV ; B ~ 1/a, dV ~ a³ → K ~ a
        np.testing.assert_allclose(K2, K1 * 2.0, rtol=1e-12)


# ---------------------------------------------------------------------------
# 4. Matrice de masse
# ---------------------------------------------------------------------------


class TestHexa8MassMatrix:
    """Vérification de la matrice de masse consistante M_e (24×24)."""

    def test_shape(self) -> None:
        """M_e est de shape (24, 24)."""
        M_e = Hexa8().mass_matrix(MAT_SIMPLE, NODES_UNIT, {})
        assert M_e.shape == (24, 24)

    def test_symmetry(self) -> None:
        """M_e est symétrique."""
        M_e = Hexa8().mass_matrix(MAT_STEEL, NODES_UNIT, {})
        np.testing.assert_allclose(M_e, M_e.T, atol=1e-8)

    def test_total_mass_unit_cube(self) -> None:
        """Masse totale = ρV = 1×1 = 1 kg pour le cube unitaire (ρ=1, V=1).

        La somme de toutes les entrées de M_e, divisée par le nombre de
        directions (3), doit valoir ρV.
        """
        M_e = Hexa8().mass_matrix(MAT_SIMPLE, NODES_UNIT, {})
        # Σ M_e / 3 = ρV  (car chaque direction contribue identiquement)
        np.testing.assert_allclose(M_e.sum() / 3.0, 1.0, rtol=1e-12)

    def test_total_mass_scaled_cube(self) -> None:
        """Masse totale = ρ·a·b·c pour un cube a×b×c.

        Référence analytique : m = ρ × volume.
        """
        rho = 2700.0   # aluminium
        a, b, c = 0.3, 0.5, 0.8
        mat = ElasticMaterial(E=70e9, nu=0.33, rho=rho)
        nodes = NODES_UNIT * np.array([a, b, c])
        M_e = Hexa8().mass_matrix(mat, nodes, {})
        expected_mass = rho * a * b * c
        np.testing.assert_allclose(M_e.sum() / 3.0, expected_mass, rtol=1e-10)

    def test_positive_definite(self) -> None:
        """M_e est définie positive (toutes valeurs propres > 0)."""
        M_e = Hexa8().mass_matrix(MAT_STEEL, NODES_UNIT, {})
        eigenvalues = np.linalg.eigvalsh(M_e)
        assert np.all(eigenvalues > 0.0), \
            f"M_e doit être définie positive; min(λ) = {eigenvalues.min():.3g}"

    def test_mass_scales_with_rho(self) -> None:
        """M_e est proportionnel à ρ."""
        mat1 = ElasticMaterial(E=1.0, nu=0.3, rho=1000.0)
        mat2 = ElasticMaterial(E=1.0, nu=0.3, rho=3000.0)
        M1 = Hexa8().mass_matrix(mat1, NODES_UNIT, {})
        M2 = Hexa8().mass_matrix(mat2, NODES_UNIT, {})
        np.testing.assert_allclose(M2, 3.0 * M1, rtol=1e-12)


# ---------------------------------------------------------------------------
# 5. Patch test — cube [0,1]³ en traction uniaxiale
# ---------------------------------------------------------------------------


class TestHexa8PatchTest:
    """Patch test : cube [0,1]³ (un seul Hexa8) en traction axiale selon z.

    Configuration
    -------------
    - Un seul élément Hexa8 (cube unitaire).
    - Base (z=0, nœuds 0..3) : entièrement encastrée (ux=uy=uz=0).
    - Dessus (z=1, nœuds 4..7) : force verticale F_z = σ₀/4 par nœud
      (équivalent à une pression uniforme σ₀ = 1 Pa sur la face).

    Solution analytique (E=1, ν=0, σ₀=1)
    --------------------------------------
    - u_z(z) = σ₀/E · z = z  → u_z = 1 m pour les nœuds du dessus
    - u_x = u_y = 0 (pas d'effet Poisson car ν=0)
    - σ = [0, 0, 1, 0, 0, 0] dans tout l'élément

    L'élément Hexa8 représente exactement les champs linéaires en z
    (ses fonctions de forme contiennent ζ) → erreur numérique ≈ 0.
    """

    def _setup_cube_tension(
        self,
        E: float = 1.0,
        nu: float = 0.0,
        sigma0: float = 1.0,
    ) -> tuple[Mesh, BoundaryConditions]:
        """Construit le maillage et les CL pour le patch test."""
        mat = ElasticMaterial(E=E, nu=nu, rho=1.0)

        elements = (
            ElementData(
                etype=Hexa8,
                node_ids=tuple(range(8)),
                material=mat,
                properties={},
            ),
        )
        mesh = Mesh(nodes=NODES_UNIT.copy(), elements=elements, n_dim=3)

        # Conditions aux limites
        # Base (z=0) : encastrement complet (ux=uy=uz=0) sur les 4 nœuds 0..3
        dirichlet: dict[int, dict[int, float]] = {
            0: {0: 0.0, 1: 0.0, 2: 0.0},
            1: {0: 0.0, 1: 0.0, 2: 0.0},
            2: {0: 0.0, 1: 0.0, 2: 0.0},
            3: {0: 0.0, 1: 0.0, 2: 0.0},
        }

        # Dessus (z=1) : force F_z = σ₀ × aire / 4 nœuds = σ₀/4 par nœud
        # (charge cohérente avec le théorème de la convergence de la force)
        f_nodal = sigma0 / 4.0
        neumann: dict[int, dict[int, float]] = {
            4: {2: f_nodal},
            5: {2: f_nodal},
            6: {2: f_nodal},
            7: {2: f_nodal},
        }

        bc = BoundaryConditions(dirichlet=dirichlet, neumann=neumann)
        return mesh, bc

    def test_axial_displacement(self) -> None:
        """u_z des nœuds du dessus = σ₀/E · 1 = 1 m (solution exacte).

        Référence : δ = FL/(EA) = (σ₀ · A) · L / (E · A) = σ₀ · L / E.
        Pour L=1, E=1, σ₀=1 : δ = 1 m.
        L'élément Hexa8 reproduit exactement ce champ linéaire en z.
        """
        mesh, bc = self._setup_cube_tension(E=1.0, nu=0.0, sigma0=1.0)
        assembler = Assembler(mesh)
        K = assembler.assemble_stiffness()
        F = assembler.assemble_forces(bc)
        K_bc, F_bc = apply_dirichlet(K, F, mesh, bc)
        u = StaticSolver().solve(K_bc, F_bc)

        # Nœuds du dessus (4..7) : u_z (DDL 2) doit valoir 1 m
        for node in range(4, 8):
            dof_z = 3 * node + 2
            np.testing.assert_allclose(
                u[dof_z], 1.0, atol=1e-10,
                err_msg=f"u_z au nœud {node} doit valoir 1 m"
            )

    def test_no_lateral_displacement_nu0(self) -> None:
        """u_x = u_y = 0 partout pour ν = 0 (pas d'effet Poisson).

        Pour ν=0, σzz ne génère pas de déformation transversale.
        Les DDL x et y des nœuds non encastrés doivent rester nuls.
        """
        mesh, bc = self._setup_cube_tension(E=1.0, nu=0.0, sigma0=1.0)
        assembler = Assembler(mesh)
        K = assembler.assemble_stiffness()
        F = assembler.assemble_forces(bc)
        K_bc, F_bc = apply_dirichlet(K, F, mesh, bc)
        u = StaticSolver().solve(K_bc, F_bc)

        # Nœuds du dessus (4..7) : u_x et u_y doivent être nuls
        for node in range(4, 8):
            np.testing.assert_allclose(
                u[3 * node    ], 0.0, atol=1e-10,
                err_msg=f"u_x au nœud {node} doit être nul (ν=0)"
            )
            np.testing.assert_allclose(
                u[3 * node + 1], 0.0, atol=1e-10,
                err_msg=f"u_y au nœud {node} doit être nul (ν=0)"
            )

    def test_uniform_stress_field(self) -> None:
        """σzz = σ₀ = 1 Pa dans l'élément, autres composantes nulles.

        La contrainte doit être uniforme (σzz = 1) car le champ de
        déplacement linéaire en z est représenté exactement par Hexa8.
        """
        mesh, bc = self._setup_cube_tension(E=1.0, nu=0.0, sigma0=1.0)
        assembler = Assembler(mesh)
        K = assembler.assemble_stiffness()
        F = assembler.assemble_forces(bc)
        K_bc, F_bc = apply_dirichlet(K, F, mesh, bc)
        u = StaticSolver().solve(K_bc, F_bc)

        # Extraire u_e pour l'élément unique
        dofs = list(range(24))
        u_e = u[dofs]

        mat = ElasticMaterial(E=1.0, nu=0.0, rho=1.0)
        # Évaluer la contrainte au centre de l'élément
        sigma = Hexa8().stress(mat, NODES_UNIT, u_e, xi=0.0, eta=0.0, zeta=0.0)
        # σzz = 1 Pa
        np.testing.assert_allclose(sigma[2], 1.0, atol=1e-10, err_msg="σzz doit valoir 1 Pa")
        # Autres composantes nulles
        np.testing.assert_allclose(
            sigma[[0, 1, 3, 4, 5]], 0.0, atol=1e-10,
            err_msg="Toutes les autres contraintes doivent être nulles"
        )

    def test_poisson_effect_nonzero_nu(self) -> None:
        """Avec ν ≠ 0 : contraction latérale des nœuds du dessus.

        Solution analytique pour traction uniaxiale de contrainte σ₀ :
            ε_xx = ε_yy = -ν · σ₀/E  (contraction transversale)
            → u_x(x=1) = -ν · σ₀/E  (nœuds 1,2,5,6 ont x=1)
            → u_y(y=1) = -ν · σ₀/E  (nœuds 2,3,6,7 ont y=1)

        Mais attention : ici les nœuds de la BASE sont encastrés en x,y,z.
        L'encastrement de la base empêche la contraction transversale en z=0,
        ce qui crée un état de contrainte non uniforme. Le test est donc
        qualitatif : la contraction latérale en z=1 doit être négative.
        """
        nu = 0.3
        mesh, bc = self._setup_cube_tension(E=1.0, nu=nu, sigma0=1.0)
        assembler = Assembler(mesh)
        K = assembler.assemble_stiffness()
        F = assembler.assemble_forces(bc)
        K_bc, F_bc = apply_dirichlet(K, F, mesh, bc)
        u = StaticSolver().solve(K_bc, F_bc)

        # Les nœuds du dessus avec x=1 (nœuds 5,6) doivent avoir u_x < 0
        u_x_5 = u[3 * 5]       # nœud 5 (x=1, y=0, z=1)
        u_x_6 = u[3 * 6]       # nœud 6 (x=1, y=1, z=1)
        assert u_x_5 < 0, f"Contraction latérale attendue en nœud 5, u_x={u_x_5}"
        assert u_x_6 < 0, f"Contraction latérale attendue en nœud 6, u_x={u_x_6}"

    def test_equilibrium_base_reactions(self) -> None:
        """Les réactions à la base équilibrent la charge appliquée.

        Équilibre statique : somme des forces en z = 0.
        Force appliquée = σ₀ = 1 N. Réactions = -1 N (total).
        """
        mesh, bc = self._setup_cube_tension(E=1.0, nu=0.0, sigma0=1.0)
        assembler = Assembler(mesh)
        K = assembler.assemble_stiffness()
        F_ext = assembler.assemble_forces(bc)
        K_bc, F_bc = apply_dirichlet(K, F_ext, mesh, bc)
        u = StaticSolver().solve(K_bc, F_bc)

        # Réactions = K · u - F_ext
        from scipy.sparse import csr_matrix
        reactions = K @ u - F_ext

        # Réaction en z sur la base (nœuds 0..3, DDL z = 2, 5, 8, 11)
        dof_z_base = [2, 5, 8, 11]
        total_reaction_z = reactions[dof_z_base].sum()
        applied_force_z = 1.0
        np.testing.assert_allclose(
            total_reaction_z, -applied_force_z, atol=1e-8,
            err_msg="Équilibre global en z non vérifié"
        )

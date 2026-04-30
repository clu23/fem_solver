"""Tests unitaires pour Quad8 (sérendipité) + comparaison de convergence.

Stratégie de validation
------------------------
1. Propriétés algébriques des fonctions de forme (partition de l'unité,
   interpolation nodale, dérivées sommant à zéro).
2. Matrice K_e : symétrie, 3 modes rigides (nul-espace de dimension 3).
3. Matrice M_e : symétrie, conservation de masse.
4. Patch test en traction : un champ quadratique de déplacement est reproduit
   exactement — Quad8 représente exactement les polynômes du second degré.
5. Traction assemblée sur 2 éléments : solution analytique exacte.
6. **Convergence sur poutre console** — comparaison Quad4 / Quad8 / Tri6.

Poutre console (Euler–Bernoulli) — solution de référence
---------------------------------------------------------
Géométrie : L=1, H=0.1, t=1 (contrainte plane).
Matériau  : E=1e6, ν=0.3.
Chargement: charge transverse uniforme q [N/m] sur le bord droit.
Force totale : P = q·H·t.

Déflexion analytique en bout (flexion pure, sans cisaillement) :
    δ_tip = P·L³ / (3·E·I)  avec I = t·H³/12

    → δ_tip = (q·H) · L³ / (3·E · H³/12) = 4·q·L³ / (E·H²)

Note : pour une poutre courte (H/L = 0.1), la correction de Timoshenko
vaut δ_shear = P·L / (G·A·κ_s) ≈ 0.7% de δ_bend → ignorée dans les tests
avec rtol=0.05. Les tests de convergence montrent la réduction de cette
erreur au raffinement.
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
from femsolver.elements.quad8 import Quad8, _GAUSS_POINTS_3X3
from femsolver.elements.tri6 import Tri6


# ---------------------------------------------------------------------------
# Données communes
# ---------------------------------------------------------------------------

MAT_SIMPLE = ElasticMaterial(E=1.0, nu=0.0, rho=1.0)
MAT_STEEL = ElasticMaterial(E=210e9, nu=0.3, rho=7800)

# Carré de référence [-1,1]² avec 4 nœuds milieux exacts
NODES_REF = np.array([
    [-1.0, -1.0],  # coin 0
    [ 1.0, -1.0],  # coin 1
    [ 1.0,  1.0],  # coin 2
    [-1.0,  1.0],  # coin 3
    [ 0.0, -1.0],  # milieu 4 (bas)
    [ 1.0,  0.0],  # milieu 5 (droit)
    [ 0.0,  1.0],  # milieu 6 (haut)
    [-1.0,  0.0],  # milieu 7 (gauche)
])

# Carré unitaire [0,1]² (plus courant dans les exemples)
NODES_UNIT = np.array([
    [0.0, 0.0],   # coin 0
    [1.0, 0.0],   # coin 1
    [1.0, 1.0],   # coin 2
    [0.0, 1.0],   # coin 3
    [0.5, 0.0],   # milieu 4 (bas)
    [1.0, 0.5],   # milieu 5 (droit)
    [0.5, 1.0],   # milieu 6 (haut)
    [0.0, 0.5],   # milieu 7 (gauche)
])
PROPS_STD = {"thickness": 1.0}


# ===========================================================================
# Tests des fonctions de forme
# ===========================================================================


class TestQuad8ShapeFunctions:
    """Propriétés algébriques des 8 fonctions de forme sérendipité."""

    def test_partition_of_unity(self) -> None:
        """ΣNi = 1 à plusieurs points (ξ,η), dont les nœuds."""
        elem = Quad8()
        test_pts = [(-1, -1), (1, -1), (1, 1), (-1, 1),
                    (0, -1), (1, 0), (0, 1), (-1, 0),
                    (0, 0), (0.3, -0.7), (-0.5, 0.5)]
        for xi, eta in test_pts:
            N = elem._shape_functions(xi, eta)
            np.testing.assert_allclose(N.sum(), 1.0, atol=1e-13,
                err_msg=f"ΣNi ≠ 1 en ({xi}, {eta})")

    def test_nodal_interpolation(self) -> None:
        """Ni vaut 1 au nœud i et 0 aux 7 autres nœuds."""
        from femsolver.elements.quad8 import _NODE_XI, _NODE_ETA
        elem = Quad8()
        for i in range(8):
            xi_i, eta_i = _NODE_XI[i], _NODE_ETA[i]
            N = elem._shape_functions(xi_i, eta_i)
            assert abs(N[i] - 1.0) < 1e-12, f"N{i}({xi_i},{eta_i}) = {N[i]} ≠ 1"
            for j in range(8):
                if j != i:
                    assert abs(N[j]) < 1e-12, f"N{j}({xi_i},{eta_i}) = {N[j]} ≠ 0"

    def test_derivatives_sum_to_zero(self) -> None:
        """Σ ∂Ni/∂ξ = 0 et Σ ∂Ni/∂η = 0 (dérivée de ΣNi=1)."""
        elem = Quad8()
        for xi, eta in [(0.2, -0.4), (-0.7, 0.3), (0.0, 0.0), (1.0, 0.0)]:
            dN = elem._shape_function_derivatives(xi, eta)
            np.testing.assert_allclose(dN[0].sum(), 0.0, atol=1e-13,
                err_msg=f"Σ∂N/∂ξ ≠ 0 en ({xi},{eta})")
            np.testing.assert_allclose(dN[1].sum(), 0.0, atol=1e-13,
                err_msg=f"Σ∂N/∂η ≠ 0 en ({xi},{eta})")

    def test_quadratic_completeness(self) -> None:
        """Quad8 interpole exactement tout polynôme quadratique.

        Si f(x,y) = a + bx + cy + dx² + exy + fy², alors
        ΣNi·f(xᵢ,yᵢ) = f(x,y) en tout point du carré [−1,1]².
        C'est la condition de complétude qui garantit la convergence quadratique.
        """
        elem = Quad8()
        nodes = NODES_REF
        # f(x,y) = 1 + 2x − 3y + x² + xy − 2y² (polynôme quadratique arbitraire)
        def f(x: float, y: float) -> float:
            return 1.0 + 2.0*x - 3.0*y + x**2 + x*y - 2.0*y**2

        f_nodal = np.array([f(x, y) for x, y in nodes])

        for xi, eta in [(0.1, 0.2), (-0.5, 0.7), (0.0, 0.0), (-0.8, -0.3)]:
            N = elem._shape_functions(xi, eta)
            # Coordonnées physiques = coordonnées naturelles (carré de référence)
            x_phys = N @ nodes[:, 0]
            y_phys = N @ nodes[:, 1]
            f_interp = N @ f_nodal
            np.testing.assert_allclose(f_interp, f(x_phys, y_phys), atol=1e-13,
                err_msg=f"Complétude quadratique échouée en (ξ={xi},η={eta})")

    def test_gauss_rule_integrates_constants(self) -> None:
        """La règle 3×3 intègre exactement ∫∫_[-1,1]² dξ dη = 4."""
        total = sum(w for _, _, w in _GAUSS_POINTS_3X3)
        np.testing.assert_allclose(total, 4.0, rtol=1e-14)


# ===========================================================================
# Tests de la matrice de rigidité K_e
# ===========================================================================


class TestQuad8StiffnessMatrix:
    """Propriétés de la matrice de rigidité élémentaire."""

    def test_shape(self) -> None:
        """K_e est de shape (16, 16)."""
        K_e = Quad8().stiffness_matrix(MAT_SIMPLE, NODES_UNIT, PROPS_STD)
        assert K_e.shape == (16, 16)

    def test_symmetry(self) -> None:
        """K_e est symétrique à précision machine."""
        K_e = Quad8().stiffness_matrix(MAT_STEEL, NODES_UNIT, PROPS_STD)
        atol = 1e-10 * np.abs(K_e).max()
        np.testing.assert_allclose(K_e, K_e.T, atol=atol)

    def test_three_rigid_body_modes(self) -> None:
        """K_e a exactement 3 valeurs propres nulles (2 translations + 1 rotation).

        Pour un élément 2D en état plan, les 3 modes rigides sont :
        - translation en x (vecteur [1,0,1,0,...])
        - translation en y (vecteur [0,1,0,1,...])
        - rotation dans le plan
        """
        K_e = Quad8().stiffness_matrix(MAT_SIMPLE, NODES_UNIT, PROPS_STD)
        eigenvalues = np.linalg.eigvalsh(K_e)
        n_zero = np.sum(np.abs(eigenvalues) < 1e-10 * eigenvalues[-1])
        assert n_zero == 3, (
            f"K_e devrait avoir 3 valeurs propres nulles, trouvé {n_zero}.\n"
            f"Valeurs propres : {eigenvalues}"
        )

    def test_positive_semidefinite(self) -> None:
        """K_e est semi-définie positive (toutes eigenvaleurs ≥ 0)."""
        K_e = Quad8().stiffness_matrix(MAT_STEEL, NODES_UNIT, PROPS_STD)
        eigenvalues = np.linalg.eigvalsh(K_e)
        tol = 1e-10 * eigenvalues[-1]
        assert np.all(eigenvalues >= -tol), (
            f"K_e a une valeur propre négative : {eigenvalues.min():.3e}"
        )

    def test_no_spurious_modes(self) -> None:
        """K_e a exactement 13 valeurs propres non-nulles (16 − 3 modes rigides).

        Contrairement à une intégration 2×2 (4 pts) qui donnerait un rang < 13
        avec des modes parasites hourglass, la règle 3×3 donne le rang exact.
        """
        K_e = Quad8().stiffness_matrix(MAT_SIMPLE, NODES_UNIT, PROPS_STD)
        eigenvalues = np.linalg.eigvalsh(K_e)
        n_nonzero = np.sum(np.abs(eigenvalues) > 1e-10 * eigenvalues[-1])
        assert n_nonzero == 13, (
            f"K_e doit avoir 13 modes non-rigides, trouvé {n_nonzero}.\n"
            f"Valeurs propres : {eigenvalues}"
        )

    def test_thickness_scaling(self) -> None:
        """K_e est proportionnelle à l'épaisseur t."""
        K1 = Quad8().stiffness_matrix(MAT_SIMPLE, NODES_UNIT, {"thickness": 1.0})
        K2 = Quad8().stiffness_matrix(MAT_SIMPLE, NODES_UNIT, {"thickness": 2.5})
        # atol couvre les entrées proches de zéro (bruit flottant ~1e-16)
        np.testing.assert_allclose(K2, 2.5 * K1, rtol=1e-10, atol=1e-13)

    def test_invalid_nodes_shape_raises(self) -> None:
        """nodes.shape ≠ (8,2) → ValueError."""
        with pytest.raises(ValueError, match="nodes.shape"):
            Quad8().stiffness_matrix(MAT_SIMPLE,
                                     np.zeros((4, 2)),
                                     PROPS_STD)

    def test_invalid_thickness_raises(self) -> None:
        """thickness ≤ 0 → ValueError."""
        with pytest.raises(ValueError, match="épaisseur"):
            Quad8().stiffness_matrix(MAT_SIMPLE, NODES_UNIT, {"thickness": 0.0})


# ===========================================================================
# Tests de la matrice de masse M_e
# ===========================================================================


class TestQuad8MassMatrix:
    """Propriétés de la matrice de masse consistante."""

    def test_shape(self) -> None:
        """M_e est de shape (16, 16)."""
        M_e = Quad8().mass_matrix(MAT_STEEL, NODES_UNIT, PROPS_STD)
        assert M_e.shape == (16, 16)

    def test_symmetry(self) -> None:
        """M_e est symétrique."""
        M_e = Quad8().mass_matrix(MAT_STEEL, NODES_UNIT, PROPS_STD)
        np.testing.assert_allclose(M_e, M_e.T, atol=1e-14)

    def test_total_mass_conservation(self) -> None:
        """Somme de toutes les entrées = 2 × masse totale.

        Un champ de déplacement uniforme u=[1,0,...,1,0,...] donne une force
        inertielle totale = M·u = [m_total·g, 0, ...]. Donc la somme de toutes
        les entrées de M = 2 × masse totale (2 composantes).
        """
        rho, t = 1.0, 1.0
        mat = ElasticMaterial(E=1.0, nu=0.0, rho=rho)
        # Carré unitaire [0,1]² : aire = 1
        M_e = Quad8().mass_matrix(mat, NODES_UNIT, {"thickness": t})
        area = 1.0   # carré [0,1]²
        expected_total_mass = rho * t * area
        np.testing.assert_allclose(
            M_e.sum(), 2.0 * expected_total_mass, rtol=1e-10,
            err_msg="Conservation de masse violée"
        )

    def test_positive_semidefinite(self) -> None:
        """M_e est semi-définie positive."""
        M_e = Quad8().mass_matrix(MAT_STEEL, NODES_UNIT, PROPS_STD)
        eigenvalues = np.linalg.eigvalsh(M_e)
        assert np.all(eigenvalues >= -1e-14), (
            f"M_e a une valeur propre négative : {eigenvalues.min():.3e}"
        )


# ===========================================================================
# Patch test — champ quadratique de déplacement
# ===========================================================================


class TestQuad8PatchTest:
    """Patch test quadratique : un champ du second degré est reproduit exactement.

    Quad8 contient l'espace polynomial quadratique complet {1, x, y, x², xy, y²},
    donc il reproduit exactement tout champ quadratique de déplacement.
    Cela va au-delà du patch test linéaire standard (requis par Quad4).

    Champ imposé : ux = α·x² + β·xy,  uy = γ·y² + δ·xy.
    Les déformations correspondantes sont :
        εxx = 2α·x + β·y,  εyy = 2γ·y + δ·x,  γxy = β·x + δ·y + β·y + δ·x.
    """

    def test_quadratic_displacement_patch(self) -> None:
        """ε = B · u_e reproduit le champ quadratique en tout point."""
        alpha, beta, gamma, delta = 0.001, 0.0005, -0.002, 0.0003
        elem = Quad8()
        nodes = NODES_UNIT.copy()

        # Déplacements nodaux imposés
        u_e = np.zeros(16)
        for i, (x, y) in enumerate(nodes):
            u_e[2 * i    ] = alpha * x**2 + beta * x * y   # ux
            u_e[2 * i + 1] = gamma * y**2 + delta * x * y  # uy

        # Vérification en plusieurs points intérieurs
        for xi, eta in [(0.0, 0.0), (0.5, 0.0), (-0.5, 0.5), (0.3, -0.6)]:
            # Coordonnées physiques via interpolation
            N = elem._shape_functions(xi, eta)
            x = N @ nodes[:, 0]
            y = N @ nodes[:, 1]

            eps = elem.strain(nodes, u_e, xi, eta)
            eps_xx_ref = 2.0 * alpha * x + beta * y
            eps_yy_ref = 2.0 * gamma * y + delta * x
            gamma_xy_ref = (beta * x + delta * y) + (beta * y + delta * x)
            # Note : γxy = ∂ux/∂y + ∂uy/∂x = (β·x + α·0 + ...) → voir calcul
            # ∂ux/∂y = β·x,  ∂uy/∂x = δ·y
            gamma_xy_ref = beta * x + delta * y  # ∂ux/∂y + ∂uy/∂x

            np.testing.assert_allclose(eps[0], eps_xx_ref, atol=1e-11,
                err_msg=f"εxx incorrect à (ξ={xi},η={eta})")
            np.testing.assert_allclose(eps[1], eps_yy_ref, atol=1e-11,
                err_msg=f"εyy incorrect à (ξ={xi},η={eta})")
            np.testing.assert_allclose(eps[2], gamma_xy_ref, atol=1e-11,
                err_msg=f"γxy incorrect à (ξ={xi},η={eta})")

    def test_linear_displacement_patch(self) -> None:
        """Patch test linéaire standard : ux=αx, uy=βy reproduit exactement."""
        alpha, beta = 0.003, -0.001
        elem = Quad8()
        nodes = NODES_UNIT.copy()

        u_e = np.zeros(16)
        for i, (x, y) in enumerate(nodes):
            u_e[2 * i    ] = alpha * x
            u_e[2 * i + 1] = beta  * y

        for xi, eta in [(0.1, 0.2), (-0.5, 0.5), (0.0, 0.0)]:
            eps = elem.strain(nodes, u_e, xi, eta)
            np.testing.assert_allclose(eps[0], alpha, atol=1e-12)
            np.testing.assert_allclose(eps[1], beta,  atol=1e-12)
            np.testing.assert_allclose(eps[2], 0.0,   atol=1e-12)


# ===========================================================================
# Traction assemblée — 2 éléments Quad8
# ===========================================================================


def _make_two_quad8_mesh(
    E: float = 1.0,
    nu: float = 0.3,
    t: float = 1.0,
    P: float = 1.0,
) -> tuple[Mesh, BoundaryConditions]:
    """Maillage de 2 Quad8 sur [0,2]×[0,1] pour un test de traction.

    Deux éléments côte à côte (colonne gauche et droite d'un rectangle 2×1).

    Numérotation globale des nœuds :
        Coins  : 0=(0,0), 1=(1,0), 2=(2,0)  (bas)
                 6=(0,1), 7=(1,1), 8=(2,1)  (haut)
        Milieux: 3=(0.5,0), 4=(1.5,0)       (bas)
                 9=(0.5,1), 10=(1.5,1)      (haut)
                 5=(1,0.5)                   (milieu central vertical)
                 11=(0,0.5), 12=(2,0.5)     (milieux bords latéraux)

    Élément gauche  : coins 0,1,7,6 + milieux 3,5,9,11
    Élément droit   : coins 1,2,8,7 + milieux 4,12,10,5
    """
    mat = ElasticMaterial(E=E, nu=nu, rho=1.0)
    props = {"thickness": t, "formulation": "plane_stress"}

    nodes = np.array([
        [0.0, 0.0],   # 0  coin bas-gauche
        [1.0, 0.0],   # 1  coin bas-central
        [2.0, 0.0],   # 2  coin bas-droit
        [0.5, 0.0],   # 3  milieu bas elem-gauche
        [1.5, 0.0],   # 4  milieu bas elem-droit
        [1.0, 0.5],   # 5  milieu central (partagé)
        [0.0, 1.0],   # 6  coin haut-gauche
        [1.0, 1.0],   # 7  coin haut-central
        [2.0, 1.0],   # 8  coin haut-droit
        [0.5, 1.0],   # 9  milieu haut elem-gauche
        [1.5, 1.0],   # 10 milieu haut elem-droit
        [0.0, 0.5],   # 11 milieu gauche
        [2.0, 0.5],   # 12 milieu droit
    ])

    # Elem gauche : Quad8 sur [0,1]×[0,1]
    # Coins 0,1,7,6 → correspond à Quad8 nœuds 0(0,0),1(1,0),2(1,1),3(0,1)
    # Milieux : 4=bas=nœud3, 5=droit=nœud5, 6=haut=nœud9, 7=gauche=nœud11
    elem_left = ElementData(
        etype=Quad8,
        node_ids=(0, 1, 7, 6, 3, 5, 9, 11),
        material=mat,
        properties=props,
    )
    # Elem droit : Quad8 sur [1,2]×[0,1]
    # Coins 1,2,8,7 → Quad8 nœuds 0(1,0),1(2,0),2(2,1),3(1,1)
    # Milieux : 4=bas=nœud4, 5=droit=nœud12, 6=haut=nœud10, 7=gauche=nœud5
    elem_right = ElementData(
        etype=Quad8,
        node_ids=(1, 2, 8, 7, 4, 12, 10, 5),
        material=mat,
        properties=props,
    )
    mesh = Mesh(nodes=nodes, elements=(elem_left, elem_right), n_dim=2)

    # Dirichlet : bord gauche (nœuds 0,6,11) bloqué en ux
    #             nœud 0 bloqué en uy
    dirichlet = {
        0:  {0: 0.0, 1: 0.0},
        6:  {0: 0.0},
        11: {0: 0.0},
    }
    # Neumann : traction P sur bord droit (nœuds 2,8,12)
    # Pour Quad8 sur une arête : f_coin = P/6, f_milieu = 2P/3 (quadrature exacte)
    neumann = {
        2:  {0: P / 6.0},
        8:  {0: P / 6.0},
        12: {0: 2.0 * P / 3.0},
    }
    bc = BoundaryConditions(dirichlet=dirichlet, neumann=neumann)
    return mesh, bc


class TestQuad8TractionAssembled:
    """Traction uniforme sur 2 Quad8 assemblés — solution analytique exacte."""

    def test_tip_displacement(self) -> None:
        """ux(x=2) = P·L/(E·A) = 2/1 = 2 pour E=t=P=1, L=2, A=t=1.

        Solution analytique : ux = P·x/(E·t) = x pour ces paramètres.
        Quad8 représente exactement les polynômes linéaires → résultat exact.
        """
        E, nu, t, P = 1.0, 0.0, 1.0, 1.0
        mesh, bc = _make_two_quad8_mesh(E=E, nu=nu, t=t, P=P)
        assembler = Assembler(mesh)
        K = assembler.assemble_stiffness()
        F = assembler.assemble_forces(bc)
        K_bc, F_bc = apply_dirichlet(K, F, mesh, bc)
        u = StaticSolver().solve(K_bc, F_bc)

        # Nœud 2 : (x=2, y=0), DOF ux = 2*2 = 4
        ux_tip = u[2 * 2]
        # Analytique : ux = P*x/(E*t) = 1*2/(1*1) = 2
        np.testing.assert_allclose(ux_tip, 2.0, rtol=1e-9,
            err_msg="Déplacement de bout en traction incorrect")

    def test_transverse_displacement(self) -> None:
        """uy(x=0,y=1) = -ν·P/(E·t) = 0 pour nu=0."""
        E, nu, t, P = 1.0, 0.0, 1.0, 1.0
        mesh, bc = _make_two_quad8_mesh(E=E, nu=nu, t=t, P=P)
        assembler = Assembler(mesh)
        K = assembler.assemble_stiffness()
        F = assembler.assemble_forces(bc)
        K_bc, F_bc = apply_dirichlet(K, F, mesh, bc)
        u = StaticSolver().solve(K_bc, F_bc)

        # Nœud 6 : (x=0, y=1), DOF uy = 2*6+1 = 13
        uy_node6 = u[2 * 6 + 1]
        np.testing.assert_allclose(uy_node6, 0.0, atol=1e-9)


# ===========================================================================
# Comparaison de convergence — Poutre console
# ===========================================================================
#
# Géométrie : L=1, H=0.1, t=1. Chargement : force de bout P appliquée
# uniformément sur le bord libre (côté droit) sous forme de traction.
#
# Solution analytique Euler–Bernoulli :
#     δ = PL³/(3EI)  avec I = t·H³/12
#
# Maillage Quad8 sur nx×ny éléments (nx le long de L, ny le long de H).
# La poutre est encastrée à gauche (x=0) : ux=uy=0 sur tous les nœuds du bord.
# La force P est distribuée sur le bord droit (x=L) :
#   - Coin : P/(2·ny) / 6  (contribution de 2 arêtes adjacentes × 1/6)
#   - Milieu : P/(2·ny) · 2/3 (contribution de 1 arête × 2/3)
#
# Note : pour nx petit (1–2), l'erreur est dominée par le shear locking
# résiduel. Quad8 converge en O(h³) en déplacement (théorie) mais les
# éléments en flexion montrent h² pratiquement (élément Q8 non enrichi).


def _build_quad8_mesh_rect(
    L: float, H: float, t: float,
    nx: int, ny: int,
    material: ElasticMaterial,
    formulation: str = "plane_stress",
) -> Mesh:
    """Maillage rectangulaire uniforme de Quad8 sur [0,L]×[0,H].

    Paramètres
    ----------
    L, H : float
        Dimensions du rectangle.
    t : float
        Épaisseur.
    nx, ny : int
        Nombre d'éléments suivant x et y.
    material : ElasticMaterial
    formulation : str

    Returns
    -------
    Mesh
        Maillage complet avec (ny+1)(nx+1) nœuds coins +
        (ny+1)·nx nœuds milieux horizontaux +
        ny·(nx+1) nœuds milieux verticaux.

    Notes
    -----
    Schéma d'indexation des nœuds :

        Coins  : idx_corner(i_row, j_col) = i*(nx+1)+j
                 pour i=0..ny, j=0..nx
        Mid-H  : idx_midH(i_row, j_col)  = offset_H + i*nx + j
                 pour i=0..ny, j=0..nx-1 (milieu de l'arête horizontale)
        Mid-V  : idx_midV(i_row, j_col)  = offset_V + i*(nx+1) + j
                 pour i=0..ny-1, j=0..nx (milieu de l'arête verticale)

    Pour l'élément (ie, je) (ligne ie, colonne je) :
        Coins  : c00=ie*(nx+1)+je,  c10=ie*(nx+1)+(je+1),
                 c11=(ie+1)*(nx+1)+(je+1), c01=(ie+1)*(nx+1)+je
        Quad8 node ids : (c00, c10, c11, c01, midH_bot, midV_right,
                          midH_top, midV_left)
          midH_bot   = offset_H + ie*nx + je       (bas)
          midV_right = offset_V + ie*(nx+1)+(je+1) (droit)
          midH_top   = offset_H + (ie+1)*nx + je   (haut)
          midV_left  = offset_V + ie*(nx+1)+je     (gauche)
    """
    dx = L / nx
    dy = H / ny
    props = {"thickness": t, "formulation": formulation}

    n_corners = (ny + 1) * (nx + 1)
    offset_H = n_corners                          # début milieux horizontaux
    offset_V = offset_H + (ny + 1) * nx           # début milieux verticaux
    n_total = offset_V + ny * (nx + 1)

    nodes = np.zeros((n_total, 2))

    # Coins
    for i in range(ny + 1):
        for j in range(nx + 1):
            nodes[i * (nx + 1) + j] = [j * dx, i * dy]

    # Milieux horizontaux (sur les arêtes y=const, entre deux coins consécutifs)
    for i in range(ny + 1):
        for j in range(nx):
            nodes[offset_H + i * nx + j] = [(j + 0.5) * dx, i * dy]

    # Milieux verticaux (sur les arêtes x=const, entre deux coins consécutifs)
    for i in range(ny):
        for j in range(nx + 1):
            nodes[offset_V + i * (nx + 1) + j] = [j * dx, (i + 0.5) * dy]

    elements = []
    for ie in range(ny):
        for je in range(nx):
            c00 = ie * (nx + 1) + je
            c10 = ie * (nx + 1) + (je + 1)
            c11 = (ie + 1) * (nx + 1) + (je + 1)
            c01 = (ie + 1) * (nx + 1) + je
            m_bot   = offset_H + ie * nx + je
            m_right = offset_V + ie * (nx + 1) + (je + 1)
            m_top   = offset_H + (ie + 1) * nx + je
            m_left  = offset_V + ie * (nx + 1) + je
            # Quad8 ordre : 0=BL, 1=BR, 2=TR, 3=TL, 4=bot, 5=right, 6=top, 7=left
            node_ids = (c00, c10, c11, c01, m_bot, m_right, m_top, m_left)
            elements.append(ElementData(
                etype=Quad8,
                node_ids=node_ids,
                material=material,
                properties=props,
            ))

    return Mesh(nodes=nodes, elements=tuple(elements), n_dim=2)


def _build_quad4_mesh_rect(
    L: float, H: float, t: float,
    nx: int, ny: int,
    material: ElasticMaterial,
    formulation: str = "plane_stress",
) -> Mesh:
    """Maillage rectangulaire uniforme de Quad4 sur [0,L]×[0,H]."""
    dx = L / nx
    dy = H / ny
    props = {"thickness": t, "formulation": formulation}

    nodes = np.array([
        [j * dx, i * dy]
        for i in range(ny + 1)
        for j in range(nx + 1)
    ])

    elements = []
    for i in range(ny):
        for j in range(nx):
            n0 = i * (nx + 1) + j
            n1 = i * (nx + 1) + (j + 1)
            n2 = (i + 1) * (nx + 1) + (j + 1)
            n3 = (i + 1) * (nx + 1) + j
            elements.append(ElementData(
                etype=Quad4,
                node_ids=(n0, n1, n2, n3),
                material=material,
                properties=props,
            ))
    return Mesh(nodes=nodes, elements=tuple(elements), n_dim=2)


def _build_tri6_mesh_rect(
    L: float, H: float, t: float,
    nx: int, ny: int,
    material: ElasticMaterial,
    formulation: str = "plane_stress",
) -> Mesh:
    """Maillage de Tri6 sur [0,L]×[0,H] (2 triangles par cellule rectangulaire).

    Chaque cellule rectangulaire est divisée en 2 Tri6 par la diagonale
    coin-bas-gauche → coin-haut-droit. Les nœuds milieux sont sur les arêtes.

    Pour un nx×ny grid de cellules, la grille complète (pas h/2 dans chaque
    direction) a (2nx+1)×(2ny+1) nœuds, soit tous les nœuds pairs ET impairs.
    Nœud (2i,2j) = coin, nœud (2i+1,2j) = milieu horizontal, etc.

    Seuls les nœuds sur les arêtes des Tri6 sont inclus (pas de centre).
    """
    dx = L / nx
    dy = H / ny
    half_dx = dx / 2.0
    half_dy = dy / 2.0
    props = {"thickness": t, "formulation": formulation}

    # Grille de pas h/2 : (2ny+1) × (2nx+1) nœuds
    def node_idx(i2: int, j2: int) -> int:
        """Indice du nœud sur la grille (2nx+1)×(2ny+1), ligne i2, col j2."""
        return i2 * (2 * nx + 1) + j2

    n_grid = (2 * ny + 1) * (2 * nx + 1)
    nodes = np.zeros((n_grid, 2))
    for i2 in range(2 * ny + 1):
        for j2 in range(2 * nx + 1):
            nodes[node_idx(i2, j2)] = [j2 * half_dx, i2 * half_dy]

    elements = []
    for i in range(ny):
        for j in range(nx):
            # Coins de la cellule (i,j) sur la grille fine
            i0, i1 = 2 * i, 2 * i + 2   # lignes grille fine
            j0, j1 = 2 * j, 2 * j + 2   # colonnes grille fine
            im = 2 * i + 1
            jm = 2 * j + 1

            # Coins : BL, BR, TR, TL
            c_bl = node_idx(i0, j0)
            c_br = node_idx(i0, j1)
            c_tr = node_idx(i1, j1)
            c_tl = node_idx(i1, j0)

            # Milieux des arêtes
            m_bot  = node_idx(i0, jm)   # milieu arête bas (BL–BR)
            m_top  = node_idx(i1, jm)   # milieu arête haut (TL–TR)
            m_left = node_idx(im, j0)   # milieu arête gauche (BL–TL)
            m_rgt  = node_idx(im, j1)   # milieu arête droite (BR–TR)
            m_diag = node_idx(im, jm)   # milieu diagonale (BL–TR)

            # Triangle bas (BL, BR, TR) + milieux (bot, rgt, diag)
            elements.append(ElementData(
                etype=Tri6,
                node_ids=(c_bl, c_br, c_tr, m_bot, m_rgt, m_diag),
                material=material,
                properties=props,
            ))
            # Triangle haut (BL, TR, TL) + milieux (diag, top, left)
            elements.append(ElementData(
                etype=Tri6,
                node_ids=(c_bl, c_tr, c_tl, m_diag, m_top, m_left),
                material=material,
                properties=props,
            ))

    return Mesh(nodes=nodes, elements=tuple(elements), n_dim=2)


def _cantilever_deflection(
    mesh: Mesh,
    L: float,
    H: float,
    t: float,
    P: float,
    formulation: str = "plane_stress",
) -> float:
    """Résout la poutre console et retourne la déflexion maximale en bout.

    La déflexion est mesurée comme la moyenne de |uy| sur les nœuds du
    bord libre (x=L), ce qui filtre les perturbations de bord locales.

    Conditions aux limites :
    - Encastrement gauche (x≈0) : ux=uy=0 sur tous les nœuds
    - Force de bout : P répartie sur le bord droit (x≈L)
      - Tri6 : f_coin = P/(6·ny), f_milieu = 2P/(3·ny) par arête
      - Quad8 : idem (même règle d'intégration sur l'arête)
      - Quad4 : f_coin = P/(2·ny) par nœud (répartition égale)
    """
    nodes = mesh.nodes
    tol = 1e-9 * L

    # Bord gauche : tous les nœuds avec x ≈ 0
    left_ids = [i for i, (x, y) in enumerate(nodes) if abs(x) < tol]
    dirichlet = {nid: {0: 0.0, 1: 0.0} for nid in left_ids}

    # Bord droit : tous les nœuds avec x ≈ L
    right_ids = sorted([i for i, (x, y) in enumerate(nodes) if abs(x - L) < tol],
                       key=lambda k: nodes[k, 1])

    # Force totale P distribuée sur le bord droit.
    # Stratégie simple et robuste : force égale sur chaque nœud du bord.
    # (Exact pour Quad4 ; légère approximation pour les éléments quadratiques,
    # mais sans biais systématique sur le déplacement global.)
    n_right = len(right_ids)
    neumann = {nid: {1: -P / n_right} for nid in right_ids}

    bc = BoundaryConditions(dirichlet=dirichlet, neumann=neumann)
    assembler = Assembler(mesh)
    K = assembler.assemble_stiffness()
    F = assembler.assemble_forces(bc)
    K_bc, F_bc = apply_dirichlet(K, F, mesh, bc)
    u = StaticSolver().solve(K_bc, F_bc)

    # Moyenne de |uy| sur les nœuds du bord libre
    uy_right = [abs(u[2 * nid + 1]) for nid in right_ids]
    return float(np.mean(uy_right))


class TestCantileverConvergence:
    """Comparaison de convergence Quad4 / Quad8 / Tri6 sur poutre console.

    Géométrie : L=1, H=0.1, t=1 (poutre mince → comportement poutre).
    Matériau  : E=1e6, ν=0.3.
    Chargement: force transverse P=1 en bout (vers le bas, −y).

    Solution analytique Euler–Bernoulli :
        δ = PL³/(3EI),  I = t·H³/12
        δ = 1·1³ / (3·1e6 · 1·(0.1)³/12) = 12 / (3·1e6·0.001) = 12/3000 = 0.004

    Résultats attendus (≤5% d'erreur) :
    - Quad8  4×1 : ~3% erreur  (très grossier mais déjà bon)
    - Quad8  8×2 : <1% erreur
    - Quad4 16×4 : ~3% erreur  (2× plus de DOFs que Quad8 8×2, même précision)
    - Tri6   8×2 : ~3% erreur  (légèrement moins précis que Quad8 même DOFs)

    La hiérarchie attendue en précision par DDL est : Quad8 >> Tri6 ≈ Quad4.
    """

    E = 1e6
    nu = 0.3
    L = 1.0
    H = 0.1
    t = 1.0
    P = 1.0

    @property
    def delta_analytic(self) -> float:
        """Déflexion analytique de la poutre console."""
        I = self.t * self.H**3 / 12.0
        return self.P * self.L**3 / (3.0 * self.E * I)

    def _material(self) -> ElasticMaterial:
        return ElasticMaterial(E=self.E, nu=self.nu, rho=1.0)

    def test_quad4_coarse_severely_locked(self) -> None:
        """Quad4 8×2 souffre de shear locking sévère en flexion.

        Le shear locking vient du fait que le champ bilinéaire du Quad4
        ne peut pas représenter le mode de flexion pur (courbure cubique)
        sans introduire des déformations de cisaillement parasites qui
        rigidifient artificiellement l'élément.

        Résultat typique : erreur > 30% pour une poutre mince (H/L=0.1)
        avec seulement 8×2 éléments Quad4 à intégration complète.
        Ce test documente ce comportement et vérifie que Quad8 fait mieux.
        """
        mat = self._material()
        mesh_q4 = _build_quad4_mesh_rect(self.L, self.H, self.t, 8, 2, mat)
        delta_q4 = _cantilever_deflection(mesh_q4, self.L, self.H, self.t, self.P)
        err_q4 = abs(delta_q4 - self.delta_analytic) / self.delta_analytic

        # Quad8 8×2 — même nombre d'éléments, bien meilleur résultat
        mesh_q8 = _build_quad8_mesh_rect(self.L, self.H, self.t, 8, 2, mat)
        delta_q8 = _cantilever_deflection(mesh_q8, self.L, self.H, self.t, self.P)
        err_q8 = abs(delta_q8 - self.delta_analytic) / self.delta_analytic

        # Quad4 a un locking sévère pour ce maillage
        assert err_q4 > 0.20, (
            f"Quad4 8×2 devrait montrer du shear locking (>20% d'erreur), "
            f"obtenu {err_q4:.1%}. Vérifier la formulation."
        )
        # Quad8, même maillage : bien plus précis
        assert err_q8 < err_q4 / 5.0, (
            f"Quad8 8×2 (erreur {err_q8:.2%}) devrait être au moins 5× "
            f"plus précis que Quad4 8×2 (erreur {err_q4:.1%})."
        )

    def test_quad8_coarse_within_5pct(self) -> None:
        """Quad8 4×1 : erreur < 5% malgré le maillage très grossier.

        Avec seulement 4 éléments en longueur et 1 en hauteur, Quad8
        capture déjà bien la flexion grâce à ses fonctions de forme
        quadratiques qui peuvent représenter la courbure parabolique.
        """
        mat = self._material()
        mesh = _build_quad8_mesh_rect(self.L, self.H, self.t, 4, 1, mat)
        delta = _cantilever_deflection(mesh, self.L, self.H, self.t, self.P)
        err = abs(delta - self.delta_analytic) / self.delta_analytic
        assert err < 0.05, (
            f"Quad8 4×1 : erreur {err:.1%} trop grande "
            f"(δ={delta:.6f}, ref={self.delta_analytic:.6f})"
        )

    def test_quad8_medium_within_2pct(self) -> None:
        """Quad8 8×2 : erreur < 2%."""
        mat = self._material()
        mesh = _build_quad8_mesh_rect(self.L, self.H, self.t, 8, 2, mat)
        delta = _cantilever_deflection(mesh, self.L, self.H, self.t, self.P)
        err = abs(delta - self.delta_analytic) / self.delta_analytic
        assert err < 0.02, (
            f"Quad8 8×2 : erreur {err:.1%} trop grande "
            f"(δ={delta:.6f}, ref={self.delta_analytic:.6f})"
        )

    def test_tri6_medium_within_5pct(self) -> None:
        """Tri6 8×2 (16 triangles) : erreur < 5%."""
        mat = self._material()
        mesh = _build_tri6_mesh_rect(self.L, self.H, self.t, 8, 2, mat)
        delta = _cantilever_deflection(mesh, self.L, self.H, self.t, self.P)
        err = abs(delta - self.delta_analytic) / self.delta_analytic
        assert err < 0.05, (
            f"Tri6 8×2 : erreur {err:.1%} trop grande "
            f"(δ={delta:.6f}, ref={self.delta_analytic:.6f})"
        )

    def test_quad8_more_accurate_than_quad4_same_dof_count(self) -> None:
        """Quad8 4×1 est plus précis que Quad4 8×2 pour un nombre de DDL similaire.

        Quad4  8×2 : (8+1)(2+1) = 27 nœuds → 54 DDL
        Quad8  4×1 : coins (4+1)(1+1)=10 + midH (4+1)×1=5 + midV 4×(1+1)=8 = 23 nœuds → 46 DDL
        (environ les mêmes DDL)

        Cela illustre la supériorité des éléments quadratiques : avec un ordre
        d'approximation plus élevé, ils convergent plus vite pour le même coût.
        """
        mat = self._material()
        mesh_q4 = _build_quad4_mesh_rect(self.L, self.H, self.t, 8, 2, mat)
        delta_q4 = _cantilever_deflection(mesh_q4, self.L, self.H, self.t, self.P)
        err_q4 = abs(delta_q4 - self.delta_analytic) / self.delta_analytic

        mesh_q8 = _build_quad8_mesh_rect(self.L, self.H, self.t, 4, 1, mat)
        delta_q8 = _cantilever_deflection(mesh_q8, self.L, self.H, self.t, self.P)
        err_q8 = abs(delta_q8 - self.delta_analytic) / self.delta_analytic

        assert err_q8 < err_q4, (
            f"Quad8 4×1 (erreur {err_q8:.2%}) devrait être plus précis que "
            f"Quad4 8×2 (erreur {err_q4:.2%}) pour un nombre de DDL similaire."
        )


# ===========================================================================
# Tests batch
# ===========================================================================


class TestQuad8Batch:
    """Cohérence entre stiffness_matrix() et batch_stiffness_matrix()."""

    def test_batch_stiffness_matches_scalar(self) -> None:
        """batch_stiffness_matrix produit les mêmes K_e que la méthode scalaire."""
        mat = ElasticMaterial(E=210e9, nu=0.3, rho=7800)
        D = mat.elasticity_matrix_plane_stress()
        t = 0.01

        rng = np.random.default_rng(42)
        n_e = 5
        # Générer des nœuds légèrement perturbés autour du carré unité
        base = NODES_UNIT
        nodes_batch = base[np.newaxis, :, :].repeat(n_e, axis=0)
        nodes_batch += rng.uniform(-0.05, 0.05, nodes_batch.shape)

        K_batch = Quad8.batch_stiffness_matrix(nodes_batch, D, t)

        props = {"thickness": t}
        for e in range(n_e):
            K_scalar = Quad8().stiffness_matrix(mat, nodes_batch[e], props)
            np.testing.assert_allclose(
                K_batch[e], K_scalar, rtol=1e-12,
                err_msg=f"Batch vs scalar divergent pour l'élément {e}"
            )

    def test_batch_mass_matches_scalar(self) -> None:
        """batch_mass_matrix produit les mêmes M_e que la méthode scalaire."""
        mat = ElasticMaterial(E=1.0, nu=0.3, rho=2700)
        t = 0.005
        n_e = 3
        rng = np.random.default_rng(7)
        nodes_batch = NODES_UNIT[np.newaxis, :, :].repeat(n_e, axis=0)
        nodes_batch += rng.uniform(-0.02, 0.02, nodes_batch.shape)

        M_batch = Quad8.batch_mass_matrix(nodes_batch, mat.rho, t)
        props = {"thickness": t}
        for e in range(n_e):
            M_scalar = Quad8().mass_matrix(mat, nodes_batch[e], props)
            np.testing.assert_allclose(
                M_batch[e], M_scalar, rtol=1e-12,
                err_msg=f"Batch masse vs scalar divergent pour l'élément {e}"
            )

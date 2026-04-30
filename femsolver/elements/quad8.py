"""Élément quadrilatère sérendipité Q8 isoparamétrique à 8 nœuds.

Sérendipité vs Lagrange — pourquoi deux familles d'éléments quadratiques ?
===========================================================================

**Famille de Lagrange**

La construction naturelle d'un élément 2D consiste à prendre le produit
tensoriel des fonctions de Lagrange 1D :

    Espace 1D linéaire   : {1, ξ}         → 2 nœuds par arête
    Espace 1D quadratique: {1, ξ, ξ²}     → 3 nœuds par arête

Produit tensoriel 2D quadratique :
    {1, ξ, ξ², η, ξη, ξ²η, η², ξη², ξ²η²}  → 9 termes → 9 nœuds (Q9)

Le Q9 a donc un **nœud central** (ξ=0, η=0) dont la fonction de forme
associée est ξ²η² (terme "bulle"). Ce terme améliore légèrement la capture
des modes en flexion antisymétrique, mais :
- augmente le nombre de DDL (+12 par maille de Q9 vs Q8),
- ce terme n'apporte que peu en pratique sur des maillages 2D typiques.

**Famille sérendipité**

La famille *sérendipité* supprime les nœuds intérieurs tout en conservant
les nœuds sur les arêtes. Pour Q8 (2D quadratique) :
- 4 nœuds aux coins
- 4 nœuds aux milieux des arêtes
- **pas de nœud central**

L'espace polynomial sérendipité Q8 est :
    {1, ξ, η, ξ², ξη, η², ξ²η, ξη²}   — 8 termes

C'est un sous-ensemble *incomplet* du degré 3, mais il contient tout le
degré 2 (comme Tri6). Les fonctions de forme sont construites par correction :

    Coin i (ξᵢ, ηᵢ) = (±1, ±1) :
        Nᵢ = ¼(1+ξᵢξ)(1+ηᵢη)(ξᵢξ + ηᵢη − 1)

    Milieu d'une arête η=const (ξᵢ=0, ηᵢ=±1) :
        Nᵢ = ½(1−ξ²)(1+ηᵢη)

    Milieu d'une arête ξ=const (ξᵢ=±1, ηᵢ=0) :
        Nᵢ = ½(1+ξᵢξ)(1−η²)

Ces formules se déduisent du Q9 en soustrayant la contribution du nœud
central : elles vérifient la partition de l'unité ΣNi=1 sans le nœud bulle.

**Comparaison Q4 / Q8(sérendipité) / Q9(Lagrange)**

+----------------+---------+---------+---------+
| Propriété      |  Q4     |  Q8     |  Q9     |
+----------------+---------+---------+---------+
| Nœuds          |  4      |  8      |  9      |
| DDL (2D)       |  8      |  16     |  18     |
| Espace poly.   | bilin.  | sérénd. | Lagrange|
| Terme max      | ξη      | ξ²η,ξη² | ξ²η²   |
| Gauss optimal  | 2×2     | 3×3     | 3×3     |
| Convergence    | O(h²)u  | O(h³)u  | O(h³)u  |
| Shear locking  | fort    | faible  | faible  |
+----------------+---------+---------+---------+

**Pourquoi 3×3 Gauss et non 2×2 ?**

Pour Q4, B(ξ,η) est bilinéaire (degré 1 en ξ, degré 1 en η), donc
B^T D B est de degré 2 en chaque variable. La règle 2×2 intègre exactement
les polynômes de degré ≤ 3 en chaque direction → Q4 est intégré exactement.

Pour Q8, les fonctions de forme sont *quadratiques* (degré 2 en (ξ,η)).
Les dérivées ∂N/∂ξ contiennent des termes jusqu'à ξη² → degré 2 en η pour
les nœuds coins. Sur un élément général (non rectangulaire), J⁻¹ introduit
des termes rationnels supplémentaires.

Pour un rectangle : B a des termes de degré 2 en chaque variable ;
B^T D B a des termes de degré 4. La règle 2×2 (exacte jusqu'au degré 3)
n'est plus suffisante → erreur d'intégration.

La règle **3×3** (exacte jusqu'au degré 5 en chaque variable) intègre
exactement le Q8 rectangulaire.

De plus, argument de rang : avec seulement 4 points de Gauss (2×2), la
matrice K_e de taille 16×16 a au plus rang 4×3=12 (contraintes de déform.),
alors qu'il en faut 16−3=13 (en retirant 3 modes rigides). Cela induit des
**modes zéro-énergie parasites** (hourglass). Avec 9 points (3×3), le rang
est bien 13, sans mode parasite.

References
----------
Cook et al., « Concepts and Applications of FEA », 4th ed., chap. 6–8.
Zienkiewicz & Taylor, « The FEM for Solid Mechanics », vol. 1, §9.3–9.4.
Hughes, « The Finite Element Method », §3.7 (serendipity vs. Lagrange).
"""

from __future__ import annotations

import numpy as np

from femsolver.core.element import Element
from femsolver.core.material import ElasticMaterial


# ---------------------------------------------------------------------------
# Règle de Gauss-Legendre 3×3 (intégration exacte jusqu'au degré 5)
# ---------------------------------------------------------------------------
#
# Règle 1D à 3 points :
#   ξ₁ = −√(3/5), w₁ = 5/9
#   ξ₂ = 0,        w₂ = 8/9
#   ξ₃ = +√(3/5), w₃ = 5/9
#
# La règle 2D est le produit tensoriel de deux règles 1D.
# Chaque point de Gauss 2D : (ξᵢ, ηⱼ, wᵢ·wⱼ).

_G3 = np.sqrt(3.0 / 5.0)          # ≈ 0.7745966692414834
_W3 = (5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0)
_X3 = (-_G3, 0.0, _G3)

_GAUSS_POINTS_3X3: list[tuple[float, float, float]] = [
    (xi, eta, wx * wy)
    for xi, wx in zip(_X3, _W3)
    for eta, wy in zip(_X3, _W3)
]   # 9 points, poids somme = 4.0 = aire du carré de référence [-1,1]²


# ---------------------------------------------------------------------------
# Coordonnées naturelles des 8 nœuds du Quad8
# ---------------------------------------------------------------------------
#
# Numérotation (sens trigonométrique, 0-indexé) :
#
#   3 ---6--- 2
#   |         |
#   7         5
#   |         |
#   0 ---4--- 1
#
# Coins (0–3) : (ξᵢ, ηᵢ) = (±1, ±1)
# Milieux (4–7) :
#   4 : (0, −1)  milieu de l'arête 0–1 (bas)
#   5 : (+1, 0)  milieu de l'arête 1–2 (droite)
#   6 : (0, +1)  milieu de l'arête 2–3 (haut)
#   7 : (−1, 0)  milieu de l'arête 3–0 (gauche)

_NODE_XI = np.array([-1.0,  1.0,  1.0, -1.0,  0.0, 1.0,  0.0, -1.0])
_NODE_ETA = np.array([-1.0, -1.0,  1.0,  1.0, -1.0, 0.0,  1.0,  0.0])
# Type de nœud : "corner" = 0, "mid_eta" (ηᵢ=±1, ξᵢ=0) = 1, "mid_xi" (ξᵢ=±1, ηᵢ=0) = 2
_NODE_TYPE = np.array([0, 0, 0, 0, 1, 2, 1, 2])   # nœuds 4,6 : mid_eta ; 5,7 : mid_xi


class Quad8(Element):
    """Quadrilatère sérendipité Q8 isoparamétrique — état plan.

    8 nœuds, 2 DDL par nœud (ux, uy) → matrices élémentaires 16×16.
    Intégration numérique 3×3 points de Gauss (9 points).

    Numérotation des nœuds (sens trigonométrique, 0-indexé) :

        3 ---6--- 2
        |         |
        7         5
        |         |
        0 ---4--- 1

    Coordonnées naturelles (ξ,η) ∈ [-1,1]² :
        0: (−1,−1)  1: (+1,−1)  2: (+1,+1)  3: (−1,+1)
        4: (0,−1)   5: (+1,0)   6: (0,+1)   7: (−1,0)

    Ordre des DDL : [u0,v0, u1,v1, …, u7,v7].

    Fonctions de forme (sérendipité) :
        Coins i=0..3 :   Nᵢ = ¼(1+ξᵢξ)(1+ηᵢη)(ξᵢξ + ηᵢη − 1)
        Milieu η=±1 (nœuds 4,6) : Nᵢ = ½(1−ξ²)(1+ηᵢη)
        Milieu ξ=±1 (nœuds 5,7) : Nᵢ = ½(1+ξᵢξ)(1−η²)

    Parameters (via ``properties``)
    --------------------------------
    thickness : float
        Épaisseur de la plaque [m].
    formulation : str
        ``"plane_stress"`` (défaut) ou ``"plane_strain"``.

    Examples
    --------
    >>> import numpy as np
    >>> from femsolver.core.material import ElasticMaterial
    >>> mat = ElasticMaterial(E=1.0, nu=0.0, rho=1.0)
    >>> nodes = np.array([
    ...     [-1.,-1.],[1.,-1.],[1.,1.],[-1.,1.],
    ...     [0.,-1.],[1.,0.],[0.,1.],[-1.,0.]])
    >>> K_e = Quad8().stiffness_matrix(mat, nodes, {"thickness": 1.0})
    >>> K_e.shape
    (16, 16)
    """

    def dof_per_node(self) -> int:
        """2 DDL par nœud : ux et uy."""
        return 2

    def n_nodes(self) -> int:
        """8 nœuds."""
        return 8

    # ------------------------------------------------------------------
    # Fonctions de forme et leurs dérivées
    # ------------------------------------------------------------------

    @staticmethod
    def _shape_functions(xi: float, eta: float) -> np.ndarray:
        """Évaluer les 8 fonctions de forme sérendipité au point (ξ, η).

        Parameters
        ----------
        xi, eta : float
            Coordonnées naturelles dans [−1, 1].

        Returns
        -------
        N : np.ndarray, shape (8,)
            [N0, N1, …, N7].

        Notes
        -----
        Partition de l'unité : ΣNi = 1.

        Formules :
            Coins (i=0..3) :  Nᵢ = ¼(1+ξᵢξ)(1+ηᵢη)(ξᵢξ+ηᵢη−1)
            Milieu η=±1 (4,6): Nᵢ = ½(1−ξ²)(1+ηᵢη)
            Milieu ξ=±1 (5,7): Nᵢ = ½(1+ξᵢξ)(1−η²)

        Vérification nœud 0 (ξ₀=−1, η₀=−1) :
            N0(−1,−1) = ¼(2)(2)(1+1−1) = 1 ✓
            N0(+1,−1) = ¼(0)(2)(…)     = 0 ✓   (nœud 1)
            N0(0,−1)  = ¼(1)(2)(0+1−1) = 0 ✓   (nœud 4)
        """
        N = np.empty(8)
        # Coins (0–3)
        for i in range(4):
            xi_i = _NODE_XI[i]
            eta_i = _NODE_ETA[i]
            N[i] = 0.25 * (1.0 + xi_i * xi) * (1.0 + eta_i * eta) * (xi_i * xi + eta_i * eta - 1.0)
        # Nœud 4 (0, −1) : milieu bas
        N[4] = 0.5 * (1.0 - xi**2) * (1.0 - eta)
        # Nœud 5 (+1, 0) : milieu droit
        N[5] = 0.5 * (1.0 + xi) * (1.0 - eta**2)
        # Nœud 6 (0, +1) : milieu haut
        N[6] = 0.5 * (1.0 - xi**2) * (1.0 + eta)
        # Nœud 7 (−1, 0) : milieu gauche
        N[7] = 0.5 * (1.0 - xi) * (1.0 - eta**2)
        return N

    @staticmethod
    def _shape_function_derivatives(xi: float, eta: float) -> np.ndarray:
        """Dérivées ∂Nᵢ/∂ξ et ∂Nᵢ/∂η au point (ξ, η).

        Parameters
        ----------
        xi, eta : float
            Coordonnées naturelles dans [−1, 1].

        Returns
        -------
        dN : np.ndarray, shape (2, 8)
            dN[0, i] = ∂Nᵢ/∂ξ,  dN[1, i] = ∂Nᵢ/∂η.

        Notes
        -----
        Coins i (ξᵢ, ηᵢ) :
            ∂Nᵢ/∂ξ = ¼ξᵢ(1+ηᵢη)(2ξᵢξ+ηᵢη)
            ∂Nᵢ/∂η = ¼ηᵢ(1+ξᵢξ)(ξᵢξ+2ηᵢη)

        Milieux ηᵢ=±1, ξᵢ=0 (nœuds 4, 6) :
            ∂Nᵢ/∂ξ = −ξ(1+ηᵢη)
            ∂Nᵢ/∂η = ½ηᵢ(1−ξ²)

        Milieux ξᵢ=±1, ηᵢ=0 (nœuds 5, 7) :
            ∂Nᵢ/∂ξ = ½ξᵢ(1−η²)
            ∂Nᵢ/∂η = −η(1+ξᵢξ)

        La somme de toutes les dérivées est nulle (ΣNi=1 → Σ∂Ni/∂ξ=0).
        """
        dN = np.empty((2, 8))
        # Coins (0–3)
        for i in range(4):
            xi_i = _NODE_XI[i]
            eta_i = _NODE_ETA[i]
            a = 1.0 + xi_i * xi
            b = 1.0 + eta_i * eta
            dN[0, i] = 0.25 * xi_i * b * (2.0 * xi_i * xi + eta_i * eta)
            dN[1, i] = 0.25 * eta_i * a * (xi_i * xi + 2.0 * eta_i * eta)
        # Nœud 4 : (0, −1)
        dN[0, 4] = -xi * (1.0 - eta)
        dN[1, 4] = -0.5 * (1.0 - xi**2)
        # Nœud 5 : (+1, 0)
        dN[0, 5] = 0.5 * (1.0 - eta**2)
        dN[1, 5] = -eta * (1.0 + xi)
        # Nœud 6 : (0, +1)
        dN[0, 6] = -xi * (1.0 + eta)
        dN[1, 6] = 0.5 * (1.0 - xi**2)
        # Nœud 7 : (−1, 0)
        dN[0, 7] = -0.5 * (1.0 - eta**2)
        dN[1, 7] = -eta * (1.0 - xi)
        return dN

    # ------------------------------------------------------------------
    # Jacobien et matrice B
    # ------------------------------------------------------------------

    def _jacobian_and_B(
        self,
        xi: float,
        eta: float,
        nodes: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """det(J) et matrice B (3×16) au point de Gauss (ξ, η).

        Parameters
        ----------
        xi, eta : float
            Coordonnées naturelles.
        nodes : np.ndarray, shape (8, 2)
            Coordonnées physiques des 8 nœuds [[x0,y0], …, [x7,y7]].

        Returns
        -------
        det_J : float
            Déterminant du Jacobien (> 0 si nœuds bien orientés).
        B : np.ndarray, shape (3, 16)
            Matrice déformation–déplacement. ε = B · u_e.

        Raises
        ------
        ValueError
            Si det(J) ≤ 0 (élément dégénéré ou nœuds dans le mauvais sens).

        Notes
        -----
        J = dN/d(ξ,η) @ nodes  (2×2).
        B[0, 2i]   = ∂Nᵢ/∂x  (εxx)
        B[1, 2i+1] = ∂Nᵢ/∂y  (εyy)
        B[2, 2i]   = ∂Nᵢ/∂y  (γxy — terme ux)
        B[2, 2i+1] = ∂Nᵢ/∂x  (γxy — terme uy)
        """
        dN = self._shape_function_derivatives(xi, eta)   # (2, 8)
        J = dN @ nodes                                    # (2, 2)
        det_J = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]

        if det_J <= 0.0:
            raise ValueError(
                f"det(J) = {det_J:.6g} ≤ 0 au point (ξ={xi:.4f}, η={eta:.4f}). "
                "Vérifier l'orientation et la position des nœuds milieux."
            )

        # Inverse explicite 2×2 (évite np.linalg.inv)
        J_inv = np.array([[J[1, 1], -J[0, 1]], [-J[1, 0], J[0, 0]]]) / det_J
        dN_phys = J_inv @ dN   # (2, 8) : dN_phys[0]=∂N/∂x, [1]=∂N/∂y

        B = np.zeros((3, 16))
        for i in range(8):
            c = 2 * i
            B[0, c    ] = dN_phys[0, i]   # εxx = ∂ux/∂x
            B[1, c + 1] = dN_phys[1, i]   # εyy = ∂uy/∂y
            B[2, c    ] = dN_phys[1, i]   # γxy (terme ux) = ∂ux/∂y
            B[2, c + 1] = dN_phys[0, i]   # γxy (terme uy) = ∂uy/∂x

        return det_J, B

    def _elasticity_matrix(
        self, material: ElasticMaterial, formulation: str
    ) -> np.ndarray:
        """Matrice D selon la formulation (plane_stress ou plane_strain)."""
        if formulation == "plane_stress":
            return material.elasticity_matrix_plane_stress()
        if formulation == "plane_strain":
            return material.elasticity_matrix_plane_strain()
        raise ValueError(
            f"formulation doit être 'plane_stress' ou 'plane_strain', "
            f"reçu '{formulation}'"
        )

    # ------------------------------------------------------------------
    # Matrices élémentaires
    # ------------------------------------------------------------------

    def stiffness_matrix(
        self,
        material: ElasticMaterial,
        nodes: np.ndarray,
        properties: dict,
    ) -> np.ndarray:
        """Matrice de rigidité élémentaire K_e (16×16).

        Parameters
        ----------
        material : ElasticMaterial
            Matériau (E, nu utilisés).
        nodes : np.ndarray, shape (8, 2)
            Coordonnées [[x0,y0], …, [x7,y7]].
            Nœuds 0–3 : coins, nœuds 4–7 : milieux des arêtes.
        properties : dict
            ``"thickness"`` : épaisseur [m].
            ``"formulation"`` : ``"plane_stress"`` (défaut) ou ``"plane_strain"``.

        Returns
        -------
        K_e : np.ndarray, shape (16, 16)
            Matrice de rigidité symétrique.

        Notes
        -----
        Intégration 3×3 Gauss (9 points) :

            K_e = t · Σₚ wₚ · Bₚᵀ · D · Bₚ · |det Jₚ|

        La règle 3×3 est exacte pour des polynômes de degré ≤ 5 en chaque
        variable, ce qui couvre exactement l'intégrande du Q8 rectangulaire
        (B^T D B de degré 4) avec une marge de sécurité pour les distorsions.
        """
        if nodes.shape != (8, 2):
            raise ValueError(f"Quad8 attend nodes.shape == (8, 2), reçu {nodes.shape}")
        t = properties["thickness"]
        if t <= 0.0:
            raise ValueError(f"L'épaisseur doit être > 0, reçu thickness={t}")
        formulation = properties.get("formulation", "plane_stress")
        D = self._elasticity_matrix(material, formulation)

        K_e = np.zeros((16, 16))
        for xi, eta, w in _GAUSS_POINTS_3X3:
            det_J, B = self._jacobian_and_B(xi, eta, nodes)
            K_e += (w * t * det_J) * (B.T @ D @ B)
        return K_e

    def mass_matrix(
        self,
        material: ElasticMaterial,
        nodes: np.ndarray,
        properties: dict,
    ) -> np.ndarray:
        """Matrice de masse consistante M_e (16×16).

        Parameters
        ----------
        material : ElasticMaterial
            Matériau (rho utilisé).
        nodes : np.ndarray, shape (8, 2)
            Coordonnées nodales.
        properties : dict
            ``"thickness"`` : épaisseur [m].

        Returns
        -------
        M_e : np.ndarray, shape (16, 16)
            Matrice de masse consistante par intégration 3×3 Gauss.

        Notes
        -----
        M_e = ρt · ∫∫ Nᵀ N |det J| dξ dη

        Matrice N (2×16) :
            N[0, 2i]   = Nᵢ   (composante ux)
            N[1, 2i+1] = Nᵢ   (composante uy)
        """
        if nodes.shape != (8, 2):
            raise ValueError(f"Quad8 attend nodes.shape == (8, 2), reçu {nodes.shape}")
        t = properties["thickness"]

        M_e = np.zeros((16, 16))
        for xi, eta, w in _GAUSS_POINTS_3X3:
            Nv = self._shape_functions(xi, eta)      # (8,)
            dN = self._shape_function_derivatives(xi, eta)
            J = dN @ nodes
            det_J = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]

            N_mat = np.zeros((2, 16))
            for i in range(8):
                N_mat[0, 2 * i]     = Nv[i]
                N_mat[1, 2 * i + 1] = Nv[i]

            M_e += w * (N_mat.T @ N_mat) * det_J

        return M_e * (material.rho * t)

    def body_force_vector(
        self,
        material: ElasticMaterial,
        nodes: np.ndarray,
        properties: dict,
        b: np.ndarray,
    ) -> np.ndarray:
        """Forces nodales équivalentes pour une force de volume uniforme b [N/m³].

        f_e = ρ · t · ∫∫ Nᵀ · b |det J| dξ dη  (intégration 3×3 Gauss)

        Parameters
        ----------
        material : ElasticMaterial
            Matériau (rho utilisé).
        nodes : np.ndarray, shape (8, 2)
            Coordonnées nodales.
        properties : dict
            ``"thickness"`` : épaisseur [m].
        b : np.ndarray, shape (2,)
            Force de volume [N/m³].

        Returns
        -------
        f_e : np.ndarray, shape (16,)
            Forces nodales équivalentes [N].
        """
        t = properties.get("thickness", 1.0)
        f_e = np.zeros(16)
        for xi, eta, w in _GAUSS_POINTS_3X3:
            Nv = self._shape_functions(xi, eta)
            dN = self._shape_function_derivatives(xi, eta)
            J = dN @ nodes
            det_J = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
            coeff = w * det_J * material.rho * t
            for i in range(8):
                f_e[2 * i    ] += coeff * Nv[i] * b[0]
                f_e[2 * i + 1] += coeff * Nv[i] * b[1]
        return f_e

    # ------------------------------------------------------------------
    # Post-traitement
    # ------------------------------------------------------------------

    def strain(
        self,
        nodes: np.ndarray,
        u_e: np.ndarray,
        xi: float = 0.0,
        eta: float = 0.0,
    ) -> np.ndarray:
        """Vecteur de déformations ε = B(ξ,η) · u_e.

        Parameters
        ----------
        nodes : np.ndarray, shape (8, 2)
            Coordonnées nodales.
        u_e : np.ndarray, shape (16,)
            Déplacements élémentaires.
        xi, eta : float
            Point d'évaluation (défaut : centre de l'élément).

        Returns
        -------
        epsilon : np.ndarray, shape (3,)
            [εxx, εyy, γxy].
        """
        _, B = self._jacobian_and_B(xi, eta, nodes)
        return B @ u_e

    def stress(
        self,
        material: ElasticMaterial,
        nodes: np.ndarray,
        u_e: np.ndarray,
        xi: float = 0.0,
        eta: float = 0.0,
        formulation: str = "plane_stress",
    ) -> np.ndarray:
        """Vecteur de contraintes σ = D · B(ξ,η) · u_e.

        Parameters
        ----------
        material : ElasticMaterial
            Matériau.
        nodes : np.ndarray, shape (8, 2)
            Coordonnées nodales.
        u_e : np.ndarray, shape (16,)
            Déplacements élémentaires.
        xi, eta : float
            Point d'évaluation (défaut : centre).
        formulation : str
            ``"plane_stress"`` ou ``"plane_strain"``.

        Returns
        -------
        sigma : np.ndarray, shape (3,)
            [σxx, σyy, τxy] [Pa].
        """
        D = self._elasticity_matrix(material, formulation)
        return D @ self.strain(nodes, u_e, xi, eta)

    # ------------------------------------------------------------------
    # Interface batch
    # ------------------------------------------------------------------

    @classmethod
    def batch_stiffness_matrix(
        cls,
        nodes_batch: np.ndarray,
        D: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """Matrices de rigidité pour N_e Quad8 en une seule passe tenseur.

        Parameters
        ----------
        nodes_batch : np.ndarray, shape (N_e, 8, 2)
            Coordonnées nodales de tous les éléments.
        D : np.ndarray, shape (3, 3)
            Matrice d'élasticité (identique pour tout le groupe).
        t : float
            Épaisseur [m].

        Returns
        -------
        K_e_all : np.ndarray, shape (N_e, 16, 16)
            Matrices de rigidité élémentaires.

        Notes
        -----
        1. ``dN_nat`` (9_gp, 2, 8) — dérivées naturelles (constantes par GP).
        2. ``J = einsum('gin,enj->geij', dN_nat, nodes_batch)`` → (9, N_e, 2, 2).
        3. Dérivées physiques via J⁻¹.
        4. B tenseur (9_gp, N_e, 3, 16).
        5. K_e = t · einsum('g,ge,gepi,pq,geqj->eij', w, det_J, B, D, B).
        """
        n_e = nodes_batch.shape[0]

        dN_nat = np.stack([
            cls._shape_function_derivatives(xi, eta)
            for xi, eta, _ in _GAUSS_POINTS_3X3
        ])   # (9_gp, 2, 8)
        w = np.array([wg for _, _, wg in _GAUSS_POINTS_3X3])   # (9,)

        J = np.einsum('gin,enj->geij', dN_nat, nodes_batch)   # (9, N_e, 2, 2)
        det_J = np.linalg.det(J)                               # (9, N_e)
        J_inv = np.linalg.inv(J)                               # (9, N_e, 2, 2)

        dN_phys = np.einsum('geij,gjn->gein', J_inv, dN_nat)  # (9, N_e, 2, 8)

        B = np.zeros((9, n_e, 3, 16))
        for i in range(8):
            c = 2 * i
            B[:, :, 0, c    ] = dN_phys[:, :, 0, i]
            B[:, :, 1, c + 1] = dN_phys[:, :, 1, i]
            B[:, :, 2, c    ] = dN_phys[:, :, 1, i]
            B[:, :, 2, c + 1] = dN_phys[:, :, 0, i]

        K_e_all = np.einsum('g,ge,gepi,pq,geqj->eij', w, det_J, B, D, B) * t
        return K_e_all   # (N_e, 16, 16)

    @classmethod
    def batch_mass_matrix(
        cls,
        nodes_batch: np.ndarray,
        rho: float,
        t: float,
    ) -> np.ndarray:
        """Matrices de masse pour N_e Quad8 en une seule passe tenseur.

        Parameters
        ----------
        nodes_batch : np.ndarray, shape (N_e, 8, 2)
            Coordonnées nodales.
        rho : float
            Masse volumique [kg/m³].
        t : float
            Épaisseur [m].

        Returns
        -------
        M_e_all : np.ndarray, shape (N_e, 16, 16)
        """
        # NtN[g] : (9, 16, 16) — constant, indépendant de la géométrie
        NtN = np.zeros((9, 16, 16))
        for g, (xi, eta, _) in enumerate(_GAUSS_POINTS_3X3):
            Nv = cls._shape_functions(xi, eta)
            N_mat = np.zeros((2, 16))
            for i in range(8):
                N_mat[0, 2 * i]     = Nv[i]
                N_mat[1, 2 * i + 1] = Nv[i]
            NtN[g] = N_mat.T @ N_mat

        dN_nat = np.stack([
            cls._shape_function_derivatives(xi, eta)
            for xi, eta, _ in _GAUSS_POINTS_3X3
        ])
        w = np.array([wg for _, _, wg in _GAUSS_POINTS_3X3])
        J = np.einsum('gin,enj->geij', dN_nat, nodes_batch)
        det_J = np.linalg.det(J)   # (9, N_e)

        M_e_all = np.einsum('g,ge,gij->eij', w, det_J, NtN) * (rho * t)
        return M_e_all   # (N_e, 16, 16)

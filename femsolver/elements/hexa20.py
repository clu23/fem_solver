"""Élément hexaèdre sérendipité H20 isoparamétrique à 20 nœuds (C3D20).

Extension 3D du Quad8 sérendipité
===================================

**Du Quad8 au Hexa20 — même principe, une dimension de plus**

Le Quad8 2D utilise 8 nœuds : 4 coins + 4 milieux d'arêtes.
Le H20 3D est son extension directe : 20 nœuds = 8 coins + 12 milieux d'arêtes.
(L'hexaèdre a 12 arêtes, d'où les 12 nœuds milieux.)

Un H27 Lagrange (équivalent du Q9 en 3D) aurait 27 nœuds = 8 coins +
12 milieux d'arêtes + 6 milieux de faces + 1 nœud central.
Comme pour le Quad8 vs Q9, le H20 supprime ces 7 nœuds intérieurs tout
en conservant la qualité quadratique sur les bords.

**Fonctions de forme sérendipité 3D**

Coins (ξᵢ, ηᵢ, ζᵢ) = (±1, ±1, ±1) :

    Nᵢ = ⅛(1+ξᵢξ)(1+ηᵢη)(1+ζᵢζ)(ξᵢξ + ηᵢη + ζᵢζ − 2)

Milieux sur arêtes ξ (ξᵢ=0, ηᵢ=±1, ζᵢ=±1) :

    Nᵢ = ¼(1−ξ²)(1+ηᵢη)(1+ζᵢζ)

Milieux sur arêtes η (ξᵢ=±1, ηᵢ=0, ζᵢ=±1) :

    Nᵢ = ¼(1+ξᵢξ)(1−η²)(1+ζᵢζ)

Milieux sur arêtes ζ (ξᵢ=±1, ηᵢ=±1, ζᵢ=0) :

    Nᵢ = ¼(1+ξᵢξ)(1+ηᵢη)(1−ζ²)

**Vérification corner formula**

Pour le nœud 0 (ξ₀=η₀=ζ₀=−1), au point (−1,−1,−1) :
    N0 = ⅛(2)(2)(2)(1+1+1−2) = ⅛ · 8 · 1 = 1 ✓
Aux 7 autres coins, au moins un facteur (1+ξᵢξ) est nul → Nᵢ=0 ✓
Au milieu d'une arête (ζᵢ=0, ex nœud 16 à (−1,−1,0)) :
    N0(−1,−1,0) = ⅛(2)(2)(1)(1+1+0−2) = ⅛·4·0 = 0 ✓

**Tableau comparatif**

+----------------+----------+----------+----------+
| Propriété      | Hexa8    | Hexa20   | Hexa27   |
+----------------+----------+----------+----------+
| Nœuds          | 8        | 20       | 27       |
| DDL (3D)       | 24       | 60       | 81       |
| Espace poly.   | trilin.  | sérénd.  | Lagrange |
| Convergence    | O(h²)u   | O(h³)u   | O(h³)u   |
| Locking volume | fort     | modéré   | modéré   |
| Gauss optimal  | 2×2×2    | 3×3×3    | 3×3×3    |
+----------------+----------+----------+----------+

**Pourquoi 3×3×3 Gauss pour le H20 ?**

Même raisonnement que pour le Quad8 :
- Les fonctions de forme sont quadratiques → B a des termes de degré 2.
- B^T D B est de degré 4 en chaque variable.
- La règle n×n×n intègre exactement les polynômes de degré 2n−1.
  Pour degré 4 : n ≥ 2.5, soit n=3.
- Argument de rang : avec 2×2×2 = 8 points, on a au plus 8×6 = 48
  contraintes de déformation pour une matrice 60×60, dont le rang requis
  est 60−6 = 54. Insuffisant → modes parasites. Avec 3×3×3 = 27 points,
  27×6 = 162 contraintes → rang correct.

**Numérotation des 20 nœuds**

Coins (0–7) = même numérotation que Hexa8 :
    0: (−1,−1,−1)  1: (+1,−1,−1)  2: (+1,+1,−1)  3: (−1,+1,−1)
    4: (−1,−1,+1)  5: (+1,−1,+1)  6: (+1,+1,+1)  7: (−1,+1,+1)

Milieux sur face ζ=−1 (arêtes du bas, nœuds 8–11) :
     8: (0,−1,−1)  milieu 0–1
     9: (+1,0,−1)  milieu 1–2
    10: (0,+1,−1)  milieu 2–3
    11: (−1,0,−1)  milieu 3–0

Milieux sur face ζ=+1 (arêtes du haut, nœuds 12–15) :
    12: (0,−1,+1)  milieu 4–5
    13: (+1,0,+1)  milieu 5–6
    14: (0,+1,+1)  milieu 6–7
    15: (−1,0,+1)  milieu 7–4

Milieux sur arêtes ζ (verticales, nœuds 16–19) :
    16: (−1,−1,0)  milieu 0–4
    17: (+1,−1,0)  milieu 1–5
    18: (+1,+1,0)  milieu 2–6
    19: (−1,+1,0)  milieu 3–7

References
----------
Cook et al., « Concepts and Applications of FEA », 4th ed., chap. 6–8.
Zienkiewicz & Taylor, « The FEM for Solid Mechanics », vol. 1, §9.3.
Bathe, « Finite Element Procedures », chap. 5.
"""

from __future__ import annotations

import numpy as np

from femsolver.core.element import Element
from femsolver.core.material import ElasticMaterial


# ---------------------------------------------------------------------------
# Règle de Gauss-Legendre 3×3×3 (27 points)
# ---------------------------------------------------------------------------

_G3 = np.sqrt(3.0 / 5.0)
_W3 = (5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0)
_X3 = (-_G3, 0.0, _G3)

_GAUSS_POINTS_3X3X3: list[tuple[float, float, float, float]] = [
    (xi, eta, zeta, wx * wy * wz)
    for xi, wx in zip(_X3, _W3)
    for eta, wy in zip(_X3, _W3)
    for zeta, wz in zip(_X3, _W3)
]   # 27 points, poids somme = 8.0 = volume du cube [-1,1]³


# ---------------------------------------------------------------------------
# Coordonnées naturelles des 20 nœuds du Hexa20
# ---------------------------------------------------------------------------

# Shape (20, 3) : colonnes = [ξᵢ, ηᵢ, ζᵢ]
_NODE_COORDS = np.array([
    # Coins (0–7)
    [-1, -1, -1],  # 0
    [ 1, -1, -1],  # 1
    [ 1,  1, -1],  # 2
    [-1,  1, -1],  # 3
    [-1, -1,  1],  # 4
    [ 1, -1,  1],  # 5
    [ 1,  1,  1],  # 6
    [-1,  1,  1],  # 7
    # Milieux face ζ=−1 (8–11)
    [ 0, -1, -1],  # 8
    [ 1,  0, -1],  # 9
    [ 0,  1, -1],  # 10
    [-1,  0, -1],  # 11
    # Milieux face ζ=+1 (12–15)
    [ 0, -1,  1],  # 12
    [ 1,  0,  1],  # 13
    [ 0,  1,  1],  # 14
    [-1,  0,  1],  # 15
    # Milieux arêtes ζ (16–19)
    [-1, -1,  0],  # 16
    [ 1, -1,  0],  # 17
    [ 1,  1,  0],  # 18
    [-1,  1,  0],  # 19
], dtype=float)


class Hexa20(Element):
    """Hexaèdre sérendipité H20 isoparamétrique — élasticité 3D.

    20 nœuds, 3 DDL par nœud (ux, uy, uz) → matrices élémentaires 60×60.
    Intégration numérique 3×3×3 points de Gauss (27 points).

    Numérotation des nœuds :
        Coins (0–7) : même que Hexa8
            7-----14-----6
           /|            /|      ζ
          15 19    13  18 |      |   η
         /  |       /    |      |  /
        4-----12-----5   |      | /
        |   3----10--|---2      +-----ξ
        16  |    17  18  |
        |   11       |   9
        |  /         |  /
        |15           |13
        |/            |/
        0------8------1

    Ordre des DDL : [ux0,uy0,uz0, ux1,uy1,uz1, …, ux19,uy19,uz19].

    Parameters (via ``properties``)
    --------------------------------
    formulation : str
        Ignoré (toujours 3D isotrope). Présent pour compatibilité d'interface.

    Examples
    --------
    >>> import numpy as np
    >>> from femsolver.core.material import ElasticMaterial
    >>> mat = ElasticMaterial(E=1.0, nu=0.0, rho=1.0)
    >>> nodes = _NODE_COORDS.copy()  # cube de référence [-1,1]³
    >>> from femsolver.elements.hexa20 import Hexa20, _NODE_COORDS
    >>> K_e = Hexa20().stiffness_matrix(mat, nodes, {})
    >>> K_e.shape
    (60, 60)
    """

    def dof_per_node(self) -> int:
        """3 DDL par nœud : ux, uy, uz."""
        return 3

    def n_nodes(self) -> int:
        """20 nœuds."""
        return 20

    # ------------------------------------------------------------------
    # Fonctions de forme et leurs dérivées
    # ------------------------------------------------------------------

    @staticmethod
    def _shape_functions(xi: float, eta: float, zeta: float) -> np.ndarray:
        """Évaluer les 20 fonctions de forme sérendipité au point (ξ, η, ζ).

        Parameters
        ----------
        xi, eta, zeta : float
            Coordonnées naturelles dans [−1, 1].

        Returns
        -------
        N : np.ndarray, shape (20,)
            [N0, N1, …, N19].

        Notes
        -----
        Coins i=0..7 :
            Nᵢ = ⅛(1+ξᵢξ)(1+ηᵢη)(1+ζᵢζ)(ξᵢξ + ηᵢη + ζᵢζ − 2)

        Milieux arêtes ξ (ξᵢ=0, ηᵢ=±1, ζᵢ=±1), nœuds 8–15 :
            Nᵢ = ¼(1−ξ²)(1+ηᵢη)(1+ζᵢζ)

        Milieux arêtes η (ξᵢ=±1, ηᵢ=0, ζᵢ=±1), nœuds 9,11,13,15 :
            Nᵢ = ¼(1+ξᵢξ)(1−η²)(1+ζᵢζ)

        Milieux arêtes ζ (ξᵢ=±1, ηᵢ=±1, ζᵢ=0), nœuds 16–19 :
            Nᵢ = ¼(1+ξᵢξ)(1+ηᵢη)(1−ζ²)
        """
        N = np.empty(20)
        # Coins (0–7) : les 3 coords valent ±1
        for i in range(8):
            xi_i, eta_i, zeta_i = _NODE_COORDS[i]
            a = 1.0 + xi_i * xi
            b = 1.0 + eta_i * eta
            c = 1.0 + zeta_i * zeta
            N[i] = 0.125 * a * b * c * (xi_i * xi + eta_i * eta + zeta_i * zeta - 2.0)

        # Nœuds 8–15 : milieux des arêtes de la face ζ=−1 et ζ=+1
        # Ces nœuds ont ξᵢ=0 ou ηᵢ=0 (mais pas les deux)
        for i in range(8, 20):
            xi_i, eta_i, zeta_i = _NODE_COORDS[i]
            if xi_i == 0.0:
                # Milieu d'une arête ξ (ξᵢ=0, ηᵢ=±1, ζᵢ=±1)
                N[i] = 0.25 * (1.0 - xi**2) * (1.0 + eta_i * eta) * (1.0 + zeta_i * zeta)
            elif eta_i == 0.0:
                # Milieu d'une arête η (ξᵢ=±1, ηᵢ=0, ζᵢ=±1)
                N[i] = 0.25 * (1.0 + xi_i * xi) * (1.0 - eta**2) * (1.0 + zeta_i * zeta)
            else:
                # Milieu d'une arête ζ (ξᵢ=±1, ηᵢ=±1, ζᵢ=0)
                N[i] = 0.25 * (1.0 + xi_i * xi) * (1.0 + eta_i * eta) * (1.0 - zeta**2)
        return N

    @staticmethod
    def _shape_function_derivatives(xi: float, eta: float, zeta: float) -> np.ndarray:
        """Dérivées ∂Nᵢ/∂ξ, ∂Nᵢ/∂η, ∂Nᵢ/∂ζ au point (ξ, η, ζ).

        Parameters
        ----------
        xi, eta, zeta : float
            Coordonnées naturelles.

        Returns
        -------
        dN : np.ndarray, shape (3, 20)
            dN[0, i] = ∂Nᵢ/∂ξ,
            dN[1, i] = ∂Nᵢ/∂η,
            dN[2, i] = ∂Nᵢ/∂ζ.

        Notes
        -----
        Coins (ξᵢ, ηᵢ, ζᵢ) = (±1, ±1, ±1) :
            ∂Nᵢ/∂ξ = ⅛ξᵢ(1+ηᵢη)(1+ζᵢζ)(2ξᵢξ + ηᵢη + ζᵢζ − 1)
            ∂Nᵢ/∂η = ⅛ηᵢ(1+ξᵢξ)(1+ζᵢζ)(ξᵢξ + 2ηᵢη + ζᵢζ − 1)
            ∂Nᵢ/∂ζ = ⅛ζᵢ(1+ξᵢξ)(1+ηᵢη)(ξᵢξ + ηᵢη + 2ζᵢζ − 1)

        Milieux ξᵢ=0 :
            ∂Nᵢ/∂ξ = −½ξ(1+ηᵢη)(1+ζᵢζ)
            ∂Nᵢ/∂η = ¼ηᵢ(1−ξ²)(1+ζᵢζ)
            ∂Nᵢ/∂ζ = ¼ζᵢ(1−ξ²)(1+ηᵢη)

        Milieux ηᵢ=0 :
            ∂Nᵢ/∂ξ = ¼ξᵢ(1−η²)(1+ζᵢζ)
            ∂Nᵢ/∂η = −½η(1+ξᵢξ)(1+ζᵢζ)
            ∂Nᵢ/∂ζ = ¼ζᵢ(1+ξᵢξ)(1−η²)

        Milieux ζᵢ=0 :
            ∂Nᵢ/∂ξ = ¼ξᵢ(1+ηᵢη)(1−ζ²)
            ∂Nᵢ/∂η = ¼ηᵢ(1+ξᵢξ)(1−ζ²)
            ∂Nᵢ/∂ζ = −½ζ(1+ξᵢξ)(1+ηᵢη)
        """
        dN = np.empty((3, 20))
        # Coins (0–7)
        for i in range(8):
            xi_i, eta_i, zeta_i = _NODE_COORDS[i]
            a = 1.0 + xi_i * xi
            b = 1.0 + eta_i * eta
            c = 1.0 + zeta_i * zeta
            s = xi_i * xi + eta_i * eta + zeta_i * zeta - 2.0   # term (sum−2)
            # d(a·b·c·s)/da · da/dξ = ξᵢ · b·c·s + a·b·c·ξᵢ = ξᵢ·b·c·(s+a)
            # s + a = ξᵢξ+ηᵢη+ζᵢζ−2 + 1+ξᵢξ = 2ξᵢξ+ηᵢη+ζᵢζ−1
            dN[0, i] = 0.125 * xi_i   * b * c * (2.0 * xi_i   * xi  + eta_i * eta + zeta_i * zeta - 1.0)
            dN[1, i] = 0.125 * eta_i  * a * c * (xi_i * xi  + 2.0 * eta_i  * eta + zeta_i * zeta - 1.0)
            dN[2, i] = 0.125 * zeta_i * a * b * (xi_i * xi  + eta_i * eta + 2.0 * zeta_i * zeta - 1.0)

        # Nœuds milieux (8–19)
        for i in range(8, 20):
            xi_i, eta_i, zeta_i = _NODE_COORDS[i]
            if xi_i == 0.0:
                b = 1.0 + eta_i * eta
                c = 1.0 + zeta_i * zeta
                dN[0, i] = -0.5 * xi  * b * c
                dN[1, i] =  0.25 * eta_i  * (1.0 - xi**2) * c
                dN[2, i] =  0.25 * zeta_i * (1.0 - xi**2) * b
            elif eta_i == 0.0:
                a = 1.0 + xi_i * xi
                c = 1.0 + zeta_i * zeta
                dN[0, i] =  0.25 * xi_i   * (1.0 - eta**2) * c
                dN[1, i] = -0.5 * eta  * a * c
                dN[2, i] =  0.25 * zeta_i * a * (1.0 - eta**2)
            else:  # zeta_i == 0.0
                a = 1.0 + xi_i * xi
                b = 1.0 + eta_i * eta
                dN[0, i] =  0.25 * xi_i  * b * (1.0 - zeta**2)
                dN[1, i] =  0.25 * eta_i * a * (1.0 - zeta**2)
                dN[2, i] = -0.5 * zeta * a * b

        return dN

    # ------------------------------------------------------------------
    # Jacobien et matrice B
    # ------------------------------------------------------------------

    def _jacobian_and_B(
        self,
        xi: float,
        eta: float,
        zeta: float,
        nodes: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """det(J) et matrice B (6×60) au point de Gauss (ξ, η, ζ).

        Parameters
        ----------
        xi, eta, zeta : float
            Coordonnées naturelles.
        nodes : np.ndarray, shape (20, 3)
            Coordonnées physiques des 20 nœuds.

        Returns
        -------
        det_J : float
            Déterminant du Jacobien (> 0 si nœuds bien orientés).
        B : np.ndarray, shape (6, 60)
            Matrice déformation–déplacement (notation de Voigt).
            ε = [εxx, εyy, εzz, γyz, γxz, γxy] = B · u_e.

        Raises
        ------
        ValueError
            Si det(J) ≤ 0.

        Notes
        -----
        Jacobien J (3×3) = dN/d(ξ,η,ζ) @ nodes.

        Pour le nœud i (colonnes 3i, 3i+1, 3i+2) :
            B[0, 3i]   = ∂Nᵢ/∂x            (εxx)
            B[1, 3i+1] = ∂Nᵢ/∂y            (εyy)
            B[2, 3i+2] = ∂Nᵢ/∂z            (εzz)
            B[3, 3i+1] = ∂Nᵢ/∂z  (γyz, uy) B[3, 3i+2] = ∂Nᵢ/∂y  (γyz, uz)
            B[4, 3i]   = ∂Nᵢ/∂z  (γxz, ux) B[4, 3i+2] = ∂Nᵢ/∂x  (γxz, uz)
            B[5, 3i]   = ∂Nᵢ/∂y  (γxy, ux) B[5, 3i+1] = ∂Nᵢ/∂x  (γxy, uy)
        """
        dN = self._shape_function_derivatives(xi, eta, zeta)   # (3, 20)
        J = dN @ nodes                                           # (3, 3)
        det_J = np.linalg.det(J)

        if det_J <= 0.0:
            raise ValueError(
                f"det(J) = {det_J:.6g} ≤ 0 au point "
                f"(ξ={xi:.4f}, η={eta:.4f}, ζ={zeta:.4f}). "
                "Vérifier l'orientation et la position des nœuds milieux."
            )

        dN_phys = np.linalg.solve(J, dN)   # (3, 20) : dN_phys[k]=∂N/∂xk

        B = np.zeros((6, 60))
        for i in range(20):
            c = 3 * i
            dNx, dNy, dNz = dN_phys[0, i], dN_phys[1, i], dN_phys[2, i]
            B[0, c    ] = dNx   # εxx
            B[1, c + 1] = dNy   # εyy
            B[2, c + 2] = dNz   # εzz
            B[3, c + 1] = dNz   # γyz (uy)
            B[3, c + 2] = dNy   # γyz (uz)
            B[4, c    ] = dNz   # γxz (ux)
            B[4, c + 2] = dNx   # γxz (uz)
            B[5, c    ] = dNy   # γxy (ux)
            B[5, c + 1] = dNx   # γxy (uy)

        return det_J, B

    # ------------------------------------------------------------------
    # Matrices élémentaires
    # ------------------------------------------------------------------

    def stiffness_matrix(
        self,
        material: ElasticMaterial,
        nodes: np.ndarray,
        properties: dict,
    ) -> np.ndarray:
        """Matrice de rigidité élémentaire K_e (60×60).

        Parameters
        ----------
        material : ElasticMaterial
            Matériau (E, nu utilisés).
        nodes : np.ndarray, shape (20, 3)
            Coordonnées [[x0,y0,z0], …, [x19,y19,z19]].
        properties : dict
            Non utilisé pour les éléments 3D (conservé pour compatibilité).

        Returns
        -------
        K_e : np.ndarray, shape (60, 60)
            Matrice de rigidité symétrique.

        Notes
        -----
        Intégration 3×3×3 Gauss (27 points) :

            K_e = Σₚ wₚ · Bₚᵀ · D · Bₚ · |det Jₚ|

        avec D (6×6) la matrice d'élasticité isotrope 3D :

                  [1−ν  ν   ν   0        0        0       ]
              E   [ ν  1−ν  ν   0        0        0       ]
        D = ──── · [ ν   ν  1−ν  0        0        0       ]
            (1+ν)(1−2ν)[ 0   0   0  (1−2ν)/2   0        0       ]
                  [ 0   0   0   0   (1−2ν)/2   0       ]
                  [ 0   0   0   0        0   (1−2ν)/2 ]
        """
        if nodes.shape != (20, 3):
            raise ValueError(f"Hexa20 attend nodes.shape == (20, 3), reçu {nodes.shape}")
        D = material.elasticity_matrix_3d()

        K_e = np.zeros((60, 60))
        for xi, eta, zeta, w in _GAUSS_POINTS_3X3X3:
            det_J, B = self._jacobian_and_B(xi, eta, zeta, nodes)
            K_e += (w * det_J) * (B.T @ D @ B)
        return K_e

    def mass_matrix(
        self,
        material: ElasticMaterial,
        nodes: np.ndarray,
        properties: dict,
    ) -> np.ndarray:
        """Matrice de masse consistante M_e (60×60).

        Parameters
        ----------
        material : ElasticMaterial
            Matériau (rho utilisé).
        nodes : np.ndarray, shape (20, 3)
            Coordonnées nodales.
        properties : dict
            Non utilisé (conservé pour compatibilité d'interface).

        Returns
        -------
        M_e : np.ndarray, shape (60, 60)
            Matrice de masse consistante par intégration 3×3×3 Gauss.

        Notes
        -----
        M_e = ρ · ∫∫∫ Nᵀ N |det J| dξ dη dζ

        Matrice N (3×60) :
            N[0, 3i]   = Nᵢ   (composante ux)
            N[1, 3i+1] = Nᵢ   (composante uy)
            N[2, 3i+2] = Nᵢ   (composante uz)
        """
        if nodes.shape != (20, 3):
            raise ValueError(f"Hexa20 attend nodes.shape == (20, 3), reçu {nodes.shape}")

        M_e = np.zeros((60, 60))
        for xi, eta, zeta, w in _GAUSS_POINTS_3X3X3:
            Nv = self._shape_functions(xi, eta, zeta)   # (20,)
            dN = self._shape_function_derivatives(xi, eta, zeta)
            J = dN @ nodes
            det_J = np.linalg.det(J)

            N_mat = np.zeros((3, 60))
            for i in range(20):
                N_mat[0, 3 * i]     = Nv[i]
                N_mat[1, 3 * i + 1] = Nv[i]
                N_mat[2, 3 * i + 2] = Nv[i]

            M_e += w * (N_mat.T @ N_mat) * det_J

        return M_e * material.rho

    def body_force_vector(
        self,
        material: ElasticMaterial,
        nodes: np.ndarray,
        properties: dict,
        b: np.ndarray,
    ) -> np.ndarray:
        """Forces nodales équivalentes pour une force de volume uniforme b [N/m³].

        f_e = ρ · ∫∫∫ Nᵀ · b |det J| dξ dη dζ

        Parameters
        ----------
        material : ElasticMaterial
            Matériau (rho utilisé).
        nodes : np.ndarray, shape (20, 3)
        properties : dict
            Non utilisé.
        b : np.ndarray, shape (3,)
            Force de volume [N/m³].

        Returns
        -------
        f_e : np.ndarray, shape (60,)
        """
        f_e = np.zeros(60)
        for xi, eta, zeta, w in _GAUSS_POINTS_3X3X3:
            Nv = self._shape_functions(xi, eta, zeta)
            dN = self._shape_function_derivatives(xi, eta, zeta)
            J = dN @ nodes
            det_J = np.linalg.det(J)
            coeff = w * det_J * material.rho
            for i in range(20):
                f_e[3 * i    ] += coeff * Nv[i] * b[0]
                f_e[3 * i + 1] += coeff * Nv[i] * b[1]
                f_e[3 * i + 2] += coeff * Nv[i] * b[2]
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
        zeta: float = 0.0,
    ) -> np.ndarray:
        """Vecteur de déformations ε = B(ξ,η,ζ) · u_e.

        Parameters
        ----------
        nodes : np.ndarray, shape (20, 3)
        u_e : np.ndarray, shape (60,)
        xi, eta, zeta : float
            Point d'évaluation (défaut : centre).

        Returns
        -------
        epsilon : np.ndarray, shape (6,)
            [εxx, εyy, εzz, γyz, γxz, γxy].
        """
        _, B = self._jacobian_and_B(xi, eta, zeta, nodes)
        return B @ u_e

    def stress(
        self,
        material: ElasticMaterial,
        nodes: np.ndarray,
        u_e: np.ndarray,
        xi: float = 0.0,
        eta: float = 0.0,
        zeta: float = 0.0,
    ) -> np.ndarray:
        """Vecteur de contraintes σ = D · B · u_e.

        Parameters
        ----------
        material : ElasticMaterial
        nodes : np.ndarray, shape (20, 3)
        u_e : np.ndarray, shape (60,)
        xi, eta, zeta : float
            Point d'évaluation (défaut : centre).

        Returns
        -------
        sigma : np.ndarray, shape (6,)
            [σxx, σyy, σzz, τyz, τxz, τxy] [Pa].
        """
        D = material.elasticity_matrix_3d()
        return D @ self.strain(nodes, u_e, xi, eta, zeta)

    # ------------------------------------------------------------------
    # Interface batch
    # ------------------------------------------------------------------

    @classmethod
    def batch_stiffness_matrix(
        cls,
        nodes_batch: np.ndarray,
        D: np.ndarray,
        properties: dict | None = None,
    ) -> np.ndarray:
        """Matrices de rigidité pour N_e Hexa20 en une seule passe tenseur.

        Parameters
        ----------
        nodes_batch : np.ndarray, shape (N_e, 20, 3)
            Coordonnées nodales.
        D : np.ndarray, shape (6, 6)
            Matrice d'élasticité 3D (identique pour tout le groupe).
        properties : dict or None
            Non utilisé.

        Returns
        -------
        K_e_all : np.ndarray, shape (N_e, 60, 60)
        """
        n_e = nodes_batch.shape[0]

        dN_nat = np.stack([
            cls._shape_function_derivatives(xi, eta, zeta)
            for xi, eta, zeta, _ in _GAUSS_POINTS_3X3X3
        ])   # (27_gp, 3, 20)
        w = np.array([wg for _, _, _, wg in _GAUSS_POINTS_3X3X3])   # (27,)

        J = np.einsum('gin,enj->geij', dN_nat, nodes_batch)   # (27, N_e, 3, 3)
        det_J = np.linalg.det(J)                               # (27, N_e)
        J_inv = np.linalg.inv(J)                               # (27, N_e, 3, 3)

        dN_phys = np.einsum('geij,gjn->gein', J_inv, dN_nat)  # (27, N_e, 3, 20)

        B = np.zeros((27, n_e, 6, 60))
        for i in range(20):
            c = 3 * i
            B[:, :, 0, c    ] = dN_phys[:, :, 0, i]
            B[:, :, 1, c + 1] = dN_phys[:, :, 1, i]
            B[:, :, 2, c + 2] = dN_phys[:, :, 2, i]
            B[:, :, 3, c + 1] = dN_phys[:, :, 2, i]
            B[:, :, 3, c + 2] = dN_phys[:, :, 1, i]
            B[:, :, 4, c    ] = dN_phys[:, :, 2, i]
            B[:, :, 4, c + 2] = dN_phys[:, :, 0, i]
            B[:, :, 5, c    ] = dN_phys[:, :, 1, i]
            B[:, :, 5, c + 1] = dN_phys[:, :, 0, i]

        K_e_all = np.einsum('g,ge,gepi,pq,geqj->eij', w, det_J, B, D, B)
        return K_e_all   # (N_e, 60, 60)

    @classmethod
    def batch_mass_matrix(
        cls,
        nodes_batch: np.ndarray,
        rho: float,
        properties: dict | None = None,
    ) -> np.ndarray:
        """Matrices de masse pour N_e Hexa20 en une seule passe tenseur.

        Parameters
        ----------
        nodes_batch : np.ndarray, shape (N_e, 20, 3)
        rho : float
        properties : dict or None

        Returns
        -------
        M_e_all : np.ndarray, shape (N_e, 60, 60)
        """
        NtN = np.zeros((27, 60, 60))
        for g, (xi, eta, zeta, _) in enumerate(_GAUSS_POINTS_3X3X3):
            Nv = cls._shape_functions(xi, eta, zeta)
            N_mat = np.zeros((3, 60))
            for i in range(20):
                N_mat[0, 3 * i]     = Nv[i]
                N_mat[1, 3 * i + 1] = Nv[i]
                N_mat[2, 3 * i + 2] = Nv[i]
            NtN[g] = N_mat.T @ N_mat

        dN_nat = np.stack([
            cls._shape_function_derivatives(xi, eta, zeta)
            for xi, eta, zeta, _ in _GAUSS_POINTS_3X3X3
        ])
        w = np.array([wg for _, _, _, wg in _GAUSS_POINTS_3X3X3])
        J = np.einsum('gin,enj->geij', dN_nat, nodes_batch)
        det_J = np.linalg.det(J)

        M_e_all = np.einsum('g,ge,gij->eij', w, det_J, NtN) * rho
        return M_e_all

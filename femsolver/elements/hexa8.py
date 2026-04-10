"""Élément hexaèdre trilinéaire isoparamétrique à 8 nœuds (C3D8).

Extension naturelle du Quad4 en 3D : fonctions de forme trilinéaires
======================================================================

**Du Quad4 au Hexa8 — la même idée, une dimension de plus**

Le Quad4 2D utilise des fonctions de forme *bilinéaires* :

    Ni(ξ, η) = ¼ (1 + ξᵢ ξ)(1 + ηᵢ η)          (ξ, η) ∈ [-1, 1]²

C'est le produit tensoriel de deux fonctions de forme 1D linéaires.
Le terme "bili" vient du produit ξη qui apparaît quand on développe.

Le Hexa8 3D est son extension directe :

    Ni(ξ, η, ζ) = ⅛ (1 + ξᵢ ξ)(1 + ηᵢ η)(1 + ζᵢ ζ)    (ξ,η,ζ) ∈ [-1, 1]³

C'est le produit tensoriel de *trois* fonctions 1D — d'où "trilinéaire".

Comparaison côte à côte
-----------------------

+------------------+-----------------------------+--------------------------------+
| Propriété        | Quad4                       | Hexa8                          |
+------------------+-----------------------------+--------------------------------+
| Dimensions       | 2D, (ξ,η)                  | 3D, (ξ,η,ζ)                   |
| Nœuds            | 4                           | 8                              |
| DDL totaux       | 8 (2/nœud)                  | 24 (3/nœud)                    |
| Fonctions de forme| bilinéaires                 | trilinéaires                   |
| Champs exacts    | {1, ξ, η, ξη}              | {1, ξ, η, ζ, ξη, ηζ, ξζ, ξηζ}|
| Points de Gauss  | 2×2 = 4                     | 2×2×2 = 8                     |
| Matrice B        | 3×8                         | 6×24                           |
| K_e, M_e         | 8×8                         | 24×24                          |
+------------------+-----------------------------+--------------------------------+

Fonctions de forme trilinéaires — géométrie
-------------------------------------------
Les 8 nœuds occupent les coins du cube de référence [-1,1]³ :

        7-----6
       /|    /|       ζ
      4-----5 |       |   η
      | 3---|-2       |  /
      |/    |/        | /
      0-----1         +-----ξ

Coordonnées naturelles des nœuds :
    0: (-1,-1,-1)   1: (+1,-1,-1)   2: (+1,+1,-1)   3: (-1,+1,-1)
    4: (-1,-1,+1)   5: (+1,-1,+1)   6: (+1,+1,+1)   7: (-1,+1,+1)

Les ξᵢ, ηᵢ, ζᵢ valent ±1 selon le coin. Chaque Nᵢ vaut 1 en son propre nœud
et 0 dans les 7 autres (partition de l'unité : Σ Nᵢ = 1 partout).

Développé pour le nœud 0 (ξ₀=-1, η₀=-1, ζ₀=-1) :
    N0 = ⅛ (1-ξ)(1-η)(1-ζ)

Ce terme "s'allume" uniquement quand ξ→-1, η→-1, ζ→-1, ce qui correspond
exactement au coin 0. Tous les autres nœuds ont au moins un facteur nul.

Intégration numérique 2×2×2 Gauss
----------------------------------
Les points de Gauss sont aux positions (±1/√3, ±1/√3, ±1/√3), poids = 1.
C'est exactement la règle de Gauss-Legendre 1D appliquée indépendamment
sur chaque direction — produit tensoriel de 3 règles 1D à 2 points.

Pour un hexaèdre régulier (parallélépipède), 2×2×2 Gauss est exact pour des
polynômes de degré ≤ 3 en chaque variable. K_e est intégré exactement car
les intégrandes sont au maximum de degré 2 en (ξ,η,ζ) pour les éléments
réguliers. Pour les éléments distordus, l'erreur est du même ordre que la
distorsion.

Matrice B 3D (6×24)
-------------------
Même structure que le Tetra4, mais pour 8 nœuds.
Le vecteur de déformation (notation de Voigt) :
    ε = [εxx, εyy, εzz, γyz, γxz, γxy]

Pour le nœud i (colonnes 3i, 3i+1, 3i+2) :
    B[:, 3i:3i+3] =
        [∂Ni/∂x,    0,        0      ]   ← εxx = ∂ux/∂x
        [0,         ∂Ni/∂y,   0      ]   ← εyy = ∂uy/∂y
        [0,         0,        ∂Ni/∂z ]   ← εzz = ∂uz/∂z
        [0,         ∂Ni/∂z,   ∂Ni/∂y ]  ← γyz = ∂uy/∂z + ∂uz/∂y
        [∂Ni/∂z,   0,        ∂Ni/∂x ]   ← γxz = ∂ux/∂z + ∂uz/∂x
        [∂Ni/∂y,   ∂Ni/∂x,   0      ]   ← γxy = ∂ux/∂y + ∂uy/∂x

Contrairement au Tetra4, B varie d'un point de Gauss à l'autre (car les
fonctions de forme sont trilinéaires, non linéaires). On intègre donc :

    K_e = Σₚ wₚ Bₚᵀ D Bₚ |det Jₚ|

Matrice de masse consistante (24×24)
-------------------------------------
M_e = ρ · Σₚ wₚ Nᵀ N |det Jₚ|

où N est la matrice d'interpolation 3×24.
Pour un cube a×b×c, la masse est ρabc (exacte par construction).

Numérotation et orientation
---------------------------
La face inférieure (ζ=-1) est 0→1→2→3 dans le sens trigonométrique vue
de bas en haut. La face supérieure (ζ=+1) est 4→5→6→7.
Cette convention assure det(J) > 0 pour tout hexaèdre convexe bien orienté.

References
----------
Cook et al., « Concepts and Applications of FEA », 4th ed., chap. 6.
Bathe, « Finite Element Procedures », 2nd ed., chap. 5.
Hughes, « The Finite Element Method », chap. 3.
Zienkiewicz & Taylor, « The FEM — Solid Mechanics », vol. 2, chap. 8.
"""

from __future__ import annotations

import numpy as np

from femsolver.core.element import Element
from femsolver.core.material import ElasticMaterial


# ---------------------------------------------------------------------------
# Points et poids de Gauss 2×2×2
# ---------------------------------------------------------------------------

_GP = 1.0 / np.sqrt(3.0)  # ±1/√3 ≈ ±0.5774

# 8 points de Gauss : (ξ, η, ζ, poids)  — tous de poids 1
_GAUSS_POINTS_2X2X2: list[tuple[float, float, float, float]] = [
    (-_GP, -_GP, -_GP, 1.0),
    ( _GP, -_GP, -_GP, 1.0),
    ( _GP,  _GP, -_GP, 1.0),
    (-_GP,  _GP, -_GP, 1.0),
    (-_GP, -_GP,  _GP, 1.0),
    ( _GP, -_GP,  _GP, 1.0),
    ( _GP,  _GP,  _GP, 1.0),
    (-_GP,  _GP,  _GP, 1.0),
]

# ---------------------------------------------------------------------------
# Coordonnées naturelles des 8 nœuds (chacune vaut ±1)
# Forme : (8, 3)  — colonnes = (ξᵢ, ηᵢ, ζᵢ)
# ---------------------------------------------------------------------------

_NODE_NAT: np.ndarray = np.array([
    [-1.0, -1.0, -1.0],  # nœud 0  coin bas-avant-gauche
    [ 1.0, -1.0, -1.0],  # nœud 1  coin bas-avant-droit
    [ 1.0,  1.0, -1.0],  # nœud 2  coin bas-arrière-droit
    [-1.0,  1.0, -1.0],  # nœud 3  coin bas-arrière-gauche
    [-1.0, -1.0,  1.0],  # nœud 4  coin haut-avant-gauche
    [ 1.0, -1.0,  1.0],  # nœud 5  coin haut-avant-droit
    [ 1.0,  1.0,  1.0],  # nœud 6  coin haut-arrière-droit
    [-1.0,  1.0,  1.0],  # nœud 7  coin haut-arrière-gauche
], dtype=float)


class Hexa8(Element):
    """Hexaèdre trilinéaire isoparamétrique à 8 nœuds — élasticité 3D.

    8 nœuds, 3 DDL par nœud (ux, uy, uz) → matrices élémentaires 24×24.
    Intégration numérique 2×2×2 points de Gauss.

    Numérotation des nœuds (face ζ=-1 puis ζ=+1) :

           7-----6
          /|    /|       ζ
         4-----5 |       |   η
         | 3---|-2       |  /
         |/    |/        | /
         0-----1         +-----ξ

    Face inférieure (ζ = -1) : 0→1→2→3 sens trigonométrique vue de dessous.
    Face supérieure (ζ = +1) : 4→5→6→7, nœuds i+4 au-dessus du nœud i.

    Ordre des DDL : [u0,v0,w0, u1,v1,w1, …, u7,v7,w7].

    Parameters (via ``properties``)
    --------------------------------
    Aucun paramètre requis. Le dict peut être vide ``{}``.

    Raises
    ------
    ValueError
        Si nodes.shape ≠ (8, 3).
    ValueError
        Si det(J) ≤ 0 à un point de Gauss (élément dégénéré ou mal orienté).

    Examples
    --------
    >>> import numpy as np
    >>> from femsolver.core.material import ElasticMaterial
    >>> mat = ElasticMaterial(E=210e9, nu=0.3, rho=7800)
    >>> nodes = np.array([
    ...     [0.,0.,0.],[1.,0.,0.],[1.,1.,0.],[0.,1.,0.],
    ...     [0.,0.,1.],[1.,0.,1.],[1.,1.,1.],[0.,1.,1.],
    ... ])
    >>> K_e = Hexa8().stiffness_matrix(mat, nodes, {})
    >>> K_e.shape
    (24, 24)
    """

    def dof_per_node(self) -> int:
        """3 DDL par nœud : ux, uy, uz."""
        return 3

    def n_nodes(self) -> int:
        """8 nœuds."""
        return 8

    # ------------------------------------------------------------------
    # Fonctions de forme et leurs dérivées
    # ------------------------------------------------------------------

    @staticmethod
    def _shape_functions(xi: float, eta: float, zeta: float) -> np.ndarray:
        """Fonctions de forme trilinéaires N₀…N₇ au point (ξ, η, ζ).

        Parameters
        ----------
        xi, eta, zeta : float
            Coordonnées naturelles dans [-1, 1].

        Returns
        -------
        N : np.ndarray, shape (8,)
            Valeurs des 8 fonctions de forme.

        Notes
        -----
        Nᵢ = ⅛ (1 + ξᵢ ξ)(1 + ηᵢ η)(1 + ζᵢ ζ)

        Extension directe du Quad4 : le facteur ¼ devient ⅛, et on ajoute
        le troisième facteur (1 + ζᵢ ζ). Chaque Nᵢ vaut 1 en son nœud et
        0 dans les 7 autres.
        """
        return (
            0.125
            * (1.0 + _NODE_NAT[:, 0] * xi)
            * (1.0 + _NODE_NAT[:, 1] * eta)
            * (1.0 + _NODE_NAT[:, 2] * zeta)
        )

    @staticmethod
    def _shape_function_derivatives(
        xi: float, eta: float, zeta: float
    ) -> np.ndarray:
        """Dérivées ∂Nᵢ/∂ξ, ∂Nᵢ/∂η, ∂Nᵢ/∂ζ au point (ξ, η, ζ).

        Parameters
        ----------
        xi, eta, zeta : float
            Coordonnées naturelles.

        Returns
        -------
        dN : np.ndarray, shape (3, 8)
            dN[0, i] = ∂Nᵢ/∂ξ
            dN[1, i] = ∂Nᵢ/∂η
            dN[2, i] = ∂Nᵢ/∂ζ

        Notes
        -----
        ∂Nᵢ/∂ξ = ⅛ ξᵢ (1 + ηᵢ η)(1 + ζᵢ ζ)   ← dérivée du facteur ξ
        ∂Nᵢ/∂η = ⅛ ηᵢ (1 + ξᵢ ξ)(1 + ζᵢ ζ)   ← dérivée du facteur η
        ∂Nᵢ/∂ζ = ⅛ ζᵢ (1 + ξᵢ ξ)(1 + ηᵢ η)   ← dérivée du facteur ζ
        """
        xi_n   = _NODE_NAT[:, 0]  # ξᵢ pour i=0…7
        eta_n  = _NODE_NAT[:, 1]  # ηᵢ
        zeta_n = _NODE_NAT[:, 2]  # ζᵢ

        dN = np.empty((3, 8))
        dN[0] = 0.125 * xi_n   * (1.0 + eta_n  * eta ) * (1.0 + zeta_n * zeta)
        dN[1] = 0.125 * eta_n  * (1.0 + xi_n   * xi  ) * (1.0 + zeta_n * zeta)
        dN[2] = 0.125 * zeta_n * (1.0 + xi_n   * xi  ) * (1.0 + eta_n  * eta )
        return dN

    # ------------------------------------------------------------------
    # Jacobien et matrice B
    # ------------------------------------------------------------------

    def _jacobian(
        self,
        dN: np.ndarray,
        nodes: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Jacobien J (3×3) et son déterminant.

        Parameters
        ----------
        dN : np.ndarray, shape (3, 8)
            Dérivées des fonctions de forme en (ξ, η, ζ).
        nodes : np.ndarray, shape (8, 3)
            Coordonnées physiques des 8 nœuds.

        Returns
        -------
        J : np.ndarray, shape (3, 3)
            Jacobien J[i,j] = ∂xⱼ/∂ξᵢ.
        det_J : float
            Déterminant du Jacobien (> 0 si nœuds bien orientés).

        Raises
        ------
        ValueError
            Si |det(J)| < 1e-14 ou det(J) < 0.

        Notes
        -----
        J = dN · nodes   (3,8) @ (8,3) → (3,3)
        J[i, j] = Σ_k (∂Nk/∂ξi) · xj_k
        """
        J = dN @ nodes  # (3,8) @ (8,3) = (3,3)
        det_J = np.linalg.det(J)
        if abs(det_J) < 1e-14:
            raise ValueError(
                f"Jacobien singulier pour Hexa8 : det(J) ≈ 0.\n"
                f"Vérifier la numérotation et les coordonnées :\n{nodes}"
            )
        if det_J < 0.0:
            raise ValueError(
                f"det(J) = {det_J:.6g} < 0 — vérifier l'ordre des nœuds "
                f"du Hexa8 (orientation positive requise)."
            )
        return J, det_J

    def _strain_displacement_matrix(
        self,
        xi: float,
        eta: float,
        zeta: float,
        nodes: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Matrice B (6×24) et det(J) au point de Gauss (ξ, η, ζ).

        Parameters
        ----------
        xi, eta, zeta : float
            Coordonnées naturelles du point de Gauss.
        nodes : np.ndarray, shape (8, 3)
            Coordonnées physiques des nœuds.

        Returns
        -------
        B : np.ndarray, shape (6, 24)
            Matrice déformation–déplacement.
        det_J : float
            Déterminant du Jacobien (> 0).

        Notes
        -----
        Les dérivées physiques sont obtenues par :
            dN_phys = J⁻¹ · dN_nat    (3×3 @ 3×8 → 3×8)

        Contrairement au Tetra4, B n'est pas constante : elle varie d'un
        point de Gauss à l'autre car les fonctions de forme trilinéaires ont
        des dérivées qui dépendent de (ξ, η, ζ).
        """
        dN = self._shape_function_derivatives(xi, eta, zeta)  # (3, 8)
        J, det_J = self._jacobian(dN, nodes)
        # dN_phys[i, k] = ∂Nk/∂xᵢ  (i=0→x, 1→y, 2→z)
        dN_phys = np.linalg.solve(J, dN)  # (3,3)⁻¹ · (3,8) = (3,8)

        B = np.zeros((6, 24))
        for i in range(8):
            c = 3 * i        # indice de colonne pour ux_i
            dNx = dN_phys[0, i]   # ∂Ni/∂x
            dNy = dN_phys[1, i]   # ∂Ni/∂y
            dNz = dN_phys[2, i]   # ∂Ni/∂z
            B[0, c    ] = dNx          # εxx = ∂ux/∂x
            B[1, c + 1] = dNy          # εyy = ∂uy/∂y
            B[2, c + 2] = dNz          # εzz = ∂uz/∂z
            B[3, c + 1] = dNz          # γyz : ∂uy/∂z
            B[3, c + 2] = dNy          # γyz : ∂uz/∂y
            B[4, c    ] = dNz          # γxz : ∂ux/∂z
            B[4, c + 2] = dNx          # γxz : ∂uz/∂x
            B[5, c    ] = dNy          # γxy : ∂ux/∂y
            B[5, c + 1] = dNx          # γxy : ∂uy/∂x

        return B, det_J

    # ------------------------------------------------------------------
    # Matrices élémentaires
    # ------------------------------------------------------------------

    def stiffness_matrix(
        self,
        material: ElasticMaterial,
        nodes: np.ndarray,
        properties: dict,
    ) -> np.ndarray:
        """Matrice de rigidité élémentaire K_e (24×24).

        Parameters
        ----------
        material : ElasticMaterial
            Matériau (E, nu utilisés pour construire D 6×6).
        nodes : np.ndarray, shape (8, 3)
            Coordonnées des 8 nœuds en repère global.
        properties : dict
            Non utilisé pour Hexa8 (peut être ``{}``).

        Returns
        -------
        K_e : np.ndarray, shape (24, 24)
            Matrice de rigidité symétrique définie positive
            (6 valeurs propres nulles correspondant aux modes rigides).

        Notes
        -----
        Intégration 2×2×2 Gauss :

            K_e = Σₚ wₚ · Bₚᵀ · D · Bₚ · |det Jₚ|

        Pour un hexaèdre régulier (parallélépipède), l'intégration est
        exacte. Pour des éléments distordus, l'erreur est O(distorsion²).

        Examples
        --------
        >>> mat = ElasticMaterial(E=1.0, nu=0.0, rho=1.0)
        >>> nodes = np.array([
        ...     [0.,0.,0.],[1.,0.,0.],[1.,1.,0.],[0.,1.,0.],
        ...     [0.,0.,1.],[1.,0.,1.],[1.,1.,1.],[0.,1.,1.],
        ... ])
        >>> K_e = Hexa8().stiffness_matrix(mat, nodes, {})
        >>> K_e.shape
        (24, 24)
        """
        if nodes.shape != (8, 3):
            raise ValueError(
                f"Hexa8 attend nodes.shape == (8, 3), reçu {nodes.shape}"
            )

        D = material.elasticity_matrix_3d()
        K_e = np.zeros((24, 24))

        for xi, eta, zeta, w in _GAUSS_POINTS_2X2X2:
            B, det_J = self._strain_displacement_matrix(xi, eta, zeta, nodes)
            K_e += w * (B.T @ D @ B) * det_J

        return K_e

    def mass_matrix(
        self,
        material: ElasticMaterial,
        nodes: np.ndarray,
        properties: dict,
    ) -> np.ndarray:
        """Matrice de masse consistante M_e (24×24) par intégration 2×2×2 Gauss.

        Parameters
        ----------
        material : ElasticMaterial
            Matériau (rho utilisé).
        nodes : np.ndarray, shape (8, 3)
            Coordonnées nodales.
        properties : dict
            Non utilisé pour Hexa8.

        Returns
        -------
        M_e : np.ndarray, shape (24, 24)
            Matrice de masse consistante symétrique.

        Notes
        -----
        M_e = ρ · ∫∫∫ Nᵀ N |det J| dξ dη dζ

        où N est la matrice d'interpolation 3×24 :
            N[0, 3i  ] = Ni,  N[1, 3i+1] = Ni,  N[2, 3i+2] = Ni

        La masse totale est conservée : trace(M_e) / 3 = ρ · V.

        Examples
        --------
        >>> mat = ElasticMaterial(E=1.0, nu=0.0, rho=2500.0)
        >>> nodes = np.array([
        ...     [0.,0.,0.],[1.,0.,0.],[1.,1.,0.],[0.,1.,0.],
        ...     [0.,0.,1.],[1.,0.,1.],[1.,1.,1.],[0.,1.,1.],
        ... ])
        >>> M_e = Hexa8().mass_matrix(mat, nodes, {})
        >>> # Masse totale = ρV = 2500 * 1 = 2500 kg
        >>> float(M_e.sum() / 3)  # doctest: +ELLIPSIS
        2500.0...
        """
        if nodes.shape != (8, 3):
            raise ValueError(
                f"Hexa8 attend nodes.shape == (8, 3), reçu {nodes.shape}"
            )

        M_e = np.zeros((24, 24))

        for xi, eta, zeta, w in _GAUSS_POINTS_2X2X2:
            Nv = self._shape_functions(xi, eta, zeta)   # (8,)
            dN = self._shape_function_derivatives(xi, eta, zeta)
            J, det_J = self._jacobian(dN, nodes)

            # Matrice d'interpolation N (3×24)
            N_mat = np.zeros((3, 24))
            for i in range(8):
                N_mat[0, 3 * i    ] = Nv[i]
                N_mat[1, 3 * i + 1] = Nv[i]
                N_mat[2, 3 * i + 2] = Nv[i]

            M_e += w * (N_mat.T @ N_mat) * det_J

        return M_e * material.rho

    def stiffness_matrix_sri(
        self,
        material: ElasticMaterial,
        nodes: np.ndarray,
        properties: dict,
    ) -> np.ndarray:
        """Matrice de rigidité par intégration réduite sélective (SRI) 24×24.

        Formulation de Hughes (1980) — split dilatation / déviatorique :

        * **D_dev** (partie déviatorique — contraintes à trace nulle + cisaillement) :
          intégrée avec **2×2×2** points de Gauss (ordre complet).
          Contient la réponse 2μ deviatoric + tous les termes de cisaillement μ.

        * **D_vol** (partie volumétrique pure — K_bulk · m·mᵀ) :
          intégrée avec **1 point** au centre (ξ=η=ζ=0), poids = 8.
          Cible le *volumetric locking* des matériaux quasi-incompressibles.

        K_e = K_dev + K_vol

        avec ::

            K_dev = Σ_{2×2×2} wₚ · Bₚᵀ · D_dev · Bₚ · |det Jₚ|   (ordre complet)
            K_vol = 8 · Bc^T · D_vol · Bc · |det Jc|               (1 point central)

        Décomposition de D
        ------------------
        Constante de Lamé bulk : K = λ + 2μ/3 = E / (3(1−2ν))

            D_vol = K · m · mᵀ          m = [1,1,1,0,0,0]ᵀ
            D_dev = D − D_vol

        D = D_dev + D_vol (partition exacte).

        Absence de modes hourglass
        --------------------------
        K_dev est calculé en 2×2×2 complet : il a rang 18 seul.
        L'ajout de K_vol (rang 1) ne peut qu'augmenter le rang → pas de modes
        zéro-énergie introduits par la réduction.

        Contraste avec Quad4
        ---------------------
        Pour Quad4, le split est membrane (2×2) / cisaillement (1 pt) et cible
        le *shear locking* en flexion.  Pour Hexa8, le split dilatation/déviatorique
        de Hughes cible le *volumetric locking*.  Le shear locking en flexion 3D
        nécessite des méthodes avancées (ANS/EAS).

        Parameters
        ----------
        material : ElasticMaterial
            Matériau (E, nu).
        nodes : np.ndarray, shape (8, 3)
            Coordonnées nodales.
        properties : dict
            Non utilisé pour Hexa8 (peut être ``{}``).

        Returns
        -------
        K_e : np.ndarray, shape (24, 24)
            Matrice de rigidité SRI symétrique.

        References
        ----------
        Hughes, T.J.R. (1980). Generalization of selective integration procedures
        to anisotropic and nonlinear media. *Int. J. Num. Meth. Engng*, 15, 1413–1418.

        Examples
        --------
        >>> import numpy as np
        >>> from femsolver.core.material import ElasticMaterial
        >>> mat = ElasticMaterial(E=210e9, nu=0.3, rho=7800)
        >>> nodes = np.array([
        ...     [0.,0.,0.],[1.,0.,0.],[1.,1.,0.],[0.,1.,0.],
        ...     [0.,0.,1.],[1.,0.,1.],[1.,1.,1.],[0.,1.,1.],
        ... ])
        >>> K_sri = Hexa8().stiffness_matrix_sri(mat, nodes, {})
        >>> K_sri.shape
        (24, 24)
        """
        if nodes.shape != (8, 3):
            raise ValueError(
                f"Hexa8 attend nodes.shape == (8, 3), reçu {nodes.shape}"
            )

        D = material.elasticity_matrix_3d()

        # --- Décomposition de D selon Hughes (1980) -----------------------
        # K_bulk = λ + 2μ/3 = E / (3(1-2ν))  [module de compressibilité 3D]
        E, nu = material.E, material.nu
        K_bulk = E / (3.0 * (1.0 - 2.0 * nu))

        # D_vol = K_bulk · m · mᵀ  où  m = [1,1,1,0,0,0]ᵀ
        # Seul le bloc 3×3 supérieur gauche est non nul (tous les termes = K_bulk)
        D_vol = np.zeros((6, 6))
        D_vol[:3, :3] = K_bulk   # m·mᵀ top-left = ones(3,3) × K_bulk

        # D_dev = D − D_vol  (partie déviatorique + cisaillement)
        D_dev = D - D_vol

        # --- Partie déviatorique : intégration 2×2×2 (ordre complet) -----
        K_dev = np.zeros((24, 24))
        for xi, eta, zeta, w in _GAUSS_POINTS_2X2X2:
            B, det_J = self._strain_displacement_matrix(xi, eta, zeta, nodes)
            K_dev += w * (B.T @ D_dev @ B) * det_J

        # --- Partie volumétrique : 1 point au centre (poids = 8) ---------
        B_c, det_J_c = self._strain_displacement_matrix(0.0, 0.0, 0.0, nodes)
        K_vol = 8.0 * (B_c.T @ D_vol @ B_c) * det_J_c

        return K_dev + K_vol

    def body_force_vector(
        self,
        material: ElasticMaterial,
        nodes: np.ndarray,
        properties: dict,
        b: np.ndarray,
    ) -> np.ndarray:
        """Forces nodales équivalentes pour une force de volume uniforme b [N/m³].

        f_e = ρ · ∫∫∫ Nᵀ · b |det J| dξ dη dζ   (intégration 2×2×2 Gauss)

        Parameters
        ----------
        material : ElasticMaterial
            Matériau (rho utilisé).
        nodes : np.ndarray, shape (8, 3)
            Coordonnées nodales.
        properties : dict
            Non utilisé pour Hexa8.
        b : np.ndarray, shape (3,)
            Force de volume [N/m³] = ρ · acceleration.

        Returns
        -------
        f_e : np.ndarray, shape (24,)
            Forces nodales équivalentes [N].

        Notes
        -----
        Pour un hexaèdre régulier (parallélépipède), chaque nœud reçoit
        exactement ρ·V/8 · b (équipartition — vérifiable analytiquement).
        """
        if nodes.shape != (8, 3):
            raise ValueError(
                f"Hexa8 attend nodes.shape == (8, 3), reçu {nodes.shape}"
            )
        f_e = np.zeros(24)
        for xi, eta, zeta, w in _GAUSS_POINTS_2X2X2:
            Nv = self._shape_functions(xi, eta, zeta)   # (8,)
            dN = self._shape_function_derivatives(xi, eta, zeta)
            _, det_J = self._jacobian(dN, nodes)
            # np.kron(Nv, b) → [N0*bx, N0*by, N0*bz, N1*bx, …]
            f_e += w * np.kron(Nv, b) * det_J
        return f_e * material.rho

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
        """Vecteur de déformations ε = B · u_e au point (ξ, η, ζ).

        Parameters
        ----------
        nodes : np.ndarray, shape (8, 3)
            Coordonnées nodales.
        u_e : np.ndarray, shape (24,)
            Déplacements élémentaires [u0,v0,w0, u1,v1,w1, …].
        xi, eta, zeta : float
            Point d'évaluation en coordonnées naturelles (défaut : centre).

        Returns
        -------
        epsilon : np.ndarray, shape (6,)
            [εxx, εyy, εzz, γyz, γxz, γxy].
        """
        B, _ = self._strain_displacement_matrix(xi, eta, zeta, nodes)
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
        """Vecteur de contraintes σ = D · B · u_e au point (ξ, η, ζ).

        Parameters
        ----------
        material : ElasticMaterial
            Matériau (E, nu).
        nodes : np.ndarray, shape (8, 3)
            Coordonnées nodales.
        u_e : np.ndarray, shape (24,)
            Déplacements élémentaires.
        xi, eta, zeta : float
            Point d'évaluation (défaut : centre de l'élément).

        Returns
        -------
        sigma : np.ndarray, shape (6,)
            [σxx, σyy, σzz, τyz, τxz, τxy] [Pa].
        """
        D = material.elasticity_matrix_3d()
        return D @ self.strain(nodes, u_e, xi, eta, zeta)

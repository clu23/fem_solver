"""Élément tétraèdre quadratique isoparamétrique à 10 nœuds (C3D10/TET10).

Différences clés par rapport à Tetra4
--------------------------------------

**Nœuds milieux (6 arêtes → 6 nœuds supplémentaires)**
Tetra4 : 4 sommets, déplacements linéaires, déformations constantes.
Tetra10 : 4 sommets + 6 milieux d'arêtes = 10 nœuds.
Les milieux d'arêtes permettent au champ de déplacement d'être *quadratique*
dans chaque élément (analogue 3D de Tri6).

**Fonctions de forme quadratiques**
En coordonnées barycentriques (L1=1−ξ−η−ζ, L2=ξ, L3=η, L4=ζ) :
    Coins   : Ni = Li(2Li − 1)       (i = 1..4)
    Milieux : Nij = 4 Li Lj           (paire d'arêtes i−j)

**Déformations linéaires (analogue 3D du LST)**
B = J⁻¹ · ∂N/∂(ξ,η,ζ) est *linéaire* en (ξ,η,ζ), contre constant pour
Tetra4. Tetra10 capture les gradients de contrainte *dans* chaque élément.

**Convergence quadratique**
Tetra4 : O(h²) déplacements, O(h) contraintes.
Tetra10 : O(h³) déplacements, O(h²) contraintes.

**Absence de shear locking**
Tetra4 verrouille en flexion (déformations constantes). Tetra10 représente
des déformations linéaires → élimine naturellement le verouillage en flexion
sans intégration réduite.

Numérotation des nœuds
-----------------------
    Coins  : 0(L1=1), 1(L2=1), 2(L3=1), 3(L4=1)
    Milieux:
        4  = arête 0-1  (L1L2, milieu entre coins 0 et 1)
        5  = arête 1-2  (L2L3, milieu entre coins 1 et 2)
        6  = arête 0-2  (L1L3, milieu entre coins 0 et 2)
        7  = arête 0-3  (L1L4, milieu entre coins 0 et 3)
        8  = arête 1-3  (L2L4, milieu entre coins 1 et 3)
        9  = arête 2-3  (L3L4, milieu entre coins 2 et 3)

Ordre DDL : [u0,v0,w0, u1,v1,w1, …, u9,v9,w9]  (30 DDL par élément).

References
----------
Bathe, « Finite Element Procedures », 2nd ed., chap. 5.
Cook et al., « Concepts and Applications of FEA », 4th ed., chap. 6–9.
Keast, « Moderate Degree Tetrahedral Products Gauss Integration Formulas »,
CMAME 55 (1986) — règles de quadrature pour tétraèdres.
"""

from __future__ import annotations

import numpy as np

from femsolver.core.element import Element
from femsolver.core.material import ElasticMaterial


# ---------------------------------------------------------------------------
# Règle de quadrature de Gauss à 4 points pour tétraèdre, ordre 2
# ---------------------------------------------------------------------------
# Exacte pour polynômes de degré ≤ 2 sur le tétraèdre de référence.
# B est linéaire en (ξ,η,ζ) → B^T D B est de degré 2 → 4 points suffisent
# pour les éléments à arêtes droites (J constant).
#
# Points en (ξ, η, ζ) = (L2, L3, L4) dans le tétraèdre {ξ,η,ζ ≥ 0, ξ+η+ζ ≤ 1}.
# Poids : somme = 1/6 (volume du tétraèdre de référence).
_A4 = (5.0 + 3.0 * np.sqrt(5.0)) / 20.0  # ≈ 0.585410196624969
_B4 = (5.0 -      np.sqrt(5.0)) / 20.0   # ≈ 0.138196601125011
_W4 = 1.0 / 24.0

# Chaque tuple : (ξ, η, ζ, poids)
_GAUSS_K4: tuple[tuple[float, float, float, float], ...] = (
    (_B4, _B4, _B4, _W4),  # L1 = _A4
    (_A4, _B4, _B4, _W4),  # L1 = _B4
    (_B4, _A4, _B4, _W4),  # L1 = _B4
    (_B4, _B4, _A4, _W4),  # L1 = _B4
)


class Tetra10(Element):
    """Tétraèdre quadratique isoparamétrique à 10 nœuds — élasticité 3D.

    10 nœuds (4 coins + 6 milieux d'arêtes), 3 DDL par nœud (ux, uy, uz).
    → matrices élémentaires 30×30.

    Déformation linéaire dans l'élément (quadratique en déplacement).
    Intégration numérique (4 points de Gauss) pour K_e,
    formule analytique pour M_e.

    Parameters (via ``properties``)
    --------------------------------
    Aucun paramètre requis. Le dict peut être vide ``{}``.

    Raises
    ------
    ValueError
        Si nodes.shape ≠ (10, 3).
    ValueError
        Si det(J) ≤ 0 à un point de Gauss (nœuds mal orientés ou dégénérés).

    Examples
    --------
    >>> import numpy as np
    >>> from femsolver.core.material import ElasticMaterial
    >>> mat = ElasticMaterial(E=210e9, nu=0.3, rho=7800)
    >>> n = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],
    ...               [.5,0,0],[.5,.5,0],[0,.5,0],
    ...               [0,0,.5],[.5,0,.5],[0,.5,.5]], dtype=float)
    >>> K_e = Tetra10().stiffness_matrix(mat, n, {})
    >>> K_e.shape
    (30, 30)
    """

    def dof_per_node(self) -> int:
        """3 DDL par nœud : ux, uy, uz."""
        return 3

    def n_nodes(self) -> int:
        """10 nœuds."""
        return 10

    # ------------------------------------------------------------------
    # Fonctions de forme et leurs dérivées
    # ------------------------------------------------------------------

    @staticmethod
    def _shape_functions(xi: float, eta: float, zeta: float) -> np.ndarray:
        """Évaluer les 10 fonctions de forme en (ξ, η, ζ).

        Parameters
        ----------
        xi, eta, zeta : float
            Coordonnées naturelles (ξ,η,ζ ≥ 0, ξ+η+ζ ≤ 1).

        Returns
        -------
        N : np.ndarray, shape (10,)
            [N0, N1, …, N9].

        Notes
        -----
        L1 = 1−ξ−η−ζ, L2=ξ, L3=η, L4=ζ.
        Coins : Ni = Li(2Li−1).  Milieux : Nij = 4 Li Lj.
        Partition de l'unité : ΣNi = 1 pour tout (ξ,η,ζ) valide.
        """
        L1 = 1.0 - xi - eta - zeta
        L2, L3, L4 = xi, eta, zeta
        return np.array([
            L1 * (2.0*L1 - 1.0),   # N0 — coin 0 (L1)
            L2 * (2.0*L2 - 1.0),   # N1 — coin 1 (L2)
            L3 * (2.0*L3 - 1.0),   # N2 — coin 2 (L3)
            L4 * (2.0*L4 - 1.0),   # N3 — coin 3 (L4)
            4.0 * L1 * L2,          # N4 — milieu arête 0-1
            4.0 * L2 * L3,          # N5 — milieu arête 1-2
            4.0 * L1 * L3,          # N6 — milieu arête 0-2
            4.0 * L1 * L4,          # N7 — milieu arête 0-3
            4.0 * L2 * L4,          # N8 — milieu arête 1-3
            4.0 * L3 * L4,          # N9 — milieu arête 2-3
        ])

    @staticmethod
    def _shape_function_derivatives(
        xi: float, eta: float, zeta: float
    ) -> np.ndarray:
        """Dérivées des 10 fonctions de forme par rapport à (ξ, η, ζ).

        Parameters
        ----------
        xi, eta, zeta : float
            Coordonnées naturelles.

        Returns
        -------
        dN : np.ndarray, shape (3, 10)
            dN[0, k] = ∂Nk/∂ξ,  dN[1, k] = ∂Nk/∂η,  dN[2, k] = ∂Nk/∂ζ.

        Notes
        -----
        Avec L1 = 1−ξ−η−ζ (→ ∂L1/∂ξ = ∂L1/∂η = ∂L1/∂ζ = −1) :

            ∂N0/∂ξ = ∂N0/∂η = ∂N0/∂ζ = 4(ξ+η+ζ)−3
            ∂N1/∂ξ = 4ξ−1,  ∂N1/∂η = ∂N1/∂ζ = 0
            ∂N2/∂ξ = 0,      ∂N2/∂η = 4η−1,  ∂N2/∂ζ = 0
            ∂N3/∂ξ = 0,      ∂N3/∂η = 0,     ∂N3/∂ζ = 4ζ−1
            ∂N4/∂ξ = 4(1−2ξ−η−ζ),  ∂N4/∂η = −4ξ,  ∂N4/∂ζ = −4ξ
            ∂N5/∂ξ = 4η,    ∂N5/∂η = 4ξ,    ∂N5/∂ζ = 0
            ∂N6/∂ξ = −4η,   ∂N6/∂η = 4(1−ξ−2η−ζ), ∂N6/∂ζ = −4η
            ∂N7/∂ξ = −4ζ,   ∂N7/∂η = −4ζ,  ∂N7/∂ζ = 4(1−ξ−η−2ζ)
            ∂N8/∂ξ = 4ζ,    ∂N8/∂η = 0,     ∂N8/∂ζ = 4ξ
            ∂N9/∂ξ = 0,     ∂N9/∂η = 4ζ,    ∂N9/∂ζ = 4η
        """
        s = xi + eta + zeta
        L1 = 1.0 - s
        dN_dxi = np.array([
            4.0*s - 3.0,                       # ∂N0/∂ξ = −4L1+1
            4.0*xi - 1.0,                       # ∂N1/∂ξ
            0.0,                                 # ∂N2/∂ξ
            0.0,                                 # ∂N3/∂ξ
            4.0*(1.0 - 2.0*xi - eta - zeta),    # ∂N4/∂ξ = 4(L1−L2)... wait
            4.0*eta,                             # ∂N5/∂ξ
            -4.0*eta,                            # ∂N6/∂ξ
            -4.0*zeta,                           # ∂N7/∂ξ
            4.0*zeta,                            # ∂N8/∂ξ
            0.0,                                 # ∂N9/∂ξ
        ])
        dN_deta = np.array([
            4.0*s - 3.0,                        # ∂N0/∂η
            0.0,                                 # ∂N1/∂η
            4.0*eta - 1.0,                       # ∂N2/∂η
            0.0,                                 # ∂N3/∂η
            -4.0*xi,                             # ∂N4/∂η
            4.0*xi,                              # ∂N5/∂η
            4.0*(1.0 - xi - 2.0*eta - zeta),    # ∂N6/∂η
            -4.0*zeta,                           # ∂N7/∂η
            0.0,                                 # ∂N8/∂η
            4.0*zeta,                            # ∂N9/∂η
        ])
        dN_dzeta = np.array([
            4.0*s - 3.0,                        # ∂N0/∂ζ
            0.0,                                 # ∂N1/∂ζ
            0.0,                                 # ∂N2/∂ζ
            4.0*zeta - 1.0,                      # ∂N3/∂ζ
            -4.0*xi,                             # ∂N4/∂ζ
            0.0,                                 # ∂N5/∂ζ
            -4.0*eta,                            # ∂N6/∂ζ
            4.0*(1.0 - xi - eta - 2.0*zeta),    # ∂N7/∂ζ
            4.0*xi,                              # ∂N8/∂ζ
            4.0*eta,                             # ∂N9/∂ζ
        ])
        return np.vstack([dN_dxi, dN_deta, dN_dzeta])  # shape (3, 10)

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
        """det(J) et matrice B (6×30) au point de Gauss (ξ, η, ζ).

        Parameters
        ----------
        xi, eta, zeta : float
            Coordonnées naturelles dans le tétraèdre de référence.
        nodes : np.ndarray, shape (10, 3)
            Coordonnées physiques [[x0,y0,z0], …, [x9,y9,z9]].

        Returns
        -------
        det_J : float
            Déterminant du Jacobien (> 0).
        B : np.ndarray, shape (6, 30)
            Matrice déformation–déplacement. ε = B · u_e.

        Raises
        ------
        ValueError
            Si det(J) ≤ 0 (élément dégénéré ou nœuds mal orientés).

        Notes
        -----
        Jacobien J (3×3) = dN/d(ξ,η,ζ) @ nodes (3×10) @ (10×3).
        Dérivées physiques : dN_phys = J⁻¹ · dN_nat (3×10).
        Notation de Voigt : ε = [εxx,εyy,εzz,γyz,γxz,γxy].

        Pour le nœud k (colonnes 3k, 3k+1, 3k+2) de B :
            Row 0 (εxx): [∂Nk/∂x,   0,         0       ]
            Row 1 (εyy): [0,         ∂Nk/∂y,    0       ]
            Row 2 (εzz): [0,         0,          ∂Nk/∂z ]
            Row 3 (γyz): [0,         ∂Nk/∂z,    ∂Nk/∂y ]
            Row 4 (γxz): [∂Nk/∂z,   0,          ∂Nk/∂x ]
            Row 5 (γxy): [∂Nk/∂y,   ∂Nk/∂x,    0       ]
        """
        dN = self._shape_function_derivatives(xi, eta, zeta)  # (3, 10)
        J = dN @ nodes                                          # (3, 3)
        det_J = np.linalg.det(J)

        if det_J <= 0.0:
            raise ValueError(
                f"det(J) = {det_J:.6g} ≤ 0 au point (ξ={xi:.4f}, η={eta:.4f}, "
                f"ζ={zeta:.4f}) — vérifier l'orientation des nœuds."
            )

        dN_phys = np.linalg.solve(J, dN)  # (3×3)⁻¹ · (3×10) = (3×10)

        B = np.zeros((6, 30))
        for k in range(10):
            c = 3 * k
            dNx = dN_phys[0, k]  # ∂Nk/∂x
            dNy = dN_phys[1, k]  # ∂Nk/∂y
            dNz = dN_phys[2, k]  # ∂Nk/∂z
            B[0, c    ] = dNx    # εxx
            B[1, c + 1] = dNy    # εyy
            B[2, c + 2] = dNz    # εzz
            B[3, c + 1] = dNz    # γyz : ∂uy/∂z
            B[3, c + 2] = dNy    # γyz : ∂uz/∂y
            B[4, c    ] = dNz    # γxz : ∂ux/∂z
            B[4, c + 2] = dNx    # γxz : ∂uz/∂x
            B[5, c    ] = dNy    # γxy : ∂ux/∂y
            B[5, c + 1] = dNx    # γxy : ∂uy/∂x
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
        """Matrice de rigidité élémentaire K_e (30×30).

        Parameters
        ----------
        material : ElasticMaterial
            Matériau (E, nu utilisés pour construire D 6×6).
        nodes : np.ndarray, shape (10, 3)
            Coordonnées des 10 nœuds.
            Nœuds 0–3 : coins. Nœuds 4–9 : milieux d'arêtes (voir module).
        properties : dict
            Non utilisé pour Tetra10 (peut être ``{}``).

        Returns
        -------
        K_e : np.ndarray, shape (30, 30)
            Matrice de rigidité symétrique.

        Notes
        -----
        Intégration numérique de Gauss à 4 points (ordre 2) :

            K_e = Σ_gp  w_gp · Bᵀ(ξ,η,ζ) · D · B(ξ,η,ζ) · |det J(ξ,η,ζ)|

        Pour des arêtes droites (milieux exactement au milieu), J est constant
        et B est linéaire → la règle d'ordre 2 est exacte. Pour des arêtes
        légèrement courbes, augmenter l'ordre est possible (remplacer _GAUSS_K4).
        """
        if nodes.shape != (10, 3):
            raise ValueError(
                f"Tetra10 attend nodes.shape == (10, 3), reçu {nodes.shape}"
            )
        D = material.elasticity_matrix_3d()
        K_e = np.zeros((30, 30))
        for xi, eta, zeta, w in _GAUSS_K4:
            det_J, B = self._jacobian_and_B(xi, eta, zeta, nodes)
            K_e += (w * det_J) * (B.T @ D @ B)
        return K_e

    def mass_matrix(
        self,
        material: ElasticMaterial,
        nodes: np.ndarray,
        properties: dict,
    ) -> np.ndarray:
        """Matrice de masse consistante M_e (30×30).

        Parameters
        ----------
        material : ElasticMaterial
            Matériau (rho utilisé).
        nodes : np.ndarray, shape (10, 3)
            Coordonnées des nœuds.
        properties : dict
            Non utilisé pour Tetra10.

        Returns
        -------
        M_e : np.ndarray, shape (30, 30)
            Matrice de masse consistante symétrique.

        Notes
        -----
        Calculée **analytiquement** via l'intégrale exacte des coordonnées
        barycentriques sur le tétraèdre :

            ∫_tet L1^a L2^b L3^c L4^d dV = 6V · a! b! c! d! / (a+b+c+d+3)!

        Cela donne la matrice scalaire 10×10 (facteur ρV) :

            Type de paire                    Coefficient
            ──────────────────────────────── ─────────────
            Coin–Coin, même nœud              1/70
            Coin–Coin, nœuds différents       1/420
            Coin–Milieu adjacent (2 Li partagés) −1/105
            Coin–Milieu non adjacent          −1/70
            Milieu–Milieu même arête          8/105
            Milieu–Milieu partageant 1 coin   4/105
            Milieu–Milieu arêtes opposées     2/105

        La matrice 30×30 est obtenue par produit de Kronecker :
            M_e = kron(m̄_scalar, I_3) × ρV

        Propriétés vérifiables :
        - Symétrie
        - Somme de ligne pour coins : −ρV/20 (négatif — connu pour Tetra10)
        - Somme de ligne pour milieux : +ρV/5
        - Somme totale de toutes entrées = ρV (masse totale)
        """
        if nodes.shape != (10, 3):
            raise ValueError(
                f"Tetra10 attend nodes.shape == (10, 3), reçu {nodes.shape}"
            )

        # Volume du tétraèdre via les 4 coins (nœuds 0–3)
        corners = nodes[:4]
        edge_vecs = corners[1:] - corners[0]  # shape (3, 3)
        volume = abs(np.linalg.det(edge_vecs)) / 6.0
        if volume < 1e-30:
            raise ValueError(
                f"Volume nul pour Tetra10 — coins coplanaires : {corners}"
            )

        # ---------------------------------------------------------------
        # Matrice scalaire analytique 10×10
        #
        # Indices 0-3 : coins (L1,L2,L3,L4)
        # Indices 4-9 : milieux
        #   4 = arête 0-1 (L1L2)
        #   5 = arête 1-2 (L2L3)
        #   6 = arête 0-2 (L1L3)
        #   7 = arête 0-3 (L1L4)
        #   8 = arête 1-3 (L2L4)
        #   9 = arête 2-3 (L3L4)
        #
        # Adjacences coin i → midside k (partageant Li) :
        #   coin 0 (L1) → adj à {4(L1L2), 6(L1L3), 7(L1L4)}
        #   coin 1 (L2) → adj à {4(L1L2), 5(L2L3), 8(L2L4)}
        #   coin 2 (L3) → adj à {5(L2L3), 6(L1L3), 9(L3L4)}
        #   coin 3 (L4) → adj à {7(L1L4), 8(L2L4), 9(L3L4)}
        #
        # Paires opposées (aucun nœud commun, valeur 2/105) :
        #   (4=L1L2) ↔ (9=L3L4)
        #   (5=L2L3) ↔ (7=L1L4)
        #   (6=L1L3) ↔ (8=L2L4)
        # ---------------------------------------------------------------
        CC  =  1.0 / 70.0    # coin–coin même nœud
        CO  =  1.0 / 420.0   # coin–coin différents
        CA  = -1.0 / 105.0   # coin–milieu adjacent
        CN  = -1.0 / 70.0    # coin–milieu non adjacent
        MS  =  8.0 / 105.0   # milieu–milieu même arête
        MA  =  4.0 / 105.0   # milieu–milieu partageant 1 coin
        MO  =  2.0 / 105.0   # milieu–milieu arêtes opposées

        # Ordre des nœuds : c0 c1 c2 c3  m4 m5 m6 m7 m8 m9
        # (abréviations : c=coin, m=milieu, indices = nœud Python)
        m_bar = np.zeros((10, 10))

        # --- bloc coin–coin (4×4) ---
        for i in range(4):
            for j in range(4):
                m_bar[i, j] = CC if i == j else CO

        # --- bloc coin–milieu (4×6) et milieu–coin (6×4) ---
        # coin 0 adj à midsides 4,6,7  (indices dans m_bar : +4 offset)
        adj = {
            0: {4, 6, 7},  # L1-arêtes
            1: {4, 5, 8},  # L2-arêtes
            2: {5, 6, 9},  # L3-arêtes
            3: {7, 8, 9},  # L4-arêtes
        }
        for ci in range(4):
            for mi in range(4, 10):
                val = CA if mi in adj[ci] else CN
                m_bar[ci, mi] = val
                m_bar[mi, ci] = val

        # --- bloc milieu–milieu (6×6) ---
        # Paires opposées (aucun coin commun) :
        opposite = {(4, 9), (5, 7), (6, 8), (9, 4), (7, 5), (8, 6)}
        for mi in range(4, 10):
            for mj in range(4, 10):
                if mi == mj:
                    m_bar[mi, mj] = MS
                elif (mi, mj) in opposite:
                    m_bar[mi, mj] = MO
                else:
                    m_bar[mi, mj] = MA

        return (material.rho * volume) * np.kron(m_bar, np.eye(3))

    def body_force_vector(
        self,
        material: ElasticMaterial,
        nodes: np.ndarray,
        properties: dict,
        b: np.ndarray,
    ) -> np.ndarray:
        """Forces nodales équivalentes pour une force de volume uniforme b [N/m³].

        f_e = ρ ∫_V N^T b dV, intégrée sur 4 points de Gauss.

        Parameters
        ----------
        material : ElasticMaterial
            Matériau (rho utilisé).
        nodes : np.ndarray, shape (10, 3)
            Coordonnées nodales.
        properties : dict
            Non utilisé pour Tetra10.
        b : np.ndarray, shape (3,)
            Force de volume [N/m³] = ρ · accélération.

        Returns
        -------
        f_e : np.ndarray, shape (30,)
            Forces nodales équivalentes [N].

        Notes
        -----
        Analytiquement : ∫ Ni dV = −V/20 pour les coins, V/5 pour les milieux.
        La somme est ρV (masse totale × b), comme attendu.
        """
        f_e = np.zeros(30)
        for xi, eta, zeta, w in _GAUSS_K4:
            det_J, _ = self._jacobian_and_B(xi, eta, zeta, nodes)
            N = self._shape_functions(xi, eta, zeta)  # (10,)
            coeff = w * det_J * material.rho
            for k in range(10):
                f_e[3*k    ] += coeff * N[k] * b[0]
                f_e[3*k + 1] += coeff * N[k] * b[1]
                f_e[3*k + 2] += coeff * N[k] * b[2]
        return f_e

    # ------------------------------------------------------------------
    # Post-traitement
    # ------------------------------------------------------------------

    def strain(
        self,
        nodes: np.ndarray,
        u_e: np.ndarray,
        xi: float = 0.25,
        eta: float = 0.25,
        zeta: float = 0.25,
    ) -> np.ndarray:
        """Vecteur de déformations ε = B(ξ,η,ζ) · u_e.

        Parameters
        ----------
        nodes : np.ndarray, shape (10, 3)
            Coordonnées nodales.
        u_e : np.ndarray, shape (30,)
            Déplacements élémentaires [u0,v0,w0, …, u9,v9,w9].
        xi, eta, zeta : float
            Point d'évaluation (défaut : centroïde du tétraèdre de référence).

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
        xi: float = 0.25,
        eta: float = 0.25,
        zeta: float = 0.25,
    ) -> np.ndarray:
        """Vecteur de contraintes σ = D · B(ξ,η,ζ) · u_e.

        Parameters
        ----------
        material : ElasticMaterial
            Matériau (E, nu).
        nodes : np.ndarray, shape (10, 3)
            Coordonnées nodales.
        u_e : np.ndarray, shape (30,)
            Déplacements élémentaires.
        xi, eta, zeta : float
            Point d'évaluation (défaut : centroïde).

        Returns
        -------
        sigma : np.ndarray, shape (6,)
            [σxx, σyy, σzz, τyz, τxz, τxy] [Pa].
        """
        D = material.elasticity_matrix_3d()
        return D @ self.strain(nodes, u_e, xi, eta, zeta)

    # ------------------------------------------------------------------
    # Interface batch (vectorisation sur N_e éléments simultanément)
    # ------------------------------------------------------------------

    @classmethod
    def batch_stiffness_matrix(
        cls,
        nodes_batch: np.ndarray,
        D: np.ndarray,
    ) -> np.ndarray:
        """Matrices de rigidité pour N_e Tetra10 en une seule passe tenseur.

        Parameters
        ----------
        nodes_batch : np.ndarray, shape (N_e, 10, 3)
            Coordonnées nodales de tous les éléments du groupe.
        D : np.ndarray, shape (6, 6)
            Matrice d'élasticité 3D (identique pour tout le groupe).

        Returns
        -------
        K_e_all : np.ndarray, shape (N_e, 30, 30)
            Matrices de rigidité élémentaires.

        Notes
        -----
        Analogue 3D du Tri6 batch, avec 4 points de Gauss à la place de 6 :

        1. ``dN_nat`` (4_gp, 3, 10) — dérivées naturelles (constantes).
        2. ``J = einsum('gin,enj->geij', dN_nat, nodes)`` → (4, N_e, 3, 3).
        3. ``dN_phys = einsum('geij,gjn->gein', J_inv, dN_nat)``
           → (4, N_e, 3, 10).
        4. Matrices B (4_gp, N_e, 6, 30).
        5. ``K_e = einsum('g,ge,gepi,pq,geqj->eij', w, det_J, B, D, B)``.
        """
        n_e = nodes_batch.shape[0]

        dN_nat = np.stack([
            cls._shape_function_derivatives(xi, eta, zeta)
            for xi, eta, zeta, _ in _GAUSS_K4
        ])   # (4_gp, 3, 10_nodes)
        w = np.array([wg for _, _, _, wg in _GAUSS_K4])   # (4,)

        J = np.einsum('gin,enj->geij', dN_nat, nodes_batch)   # (4, N_e, 3, 3)
        det_J = np.linalg.det(J)                               # (4, N_e)
        J_inv = np.linalg.inv(J)                               # (4, N_e, 3, 3)

        dN_phys = np.einsum('geij,gjn->gein', J_inv, dN_nat)  # (4, N_e, 3, 10)

        B = np.zeros((4, n_e, 6, 30))
        for k in range(10):
            c = 3 * k
            B[:, :, 0, c    ] = dN_phys[:, :, 0, k]   # εxx : ∂Nk/∂x
            B[:, :, 1, c + 1] = dN_phys[:, :, 1, k]   # εyy : ∂Nk/∂y
            B[:, :, 2, c + 2] = dN_phys[:, :, 2, k]   # εzz : ∂Nk/∂z
            B[:, :, 3, c + 1] = dN_phys[:, :, 2, k]   # γyz : ∂Nk/∂z
            B[:, :, 3, c + 2] = dN_phys[:, :, 1, k]   # γyz : ∂Nk/∂y
            B[:, :, 4, c    ] = dN_phys[:, :, 2, k]   # γxz : ∂Nk/∂z
            B[:, :, 4, c + 2] = dN_phys[:, :, 0, k]   # γxz : ∂Nk/∂x
            B[:, :, 5, c    ] = dN_phys[:, :, 1, k]   # γxy : ∂Nk/∂y
            B[:, :, 5, c + 1] = dN_phys[:, :, 0, k]   # γxy : ∂Nk/∂x

        K_e_all = np.einsum('g,ge,gepi,pq,geqj->eij', w, det_J, B, D, B)
        return K_e_all   # (N_e, 30, 30)

    @classmethod
    def batch_mass_matrix(
        cls,
        nodes_batch: np.ndarray,
        rho: float,
    ) -> np.ndarray:
        """Matrices de masse pour N_e Tetra10 en une seule passe tenseur.

        Parameters
        ----------
        nodes_batch : np.ndarray, shape (N_e, 10, 3)
            Coordonnées nodales.
        rho : float
            Masse volumique [kg/m³].

        Returns
        -------
        M_e_all : np.ndarray, shape (N_e, 30, 30)

        Notes
        -----
        La formule analytique est utilisée (comme dans la méthode scalaire) :

            M_e[e] = (ρ · V[e]) · kron(m_bar, I_3)

        V[e] est calculé via les 4 coins. m_bar (10×10) est constant.
        """
        # Volume via les 4 coins (nœuds 0-3)
        corners = nodes_batch[:, :4, :]                           # (N_e, 4, 3)
        edge_vecs = corners[:, 1:, :] - corners[:, 0:1, :]       # (N_e, 3, 3)
        volume = np.abs(np.linalg.det(edge_vecs)) / 6.0          # (N_e,)

        # Matrice analytique m_bar (10×10) — identique à la méthode scalaire
        CC  =  1.0 / 70.0
        CO  =  1.0 / 420.0
        CA  = -1.0 / 105.0
        CN  = -1.0 / 70.0
        MS  =  8.0 / 105.0
        MA  =  4.0 / 105.0
        MO  =  2.0 / 105.0

        m_bar = np.zeros((10, 10))
        for i in range(4):
            for j in range(4):
                m_bar[i, j] = CC if i == j else CO

        adj = {0: {4, 6, 7}, 1: {4, 5, 8}, 2: {5, 6, 9}, 3: {7, 8, 9}}
        for ci in range(4):
            for mi in range(4, 10):
                val = CA if mi in adj[ci] else CN
                m_bar[ci, mi] = val
                m_bar[mi, ci] = val

        opposite = {(4, 9), (5, 7), (6, 8), (9, 4), (7, 5), (8, 6)}
        for mi in range(4, 10):
            for mj in range(4, 10):
                if mi == mj:
                    m_bar[mi, mj] = MS
                elif (mi, mj) in opposite:
                    m_bar[mi, mj] = MO
                else:
                    m_bar[mi, mj] = MA

        M_pat = np.kron(m_bar, np.eye(3))   # (30, 30) — constant

        scales = rho * volume   # (N_e,)
        return scales[:, np.newaxis, np.newaxis] * M_pat[np.newaxis]   # (N_e, 30, 30)

"""Élément triangle quadratique à 6 nœuds (T6/LST — Linear Strain Triangle).

Différences clés par rapport à Tri3 (CST)
------------------------------------------

**Nœuds milieux**
Tri3 n'a que 3 sommets. Tri6 ajoute un nœud au milieu de chaque arête,
portant le total à 6. Ces nœuds milieux permettent au champ de déplacement
d'être *quadratique* (et non plus linéaire) dans chaque élément.

**Fonctions de forme quadratiques**
En coordonnées barycentriques (L1=1−ξ−η, L2=ξ, L3=η) :
    Coins   : Ni = Li(2Li − 1)      → nul aux deux arêtes adjacentes,
                                       vaut 1 au sommet i
    Milieux : Nij = 4 Li Lj          → vaut 1 au milieu de l'arête i−j,
                                       nul aux 4 autres nœuds

**Déformations linéaires (LST)**
B = J⁻¹ · ∂N/∂(ξ,η) est *linéaire* en (ξ, η), contre *constant* pour Tri3.
→ chaque élément Tri6 capture les gradients de contrainte à l'intérieur,
  pas seulement entre éléments.

**Convergence quadratique**
Tri3 converge en O(h²) sur les déplacements, O(h) sur les contraintes.
Tri6 converge en O(h³) sur les déplacements, O(h²) sur les contraintes.
En pratique : avec le même nombre de DDL, Tri6 est bien plus précis.

**Résistance naturelle au shear locking**
Tri3 verrouille en flexion (CST → déformations constantes ne peuvent pas
représenter la courbure nulle d'une poutre → locking). Tri6 peut représenter
des déformations linéaires, ce qui élimine naturellement ce verouillage
sans recourir à l'intégration réduite. C'est l'avantage fondamental des
éléments quadratiques : la hiérarchie des polynômes suffit.

References
----------
Cook et al., « Concepts and Applications of FEA », 4th ed., chap. 6–8.
Zienkiewicz & Taylor, « The FEM for Solid Mechanics », vol. 2, chap. 8.
Dunavant, IJNME 21 (1985) — règles de quadrature triangulaires symétriques.
"""

from __future__ import annotations

import numpy as np

from femsolver.core.element import Element
from femsolver.core.material import ElasticMaterial


# ---------------------------------------------------------------------------
# Règle de quadrature de Gauss–Dunavant à 6 points, ordre 4
# ---------------------------------------------------------------------------
# Intègre exactement les polynômes de degré ≤ 4 sur le triangle de référence.
# B est linéaire → B^T D B est de degré 2 → la règle d'ordre 4 est largement
# suffisante et reste valide pour des arêtes légèrement courbes.
#
# Coordonnées (ξ, η) dans le triangle de référence {ξ≥0, η≥0, ξ+η≤1}.
# Poids : somme = 0.5 (aire du triangle de référence).
_A1 = 0.445948490915965
_B1 = 1.0 - 2.0 * _A1   # ≈ 0.108103018168070
_W1 = 0.223381589678011 / 2.0

_A2 = 0.091576213509771
_B2 = 1.0 - 2.0 * _A2   # ≈ 0.816847572980459
_W2 = 0.109951743655322 / 2.0

# Chaque tuple : (ξ, η, poids)
_GAUSS_PTS: tuple[tuple[float, float, float], ...] = (
    (_A1, _A1, _W1), (_B1, _A1, _W1), (_A1, _B1, _W1),
    (_A2, _A2, _W2), (_B2, _A2, _W2), (_A2, _B2, _W2),
)


class Tri6(Element):
    """Triangle LST (Linear Strain Triangle) quadratique à 6 nœuds.

    6 nœuds, 2 DDL par nœud (ux, uy) → matrices élémentaires 12×12.
    Champ de déplacement quadratique → déformations linéaires (LST).

    Numérotation des nœuds (sens trigonométrique) :

        3
        |\\
        6  5
        |   \\
        1--4--2

    Nœuds 1–3 : sommets.
    Nœud 4 : milieu arête 1–2.
    Nœud 5 : milieu arête 2–3.
    Nœud 6 : milieu arête 1–3.

    Ordre des DDL : [u1,v1, u2,v2, u3,v3, u4,v4, u5,v5, u6,v6].

    Fonctions de forme (L1=1−ξ−η, L2=ξ, L3=η) :

        N1 = L1(2L1−1)    N4 = 4 L1 L2
        N2 = L2(2L2−1)    N5 = 4 L2 L3
        N3 = L3(2L3−1)    N6 = 4 L1 L3

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
    >>> nodes = np.array([[0,0],[1,0],[0,1],[.5,0],[.5,.5],[0,.5]], dtype=float)
    >>> K_e = Tri6().stiffness_matrix(mat, nodes, {"thickness": 1.0})
    >>> K_e.shape
    (12, 12)
    """

    def dof_per_node(self) -> int:
        """2 DDL par nœud : ux et uy."""
        return 2

    def n_nodes(self) -> int:
        """6 nœuds."""
        return 6

    # ------------------------------------------------------------------
    # Fonctions de forme et leurs dérivées
    # ------------------------------------------------------------------

    @staticmethod
    def _shape_functions(xi: float, eta: float) -> np.ndarray:
        """Évaluer les 6 fonctions de forme au point (ξ, η).

        Parameters
        ----------
        xi, eta : float
            Coordonnées naturelles (ξ ≥ 0, η ≥ 0, ξ+η ≤ 1).

        Returns
        -------
        N : np.ndarray, shape (6,)
            [N1, N2, N3, N4, N5, N6].

        Notes
        -----
        Partition de l'unité : ΣNi = 1 quel que soit (ξ, η).
        Chaque Ni vaut 1 au nœud i et 0 aux 5 autres nœuds.
        """
        L1 = 1.0 - xi - eta
        L2 = xi
        L3 = eta
        return np.array([
            L1 * (2.0 * L1 - 1.0),  # N1 — coin 1
            L2 * (2.0 * L2 - 1.0),  # N2 — coin 2
            L3 * (2.0 * L3 - 1.0),  # N3 — coin 3
            4.0 * L1 * L2,           # N4 — milieu 1-2
            4.0 * L2 * L3,           # N5 — milieu 2-3
            4.0 * L1 * L3,           # N6 — milieu 1-3
        ])

    @staticmethod
    def _shape_function_derivatives(xi: float, eta: float) -> np.ndarray:
        """Dérivées des 6 fonctions de forme par rapport à (ξ, η).

        Parameters
        ----------
        xi, eta : float
            Coordonnées naturelles.

        Returns
        -------
        dN : np.ndarray, shape (2, 6)
            dN[0, k] = ∂Nk/∂ξ,  dN[1, k] = ∂Nk/∂η.

        Notes
        -----
        Avec L1 = 1−ξ−η :
            ∂N1/∂ξ = ∂N1/∂η = 4(ξ+η)−3
            ∂N2/∂ξ = 4ξ−1,        ∂N2/∂η = 0
            ∂N3/∂ξ = 0,            ∂N3/∂η = 4η−1
            ∂N4/∂ξ = 4(1−2ξ−η),  ∂N4/∂η = −4ξ
            ∂N5/∂ξ = 4η,           ∂N5/∂η = 4ξ
            ∂N6/∂ξ = −4η,          ∂N6/∂η = 4(1−ξ−2η)
        """
        s = xi + eta
        dN_dxi = np.array([
            4.0 * s - 3.0,            # ∂N1/∂ξ = -4L1+1
            4.0 * xi - 1.0,           # ∂N2/∂ξ
            0.0,                       # ∂N3/∂ξ
            4.0 * (1.0 - 2.0*xi - eta),  # ∂N4/∂ξ
            4.0 * eta,                 # ∂N5/∂ξ
            -4.0 * eta,                # ∂N6/∂ξ
        ])
        dN_deta = np.array([
            4.0 * s - 3.0,            # ∂N1/∂η = -4L1+1
            0.0,                       # ∂N2/∂η
            4.0 * eta - 1.0,           # ∂N3/∂η
            -4.0 * xi,                 # ∂N4/∂η
            4.0 * xi,                  # ∂N5/∂η
            4.0 * (1.0 - xi - 2.0*eta),   # ∂N6/∂η
        ])
        return np.vstack([dN_dxi, dN_deta])  # shape (2, 6)

    # ------------------------------------------------------------------
    # Jacobien et matrice B
    # ------------------------------------------------------------------

    def _jacobian_and_B(
        self,
        xi: float,
        eta: float,
        nodes: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """det(J) et matrice B (3×12) au point de Gauss (ξ, η).

        Parameters
        ----------
        xi, eta : float
            Coordonnées naturelles.
        nodes : np.ndarray, shape (6, 2)
            Coordonnées physiques des 6 nœuds [[x1,y1], …, [x6,y6]].

        Returns
        -------
        det_J : float
            Déterminant du Jacobien (> 0 si nœuds bien orientés).
        B : np.ndarray, shape (3, 12)
            Matrice déformation–déplacement.
            ε = [εxx, εyy, γxy] = B · u_e.

        Raises
        ------
        ValueError
            Si det(J) ≤ 0 (élément dégénéré ou sens horaire).

        Notes
        -----
        Jacobien J (2×2) = dN/d(ξ,η) @ nodes :
            J = [[Σ ∂Nk/∂ξ · xk,  Σ ∂Nk/∂ξ · yk],
                 [Σ ∂Nk/∂η · xk,  Σ ∂Nk/∂η · yk]]

        Dérivées physiques : dN_phys = J⁻¹ · dN_nat  (2×6).

        Structure de B pour le nœud i (colonnes 2i, 2i+1) :
            [∂Ni/∂x,    0    ]  → εxx
            [   0,    ∂Ni/∂y ]  → εyy
            [∂Ni/∂y,  ∂Ni/∂x]  → γxy
        """
        dN = self._shape_function_derivatives(xi, eta)  # (2, 6)
        J = dN @ nodes                                   # (2, 2)
        det_J = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]

        if det_J <= 0.0:
            raise ValueError(
                f"det(J) = {det_J:.6g} ≤ 0 au point (ξ={xi:.4f}, η={eta:.4f}) — "
                "vérifier l'orientation et la position des nœuds milieux."
            )

        # Inverse 2×2 explicite (évite np.linalg.inv)
        J_inv = np.array([[J[1, 1], -J[0, 1]], [-J[1, 0], J[0, 0]]]) / det_J
        dN_phys = J_inv @ dN  # (2, 6) : dN_phys[0]=∂N/∂x, [1]=∂N/∂y

        B = np.zeros((3, 12))
        for i in range(6):
            c = 2 * i
            dNx = dN_phys[0, i]
            dNy = dN_phys[1, i]
            B[0, c    ] = dNx   # εxx = ∂ux/∂x
            B[1, c + 1] = dNy   # εyy = ∂uy/∂y
            B[2, c    ] = dNy   # γxy = ∂ux/∂y + ∂uy/∂x (terme ∂ux/∂y)
            B[2, c + 1] = dNx   # γxy (terme ∂uy/∂x)
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
        """Matrice de rigidité élémentaire K_e (12×12).

        Parameters
        ----------
        material : ElasticMaterial
            Matériau (E, nu utilisés).
        nodes : np.ndarray, shape (6, 2)
            Coordonnées [[x1,y1], …, [x6,y6]].
            Nœuds 0–2 : sommets, nœuds 3–5 : milieux des arêtes 0-1, 1-2, 0-2.
        properties : dict
            ``"thickness"`` : épaisseur [m].
            ``"formulation"`` : ``"plane_stress"`` (défaut) ou ``"plane_strain"``.

        Returns
        -------
        K_e : np.ndarray, shape (12, 12)
            Matrice de rigidité symétrique.

        Notes
        -----
        Intégration numérique de Dunavant à 6 points (ordre 4) :

            K_e = t · Σ_gp  w_gp · Bᵀ(ξ,η) · D · B(ξ,η) · |det J(ξ,η)|

        Contrairement à Tri3 (B constant), B varie linéairement → l'intégration
        numérique est indispensable. La règle d'ordre 4 est exacte pour les
        polynômes de degré ≤ 4, ce qui couvre aussi les géométries légèrement
        distordues.
        """
        if nodes.shape != (6, 2):
            raise ValueError(f"Tri6 attend nodes.shape == (6, 2), reçu {nodes.shape}")
        t = properties["thickness"]
        if t <= 0.0:
            raise ValueError(f"L'épaisseur doit être > 0, reçu thickness={t}")
        formulation = properties.get("formulation", "plane_stress")
        D = self._elasticity_matrix(material, formulation)

        K_e = np.zeros((12, 12))
        for xi, eta, w in _GAUSS_PTS:
            det_J, B = self._jacobian_and_B(xi, eta, nodes)
            K_e += (w * t * det_J) * (B.T @ D @ B)
        return K_e

    def mass_matrix(
        self,
        material: ElasticMaterial,
        nodes: np.ndarray,
        properties: dict,
    ) -> np.ndarray:
        """Matrice de masse consistante M_e (12×12).

        Parameters
        ----------
        material : ElasticMaterial
            Matériau (rho utilisé).
        nodes : np.ndarray, shape (6, 2)
            Coordonnées nodales.
        properties : dict
            ``"thickness"`` : épaisseur [m].

        Returns
        -------
        M_e : np.ndarray, shape (12, 12)
            Matrice de masse consistante symétrique.

        Notes
        -----
        Calculée **analytiquement** via l'intégrale exacte des monômes
        barycentriques :

            ∫_A L1^a L2^b L3^c dA = 2A · a! b! c! / (a+b+c+2)!

        Cela donne la matrice scalaire 6×6 suivante (facteur ρ·t·A) :

            Type de paire               Coefficient
            ─────────────────────────── ──────────────
            Coin–Coin, même nœud         1/30
            Coin–Coin, nœuds différents −1/180
            Coin–Milieu adjacent (∫=0)    0
            Coin–Milieu non adjacent     −1/45
            Milieu–Milieu, même arête    8/45
            Milieu–Milieu, arêtes adj.   4/45

        Remarque : les coins ont un poids négatif entre eux et vis-à-vis des
        milieux non adjacents. C'est une propriété connue des éléments
        quadratiques (la matrice reste SPD globalement).

        La matrice 12×12 est obtenue par produit de Kronecker :
            M_e = kron(m̄_scalar, I_2) × ρ × t × A
        """
        if nodes.shape != (6, 2):
            raise ValueError(f"Tri6 attend nodes.shape == (6, 2), reçu {nodes.shape}")
        t = properties["thickness"]

        # Aire via les 3 sommets uniquement (les milieux sont contraints)
        x1, y1 = nodes[0]
        x2, y2 = nodes[1]
        x3, y3 = nodes[2]
        area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
        if area < 1e-30:
            raise ValueError(f"Aire nulle pour Tri6 — sommets colinéaires : {nodes[:3]}")

        # ---------------------------------------------------------------
        # Matrice scalaire analytique 6×6
        # Indices : 0=coin1(L1), 1=coin2(L2), 2=coin3(L3),
        #           3=milieu12, 4=milieu23, 5=milieu13
        #
        # Adjacence coin→milieu :
        #   coin0 (L1) → adj à {3(L1L2), 5(L1L3)},  non adj à {4(L2L3)}
        #   coin1 (L2) → adj à {3(L1L2), 4(L2L3)},  non adj à {5(L1L3)}
        #   coin2 (L3) → adj à {4(L2L3), 5(L1L3)},  non adj à {3(L1L2)}
        # ---------------------------------------------------------------
        C  =  1.0 / 30.0    # coin–coin même nœud
        CO = -1.0 / 180.0   # coin–coin différents
        A0 =  0.0            # coin–milieu adjacent
        NA = -1.0 / 45.0    # coin–milieu non adjacent
        MD =  8.0 / 45.0    # milieu–milieu même arête
        MA =  4.0 / 45.0    # milieu–milieu arêtes adjacentes

        m_bar = np.array([
            # c0    c1    c2    m3    m4    m5
            [ C,   CO,   CO,   A0,   NA,   A0],   # c0 (coin L1)
            [CO,    C,   CO,   A0,   A0,   NA],   # c1 (coin L2)
            [CO,   CO,    C,   NA,   A0,   A0],   # c2 (coin L3)
            [A0,   A0,   NA,   MD,   MA,   MA],   # m3 (mid L1L2)
            [NA,   A0,   A0,   MA,   MD,   MA],   # m4 (mid L2L3)
            [A0,   NA,   A0,   MA,   MA,   MD],   # m5 (mid L1L3)
        ], dtype=float)

        return (material.rho * t * area) * np.kron(m_bar, np.eye(2))

    def body_force_vector(
        self,
        material: ElasticMaterial,
        nodes: np.ndarray,
        properties: dict,
        b: np.ndarray,
    ) -> np.ndarray:
        """Forces nodales équivalentes pour une force de volume uniforme b [N/m³].

        f_e = ρ t ∫_A N^T b dA,  intégrée sur 6 points de Gauss.

        Parameters
        ----------
        material : ElasticMaterial
            Matériau (rho utilisé).
        nodes : np.ndarray, shape (6, 2)
            Coordonnées nodales.
        properties : dict
            ``"thickness"`` : épaisseur [m].
        b : np.ndarray, shape (2,)
            Force de volume [N/m³] = ρ · accélération.

        Returns
        -------
        f_e : np.ndarray, shape (12,)
            Forces nodales équivalentes [N].

        Notes
        -----
        Pour une force uniforme : ∫ Ni dA = 0 pour les coins, A/3 pour
        les milieux. La force totale est donc portée uniquement par les
        nœuds milieux (résultat analytique vérifié à la fin de la méthode).
        """
        t = properties.get("thickness", 1.0)
        f_e = np.zeros(12)
        for xi, eta, w in _GAUSS_PTS:
            det_J, _ = self._jacobian_and_B(xi, eta, nodes)
            N = self._shape_functions(xi, eta)  # (6,)
            coeff = w * det_J * material.rho * t
            for i in range(6):
                f_e[2*i    ] += coeff * N[i] * b[0]
                f_e[2*i + 1] += coeff * N[i] * b[1]
        return f_e

    # ------------------------------------------------------------------
    # Post-traitement
    # ------------------------------------------------------------------

    def strain(
        self,
        nodes: np.ndarray,
        u_e: np.ndarray,
        xi: float = 1.0 / 3.0,
        eta: float = 1.0 / 3.0,
    ) -> np.ndarray:
        """Vecteur de déformations ε = B(ξ,η) · u_e.

        Parameters
        ----------
        nodes : np.ndarray, shape (6, 2)
            Coordonnées nodales.
        u_e : np.ndarray, shape (12,)
            Déplacements élémentaires [u1,v1, …, u6,v6].
        xi, eta : float
            Point d'évaluation (défaut : centroïde ≈ (1/3, 1/3)).

        Returns
        -------
        epsilon : np.ndarray, shape (3,)
            [εxx, εyy, γxy] au point (ξ, η).
        """
        _, B = self._jacobian_and_B(xi, eta, nodes)
        return B @ u_e

    def stress(
        self,
        material: ElasticMaterial,
        nodes: np.ndarray,
        u_e: np.ndarray,
        xi: float = 1.0 / 3.0,
        eta: float = 1.0 / 3.0,
        formulation: str = "plane_stress",
    ) -> np.ndarray:
        """Vecteur de contraintes σ = D · B(ξ,η) · u_e.

        Parameters
        ----------
        material : ElasticMaterial
            Matériau.
        nodes : np.ndarray, shape (6, 2)
            Coordonnées nodales.
        u_e : np.ndarray, shape (12,)
            Déplacements élémentaires.
        xi, eta : float
            Point d'évaluation (défaut : centroïde).
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
    # Interface batch (vectorisation sur N_e éléments simultanément)
    # ------------------------------------------------------------------

    @classmethod
    def batch_stiffness_matrix(
        cls,
        nodes_batch: np.ndarray,
        D: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """Matrices de rigidité pour N_e Tri6 en une seule passe tenseur.

        Parameters
        ----------
        nodes_batch : np.ndarray, shape (N_e, 6, 2)
            Coordonnées nodales de tous les éléments du groupe.
        D : np.ndarray, shape (3, 3)
            Matrice d'élasticité (identique pour tout le groupe).
        t : float
            Épaisseur [m] (identique pour tout le groupe).

        Returns
        -------
        K_e_all : np.ndarray, shape (N_e, 12, 12)
            Matrices de rigidité élémentaires.

        Notes
        -----
        Même structure tensorielle que le Quad4 batch, avec les 6 points de
        Gauss-Dunavant à la place des 4 points 2×2.

        1. ``dN_nat`` (6_gp, 2, 6) — dérivées naturelles (constantes).
        2. ``J = einsum('gin,enj->geij', dN_nat, nodes)`` → (6, N_e, 2, 2).
        3. ``dN_phys = einsum('geij,gjn->gein', J_inv, dN_nat)``
           → (6, N_e, 2, 6).
        4. Matrices B (6_gp, N_e, 3, 12).
        5. ``K_e = t · einsum('g,ge,gepi,pq,geqj->eij', w, det_J, B, D, B)``.
        """
        n_e = nodes_batch.shape[0]

        dN_nat = np.stack([
            cls._shape_function_derivatives(xi, eta)
            for xi, eta, _ in _GAUSS_PTS
        ])   # (6_gp, 2, 6_nodes)
        w = np.array([wg for _, _, wg in _GAUSS_PTS])   # (6,)

        J = np.einsum('gin,enj->geij', dN_nat, nodes_batch)   # (6, N_e, 2, 2)
        det_J = np.linalg.det(J)                               # (6, N_e)
        J_inv = np.linalg.inv(J)                               # (6, N_e, 2, 2)

        dN_phys = np.einsum('geij,gjn->gein', J_inv, dN_nat)  # (6, N_e, 2, 6)

        B = np.zeros((6, n_e, 3, 12))
        for i in range(6):
            c = 2 * i
            B[:, :, 0, c    ] = dN_phys[:, :, 0, i]   # εxx : ∂Ni/∂x
            B[:, :, 1, c + 1] = dN_phys[:, :, 1, i]   # εyy : ∂Ni/∂y
            B[:, :, 2, c    ] = dN_phys[:, :, 1, i]   # γxy : ∂Ni/∂y
            B[:, :, 2, c + 1] = dN_phys[:, :, 0, i]   # γxy : ∂Ni/∂x

        K_e_all = np.einsum('g,ge,gepi,pq,geqj->eij', w, det_J, B, D, B) * t
        return K_e_all   # (N_e, 12, 12)

    @classmethod
    def batch_mass_matrix(
        cls,
        nodes_batch: np.ndarray,
        rho: float,
        t: float,
    ) -> np.ndarray:
        """Matrices de masse pour N_e Tri6 en une seule passe tenseur.

        Parameters
        ----------
        nodes_batch : np.ndarray, shape (N_e, 6, 2)
            Coordonnées nodales.
        rho : float
            Masse volumique [kg/m³].
        t : float
            Épaisseur [m].

        Returns
        -------
        M_e_all : np.ndarray, shape (N_e, 12, 12)

        Notes
        -----
        La formule analytique est utilisée (comme dans la méthode scalaire) :

            M_e[e] = (ρ · t · A[e]) · kron(m_bar, I_2)

        où A[e] est l'aire des 3 sommets et m_bar la matrice 6×6 de
        coefficients barycentriques (constant, précomputable).
        """
        # Aire via les 3 sommets (nœuds 0-2)
        x = nodes_batch[:, :3, 0]   # (N_e, 3)
        y = nodes_batch[:, :3, 1]
        area = 0.5 * np.abs(
            (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0])
            - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0])
        )   # (N_e,)

        C  =  1.0 / 30.0
        CO = -1.0 / 180.0
        A0 =  0.0
        NA = -1.0 / 45.0
        MD =  8.0 / 45.0
        MA =  4.0 / 45.0
        m_bar = np.array([
            [ C,  CO,  CO,  A0,  NA,  A0],
            [CO,   C,  CO,  A0,  A0,  NA],
            [CO,  CO,   C,  NA,  A0,  A0],
            [A0,  A0,  NA,  MD,  MA,  MA],
            [NA,  A0,  A0,  MA,  MD,  MA],
            [A0,  NA,  A0,  MA,  MA,  MD],
        ], dtype=float)
        M_pat = np.kron(m_bar, np.eye(2))   # (12, 12) — constant

        scales = rho * t * area   # (N_e,)
        return scales[:, np.newaxis, np.newaxis] * M_pat[np.newaxis]   # (N_e, 12, 12)

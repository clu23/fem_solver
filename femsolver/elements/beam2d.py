"""Élément poutre 2D d'Euler–Bernoulli à 2 nœuds.

Formulation
-----------
Chaque nœud possède 3 DDL : ux (axial), uy (transverse), θz (rotation).
L'élément est une combinaison de :

- Rigidité **axiale** (comme Bar2D) : K_axial = (EA/L) · [[1,-1],[-1,1]]
  sur les DDL {ux₁, ux₂}.

- Rigidité de **flexion Euler–Bernoulli** via les polynômes de Hermite
  cubiques (C¹ continus) sur les DDL {uy₁, θ₁, uy₂, θ₂} :

      v(x) = H₁·uy₁ + H₂·θ₁ + H₃·uy₂ + H₄·θ₂

  avec (ξ = x/L ∈ [0,1]) :
      H₁ = 1 − 3ξ² + 2ξ³
      H₂ = L·ξ(1−ξ)²
      H₃ = 3ξ² − 2ξ³
      H₄ = L·ξ²(ξ−1)

  La rigidité de flexion intègre la courbure κ = v'' :

      k = EI/L³

      K_flex = k · [[ 12,   6L,  −12,   6L ],
                    [  6L,  4L²,  −6L,  2L²],
                    [−12,  −6L,   12,  −6L ],
                    [  6L,  2L²,  −6L,  4L²]]

Matrice 6×6 (repère local, ordre [ux₁, uy₁, θ₁, ux₂, uy₂, θ₂])
-----------------------------------------------------------------

      ux₁    uy₁     θ₁     ux₂    uy₂     θ₂
  [  a       0       0     −a       0       0    ]
  [  0      12k     6kL    0     −12k     6kL    ]
  [  0      6kL    4kL²    0      −6kL   2kL²   ]
  [ −a       0       0      a       0       0    ]
  [  0    −12k    −6kL     0      12k    −6kL    ]
  [  0      6kL   2kL²     0      −6kL   4kL²   ]

avec a = EA/L, k = EI/L³.

Rotation repère local → global
-------------------------------
Pour une poutre inclinée d'angle θ (cosinus c, sinus s) :

    T_nœud = [[ c,  s,  0],
              [−s,  c,  0],
              [ 0,  0,  1]]   ← θz invariant par rotation de repère

    T = block_diag(T_nœud, T_nœud)    (6×6)

    K_global = Tᵀ · K_local · T

Le DDL θz (rotation angulaire 2D) est un scalaire invariant : il se
transforme via la matrice identité dans T, contrairement aux DDL de
translation (ux, uy) qui se transforment avec (c, s).

Propriétés requises (``properties`` dict)
------------------------------------------
- ``"area"`` : section transversale A [m²]
- ``"inertia"`` : moment quadratique I [m⁴] (I = bh³/12 pour section rect.)

Notes
-----
L'hypothèse Euler–Bernoulli (section plane reste plane, pas de déformation
de cisaillement transverse) est valide pour les poutres élancées L/h ≳ 10.
Pour les poutres courtes, utiliser Timoshenko (non implémenté ici).

Références
----------
Cook et al., «Concepts and Applications of FEA», 4th ed., chap. 4-5.
Bathe, «Finite Element Procedures», chap. 5.
"""

from __future__ import annotations

import numpy as np

from femsolver.core.element import Element
from femsolver.core.material import ElasticMaterial
from femsolver.core.sections import Section


class Beam2D(Element):
    """Élément poutre 2D Euler–Bernoulli à 2 nœuds (6 DDL au total).

    Ordre des DDL : [ux₁, uy₁, θ₁, ux₂, uy₂, θ₂].
    Propriétés : ``{"area": A, "inertia": I}``.
    """

    def dof_per_node(self) -> int:
        """3 DDL par nœud : ux, uy, θz."""
        return 3

    def n_nodes(self) -> int:
        """2 nœuds."""
        return 2

    # ------------------------------------------------------------------
    # Géométrie
    # ------------------------------------------------------------------

    def _geometry(self, nodes: np.ndarray) -> tuple[float, float, float]:
        """Longueur et cosinus directeurs de l'élément.

        Parameters
        ----------
        nodes : np.ndarray, shape (2, 2)
            Coordonnées [[x₁, y₁], [x₂, y₂]] en repère global.

        Returns
        -------
        L : float
            Longueur [m].
        c : float
            cos θ (angle entre l'axe local et l'axe x global).
        s : float
            sin θ.

        Raises
        ------
        ValueError
            Si les nœuds sont confondus (L ≈ 0).
        """
        if nodes.shape != (2, 2):
            raise ValueError(
                f"Beam2D attend nodes.shape == (2, 2), reçu {nodes.shape}"
            )
        dx = nodes[1, 0] - nodes[0, 0]
        dy = nodes[1, 1] - nodes[0, 1]
        L = float(np.hypot(dx, dy))
        if L < 1e-14:
            raise ValueError(
                f"Longueur nulle pour Beam2D (nœuds confondus) : nodes={nodes}"
            )
        return L, dx / L, dy / L

    # ------------------------------------------------------------------
    # Matrices locales
    # ------------------------------------------------------------------

    def _stiffness_local(self, ea: float, ei: float, L: float) -> np.ndarray:
        """Matrice de rigidité 6×6 en repère local.

        Parameters
        ----------
        ea : float
            Rigidité axiale E·A [N].
        ei : float
            Rigidité de flexion E·I [N·m²].
        L : float
            Longueur de l'élément [m].

        Returns
        -------
        K_local : np.ndarray, shape (6, 6)
        """
        a = ea / L          # rigidité axiale [N/m]
        k = ei / L**3       # rigidité de flexion de base [N/m³·m³ = N/m]

        # Pré-calcul des coefficients de flexion
        k12  = 12.0 * k
        k6L  =  6.0 * k * L
        k4L2 =  4.0 * k * L**2
        k2L2 =  2.0 * k * L**2

        return np.array([
            [ a,      0,      0,     -a,      0,      0    ],
            [ 0,    k12,    k6L,      0,   -k12,    k6L    ],
            [ 0,    k6L,   k4L2,     0,   -k6L,   k2L2    ],
            [-a,      0,      0,      a,      0,      0    ],
            [ 0,   -k12,   -k6L,     0,    k12,   -k6L    ],
            [ 0,    k6L,   k2L2,     0,   -k6L,   k4L2    ],
        ])

    def _mass_local(self, rho_a: float, L: float) -> np.ndarray:
        """Matrice de masse consistante 6×6 en repère local.

        Découplage axial/flexion :
        - Axial (DDL ux₁, ux₂) : M = (ρAL/6)·[[2,1],[1,2]]
        - Flexion (DDL uy₁, θ₁, uy₂, θ₂) : intégrale de Hermite
          M = (ρAL/420)·[[156, 22L, 54, -13L], ...]

        Parameters
        ----------
        rho_a : float
            Masse linéique ρ·A [kg/m].
        L : float
            Longueur [m].

        Returns
        -------
        M_local : np.ndarray, shape (6, 6)
        """
        # Partie axiale : (ρAL/6)·[[2,1],[1,2]] sur DDL {ux1, ux2}
        ma = rho_a * L / 6.0

        # Partie de flexion : (ρAL/420)·matrice de Hermite sur {uy1, θ1, uy2, θ2}
        mb = rho_a * L / 420.0

        M = np.zeros((6, 6))

        # Termes axiaux (indices 0 et 3)
        M[0, 0] = 2.0 * ma;  M[0, 3] = ma
        M[3, 0] = ma;         M[3, 3] = 2.0 * ma

        # Termes de flexion (indices 1, 2, 4, 5 → uy1, θ1, uy2, θ2)
        L2 = L * L
        M[1, 1] = 156.0  * mb;   M[1, 2] =  22.0 * L  * mb
        M[1, 4] =  54.0  * mb;   M[1, 5] = -13.0 * L  * mb

        M[2, 1] =  22.0 * L  * mb;  M[2, 2] =   4.0 * L2 * mb
        M[2, 4] =  13.0 * L  * mb;  M[2, 5] =  -3.0 * L2 * mb

        M[4, 1] =  54.0  * mb;   M[4, 2] =  13.0 * L  * mb
        M[4, 4] = 156.0  * mb;   M[4, 5] = -22.0 * L  * mb

        M[5, 1] = -13.0 * L  * mb;  M[5, 2] =  -3.0 * L2 * mb
        M[5, 4] = -22.0 * L  * mb;  M[5, 5] =   4.0 * L2 * mb

        return M

    # ------------------------------------------------------------------
    # Matrice de rotation
    # ------------------------------------------------------------------

    def _rotation_matrix(self, c: float, s: float) -> np.ndarray:
        """Matrice de rotation locale → globale 6×6.

        Chaque bloc nœud est :

            T_nœud = [[ c,  s,  0],
                      [−s,  c,  0],
                      [ 0,  0,  1]]

        Le DDL θz (indice 2 et 5) est invariant par rotation : c'est un
        scalaire angulaire, pas un vecteur de translation.

        Parameters
        ----------
        c, s : float
            cos θ et sin θ de l'orientation de la poutre.

        Returns
        -------
        T : np.ndarray, shape (6, 6)
            Transforme les DDL locaux en DDL globaux :
            u_local = T @ u_global.
        """
        T = np.zeros((6, 6))
        for i in range(2):   # bloc nœud 1 (i=0) et nœud 2 (i=1)
            j = 3 * i
            T[j,   j  ] =  c;  T[j,   j+1] = s
            T[j+1, j  ] = -s;  T[j+1, j+1] = c
            T[j+2, j+2] =  1.0
        return T

    # ------------------------------------------------------------------
    # Extraction des propriétés de section
    # ------------------------------------------------------------------

    @staticmethod
    def _beam_props(
        material: ElasticMaterial,
        properties: dict,
    ) -> tuple[float, float]:
        """Retourne (EA, EI) depuis ``properties``.

        Accepte deux formes :

        - ``{"section": Section}``      → utilise ``section.area`` et ``section.Iz``.
        - ``{"area": A, "inertia": I}`` → compatibilité ascendante.

        Parameters
        ----------
        material : ElasticMaterial
            Matériau (module d'Young E).
        properties : dict
            Propriétés géométriques.

        Returns
        -------
        ea : float
            Rigidité axiale E·A [N].
        ei : float
            Rigidité de flexion E·I [N·m²].
        """
        if "section" in properties:
            sec: Section = properties["section"]
            return material.E * sec.area, material.E * sec.Iz
        A = float(properties["area"])
        I = float(properties["inertia"])
        if A <= 0:
            raise ValueError(f"area doit être > 0, reçu {A}")
        if I <= 0:
            raise ValueError(f"inertia doit être > 0, reçu {I}")
        return material.E * A, material.E * I

    @staticmethod
    def _rho_a(material: ElasticMaterial, properties: dict) -> float:
        """Retourne ρ·A [kg/m] pour la matrice de masse.

        Accepte ``{"section": Section}`` ou ``{"area": A, ...}``.
        """
        if "section" in properties:
            return material.rho * properties["section"].area
        A = float(properties["area"])
        if A <= 0:
            raise ValueError(f"area doit être > 0, reçu {A}")
        return material.rho * A

    # ------------------------------------------------------------------
    # Interface publique
    # ------------------------------------------------------------------

    def stiffness_matrix(
        self,
        material: ElasticMaterial,
        nodes: np.ndarray,
        properties: dict,
    ) -> np.ndarray:
        """Matrice de rigidité élémentaire 6×6 en repère global.

        Parameters
        ----------
        material : ElasticMaterial
            Matériau (E utilisé).
        nodes : np.ndarray, shape (2, 2)
            Coordonnées [[x₁, y₁], [x₂, y₂]] en repère global.
        properties : dict
            Deux formes acceptées :

            - ``{"section": Section}``      — objet Section (Beam2D utilise
              ``section.area`` et ``section.Iz``).
            - ``{"area": A, "inertia": I}`` — scalaires (compatibilité ascendante).

        Returns
        -------
        K_e : np.ndarray, shape (6, 6)
            Matrice de rigidité symétrique définie positive.

        Examples
        --------
        Avec un dict scalaire (ancienne interface) :

        >>> mat = ElasticMaterial(E=210e9, nu=0.3, rho=7800)
        >>> nodes = np.array([[0., 0.], [1., 0.]])
        >>> K = Beam2D().stiffness_matrix(mat, nodes, {"area": 0.01, "inertia": 8.333e-6})
        >>> K.shape
        (6, 6)

        Avec un objet Section :

        >>> from femsolver.core.sections import RectangularSection
        >>> sec = RectangularSection(width=0.1, height=0.1)
        >>> K = Beam2D().stiffness_matrix(mat, nodes, {"section": sec})
        >>> K.shape
        (6, 6)
        """
        L, c, s = self._geometry(nodes)
        ea, ei = self._beam_props(material, properties)
        K_local = self._stiffness_local(ea, ei, L)
        T = self._rotation_matrix(c, s)
        return T.T @ K_local @ T

    def mass_matrix(
        self,
        material: ElasticMaterial,
        nodes: np.ndarray,
        properties: dict,
    ) -> np.ndarray:
        """Matrice de masse consistante 6×6 en repère global.

        Parameters
        ----------
        material : ElasticMaterial
            Matériau (rho utilisé).
        nodes : np.ndarray, shape (2, 2)
            Coordonnées en repère global.
        properties : dict
            Accepte ``{"section": Section}`` ou ``{"area": A, ...}``.
            ``inertia`` n'est pas utilisé (inertie de rotation ρI
            négligée, hypothèse Euler–Bernoulli).

        Returns
        -------
        M_e : np.ndarray, shape (6, 6)
            Matrice de masse consistante.

        Notes
        -----
        L'inertie de rotation ρI (terme correctif de Timoshenko) est
        omise : elle est négligeable pour les poutres élancées L/h ≳ 10.
        """
        L, c, s = self._geometry(nodes)
        rho_a = self._rho_a(material, properties)
        M_local = self._mass_local(rho_a, L)
        T = self._rotation_matrix(c, s)
        return T.T @ M_local @ T

    # ------------------------------------------------------------------
    # Post-traitement : efforts internes
    # ------------------------------------------------------------------

    def geometric_stiffness_matrix(
        self,
        material: ElasticMaterial,
        nodes: np.ndarray,
        properties: dict,
        u_e: np.ndarray,
    ) -> np.ndarray:
        """Matrice de rigidité géométrique 6×6 pour la poutre Euler–Bernoulli.

        L'effort axial N de l'état pré-flambement modifie la rigidité
        de flexion via l'intégrale de l'énergie de courbure due à N :

            δ²W = ∫₀ᴸ N · (dv/dx)(dδv/dx) dx

        Avec les polynômes de Hermite, l'intégration donne la sous-matrice
        de flexion 4×4 (DDL {uy₁, θ₁, uy₂, θ₂}) :

            K_g_flex = (N / 30L) · [[ 36,  3L, −36,  3L],
                                      [  3L, 4L²,−3L, −L²],
                                      [−36, −3L,  36, −3L],
                                      [  3L, −L², −3L, 4L²]]

        Les DDL axiaux {ux₁, ux₂} (indices 0 et 3) ont une contribution
        nulle à K_g dans la formulation linéaire.

        Parameters
        ----------
        material : ElasticMaterial
            Matériau (E·A utilisé pour calculer N).
        nodes : np.ndarray, shape (2, 2)
            Coordonnées [[x₁, y₁], [x₂, y₂]].
        properties : dict
            ``"area"`` et ``"inertia"`` (ou ``"section"``).
        u_e : np.ndarray, shape (6,)
            Déplacements [ux₁, uy₁, θ₁, ux₂, uy₂, θ₂] (repère global).

        Returns
        -------
        K_g_e : np.ndarray, shape (6, 6)
            Rigidité géométrique en repère global.
            Négative dans le bloc de flexion si N < 0 (compression).

        Notes
        -----
        Référence : Cook et al., « Concepts and Applications of FEA »,
        4th ed., §15.1 ; Bathe, « FE Procedures », §6.3.4.
        """
        L, c, s = self._geometry(nodes)
        T = self._rotation_matrix(c, s)
        u_local = T @ u_e
        ea, _ = self._beam_props(material, properties)
        N = ea / L * (u_local[3] - u_local[0])   # N < 0 = compression

        L2 = L * L
        a = N / (30.0 * L)

        K_g_local = a * np.array([
            [ 0.0,   0.0,    0.0,    0.0,   0.0,    0.0  ],
            [ 0.0,  36.0,   3.0*L,   0.0, -36.0,   3.0*L ],
            [ 0.0,   3.0*L, 4.0*L2,  0.0,  -3.0*L, -L2   ],
            [ 0.0,   0.0,    0.0,    0.0,   0.0,    0.0  ],
            [ 0.0, -36.0,  -3.0*L,   0.0,  36.0,  -3.0*L ],
            [ 0.0,   3.0*L, -L2,     0.0,  -3.0*L,  4.0*L2],
        ])
        return T.T @ K_g_local @ T

    def distributed_load_vector(
        self,
        material: ElasticMaterial,
        nodes: np.ndarray,
        properties: dict,
        qx: float,
        qy: float,
    ) -> np.ndarray:
        """Forces nodales équivalentes pour une charge distribuée uniforme.

        Utilise les fonctions de forme de Hermite pour la partie transverse
        (charge qy) et des fonctions de forme linéaires pour la partie axiale
        (charge qx). La solution est **exacte aux nœuds** pour 1 seul élément.

        Parameters
        ----------
        material : ElasticMaterial
            Non utilisé (interface commune avec les autres éléments).
        nodes : np.ndarray, shape (2, 2)
            Coordonnées des nœuds en repère global.
        properties : dict
            Propriétés de section (non utilisées ici, interface uniforme).
        qx : float
            Charge axiale distribuée dans le repère local [N/m].
            Positif dans le sens nœud 1 → nœud 2.
        qy : float
            Charge transverse distribuée dans le repère local [N/m].
            Positif dans le sens +y local (vers le haut pour une poutre
            horizontale).

        Returns
        -------
        f_e : np.ndarray, shape (6,)
            Forces nodales équivalentes [N, N, N·m, N, N, N·m] en repère
            global, dans l'ordre [Fx₁, Fy₁, Mz₁, Fx₂, Fy₂, Mz₂].

        Notes
        -----
        Intégrales des fonctions de Hermite sur [0, L] :

        ∫₀ᴸ H₁ dx = L/2,   ∫₀ᴸ H₂ dx = L²/12
        ∫₀ᴸ H₃ dx = L/2,   ∫₀ᴸ H₄ dx = −L²/12

        Vecteur local (ordre [ux₁, uy₁, θ₁, ux₂, uy₂, θ₂]) :

            f_local = [qx·L/2,   qy·L/2,   qy·L²/12,
                       qx·L/2,   qy·L/2,  −qy·L²/12]

        Validation analytique — console encastrée-libre (L, EI, q uniforme) :

            δ_tip  = q·L⁴/(8·EI)    ← exact avec 1 seul élément
            θ_tip  = q·L³/(6·EI)    ← exact avec 1 seul élément

        Examples
        --------
        >>> mat = ElasticMaterial(E=210e9, nu=0.3, rho=7800)
        >>> nodes = np.array([[0., 0.], [1., 0.]])
        >>> f = Beam2D().distributed_load_vector(mat, nodes, {}, qx=0., qy=1000.)
        >>> f  # [0, 500, 83.33, 0, 500, -83.33]
        array([  0.        , 500.        ,  83.33333333,   0.        ,
               500.        , -83.33333333])
        """
        L, c, s = self._geometry(nodes)
        f_local = np.array([
            qx * L / 2.0,         # Fx1 : axial (linéaire)
            qy * L / 2.0,         # Fy1 : transverse ∫H₁·qy dx
            qy * L**2 / 12.0,     # Mz1 : moment ∫H₂·qy dx
            qx * L / 2.0,         # Fx2 : axial (linéaire)
            qy * L / 2.0,         # Fy2 : transverse ∫H₃·qy dx
            -qy * L**2 / 12.0,    # Mz2 : moment ∫H₄·qy dx (négatif)
        ])
        T = self._rotation_matrix(c, s)
        return T.T @ f_local

    def section_forces(
        self,
        material: ElasticMaterial,
        nodes: np.ndarray,
        properties: dict,
        u_e: np.ndarray,
    ) -> dict[str, float]:
        """Efforts internes aux deux extrémités de l'élément.

        Calcule l'effort normal N, l'effort tranchant V et le moment
        fléchissant M à partir du champ de déplacements nodaux.

        Parameters
        ----------
        material : ElasticMaterial
            Matériau.
        nodes : np.ndarray, shape (2, 2)
            Coordonnées des nœuds.
        properties : dict
            ``"area"`` et ``"inertia"``.
        u_e : np.ndarray, shape (6,)
            Déplacements globaux [ux₁, uy₁, θ₁, ux₂, uy₂, θ₂].

        Returns
        -------
        dict with keys :
            ``N1``, ``V1``, ``M1`` : effort normal, tranchant, moment au nœud 1.
            ``N2``, ``V2``, ``M2`` : idem au nœud 2.

        Notes
        -----
        Calcul en repère local :
            f_local = K_local · u_local
        Les forces aux nœuds sont les réactions internes (signe convention
        de la poutre, non des réactions d'appui).
        """
        L, c, s = self._geometry(nodes)
        T = self._rotation_matrix(c, s)
        u_local = T @ u_e
        ea, ei = self._beam_props(material, properties)
        K_local = self._stiffness_local(ea, ei, L)
        f_local = K_local @ u_local   # [N1, V1, M1, N2, V2, M2]
        return {
            "N1": float(f_local[0]),
            "V1": float(f_local[1]),
            "M1": float(f_local[2]),
            "N2": float(f_local[3]),
            "V2": float(f_local[4]),
            "M2": float(f_local[5]),
        }

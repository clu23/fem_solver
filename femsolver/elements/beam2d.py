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
            Doit contenir :
            - ``"area"``    : section transversale A [m²]
            - ``"inertia"`` : moment quadratique I [m⁴]

        Returns
        -------
        K_e : np.ndarray, shape (6, 6)
            Matrice de rigidité symétrique définie positive.

        Examples
        --------
        Poutre horizontale 1 m, acier, section 10 × 10 cm :

        >>> mat = ElasticMaterial(E=210e9, nu=0.3, rho=7800)
        >>> nodes = np.array([[0., 0.], [1., 0.]])
        >>> props = {"area": 0.01, "inertia": 8.333e-6}
        >>> K = Beam2D().stiffness_matrix(mat, nodes, props)
        >>> K.shape
        (6, 6)
        """
        A = properties["area"]
        I = properties["inertia"]
        if A <= 0:
            raise ValueError(f"area doit être > 0, reçu {A}")
        if I <= 0:
            raise ValueError(f"inertia doit être > 0, reçu {I}")

        L, c, s = self._geometry(nodes)
        K_local = self._stiffness_local(material.E * A, material.E * I, L)
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
            Doit contenir ``"area"`` [m²]. ``"inertia"`` non utilisé ici
            (la masse de rotation ρI est négligée, hypothèse Euler–Bernoulli).

        Returns
        -------
        M_e : np.ndarray, shape (6, 6)
            Matrice de masse consistante.

        Notes
        -----
        L'inertie de rotation ρI (terme correctif de Timoshenko) est
        omise : elle est négligeable pour les poutres élancées L/h ≳ 10.
        """
        A = properties["area"]
        if A <= 0:
            raise ValueError(f"area doit être > 0, reçu {A}")

        L, c, s = self._geometry(nodes)
        M_local = self._mass_local(material.rho * A, L)
        T = self._rotation_matrix(c, s)
        return T.T @ M_local @ T

    # ------------------------------------------------------------------
    # Post-traitement : efforts internes
    # ------------------------------------------------------------------

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
        A = properties["area"]
        I = properties["inertia"]
        L, c, s = self._geometry(nodes)
        T = self._rotation_matrix(c, s)
        u_local = T @ u_e
        K_local = self._stiffness_local(material.E * A, material.E * I, L)
        f_local = K_local @ u_local   # [N1, V1, M1, N2, V2, M2]
        return {
            "N1": float(f_local[0]),
            "V1": float(f_local[1]),
            "M1": float(f_local[2]),
            "N2": float(f_local[3]),
            "V2": float(f_local[4]),
            "M2": float(f_local[5]),
        }

"""Élément poutre 2D de Timoshenko à 2 nœuds.

Différences avec Euler–Bernoulli (``Beam2D``)
---------------------------------------------

Dans la théorie d'Euler–Bernoulli (EB), la section droite reste
perpendiculaire à l'axe neutre après déformation : il n'y a pas de
déformation de cisaillement transverse.  C'est une bonne approximation
pour les poutres élancées (L/h ≳ 10), mais elle surestime la rigidité
des poutres courtes ou des sections de grande épaisseur.

La théorie de Timoshenko lève cette hypothèse en introduisant un angle de
cisaillement γ indépendant de la courbure :

    γ = v' − ψ        (glissement transverse)

où v est le déplacement transverse et ψ la rotation de la section (DDL
indépendant de v').

Paramètre de cisaillement Φ
----------------------------
La relative importance du cisaillement est mesurée par le paramètre
adimensionnel Φ (parfois noté α²) :

    Φ = 12 E I / (G A_s L²)

- **G A_s = G κ A** est la rigidité de cisaillement effective.
- **κ** (kappa) est le facteur de correction de cisaillement de la
  section (fourni par :meth:`~femsolver.core.sections.Section.shear_correction_factor`).
- Quand Φ → 0 (L grand, section élancée), la rigidité de Timoshenko
  converge vers celle d'Euler–Bernoulli.

Section cisaillée A_s
---------------------
Le facteur κ (Cowper 1966) compense le fait que la contrainte de
cisaillement n'est pas uniformément répartie sur la section :

    G A_s = G κ A   avec κ ∈ (0, 1]

Valeurs typiques (Cowper 1966) :

- Section rectangulaire : κ = 10(1+ν)/(12+11ν) ≈ 0.833  (ν = 0.3)
- Section circulaire     : κ = 6(1+ν)/(7+6ν)   ≈ 0.886  (ν = 0.3)
- Profilé en I (approx.) : κ ≈ A_web / A        ≈ 0.3–0.6

Matrice de rigidité 6×6 locale (Timoshenko exact)
--------------------------------------------------
Ordre des DDL : [ux₁, uy₁, θ₁, ux₂, uy₂, θ₂].

La partie axiale (ux) est identique à EB.

Pour la partie flexion-cisaillement on pose :

    k = E I / ((1 + Φ) L³)

La matrice de flexion 4×4 (sur {uy₁, θ₁, uy₂, θ₂}) devient :

    k · [[ 12,      6L,       −12,      6L      ],
         [  6L,  (4+Φ)L²,    −6L,   (2−Φ)L²  ],
         [−12,     −6L,        12,     −6L      ],
         [  6L,  (2−Φ)L²,    −6L,   (4+Φ)L²  ]]

Comparaison avec EB :
  - La rigidité globale est divisée par (1+Φ) : poutres courtes ⟹ Φ grand
    ⟹ deflexion plus grande qu'en EB.
  - Les termes de moment (2,2) et (4,4) passent de 4kL² à (4+Φ)kL² ;
    les termes couplés (2,5) de 2kL² à (2−Φ)kL².
  - Pour Φ = 0 : K_Timoshenko ≡ K_EB (aucun verrouillage en cisaillement
    car la formulation utilise les fonctions de forme exactes d'équilibre).

Matrice de masse consistante 6×6 locale
----------------------------------------
La matrice de masse inclut l'**inertie de rotation** ρI (terme correctif
important pour les poutres courtes ou les hautes fréquences) et les
corrections en Φ.  Formulation de Friedman & Kosmatka (1993).

Références
----------
Friedman Z. & Kosmatka J.B. (1993), *An improved two-node Timoshenko beam
finite element*, CMAME 105, 187–199.

Cowper G.R. (1966), *The Shear Coefficient in Timoshenko's Beam Theory*,
J. Appl. Mech. 33(2), 335–340.

Cook R.D. et al., «Concepts and Applications of FEA», 4th ed., chap. 4.
"""

from __future__ import annotations

import numpy as np

from femsolver.core.material import ElasticMaterial
from femsolver.core.sections import Section
from femsolver.elements.beam2d import Beam2D


class Beam2DTimoshenko(Beam2D):
    """Élément poutre 2D de Timoshenko à 2 nœuds (6 DDL au total).

    Hérite de :class:`~femsolver.elements.beam2d.Beam2D` pour la géométrie
    et la matrice de rotation (même structure DOF : [ux₁, uy₁, θ₁, ux₂, uy₂, θ₂]).

    Propriétés requises (``properties`` dict)
    ------------------------------------------
    - ``"section": Section``  — objet Section avec :meth:`shear_correction_factor`.
    - **ou** ``{"area": A, "inertia": I, "shear_area": A_s}`` — scalaires
      (A_s = κ·A, rigidité de cisaillement directement spécifiée).

    Notes
    -----
    Contrairement à ``Beam2D``, les propriétés scalaires requièrent
    ``"shear_area"`` en plus de ``"area"`` et ``"inertia"``.
    """

    # ------------------------------------------------------------------
    # Extraction des propriétés de section (Timoshenko)
    # ------------------------------------------------------------------

    @staticmethod
    def _beam_props_timoshenko(
        material: ElasticMaterial,
        properties: dict,
    ) -> tuple[float, float, float]:
        """Retourne (EA, EI, GA_s) depuis ``properties``.

        Parameters
        ----------
        material : ElasticMaterial
            Matériau.
        properties : dict
            ``{"section": Section}`` ou ``{"area": A, "inertia": I, "shear_area": A_s}``.

        Returns
        -------
        ea : float
            Rigidité axiale E·A [N].
        ei : float
            Rigidité de flexion E·I [N·m²].
        gas : float
            Rigidité de cisaillement effective G·κ·A [N].
        """
        if "section" in properties:
            sec: Section = properties["section"]
            kappa = sec.shear_correction_factor(material.nu)
            return (
                material.E * sec.area,
                material.E * sec.Iz,
                material.G * kappa * sec.area,
            )
        A = float(properties["area"])
        I = float(properties["inertia"])
        if "shear_area" not in properties:
            raise ValueError(
                "Beam2DTimoshenko : 'shear_area' (= κ·A) est requis dans properties "
                "quand 'section' n'est pas fourni."
            )
        A_s = float(properties["shear_area"])
        if A <= 0:
            raise ValueError(f"area doit être > 0, reçu {A}")
        if I <= 0:
            raise ValueError(f"inertia doit être > 0, reçu {I}")
        if A_s <= 0:
            raise ValueError(f"shear_area doit être > 0, reçu {A_s}")
        return material.E * A, material.E * I, material.G * A_s

    @staticmethod
    def _rho_i(material: ElasticMaterial, properties: dict) -> float:
        """Retourne ρ·I [kg·m] pour l'inertie de rotation.

        Parameters
        ----------
        material : ElasticMaterial
        properties : dict

        Returns
        -------
        float
        """
        if "section" in properties:
            return material.rho * properties["section"].Iz
        return material.rho * float(properties["inertia"])

    # ------------------------------------------------------------------
    # Matrices locales de Timoshenko
    # ------------------------------------------------------------------

    @staticmethod
    def _stiffness_local_timoshenko(
        ea: float,
        ei: float,
        gas: float,
        L: float,
    ) -> np.ndarray:
        """Matrice de rigidité 6×6 de Timoshenko en repère local.

        Formulation exacte basée sur les fonctions de forme d'équilibre
        (sans verrouillage en cisaillement).

        Parameters
        ----------
        ea : float
            Rigidité axiale E·A [N].
        ei : float
            Rigidité de flexion E·I [N·m²].
        gas : float
            Rigidité de cisaillement effective G·κ·A [N].
        L : float
            Longueur [m].

        Returns
        -------
        K_local : np.ndarray, shape (6, 6)

        Notes
        -----
        Paramètre de cisaillement : Φ = 12·EI / (GA_s·L²).

        Avec k = EI / ((1+Φ)·L³) :

            K_flex = k · [[ 12,      6L,       −12,      6L      ],
                          [  6L,  (4+Φ)L²,    −6L,   (2−Φ)L²  ],
                          [−12,     −6L,        12,     −6L      ],
                          [  6L,  (2−Φ)L²,    −6L,   (4+Φ)L²  ]]
        """
        a = ea / L
        Phi = 12.0 * ei / (gas * L ** 2)
        k = ei / ((1.0 + Phi) * L ** 3)

        k12 = 12.0 * k
        k6L = 6.0 * k * L
        k4p = (4.0 + Phi) * k * L ** 2
        k2m = (2.0 - Phi) * k * L ** 2

        return np.array([
            [ a,     0,     0,    -a,     0,     0  ],
            [ 0,   k12,   k6L,    0,   -k12,   k6L  ],
            [ 0,   k6L,   k4p,    0,   -k6L,   k2m  ],
            [-a,     0,     0,     a,     0,     0  ],
            [ 0,  -k12,  -k6L,    0,    k12,  -k6L  ],
            [ 0,   k6L,   k2m,    0,   -k6L,   k4p  ],
        ])

    @staticmethod
    def _mass_local_timoshenko(
        rho_a: float,
        rho_i: float,
        Phi: float,
        L: float,
    ) -> np.ndarray:
        """Matrice de masse consistante 6×6 de Timoshenko en repère local.

        Inclut l'inertie de rotation ρ·I (termes correctifs pour poutres
        courtes).  Formulation de Friedman & Kosmatka (1993).

        Parameters
        ----------
        rho_a : float
            Masse linéique ρ·A [kg/m].
        rho_i : float
            Inertie de rotation linéique ρ·I [kg·m].
        Phi : float
            Paramètre de cisaillement Φ = 12EI/(GA_s·L²).
        L : float
            Longueur [m].

        Returns
        -------
        M_local : np.ndarray, shape (6, 6)

        Notes
        -----
        La matrice de flexion se décompose en partie translationnelle M_T
        et partie rotationnelle M_R :

        M_T = (ρAL / (1+Φ)²) × m_T(Φ, L)
        M_R = (ρI / (L(1+Φ)²)) × m_R(Φ, L)

        Pour Φ = 0 : M_T → matrice de Hermite EB (156, 22L, …/420),
                     M_R → matrice d'inertie de rotation EB.

        References
        ----------
        Friedman & Kosmatka (1993), CMAME 105, Table 1.
        """
        denom = (1.0 + Phi) ** 2
        m_t = rho_a * L / denom            # facteur translationnelle
        m_r = rho_i / (L * denom)          # facteur rotationnelle

        L2 = L * L

        # ------------------------------------------------------------------
        # Partie translationnelle (bending DOF : uy1, θ1, uy2, θ2)
        # Ligne/colonne : 0=uy1, 1=θ1, 2=uy2, 3=θ2  (bloc 4×4)
        # ------------------------------------------------------------------
        P = Phi
        P2 = Phi * Phi

        t11 = 13.0 / 35.0 + 7.0 * P / 10.0 + P2 / 3.0
        t12 = (11.0 / 210.0 + 11.0 * P / 120.0 + P2 / 24.0) * L
        t13 = 9.0 / 70.0 + 3.0 * P / 10.0 + P2 / 6.0
        t14 = -(13.0 / 420.0 + 3.0 * P / 40.0 + P2 / 24.0) * L
        t22 = (1.0 / 105.0 + P / 60.0 + P2 / 120.0) * L2
        t23 = (13.0 / 420.0 + 3.0 * P / 40.0 + P2 / 24.0) * L
        t24 = -(1.0 / 140.0 + P / 60.0 + P2 / 120.0) * L2
        # t33 = t11  (symétrie)
        # t34 = -t12
        # t44 = t22  (symétrie)

        # ------------------------------------------------------------------
        # Partie rotationnelle
        # ------------------------------------------------------------------
        r11 = 6.0 / 5.0
        r12 = (1.0 / 10.0 - P / 2.0) * L
        # r13 = -r11
        # r14 = r12
        r22 = (2.0 / 15.0 + P / 6.0 + P2 / 3.0) * L2
        # r23 = -r12
        r24 = -(1.0 / 30.0 - P / 6.0 + P2 / 6.0) * L2
        # r33 = r11
        # r34 = -r12
        # r44 = r22

        # ------------------------------------------------------------------
        # Assemblage du bloc de flexion 4×4
        # (indices 0=uy1, 1=θ1, 2=uy2, 3=θ2)
        # ------------------------------------------------------------------
        Mb = np.zeros((4, 4))

        # Colonne 0 (uy1)
        Mb[0, 0] = m_t * t11 + m_r * r11
        Mb[1, 0] = m_t * t12 + m_r * r12
        Mb[2, 0] = m_t * t13 - m_r * r11
        Mb[3, 0] = m_t * t14 + m_r * r12

        # Colonne 1 (θ1)
        Mb[0, 1] = Mb[1, 0]
        Mb[1, 1] = m_t * t22 + m_r * r22
        Mb[2, 1] = m_t * t23 - m_r * r12
        Mb[3, 1] = m_t * t24 + m_r * r24

        # Colonne 2 (uy2)
        Mb[0, 2] = Mb[2, 0]
        Mb[1, 2] = Mb[2, 1]
        Mb[2, 2] = m_t * t11 + m_r * r11
        Mb[3, 2] = -(m_t * t12 + m_r * r12)

        # Colonne 3 (θ2)
        Mb[0, 3] = Mb[3, 0]
        Mb[1, 3] = Mb[3, 1]
        Mb[2, 3] = Mb[3, 2]
        Mb[3, 3] = m_t * t22 + m_r * r22

        # ------------------------------------------------------------------
        # Matrice axiale 2×2 (inchangée vs EB)
        # (indices 0=ux1, 1=ux2)
        # ------------------------------------------------------------------
        ma = rho_a * L / 6.0
        Ma = np.array([[2.0 * ma, ma], [ma, 2.0 * ma]])

        # ------------------------------------------------------------------
        # Assemblage 6×6
        # Ordre DDL : ux1(0), uy1(1), θ1(2), ux2(3), uy2(4), θ2(5)
        # Bending block Mb → indices (1,2,4,5)
        # Axial block Ma  → indices (0,3)
        # ------------------------------------------------------------------
        M = np.zeros((6, 6))

        # Axial
        M[0, 0] = Ma[0, 0];  M[0, 3] = Ma[0, 1]
        M[3, 0] = Ma[1, 0];  M[3, 3] = Ma[1, 1]

        # Flexion — mappage : Mb(0,1,2,3) → M(1,2,4,5)
        idx = [1, 2, 4, 5]
        for i, gi in enumerate(idx):
            for j, gj in enumerate(idx):
                M[gi, gj] = Mb[i, j]

        return M

    # ------------------------------------------------------------------
    # Interface publique (override Beam2D)
    # ------------------------------------------------------------------

    def stiffness_matrix(
        self,
        material: ElasticMaterial,
        nodes: np.ndarray,
        properties: dict,
    ) -> np.ndarray:
        """Matrice de rigidité de Timoshenko 6×6 en repère global.

        Parameters
        ----------
        material : ElasticMaterial
            Matériau (E, nu utilisés).
        nodes : np.ndarray, shape (2, 2)
            Coordonnées [[x₁, y₁], [x₂, y₂]].
        properties : dict
            ``{"section": Section}`` ou
            ``{"area": A, "inertia": I, "shear_area": A_s}``.

        Returns
        -------
        K_e : np.ndarray, shape (6, 6)

        Examples
        --------
        Poutre rectangulaire carrée :

        >>> import numpy as np
        >>> from femsolver.core.material import ElasticMaterial
        >>> from femsolver.core.sections import RectangularSection
        >>> mat = ElasticMaterial(E=210e9, nu=0.3, rho=7800)
        >>> sec = RectangularSection(width=0.1, height=0.1)
        >>> nodes = np.array([[0., 0.], [1., 0.]])
        >>> K = Beam2DTimoshenko().stiffness_matrix(mat, nodes, {"section": sec})
        >>> K.shape
        (6, 6)
        """
        L, c, s = self._geometry(nodes)
        ea, ei, gas = self._beam_props_timoshenko(material, properties)
        K_local = self._stiffness_local_timoshenko(ea, ei, gas, L)
        T = self._rotation_matrix(c, s)
        return T.T @ K_local @ T

    def mass_matrix(
        self,
        material: ElasticMaterial,
        nodes: np.ndarray,
        properties: dict,
    ) -> np.ndarray:
        """Matrice de masse de Timoshenko 6×6 en repère global.

        Inclut l'inertie de rotation ρ·I et les corrections en Φ.

        Parameters
        ----------
        material : ElasticMaterial
        nodes : np.ndarray, shape (2, 2)
        properties : dict

        Returns
        -------
        M_e : np.ndarray, shape (6, 6)
        """
        L, c, s = self._geometry(nodes)
        ea, ei, gas = self._beam_props_timoshenko(material, properties)
        rho_a = self._rho_a(material, properties)
        rho_i = self._rho_i(material, properties)
        Phi = 12.0 * ei / (gas * L ** 2)
        M_local = self._mass_local_timoshenko(rho_a, rho_i, Phi, L)
        T = self._rotation_matrix(c, s)
        return T.T @ M_local @ T

    def section_forces(
        self,
        material: ElasticMaterial,
        nodes: np.ndarray,
        properties: dict,
        u_e: np.ndarray,
    ) -> dict[str, float]:
        """Efforts internes aux deux extrémités (Timoshenko).

        Parameters
        ----------
        material : ElasticMaterial
        nodes : np.ndarray, shape (2, 2)
        properties : dict
        u_e : np.ndarray, shape (6,)
            Déplacements globaux.

        Returns
        -------
        dict with keys ``N1``, ``V1``, ``M1``, ``N2``, ``V2``, ``M2``.
        """
        L, c, s = self._geometry(nodes)
        T = self._rotation_matrix(c, s)
        u_local = T @ u_e
        ea, ei, gas = self._beam_props_timoshenko(material, properties)
        K_local = self._stiffness_local_timoshenko(ea, ei, gas, L)
        f_local = K_local @ u_local
        return {
            "N1": float(f_local[0]),
            "V1": float(f_local[1]),
            "M1": float(f_local[2]),
            "N2": float(f_local[3]),
            "V2": float(f_local[4]),
            "M2": float(f_local[5]),
        }

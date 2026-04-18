"""Élément poutre 3D de Timoshenko à 2 nœuds — 6 DDL/nœud.

Degrés de liberté (repère global)
----------------------------------
Ordre DDL : [ux₁, uy₁, uz₁, θx₁, θy₁, θz₁,  ux₂, uy₂, uz₂, θx₂, θy₂, θz₂]
Indices   : [  0,   1,   2,   3,   4,   5,     6,   7,   8,   9,  10,  11 ]

Convention de signe des rotations : règle de la main droite pour θx, θy, θz
(rotation positive = sens antihoraire vu de l'extrémité positive de l'axe).

Repère local de l'élément
--------------------------
Trois vecteurs unitaires orthonormaux (e₁, e₂, e₃) construits depuis la
géométrie et le vecteur d'orientation v (« v-vector ») fourni par l'utilisateur,
qui joue le même rôle que le v-vector de la carte CBAR/CBEAM de Nastran.

    e₁ = (node₂ − node₁) / L           ← axe local x̂ (axe de l'élément)
    e₃ = normalise(e₁ × v)             ← axe local ẑ (normal au plan e₁–v)
    e₂ = e₃ × e₁                       ← axe local ŷ (complète le trièdre direct)

Le plan e₁–e₂ est le « plan de la section web » (plan de flexion forte Iz).

v-vector par défaut :
  - Si e₁ n'est pas parallèle à Z_global = [0, 0, 1] : v = [0, 0, 1]
  - Sinon : v = [0, 1, 0]
  (pour les poutres horizontales, e₂ pointe vers le haut, convention la plus naturelle)

Matrice de rotation locale → globale (12×12)
---------------------------------------------
La matrice de direction λ (3×3) contient les cosinus directeurs :
    λ = [[e₁ᵀ],   ← e₁ exprimé dans le repère global
         [e₂ᵀ],
         [e₃ᵀ]]

La transformation complète pour les DDL d'un nœud est :
    T_nœud = [[λ  0],     u_local = T_nœud · u_global  (translations ET rotations)
              [0  λ]]

Cela fonctionne car, avec la règle de la main droite, les pseudo-vecteurs
de rotation se transforment identiquement aux vecteurs de translation.

    T = block_diag(T_nœud₁, T_nœud₂)   (12×12)
    K_global = Tᵀ · K_local · T

Matrice de rigidité locale 12×12 (Timoshenko)
----------------------------------------------
Structure par blocs (Φy, Φz : paramètres de cisaillement de Timoshenko) :

    Axial   → bloc 2×2 sur {ux₁(0), ux₂(6)} :  EA/L · [[1,-1],[-1,1]]
    Torsion → bloc 2×2 sur {θx₁(3), θx₂(9)} :  GJ/L · [[1,-1],[-1,1]]

    Plan xy → bloc 4×4 sur {uy₁(1), θz₁(5), uy₂(7), θz₂(11)} :
        ky = EIz / ((1+Φy)L³)    avec Φy = 12EIz/(GAsy·L²)

        ky · [[ 12,      6L,       −12,      6L      ],
              [  6L,  (4+Φy)L²,   −6L,   (2−Φy)L²  ],
              [−12,     −6L,        12,     −6L      ],
              [  6L,  (2−Φy)L²,   −6L,   (4+Φy)L²  ]]

    Plan xz → bloc 4×4 sur {uz₁(2), θy₁(4), uz₂(8), θy₂(10)} :
        kz = EIy / ((1+Φz)L³)    avec Φz = 12EIy/(GAsz·L²)

        ATTENTION — signe négatif sur les couplages uz/θy (règle main droite) :
        Rotation positive θy ≡ duz/dx négatif (contrairement à θz ≡ +duy/dx).

        kz · [[ 12,     −6L,       −12,     −6L      ],
              [ −6L,  (4+Φz)L²,    6L,   (2−Φz)L²  ],
              [−12,     6L,         12,      6L      ],
              [ −6L,  (2−Φz)L²,    6L,   (4+Φz)L²  ]]

Support d'offset (excentrement)
--------------------------------
Un offset a_i = [aix, aiy, aiz] au nœud i déplace le point de connexion de la
poutre (son axe élastique) par rapport au nœud du maillage. La transformation
est une liaison rigide :

    u_beam_end = G_i · u_node_i     avec   G_i = [[I,  −ã_i],
                                                   [0,   I  ]]

    ã_i = skew(a_i) = [[ 0,   −aiz,  aiy],
                       [ aiz,   0,  −aix],
                       [−aiy,  aix,   0 ]]

    T_off = block_diag(G₁, G₂)   (12×12)
    K_off = T_off^T · K_e · T_off

Propriétés requises (``properties`` dict)
-----------------------------------------
- ``"section"`` : objet :class:`~femsolver.core.sections.Section`
    Fournit A, Iy, Iz, J, κ (via ``shear_correction_factor``).
    La même valeur κ est utilisée pour les deux plans de flexion (Asy = Asz = κA).
    Pour des κy ≠ κz, surcharger avec ``"kappa_y"`` et/ou ``"kappa_z"`` (float).

- ``"v_vec"`` : np.ndarray, shape (3,), optionnel
    Vecteur d'orientation définissant le plan de flexion forte (plan xy local).
    Doit être non parallèle à l'axe de la poutre.
    Défaut : [0,0,1] si la poutre n'est pas verticale, [0,1,0] sinon.

- ``"offset_i"`` : np.ndarray, shape (3,), optionnel
    Vecteur d'excentrement au nœud 1 (en repère global) [m]. Défaut : zéro.

- ``"offset_j"`` : np.ndarray, shape (3,), optionnel
    Vecteur d'excentrement au nœud 2 (en repère global) [m]. Défaut : zéro.

Références
----------
Przemieniecki J.S. (1968). *Theory of Matrix Structural Analysis*, chap. 5.
McGraw-Hill.

Friedman Z. & Kosmatka J.B. (1993). An improved two-node Timoshenko beam
finite element. *CMAME* 105, 187–199.

Cook R.D. et al. (2002). *Concepts and Applications of FEA*, 4th ed.,
chap. 5. Wiley.

Nastran Quick Reference Guide — CBAR / CBEAM element definition, §4.
"""

from __future__ import annotations

import numpy as np

from femsolver.core.element import Element
from femsolver.core.material import ElasticMaterial
from femsolver.core.sections import Section


class Beam3D(Element):
    """Poutre Timoshenko 3D à 2 nœuds, 6 DDL/nœud (12 DDL au total).

    Paramètres de l'élément via le dict ``properties`` :

    - ``"section"``   : :class:`~femsolver.core.sections.Section` — obligatoire.
    - ``"v_vec"``     : np.ndarray (3,) — vecteur d'orientation (optionnel).
    - ``"offset_i"``  : np.ndarray (3,) — offset nœud 1 en repère global (optionnel).
    - ``"offset_j"``  : np.ndarray (3,) — offset nœud 2 en repère global (optionnel).
    - ``"kappa_y"``   : float — surcharge du facteur κ pour le plan xy (optionnel).
    - ``"kappa_z"``   : float — surcharge du facteur κ pour le plan xz (optionnel).

    Voir le module docstring pour la description complète de la formulation.
    """

    # ------------------------------------------------------------------
    # Interface Element
    # ------------------------------------------------------------------

    def dof_per_node(self) -> int:
        """6 DDL par nœud : ux, uy, uz, θx, θy, θz."""
        return 6

    def n_nodes(self) -> int:
        """2 nœuds."""
        return 2

    # ------------------------------------------------------------------
    # Géométrie et repère local
    # ------------------------------------------------------------------

    @staticmethod
    def _local_frame(
        nodes: np.ndarray,
        v_vec: np.ndarray | None,
    ) -> tuple[float, np.ndarray]:
        """Longueur et matrice de direction cosinus λ (3×3).

        Parameters
        ----------
        nodes : np.ndarray, shape (2, 3)
            Coordonnées 3D des deux nœuds [[x₁,y₁,z₁], [x₂,y₂,z₂]].
        v_vec : np.ndarray, shape (3,) or None
            Vecteur d'orientation.  Si None, choisit le défaut.

        Returns
        -------
        L : float
            Longueur de l'élément [m].
        lam : np.ndarray, shape (3, 3)
            Matrice de direction cosinus — chaque ligne est un vecteur
            de base local exprimé en repère global :
                lam[0] = e₁ (axe de la poutre)
                lam[1] = e₂ (axe local ŷ)
                lam[2] = e₃ (axe local ẑ)

        Raises
        ------
        ValueError
            Si les nœuds sont confondus (L < ε) ou si v_vec est parallèle
            à e₁ (section ne peut pas être orientée).
        """
        if nodes.shape != (2, 3):
            raise ValueError(
                f"Beam3D attend nodes.shape == (2, 3), reçu {nodes.shape}"
            )
        d = nodes[1] - nodes[0]
        L = float(np.linalg.norm(d))
        if L < 1e-14:
            raise ValueError(
                f"Longueur nulle pour Beam3D (nœuds confondus) : nodes={nodes}"
            )
        e1 = d / L

        # --- Choix du v-vector par défaut ---
        if v_vec is None:
            z_global = np.array([0.0, 0.0, 1.0])
            if abs(abs(np.dot(e1, z_global)) - 1.0) < 1e-6:
                # Poutre quasi-verticale : v = X global
                v_vec = np.array([1.0, 0.0, 0.0])
            else:
                v_vec = z_global

        v = np.asarray(v_vec, dtype=float)
        if np.linalg.norm(v) < 1e-14:
            raise ValueError("v_vec ne doit pas être le vecteur nul.")

        # --- e₃ = normalise(e₁ × v)
        e3 = np.cross(e1, v)
        norm_e3 = float(np.linalg.norm(e3))
        if norm_e3 < 1e-10:
            raise ValueError(
                f"v_vec est quasi-parallèle à l'axe de la poutre e₁={e1}.  "
                "Choisir un v_vec non parallèle à l'axe de la poutre."
            )
        e3 = e3 / norm_e3

        # --- e₂ = e₃ × e₁  (trièdre direct)
        e2 = np.cross(e3, e1)

        lam = np.stack([e1, e2, e3])   # (3, 3), lignes = vecteurs de base locaux
        return L, lam

    @staticmethod
    def _rotation_matrix(lam: np.ndarray) -> np.ndarray:
        """Matrice de rotation 12×12 local → global.

        T = block_diag(T_n1, T_n2)  avec  T_nœud = [[λ  0], [0  λ]]

        u_local = T · u_global   →   K_global = Tᵀ · K_local · T

        Parameters
        ----------
        lam : np.ndarray, shape (3, 3)
            Matrice de cosinus directeurs (lignes = axes locaux dans global).

        Returns
        -------
        T : np.ndarray, shape (12, 12)
        """
        T = np.zeros((12, 12))
        for n in range(2):          # nœud 0 et nœud 1
            off = 6 * n
            # bloc translations (off:off+3, off:off+3)
            T[off:off+3, off:off+3] = lam
            # bloc rotations  (off+3:off+6, off+3:off+6) — même λ
            T[off+3:off+6, off+3:off+6] = lam
        return T

    # ------------------------------------------------------------------
    # Extraction des propriétés de section
    # ------------------------------------------------------------------

    @staticmethod
    def _beam_props(
        material: ElasticMaterial,
        properties: dict,
    ) -> tuple[float, float, float, float, float, float]:
        """Lit la section et retourne les rigidités scalaires.

        Parameters
        ----------
        material : ElasticMaterial
        properties : dict
            Doit contenir ``"section"``.  Les clés ``"kappa_y"`` et
            ``"kappa_z"`` permettent de surcharger le facteur de
            cisaillement de la section pour chaque plan.

        Returns
        -------
        EA   : float  — rigidité axiale [N]
        EIy  : float  — flexion en xz (Iy = ∫z²dA) [N·m²]
        EIz  : float  — flexion en xy (Iz = ∫y²dA) [N·m²]
        GJ   : float  — rigidité de torsion [N·m²]
        GAsy : float  — rigidité de cisaillement plan xy [N]
        GAsz : float  — rigidité de cisaillement plan xz [N]
        """
        if "section" not in properties:
            raise KeyError(
                "Beam3D : 'section' (objet Section) est requis dans properties."
            )
        sec: Section = properties["section"]
        kappa = sec.shear_correction_factor(material.nu)
        kappa_y = float(properties.get("kappa_y", kappa))
        kappa_z = float(properties.get("kappa_z", kappa))

        EA   = material.E * sec.area
        EIy  = material.E * sec.Iy
        EIz  = material.E * sec.Iz
        GJ   = material.G * sec.J
        GAsy = material.G * kappa_y * sec.area
        GAsz = material.G * kappa_z * sec.area
        return EA, EIy, EIz, GJ, GAsy, GAsz

    # ------------------------------------------------------------------
    # Matrice de rigidité locale
    # ------------------------------------------------------------------

    @staticmethod
    def _stiffness_local(
        EA: float,
        EIy: float,
        EIz: float,
        GJ: float,
        GAsy: float,
        GAsz: float,
        L: float,
    ) -> np.ndarray:
        """Matrice de rigidité 12×12 en repère local (Timoshenko exact).

        Parameters
        ----------
        EA, EIy, EIz, GJ, GAsy, GAsz : float
            Rigidités scalaires (voir :meth:`_beam_props`).
        L : float
            Longueur [m].

        Returns
        -------
        K_local : np.ndarray, shape (12, 12)

        Notes
        -----
        Le signe négatif sur les couplages uz–θy (plan xz) résulte de la
        convention règle de la main droite pour θy : rotation positive θy
        autour de +ŷ fait tourner +x̂ vers −ẑ, soit duz/dx négatif.
        Les blocs xy et xz sont donc différents :

        Plan xy (coupling positif, θz = +duy/dx) :
            K[uz₁, θy₁] = −6·kz·L   ← signe − imposé par la RMD

        Plan xz (coupling négatif, θy = −duz/dx) :
            K[uy₁, θz₁] = +6·ky·L   ← signe + (convention positive pour θz)

        Références : Przemieniecki (1968) eq. 5.62 ;
                     Cook et al. (2002) eq. 5.7-1.
        """
        K = np.zeros((12, 12))
        L2 = L * L
        L3 = L2 * L

        # ── Axial : DDL 0 (ux₁) et 6 (ux₂) ─────────────────────────────────
        a = EA / L
        K[0, 0] =  a;  K[0, 6] = -a
        K[6, 0] = -a;  K[6, 6] =  a

        # ── Torsion : DDL 3 (θx₁) et 9 (θx₂) ───────────────────────────────
        t = GJ / L
        K[3, 3] =  t;  K[3, 9] = -t
        K[9, 3] = -t;  K[9, 9] =  t

        # ── Plan xy : DDL 1(uy₁), 5(θz₁), 7(uy₂), 11(θz₂) ─────────────────
        # Paramètre de Timoshenko : Φy = 12·EIz / (GAsy·L²)
        # Coupling positif (θz = +duy/dx convention)
        Phi_y = 12.0 * EIz / (GAsy * L2)
        ky = EIz / ((1.0 + Phi_y) * L3)

        ky12  = 12.0 * ky
        ky6L  =  6.0 * ky * L
        ky4p  = (4.0 + Phi_y) * ky * L2
        ky2m  = (2.0 - Phi_y) * ky * L2

        iy = [1, 5, 7, 11]          # indices globaux des DDL du plan xy
        Ky = np.array([
            [ ky12,  ky6L, -ky12,  ky6L],
            [ ky6L,  ky4p, -ky6L,  ky2m],
            [-ky12, -ky6L,  ky12, -ky6L],
            [ ky6L,  ky2m, -ky6L,  ky4p],
        ])
        for i, gi in enumerate(iy):
            for j, gj in enumerate(iy):
                K[gi, gj] = Ky[i, j]

        # ── Plan xz : DDL 2(uz₁), 4(θy₁), 8(uz₂), 10(θy₂) ─────────────────
        # Paramètre de Timoshenko : Φz = 12·EIy / (GAsz·L²)
        # Coupling négatif (θy = −duz/dx par règle de la main droite)
        Phi_z = 12.0 * EIy / (GAsz * L2)
        kz = EIy / ((1.0 + Phi_z) * L3)

        kz12  = 12.0 * kz
        kz6L  =  6.0 * kz * L
        kz4p  = (4.0 + Phi_z) * kz * L2
        kz2m  = (2.0 - Phi_z) * kz * L2

        iz = [2, 4, 8, 10]          # indices globaux des DDL du plan xz
        Kz = np.array([
            [ kz12, -kz6L, -kz12, -kz6L],
            [-kz6L,  kz4p,  kz6L,  kz2m],
            [-kz12,  kz6L,  kz12,  kz6L],
            [-kz6L,  kz2m,  kz6L,  kz4p],
        ])
        for i, gi in enumerate(iz):
            for j, gj in enumerate(iz):
                K[gi, gj] = Kz[i, j]

        return K

    # ------------------------------------------------------------------
    # Matrice de masse locale (Friedman–Kosmatka étendue en 3D)
    # ------------------------------------------------------------------

    @staticmethod
    def _mass_local(
        rho_A: float,
        rho_Iy: float,
        rho_Iz: float,
        rho_Ip: float,
        Phi_y: float,
        Phi_z: float,
        L: float,
    ) -> np.ndarray:
        """Matrice de masse consistante 12×12 en repère local.

        Inclut les inerties de rotation ρ·Iy et ρ·Iz (importantes pour
        poutres courtes ou hautes fréquences). Formulation Friedman–Kosmatka
        (1993) pour chaque plan de flexion, masse consistante pour axial
        et torsion.

        Parameters
        ----------
        rho_A  : float — ρ·A [kg/m] (masse linéique)
        rho_Iy : float — ρ·Iy [kg·m] (inertie de rotation, plan xz)
        rho_Iz : float — ρ·Iz [kg·m] (inertie de rotation, plan xy)
        rho_Ip : float — ρ·Ip [kg·m] (torsion, Ip = Iy + Iz)
        Phi_y  : float — paramètre de cisaillement plan xy
        Phi_z  : float — paramètre de cisaillement plan xz
        L      : float — longueur [m]

        Returns
        -------
        M_local : np.ndarray, shape (12, 12)

        Notes
        -----
        Les signes des couplages uz–θy dans le bloc xz sont NÉGATIFS,
        cohérents avec la convention θy = −duz/dx (règle main droite).

        Références : Friedman & Kosmatka (1993), CMAME 105, Table 1.
        """
        M = np.zeros((12, 12))
        L2 = L * L

        # ── Axial : DDL 0 (ux₁) et 6 (ux₂) — consistante linéaire ──────────
        ma = rho_A * L / 6.0
        M[0, 0] = 2.0 * ma;  M[0, 6] = ma
        M[6, 0] = ma;         M[6, 6] = 2.0 * ma

        # ── Torsion : DDL 3 (θx₁) et 9 (θx₂) — consistante linéaire ────────
        mt = rho_Ip * L / 6.0
        M[3, 3] = 2.0 * mt;  M[3, 9] = mt
        M[9, 3] = mt;         M[9, 9] = 2.0 * mt

        # ── Plans de flexion — Friedman–Kosmatka (1993) ───────────────────────
        # On calcule les deux blocs 4×4 (xy et xz) avec la même routine,
        # puis on les insère aux bons indices.  Le bloc xz a les couplages
        # uz–θy avec un signe opposé (−s sur off-diag rho_I).

        def _fk_block(
            rho_a: float,
            rho_i: float,
            Phi: float,
            sign_ri: float,
        ) -> np.ndarray:
            """Bloc de flexion 4×4 Friedman–Kosmatka.

            Parameters
            ----------
            rho_a  : masse linéique [kg/m]
            rho_i  : inertie de rotation [kg·m]
            Phi    : paramètre de cisaillement Φ
            sign_ri : signe (+1 ou -1) à appliquer aux couplages
                      déplacement–rotation de la partie ρI.
                      +1 pour le plan xy, −1 pour le plan xz.

            Returns
            -------
            Mb : np.ndarray, shape (4, 4)
                Dans l'ordre {u₁, θ₁, u₂, θ₂} (u = déplacement transverse,
                θ = rotation de section).
            """
            denom = (1.0 + Phi) ** 2
            m_t = rho_a * L / denom
            m_r = rho_i / (L * denom)

            P = Phi;  P2 = Phi * Phi

            t11 = 13.0/35.0 + 7.0*P/10.0 + P2/3.0
            t12 = (11.0/210.0 + 11.0*P/120.0 + P2/24.0) * L
            t13 = 9.0/70.0 + 3.0*P/10.0 + P2/6.0
            t14 = -(13.0/420.0 + 3.0*P/40.0 + P2/24.0) * L
            t22 = (1.0/105.0 + P/60.0 + P2/120.0) * L2
            t23 = (13.0/420.0 + 3.0*P/40.0 + P2/24.0) * L
            t24 = -(1.0/140.0 + P/60.0 + P2/120.0) * L2

            r11 = 6.0/5.0
            r12 = (1.0/10.0 - P/2.0) * L
            r22 = (2.0/15.0 + P/6.0 + P2/3.0) * L2
            r24 = -(1.0/30.0 - P/6.0 + P2/6.0) * L2

            # Matrice 4×4 symétrique
            Mb = np.zeros((4, 4))

            # Colonne 0 (u₁) — translationnel seul pour M_T, couplé pour M_R
            Mb[0, 0] = m_t * t11 + m_r * r11
            Mb[1, 0] = sign_ri * (m_t * t12 + m_r * r12)
            Mb[2, 0] = m_t * t13 - m_r * r11
            Mb[3, 0] = sign_ri * (m_t * t14 + m_r * r12)

            # Colonne 1 (θ₁)
            Mb[0, 1] = Mb[1, 0]
            Mb[1, 1] = m_t * t22 + m_r * r22
            Mb[2, 1] = sign_ri * (m_t * t23 - m_r * r12)
            Mb[3, 1] = m_t * t24 + m_r * r24

            # Colonne 2 (u₂)
            Mb[0, 2] = Mb[2, 0]
            Mb[1, 2] = Mb[2, 1]
            Mb[2, 2] = m_t * t11 + m_r * r11
            Mb[3, 2] = -sign_ri * (m_t * t12 + m_r * r12)

            # Colonne 3 (θ₂)
            Mb[0, 3] = Mb[3, 0]
            Mb[1, 3] = Mb[3, 1]
            Mb[2, 3] = Mb[3, 2]
            Mb[3, 3] = m_t * t22 + m_r * r22

            return Mb

        # Bloc xy : {uy₁(1), θz₁(5), uy₂(7), θz₂(11)} — sign_ri = +1
        Mxy = _fk_block(rho_A, rho_Iz, Phi_y, sign_ri=+1.0)
        iy = [1, 5, 7, 11]
        for i, gi in enumerate(iy):
            for j, gj in enumerate(iy):
                M[gi, gj] = Mxy[i, j]

        # Bloc xz : {uz₁(2), θy₁(4), uz₂(8), θy₂(10)} — sign_ri = −1
        Mxz = _fk_block(rho_A, rho_Iy, Phi_z, sign_ri=-1.0)
        iz = [2, 4, 8, 10]
        for i, gi in enumerate(iz):
            for j, gj in enumerate(iz):
                M[gi, gj] = Mxz[i, j]

        return M

    # ------------------------------------------------------------------
    # Matrice d'offset (liaison rigide excentrée)
    # ------------------------------------------------------------------

    @staticmethod
    def _offset_transform(
        offset_i: np.ndarray | None,
        offset_j: np.ndarray | None,
    ) -> np.ndarray | None:
        """Matrice de transformation d'offset 12×12, ou None si pas d'offset.

        Parameters
        ----------
        offset_i : np.ndarray, shape (3,) or None
            Vecteur d'excentrement a_i en repère global au nœud i [m].
        offset_j : np.ndarray, shape (3,) or None
            Vecteur d'excentrement a_j en repère global au nœud j [m].

        Returns
        -------
        T_off : np.ndarray, shape (12, 12) or None
            ``None`` si les deux offsets sont nuls (optimisation).

        Notes
        -----
        La liaison rigide transforme les DDL d'un nœud (u_node) vers
        les DDL du point de connexion de la poutre (u_beam) :

            u_beam = G · u_node   avec  G = [[I,  −ã],
                                            [0,   I ]]

        où ã = skew(a) est la matrice antisymétrique de a.

            K_offset = T_offᵀ · K_local · T_off

        Les forces nodales équivalentes à un chargement distribué sur la
        poutre excentrée sont aussi transformées par T_off.
        """
        def _G(a: np.ndarray) -> np.ndarray:
            """Matrice de transformation rigide 6×6 pour un offset a."""
            G = np.eye(6)
            # Sous-bloc (0:3, 3:6) = −ã
            ax, ay, az = a
            G[0, 4] =  az    # ux déplacé par +az·θy
            G[0, 5] = -ay    # ux déplacé par −ay·θz
            G[1, 3] = -az    # uy déplacé par −az·θx
            G[1, 5] =  ax    # uy déplacé par +ax·θz
            G[2, 3] =  ay    # uz déplacé par +ay·θx
            G[2, 4] = -ax    # uz déplacé par −ax·θy
            return G

        ai = np.zeros(3) if offset_i is None else np.asarray(offset_i, dtype=float)
        aj = np.zeros(3) if offset_j is None else np.asarray(offset_j, dtype=float)

        if np.allclose(ai, 0.0) and np.allclose(aj, 0.0):
            return None    # pas d'offset → pas de transformation

        T_off = np.zeros((12, 12))
        T_off[0:6,  0:6 ] = _G(ai)
        T_off[6:12, 6:12] = _G(aj)
        return T_off

    # ------------------------------------------------------------------
    # Interface publique
    # ------------------------------------------------------------------

    def stiffness_matrix(
        self,
        material: ElasticMaterial,
        nodes: np.ndarray,
        properties: dict,
    ) -> np.ndarray:
        """Matrice de rigidité élémentaire 12×12 en repère global.

        Parameters
        ----------
        material : ElasticMaterial
            Matériau (E, nu, G).
        nodes : np.ndarray, shape (2, 3)
            Coordonnées 3D [[x₁,y₁,z₁], [x₂,y₂,z₂]].
        properties : dict
            Voir le module docstring.  Clé obligatoire : ``"section"``.

        Returns
        -------
        K_e : np.ndarray, shape (12, 12)
            Matrice de rigidité symétrique en repère global.

        Examples
        --------
        >>> import numpy as np
        >>> from femsolver.core.material import ElasticMaterial
        >>> from femsolver.core.sections import RectangularSection
        >>> mat = ElasticMaterial(E=210e9, nu=0.3, rho=7800)
        >>> sec = RectangularSection(width=0.1, height=0.2)
        >>> nodes = np.array([[0., 0., 0.], [1., 0., 0.]])
        >>> K = Beam3D().stiffness_matrix(mat, nodes, {"section": sec})
        >>> K.shape
        (12, 12)
        """
        v_vec   = properties.get("v_vec",    None)
        off_i   = properties.get("offset_i", None)
        off_j   = properties.get("offset_j", None)

        L, lam = self._local_frame(nodes, v_vec)
        EA, EIy, EIz, GJ, GAsy, GAsz = self._beam_props(material, properties)
        K_loc = self._stiffness_local(EA, EIy, EIz, GJ, GAsy, GAsz, L)

        T = self._rotation_matrix(lam)
        K_e = T.T @ K_loc @ T

        T_off = self._offset_transform(off_i, off_j)
        if T_off is not None:
            K_e = T_off.T @ K_e @ T_off

        return K_e

    def mass_matrix(
        self,
        material: ElasticMaterial,
        nodes: np.ndarray,
        properties: dict,
    ) -> np.ndarray:
        """Matrice de masse consistante 12×12 en repère global.

        Inclut les inerties de rotation ρ·Iy et ρ·Iz (Friedman–Kosmatka 1993).

        Parameters
        ----------
        material : ElasticMaterial
        nodes : np.ndarray, shape (2, 3)
        properties : dict

        Returns
        -------
        M_e : np.ndarray, shape (12, 12)
        """
        v_vec = properties.get("v_vec", None)
        off_i = properties.get("offset_i", None)
        off_j = properties.get("offset_j", None)

        L, lam = self._local_frame(nodes, v_vec)
        sec: Section = properties["section"]
        EA, EIy, EIz, GJ, GAsy, GAsz = self._beam_props(material, properties)

        rho_A  = material.rho * sec.area
        rho_Iy = material.rho * sec.Iy
        rho_Iz = material.rho * sec.Iz
        rho_Ip = rho_Iy + rho_Iz           # moment polaire = Iy + Iz

        L2 = L * L
        Phi_y = 12.0 * EIz / (GAsy * L2)
        Phi_z = 12.0 * EIy / (GAsz * L2)

        M_loc = self._mass_local(rho_A, rho_Iy, rho_Iz, rho_Ip, Phi_y, Phi_z, L)

        T = self._rotation_matrix(lam)
        M_e = T.T @ M_loc @ T

        T_off = self._offset_transform(off_i, off_j)
        if T_off is not None:
            M_e = T_off.T @ M_e @ T_off

        return M_e

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
        """Efforts internes en repère local aux deux extrémités.

        Parameters
        ----------
        material : ElasticMaterial
        nodes : np.ndarray, shape (2, 3)
        properties : dict
        u_e : np.ndarray, shape (12,)
            Déplacements globaux.

        Returns
        -------
        dict avec les clés :
            ``N1``, ``Vy1``, ``Vz1``, ``Tx1``, ``My1``, ``Mz1`` — au nœud 1.
            ``N2``, ``Vy2``, ``Vz2``, ``Tx2``, ``My2``, ``Mz2`` — au nœud 2.

        Notes
        -----
        Calcul en repère local : f_loc = K_loc · (T · u_e).
        Les efforts sont ceux de la poutre (convention poutres : N positif
        en traction, Mz positif en sagitta, etc.).
        """
        v_vec = properties.get("v_vec", None)
        L, lam = self._local_frame(nodes, v_vec)
        EA, EIy, EIz, GJ, GAsy, GAsz = self._beam_props(material, properties)
        K_loc = self._stiffness_local(EA, EIy, EIz, GJ, GAsy, GAsz, L)
        T = self._rotation_matrix(lam)
        u_local = T @ u_e
        f_local = K_loc @ u_local   # (12,)
        return {
            "N1":  float(f_local[0]),
            "Vy1": float(f_local[1]),
            "Vz1": float(f_local[2]),
            "Tx1": float(f_local[3]),
            "My1": float(f_local[4]),
            "Mz1": float(f_local[5]),
            "N2":  float(f_local[6]),
            "Vy2": float(f_local[7]),
            "Vz2": float(f_local[8]),
            "Tx2": float(f_local[9]),
            "My2": float(f_local[10]),
            "Mz2": float(f_local[11]),
        }

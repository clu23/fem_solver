"""Élément quadrilatère bilinéaire Q4 isoparamétrique à 4 nœuds."""

from __future__ import annotations

import numpy as np

from femsolver.core.element import Element
from femsolver.core.material import ElasticMaterial


# ---------------------------------------------------------------------------
# Points et poids de Gauss 2×2 (intégration exacte jusqu'au degré 3)
# ---------------------------------------------------------------------------

_GP = 1.0 / np.sqrt(3.0)  # ±1/√3

# 4 points de Gauss : (ξ, η, poids)
_GAUSS_POINTS_2X2: list[tuple[float, float, float]] = [
    (-_GP, -_GP, 1.0),
    ( _GP, -_GP, 1.0),
    ( _GP,  _GP, 1.0),
    (-_GP,  _GP, 1.0),
]


class Quad4(Element):
    """Quadrilatère bilinéaire Q4 isoparamétrique — état plan.

    4 nœuds, 2 DDL par nœud (ux, uy) → matrices élémentaires 8×8.
    Intégration numérique 2×2 points de Gauss.

    Numérotation des nœuds (sens trigonométrique) :

        4 ------- 3
        |         |
        |         |
        1 ------- 2

    Coordonnées naturelles : (ξ, η) ∈ [-1, 1]².

    Ordre des DDL : [u1, v1, u2, v2, u3, v3, u4, v4].

    Parameters (via ``properties``)
    --------------------------------
    thickness : float
        Épaisseur de la plaque [m].
    formulation : str
        ``"plane_stress"`` (défaut) ou ``"plane_strain"``.

    References
    ----------
    Cook et al., « Concepts and Applications of FEA », 4th ed., chap. 6–7.
    Hughes, « The Finite Element Method », chap. 3.
    """

    def dof_per_node(self) -> int:
        """2 DDL par nœud : ux et uy."""
        return 2

    def n_nodes(self) -> int:
        """4 nœuds."""
        return 4

    @staticmethod
    def _shape_functions(xi: float, eta: float) -> np.ndarray:
        """Fonctions de forme N1…N4 au point (ξ, η).

        Parameters
        ----------
        xi, eta : float
            Coordonnées naturelles dans [-1, 1].

        Returns
        -------
        N : np.ndarray, shape (4,)
            [N1, N2, N3, N4].

        Notes
        -----
        N1 = ¼(1-ξ)(1-η)   N2 = ¼(1+ξ)(1-η)
        N3 = ¼(1+ξ)(1+η)   N4 = ¼(1-ξ)(1+η)
        """
        return 0.25 * np.array([
            (1 - xi) * (1 - eta),
            (1 + xi) * (1 - eta),
            (1 + xi) * (1 + eta),
            (1 - xi) * (1 + eta),
        ])

    @staticmethod
    def _shape_function_derivatives(xi: float, eta: float) -> np.ndarray:
        """Dérivées ∂Ni/∂ξ et ∂Ni/∂η au point (ξ, η).

        Parameters
        ----------
        xi, eta : float
            Coordonnées naturelles.

        Returns
        -------
        dN : np.ndarray, shape (2, 4)
            dN[0, i] = ∂Ni/∂ξ,  dN[1, i] = ∂Ni/∂η.

        Notes
        -----
        ∂N1/∂ξ = -¼(1-η)   ∂N1/∂η = -¼(1-ξ)
        ∂N2/∂ξ =  ¼(1-η)   ∂N2/∂η = -¼(1+ξ)
        ∂N3/∂ξ =  ¼(1+η)   ∂N3/∂η =  ¼(1+ξ)
        ∂N4/∂ξ = -¼(1+η)   ∂N4/∂η =  ¼(1-ξ)
        """
        return 0.25 * np.array([
            [-(1 - eta),  (1 - eta), (1 + eta), -(1 + eta)],  # ∂N/∂ξ
            [-(1 - xi), -(1 + xi),  (1 + xi),   (1 - xi)],   # ∂N/∂η
        ])

    @staticmethod
    def _jacobian(dN: np.ndarray, nodes: np.ndarray) -> np.ndarray:
        """Matrice Jacobienne J = dN · nodes (2×2).

        Parameters
        ----------
        dN : np.ndarray, shape (2, 4)
            Dérivées des fonctions de forme en (ξ, η).
        nodes : np.ndarray, shape (4, 2)
            Coordonnées physiques [[x1,y1], …, [x4,y4]].

        Returns
        -------
        J : np.ndarray, shape (2, 2)
            J[i,j] = ∂xⱼ/∂ξᵢ  avec ξ0=ξ, ξ1=η et x0=x, x1=y.

        Raises
        ------
        ValueError
            Si le Jacobien est singulier (élément dégénéré).
        """
        J = dN @ nodes  # (2,4) @ (4,2) → (2,2)
        if abs(np.linalg.det(J)) < 1e-14:
            raise ValueError(
                f"Jacobien singulier pour Quad4 : det(J) ≈ 0.\n"
                f"Vérifier la numérotation et les coordonnées des nœuds :\n{nodes}"
            )
        return J

    def _strain_displacement_matrix(
        self,
        xi: float,
        eta: float,
        nodes: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Matrice B (3×8) et det(J) au point de Gauss (ξ, η).

        Parameters
        ----------
        xi, eta : float
            Coordonnées naturelles du point de Gauss.
        nodes : np.ndarray, shape (4, 2)
            Coordonnées physiques.

        Returns
        -------
        B : np.ndarray, shape (3, 8)
            Matrice déformation–déplacement au point (ξ, η).
        det_J : float
            Déterminant du Jacobien (> 0 si nœuds en sens trigo).

        Notes
        -----
        ε = B · u_e  avec u_e = [u1,v1,u2,v2,u3,v3,u4,v4]ᵀ

        Les dérivées physiques s'obtiennent par :
            [∂Ni/∂x, ∂Ni/∂y]ᵀ = J⁻¹ · [∂Ni/∂ξ, ∂Ni/∂η]ᵀ

        B = [ ∂N1/∂x    0    ∂N2/∂x    0    ∂N3/∂x    0    ∂N4/∂x    0   ]
            [    0    ∂N1/∂y    0    ∂N2/∂y    0    ∂N3/∂y    0    ∂N4/∂y ]
            [ ∂N1/∂y ∂N1/∂x ∂N2/∂y ∂N2/∂x ∂N3/∂y ∂N3/∂x ∂N4/∂y ∂N4/∂x ]
        """
        dN = self._shape_function_derivatives(xi, eta)      # (2, 4)
        J = self._jacobian(dN, nodes)                        # (2, 2)
        det_J = np.linalg.det(J)
        dN_phys = np.linalg.solve(J, dN)                    # J⁻¹ · dN → (2, 4)

        # dN_phys[0, i] = ∂Ni/∂x,  dN_phys[1, i] = ∂Ni/∂y
        B = np.zeros((3, 8))
        for i in range(4):
            col = 2 * i
            B[0, col]     = dN_phys[0, i]   # ∂Ni/∂x → εxx
            B[1, col + 1] = dN_phys[1, i]   # ∂Ni/∂y → εyy
            B[2, col]     = dN_phys[1, i]   # ∂Ni/∂y → γxy (ligne u)
            B[2, col + 1] = dN_phys[0, i]   # ∂Ni/∂x → γxy (ligne v)

        return B, det_J

    def _elasticity_matrix(
        self, material: ElasticMaterial, formulation: str
    ) -> np.ndarray:
        """Sélectionne D selon la formulation."""
        if formulation == "plane_stress":
            return material.elasticity_matrix_plane_stress()
        if formulation == "plane_strain":
            return material.elasticity_matrix_plane_strain()
        raise ValueError(
            f"formulation doit être 'plane_stress' ou 'plane_strain', "
            f"reçu '{formulation}'"
        )

    def stiffness_matrix(
        self,
        material: ElasticMaterial,
        nodes: np.ndarray,
        properties: dict,
    ) -> np.ndarray:
        """Matrice de rigidité élémentaire 8×8.

        Parameters
        ----------
        material : ElasticMaterial
            Matériau (E, nu utilisés).
        nodes : np.ndarray, shape (4, 2)
            Coordonnées [[x1,y1], [x2,y2], [x3,y3], [x4,y4]].
        properties : dict
            ``"thickness"`` : épaisseur [m].
            ``"formulation"`` : ``"plane_stress"`` (défaut) ou ``"plane_strain"``.

        Returns
        -------
        K_e : np.ndarray, shape (8, 8)
            Matrice de rigidité symétrique.

        Notes
        -----
        Intégration 2×2 Gauss :

            K_e = t · Σₚ wₚ · Bₚᵀ · D · Bₚ · |det Jₚ|

        Exacte pour des quads rectangulaires, très précise pour des quads peu
        distordus (recommandation : ratio d'aspect < 5).

        Examples
        --------
        >>> import numpy as np
        >>> from femsolver.core.material import ElasticMaterial
        >>> mat = ElasticMaterial(E=1.0, nu=0.0, rho=1.0)
        >>> nodes = np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.]])
        >>> K_e = Quad4().stiffness_matrix(mat, nodes, {"thickness": 1.0})
        >>> K_e.shape
        (8, 8)
        """
        t = properties["thickness"]
        formulation = properties.get("formulation", "plane_stress")

        if t <= 0:
            raise ValueError(f"L'épaisseur doit être > 0, reçu thickness={t}")
        if nodes.shape != (4, 2):
            raise ValueError(f"Quad4 attend nodes.shape == (4, 2), reçu {nodes.shape}")

        D = self._elasticity_matrix(material, formulation)
        K_e = np.zeros((8, 8))

        for xi, eta, w in _GAUSS_POINTS_2X2:
            B, det_J = self._strain_displacement_matrix(xi, eta, nodes)
            K_e += w * (B.T @ D @ B) * det_J

        return K_e * t

    def mass_matrix(
        self,
        material: ElasticMaterial,
        nodes: np.ndarray,
        properties: dict,
    ) -> np.ndarray:
        """Matrice de masse consistante 8×8 par intégration 2×2 Gauss.

        Parameters
        ----------
        material : ElasticMaterial
            Matériau (rho utilisé).
        nodes : np.ndarray, shape (4, 2)
            Coordonnées nodales.
        properties : dict
            ``"thickness"`` : épaisseur [m].

        Returns
        -------
        M_e : np.ndarray, shape (8, 8)
            Matrice de masse consistante.

        Notes
        -----
        M_e = ρt · ∫∫ Nᵀ N |det J| dξ dη

        où N est la matrice d'interpolation 2×8 :
            N = [[N1, 0, N2, 0, N3, 0, N4, 0],
                 [0, N1,  0, N2,  0, N3,  0, N4]]

        Intégrée par quadrature 2×2 Gauss.
        """
        t = properties["thickness"]
        if nodes.shape != (4, 2):
            raise ValueError(f"Quad4 attend nodes.shape == (4, 2), reçu {nodes.shape}")

        M_e = np.zeros((8, 8))

        for xi, eta, w in _GAUSS_POINTS_2X2:
            Nv = self._shape_functions(xi, eta)   # shape (4,)
            dN = self._shape_function_derivatives(xi, eta)
            J = self._jacobian(dN, nodes)
            det_J = np.linalg.det(J)

            # Matrice N (2×8)
            N_mat = np.zeros((2, 8))
            for i in range(4):
                N_mat[0, 2 * i]     = Nv[i]
                N_mat[1, 2 * i + 1] = Nv[i]

            M_e += w * (N_mat.T @ N_mat) * det_J

        return M_e * (material.rho * t)

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
        """Matrices de rigidité pour N_e Quad4 en une seule passe tenseur.

        Remplace N_e appels à ``stiffness_matrix()`` par un calcul batch
        vectorisé sans boucle Python sur les éléments.

        Parameters
        ----------
        nodes_batch : np.ndarray, shape (N_e, 4, 2)
            Coordonnées nodales de tous les éléments du groupe.
        D : np.ndarray, shape (3, 3)
            Matrice d'élasticité (identique pour tous les éléments du groupe).
        t : float
            Épaisseur [m] (identique pour tous les éléments du groupe).

        Returns
        -------
        K_e_all : np.ndarray, shape (N_e, 8, 8)
            Matrices de rigidité élémentaires pour les N_e éléments.

        Notes
        -----
        Algorithme :

        1. ``dN_nat`` (4_gp, 2, 4) — dérivées en coordonnées naturelles,
           constantes (ne dépendent pas de la géométrie).
        2. Jacobiens batch : ``J = einsum('gin,enj->geij', dN_nat, nodes_batch)``
           → (4_gp, N_e, 2, 2).
        3. ``det_J``, ``J_inv`` via ``np.linalg.det/inv`` qui broadcast nativement.
        4. Dérivées physiques : ``dN_phys = einsum('geij,gjn->gein', J_inv, dN_nat)``
           → (4_gp, N_e, 2, 4).
        5. Matrices B (4_gp, N_e, 3, 8) par assignation vectorisée sur les 4 nœuds.
        6. ``K_e = t · einsum('g,ge,gepi,pq,geqj->eij', w, det_J, B, D, B)``
           → (N_e, 8, 8).
        """
        n_e = nodes_batch.shape[0]
        gp = _GP
        xi_eta_w = np.array([
            [-gp, -gp, 1.0],
            [ gp, -gp, 1.0],
            [ gp,  gp, 1.0],
            [-gp,  gp, 1.0],
        ])   # (4, 3)

        # dN/dξ, dN/dη aux 4 points de Gauss : (4, 2, 4) — constant
        dN_nat = np.stack([
            cls._shape_function_derivatives(xi, eta)
            for xi, eta, _ in xi_eta_w
        ])   # (4_gp, 2, 4)
        w = xi_eta_w[:, 2]   # (4,)

        # Jacobiens batch : J[g,e] = dN_nat[g] @ nodes_batch[e]
        # (4,2,4) × (N_e,4,2) → (4, N_e, 2, 2)
        J = np.einsum('gin,enj->geij', dN_nat, nodes_batch)
        det_J = np.linalg.det(J)        # (4, N_e)
        J_inv = np.linalg.inv(J)        # (4, N_e, 2, 2)

        # Dérivées physiques : dN_phys[g,e] = J_inv[g,e] @ dN_nat[g]
        # (4,N_e,2,2) × (4,2,4) → (4, N_e, 2, 4)
        dN_phys = np.einsum('geij,gjn->gein', J_inv, dN_nat)

        # Matrices B : (4_gp, N_e, 3, 8)
        B = np.zeros((4, n_e, 3, 8))
        for i in range(4):
            B[:, :, 0, 2 * i]     = dN_phys[:, :, 0, i]   # ∂Ni/∂x → εxx
            B[:, :, 1, 2 * i + 1] = dN_phys[:, :, 1, i]   # ∂Ni/∂y → εyy
            B[:, :, 2, 2 * i]     = dN_phys[:, :, 1, i]   # ∂Ni/∂y → γxy
            B[:, :, 2, 2 * i + 1] = dN_phys[:, :, 0, i]   # ∂Ni/∂x → γxy

        # K_e = t · Σ_g w_g · |det_J[g,e]| · B[g,e]^T · D · B[g,e]
        # einsum indices : g=Gauss, e=elem, p/q=stress(3), i/j=dof(8)
        K_e_all = np.einsum('g,ge,gepi,pq,geqj->eij', w, det_J, B, D, B) * t
        return K_e_all   # (N_e, 8, 8)

    @classmethod
    def batch_mass_matrix(
        cls,
        nodes_batch: np.ndarray,
        rho: float,
        t: float,
    ) -> np.ndarray:
        """Matrices de masse pour N_e Quad4 en une seule passe tenseur.

        Parameters
        ----------
        nodes_batch : np.ndarray, shape (N_e, 4, 2)
            Coordonnées nodales des N_e éléments.
        rho : float
            Masse volumique [kg/m³] (identique pour tout le groupe).
        t : float
            Épaisseur [m].

        Returns
        -------
        M_e_all : np.ndarray, shape (N_e, 8, 8)

        Notes
        -----
        NtN[g] = N_mat[g]^T @ N_mat[g] est une constante (ne dépend que
        de ξ,η, pas de la géométrie) : précomputation hors de toute boucle.

            M_e = ρ·t · einsum('g,ge,gij->eij', w, det_J, NtN)
        """
        gp = _GP
        xi_eta_w = np.array([
            [-gp, -gp, 1.0],
            [ gp, -gp, 1.0],
            [ gp,  gp, 1.0],
            [-gp,  gp, 1.0],
        ])
        w = xi_eta_w[:, 2]

        # NtN[g] = N_mat[g]^T @ N_mat[g] : (4, 8, 8) — constant, indépendant des nœuds
        NtN = np.zeros((4, 8, 8))
        for g, (xi, eta, _) in enumerate(xi_eta_w):
            Nv = cls._shape_functions(xi, eta)   # (4,)
            N_mat = np.zeros((2, 8))
            for i in range(4):
                N_mat[0, 2 * i]     = Nv[i]
                N_mat[1, 2 * i + 1] = Nv[i]
            NtN[g] = N_mat.T @ N_mat   # (8, 8)

        # Jacobiens batch (pour det_J uniquement — même calcul que batch_stiffness)
        dN_nat = np.stack([
            cls._shape_function_derivatives(xi, eta)
            for xi, eta, _ in xi_eta_w
        ])
        J = np.einsum('gin,enj->geij', dN_nat, nodes_batch)
        det_J = np.linalg.det(J)   # (4, N_e)

        # M_e = ρ·t · Σ_g w_g · det_J[g,e] · NtN[g]
        M_e_all = np.einsum('g,ge,gij->eij', w, det_J, NtN) * (rho * t)
        return M_e_all   # (N_e, 8, 8)

    def geometric_stiffness_matrix(
        self,
        material: ElasticMaterial,
        nodes: np.ndarray,
        properties: dict,
        u_e: np.ndarray,
    ) -> np.ndarray:
        """Matrice de rigidité géométrique 8×8 par intégration 2×2 Gauss.

        Pour un élément plan en état pré-contraint σ₀ = [σxx, σyy, τxy],
        la 2ᵉ variation de l'énergie potentielle donne :

            δ²W = t ∫∫ σᵢⱼ (∂δuₖ/∂xᵢ)(∂δuₖ/∂xⱼ) dA

        Cela se factorise comme :

            K_g = t ∫∫ Gᵀ σ̃ G dA

        où :
        - G = dN_phys (2×4) : dérivées physiques des fonctions de forme
          G[α, I] = ∂N_I/∂x_α
        - σ̃ = [[σxx, τxy], [τxy, σyy]] : tenseur de contraintes 2D

        Structure de K_g :
            K_g[2I+α, 2J+β] = δ_{αβ} × ∫∫ (∇N_I)ᵀ σ̃ (∇N_J) t dA

        Les composantes ux et uy sont couplées identiquement (pas de termes
        croisés ux–uy dans K_g pour les éléments 2D isoparamétriques).

        Parameters
        ----------
        material : ElasticMaterial
            Matériau (E, nu utilisés pour calculer σ = D B u_e).
        nodes : np.ndarray, shape (4, 2)
            Coordonnées nodales.
        properties : dict
            ``"thickness"`` et optionnellement ``"formulation"``.
        u_e : np.ndarray, shape (8,)
            Déplacements [u1, v1, u2, v2, u3, v3, u4, v4] (repère global).

        Returns
        -------
        K_g_e : np.ndarray, shape (8, 8)
            Matrice de rigidité géométrique symétrique.

        Notes
        -----
        La contrainte σ₀ est calculée à partir de u_e (état pré-flambement).
        Pour flambage sous compression uniaxiale σxx < 0, K_g < 0 dans les
        DDL transversaux → réduction de la rigidité apparente.

        Référence : Bathe, « FE Procedures », §6.3 ; Zienkiewicz & Taylor,
        vol. 2, §9.1.
        """
        t = properties["thickness"]
        formulation = properties.get("formulation", "plane_stress")
        D = self._elasticity_matrix(material, formulation)

        K_g = np.zeros((8, 8))
        idx = np.arange(4)

        for xi, eta, w in _GAUSS_POINTS_2X2:
            B, det_J = self._strain_displacement_matrix(xi, eta, nodes)
            sigma = D @ (B @ u_e)   # [σxx, σyy, τxy]
            sigma_tensor = np.array([
                [sigma[0], sigma[2]],
                [sigma[2], sigma[1]],
            ])

            dN = self._shape_function_derivatives(xi, eta)
            J = self._jacobian(dN, nodes)
            G = np.linalg.solve(J, dN)   # dN_phys : (2, 4)

            k_nodal = G.T @ sigma_tensor @ G   # (4, 4)
            contrib = w * det_J * k_nodal * t

            # K_g[2I+α, 2J+α] += k_IJ pour α ∈ {0, 1} (ux et uy identiques)
            K_g[np.ix_(2 * idx,     2 * idx    )] += contrib
            K_g[np.ix_(2 * idx + 1, 2 * idx + 1)] += contrib

        return K_g

    def stiffness_matrix_sri(
        self,
        material: ElasticMaterial,
        nodes: np.ndarray,
        properties: dict,
    ) -> np.ndarray:
        """Matrice de rigidité par intégration réduite sélective (SRI) 8×8.

        Élimine le *shear locking* en flexion sans introduire de modes
        hourglass en séparant D en deux contributions :

        * **D_dil** (partie dilatation/membrane — εxx, εyy) :
          intégrée avec 2×2 points de Gauss (ordre complet).  Cette partie
          garantit le rang correct de K_e et bloque les modes hourglass.

        * **D_dev** (partie déviatorique/cisaillement — γxy, G = D[2,2]) :
          intégrée avec **1 point** au centre (ξ=0, η=0), poids = 4.
          En flexion pure, γxy = 0 au centre d'un élément rectangulaire,
          donc la rigidité parasitaire de cisaillement est annulée.

        K_e = K_dil + K_dev

        avec ::

            K_dil = t · Σ_{2×2} wₚ · Bₚᵀ · D_dil · Bₚ · |det Jₚ|
            K_dev = t · 4 · Bc^T · D_dev · Bc · |det Jc|   (1 point central)

        Parameters
        ----------
        material : ElasticMaterial
            Matériau (E, nu).
        nodes : np.ndarray, shape (4, 2)
            Coordonnées nodales.
        properties : dict
            ``"thickness"`` : épaisseur [m].
            ``"formulation"`` : ``"plane_stress"`` (défaut) ou
            ``"plane_strain"``.

        Returns
        -------
        K_e : np.ndarray, shape (8, 8)
            Matrice de rigidité SRI symétrique.

        Notes
        -----
        Référence : Hughes, « The FEM », §4.5 (selective/reduced integration).
        Zienkiewicz & Taylor, vol. 1, §9.9 (méthode de split dilatation/déviatoire).

        Pour un état de contrainte uniforme (patch test), K_dil · u_e + K_dev · u_e
        donne les mêmes forces nodales que l'intégration complète, donc le
        patch test en traction et en cisaillement pur passe exactement.

        Examples
        --------
        >>> import numpy as np
        >>> from femsolver.core.material import ElasticMaterial
        >>> mat = ElasticMaterial(E=1.0, nu=0.0, rho=1.0)
        >>> nodes = np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.]])
        >>> K_sri = Quad4().stiffness_matrix_sri(mat, nodes, {"thickness": 1.0})
        >>> K_sri.shape
        (8, 8)
        """
        t = properties["thickness"]
        formulation = properties.get("formulation", "plane_stress")

        if t <= 0:
            raise ValueError(f"L'épaisseur doit être > 0, reçu thickness={t}")
        if nodes.shape != (4, 2):
            raise ValueError(f"Quad4 attend nodes.shape == (4, 2), reçu {nodes.shape}")

        D = self._elasticity_matrix(material, formulation)

        # --- Décomposition de D -------------------------------------------
        # D_dil : partie dilatation/membrane (εxx, εyy) — intégrée en 2×2
        D_dil = np.zeros((3, 3))
        D_dil[:2, :2] = D[:2, :2]

        # D_dev : partie déviatorique/cisaillement (γxy = G) — intégrée en 1 pt
        D_dev = np.zeros((3, 3))
        D_dev[2, 2] = D[2, 2]   # G = E/(2(1+ν)) pour plane_stress

        # --- Partie dilatation : intégration 2×2 ---------------------------
        K_dil = np.zeros((8, 8))
        for xi, eta, w in _GAUSS_POINTS_2X2:
            B, det_J = self._strain_displacement_matrix(xi, eta, nodes)
            K_dil += w * (B.T @ D_dil @ B) * det_J

        # --- Partie déviatorique : 1 point au centre (poids = 4) -----------
        B_c, det_J_c = self._strain_displacement_matrix(0.0, 0.0, nodes)
        K_dev = 4.0 * (B_c.T @ D_dev @ B_c) * det_J_c

        return (K_dil + K_dev) * t

    def body_force_vector(
        self,
        material: ElasticMaterial,
        nodes: np.ndarray,
        properties: dict,
        b: np.ndarray,
    ) -> np.ndarray:
        """Forces nodales équivalentes pour une force de volume uniforme b [N/m³].

        f_e = ρ · t · ∫∫ Nᵀ · b |det J| dξ dη   (intégration 2×2 Gauss)

        Parameters
        ----------
        material : ElasticMaterial
            Matériau (rho utilisé).
        nodes : np.ndarray, shape (4, 2)
            Coordonnées nodales.
        properties : dict
            ``"thickness"`` : épaisseur [m].
        b : np.ndarray, shape (2,)
            Force de volume [N/m³] = ρ · acceleration.

        Returns
        -------
        f_e : np.ndarray, shape (8,)
            Forces nodales équivalentes [N].

        Notes
        -----
        Pour un rectangle de dimensions a×b, chaque nœud reçoit exactement
        ρ·t·a·b/4 · b_vect (équipartition — vérifiable analytiquement).
        """
        t = properties.get("thickness", 1.0)
        f_e = np.zeros(8)
        for xi, eta, w in _GAUSS_POINTS_2X2:
            Nv = self._shape_functions(xi, eta)   # shape (4,)
            dN = self._shape_function_derivatives(xi, eta)
            J = self._jacobian(dN, nodes)
            det_J = np.linalg.det(J)
            # np.kron(Nv, b) crée [N1*bx, N1*by, N2*bx, N2*by, N3*bx, N3*by, N4*bx, N4*by]
            f_e += w * np.kron(Nv, b) * det_J
        return f_e * (material.rho * t)

    def strain(
        self,
        nodes: np.ndarray,
        u_e: np.ndarray,
        xi: float = 0.0,
        eta: float = 0.0,
    ) -> np.ndarray:
        """Vecteur de déformations ε = B · u_e au point (ξ, η).

        Parameters
        ----------
        nodes : np.ndarray, shape (4, 2)
            Coordonnées nodales.
        u_e : np.ndarray, shape (8,)
            Déplacements élémentaires.
        xi, eta : float
            Point d'évaluation en coordonnées naturelles (défaut : centre).

        Returns
        -------
        epsilon : np.ndarray, shape (3,)
            [εxx, εyy, γxy].
        """
        B, _ = self._strain_displacement_matrix(xi, eta, nodes)
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
        """Vecteur de contraintes σ = D · B · u_e au point (ξ, η).

        Parameters
        ----------
        material : ElasticMaterial
            Matériau.
        nodes : np.ndarray, shape (4, 2)
            Coordonnées nodales.
        u_e : np.ndarray, shape (8,)
            Déplacements élémentaires.
        xi, eta : float
            Point d'évaluation (défaut : centre de l'élément).
        formulation : str
            ``"plane_stress"`` ou ``"plane_strain"``.

        Returns
        -------
        sigma : np.ndarray, shape (3,)
            [σxx, σyy, τxy] [Pa].
        """
        D = self._elasticity_matrix(material, formulation)
        return D @ self.strain(nodes, u_e, xi, eta)

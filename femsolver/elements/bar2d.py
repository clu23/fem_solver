"""Élément barre 2D — treillis (effort axial uniquement)."""

from __future__ import annotations

import numpy as np

from femsolver.core.element import Element
from femsolver.core.material import ElasticMaterial


class Bar2D(Element):
    """Élément barre 2D à 2 nœuds pour les treillis plans.

    Chaque nœud possède 2 DDL (ux, uy) → matrice élémentaire 4×4.
    La barre ne transmet que des efforts axiaux (pas de flexion).

    La matrice de rigidité est obtenue analytiquement par rotation du
    repère local (axial) vers le repère global :

        K_e = T_bar^T · K_local · T_bar

    où T_bar = [[c, s, 0, 0], [0, 0, c, s]] et K_local = (EA/L)·[[1,-1],[-1,1]].

    Notes
    -----
    Ordre des DDL : [ux_1, uy_1, ux_2, uy_2].

    La propriété requise dans ``properties`` est ``"area"`` (section [m²]).

    Référence : Cook et al., « Concepts and Applications of FEA »,
    4th ed., chap. 2.
    """

    def dof_per_node(self) -> int:
        """2 DDL par nœud : ux et uy."""
        return 2

    def n_nodes(self) -> int:
        """2 nœuds."""
        return 2

    def _geometry(self, nodes: np.ndarray) -> tuple[float, float, float]:
        """Longueur et cosinus directeurs de la barre.

        Parameters
        ----------
        nodes : np.ndarray, shape (2, 2)
            Coordonnées [[x1, y1], [x2, y2]] des deux nœuds.

        Returns
        -------
        L : float
            Longueur de la barre [m].
        c : float
            cos θ (cosinus de l'angle avec l'axe x global).
        s : float
            sin θ (sinus de l'angle avec l'axe x global).

        Raises
        ------
        ValueError
            Si la longueur est nulle (nœuds confondus).
        """
        if nodes.shape != (2, 2):
            raise ValueError(
                f"Bar2D attend nodes.shape == (2, 2), reçu {nodes.shape}"
            )
        dx = nodes[1, 0] - nodes[0, 0]
        dy = nodes[1, 1] - nodes[0, 1]
        L = float(np.hypot(dx, dy))
        if L < 1e-14:
            raise ValueError(
                f"Longueur de barre nulle (nœuds confondus) : nodes={nodes}"
            )
        return L, dx / L, dy / L

    def _rotation_matrix(self, c: float, s: float) -> np.ndarray:
        """Matrice de rotation globale → locale 4×4.

        Transforme les déplacements globaux [ux1, uy1, ux2, uy2] en
        déplacements locaux [u_axial_1, u_transv_1, u_axial_2, u_transv_2].

        Construction
        ------------
        On place deux blocs R identiques sur la diagonale — un par nœud :

            R = [[c,  s],   (rotation 2D : projette (x,y) global sur (axial, transv) local)
                 [-s, c]]

            T = block_diag(R, R) = [[ c,  s,  0,  0],
                                    [-s,  c,  0,  0],
                                    [ 0,  0,  c,  s],
                                    [ 0,  0, -s,  c]]

        La ligne 0 projette (ux1, uy1) sur l'axe de la barre → u_axial_1 = c·ux1 + s·uy1.
        La ligne 1 projette sur la direction transversale → u_transv_1 = -s·ux1 + c·uy1.
        Les lignes 2-3 font de même pour le nœud 2.

        Parameters
        ----------
        c : float
            cos θ (θ = angle barre / axe x global).
        s : float
            sin θ.

        Returns
        -------
        T : np.ndarray, shape (4, 4)
            u_local = T @ u_global.
        """
        return np.array(
            [
                [ c,  s, 0.0, 0.0],
                [-s,  c, 0.0, 0.0],
                [0.0, 0.0,  c,  s],
                [0.0, 0.0, -s,  c],
            ]
        )

    def stiffness_matrix(
        self,
        material: ElasticMaterial,
        nodes: np.ndarray,
        properties: dict,
    ) -> np.ndarray:
        """Matrice de rigidité élémentaire 4×4 en repère global.

        Parameters
        ----------
        material : ElasticMaterial
            Matériau (E utilisé).
        nodes : np.ndarray, shape (2, 2)
            Coordonnées [[x1, y1], [x2, y2]] des nœuds.
        properties : dict
            Doit contenir ``"area"`` : section transversale [m²].

        Returns
        -------
        K_e : np.ndarray, shape (4, 4)
            Matrice de rigidité globale symétrique.

        Notes
        -----
        En repère local : K_local = (EA/L) · [[1, -1], [-1, 1]]

        En repère global (développement analytique) :

            K_e = (EA/L) · [[ c²,  cs, -c², -cs],
                             [ cs,  s², -cs, -s²],
                             [-c², -cs,  c²,  cs],
                             [-cs, -s²,  cs,  s²]]

        avec c = cos θ, s = sin θ.

        Examples
        --------
        Barre horizontale de 1 m, acier, section 1 cm² :

        >>> import numpy as np
        >>> from femsolver.core.material import ElasticMaterial
        >>> mat = ElasticMaterial(E=210e9, nu=0.3, rho=7800)
        >>> nodes = np.array([[0.0, 0.0], [1.0, 0.0]])
        >>> K_e = Bar2D().stiffness_matrix(mat, nodes, {"area": 1e-4})
        >>> K_e[0, 0]   # EA/L = 210e9 * 1e-4 / 1 = 21e6 N/m
        21000000.0
        """
        area = properties["area"]
        if area <= 0:
            raise ValueError(f"La section doit être > 0, reçu area={area}")

        L, c, s = self._geometry(nodes)
        k = material.E * area / L  # EA/L [N/m]
        T = self._rotation_matrix(c, s)

        # Rigidité axiale pure en repère local (pas de rigidité transversale pour une barre)
        K_local = k * np.array(
            [
                [ 1.0, 0.0, -1.0, 0.0],
                [ 0.0, 0.0,  0.0, 0.0],
                [-1.0, 0.0,  1.0, 0.0],
                [ 0.0, 0.0,  0.0, 0.0],
            ]
        )

        return T.T @ K_local @ T

    def mass_matrix(
        self,
        material: ElasticMaterial,
        nodes: np.ndarray,
        properties: dict,
    ) -> np.ndarray:
        """Matrice de masse consistante 4×4 en repère global.

        Parameters
        ----------
        material : ElasticMaterial
            Matériau (rho utilisé).
        nodes : np.ndarray, shape (2, 2)
            Coordonnées [[x1, y1], [x2, y2]] des nœuds.
        properties : dict
            Doit contenir ``"area"`` : section transversale [m²].

        Returns
        -------
        M_e : np.ndarray, shape (4, 4)
            Matrice de masse consistante.

        Notes
        -----
        Avec les fonctions de forme linéaires N1 = (1-ξ)/2, N2 = (1+ξ)/2,
        l'intégrale ρAL/2 · ∫₋₁¹ N^T N dξ donne :

            M_e = (ρAL/6) · [[2, 0, 1, 0],
                               [0, 2, 0, 1],
                               [1, 0, 2, 0],
                               [0, 1, 0, 2]]

        La masse est répartie également en x et y (isotrope),
        indépendamment de l'orientation de la barre.
        """
        area = properties["area"]
        if area <= 0:
            raise ValueError(f"La section doit être > 0, reçu area={area}")

        L, c, s = self._geometry(nodes)
        m = material.rho * area * L / 6.0  # ρAL/6
        T = self._rotation_matrix(c, s)

        # Masse consistante en repère local : la barre a de la masse dans les deux directions.
        # M_local = ρAL/6 · [[2,0,1,0],[0,2,0,1],[1,0,2,0],[0,1,0,2]]
        # (isotrope → M_local est invariante par rotation : T.T @ M_local @ T = M_local)
        M_local = m * np.array(
            [
                [2.0, 0.0, 1.0, 0.0],
                [0.0, 2.0, 0.0, 1.0],
                [1.0, 0.0, 2.0, 0.0],
                [0.0, 1.0, 0.0, 2.0],
            ]
        )

        return T.T @ M_local @ T

    def axial_force(
        self,
        material: ElasticMaterial,
        nodes: np.ndarray,
        area: float,
        u_e: np.ndarray,
    ) -> float:
        """Effort normal dans la barre à partir du champ de déplacements.

        Parameters
        ----------
        material : ElasticMaterial
            Matériau (E utilisé).
        nodes : np.ndarray, shape (2, 2)
            Coordonnées des nœuds en repère global.
        area : float
            Section transversale [m²].
        u_e : np.ndarray, shape (4,)
            Déplacements globaux [ux1, uy1, ux2, uy2].

        Returns
        -------
        N : float
            Effort normal [N]. Positif = traction, négatif = compression.

        Notes
        -----
        u_local = T_bar @ u_e  →  ε = (u2_local - u1_local) / L
        N = E·A·ε = (EA/L) · (u2_local - u1_local)
        """
        L, c, s = self._geometry(nodes)
        T = self._rotation_matrix(c, s)
        u_local = T @ u_e              # [u1_axial, u1_transv, u2_axial, u2_transv]
        delta = u_local[2] - u_local[0]  # allongement axial = u_axial_2 − u_axial_1 [m]
        return float(material.E * area / L * delta)

    def axial_stress(
        self,
        material: ElasticMaterial,
        nodes: np.ndarray,
        area: float,
        u_e: np.ndarray,
    ) -> float:
        """Contrainte axiale σ = N/A [Pa].

        Parameters
        ----------
        material : ElasticMaterial
            Matériau.
        nodes : np.ndarray, shape (2, 2)
            Coordonnées des nœuds.
        area : float
            Section transversale [m²].
        u_e : np.ndarray, shape (4,)
            Déplacements globaux de l'élément.

        Returns
        -------
        sigma : float
            Contrainte axiale [Pa]. Positif = traction.
        """
        return self.axial_force(material, nodes, area, u_e) / area

    def distributed_load_vector(
        self,
        material: ElasticMaterial,
        nodes: np.ndarray,
        properties: dict,
        qx: float,
        qy: float,
    ) -> np.ndarray:
        """Forces nodales équivalentes pour une charge axiale uniforme qx [N/m].

        Chaque nœud reçoit la moitié de la charge totale (intégrale exacte
        pour des fonctions de forme linéaires) :

            f_local = [qx·L/2, 0, qx·L/2, 0]ᵀ   (repère local)
            f_e     = Tᵀ · f_local                (repère global)

        Parameters
        ----------
        qx : float
            Charge axiale distribuée dans le repère local [N/m].
            Positif dans le sens nœud_1 → nœud_2.
        qy : float
            Doit être 0.  Bar2D ne supporte pas de charge transverse.

        Returns
        -------
        f_e : np.ndarray, shape (4,)
            Forces nodales équivalentes [N] en repère global.

        Raises
        ------
        ValueError
            Si qy ≠ 0.

        Notes
        -----
        Validation analytique — barre encastrée-libre (L, EA, q uniforme) :

            u_tip = q·L² / (2·EA)    exact avec 1 seul élément FEM.

        Examples
        --------
        >>> mat = ElasticMaterial(E=210e9, nu=0.3, rho=7800)
        >>> nodes = np.array([[0., 0.], [1., 0.]])
        >>> f = Bar2D().distributed_load_vector(mat, nodes, {}, qx=1000.0, qy=0.0)
        >>> f
        array([500.,   0., 500.,   0.])
        """
        if qy != 0.0:
            raise ValueError(
                f"Bar2D ne supporte pas de charge transverse (qy={qy!r} ≠ 0). "
                "Utilisez Beam2D pour les charges transverses."
            )
        L, c, s = self._geometry(nodes)
        f_local = np.array([qx * L / 2.0, 0.0, qx * L / 2.0, 0.0])
        T = self._rotation_matrix(c, s)
        return T.T @ f_local

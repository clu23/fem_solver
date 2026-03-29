"""Élément triangle CST à 3 nœuds (Constant Strain Triangle)."""

from __future__ import annotations

import numpy as np

from femsolver.core.element import Element
from femsolver.core.material import ElasticMaterial


class Tri3(Element):
    """Triangle CST — état plan de contraintes ou de déformations.

    3 nœuds, 2 DDL par nœud (ux, uy) → matrice élémentaire 6×6.
    Les déformations sont constantes dans l'élément (champ de déplacement
    linéaire), ce qui donne une intégration exacte sans quadrature de Gauss.

    Numérotation des nœuds (sens trigonométrique recommandé) :

        3
        |\\
        | \\
        |  \\
        1---2

    Ordre des DDL : [u1, v1, u2, v2, u3, v3].

    Parameters (via ``properties``)
    --------------------------------
    thickness : float
        Épaisseur de la plaque [m].
    formulation : str
        ``"plane_stress"`` (défaut) ou ``"plane_strain"``.

    References
    ----------
    Cook et al., « Concepts and Applications of FEA », 4th ed., chap. 6.
    Logan, « A First Course in FEM », chap. 6.
    """

    def dof_per_node(self) -> int:
        """2 DDL par nœud : ux et uy."""
        return 2

    def n_nodes(self) -> int:
        """3 nœuds."""
        return 3

    def _geometry(
        self, nodes: np.ndarray
    ) -> tuple[float, float, float, float, float, float, float]:
        """Aire et coefficients b, c à partir des coordonnées nodales.

        Parameters
        ----------
        nodes : np.ndarray, shape (3, 2)
            Coordonnées [[x1,y1], [x2,y2], [x3,y3]].

        Returns
        -------
        area : float
            Aire du triangle [m²] (> 0 si nœuds en sens trigonométrique).
        b1, b2, b3 : float
            Coefficients b_i = y_j - y_k (dérivées des fonctions de forme / x).
        c1, c2, c3 : float
            Coefficients c_i = x_k - x_j (dérivées des fonctions de forme / y).

        Raises
        ------
        ValueError
            Si l'aire est nulle (nœuds colinéaires).

        Notes
        -----
        Indices cycliques : (i,j,k) = (1,2,3), (2,3,1), (3,1,2).

        b1 = y2 - y3,  b2 = y3 - y1,  b3 = y1 - y2
        c1 = x3 - x2,  c2 = x1 - x3,  c3 = x2 - x1

        Aire signée = ½ det([[x1,y1,1],[x2,y2,1],[x3,y3,1]])
                    = ½ (b1·(x2-x1) + ...) = ½ (b1·c2 - b2·c1)
        """
        if nodes.shape != (3, 2):
            raise ValueError(
                f"Tri3 attend nodes.shape == (3, 2), reçu {nodes.shape}"
            )
        x1, y1 = nodes[0]
        x2, y2 = nodes[1]
        x3, y3 = nodes[2]

        b1, b2, b3 = y2 - y3, y3 - y1, y1 - y2
        c1, c2, c3 = x3 - x2, x1 - x3, x2 - x1

        # Aire signée (positive si nœuds en sens trigonométrique)
        area_signed = 0.5 * (b1 * (x2 - x1) - c1 * (y2 - y1))
        # Formule équivalente directe :
        area_signed = 0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

        if abs(area_signed) < 1e-14:
            raise ValueError(
                f"Aire nulle pour Tri3 — nœuds colinéaires : {nodes}"
            )

        # On travaille avec l'aire absolue ; b et c sont définis pour sens trigo
        # Si l'ordre est horaire, area_signed < 0 → on corrige le signe
        if area_signed < 0:
            b1, b2, b3 = -b1, -b2, -b3
            c1, c2, c3 = -c1, -c2, -c3
            area_signed = -area_signed

        return area_signed, b1, b2, b3, c1, c2, c3

    def _strain_displacement_matrix(
        self, nodes: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """Matrice B (3×6) et aire du triangle.

        Parameters
        ----------
        nodes : np.ndarray, shape (3, 2)
            Coordonnées nodales.

        Returns
        -------
        B : np.ndarray, shape (3, 6)
            Matrice déformation–déplacement. Constante sur l'élément.
        area : float
            Aire du triangle [m²].

        Notes
        -----
        ε = B · u_e  avec u_e = [u1, v1, u2, v2, u3, v3]ᵀ

        B = 1/(2A) · [ b1   0   b2   0   b3   0  ]
                     [  0  c1    0  c2    0  c3  ]
                     [ c1  b1   c2  b2   c3  b3  ]
        """
        area, b1, b2, b3, c1, c2, c3 = self._geometry(nodes)
        B = np.array(
            [
                [b1, 0.0, b2, 0.0, b3, 0.0],
                [0.0, c1, 0.0, c2, 0.0, c3],
                [c1, b1, c2, b2, c3, b3],
            ]
        ) / (2.0 * area)
        return B, area

    def _elasticity_matrix(
        self, material: ElasticMaterial, formulation: str
    ) -> np.ndarray:
        """Sélectionne la matrice D selon la formulation."""
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
        """Matrice de rigidité élémentaire 6×6.

        Parameters
        ----------
        material : ElasticMaterial
            Matériau (E, nu utilisés).
        nodes : np.ndarray, shape (3, 2)
            Coordonnées [[x1,y1], [x2,y2], [x3,y3]].
        properties : dict
            ``"thickness"`` : épaisseur [m].
            ``"formulation"`` : ``"plane_stress"`` (défaut) ou ``"plane_strain"``.

        Returns
        -------
        K_e : np.ndarray, shape (6, 6)
            Matrice de rigidité symétrique.

        Notes
        -----
        B est constant → K_e = Bᵀ D B · (A · t) (intégration exacte).

        Examples
        --------
        Triangle rectangle isocèle, côté 1 m, acier, contrainte plane :

        >>> import numpy as np
        >>> from femsolver.core.material import ElasticMaterial
        >>> mat = ElasticMaterial(E=1.0, nu=0.0, rho=1.0)
        >>> nodes = np.array([[0.,0.],[1.,0.],[0.,1.]])
        >>> K_e = Tri3().stiffness_matrix(mat, nodes, {"thickness": 1.0})
        >>> K_e.shape
        (6, 6)
        """
        t = properties["thickness"]
        formulation = properties.get("formulation", "plane_stress")

        if t <= 0:
            raise ValueError(f"L'épaisseur doit être > 0, reçu thickness={t}")

        B, area = self._strain_displacement_matrix(nodes)
        D = self._elasticity_matrix(material, formulation)

        return B.T @ D @ B * (area * t)

    def mass_matrix(
        self,
        material: ElasticMaterial,
        nodes: np.ndarray,
        properties: dict,
    ) -> np.ndarray:
        """Matrice de masse consistante 6×6.

        Parameters
        ----------
        material : ElasticMaterial
            Matériau (rho utilisé).
        nodes : np.ndarray, shape (3, 2)
            Coordonnées nodales.
        properties : dict
            ``"thickness"`` : épaisseur [m].

        Returns
        -------
        M_e : np.ndarray, shape (6, 6)
            Matrice de masse consistante.

        Notes
        -----
        Avec les fonctions de forme linéaires, l'intégrale ∫ ρ t Nᵀ N dA donne :

            M_e = (ρ t A / 12) · [ 2 0 1 0 1 0 ]
                                  [ 0 2 0 1 0 1 ]
                                  [ 1 0 2 0 1 0 ]
                                  [ 0 1 0 2 0 1 ]
                                  [ 1 0 1 0 2 0 ]
                                  [ 0 1 0 1 0 2 ]

        Les termes diagonaux 2 (nœud i) et hors-diagonaux 1 (nœuds i≠j)
        correspondent aux intégrales ∫ Ni² dA = A/6 et ∫ Ni Nj dA = A/12.
        """
        t = properties["thickness"]
        _, _, _, _, _, _, _ = self._geometry(nodes)  # valide la géométrie
        area, *_ = self._geometry(nodes)

        m = material.rho * t * area / 12.0
        # Bloc 2×2 identité pour chaque paire (i,j) avec facteur 2 si i==j, 1 sinon
        M_e = m * np.array(
            [
                [2, 0, 1, 0, 1, 0],
                [0, 2, 0, 1, 0, 1],
                [1, 0, 2, 0, 1, 0],
                [0, 1, 0, 2, 0, 1],
                [1, 0, 1, 0, 2, 0],
                [0, 1, 0, 1, 0, 2],
            ],
            dtype=float,
        )
        return M_e

    def body_force_vector(
        self,
        material: ElasticMaterial,
        nodes: np.ndarray,
        properties: dict,
        b: np.ndarray,
    ) -> np.ndarray:
        """Forces nodales équivalentes pour une force de volume uniforme b [N/m³].

        f_e = ρ · t · A / 3 · [bx, by, bx, by, bx, by]ᵀ

        Résulte de l'intégration exacte des fonctions de forme linéaires du CST :
        ∫_A Ni dA = A/3 pour chaque nœud.

        Parameters
        ----------
        material : ElasticMaterial
            Matériau (rho utilisé).
        nodes : np.ndarray, shape (3, 2)
            Coordonnées nodales.
        properties : dict
            ``"thickness"`` : épaisseur [m].
        b : np.ndarray, shape (2,)
            Force de volume [N/m³] = ρ · acceleration.

        Returns
        -------
        f_e : np.ndarray, shape (6,)
            Forces nodales équivalentes [N].

        Notes
        -----
        Validation : somme des forces = ρ · t · A · b (équilibre global).

        Examples
        --------
        >>> import numpy as np
        >>> from femsolver.core.material import ElasticMaterial
        >>> mat = ElasticMaterial(E=210e9, nu=0.3, rho=7800)
        >>> nodes = np.array([[0.,0.],[1.,0.],[0.,1.]])
        >>> b = np.array([0., -9.81 * 7800])
        >>> f = Tri3().body_force_vector(mat, nodes, {"thickness": 0.01}, b)
        >>> abs(sum(f[1::2]) - (-9.81 * 7800 * 7800 * 0.01 * 0.5)) < 1.0
        True
        """
        area, *_ = self._geometry(nodes)
        t = properties.get("thickness", 1.0)
        # Chaque nœud reçoit 1/3 de la force totale (intégrale exacte CST)
        f_node = material.rho * t * area / 3.0 * b   # shape (2,)
        return np.tile(f_node, 3)

    def strain(self, nodes: np.ndarray, u_e: np.ndarray) -> np.ndarray:
        """Vecteur de déformations ε = B · u_e (constant dans l'élément).

        Parameters
        ----------
        nodes : np.ndarray, shape (3, 2)
            Coordonnées nodales.
        u_e : np.ndarray, shape (6,)
            Déplacements élémentaires [u1, v1, u2, v2, u3, v3].

        Returns
        -------
        epsilon : np.ndarray, shape (3,)
            [εxx, εyy, γxy].
        """
        B, _ = self._strain_displacement_matrix(nodes)
        return B @ u_e

    def stress(
        self,
        material: ElasticMaterial,
        nodes: np.ndarray,
        u_e: np.ndarray,
        formulation: str = "plane_stress",
    ) -> np.ndarray:
        """Vecteur de contraintes σ = D · B · u_e (constant dans l'élément).

        Parameters
        ----------
        material : ElasticMaterial
            Matériau.
        nodes : np.ndarray, shape (3, 2)
            Coordonnées nodales.
        u_e : np.ndarray, shape (6,)
            Déplacements élémentaires.
        formulation : str
            ``"plane_stress"`` ou ``"plane_strain"``.

        Returns
        -------
        sigma : np.ndarray, shape (3,)
            [σxx, σyy, τxy] [Pa].
        """
        D = self._elasticity_matrix(material, formulation)
        return D @ self.strain(nodes, u_e)

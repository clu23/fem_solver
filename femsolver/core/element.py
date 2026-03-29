"""Classe abstraite définissant l'interface de tout élément fini."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from femsolver.core.material import ElasticMaterial


class Element(ABC):
    """Interface pour tous les éléments finis.

    Tout élément concret (Bar2D, Tri3, Quad4, …) doit implémenter
    cette interface. L'assembleur et le solveur n'interagissent qu'avec
    cette abstraction, ce qui permet d'ajouter de nouveaux éléments sans
    modifier le code existant.

    Notes
    -----
    Les matrices élémentaires (K_e, M_e) sont des `np.ndarray` denses,
    car elles sont petites (4×4 à 24×24). Les matrices globales K et M
    sont TOUJOURS creuses (scipy.sparse).
    """

    @abstractmethod
    def stiffness_matrix(
        self,
        material: ElasticMaterial,
        nodes: np.ndarray,
        properties: dict,
    ) -> np.ndarray:
        """Matrice de rigidité élémentaire en repère global.

        Parameters
        ----------
        material : ElasticMaterial
            Propriétés du matériau.
        nodes : np.ndarray, shape (n_elem_nodes, n_dim)
            Coordonnées des nœuds de l'élément en repère global.
        properties : dict
            Propriétés géométriques de l'élément. Chaque type d'élément
            y lit ce dont il a besoin :
            - Bar2D  : ``{"area": float}`` — section transversale [m²]
            - Tri3   : ``{"thickness": float}`` — épaisseur [m]
            - Quad4  : ``{"thickness": float}``

        Returns
        -------
        K_e : np.ndarray, shape (n_dof_elem, n_dof_elem)
            Matrice de rigidité élémentaire symétrique définie positive
            (avant application des conditions aux limites).

        Raises
        ------
        ValueError
            Si les coordonnées sont incompatibles avec l'élément
            (mauvais nombre de nœuds, longueur nulle, etc.).
        KeyError
            Si une clé requise est absente de ``properties``.
        """
        ...

    @abstractmethod
    def mass_matrix(
        self,
        material: ElasticMaterial,
        nodes: np.ndarray,
        properties: dict,
    ) -> np.ndarray:
        """Matrice de masse consistante élémentaire en repère global.

        Parameters
        ----------
        material : ElasticMaterial
            Propriétés du matériau (densité rho utilisée).
        nodes : np.ndarray, shape (n_elem_nodes, n_dim)
            Coordonnées des nœuds de l'élément en repère global.
        properties : dict
            Propriétés géométriques (voir `stiffness_matrix`).

        Returns
        -------
        M_e : np.ndarray, shape (n_dof_elem, n_dof_elem)
            Matrice de masse consistante (intégrale de N^T·N).
        """
        ...

    @abstractmethod
    def dof_per_node(self) -> int:
        """Nombre de degrés de liberté par nœud.

        Returns
        -------
        int
            2 pour Bar2D (ux, uy), 3 pour Beam2D (ux, uy, θz), etc.
        """
        ...

    @abstractmethod
    def n_nodes(self) -> int:
        """Nombre de nœuds de l'élément.

        Returns
        -------
        int
            2 pour Bar2D et Beam2D, 3 pour Tri3, 4 pour Quad4, etc.
        """
        ...

    def n_dof(self) -> int:
        """Nombre total de DDL de l'élément.

        Returns
        -------
        int
            n_nodes() * dof_per_node()
        """
        return self.n_nodes() * self.dof_per_node()

    # ------------------------------------------------------------------
    # Méthodes optionnelles (chargements distribués / forces de volume)
    # ------------------------------------------------------------------

    def body_force_vector(
        self,
        material: ElasticMaterial,
        nodes: np.ndarray,
        properties: dict,
        b: np.ndarray,
    ) -> np.ndarray:
        """Forces nodales équivalentes pour une force de volume b [N/m³].

        f_e = ∫_Ve  N^T · b dV

        Parameters
        ----------
        material : ElasticMaterial
            Matériau (densité ρ utilisée).
        nodes : np.ndarray, shape (n_elem_nodes, n_dim)
            Coordonnées des nœuds.
        properties : dict
            Propriétés géométriques (épaisseur pour éléments 2D, etc.).
        b : np.ndarray, shape (n_dim,)
            Vecteur d'accélération [m/s²] multiplié par ρ — ou directement
            la force par unité de volume [N/m³].

        Returns
        -------
        f_e : np.ndarray, shape (n_dof_elem,)
            Vecteur de forces nodales équivalentes [N].

        Raises
        ------
        NotImplementedError
            Si l'élément ne supporte pas les forces de volume.
            Bar2D et Beam2D lèvent cette exception (pas de volume surfacique).
        """
        raise NotImplementedError(
            f"{type(self).__name__} ne supporte pas les forces de volume. "
            "Implémentez body_force_vector() dans la sous-classe."
        )

    def distributed_load_vector(
        self,
        material: ElasticMaterial,
        nodes: np.ndarray,
        properties: dict,
        qx: float,
        qy: float,
    ) -> np.ndarray:
        """Forces nodales équivalentes pour une charge linéique (Bar2D / Beam2D).

        f_e = ∫_0^L  N^T · q dx

        Parameters
        ----------
        material : ElasticMaterial
            Matériau (non utilisé pour les éléments linéaires élastiques
            mais disponible pour une éventuelle dépendance matériau).
        nodes : np.ndarray, shape (2, n_dim)
            Coordonnées des 2 nœuds de l'élément.
        properties : dict
            Propriétés géométriques.
        qx : float
            Charge axiale distribuée dans le repère local [N/m].
        qy : float
            Charge transverse distribuée dans le repère local [N/m].
            Non nul uniquement pour Beam2D.

        Returns
        -------
        f_e : np.ndarray, shape (n_dof_elem,)
            Vecteur de forces nodales équivalentes [N], en repère global.

        Raises
        ------
        NotImplementedError
            Pour les éléments qui ne sont pas des éléments 1D (Tri3, Quad4…).
        ValueError
            Si qy ≠ 0 sur Bar2D (pas de résistance transverse).
        """
        raise NotImplementedError(
            f"{type(self).__name__} ne supporte pas les charges linéiques. "
            "Implémentez distributed_load_vector() dans la sous-classe."
        )

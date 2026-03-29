"""Structures de données pour le maillage éléments finis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from femsolver.core.element import Element
    from femsolver.core.material import ElasticMaterial


@dataclass(frozen=True)
class ElementData:
    """Données d'un élément du maillage.

    Relie la géométrie (nœuds), la physique (matériau) et le type
    d'élément (formulation FEM).

    Parameters
    ----------
    etype : type[Element]
        Classe de l'élément (Bar2D, Tri3, Quad4, …). Pas une instance :
        l'assembleur instancie l'élément à la volée.
    node_ids : tuple[int, ...]
        Indices des nœuds dans `Mesh.nodes` (immutables → tuple).
    material : ElasticMaterial
        Matériau associé à cet élément.
    properties : dict
        Propriétés géométriques complémentaires. Pour Bar2D : {"area": A}.
        Pour Tri3/Quad4 : {"thickness": t}.

    Examples
    --------
    >>> from femsolver.elements.bar2d import Bar2D
    >>> from femsolver.core.material import ElasticMaterial
    >>> mat = ElasticMaterial(E=210e9, nu=0.3, rho=7800)
    >>> elem = ElementData(etype=Bar2D, node_ids=(0, 1), material=mat,
    ...                    properties={"area": 1e-4})
    """

    etype: type  # type[Element] — évite l'import circulaire au runtime
    node_ids: tuple[int, ...]
    material: "ElasticMaterial"
    properties: dict = field(default_factory=dict, compare=False, hash=False)

    def get_element(self) -> "Element":
        """Instancie et retourne l'objet élément associé.

        Returns
        -------
        Element
            Instance de l'élément FEM.
        """
        return self.etype()  # type: ignore[call-arg]


@dataclass(frozen=True)
class Mesh:
    """Maillage éléments finis 2D ou 3D — donnée purement géométrique.

    Stocke les nœuds, la connectivité et les matériaux. **Ne contient
    pas** les conditions aux limites, qui dépendent du problème posé
    sur ce maillage (voir `BoundaryConditions`). Un même maillage peut
    ainsi être réutilisé avec différents cas de charge.

    Parameters
    ----------
    nodes : np.ndarray, shape (n_nodes, n_dim)
        Coordonnées des nœuds en mètres. Chaque ligne est un nœud.
    elements : tuple[ElementData, ...]
        Éléments du maillage (connectivité + type + matériau).
    n_dim : int
        Dimension spatiale (2 ou 3).
    dof_per_node : int or None, optional
        Nombre de DDL par nœud. Si None (défaut), égal à ``n_dim``.
        Utiliser ``dof_per_node=3`` pour les éléments poutre 2D (Beam2D)
        qui ont 3 DDL/nœud (ux, uy, θz) mais des coordonnées 2D.

    Examples
    --------
    Treillis horizontal à 2 nœuds, 1 barre :

    >>> import numpy as np
    >>> from femsolver.elements.bar2d import Bar2D
    >>> from femsolver.core.material import ElasticMaterial
    >>> nodes = np.array([[0.0, 0.0], [1.0, 0.0]])
    >>> mat = ElasticMaterial(E=210e9, nu=0.3, rho=7800)
    >>> elem = ElementData(Bar2D, (0, 1), mat, {"area": 1e-4})
    >>> mesh = Mesh(nodes=nodes, elements=(elem,), n_dim=2)

    Poutre console à 3 nœuds (Beam2D, 3 DDL/nœud) :

    >>> from femsolver.elements.beam2d import Beam2D
    >>> nodes = np.array([[0., 0.], [0.5, 0.], [1., 0.]])
    >>> mesh = Mesh(nodes=nodes, elements=(...,), n_dim=2, dof_per_node=3)
    """

    nodes: np.ndarray
    elements: tuple[ElementData, ...]
    n_dim: int
    dof_per_node: int | None = None

    def __post_init__(self) -> None:
        if self.nodes.ndim != 2:
            raise ValueError(
                f"nodes doit être un tableau 2D (n_nodes, n_dim), "
                f"reçu shape={self.nodes.shape}"
            )
        if self.nodes.shape[1] != self.n_dim:
            raise ValueError(
                f"nodes.shape[1]={self.nodes.shape[1]} ≠ n_dim={self.n_dim}"
            )
        if self.n_dim not in (2, 3):
            raise ValueError(f"n_dim doit être 2 ou 3, reçu {self.n_dim}")
        if self.dof_per_node is not None and self.dof_per_node < 1:
            raise ValueError(
                f"dof_per_node doit être ≥ 1, reçu {self.dof_per_node}"
            )

    @property
    def dpn(self) -> int:
        """DDL par nœud : ``dof_per_node`` si spécifié, sinon ``n_dim``."""
        return self.dof_per_node if self.dof_per_node is not None else self.n_dim

    @property
    def n_nodes(self) -> int:
        """Nombre de nœuds du maillage."""
        return self.nodes.shape[0]

    @property
    def n_dof(self) -> int:
        """Nombre total de degrés de liberté (n_nodes × dpn)."""
        return self.n_nodes * self.dpn

    def node_coords(self, node_ids: tuple[int, ...]) -> np.ndarray:
        """Coordonnées d'un sous-ensemble de nœuds.

        Parameters
        ----------
        node_ids : tuple[int, ...]
            Indices des nœuds à extraire.

        Returns
        -------
        np.ndarray, shape (len(node_ids), n_dim)
            Coordonnées des nœuds demandés.
        """
        return self.nodes[list(node_ids)]

    def global_dofs(self, node_ids: tuple[int, ...]) -> list[int]:
        """Indices globaux des DDL associés à un ensemble de nœuds.

        Convention : nœud i → DDL [dpn·i, dpn·i+1, …, dpn·i+dpn−1]
        où ``dpn = dof_per_node`` (ou ``n_dim`` si non spécifié).

        Parameters
        ----------
        node_ids : tuple[int, ...]
            Indices des nœuds.

        Returns
        -------
        list[int]
            Liste des indices de DDL globaux, dans l'ordre
            (ux_0, uy_0, ux_1, uy_1, …) pour n_dim=2 / dof_per_node=None.

        Examples
        --------
        >>> mesh.global_dofs((0, 1))  # n_dim=2, dof_per_node=None
        [0, 1, 2, 3]
        >>> mesh_beam.global_dofs((0, 1))  # n_dim=2, dof_per_node=3
        [0, 1, 2, 3, 4, 5]
        """
        dpn = self.dpn
        dofs = []
        for nid in node_ids:
            for d in range(dpn):
                dofs.append(dpn * nid + d)
        return dofs


@dataclass(frozen=True)
class BoundaryConditions:
    """Conditions aux limites d'un problème posé sur un maillage.

    Sépare le problème physique (quelles forces, quels appuis) de la
    géométrie (`Mesh`). Un même `Mesh` peut être soumis à plusieurs
    jeux de conditions aux limites (cas de charge multiples).

    Parameters
    ----------
    dirichlet : dict[int, dict[int, float]]
        Déplacements imposés. Format : {node_id: {dof: valeur [m]}}.
        Le DDL 0 correspond à ux, 1 à uy, 2 à uz.
        Exemple : {0: {0: 0.0, 1: 0.0}} → nœud 0 bloqué en x et y.
    neumann : dict[int, dict[int, float]]
        Forces nodales appliquées. Format : {node_id: {dof: force [N]}}.
        Exemple : {3: {1: -5000.0}} → force de -5 kN en y sur nœud 3.

    Examples
    --------
    >>> bc = BoundaryConditions(
    ...     dirichlet={0: {0: 0.0, 1: 0.0}, 1: {1: 0.0}},
    ...     neumann={3: {1: -10000.0}},
    ... )
    """

    dirichlet: dict[int, dict[int, float]]
    neumann: dict[int, dict[int, float]]

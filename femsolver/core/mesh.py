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
class PressureLoad:
    """Pression surfacique sur une arête (2D) ou une face (3D).

    Les nœuds doivent être listés dans l'ordre **CCW** (sens antihoraire
    vu de l'extérieur du domaine).  La normale sortante est déduite
    automatiquement de cet ordre et vérifiée lors de l'assemblage.

    Parameters
    ----------
    node_ids : tuple[int, ...]
        Nœuds de la face dans l'ordre CCW :

        - 2 nœuds  → arête 2D (Tri3, Quad4)
        - 3 nœuds  → face triangulaire 3D (Tetra4)
        - 4 nœuds  → face quadrangulaire 3D (Hexa8)

    magnitude : float
        Intensité de la pression [Pa].  Positif = compression
        (la force est dirigée vers l'intérieur, i.e. dans le sens −n̂).

    Examples
    --------
    Pression de 10 kPa sur l'arête droite d'un carré (nœuds 1→2) :

    >>> load = PressureLoad(node_ids=(1, 2), magnitude=10_000.0)
    """

    node_ids: tuple[int, ...]
    magnitude: float


@dataclass(frozen=True)
class BodyForce:
    """Force de volume uniforme sur tout le maillage (gravité, inertie…).

    Parameters
    ----------
    acceleration : tuple[float, ...]
        Vecteur d'accélération du champ de forces [m/s²].
        La force par unité de volume est b = ρ · acceleration.

        - 2D : (ax, ay)   — ex: (0.0, −9.81) pour la gravité vers −y
        - 3D : (ax, ay, az)

    Examples
    --------
    >>> gravity_2d = BodyForce(acceleration=(0.0, -9.81))
    >>> gravity_3d = BodyForce(acceleration=(0.0, 0.0, -9.81))
    """

    acceleration: tuple[float, ...]


@dataclass(frozen=True)
class DistributedLineLoad:
    """Charge linéique sur un élément Bar2D (axiale) ou Beam2D (qx + qy).

    Les forces sont spécifiées dans le **repère local** de l'élément :

    - *qx* : force par unité de longueur dans la direction axiale [N/m].
    - *qy* : force par unité de longueur dans la direction transverse [N/m]
      (uniquement pour Beam2D ; une erreur est levée si qy ≠ 0 sur Bar2D).

    Parameters
    ----------
    node_ids : tuple[int, int]
        Nœuds de l'élément (même ordre que dans ``ElementData.node_ids``).
    qx : float
        Charge axiale distribuée [N/m].  Positif dans le sens i→j.
    qy : float
        Charge transverse distribuée [N/m].  Positif dans le sens +y local
        (vers le haut pour une poutre horizontale).

    Examples
    --------
    Charge transverse uniforme q = 5 kN/m sur l'élément (nœuds 0→1) :

    >>> load = DistributedLineLoad(node_ids=(0, 1), qy=-5_000.0)
    """

    node_ids: tuple[int, int]
    qx: float = 0.0
    qy: float = 0.0


@dataclass(frozen=True)
class MPCConstraint:
    """Contrainte linéaire multi-points : Σ aᵢ·u[nœud_i, ddl_i] = β.

    Chaque terme est un triplet ``(node_id, local_dof, coefficient)``.  Le DDL
    global correspondant est ``mesh.dpn * node_id + local_dof``.

    Convention pour la **méthode d'élimination** :

    - Le **premier terme** est l'esclave (DOF éliminé du système).
    - Exactement **2 termes** pour l'élimination (1 esclave + 1 maître).
    - La contrainte est réécrite :

          a_s · u_s + a_m · u_m = β
          → u_s = (β − a_m · u_m) / a_s = α · u_m + β̃

    Pour la **méthode des multiplicateurs de Lagrange**, tous les termes sont
    égaux (pas de notion maître/esclave) et le nombre de termes est quelconque.

    Parameters
    ----------
    terms : tuple[tuple[int, int, float], ...]
        Triplets ``(node_id, local_dof, coefficient)``.
    rhs : float
        Second membre β de la contrainte (défaut 0).

    Examples
    --------
    Liaison rigide horizontale — u_x(nœud 1) = u_x(nœud 3) :

    >>> c = MPCConstraint(terms=((1, 0, 1.0), (3, 0, -1.0)), rhs=0.0)

    Raccord poutre — θ(nœud 1) = θ(nœud 2) :

    >>> c = MPCConstraint(terms=((1, 2, 1.0), (2, 2, -1.0)), rhs=0.0)
    """

    terms: tuple[tuple[int, int, float], ...]
    rhs: float = 0.0


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
    pressure : tuple[PressureLoad, ...]
        Pressions surfaciques.  Défaut : vide.
    body_force : BodyForce or None
        Force de volume uniforme (gravité…).  Défaut : None.
    distributed : tuple[DistributedLineLoad, ...]
        Charges linéiques sur Bar2D / Beam2D.  Défaut : vide.

    Examples
    --------
    >>> bc = BoundaryConditions(
    ...     dirichlet={0: {0: 0.0, 1: 0.0}, 1: {1: 0.0}},
    ...     neumann={3: {1: -10000.0}},
    ... )
    >>> bc_gravity = BoundaryConditions(
    ...     dirichlet={0: {0: 0.0, 1: 0.0}},
    ...     neumann={},
    ...     body_force=BodyForce(acceleration=(0.0, -9.81)),
    ... )
    """

    dirichlet: dict[int, dict[int, float]]
    neumann: dict[int, dict[int, float]]
    pressure: tuple[PressureLoad, ...] = field(default_factory=tuple)
    body_force: BodyForce | None = None
    distributed: tuple[DistributedLineLoad, ...] = field(default_factory=tuple)

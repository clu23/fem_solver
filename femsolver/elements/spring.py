"""Élément ressort ponctuel — raideur discrète (type CBUSH de Nastran).

Un ressort relie soit **deux nœuds** (raideur relative), soit **un nœud et
le sol** (appui élastique). La raideur est définie DDL par DDL : translation
(kx, ky, kz) et/ou rotation (krx, kry, krz), ce qui couvre les modèles 1D, 2D
et 3D. Le ressort est aligné sur les axes globaux (pas de géométrie : il ne
dépend que des DDL, pas des coordonnées des nœuds).

Convention matricielle
-----------------------
Pour une raideur ``k`` sur un DDL donné :

- ressort sol (1 nœud)   : ``[k]`` (terme diagonal seul)
- ressort 2 nœuds        : ``[[k, -k], [-k, k]]``

Avec plusieurs DDL actifs, ``k`` devient le vecteur ``diag(k1, …, k_dpn)`` et
chaque bloc 1×1 ci-dessus devient un bloc dpn×dpn diagonal.
"""

from __future__ import annotations

import numpy as np

from femsolver.core.element import Element
from femsolver.core.material import ElasticMaterial


def _coupling_matrix(diag: np.ndarray, n_elem_nodes: int) -> np.ndarray:
    """Construit la matrice élémentaire d'un connecteur ponctuel diagonal.

    Parameters
    ----------
    diag : np.ndarray, shape (dpn,)
        Coefficients diagonaux par DDL (raideur ou amortissement).
    n_elem_nodes : int
        1 pour un connecteur au sol, 2 pour un connecteur entre deux nœuds.

    Returns
    -------
    np.ndarray
        - ``(dpn, dpn)`` si ``n_elem_nodes == 1`` : ``diag(diag)``.
        - ``(2·dpn, 2·dpn)`` si ``n_elem_nodes == 2`` : blocs
          ``[[D, -D], [-D, D]]`` avec ``D = diag(diag)``.

    Raises
    ------
    ValueError
        Si ``n_elem_nodes`` n'est ni 1 ni 2.
    """
    D = np.diag(diag)
    if n_elem_nodes == 1:
        return D.copy()
    if n_elem_nodes == 2:
        dpn = diag.shape[0]
        mat = np.zeros((2 * dpn, 2 * dpn))
        mat[:dpn, :dpn] = D
        mat[dpn:, dpn:] = D
        mat[:dpn, dpn:] = -D
        mat[dpn:, :dpn] = -D
        return mat
    raise ValueError(
        f"Un connecteur ponctuel relie 1 (sol) ou 2 nœuds, reçu {n_elem_nodes}."
    )


class SpringElement(Element):
    """Ressort ponctuel à raideur diagonale, 1 ou 2 nœuds, 1D/2D/3D.

    Chaque nœud porte ``dpn`` DDL ; la raideur est fournie DDL par DDL via la
    propriété ``"stiffness"`` (vecteur de longueur ``dpn``). Une valeur nulle
    désactive simplement ce DDL. Le ressort est aligné sur les axes globaux et
    ne dépend pas de la position des nœuds.

    L'élément est **sans état** : sa taille (nombre de nœuds, nombre de DDL)
    est entièrement déterminée à l'assemblage par ``nodes`` et ``properties``.
    Les arguments du constructeur ne servent qu'aux métadonnées
    (``dof_per_node()`` / ``n_nodes()``) consultées par les diagnostics.

    Parameters
    ----------
    n_nodes : int, optional
        Nombre de nœuds (métadonnée). 2 par défaut (ressort entre deux nœuds) ;
        1 pour un ressort au sol.
    dof_per_node : int, optional
        DDL par nœud (métadonnée). 3 par défaut.

    Notes
    -----
    Propriété requise : ``"stiffness"`` — réel ou liste de réels [N/m] pour les
    translations, [N·m/rad] pour les rotations. La longueur du vecteur fixe le
    nombre de DDL par nœud et doit coïncider avec ``mesh.dpn``.

    Référence : MSC Nastran, élément CBUSH (raideur ponctuelle K1…K6).

    Examples
    --------
    Ressort au sol de raideur k = 1000 N/m sur le DDL y (modèle 2D) :

    >>> import numpy as np
    >>> from femsolver.core.material import ElasticMaterial
    >>> mat = ElasticMaterial(E=1.0, nu=0.0, rho=0.0)
    >>> nodes = np.array([[0.0, 1.0]])           # 1 nœud → ressort au sol
    >>> K_e = SpringElement().stiffness_matrix(mat, nodes, {"stiffness": [0.0, 1000.0]})
    >>> K_e
    array([[   0.,    0.],
           [   0., 1000.]])
    """

    def __init__(self, n_nodes: int = 2, dof_per_node: int = 3) -> None:
        self._n_nodes = n_nodes
        self._dpn = dof_per_node

    def dof_per_node(self) -> int:
        """DDL par nœud (métadonnée — la taille réelle vient de ``properties``)."""
        return self._dpn

    def n_nodes(self) -> int:
        """Nombre de nœuds (métadonnée — la taille réelle vient de ``nodes``)."""
        return self._n_nodes

    @staticmethod
    def _stiffness_vector(properties: dict) -> np.ndarray:
        """Lit et valide le vecteur de raideur par DDL depuis ``properties``.

        Parameters
        ----------
        properties : dict
            Doit contenir ``"stiffness"`` : réel ou liste de réels ≥ 0.

        Returns
        -------
        np.ndarray, shape (dpn,)
            Raideurs par DDL.

        Raises
        ------
        KeyError
            Si ``"stiffness"`` est absent.
        ValueError
            Si une raideur est négative ou si toutes sont nulles.
        """
        if "stiffness" not in properties:
            raise KeyError(
                "SpringElement requiert la propriété 'stiffness' "
                "(réel ou liste de réels par DDL)."
            )
        k = np.atleast_1d(np.asarray(properties["stiffness"], dtype=float))
        if np.any(k < 0.0):
            raise ValueError(f"Les raideurs doivent être ≥ 0, reçu {k}.")
        if not np.any(k > 0.0):
            raise ValueError("Au moins une raideur doit être > 0.")
        return k

    def stiffness_matrix(
        self,
        material: ElasticMaterial,
        nodes: np.ndarray,
        properties: dict,
    ) -> np.ndarray:
        """Matrice de rigidité élémentaire diagonale en repère global.

        Parameters
        ----------
        material : ElasticMaterial
            Non utilisé (le ressort ne dépend pas d'un matériau continu).
        nodes : np.ndarray, shape (1 ou 2, n_dim)
            1 nœud → ressort au sol ; 2 nœuds → ressort entre nœuds.
        properties : dict
            ``"stiffness"`` : raideur par DDL [N/m] ou [N·m/rad].

        Returns
        -------
        K_e : np.ndarray
            ``(dpn, dpn)`` (sol) ou ``(2·dpn, 2·dpn)`` (deux nœuds).

        Notes
        -----
        - Sol     : ``K_e = diag(k)``.
        - 2 nœuds : ``K_e = [[diag(k), -diag(k)], [-diag(k), diag(k)]]``.

        Examples
        --------
        Ressort entre deux nœuds, raideur axiale k = 1000 N/m (DDL x seul) :

        >>> import numpy as np
        >>> from femsolver.core.material import ElasticMaterial
        >>> mat = ElasticMaterial(E=1.0, nu=0.0, rho=0.0)
        >>> nodes = np.array([[0.0, 0.0], [1.0, 0.0]])
        >>> K = SpringElement().stiffness_matrix(mat, nodes, {"stiffness": [1000.0, 0.0]})
        >>> K[0, 0], K[0, 2]
        (1000.0, -1000.0)
        """
        k = self._stiffness_vector(properties)
        return _coupling_matrix(k, nodes.shape[0])

    def mass_matrix(
        self,
        material: ElasticMaterial,
        nodes: np.ndarray,
        properties: dict,
    ) -> np.ndarray:
        """Matrice de masse — nulle (un ressort idéal est sans masse).

        Parameters
        ----------
        material, nodes, properties
            Voir ``stiffness_matrix``. ``properties["stiffness"]`` fixe la taille.

        Returns
        -------
        M_e : np.ndarray
            Zéros, de la même taille que ``stiffness_matrix``.
        """
        k = self._stiffness_vector(properties)
        n_dof = nodes.shape[0] * k.shape[0]
        return np.zeros((n_dof, n_dof))

"""Élément amortisseur ponctuel — amortissement visqueux discret (type CBUSH).

Pendant visqueux du ressort (``femsolver.elements.spring``). Un amortisseur
relie soit **deux nœuds**, soit **un nœud et le sol**, et oppose une force
proportionnelle à la vitesse relative : ``f = c · v``. Le coefficient ``c`` est
défini DDL par DDL (translation [N·s/m] et/ou rotation [N·m·s/rad]), couvrant
les modèles 1D, 2D et 3D.

Convention matricielle (identique au ressort)
---------------------------------------------
- amortisseur sol (1 nœud) : ``[c]``
- amortisseur 2 nœuds      : ``[[c, -c], [-c, c]]``

L'amortisseur ne contribue **ni** à la rigidité **ni** à la masse : seules ses
matrices d'amortissement ``C_e`` sont non nulles. ``Assembler.assemble_damping``
collecte ces contributions dans la matrice globale C.
"""

from __future__ import annotations

import numpy as np

from femsolver.core.element import Element
from femsolver.core.material import ElasticMaterial
from femsolver.elements.spring import _coupling_matrix


class DamperElement(Element):
    """Amortisseur visqueux ponctuel, 1 ou 2 nœuds, 1D/2D/3D.

    Chaque nœud porte ``dpn`` DDL ; le coefficient visqueux est fourni DDL par
    DDL via la propriété ``"damping"`` (vecteur de longueur ``dpn``). L'élément
    est aligné sur les axes globaux et sans état (taille déterminée à
    l'assemblage par ``nodes`` et ``properties``).

    Parameters
    ----------
    n_nodes : int, optional
        Nombre de nœuds (métadonnée). 2 par défaut ; 1 pour un amortisseur sol.
    dof_per_node : int, optional
        DDL par nœud (métadonnée). 3 par défaut.

    Notes
    -----
    Propriété requise : ``"damping"`` — réel ou liste de réels [N·s/m] (trans.)
    ou [N·m·s/rad] (rot.). La longueur fixe le nombre de DDL par nœud et doit
    coïncider avec ``mesh.dpn``.

    ``stiffness_matrix`` et ``mass_matrix`` renvoient des zéros : l'amortisseur
    n'apporte qu'un terme visqueux, assemblé via ``damping_matrix``.

    Référence : MSC Nastran, élément CBUSH (amortissement ponctuel B1…B6).

    Examples
    --------
    Amortisseur au sol c = 50 N·s/m sur le DDL y (modèle 2D) :

    >>> import numpy as np
    >>> from femsolver.core.material import ElasticMaterial
    >>> mat = ElasticMaterial(E=1.0, nu=0.0, rho=0.0)
    >>> nodes = np.array([[0.0, 1.0]])
    >>> C_e = DamperElement().damping_matrix(mat, nodes, {"damping": [0.0, 50.0]})
    >>> C_e
    array([[ 0.,  0.],
           [ 0., 50.]])
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
    def _damping_vector(properties: dict) -> np.ndarray:
        """Lit et valide le vecteur d'amortissement par DDL depuis ``properties``.

        Parameters
        ----------
        properties : dict
            Doit contenir ``"damping"`` : réel ou liste de réels ≥ 0.

        Returns
        -------
        np.ndarray, shape (dpn,)
            Coefficients visqueux par DDL.

        Raises
        ------
        KeyError
            Si ``"damping"`` est absent.
        ValueError
            Si un coefficient est négatif ou si tous sont nuls.
        """
        if "damping" not in properties:
            raise KeyError(
                "DamperElement requiert la propriété 'damping' "
                "(réel ou liste de réels par DDL)."
            )
        c = np.atleast_1d(np.asarray(properties["damping"], dtype=float))
        if np.any(c < 0.0):
            raise ValueError(f"Les coefficients visqueux doivent être ≥ 0, reçu {c}.")
        if not np.any(c > 0.0):
            raise ValueError("Au moins un coefficient visqueux doit être > 0.")
        return c

    def _zero_matrix(self, nodes: np.ndarray, properties: dict) -> np.ndarray:
        """Matrice nulle dimensionnée comme ``damping_matrix`` (rigidité/masse)."""
        c = self._damping_vector(properties)
        n_dof = nodes.shape[0] * c.shape[0]
        return np.zeros((n_dof, n_dof))

    def damping_matrix(
        self,
        material: ElasticMaterial,
        nodes: np.ndarray,
        properties: dict,
    ) -> np.ndarray:
        """Matrice d'amortissement visqueux diagonale en repère global.

        Parameters
        ----------
        material : ElasticMaterial
            Non utilisé.
        nodes : np.ndarray, shape (1 ou 2, n_dim)
            1 nœud → amortisseur au sol ; 2 nœuds → entre nœuds.
        properties : dict
            ``"damping"`` : coefficient visqueux par DDL [N·s/m] ou [N·m·s/rad].

        Returns
        -------
        C_e : np.ndarray
            ``(dpn, dpn)`` (sol) ou ``(2·dpn, 2·dpn)`` (deux nœuds).

        Notes
        -----
        - Sol     : ``C_e = diag(c)``.
        - 2 nœuds : ``C_e = [[diag(c), -diag(c)], [-diag(c), diag(c)]]``.
        """
        c = self._damping_vector(properties)
        return _coupling_matrix(c, nodes.shape[0])

    def stiffness_matrix(
        self,
        material: ElasticMaterial,
        nodes: np.ndarray,
        properties: dict,
    ) -> np.ndarray:
        """Rigidité — nulle (un amortisseur idéal n'a pas de raideur).

        Returns
        -------
        K_e : np.ndarray
            Zéros, dimensionnés comme ``damping_matrix`` (pour que l'assemblage
            global de K reste cohérent en DDL).
        """
        return self._zero_matrix(nodes, properties)

    def mass_matrix(
        self,
        material: ElasticMaterial,
        nodes: np.ndarray,
        properties: dict,
    ) -> np.ndarray:
        """Masse — nulle (un amortisseur idéal est sans masse).

        Returns
        -------
        M_e : np.ndarray
            Zéros, dimensionnés comme ``damping_matrix``.
        """
        return self._zero_matrix(nodes, properties)

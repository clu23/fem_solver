"""Import/export de maillages via meshio.

Import : fichiers .msh (Gmsh v2/v4), .vtk, .vtu, .inp (Abaqus)…
Export : .vtu (VTK XML non structuré) pour ParaView.

Correspondance des types d'éléments
------------------------------------

+-------------------+------------------+------------------------------------+
| Nom meshio        | Classe femsolver | Nœuds                              |
+-------------------+------------------+------------------------------------+
| ``"hexahedron"``  | Hexa8            | 8, convention VTK = convention H8  |
| ``"tetra"``       | Tetra4           | 4                                  |
+-------------------+------------------+------------------------------------+

Notes sur les conventions d'orientation
-----------------------------------------
meshio normalise l'ordre des nœuds selon la convention VTK, qui coïncide
exactement avec la convention interne de Hexa8 et Tetra4 de ce projet.
Aucune permutation n'est nécessaire.

Lecture d'un fichier Gmsh (.msh v2 ou v4)
------------------------------------------
meshio prend en charge les deux versions. Seuls les éléments volumiques 3D
sont importés. Les éléments de surface/ligne (utilisés par Gmsh pour marquer
les groupes physiques) sont ignorés.

Groupes physiques et conditions aux limites
--------------------------------------------
Les ``point_sets`` de meshio (section ``$NodeData`` ou ``$PhysicalNames``
de Gmsh) sont retournés dans un dictionnaire ``node_sets`` :

    node_sets["fixed"]   = [0, 1, 2, 3, ...]   (indices de nœuds)
    node_sets["loaded"]  = [12, 13, 14, ...]

L'utilisateur peut ensuite construire les `BoundaryConditions` à partir de
ces ensembles de nœuds.

Export VTK/VTU
--------------
La fonction ``write_vtu`` exporte les résultats en format VTK XML pour
ParaView. Elle accepte :
- le déplacement ``u`` (champ vectoriel aux nœuds)
- les contraintes nodales ``sigma`` (shape n×6 ou n×3)
- la contrainte de Von Mises ``sigma_vm`` (scalaire aux nœuds)

Examples
--------
Lecture et résolution d'un cube Gmsh :

>>> mesh, node_sets = read_mesh("cube.msh", material=mat)
>>> # Construire bc depuis node_sets["fixed"] et node_sets["loaded"]
>>> write_vtu("results.vtu", mesh, u=u, sigma=sigma_nodal)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from femsolver.core.material import ElasticMaterial
from femsolver.core.mesh import ElementData, Mesh

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dépendance optionnelle : meshio
# ---------------------------------------------------------------------------

try:
    import meshio as _meshio
    HAS_MESHIO = True
except ImportError:
    HAS_MESHIO = False


def _require_meshio() -> None:
    if not HAS_MESHIO:
        raise ImportError(
            "meshio est requis pour lire/écrire des maillages.\n"
            "Installer avec : pip install meshio"
        )


# Correspondance : type meshio → classe d'élément femsolver
# Importé ici pour éviter un import circulaire au niveau du paquet.
def _get_element_registry() -> dict[str, type]:
    from femsolver.elements.hexa8 import Hexa8
    from femsolver.elements.tetra4 import Tetra4
    return {
        "hexahedron": Hexa8,
        "tetra":      Tetra4,
    }


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------


def read_mesh(
    path: str | Path,
    material: ElasticMaterial,
    properties: dict[str, Any] | None = None,
) -> tuple[Mesh, dict[str, list[int]]]:
    """Lire un fichier maillage (Gmsh .msh, VTK, Abaqus…) en un objet `Mesh`.

    Parameters
    ----------
    path : str or Path
        Chemin vers le fichier maillage. Le format est déduit de l'extension.
    material : ElasticMaterial
        Matériau appliqué à **tous** les éléments volumiques importés.
    properties : dict, optional
        Propriétés complémentaires passées à ``ElementData.properties``
        (exemple : ``{"thickness": 0.01}`` pour des éléments 2D).
        Par défaut ``{}``.

    Returns
    -------
    mesh : Mesh
        Maillage femsolver (nœuds + éléments 3D uniquement).
    node_sets : dict[str, list[int]]
        Groupes de nœuds nommés (depuis les groupes physiques Gmsh ou
        les node sets Abaqus). Peut être vide si le format ne les supporte
        pas ou si aucun groupe n'est défini.

    Raises
    ------
    ImportError
        Si meshio n'est pas installé.
    ValueError
        Si aucun élément volumique reconnu n'est trouvé dans le fichier.
    FileNotFoundError
        Si le fichier n'existe pas.

    Notes
    -----
    Seuls les éléments volumiques sont importés. Les éléments de surface,
    d'arête ou de point (utilisés par Gmsh pour marquer les groupes
    physiques) sont ignorés.

    La numérotation des nœuds est conservée telle quelle depuis le fichier
    (0-based dans meshio, converti depuis 1-based Gmsh automatiquement).

    Examples
    --------
    >>> mat = ElasticMaterial(E=210e9, nu=0.3, rho=7800)
    >>> mesh, node_sets = read_mesh("bracket.msh", material=mat)
    >>> print(f"{mesh.n_nodes} nœuds, {len(mesh.elements)} éléments")
    """
    _require_meshio()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier maillage introuvable : {path}")

    props = properties or {}
    registry = _get_element_registry()

    logger.info("Lecture du maillage : %s", path)
    raw = _meshio.read(str(path))

    # --- Nœuds ---
    points = np.asarray(raw.points, dtype=float)
    n_dim = points.shape[1]
    if n_dim not in (2, 3):
        raise ValueError(f"Dimension spatiale inattendue : {n_dim} (attendu 2 ou 3)")

    # --- Éléments volumiques ---
    # Types volumiques 3D reconnus
    volume_types = {"hexahedron", "tetra"}
    # Types à ignorer (surfaciques, arêtes, points)
    ignored_types = {
        "line", "line2", "line3",
        "triangle", "triangle6",
        "quad", "quad8",
        "vertex", "point",
    }

    elements: list[ElementData] = []
    n_ignored = 0

    for cell_block in raw.cells:
        ctype = cell_block.type
        if ctype in ignored_types:
            n_ignored += cell_block.data.shape[0]
            continue
        if ctype not in registry:
            logger.warning(
                "Type d'élément non supporté '%s' (%d éléments) — ignoré.",
                ctype, cell_block.data.shape[0],
            )
            continue
        elem_class = registry[ctype]
        for connectivity in cell_block.data:
            elements.append(ElementData(
                etype=elem_class,
                node_ids=tuple(int(n) for n in connectivity),
                material=material,
                properties=props,
            ))

    if not elements:
        supported = list(volume_types & set(registry.keys()))
        raise ValueError(
            f"Aucun élément volumique reconnu dans '{path.name}'.\n"
            f"Types supportés : {supported}.\n"
            f"Types surfaciques ignorés : {n_ignored} éléments."
        )

    logger.info(
        "  %d nœuds, %d éléments importés (%d éléments non-volumiques ignorés)",
        len(points), len(elements), n_ignored,
    )

    mesh = Mesh(nodes=points, elements=tuple(elements), n_dim=n_dim)

    # --- Groupes de nœuds (point_sets) ---
    node_sets: dict[str, list[int]] = {}
    for name, indices in (raw.point_sets or {}).items():
        node_sets[name] = [int(i) for i in indices]
        logger.info("  Groupe de nœuds '%s' : %d nœuds", name, len(indices))

    return mesh, node_sets


# ---------------------------------------------------------------------------
# Export VTK/VTU
# ---------------------------------------------------------------------------

# Correspondance : classe femsolver → type de cellule meshio
def _cell_type_for(etype: type) -> str:
    from femsolver.elements.hexa8 import Hexa8
    from femsolver.elements.tetra4 import Tetra4
    mapping = {Hexa8: "hexahedron", Tetra4: "tetra"}
    result = mapping.get(etype)
    if result is None:
        raise ValueError(
            f"Export VTK non supporté pour l'élément '{etype.__name__}'. "
            f"Types supportés : {list(mapping.keys())}"
        )
    return result


def write_vtu(
    path: str | Path,
    mesh: Mesh,
    u: np.ndarray | None = None,
    sigma: np.ndarray | None = None,
    sigma_vm: np.ndarray | None = None,
) -> None:
    """Exporter le maillage et les résultats en VTU (VTK XML non structuré).

    Le fichier .vtu peut être ouvert directement dans ParaView pour
    visualiser les champs de déplacement et de contrainte.

    Parameters
    ----------
    path : str or Path
        Chemin du fichier de sortie (extension ``.vtu`` recommandée).
    mesh : Mesh
        Maillage femsolver.
    u : np.ndarray, shape (n_dof,), optional
        Vecteur de déplacements. Exporté comme champ vectoriel ``"u"``
        aux nœuds. Les composantes manquantes sont complétées à 0
        (ex : si ``n_dim=2``, la composante z est mise à 0 pour VTK).
    sigma : np.ndarray, optional
        Contraintes nodales. Shape ``(n_nodes, 3)`` [σxx, σyy, τxy] pour
        2D ou ``(n_nodes, 6)`` [σxx, σyy, σzz, τyz, τxz, τxy] pour 3D.
    sigma_vm : np.ndarray, shape (n_nodes,), optional
        Contrainte de Von Mises aux nœuds [Pa]. Exportée comme champ
        scalaire ``"von_mises"``.

    Raises
    ------
    ImportError
        Si meshio n'est pas installé.

    Examples
    --------
    >>> write_vtu("results.vtu", mesh, u=u, sigma=sigma_nodal, sigma_vm=vm)
    >>> # Ouvrir results.vtu dans ParaView pour visualiser
    """
    _require_meshio()
    path = Path(path)

    # --- Grouper les éléments par type ---
    from collections import defaultdict
    by_type: dict[str, list[list[int]]] = defaultdict(list)
    for elem in mesh.elements:
        ctype = _cell_type_for(elem.etype)
        by_type[ctype].append(list(elem.node_ids))

    cells = [
        _meshio.CellBlock(ctype, np.array(conns, dtype=int))
        for ctype, conns in by_type.items()
    ]

    # --- Champs aux nœuds ---
    point_data: dict[str, np.ndarray] = {}

    if u is not None:
        u_nodes = u.reshape(-1, mesh.n_dim)
        if mesh.n_dim == 2:
            # VTK/ParaView attend des vecteurs 3D
            u3d = np.zeros((mesh.n_nodes, 3))
            u3d[:, :2] = u_nodes
            point_data["u"] = u3d
        else:
            point_data["u"] = u_nodes.copy()

    if sigma is not None:
        sigma = np.asarray(sigma)
        if sigma.ndim == 2 and sigma.shape == (mesh.n_nodes, 3):
            # 2D : [sxx, syy, txy]
            point_data["sigma_xx"] = sigma[:, 0]
            point_data["sigma_yy"] = sigma[:, 1]
            point_data["tau_xy"]   = sigma[:, 2]
        elif sigma.ndim == 2 and sigma.shape == (mesh.n_nodes, 6):
            # 3D : [sxx, syy, szz, tyz, txz, txy]
            labels = ["sigma_xx", "sigma_yy", "sigma_zz", "tau_yz", "tau_xz", "tau_xy"]
            for k, lbl in enumerate(labels):
                point_data[lbl] = sigma[:, k]
        else:
            logger.warning(
                "Forme sigma inattendue %s — ignorée (attendu (n_nodes, 3) ou (n_nodes, 6)).",
                sigma.shape,
            )

    if sigma_vm is not None:
        point_data["von_mises"] = np.asarray(sigma_vm, dtype=float)

    # --- Écriture ---
    # Nœuds : toujours en 3D pour VTK (compléter avec des zéros si 2D)
    points = mesh.nodes
    if mesh.n_dim == 2:
        z = np.zeros((mesh.n_nodes, 1))
        points = np.hstack([points, z])

    raw = _meshio.Mesh(points=points, cells=cells, point_data=point_data)
    raw.write(str(path))
    logger.info("Résultats exportés : %s", path)

"""Visualisation 3D des maillages et résultats éléments finis via PyVista.

Ce module fournit deux fonctions principales :

- `plot_mesh_3d`    — Affiche la géométrie non déformée (fil de fer + faces).
- `plot_deformed_3d` — Affiche la déformée amplifiée et une carte de contrainte.

Rendu hors-écran (sans display X11)
--------------------------------------
Sur les environnements sans serveur X (WSL, CI, serveurs headless), PyVista
peut ne pas ouvrir de fenêtre interactive. Dans ce cas, utilisez le paramètre
``screenshot="output.png"`` pour sauvegarder l'image directement sur disque.

Le module tente automatiquement de démarrer un serveur X virtuel (via
``pyvista.start_xvfb()``) si aucun display n'est détecté. Si cela échoue,
il bascule en mode hors-écran silencieusement.

Types d'éléments supportés
----------------------------
+----------+----------------+----------------------------------------------+
| Type     | Classe         | Cellule VTK                                  |
+----------+----------------+----------------------------------------------+
| Hexa8    | `Hexa8`        | VTK_HEXAHEDRON (12)                         |
| Tetra4   | `Tetra4`       | VTK_TETRA (10)                              |
+----------+----------------+----------------------------------------------+

Examples
--------
Affichage simple d'un maillage :

>>> plot_mesh_3d(mesh, screenshot="mesh.png")

Déformée + Von Mises :

>>> sigma = nodal_stresses_3d(mesh, u)
>>> vm = von_mises_3d(sigma)
>>> plot_deformed_3d(mesh, u, sigma_vm=vm, scale=500, screenshot="vm.png")
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import numpy as np

from femsolver.core.mesh import Mesh

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dépendance optionnelle : PyVista
# ---------------------------------------------------------------------------

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False


def _require_pyvista() -> None:
    if not HAS_PYVISTA:
        raise ImportError(
            "PyVista est requis pour la visualisation 3D.\n"
            "Installer avec : pip install pyvista"
        )


# ---------------------------------------------------------------------------
# Correspondance type d'élément → code de cellule VTK
# ---------------------------------------------------------------------------

def _vtk_cell_type(etype: type) -> int:
    """Code de cellule VTK pour une classe d'élément femsolver."""
    from femsolver.elements.hexa8 import Hexa8
    from femsolver.elements.tetra4 import Tetra4
    mapping = {Hexa8: 12, Tetra4: 10}   # VTK_HEXAHEDRON=12, VTK_TETRA=10
    result = mapping.get(etype)
    if result is None:
        raise ValueError(
            f"Visualisation non supportée pour '{etype.__name__}'. "
            f"Supportés : Hexa8, Tetra4."
        )
    return result


# ---------------------------------------------------------------------------
# Construction de la grille PyVista
# ---------------------------------------------------------------------------

def _build_pyvista_grid(mesh: Mesh, u_displaced: np.ndarray | None = None) -> "pv.UnstructuredGrid":
    """Construire un `pv.UnstructuredGrid` depuis un objet `Mesh` femsolver.

    Parameters
    ----------
    mesh : Mesh
        Maillage source.
    u_displaced : np.ndarray, shape (n_dof,), optional
        Vecteur de déplacements pour déformer les nœuds. Si None, utilise
        les coordonnées originales.

    Returns
    -------
    grid : pv.UnstructuredGrid
    """
    points = mesh.nodes.copy()
    if u_displaced is not None:
        points = points + u_displaced.reshape(-1, mesh.n_dim)

    # PyVista attend des points 3D (x, y, z)
    if mesh.n_dim == 2:
        z = np.zeros((mesh.n_nodes, 1))
        points = np.hstack([points, z])

    # Construction du tableau de cellules : [n_nodes, n0, n1, ..., n_nk, ...]
    cells_arr = []
    cell_types = []
    for elem in mesh.elements:
        n = len(elem.node_ids)
        cells_arr.append(n)
        cells_arr.extend(elem.node_ids)
        cell_types.append(_vtk_cell_type(elem.etype))

    cells_np = np.array(cells_arr, dtype=int)
    types_np = np.array(cell_types, dtype=np.uint8)

    return pv.UnstructuredGrid(cells_np, types_np, points)


# ---------------------------------------------------------------------------
# Configuration du rendu (headless / interactif)
# ---------------------------------------------------------------------------

def _setup_renderer(off_screen: bool | None = None) -> bool:
    """Tente de configurer le rendu PyVista.

    Returns
    -------
    is_offscreen : bool
        True si le rendu est en mode hors-écran.
    """
    if off_screen is not None:
        pv.global_theme.backend = "static"
        return off_screen

    # Détecter si un display est disponible
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if not has_display:
        # Essayer xvfb (optionnel)
        try:
            pv.start_xvfb()
            return False
        except Exception:
            pass
        return True
    return False


# ---------------------------------------------------------------------------
# Fonctions publiques
# ---------------------------------------------------------------------------

def _auto_scale(nodes: np.ndarray, u: np.ndarray, target_ratio: float = 0.15) -> float:
    """Facteur d'amplification adapté à la géométrie et aux déplacements."""
    span = np.ptp(nodes, axis=0).max()
    u_max = np.abs(u).max()
    if u_max < 1e-14 or span < 1e-14:
        return 1.0
    return target_ratio * span / u_max


def plot_mesh_3d(
    mesh: Mesh,
    title: str = "Maillage 3D",
    show: bool = True,
    screenshot: str | None = None,
    off_screen: bool | None = None,
) -> None:
    """Afficher le maillage 3D non déformé (fil de fer + faces semi-transparentes).

    Parameters
    ----------
    mesh : Mesh
        Maillage femsolver.
    title : str
        Titre de la fenêtre.
    show : bool
        Si True, ouvre une fenêtre interactive (si possible).
    screenshot : str, optional
        Chemin d'une image PNG à sauvegarder (exemple : ``"mesh.png"``).
        Ignoré si None.
    off_screen : bool, optional
        Force le mode hors-écran. Si None, détecté automatiquement.

    Raises
    ------
    ImportError
        Si PyVista n'est pas installé.
    """
    _require_pyvista()
    is_offscreen = _setup_renderer(off_screen)
    grid = _build_pyvista_grid(mesh)

    pl = pv.Plotter(title=title, off_screen=is_offscreen or (screenshot is not None and not show))
    pl.add_mesh(
        grid,
        style="wireframe",
        color="steelblue",
        line_width=1.5,
        opacity=0.8,
        label="Maillage",
    )
    pl.add_mesh(
        grid.copy(),
        style="surface",
        color="lightsteelblue",
        opacity=0.15,
        show_edges=False,
    )
    pl.add_axes(line_width=2)
    pl.add_bounding_box(line_width=0.5, color="gray")
    _add_stats_text(pl, mesh)

    if screenshot:
        pl.screenshot(screenshot)
        logger.info("Capture sauvegardée : %s", screenshot)

    if show and not is_offscreen:
        pl.show()
    else:
        pl.close()


def plot_deformed_3d(
    mesh: Mesh,
    u: np.ndarray,
    sigma_vm: np.ndarray | None = None,
    sigma_component: np.ndarray | None = None,
    component_label: str = "σ [Pa]",
    scale: float | None = None,
    title: str = "Déformée 3D",
    show: bool = True,
    screenshot: str | None = None,
    off_screen: bool | None = None,
) -> None:
    """Afficher la structure déformée avec une carte de contrainte en couleur.

    Affiche en superposition :
    - La géométrie non déformée en fil de fer gris.
    - La déformée amplifiée avec une carte de couleur (Von Mises ou composante
      choisie).

    Parameters
    ----------
    mesh : Mesh
        Maillage femsolver.
    u : np.ndarray, shape (n_dof,)
        Vecteur de déplacements [m].
    sigma_vm : np.ndarray, shape (n_nodes,), optional
        Contrainte de Von Mises aux nœuds [Pa]. Si fournie, utilisée pour la
        carte de couleur. Priorité sur ``sigma_component``.
    sigma_component : np.ndarray, shape (n_nodes,), optional
        Composante de contrainte aux nœuds [Pa] (exemple : σzz). Utilisée
        uniquement si ``sigma_vm`` est None.
    component_label : str
        Label du champ affiché sur la barre de couleur.
    scale : float, optional
        Facteur d'amplification de la déformée. Si None, calculé
        automatiquement pour que le déplacement max représente ~15 % de la
        dimension caractéristique.
    title : str
        Titre de la fenêtre.
    show : bool
        Si True, ouvre une fenêtre interactive.
    screenshot : str, optional
        Chemin d'image PNG à sauvegarder.
    off_screen : bool, optional
        Force le mode hors-écran.

    Raises
    ------
    ImportError
        Si PyVista n'est pas installé.

    Examples
    --------
    >>> sigma = nodal_stresses_3d(mesh, u)
    >>> vm = von_mises_3d(sigma)
    >>> plot_deformed_3d(mesh, u, sigma_vm=vm, scale=200, screenshot="vm.png")
    """
    _require_pyvista()
    is_offscreen = _setup_renderer(off_screen)

    if scale is None:
        scale = _auto_scale(mesh.nodes, u)

    logger.info("Facteur d'amplification de la déformée : ×%.1f", scale)

    # Grilles
    grid_orig = _build_pyvista_grid(mesh)
    grid_def  = _build_pyvista_grid(mesh, u_displaced=scale * u)

    # Champ de couleur : Von Mises ou composante choisie
    if sigma_vm is not None:
        field = sigma_vm / 1e6   # MPa
        field_label = "σ_VM [MPa]"
    elif sigma_component is not None:
        field = sigma_component / 1e6
        field_label = component_label
    else:
        # Norme du déplacement comme champ de couleur par défaut
        u_nodes = u.reshape(-1, mesh.n_dim)
        field = np.linalg.norm(u_nodes, axis=1) * 1e3   # mm
        field_label = "‖u‖ [mm]"

    grid_def.point_data[field_label] = field

    # --- Plotter ---
    pl = pv.Plotter(
        title=title,
        off_screen=is_offscreen or (screenshot is not None and not show),
    )

    # Configuration de la barre de couleur
    sargs = dict(
        title=field_label,
        title_font_size=14,
        label_font_size=12,
        shadow=True,
        n_labels=5,
        fmt="%.2f",
        vertical=True,
        position_x=0.85,
        position_y=0.15,
    )

    # Géométrie initiale (fil de fer semi-transparent)
    pl.add_mesh(
        grid_orig,
        style="wireframe",
        color="gray",
        line_width=0.8,
        opacity=0.3,
        label="Configuration initiale",
    )

    # Déformée avec carte de couleur
    pl.add_mesh(
        grid_def,
        scalars=field_label,
        cmap="jet",
        show_edges=True,
        edge_color="black",
        line_width=0.5,
        opacity=1.0,
        scalar_bar_args=sargs,
        label="Déformée",
    )

    pl.add_axes(line_width=2)
    _add_stats_text(pl, mesh, u=u, scale=scale, sigma_vm=sigma_vm)

    if screenshot:
        pl.screenshot(screenshot)
        logger.info("Capture sauvegardée : %s", screenshot)

    if show and not is_offscreen:
        pl.show()
    else:
        pl.close()


# ---------------------------------------------------------------------------
# Utilitaires internes
# ---------------------------------------------------------------------------

def _add_stats_text(
    pl: "pv.Plotter",
    mesh: Mesh,
    u: np.ndarray | None = None,
    scale: float | None = None,
    sigma_vm: np.ndarray | None = None,
) -> None:
    """Ajoute un bloc de texte statistique dans le coin inférieur gauche."""
    lines = [
        f"Nœuds  : {mesh.n_nodes}",
        f"Éléments: {len(mesh.elements)}",
        f"DDL    : {mesh.n_dof}",
    ]
    if u is not None:
        u_max_mm = np.abs(u).max() * 1e3
        lines.append(f"u_max  : {u_max_mm:.4f} mm")
    if scale is not None:
        lines.append(f"Ampl.  : ×{scale:.1f}")
    if sigma_vm is not None:
        lines.append(f"σ_VM max: {sigma_vm.max() / 1e6:.2f} MPa")

    pl.add_text(
        "\n".join(lines),
        position="lower_left",
        font_size=9,
        color="white",
    )

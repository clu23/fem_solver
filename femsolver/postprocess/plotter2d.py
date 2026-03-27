"""Visualisation 2D des treillis et structures planes (matplotlib)."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from femsolver.core.mesh import Mesh


def plot_truss(
    mesh: Mesh,
    u: np.ndarray | None = None,
    axial_forces: Sequence[float] | None = None,
    scale: float = 1.0,
    ax: plt.Axes | None = None,
    show: bool = True,
) -> plt.Axes:
    """Afficher un treillis 2D avec la déformée et les efforts normaux.

    Parameters
    ----------
    mesh : Mesh
        Maillage du treillis (nœuds + connectivité barres).
    u : np.ndarray, shape (n_dof,), optional
        Vecteur de déplacements (pour afficher la déformée). Si None, seule
        la configuration initiale est tracée.
    axial_forces : sequence of float, optional
        Effort normal N [N] dans chaque barre (dans l'ordre des éléments).
        Positif = traction (bleu), négatif = compression (rouge).
    scale : float
        Facteur d'amplification de la déformée (pour la visualisation).
    ax : plt.Axes, optional
        Axes matplotlib existants. Si None, une nouvelle figure est créée.
    show : bool
        Si True, appelle plt.show() en fin de fonction.

    Returns
    -------
    ax : plt.Axes
        Axes contenant la figure.

    Examples
    --------
    >>> ax = plot_truss(mesh, u=u_solution, axial_forces=forces, scale=1000)
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    nodes = mesh.nodes  # shape (n_nodes, 2)
    n_dim = mesh.n_dim

    # Configuration déformée
    if u is not None:
        u_xy = u.reshape(-1, n_dim)
        nodes_def = nodes + scale * u_xy
    else:
        nodes_def = nodes

    # Tracé de la configuration initiale (gris pointillé)
    for elem_data in mesh.elements:
        nids = list(elem_data.node_ids)
        x = [nodes[nids[0], 0], nodes[nids[1], 0]]
        y = [nodes[nids[0], 1], nodes[nids[1], 1]]
        ax.plot(x, y, "k--", linewidth=0.8, alpha=0.3)

    # Tracé de la déformée avec couleur selon l'effort
    for idx, elem_data in enumerate(mesh.elements):
        nids = list(elem_data.node_ids)
        x = [nodes_def[nids[0], 0], nodes_def[nids[1], 0]]
        y = [nodes_def[nids[0], 1], nodes_def[nids[1], 1]]

        if axial_forces is not None:
            N = axial_forces[idx]
            color = "tab:blue" if N >= 0 else "tab:red"
            lw = 2.0
            label_text = f"{N/1e3:.2f} kN"
            xm = (x[0] + x[1]) / 2
            ym = (y[0] + y[1]) / 2
            ax.text(xm, ym, label_text, fontsize=7, ha="center", va="bottom",
                    color=color)
        else:
            color = "tab:blue"
            lw = 2.0

        ax.plot(x, y, color=color, linewidth=lw)

    # Nœuds
    ax.plot(nodes_def[:, 0], nodes_def[:, 1], "ko", markersize=5, zorder=5)

    # Numéros de nœuds
    for i, (x, y) in enumerate(nodes_def):
        ax.annotate(str(i), (x, y), textcoords="offset points", xytext=(4, 4),
                    fontsize=8, color="dimgray")

    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", alpha=0.5)

    title_parts = ["Treillis 2D"]
    if u is not None:
        title_parts.append(f"(déformée × {scale})")
    if axial_forces is not None:
        title_parts.append("— bleu=traction, rouge=compression")
    ax.set_title(" ".join(title_parts))
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    if show:
        plt.tight_layout()
        plt.show()

    return ax

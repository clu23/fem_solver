"""Visualisation 2D des treillis et structures planes (matplotlib)."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from femsolver.core.mesh import Mesh


def _auto_scale(nodes: np.ndarray, u: np.ndarray, target_ratio: float = 0.15) -> float:
    """Calcule un facteur d'amplification adapté à la géométrie.

    Le facteur est choisi pour que le déplacement max représente
    ``target_ratio`` de la dimension caractéristique de la structure.

    Parameters
    ----------
    nodes : np.ndarray, shape (n_nodes, 2)
        Coordonnées des nœuds non déformés.
    u : np.ndarray, shape (n_dof,)
        Vecteur de déplacements.
    target_ratio : float
        Fraction de la dimension caractéristique visée pour l'amplification.

    Returns
    -------
    scale : float
    """
    span = max(nodes[:, 0].max() - nodes[:, 0].min(), nodes[:, 1].max() - nodes[:, 1].min())
    u_max = np.abs(u).max()
    if u_max < 1e-14 or span < 1e-14:
        return 1.0
    return target_ratio * span / u_max


def plot_truss(
    mesh: Mesh,
    u: np.ndarray | None = None,
    axial_forces: Sequence[float] | None = None,
    nodal_forces: dict[int, dict[int, float]] | None = None,
    scale: float | None = None,
    ax: plt.Axes | None = None,
    show: bool = True,
) -> plt.Axes:
    """Afficher un treillis 2D avec la déformée et les efforts normaux.

    Parameters
    ----------
    mesh : Mesh
        Maillage du treillis (nœuds + connectivité barres).
    u : np.ndarray, shape (n_dof,), optional
        Vecteur de déplacements (pour afficher la déformée).
    axial_forces : sequence of float, optional
        Effort normal N [N] dans chaque barre (ordre des éléments).
        Positif = traction (bleu), négatif = compression (rouge).
    nodal_forces : dict[int, dict[int, float]], optional
        Forces nodales appliquées à afficher sous forme de flèches vertes.
        Format : ``{node_id: {dof: valeur [N]}}`` — typiquement ``bc.neumann``.
    scale : float, optional
        Facteur d'amplification de la déformée. Si None, calculé automatiquement
        pour que le déplacement max représente ~15 % de la dimension de la structure.
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
    >>> ax = plot_truss(mesh, u=u_solution, axial_forces=forces, nodal_forces=bc.neumann)
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6))

    nodes = mesh.nodes  # shape (n_nodes, 2)
    n_dim = mesh.n_dim

    # --- Calcul de la déformée ---
    if u is not None:
        if scale is None:
            scale = _auto_scale(nodes, u)
        u_xy = u.reshape(-1, n_dim)
        nodes_def = nodes + scale * u_xy
    else:
        scale = scale or 1.0
        nodes_def = nodes

    # --- Configuration initiale (gris pointillé) ---
    for elem_data in mesh.elements:
        nids = list(elem_data.node_ids)
        x0 = [nodes[nids[0], 0], nodes[nids[1], 0]]
        y0 = [nodes[nids[0], 1], nodes[nids[1], 1]]
        ax.plot(x0, y0, "--", color="#aaaaaa", linewidth=1.0, alpha=0.6, zorder=1)

    # --- Déformée avec couleur et labels d'effort ---
    N_abs_max = max((abs(n) for n in axial_forces), default=1.0) if axial_forces else 1.0

    for idx, elem_data in enumerate(mesh.elements):
        nids = list(elem_data.node_ids)
        x = [nodes_def[nids[0], 0], nodes_def[nids[1], 0]]
        y = [nodes_def[nids[0], 1], nodes_def[nids[1], 1]]

        if axial_forces is not None:
            N = axial_forces[idx]
            color = "tab:blue" if N >= 0 else "tab:red"
            lw = 1.5 + 3.0 * abs(N) / N_abs_max
        else:
            color = "tab:blue"
            lw = 2.0

        ax.plot(x, y, color=color, linewidth=lw, solid_capstyle="round", zorder=3)

        if axial_forces is not None:
            # Milieu de la barre déformée
            xm = (x[0] + x[1]) / 2.0
            ym = (y[0] + y[1]) / 2.0

            # Offset perpendiculaire à la barre (évite la superposition avec le trait)
            dx = x[1] - x[0]
            dy = y[1] - y[0]
            length = np.hypot(dx, dy)
            if length > 1e-14:
                # Vecteur perpendiculaire unitaire (rotation +90°)
                nx, ny = -dy / length, dx / length
            else:
                nx, ny = 0.0, 1.0

            # Taille de l'offset = ~4% de la dimension caractéristique
            span = max(nodes[:, 0].max() - nodes[:, 0].min(), nodes[:, 1].max() - nodes[:, 1].min())
            offset = 0.04 * span

            sign = "+" if N >= 0 else ""
            ax.text(
                xm + nx * offset,
                ym + ny * offset,
                f"{sign}{N / 1e3:.1f} kN",
                fontsize=7,
                ha="center",
                va="center",
                color=color,
                fontweight="bold",
                zorder=5,
            )

    # --- Nœuds (numérotés à partir de 1 sur le plot) ---
    ax.plot(nodes_def[:, 0], nodes_def[:, 1], "ko", markersize=5, zorder=6)
    for i, (xn, yn) in enumerate(nodes_def):
        ax.annotate(f"N{i + 1}", (xn, yn), textcoords="offset points", xytext=(5, 4),
                    fontsize=8, color="dimgray", zorder=7)

    # --- Flèches de forces nodales appliquées ---
    if nodal_forces:
        span = max(nodes[:, 0].max() - nodes[:, 0].min(), nodes[:, 1].max() - nodes[:, 1].min())
        arrow_len = 0.12 * span
        for node_id, dof_forces in nodal_forces.items():
            xn, yn = nodes[node_id]
            for dof, fval in dof_forces.items():
                if abs(fval) < 1e-10:
                    continue
                dx_arr = arrow_len * np.sign(fval) if dof == 0 else 0.0
                dy_arr = arrow_len * np.sign(fval) if dof == 1 else 0.0
                ax.annotate(
                    "",
                    xy=(xn + dx_arr, yn + dy_arr),
                    xytext=(xn, yn),
                    arrowprops=dict(arrowstyle="->", color="green", lw=2.0),
                    zorder=8,
                )
                ax.text(
                    xn - dx_arr * 0.15,
                    yn - dy_arr * 0.15,
                    f"{abs(fval) / 1e3:.0f} kN",
                    color="green",
                    fontsize=8,
                    ha="center",
                    va="center",
                    zorder=9,
                )

    # --- Limites des axes : englobent initiale ET déformée avec marge ---
    all_x = np.concatenate([nodes[:, 0], nodes_def[:, 0]])
    all_y = np.concatenate([nodes[:, 1], nodes_def[:, 1]])
    margin_x = max(0.1 * (all_x.max() - all_x.min()), 0.3)
    margin_y = max(0.1 * (all_y.max() - all_y.min()), 0.3)
    ax.set_xlim(all_x.min() - margin_x, all_x.max() + margin_x)
    ax.set_ylim(all_y.min() - margin_y, all_y.max() + margin_y)

    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", alpha=0.4)

    title_parts = ["Treillis 2D"]
    if u is not None:
        title_parts.append(f"(déformée × {scale:.0f})")
    if axial_forces is not None:
        title_parts.append("— bleu = traction, rouge = compression")
    ax.set_title(" ".join(title_parts))
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    if show:
        plt.tight_layout()
        plt.show()

    return ax

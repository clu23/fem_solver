"""Diagrammes d'efforts internes M / V / N (et T) pour les poutres.

Ce module extrait les efforts internes le long de chaque élément poutre
(``Beam2D``, ``Beam2DTimoshenko``, ``Beam3D``) à partir du champ de
déplacements nodaux, puis les trace avec Matplotlib.

Méthode — récupération par l'équilibre
---------------------------------------
Le champ de déplacement éléments finis d'Euler–Bernoulli est cubique
(polynômes de Hermite).  En dérivant ce champ, le moment ``M = EI·v''``
est **linéaire** et l'effort tranchant ``V = EI·v'''`` est **constant** :
ces dérivées ne peuvent donc PAS représenter le moment parabolique et le
tranchant linéaire induits par une charge répartie.

On reconstruit donc les efforts internes par l'**équilibre statique** à
partir des efforts d'extrémité du membre :

    s = f_local − f_eq_local

où ``f_local = K_local · u_local`` (fourni par ``element.section_forces``)
et ``f_eq_local`` est le vecteur de charge réparti équivalent (fixed-end
forces).  La soustraction retire la part « équivalente nodale » de la
charge répartie, de sorte que ``s`` représente les vrais efforts du membre.

Les efforts internes à l'abscisse ``x ∈ [0, L]`` (mesurée depuis le nœud 1
le long de l'axe local) valent alors, pour une charge répartie locale
uniforme ``(qx, qy)`` :

    N(x) = −s_N1 − qx·x                      (linéaire si qx ≠ 0)
    V(x) = −s_V1 − qy·x                      (linéaire si qy ≠ 0)
    M(x) = −s_M1 + s_V1·x + (qy/2)·x²        (parabolique si qy ≠ 0)

Validation analytique — console encastrée-libre de longueur L :

- Charge ponctuelle F en bout : V = F constant, M linéaire (|M|max = F·L).
- Charge répartie q uniforme  : V linéaire (|V|max = q·L),
  M parabolique (|M|max = q·L²/2).

En 3D, le même raisonnement donne en plus l'effort tranchant ``Vz``,
le moment ``My`` et la torsion ``T`` (les charges réparties ne sont pas
encore supportées pour ``Beam3D``, donc qx = qy = qz = 0).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from femsolver.core.mesh import BoundaryConditions, Mesh
from femsolver.elements.beam2d import Beam2D
from femsolver.elements.beam3d import Beam3D

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.figure import Figure


# Ordre des composantes tracées selon la dimension de l'élément.
_COMPONENTS_2D = ("N", "V", "M")
_COMPONENTS_3D = ("N", "Vy", "Vz", "T", "My", "Mz")

# Étiquettes longues pour les axes des graphiques.
_LABELS = {
    "N": "Effort normal N [N]",
    "V": "Effort tranchant V [N]",
    "M": "Moment fléchissant M [N·m]",
    "Vy": "Effort tranchant Vy [N]",
    "Vz": "Effort tranchant Vz [N]",
    "T": "Moment de torsion T [N·m]",
    "My": "Moment fléchissant My [N·m]",
    "Mz": "Moment fléchissant Mz [N·m]",
}


@dataclass(frozen=True)
class BeamDiagram:
    """Efforts internes échantillonnés le long d'un élément poutre.

    Attributes
    ----------
    element_index : int
        Indice de l'élément dans ``mesh.elements``.
    node_ids : tuple[int, int]
        Nœuds de l'élément (ordre local).
    length : float
        Longueur de l'élément [m].
    x_local : np.ndarray, shape (n_points,)
        Abscisses curvilignes ∈ [0, L] depuis le nœud 1.
    x_global : np.ndarray, shape (n_points, n_dim)
        Coordonnées globales des points d'échantillonnage.
    forces : dict[str, np.ndarray]
        Efforts internes par composante.  Clés selon la dimension :
        2D → ``{"N", "V", "M"}`` ; 3D → ``{"N", "Vy", "Vz", "T", "My", "Mz"}``.
        Chaque tableau a la forme ``(n_points,)``.
    """

    element_index: int
    node_ids: tuple[int, int]
    length: float
    x_local: np.ndarray
    x_global: np.ndarray
    forces: dict[str, np.ndarray]

    @property
    def components(self) -> tuple[str, ...]:
        """Composantes disponibles, dans l'ordre canonique d'affichage."""
        ordered = _COMPONENTS_3D if "Vz" in self.forces else _COMPONENTS_2D
        return tuple(c for c in ordered if c in self.forces)


# ---------------------------------------------------------------------------
# Extraction des efforts internes
# ---------------------------------------------------------------------------


def _distributed_lookup(
    bc: BoundaryConditions,
) -> dict[frozenset[int], tuple[float, float]]:
    """Indexe les charges réparties par ensemble de nœuds.

    Parameters
    ----------
    bc : BoundaryConditions
        Conditions aux limites contenant ``bc.distributed``.

    Returns
    -------
    dict[frozenset[int], tuple[float, float]]
        ``{frozenset(node_ids): (qx, qy)}`` en repère local de l'élément.
    """
    lookup: dict[frozenset[int], tuple[float, float]] = {}
    for load in bc.distributed:
        lookup[frozenset(load.node_ids)] = (load.qx, load.qy)
    return lookup


def _section_forces_local(elem, material, coords, properties, u_e) -> np.ndarray:
    """Retourne ``f_local = K_local · u_local`` comme vecteur ordonné.

    Convertit le dictionnaire renvoyé par ``element.section_forces`` en un
    vecteur dont l'ordre suit les DDL locaux de l'élément.

    Parameters
    ----------
    elem : Element
        Instance de l'élément poutre.
    material : ElasticMaterial
    coords : np.ndarray, shape (2, n_dim)
    properties : dict
    u_e : np.ndarray
        Déplacements globaux de l'élément.

    Returns
    -------
    np.ndarray
        6 composantes (2D) ou 12 composantes (3D).
    """
    sf = elem.section_forces(material, coords, properties, u_e)
    if "Vz1" in sf:  # poutre 3D
        keys = ("N1", "Vy1", "Vz1", "Tx1", "My1", "Mz1",
                "N2", "Vy2", "Vz2", "Tx2", "My2", "Mz2")
    else:  # poutre 2D
        keys = ("N1", "V1", "M1", "N2", "V2", "M2")
    return np.array([sf[k] for k in keys], dtype=float)


def _forces_2d(s: np.ndarray, x: np.ndarray, qx: float, qy: float) -> dict:
    """Efforts internes 2D le long de l'élément (récupération par équilibre).

    Parameters
    ----------
    s : np.ndarray, shape (6,)
        Efforts d'extrémité du membre ``f_local − f_eq_local``.
    x : np.ndarray, shape (n_points,)
        Abscisses ∈ [0, L].
    qx, qy : float
        Charge répartie locale uniforme [N/m].

    Returns
    -------
    dict[str, np.ndarray]
        ``{"N", "V", "M"}``.
    """
    n1, v1, m1 = s[0], s[1], s[2]
    return {
        "N": -n1 - qx * x,
        "V": -v1 - qy * x,
        "M": -m1 + v1 * x + 0.5 * qy * x * x,
    }


def _forces_3d(s: np.ndarray, x: np.ndarray) -> dict:
    """Efforts internes 3D le long de l'élément (sans charge répartie).

    Parameters
    ----------
    s : np.ndarray, shape (12,)
        Efforts d'extrémité du membre ``f_local`` (f_eq nul en 3D).
    x : np.ndarray, shape (n_points,)
        Abscisses ∈ [0, L].

    Returns
    -------
    dict[str, np.ndarray]
        ``{"N", "Vy", "Vz", "T", "My", "Mz"}``.
    """
    n1, vy1, vz1, tx1, my1, mz1 = s[:6]
    ones = np.ones_like(x)
    return {
        "N": -n1 * ones,
        "Vy": -vy1 * ones,
        "Vz": -vz1 * ones,
        "T": -tx1 * ones,
        "My": -my1 - vz1 * x,
        "Mz": -mz1 + vy1 * x,
    }


def extract_beam_diagrams(
    mesh: Mesh,
    bc: BoundaryConditions,
    u_full: np.ndarray,
    *,
    n_points: int = 21,
) -> list[BeamDiagram]:
    """Extrait les efforts internes M / V / N (et T) de chaque poutre.

    Parcourt les éléments du maillage, sélectionne les poutres
    (``Beam2D``, ``Beam2DTimoshenko``, ``Beam3D``) et reconstruit les
    efforts internes par l'équilibre (voir le docstring du module).

    Parameters
    ----------
    mesh : Mesh
        Maillage résolu.
    bc : BoundaryConditions
        Conditions aux limites (utilisées pour les charges réparties).
    u_full : np.ndarray, shape (n_dof,)
        Vecteur de déplacements nodaux complet.
    n_points : int, optional
        Nombre de points d'échantillonnage par élément (défaut 21).

    Returns
    -------
    list[BeamDiagram]
        Un diagramme par élément poutre, dans l'ordre du maillage.
        Liste vide si le maillage ne contient aucune poutre.

    Examples
    --------
    >>> diagrams = extract_beam_diagrams(mesh, bc, u_full)
    >>> diagrams[0].forces["M"].max()   # moment max sur le 1er élément  # doctest: +SKIP
    """
    if n_points < 2:
        raise ValueError(f"n_points doit être ≥ 2, reçu {n_points}")

    dist = _distributed_lookup(bc)
    t = np.linspace(0.0, 1.0, n_points)
    diagrams: list[BeamDiagram] = []

    for idx, elem_data in enumerate(mesh.elements):
        etype = elem_data.etype
        is_3d = issubclass(etype, Beam3D)
        is_2d = issubclass(etype, Beam2D)
        if not (is_2d or is_3d):
            continue

        elem = elem_data.get_element()
        node_ids = elem_data.node_ids
        coords = mesh.node_coords(node_ids)
        dofs = mesh.global_dofs(node_ids)
        u_e = u_full[dofs]

        p1, p2 = coords[0], coords[1]
        L = float(np.linalg.norm(p2 - p1))
        x_local = t * L
        x_global = p1[None, :] + np.outer(t, p2 - p1)

        f_local = _section_forces_local(
            elem, elem_data.material, coords, elem_data.properties, u_e
        )

        if is_3d:
            forces = _forces_3d(f_local, x_local)
        else:
            qx, qy = dist.get(frozenset(node_ids), (0.0, 0.0))
            # Vecteur de charge réparti équivalent (fixed-end forces),
            # ordre local [Fx1, Fy1, Mz1, Fx2, Fy2, Mz2].
            f_eq = np.array([
                qx * L / 2.0, qy * L / 2.0, qy * L**2 / 12.0,
                qx * L / 2.0, qy * L / 2.0, -qy * L**2 / 12.0,
            ])
            s = f_local - f_eq
            forces = _forces_2d(s, x_local, qx, qy)

        diagrams.append(BeamDiagram(
            element_index=idx,
            node_ids=(int(node_ids[0]), int(node_ids[1])),
            length=L,
            x_local=x_local,
            x_global=x_global,
            forces=forces,
        ))

    return diagrams


# ---------------------------------------------------------------------------
# Tracé Matplotlib
# ---------------------------------------------------------------------------


def plot_beam_diagrams(
    diagrams: list[BeamDiagram],
    *,
    title: str | None = None,
    show: bool = True,
    savefig: str | None = None,
) -> "Figure":
    """Trace les diagrammes d'efforts internes sur une figure multi-panneaux.

    Les éléments sont concaténés le long d'une abscisse curviligne cumulée
    de sorte qu'une poutre multi-éléments donne des diagrammes continus.
    Une sous-figure (panneau) est créée par composante d'effort.

    Parameters
    ----------
    diagrams : list[BeamDiagram]
        Diagrammes renvoyés par :func:`extract_beam_diagrams`.
    title : str, optional
        Titre global de la figure.
    show : bool, optional
        Si ``True``, appelle ``plt.show()`` (défaut ``True``).
    savefig : str, optional
        Si fourni, enregistre la figure à ce chemin (PNG, PDF…).

    Returns
    -------
    matplotlib.figure.Figure
        La figure créée.

    Raises
    ------
    ValueError
        Si ``diagrams`` est vide.
    """
    if not diagrams:
        raise ValueError(
            "Aucun diagramme à tracer : le modèle ne contient pas de poutre."
        )

    components = diagrams[0].components
    n_panels = len(components)

    fig, axes = plt.subplots(
        n_panels, 1, figsize=(9, 2.4 * n_panels), sharex=True, squeeze=False
    )
    axes = axes[:, 0]

    colors = {"N": "#1b7837", "V": "#2166ac", "M": "#b2182b",
              "Vy": "#2166ac", "Vz": "#4393c3", "T": "#762a83",
              "My": "#b2182b", "Mz": "#d6604d"}

    for ax, comp in zip(axes, components):
        x_offset = 0.0
        for diag in diagrams:
            xs = x_offset + diag.x_local
            ys = diag.forces[comp]
            color = colors.get(comp, "#333333")
            ax.fill_between(xs, ys, 0.0, alpha=0.25, color=color)
            ax.plot(xs, ys, color=color, lw=1.6)
            x_offset += diag.length
        # Séparateurs entre éléments + ligne zéro.
        ax.axhline(0.0, color="black", lw=0.8)
        sep = 0.0
        for diag in diagrams[:-1]:
            sep += diag.length
            ax.axvline(sep, color="gray", lw=0.5, ls=":")
        ax.set_ylabel(_LABELS.get(comp, comp))
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Abscisse curviligne s [m]")
    if title:
        fig.suptitle(title)
    fig.tight_layout()

    if savefig:
        fig.savefig(savefig, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    return fig

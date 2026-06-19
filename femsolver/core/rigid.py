"""Éléments rigides RBE2 / RBE3 exprimés comme contraintes MPC.

Ces « éléments » ne possèdent **pas** de matrice de rigidité : ils sont
traduits en contraintes linéaires multi-points (:class:`MPCConstraint`) que le
solveur applique par multiplicateurs de Lagrange (voir
:func:`femsolver.core.mpc.apply_mpc_lagrange`).

RBE2 — liaison rigide
---------------------
Un nœud **maître** (indépendant) impose un mouvement de corps rigide à N nœuds
**esclaves** (dépendants).  Chaque DDL esclave devient une combinaison linéaire
des DDL du maître.  Cinématique (le maître porte les rotations) :

    u_S = u_M + ω_M × (x_S − x_M)        (translations)
    θ_S = θ_M                            (rotations, si présentes)

C'est un corps infiniment rigide : il **ajoute** de la rigidité (distribution
d'une force/d'un boulon, raccord de maillages incompatibles).

RBE3 — distribution de forces
-----------------------------
Le mouvement d'un nœud de **référence** (dépendant) est la moyenne pondérée des
nœuds **indépendants** :

    u_ref[d] = Σ_i (w_i / Σ w) · u_i[d]

Contrairement au RBE2, le RBE3 n'**ajoute aucune rigidité** : une charge
appliquée au nœud de référence est répartie sur les nœuds indépendants via les
forces de contrainte (les multiplicateurs de Lagrange), sans rigidifier la
structure.  Sert à appliquer une charge répartie depuis un point ou à mesurer
un déplacement moyen.

Convention de DDL
-----------------
Pour ``dpn`` DDL par nœud et ``n_dim`` dimensions spatiales :

- DDL ``0 … n_dim−1``  : translations (ux, uy[, uz]).
- DDL ``n_dim … dpn−1`` : rotations (θz en 2D ; θx, θy, θz en 3D).

Les rotations ne sont présentes que si ``dpn > n_dim`` (poutres : Beam2D dpn=3,
Beam3D dpn=6).  Pour un maillage continu (dpn = n_dim) le RBE2 se réduit à une
égalité de translations (pas de bras de levier en rotation).
"""

from __future__ import annotations

import numpy as np

from femsolver.core.mesh import Mesh, MPCConstraint


def _rotation_dofs(n_dim: int, dpn: int) -> list[int]:
    """Indices locaux des DDL de rotation (``[]`` si translations seules)."""
    if dpn <= n_dim:
        return []
    return list(range(n_dim, dpn))


def _validate_node(mesh: Mesh, node: int, role: str) -> None:
    if not 0 <= node < mesh.n_nodes:
        raise ValueError(
            f"{role} {node} hors limites (maillage à {mesh.n_nodes} nœuds)."
        )


def make_rbe2_constraints(
    mesh: Mesh,
    master: int,
    slaves: tuple[int, ...] | list[int],
    dofs: tuple[int, ...] | list[int] | None = None,
) -> tuple[MPCConstraint, ...]:
    """Construit les contraintes MPC d'une liaison rigide RBE2.

    Chaque DDL esclave est lié au mouvement de corps rigide du maître.  Pour un
    DDL de translation ``d`` du nœud esclave ``S`` (bras de levier
    ``r = x_S − x_M``) :

    - 2D avec rotation (θz) :
        ``ux_S = ux_M − θz_M · ry``,  ``uy_S = uy_M + θz_M · rx``
    - 3D avec rotations (θx, θy, θz), ``ω × r`` :
        ``u_S = u_M + ω_M × r``
    - sans DDL de rotation : ``u_S = u_M`` (translation pure).

    Les DDL de rotation de l'esclave suivent ceux du maître : ``θ_S = θ_M``.

    Le **premier terme** de chaque contrainte est l'esclave (DDL dépendant),
    conformément à la convention de :class:`MPCConstraint`.

    Parameters
    ----------
    mesh : Mesh
        Maillage (coordonnées nodales et ``dpn``).
    master : int
        Indice du nœud maître (indépendant).
    slaves : sequence of int
        Indices des nœuds esclaves (dépendants).
    dofs : sequence of int, optional
        DDL locaux esclaves à contraindre.  Défaut : tous (``range(dpn)``).

    Returns
    -------
    tuple[MPCConstraint, ...]
        Une contrainte par couple (esclave, DDL contraint).

    Raises
    ------
    ValueError
        Si un indice de nœud est hors limites, si un esclave coïncide avec le
        maître, ou si un DDL demandé dépasse ``dpn``.

    Examples
    --------
    Maître 4 rigidifiant les nœuds 0–3 (toutes les composantes) :

    >>> cons = make_rbe2_constraints(mesh, master=4, slaves=[0, 1, 2, 3])
    """
    dpn = mesh.dpn
    n_dim = mesh.n_dim
    coords = mesh.nodes

    _validate_node(mesh, master, "Nœud maître RBE2")
    dof_list = list(range(dpn)) if dofs is None else list(dofs)
    for d in dof_list:
        if not 0 <= d < dpn:
            raise ValueError(
                f"DDL {d} invalide pour RBE2 (dpn={dpn}, attendu 0..{dpn - 1})."
            )

    rot_dofs = _rotation_dofs(n_dim, dpn)
    x_m = coords[master]

    constraints: list[MPCConstraint] = []
    for s in slaves:
        _validate_node(mesh, s, "Nœud esclave RBE2")
        if s == master:
            raise ValueError(
                f"Le nœud esclave {s} ne peut pas être le nœud maître RBE2."
            )
        r = coords[s] - x_m                       # bras de levier (n_dim,)

        for d in dof_list:
            if d in rot_dofs:
                # Rotation : θ_S = θ_M  →  θ_S − θ_M = 0
                terms = ((s, d, 1.0), (master, d, -1.0))
                constraints.append(MPCConstraint(terms=terms, rhs=0.0))
                continue

            # Translation d : u_S[d] = u_M[d] + (ω_M × r)[d]
            terms: list[tuple[int, int, float]] = [
                (s, d, 1.0),
                (master, d, -1.0),
            ]
            if rot_dofs:
                for rdof, coeff in _rigid_rotation_coupling(n_dim, d, r):
                    terms.append((master, rdof, coeff))
            constraints.append(MPCConstraint(terms=tuple(terms), rhs=0.0))

    return tuple(constraints)


def _rigid_rotation_coupling(
    n_dim: int, d: int, r: np.ndarray
) -> list[tuple[int, float]]:
    """Termes ``(dof_rotation_maître, coefficient)`` de ``−(ω × r)[d]``.

    La contrainte translation s'écrit ``u_S[d] − u_M[d] − (ω×r)[d] = 0``.  On
    retourne donc les coefficients de ``−(ω×r)[d]`` portés par les DDL de
    rotation du maître.

    - 2D (ω = θz·ẑ) :
        ``(ω×r) = (−θz·ry, θz·rx)``
        d=0 → −(ω×r)_x = +θz·ry  → coeff θz = +ry
        d=1 → −(ω×r)_y = −θz·rx  → coeff θz = −rx
    - 3D (ω = (θx, θy, θz)) :
        ``(ω×r)_x = θy·rz − θz·ry``
        ``(ω×r)_y = θz·rx − θx·rz``
        ``(ω×r)_z = θx·ry − θy·rx``
    """
    if n_dim == 2:
        rx, ry = float(r[0]), float(r[1])
        theta_z = 2                      # DDL local θz
        if d == 0:
            return [(theta_z, ry)]
        return [(theta_z, -rx)]          # d == 1

    # n_dim == 3 : DDL θx=3, θy=4, θz=5
    rx, ry, rz = float(r[0]), float(r[1]), float(r[2])
    theta_x, theta_y, theta_z = 3, 4, 5
    if d == 0:                           # −(θy·rz − θz·ry)
        return [(theta_y, -rz), (theta_z, ry)]
    if d == 1:                           # −(θz·rx − θx·rz)
        return [(theta_z, -rx), (theta_x, rz)]
    return [(theta_x, -ry), (theta_y, rx)]   # d == 2 : −(θx·ry − θy·rx)


def make_rbe3_constraints(
    mesh: Mesh,
    ref: int,
    nodes: tuple[int, ...] | list[int],
    weights: tuple[float, ...] | list[float] | None = None,
    dofs: tuple[int, ...] | list[int] | None = None,
) -> tuple[MPCConstraint, ...]:
    """Construit les contraintes MPC d'un élément de distribution RBE3.

    Le déplacement du nœud de **référence** (dépendant) est la moyenne pondérée
    des nœuds **indépendants**, DDL par DDL :

    .. math::

        u_\\text{ref}[d] = \\frac{\\sum_i w_i \\, u_i[d]}{\\sum_i w_i}

    soit la contrainte ``u_ref[d] − Σ_i (w_i/Σw)·u_i[d] = 0``.  Le **premier
    terme** est le DDL de référence (dépendant).

    Aucune rigidité n'est ajoutée : une charge au nœud de référence est répartie
    sur les indépendants par les forces de contrainte.

    Parameters
    ----------
    mesh : Mesh
        Maillage.
    ref : int
        Indice du nœud de référence (dépendant).
    nodes : sequence of int
        Indices des nœuds indépendants.
    weights : sequence of float, optional
        Poids ``w_i`` (un par nœud indépendant).  Défaut : poids unitaires
        (moyenne arithmétique).
    dofs : sequence of int, optional
        DDL locaux du nœud de référence à contraindre.  Défaut : tous
        (``range(dpn)``).  Pour chaque DDL contraint, les nœuds indépendants
        doivent posséder ce même DDL.

    Returns
    -------
    tuple[MPCConstraint, ...]
        Une contrainte par DDL contraint du nœud de référence.

    Raises
    ------
    ValueError
        Si la liste des nœuds indépendants est vide, si la longueur des poids ne
        correspond pas, si un poids rend la somme nulle, ou si un indice/DDL est
        invalide.

    Examples
    --------
    Référence 10, moyenne des nœuds 0, 1, 2 (poids égaux) :

    >>> cons = make_rbe3_constraints(mesh, ref=10, nodes=[0, 1, 2])
    """
    dpn = mesh.dpn
    indep = list(nodes)
    if not indep:
        raise ValueError("RBE3 : la liste des nœuds indépendants est vide.")

    _validate_node(mesh, ref, "Nœud de référence RBE3")
    for n in indep:
        _validate_node(mesh, n, "Nœud indépendant RBE3")
        if n == ref:
            raise ValueError(
                f"Le nœud indépendant {n} ne peut pas être le nœud de "
                "référence RBE3."
            )

    if weights is None:
        w = np.ones(len(indep))
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != (len(indep),):
            raise ValueError(
                f"RBE3 : {len(indep)} nœuds indépendants mais {w.size} poids."
            )
    w_sum = float(w.sum())
    if w_sum == 0.0:
        raise ValueError("RBE3 : la somme des poids est nulle.")

    dof_list = list(range(dpn)) if dofs is None else list(dofs)
    for d in dof_list:
        if not 0 <= d < dpn:
            raise ValueError(
                f"DDL {d} invalide pour RBE3 (dpn={dpn}, attendu 0..{dpn - 1})."
            )

    constraints: list[MPCConstraint] = []
    for d in dof_list:
        terms: list[tuple[int, int, float]] = [(ref, d, 1.0)]
        for n, wi in zip(indep, w):
            terms.append((n, d, -float(wi) / w_sum))
        constraints.append(MPCConstraint(terms=tuple(terms), rhs=0.0))

    return tuple(constraints)

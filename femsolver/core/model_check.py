"""Vérification de la santé du modèle avant résolution (style Abaqus).

Philosophie : **vérifier, avertir, bloquer si grave**.  Inspiré du *data
check* d'Abaqus, ce module détecte les défauts de modélisation AVANT toute
résolution, plutôt que de laisser le solveur planter ou rendre des résultats
absurdes.  Aucune correction automatique (pas d'AUTOSPC) : on bloque ou on
avertit, c'est à l'utilisateur de corriger.

Deux niveaux de gravité
------------------------
**Erreurs bloquantes** (lèvent :class:`ModelError` à la résolution) :

1. *Nœuds orphelins* — nœuds non connectés à aucun élément.  Leurs DDL
   n'ont aucune raideur → système singulier.
2. *Jacobien négatif ou nul* — élément retourné (mauvais ordre des nœuds)
   ou dégénéré (nœuds confondus/alignés).  Évalué aux points de Gauss.
3. *Singularité* — après application des BCs, K_free n'est pas factorisable
   (essai de factorisation LDLᵀ, détection des pivots nuls), avec
   identification des nœuds/DDL impliqués.

**Avertissements non bloquants** (loggés, le calcul continue) :

4. *Nœuds coïncidents* — distance < tolérance (souvent un oubli de fusion).
5. *Éléments dupliqués* — même connectivité (rigidité comptée deux fois).
6. *Qualité des éléments* — aspect ratio > 10, angle < 10° ou > 170°.
7. *Conditionnement de K* — estimation ; WARNING si cond(K) > 1e12.

Usage
-----
::

    report = run_model_checks(mesh, bc)
    report.raise_if_errors()      # lève ModelError si erreur bloquante
    # ... les warnings ont déjà été loggés, le calcul peut continuer
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csr_matrix

from femsolver.core.diagnostics import (
    _DOF_DESCRIPTION,
    _dof_labels,
    detect_mechanisms,
)
from femsolver.core.mesh import BoundaryConditions, Mesh, MPCConstraint

logger = logging.getLogger("femsolver.model_check")


# ---------------------------------------------------------------------------
# Seuils par défaut
# ---------------------------------------------------------------------------

COINCIDENT_TOL = 1e-10      # m — distance en deçà de laquelle deux nœuds coïncident
ASPECT_RATIO_MAX = 10.0     # ratio arête max / arête min toléré
ANGLE_MIN_DEG = 10.0        # angle minimal toléré (°)
ANGLE_MAX_DEG = 170.0       # angle maximal toléré (°)
CONDITION_MAX = 1e12        # conditionnement au-delà duquel on avertit
JACOBIAN_REL_TOL = 1e-10    # det(J) ≤ rel_tol · taille^n_dim ⇒ nul/négatif
_SINGULAR_REL_TOL = 1e-9    # pivot LDLᵀ ≤ rel_tol · max|pivot| ⇒ nul
_MAX_DENSE = 1500           # taille max de K_free pour la factorisation dense


# ---------------------------------------------------------------------------
# Exception et structures de rapport
# ---------------------------------------------------------------------------


class ModelError(Exception):
    """Erreur bloquante de modélisation détectée avant résolution."""


@dataclass(frozen=True)
class CheckIssue:
    """Un problème détecté (erreur ou avertissement).

    Attributes
    ----------
    code : str
        Identifiant court (ex. ``"orphan_nodes"``, ``"negative_jacobian"``).
    severity : str
        ``"error"`` (bloquant) ou ``"warning"`` (non bloquant).
    message : str
        Message lisible en français.
    nodes : tuple[int, ...]
        Nœuds concernés (si pertinent).
    elements : tuple[int, ...]
        Indices d'éléments concernés (si pertinent).
    dofs : tuple[int, ...]
        DDL globaux concernés (si pertinent).
    """

    code: str
    severity: str
    message: str
    nodes: tuple[int, ...] = ()
    elements: tuple[int, ...] = ()
    dofs: tuple[int, ...] = ()


@dataclass(frozen=True)
class ModelCheckReport:
    """Résultat de :func:`run_model_checks`.

    Attributes
    ----------
    errors : tuple[CheckIssue, ...]
        Erreurs bloquantes.
    warnings : tuple[CheckIssue, ...]
        Avertissements non bloquants.
    """

    errors: tuple[CheckIssue, ...] = ()
    warnings: tuple[CheckIssue, ...] = ()

    @property
    def ok(self) -> bool:
        """True s'il n'y a aucune erreur bloquante."""
        return len(self.errors) == 0

    def raise_if_errors(self) -> None:
        """Lève :class:`ModelError` si au moins une erreur bloquante existe.

        Raises
        ------
        ModelError
            Message agrégeant toutes les erreurs détectées.
        """
        if not self.errors:
            return
        lines = [f"{len(self.errors)} erreur(s) de modélisation bloquante(s) :"]
        for err in self.errors:
            lines.append(f"  • [{err.code}] {err.message}")
        raise ModelError("\n".join(lines))


# ---------------------------------------------------------------------------
# Registres de topologie (familles d'éléments)
# ---------------------------------------------------------------------------

# Familles continues → (sample = points où évaluer det(J), topologie qualité).
_FAMILY = {
    "Tri3": "tri", "Tri6": "tri",
    "Quad4": "quad", "Quad8": "quad",
    "Tetra4": "tet", "Tetra10": "tet",
    "Hexa8": "hex", "Hexa20": "hex",
}

# Topologie des coins : nombre de coins, arêtes (paires d'indices de coin),
# faces (suites cycliques d'indices de coin) — pour la qualité.
_TOPO = {
    "tri":  dict(n_corners=3, n_dim=2,
                 edges=[(0, 1), (1, 2), (2, 0)],
                 faces=[(0, 1, 2)]),
    "quad": dict(n_corners=4, n_dim=2,
                 edges=[(0, 1), (1, 2), (2, 3), (3, 0)],
                 faces=[(0, 1, 2, 3)]),
    "tet":  dict(n_corners=4, n_dim=3,
                 edges=[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
                 faces=[(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]),
    "hex":  dict(n_corners=8, n_dim=3,
                 edges=[(0, 1), (1, 2), (2, 3), (3, 0),
                        (4, 5), (5, 6), (6, 7), (7, 4),
                        (0, 4), (1, 5), (2, 6), (3, 7)],
                 faces=[(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
                        (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]),
}

_G = 1.0 / np.sqrt(3.0)  # point de Gauss à 2 points

# Points d'échantillonnage pour det(J) (points de Gauss de la famille).
_GAUSS = {
    "quad": [(a, b) for a in (-_G, _G) for b in (-_G, _G)],
    "tri":  [(1.0 / 6, 1.0 / 6), (2.0 / 3, 1.0 / 6), (1.0 / 6, 2.0 / 3)],
    "hex":  [(a, b, c) for a in (-_G, _G) for b in (-_G, _G) for c in (-_G, _G)],
    "tet":  [(0.5854101966, 0.1381966011, 0.1381966011),
             (0.1381966011, 0.5854101966, 0.1381966011),
             (0.1381966011, 0.1381966011, 0.5854101966),
             (0.1381966011, 0.1381966011, 0.1381966011)],
}


# ---------------------------------------------------------------------------
# 1. Nœuds orphelins (erreur)
# ---------------------------------------------------------------------------


def _check_orphan_nodes(
    mesh: Mesh, mpc_nodes: frozenset[int] = frozenset()
) -> list[CheckIssue]:
    """Détecte les nœuds non référencés par aucun élément.

    Un nœud orphelin n'a aucune raideur : ses DDL forment un mécanisme et
    rendent K singulière.  C'est une erreur bloquante.

    Les nœuds participant à une contrainte rigide (RBE2/RBE3, ``mpc_nodes``)
    sont exclus : leur cinématique est imposée par les MPC, ils ne sont donc
    pas réellement orphelins.
    """
    used: set[int] = set(mpc_nodes)
    for elem in mesh.elements:
        used.update(elem.node_ids)
    orphans = sorted(set(range(mesh.n_nodes)) - used)
    if not orphans:
        return []
    preview = ", ".join(str(n) for n in orphans[:10])
    if len(orphans) > 10:
        preview += ", …"
    msg = (f"{len(orphans)} nœud(s) orphelin(s) (non connecté(s) à un "
           f"élément) : {preview}")
    return [CheckIssue("orphan_nodes", "error", msg, nodes=tuple(orphans))]


# ---------------------------------------------------------------------------
# 2. Jacobien négatif ou nul (erreur)
# ---------------------------------------------------------------------------


def _signed_area_tri(nodes: np.ndarray) -> float:
    """Aire signée d'un triangle (positive si nœuds en sens trigonométrique)."""
    x0, y0 = nodes[0]
    x1, y1 = nodes[1]
    x2, y2 = nodes[2]
    return 0.5 * ((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))


def _signed_vol6_tet(nodes: np.ndarray) -> float:
    """6× volume signé d'un tétraèdre (positif si nœuds bien orientés)."""
    e1 = nodes[1] - nodes[0]
    e2 = nodes[2] - nodes[0]
    e3 = nodes[3] - nodes[0]
    return float(np.linalg.det(np.array([e1, e2, e3])))


def _jacobian_determinants(etype, elem, nodes: np.ndarray) -> list[float]:
    """det(J) aux points de Gauss de l'élément continu.

    Pour Tri3 / Tetra4 le Jacobien est constant : on renvoie une mesure
    signée proportionnelle à det(J) (2·aire ou 6·volume).  Pour les autres,
    on évalue ``_shape_function_derivatives`` aux points de Gauss.
    """
    name = etype.__name__
    if name == "Tri3":
        return [2.0 * _signed_area_tri(nodes[:3])]
    if name == "Tetra4":
        return [_signed_vol6_tet(nodes[:4])]

    fam = _FAMILY[name]
    dets: list[float] = []
    for pt in _GAUSS[fam]:
        dN = elem._shape_function_derivatives(*pt)   # (n_dim, n_nodes)
        J = dN @ nodes                               # (n_dim, n_dim)
        dets.append(float(np.linalg.det(J)))
    return dets


def _check_jacobians(mesh: Mesh) -> list[CheckIssue]:
    """Détecte les éléments retournés (det J < 0) ou dégénérés (det J ≈ 0).

    Le déterminant du Jacobien mesure le rapport volume physique / volume
    de référence à chaque point de Gauss.  Il doit être strictement positif :

    - det(J) < 0  → élément **retourné** (nœuds dans le mauvais ordre).
    - det(J) ≈ 0  → élément **dégénéré** (nœuds confondus/alignés/coplanaires).

    La tolérance est relative à la taille de l'élément (arête max)^n_dim,
    pour rester invariante à l'échelle.
    """
    issues: list[CheckIssue] = []
    for idx, elem_data in enumerate(mesh.elements):
        etype = elem_data.etype
        if etype.__name__ not in _FAMILY:
            continue  # éléments structuraux (barre, poutre, ressort) : pas de J
        nodes = mesh.node_coords(elem_data.node_ids)
        # Échelle de référence : (arête caractéristique)^n_dim.
        fam = _FAMILY[etype.__name__]
        n_dim = _TOPO[fam]["n_dim"]
        char = _char_size(nodes)
        scale = char ** n_dim if char > 0 else 1.0
        thr = JACOBIAN_REL_TOL * scale

        dets = _jacobian_determinants(etype, elem_data.get_element(), nodes)
        det_min = min(dets)
        if det_min < -thr:
            msg = (f"Élément {idx} ({etype.__name__}, nœuds "
                   f"{list(elem_data.node_ids)}) : Jacobien négatif "
                   f"(det J = {det_min:.3e}) — élément retourné, vérifier "
                   f"l'ordre des nœuds.")
            issues.append(CheckIssue("negative_jacobian", "error", msg,
                                     elements=(idx,),
                                     nodes=tuple(elem_data.node_ids)))
        elif det_min <= thr:
            msg = (f"Élément {idx} ({etype.__name__}, nœuds "
                   f"{list(elem_data.node_ids)}) : Jacobien nul "
                   f"(det J = {det_min:.3e}) — élément dégénéré "
                   f"(nœuds confondus ou alignés).")
            issues.append(CheckIssue("degenerate_jacobian", "error", msg,
                                     elements=(idx,),
                                     nodes=tuple(elem_data.node_ids)))
    return issues


def _char_size(nodes: np.ndarray) -> float:
    """Taille caractéristique = plus grande distance entre deux nœuds."""
    n = len(nodes)
    cmax = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(nodes[i] - nodes[j]))
            if d > cmax:
                cmax = d
    return cmax


# ---------------------------------------------------------------------------
# 3. Singularité après BCs (erreur) + 7. conditionnement (warning)
# ---------------------------------------------------------------------------


def _describe_dof(mesh: Mesh, global_dof: int) -> str:
    """Décrit un DDL global : ``"Nœud 7 (rotation θz)"``."""
    dpn = mesh.dpn
    node_id = global_dof // dpn
    local = global_dof % dpn
    labels = _dof_labels(mesh)
    lbl = labels[local] if local < len(labels) else f"D{local}"
    _, desc = _DOF_DESCRIPTION.get(lbl, ("ddl", f"ddl {lbl}"))
    return f"Nœud {node_id} ({desc})"


def _ldlt_is_singular(K_free_dense: np.ndarray) -> tuple[bool, int]:
    """Essai de factorisation LDLᵀ : renvoie (singulier, nullité).

    Utilise ``scipy.linalg.ldl`` et compte les pivots nuls de la diagonale
    de D (relativement au plus grand pivot).  Un pivot nul = une direction
    de l'espace nul = une rigidité de corps rigide non bloquée.
    """
    from scipy.linalg import ldl

    _, d, _ = ldl(K_free_dense)
    pivots = np.abs(np.diag(d))
    scale = float(pivots.max()) if pivots.size else 1.0
    if scale <= 0.0:
        scale = 1.0
    nullity = int(np.count_nonzero(pivots <= _SINGULAR_REL_TOL * scale))
    return nullity > 0, nullity


def _null_space_dofs(
    K_free_dense: np.ndarray,
    free_dofs: np.ndarray,
    nullity: int,
    mesh: Mesh,
) -> list[int]:
    """DDL globaux dominant l'espace nul de K_free (via décomposition propre).

    Les pivots LDLᵀ détectent la singularité mais pas le DDL physique
    impliqué (l'ordre d'élimination brouille la correspondance).  On revient
    donc aux vecteurs propres de plus petite valeur propre : le DDL avec la
    plus grande composante dans l'espace nul est « le plus impliqué ».
    """
    vals, vecs = np.linalg.eigh(K_free_dense)
    dofs: list[int] = []
    for k in range(min(nullity, len(vals))):
        v = np.abs(vecs[:, k])
        local = int(np.argmax(v))
        g = int(free_dofs[local])
        if g not in dofs:
            dofs.append(g)
    return dofs


def _check_stiffness(
    mesh: Mesh,
    bc: BoundaryConditions,
    *,
    static_like: bool,
) -> list[CheckIssue]:
    """Singularité (erreur) et conditionnement (warning) de K après BCs.

    Étapes :

    1. Balayage diagonal (``detect_mechanisms``) : DDL libres sans aucune
       raideur (orphelins, rotation oubliée…) — identification exacte.
    2. Essai de factorisation LDLᵀ de K_free pour repérer une déficience de
       rang (mode de corps rigide) que la diagonale ne voit pas.
    3. Si K_free est factorisable : estimation du conditionnement.

    En analyse statique, la singularité est bloquante (système insoluble).
    Pour les autres analyses (modale, flambage…), un noyau peut être
    légitime (modes de corps rigide) : la singularité est alors rétrogradée
    en avertissement.
    """
    from femsolver.core.assembler import Assembler
    from femsolver.core.boundary import apply_dirichlet

    issues: list[CheckIssue] = []

    K = Assembler(mesh).assemble_stiffness()
    severity = "error" if static_like else "warning"

    # 1. DDL libres sans raideur — identification exacte par la diagonale.
    mech = detect_mechanisms(K, mesh, bc)
    if mech.has_mechanism:
        dofs = tuple(mesh.dpn * u.node_id + u.local_dof
                     for u in mech.unconstrained)
        details = " ; ".join(mech.messages())
        msg = (f"Système singulier — {len(mech.unconstrained)} DDL libre(s) "
               f"sans raideur : {details}")
        issues.append(CheckIssue("singular_stiffness", severity, msg,
                                  dofs=dofs))
        # Inutile d'aller plus loin : K est déjà singulière.
        return issues

    # 2. Déficience de rang globale (corps rigide) via factorisation LDLᵀ.
    ds = apply_dirichlet(K, np.zeros(K.shape[0]), mesh, bc)
    free = ds.free_dofs
    if free.size == 0:
        return issues
    K_free = ds.reduce(K)

    if free.size <= _MAX_DENSE:
        dense = K_free.toarray()
        singular, nullity = _ldlt_is_singular(dense)
        if singular:
            bad = _null_space_dofs(dense, free, nullity, mesh)
            desc = " ; ".join(_describe_dof(mesh, g) for g in bad)
            msg = (f"Système singulier — K non factorisable (déficience de "
                   f"rang {nullity}, mode(s) de corps rigide). DDL impliqué(s) : "
                   f"{desc}. Ajoutez des conditions aux limites.")
            issues.append(CheckIssue("singular_stiffness", severity, msg,
                                      dofs=tuple(bad)))
            return issues
        # 3. Conditionnement (uniquement si non singulier).
        cond = _condition_estimate(K_free)
        if cond is not None and cond > CONDITION_MAX:
            msg = (f"Conditionnement élevé de K : cond ≈ {cond:.2e} "
                   f"(> {CONDITION_MAX:.0e}) — résultats peu fiables "
                   f"(unités hétérogènes, raideurs très contrastées ?).")
            issues.append(CheckIssue("ill_conditioned", "warning", msg))
    else:
        # Grand système : on teste la factorisabilité en creux (sans dense).
        singular = _splu_is_singular(K_free)
        if singular:
            msg = ("Système singulier — K_free non factorisable "
                   "(grand système, DDL non identifié). Vérifiez les "
                   "conditions aux limites.")
            issues.append(CheckIssue("singular_stiffness", severity, msg))

    return issues


def _condition_estimate(K_free: csr_matrix) -> float | None:
    """Estimation du conditionnement 1-norme de K_free (style MATLAB condest).

    cond₁(K) = ‖K‖₁ · ‖K⁻¹‖₁, où ‖K⁻¹‖₁ est estimé par ``onenormest`` sur
    l'opérateur de résolution (factorisation LU creuse).  Renvoie None si
    l'estimation échoue (matrice singulière, etc.).
    """
    from scipy.sparse.linalg import LinearOperator, onenormest, splu

    try:
        Kcsc = K_free.tocsc()
        norm_K = onenormest(Kcsc)
        lu = splu(Kcsc)
        n = Kcsc.shape[0]
        op = LinearOperator(
            (n, n),
            matvec=lambda b: lu.solve(b),
            rmatvec=lambda b: lu.solve(b, trans="T"),
        )
        norm_Kinv = onenormest(op)
        return float(norm_K * norm_Kinv)
    except (RuntimeError, ValueError):
        return None


def _splu_is_singular(K_free: csr_matrix) -> bool:
    """Teste la factorisabilité LU creuse d'un grand K_free (sans densifier)."""
    from scipy.sparse.linalg import splu

    try:
        splu(K_free.tocsc())
        return False
    except (RuntimeError, ValueError):
        return True


# ---------------------------------------------------------------------------
# 4. Nœuds coïncidents (warning)
# ---------------------------------------------------------------------------


def _check_coincident_nodes(
    mesh: Mesh, *, tol: float = COINCIDENT_TOL
) -> list[CheckIssue]:
    """Détecte les paires de nœuds quasi confondus (distance < tol).

    Souvent le symptôme d'un maillage non fusionné : deux pièces se
    superposent sans partager leurs nœuds → elles ne sont pas liées.
    Non bloquant (parfois voulu : modélisation de fissures, contacts).
    """
    from scipy.spatial import cKDTree

    if mesh.n_nodes < 2:
        return []
    tree = cKDTree(mesh.nodes)
    pairs = sorted(tree.query_pairs(tol))
    if not pairs:
        return []
    preview = ", ".join(f"({i}, {j})" for i, j in pairs[:8])
    if len(pairs) > 8:
        preview += ", …"
    nodes = tuple(sorted({n for pair in pairs for n in pair}))
    msg = (f"{len(pairs)} paire(s) de nœuds coïncidents (distance < {tol:g} m) "
           f": {preview}. Fusionner si les pièces doivent être liées.")
    return [CheckIssue("coincident_nodes", "warning", msg, nodes=nodes)]


# ---------------------------------------------------------------------------
# 5. Éléments dupliqués (warning)
# ---------------------------------------------------------------------------


def _check_duplicate_elements(mesh: Mesh) -> list[CheckIssue]:
    """Détecte les éléments de même type et même connectivité.

    Deux éléments identiques comptent leur rigidité en double → résultats
    faux localement.  Non bloquant mais presque toujours une erreur.
    """
    seen: dict[tuple, list[int]] = {}
    for idx, elem in enumerate(mesh.elements):
        key = (elem.etype.__name__, frozenset(elem.node_ids))
        seen.setdefault(key, []).append(idx)

    dups = {k: v for k, v in seen.items() if len(v) > 1}
    if not dups:
        return []
    parts = []
    flat: list[int] = []
    for (etname, _nodeset), idxs in dups.items():
        parts.append(f"{etname} {idxs}")
        flat.extend(idxs)
    msg = (f"{len(dups)} groupe(s) d'éléments dupliqués (même connectivité) : "
           f"{' ; '.join(parts[:8])}.")
    return [CheckIssue("duplicate_elements", "warning", msg,
                       elements=tuple(flat))]


# ---------------------------------------------------------------------------
# 6. Qualité des éléments (warning)
# ---------------------------------------------------------------------------


def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle (°) au sommet b entre les arêtes b→a et b→c."""
    u = a - b
    v = c - b
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu < 1e-30 or nv < 1e-30:
        return 0.0
    cos = float(np.dot(u, v) / (nu * nv))
    cos = max(-1.0, min(1.0, cos))
    return float(np.degrees(np.arccos(cos)))


def _check_element_quality(mesh: Mesh) -> list[CheckIssue]:
    """Avertit sur les éléments mal proportionnés.

    Deux indicateurs classiques, calculés sur les **nœuds de coin** :

    - *aspect ratio* = arête la plus longue / la plus courte.  Au-delà de
      10, l'élément est trop allongé (mauvais conditionnement, contraintes
      imprécises).
    - *angles* aux coins de chaque face : un angle < 10° ou > 170° annonce
      un élément « écrasé » (presque dégénéré).
    """
    issues: list[CheckIssue] = []
    for idx, elem_data in enumerate(mesh.elements):
        fam = _FAMILY.get(elem_data.etype.__name__)
        if fam is None:
            continue
        topo = _TOPO[fam]
        nc = topo["n_corners"]
        coords = mesh.node_coords(elem_data.node_ids)[:nc]

        edge_len = [float(np.linalg.norm(coords[i] - coords[j]))
                    for i, j in topo["edges"]]
        emin, emax = min(edge_len), max(edge_len)

        problems: list[str] = []
        if emin > 1e-30:
            ar = emax / emin
            if ar > ASPECT_RATIO_MAX:
                problems.append(f"aspect ratio = {ar:.1f} (> {ASPECT_RATIO_MAX:g})")

        ang_min, ang_max = 180.0, 0.0
        for face in topo["faces"]:
            m = len(face)
            for k in range(m):
                ang = _angle_deg(coords[face[k - 1]], coords[face[k]],
                                 coords[face[(k + 1) % m]])
                ang_min = min(ang_min, ang)
                ang_max = max(ang_max, ang)
        if ang_min < ANGLE_MIN_DEG:
            problems.append(f"angle min = {ang_min:.1f}° (< {ANGLE_MIN_DEG:g}°)")
        if ang_max > ANGLE_MAX_DEG:
            problems.append(f"angle max = {ang_max:.1f}° (> {ANGLE_MAX_DEG:g}°)")

        if problems:
            msg = (f"Élément {idx} ({elem_data.etype.__name__}, nœuds "
                   f"{list(elem_data.node_ids)}) de mauvaise qualité : "
                   f"{', '.join(problems)}.")
            issues.append(CheckIssue("poor_quality", "warning", msg,
                                     elements=(idx,)))
    return issues


# ---------------------------------------------------------------------------
# Orchestrateur
# ---------------------------------------------------------------------------


def run_model_checks(
    mesh: Mesh,
    bc: BoundaryConditions,
    *,
    analysis_type: str = "static",
    mpc: tuple[MPCConstraint, ...] = (),
) -> ModelCheckReport:
    """Exécute toutes les vérifications de santé du modèle.

    Ordre : on traite d'abord les défauts géométriques (orphelins, Jacobien)
    qui empêcheraient l'assemblage.  Les vérifications sur K (singularité,
    conditionnement) ne sont tentées que si la géométrie est saine.

    Les avertissements sont **loggés** (le calcul peut continuer) ; les
    erreurs sont retournées dans le rapport (l'appelant décide de lever
    :class:`ModelError` via :meth:`ModelCheckReport.raise_if_errors`).

    Parameters
    ----------
    mesh : Mesh
        Maillage à vérifier.
    bc : BoundaryConditions
        Conditions aux limites (pour la singularité).
    analysis_type : str
        Type d'analyse.  ``"static"`` rend la singularité bloquante ; pour
        les autres analyses (modale, flambage…) un noyau peut être légitime
        et la singularité devient un simple avertissement.
    mpc : tuple[MPCConstraint, ...]
        Contraintes multi-points (RBE2/RBE3…).  Si non vide : les nœuds
        impliqués sont exclus du test d'orphelins, et le test de singularité
        de K est ignoré (les MPC modifient le système — la rigidité de la
        structure nue n'est plus représentative ; un nœud de référence RBE3
        sans raideur propre est légitimement tenu par les MPC).

    Returns
    -------
    ModelCheckReport
        Rapport listant erreurs et avertissements.  Aucune exception levée.

    Examples
    --------
    >>> report = run_model_checks(mesh, bc)
    >>> report.raise_if_errors()      # lève ModelError si nécessaire
    """
    errors: list[CheckIssue] = []
    warnings: list[CheckIssue] = []

    mpc_nodes = frozenset(
        node_id for c in mpc for node_id, _, _ in c.terms
    )

    # --- Géométrie (préalable à l'assemblage) ---
    geometry_errors = (
        _check_orphan_nodes(mesh, mpc_nodes) + _check_jacobians(mesh)
    )
    errors.extend(geometry_errors)

    warnings.extend(_check_coincident_nodes(mesh))
    warnings.extend(_check_duplicate_elements(mesh))
    warnings.extend(_check_element_quality(mesh))

    # --- Rigidité (seulement si la géométrie permet l'assemblage) ---
    # Les contraintes rigides (RBE2/RBE3) modifient le système : un nœud de
    # référence RBE3 n'a pas de raideur propre mais est tenu par les MPC. Le
    # test de singularité sur la structure nue n'est donc pas représentatif.
    if not geometry_errors and not mpc:
        try:
            stiffness_issues = _check_stiffness(
                mesh, bc, static_like=(analysis_type == "static")
            )
        except Exception as exc:  # noqa: BLE001 — robustesse du pré-check
            logger.warning("Vérification de rigidité interrompue : %s", exc)
            stiffness_issues = []
        for issue in stiffness_issues:
            (errors if issue.severity == "error" else warnings).append(issue)

    report = ModelCheckReport(errors=tuple(errors), warnings=tuple(warnings))
    _log_report(report)
    return report


def _log_report(report: ModelCheckReport) -> None:
    """Journalise les avertissements (WARNING).

    Les erreurs ne sont pas loggées ici : elles sont remontées à l'appelant
    via :meth:`ModelCheckReport.raise_if_errors` (ou affichées par la
    commande CLI ``check``), ce qui évite un double affichage.
    """
    for w in report.warnings:
        logger.warning("%s", w.message)

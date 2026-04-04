"""Diagnostics non-bloquants : masse, réactions d'appui, bilan d'équilibre.

Ces vérifications reproduisent les sorties Nastran ``GPWG``, ``SPCFORCE``
et ``OLOAD`` sous forme de log lisible.  Elles n'interrompent jamais le
calcul — les anomalies sont signalées en ``WARNING``, les résultats en ``INFO``.

Usage typique
-------------
::

    from femsolver.core.diagnostics import run_diagnostics

    assembler = Assembler(mesh)
    K = assembler.assemble_stiffness()
    M = assembler.assemble_mass()
    F = assembler.assemble_forces(bc)
    K_bc, F_bc = apply_dirichlet(K, F, mesh, bc)
    u = StaticSolver().solve(K_bc, F_bc)

    run_diagnostics(mesh, K, u, F, bc, M=M)

Note : ``K`` doit être la matrice **originale** (avant modifications Dirichlet).

Conventions de signe
--------------------
- ``F`` : forces externes appliquées (Neumann, pression, gravité) — positif
  dans le sens de l'action extérieure sur la structure.
- ``R`` : réactions d'appui (SPCFORCE) — défini par ``R = K·u − F`` aux
  DDL contraints.  R positif = la liaison pousse la structure (compression).
- Bilan : ``Σ F_appliqué + Σ R = 0`` dans chaque direction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix

from femsolver.core.mesh import BoundaryConditions, Mesh

logger = logging.getLogger("femsolver.diagnostics")

# Étiquettes des directions / DDL
_DOF_LABELS_2D_TRANS = ("UX", "UY")
_DOF_LABELS_3D_TRANS = ("UX", "UY", "UZ")
_DOF_LABELS_BEAM2D   = ("UX", "UY", "THZ")
_DOF_LABELS_BEAM3D   = ("UX", "UY", "UZ", "THX", "THY", "THZ")

_HR = "=" * 72


# ---------------------------------------------------------------------------
# Dataclass résultats (facilite les tests)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiagnosticsResult:
    """Résultats numériques des diagnostics.

    Attributes
    ----------
    mass_theoretical : float
        Masse théorique calculée depuis la géométrie et le matériau [kg].
    mass_fem : float
        Masse FEM extraite de la matrice de masse (DDL de translation) [kg].
    mass_relative_error : float
        Erreur relative |m_fem − m_th| / m_th.
    center_of_gravity : np.ndarray, shape (n_dim,)
        Centre de gravité théorique [m].
    reactions : dict[int, float]
        Réactions d'appui : ``{global_dof: reaction_value [N ou N·m]}``.
    equilibrium_residuals : np.ndarray, shape (n_dim,)
        Résidu par direction : ``Σ F_appliqué_dir + Σ R_dir`` [N].
    equilibrium_ok : bool
        True si tous les résidus sont inférieurs à la tolérance.
    """

    mass_theoretical : float
    mass_fem         : float
    mass_relative_error: float
    center_of_gravity: np.ndarray
    reactions        : dict[int, float]
    equilibrium_residuals: np.ndarray
    equilibrium_ok   : bool


# ---------------------------------------------------------------------------
# Interne — utilitaires
# ---------------------------------------------------------------------------


def _dof_labels(mesh: Mesh) -> list[str]:
    """Étiquettes des DDL locaux (0..dpn-1) pour ce maillage."""
    n_dim, dpn = mesh.n_dim, mesh.dpn
    if n_dim == 2 and dpn == 2:
        return list(_DOF_LABELS_2D_TRANS)
    if n_dim == 3 and dpn == 3:
        return list(_DOF_LABELS_3D_TRANS)
    if n_dim == 2 and dpn == 3:
        return list(_DOF_LABELS_BEAM2D)
    if n_dim == 3 and dpn == 6:
        return list(_DOF_LABELS_BEAM3D)
    # Fallback générique
    return [f"DOF{d}" for d in range(dpn)]


def _constrained_dofs(mesh: Mesh, bc: BoundaryConditions) -> dict[int, float]:
    """Renvoie {global_dof: valeur_imposée} depuis bc.dirichlet."""
    result: dict[int, float] = {}
    for node_id, dof_values in bc.dirichlet.items():
        for local_dof, value in dof_values.items():
            result[mesh.dpn * node_id + local_dof] = float(value)
    return result


def _translational_dofs(mesh: Mesh) -> list[int]:
    """Indices globaux des DDL de translation (les n_dim premiers par nœud)."""
    dpn, n_dim = mesh.dpn, mesh.n_dim
    return [dpn * i + d for i in range(mesh.n_nodes) for d in range(n_dim)]


# ---------------------------------------------------------------------------
# 1. Vérification de la masse et du centre de gravité
# ---------------------------------------------------------------------------


def _element_mass_and_centroid(
    elem_data,
    nodes_e: np.ndarray,
    n_dim: int,
    M_e: np.ndarray,
    dpn_e: int,
) -> tuple[float, np.ndarray]:
    """Masse et centroïde d'un élément à partir de M_e (DDL de translation).

    Utilise ``M_e_trans.sum() / n_dim`` pour la masse.  Cette identité est
    exacte pour la masse consistante de tout élément isotrope :

        Σ M_e[i,j] (i,j translat.) = n_dim × m_e

    Elle tient aussi pour Beam2D (3 DDL/nœud) car les termes de rotation
    ne couplent pas avec les translations dans la somme globale.

    Parameters
    ----------
    elem_data : ElementData
        Données de l'élément (non utilisées ici, interface uniforme).
    nodes_e : np.ndarray, shape (n_nodes_e, n_dim)
        Coordonnées des nœuds de l'élément.
    n_dim : int
        Dimension spatiale.
    M_e : np.ndarray, shape (n_dof_e, n_dof_e)
        Matrice de masse élémentaire.
    dpn_e : int
        DDL par nœud de cet élément.

    Returns
    -------
    m_e : float
        Masse de l'élément [kg].
    centroid : np.ndarray, shape (n_dim,)
        Centroïde géométrique de l'élément [m].
    """
    n_nodes_e = len(nodes_e)
    # Indices DDL de translation dans la matrice élémentaire (local)
    trans_local = [dpn_e * k + d for k in range(n_nodes_e) for d in range(n_dim)]
    M_trans = M_e[np.ix_(trans_local, trans_local)]
    m_e = float(M_trans.sum()) / n_dim
    centroid = nodes_e.mean(axis=0)
    return m_e, centroid


def check_mass(
    mesh: Mesh,
    M: csr_matrix,
    *,
    tol: float = 1e-4,
) -> tuple[float, float, np.ndarray]:
    """Vérifie la masse totale FEM et calcule le centre de gravité.

    La masse FEM est extraite de la matrice de masse assemblée en utilisant
    uniquement les DDL de translation :

        m_FEM = M[trans_dofs, trans_dofs].sum() / n_dim

    La masse théorique est calculée depuis les masses élémentaires (même
    formule appliquée aux matrices élémentaires), cohérente par construction.

    Parameters
    ----------
    mesh : Mesh
        Maillage.
    M : csr_matrix, shape (n_dof, n_dof)
        Matrice de masse assemblée.
    tol : float
        Tolérance sur l'erreur relative pour déclencher un WARNING.

    Returns
    -------
    m_theoretical : float
        Masse théorique (somme des masses élémentaires) [kg].
    m_fem : float
        Masse FEM depuis M [kg].
    cg : np.ndarray, shape (n_dim,)
        Centre de gravité théorique [m].

    Notes
    -----
    Log ``WARNING`` si |m_fem − m_th| / m_th > tol.
    """
    n_dim = mesh.n_dim
    dpn   = mesh.dpn

    # Masse FEM globale depuis M (DDL de translation uniquement)
    trans_dofs = _translational_dofs(mesh)
    M_trans = M[np.ix_(trans_dofs, trans_dofs)]
    m_fem = float(M_trans.sum()) / n_dim

    # Masse théorique + CG depuis les matrices élémentaires
    m_total = 0.0
    cg = np.zeros(n_dim)
    for elem_data in mesh.elements:
        elem = elem_data.get_element()
        nodes_e = mesh.node_coords(elem_data.node_ids)
        M_e = elem.mass_matrix(elem_data.material, nodes_e, elem_data.properties)
        dpn_e = elem.dof_per_node()
        m_e, centroid = _element_mass_and_centroid(
            elem_data, nodes_e, n_dim, M_e, dpn_e
        )
        m_total += m_e
        cg += m_e * centroid

    if m_total > 0.0:
        cg /= m_total

    rel_err = abs(m_fem - m_total) / m_total if m_total > 0.0 else 0.0

    if rel_err > tol:
        logger.warning(
            "MASSE : erreur relative %.2e > tolérance %.2e "
            "(m_FEM=%.6g kg, m_théo=%.6g kg)",
            rel_err, tol, m_fem, m_total,
        )

    return m_total, m_fem, cg


# ---------------------------------------------------------------------------
# 2. Forces de réaction aux appuis (SPCFORCE)
# ---------------------------------------------------------------------------


def compute_reactions(
    K: csr_matrix,
    u: np.ndarray,
    F: np.ndarray,
    mesh: Mesh,
    bc: BoundaryConditions,
) -> dict[int, float]:
    """Calcule les réactions d'appui R = K·u − F aux DDL contraints.

    Analogue au SPCFORCE de Nastran : pour chaque DDL bloqué, la réaction
    est la force que l'appui exerce sur la structure pour maintenir le
    déplacement imposé.

    À l'équilibre :

        K·u = F_external + R_reactions  (sur tout le système)

    Aux DDL libres : ``(K·u − F)[free] ≈ 0`` (résidu d'équilibre).
    Aux DDL contraints : ``(K·u − F)[s] = R[s]`` (réaction d'appui).

    Parameters
    ----------
    K : csr_matrix, shape (n_dof, n_dof)
        Matrice de rigidité **originale** (avant modification Dirichlet).
    u : np.ndarray, shape (n_dof,)
        Vecteur de déplacements solution.
    F : np.ndarray, shape (n_dof,)
        Vecteur de forces externes assemblé (avant modification Dirichlet).
    mesh : Mesh
        Maillage.
    bc : BoundaryConditions
        Conditions aux limites.

    Returns
    -------
    reactions : dict[int, float]
        ``{global_dof: reaction [N or N·m]}``.  Seuls les DDL contraints
        sont présents.

    Notes
    -----
    Signe : une réaction positive signifie que l'appui pousse la structure
    dans le sens positif du DDL (convention R = K·u − F).
    """
    constrained = _constrained_dofs(mesh, bc)
    residual = K @ u - F          # shape (n_dof,)
    return {dof: float(residual[dof]) for dof in constrained}


# ---------------------------------------------------------------------------
# 3. Bilan d'équilibre global (Newton 3)
# ---------------------------------------------------------------------------


def check_equilibrium(
    F: np.ndarray,
    reactions: dict[int, float],
    mesh: Mesh,
    *,
    tol: float = 1e-6,
) -> tuple[np.ndarray, bool]:
    """Vérifie le bilan des forces dans chaque direction.

    Newton 3 : la somme des forces appliquées et des réactions doit être
    nulle dans chaque direction de translation.

        residual_d = Σ_tous_nœuds F[dpn·i+d] + Σ_contraints R[dpn·i+d]

    La tolérance relative est calculée par rapport à la force maximale :

        |residual_d| / max(|Σ F_d|, |Σ R_d|, 1.0) < tol

    Parameters
    ----------
    F : np.ndarray, shape (n_dof,)
        Forces externes appliquées (avant Dirichlet).
    reactions : dict[int, float]
        Réactions aux DDL contraints (sortie de :func:`compute_reactions`).
    mesh : Mesh
        Maillage.
    tol : float
        Tolérance sur le résidu relatif pour déclencher un WARNING.

    Returns
    -------
    residuals : np.ndarray, shape (n_dim,)
        Résidu par direction [N].
    ok : bool
        True si tous les résidus relatifs sont inférieurs à ``tol``.
    """
    n_dim, dpn = mesh.n_dim, mesh.dpn

    sum_applied   = np.zeros(n_dim)
    sum_reactions = np.zeros(n_dim)

    for i in range(mesh.n_nodes):
        for d in range(n_dim):
            g = dpn * i + d
            sum_applied[d] += F[g]

    for dof, r_val in reactions.items():
        local_d = dof % dpn
        if local_d < n_dim:         # seulement les DDL de translation
            sum_reactions[local_d] += r_val

    residuals = sum_applied + sum_reactions
    force_scale = np.maximum(np.abs(sum_applied), np.abs(sum_reactions))
    force_scale = np.where(force_scale < 1.0, 1.0, force_scale)
    rel_res = np.abs(residuals) / force_scale

    ok = bool(np.all(rel_res < tol))

    if not ok:
        for d in range(n_dim):
            if rel_res[d] >= tol:
                logger.warning(
                    "ÉQUILIBRE direction %d : résidu = %.4e N "
                    "(relatif = %.2e > tolérance %.2e)",
                    d, residuals[d], rel_res[d], tol,
                )

    return residuals, ok


# ---------------------------------------------------------------------------
# 4. Rapport complet (Nastran-style)
# ---------------------------------------------------------------------------


def run_diagnostics(
    mesh: Mesh,
    K: csr_matrix,
    u: np.ndarray,
    F: np.ndarray,
    bc: BoundaryConditions,
    M: csr_matrix | None = None,
    *,
    tol_mass: float = 1e-4,
    tol_equil: float = 1e-6,
) -> DiagnosticsResult:
    """Génère un rapport complet de diagnostics FEM (style listing Nastran).

    Enchaîne les trois vérifications dans l'ordre :
    1. Masse totale + centre de gravité (si M fourni).
    2. Réactions d'appui (SPCFORCE).
    3. Bilan d'équilibre global (Newton 3).

    Les résultats sont loggés au niveau ``INFO`` ; les anomalies au niveau
    ``WARNING``.  Aucune exception n'est jamais levée.

    Parameters
    ----------
    mesh : Mesh
        Maillage.
    K : csr_matrix, shape (n_dof, n_dof)
        Matrice de rigidité **originale** (avant Dirichlet).
    u : np.ndarray, shape (n_dof,)
        Déplacements solution.
    F : np.ndarray, shape (n_dof,)
        Forces externes assemblées (avant Dirichlet).
    bc : BoundaryConditions
        Conditions aux limites.
    M : csr_matrix or None
        Matrice de masse.  Si None, la vérification de masse est sautée.
    tol_mass : float
        Tolérance relative pour la vérification de masse.
    tol_equil : float
        Tolérance relative pour l'équilibre (résidu / max_force).

    Returns
    -------
    DiagnosticsResult
        Résultats numériques (utile pour les tests).

    Examples
    --------
    >>> result = run_diagnostics(mesh, K, u, F, bc, M=M)
    >>> assert result.equilibrium_ok
    >>> assert result.mass_relative_error < 1e-6
    """
    n_dim = mesh.n_dim
    dpn   = mesh.dpn
    labels = _dof_labels(mesh)

    lines: list[str] = [
        "",
        _HR,
        "        F E M   D I A G N O S T I C S",
        _HR,
    ]

    # ------------------------------------------------------------------
    # 1. Masse et centre de gravité
    # ------------------------------------------------------------------
    m_theo = m_fem = 0.0
    rel_err = 0.0
    cg = np.zeros(n_dim)

    if M is not None:
        m_theo, m_fem, cg = check_mass(mesh, M, tol=tol_mass)
        rel_err = abs(m_fem - m_theo) / m_theo if m_theo > 0.0 else 0.0
        mass_status = "OK" if rel_err <= tol_mass else "** WARN **"

        lines += [
            "",
            "--- MASS CHECK (GPWG) " + "-" * 50,
            f"  Theoretical mass  :  {m_theo:>14.6e} kg",
            f"  FEM mass (M/n_dim):  {m_fem:>14.6e} kg",
            f"  Relative error    :  {rel_err:>14.2e}          {mass_status}",
            "",
            "  Center of gravity (theoretical) :",
        ]
        dir_names = ("X", "Y", "Z")
        for d in range(n_dim):
            lines.append(f"    {dir_names[d]} = {cg[d]:>12.4e} m")
    else:
        lines += ["", "--- MASS CHECK  (skipped — M not provided) ---"]

    # ------------------------------------------------------------------
    # 2. Réactions d'appui (SPCFORCE)
    # ------------------------------------------------------------------
    reactions = compute_reactions(K, u, F, mesh, bc)

    lines += [
        "",
        "--- SPCFORCES (Reaction forces at constrained DOFs) " + "-" * 19,
        f"  {'Point':>6}  {'DOF':<6}  {'Reaction [N or N.m]':>22}",
        "  " + "-" * 40,
    ]

    constrained = _constrained_dofs(mesh, bc)
    # Regrouper les DDL par nœud pour un affichage compact
    node_rxn: dict[int, dict[int, float]] = {}
    for dof, r_val in sorted(reactions.items()):
        node_id  = dof // dpn
        local_d  = dof %  dpn
        node_rxn.setdefault(node_id, {})[local_d] = r_val

    for node_id in sorted(node_rxn):
        for local_d, r_val in sorted(node_rxn[node_id].items()):
            lbl = labels[local_d] if local_d < len(labels) else f"D{local_d}"
            lines.append(f"  {node_id:>6}  {lbl:<6}  {r_val:>+22.6e}")

    if not reactions:
        lines.append("  (aucun DDL contraint)")

    # ------------------------------------------------------------------
    # 3. Bilan d'équilibre
    # ------------------------------------------------------------------
    residuals, ok = check_equilibrium(F, reactions, mesh, tol=tol_equil)

    dir_names = ("X", "Y", "Z")
    sum_applied   = np.zeros(n_dim)
    sum_reactions = np.zeros(n_dim)
    for i in range(mesh.n_nodes):
        for d in range(n_dim):
            sum_applied[d] += F[dpn * i + d]
    for dof, r_val in reactions.items():
        ld = dof % dpn
        if ld < n_dim:
            sum_reactions[ld] += r_val

    lines += [
        "",
        "--- EQUILIBRIUM CHECK (Σ Applied + Σ Reactions = 0) " + "-" * 18,
        f"  {'Dir':<4}  {'Applied [N]':>16}  {'Reaction [N]':>16}  "
        f"{'Residual [N]':>16}  {'Status':<10}",
        "  " + "-" * 72,
    ]
    equil_ok_per_dir = []
    for d in range(n_dim):
        fa = sum_applied[d]
        fr = sum_reactions[d]
        res = residuals[d]
        scale = max(abs(fa), abs(fr), 1.0)
        rel = abs(res) / scale
        status = "OK" if rel < tol_equil else "** WARN **"
        equil_ok_per_dir.append(rel < tol_equil)
        lines.append(
            f"  {dir_names[d]:<4}  {fa:>+16.6e}  {fr:>+16.6e}  "
            f"{res:>+16.6e}  {status:<10}"
        )

    lines += ["", _HR, ""]

    # ------------------------------------------------------------------
    # Log du rapport complet
    # ------------------------------------------------------------------
    report = "\n".join(lines)
    logger.info(report)

    return DiagnosticsResult(
        mass_theoretical    = m_theo,
        mass_fem            = m_fem,
        mass_relative_error = rel_err,
        center_of_gravity   = cg,
        reactions           = reactions,
        equilibrium_residuals = residuals,
        equilibrium_ok      = ok,
    )

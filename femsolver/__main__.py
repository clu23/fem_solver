"""CLI pour fem-solver.

Usage
-----
    python -m femsolver run      model.json [--detailed] [--export out.vtu]
                                            [--diagrams out.png] [--quiet]
    python -m femsolver validate model.json
    python -m femsolver check    model.json
    python -m femsolver info     model.json

Commandes
---------
run
    Charge le modèle JSON, lance le calcul et affiche un résumé des résultats.
    Pour les analyses statiques, affiche les 5 nœuds les plus déplacés,
    les réactions d'appui, le bilan d'équilibre, et pour les treillis les
    barres les plus sollicitées en traction/compression.  La santé du modèle
    est vérifiée automatiquement avant la résolution (cf. ``check``).

validate
    Vérifie que le JSON est syntaxiquement correct et que le modèle peut être
    construit (types d'éléments, matériaux, sections…), sans lancer aucun calcul.

check
    Vérifie la santé du modèle (style Abaqus) : nœuds orphelins, Jacobien
    négatif/nul, singularité après BCs (erreurs bloquantes) ; nœuds coïncidents,
    éléments dupliqués, mauvaise qualité, conditionnement (avertissements).

info
    Affiche un résumé du modèle : nœuds, éléments, DDL, conditions aux limites
    et type d'analyse — sans calcul.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Helpers d'affichage
# ---------------------------------------------------------------------------

def _header(title: str) -> None:
    width = 62
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def _section(title: str) -> None:
    print(f"\n  {'─' * 56}")
    print(f"  {title}")


def _row(label: str, value: Any, indent: int = 4) -> None:
    pad = " " * indent
    print(f"{pad}{label:<28} {value}")


def _ok(msg: str) -> None:
    print(f"  \033[32m✓\033[0m  {msg}")


def _err(msg: str) -> None:
    print(f"  \033[31m✗\033[0m  {msg}", file=sys.stderr)


def _warn(msg: str) -> None:
    print(f"  \033[33m⚠\033[0m  {msg}")


# ---------------------------------------------------------------------------
# Calculs post-traitement statique
# ---------------------------------------------------------------------------

def _compute_reactions(mesh, bc, u_full: np.ndarray) -> np.ndarray:
    """Retourne R = K_orig @ u_full − F_ext (réactions aux appuis).

    Parameters
    ----------
    mesh : Mesh
    bc : BoundaryConditions
    u_full : np.ndarray, shape (n_dof,)

    Returns
    -------
    R : np.ndarray, shape (n_dof,)
        Non nul uniquement aux DDL contraints.
    """
    from femsolver.core.assembler import Assembler
    assembler = Assembler(mesh)
    K = assembler.assemble_stiffness()
    F = assembler.assemble_forces(bc)
    return K @ u_full - F


def _top_displaced_nodes(mesh, u_full: np.ndarray, n: int = 5) -> list[tuple]:
    """Retourne les n nœuds les plus déplacés.

    Returns
    -------
    list of (node_id, magnitude, coords, u_components)
    """
    dpn = mesh.dpn
    nd = mesh.n_dim
    n_nodes = mesh.n_nodes
    # Magnitude sur les DDL de translation uniquement
    mag = np.zeros(n_nodes)
    for i in range(n_nodes):
        u_trans = u_full[i * dpn: i * dpn + nd]
        mag[i] = float(np.linalg.norm(u_trans))
    idx = np.argsort(mag)[::-1][:n]
    result = []
    for i in idx:
        u_comp = u_full[i * dpn: i * dpn + dpn]
        result.append((int(i), float(mag[i]), mesh.nodes[i].tolist(), u_comp.tolist()))
    return result


def _bar_forces(mesh, u_full: np.ndarray) -> list[tuple] | None:
    """Retourne les efforts normaux de chaque barre (treillis uniquement).

    Returns
    -------
    list of (bar_idx, N_force, node_ids) or None si non applicable.
    """
    from femsolver.elements.bar2d import Bar2D
    if not all(e.etype is Bar2D for e in mesh.elements):
        return None
    bar = Bar2D()
    forces = []
    for idx, elem in enumerate(mesh.elements):
        nids = elem.node_ids
        coords = mesh.node_coords(nids)
        dofs = mesh.global_dofs(nids)
        u_e = u_full[dofs]
        area = float(elem.properties["area"])
        N = bar.axial_force(elem.material, coords, area, u_e)
        forces.append((idx, float(N), nids))
    return forces


def _equilibrium(mesh, bc, u_full: np.ndarray, R: np.ndarray) -> list[tuple]:
    """Calcule le bilan d'équilibre par direction.

    Returns
    -------
    list of (axis_label, F_applied, F_reaction, residual)
    """
    from femsolver.core.assembler import Assembler
    assembler = Assembler(mesh)
    F_ext = assembler.assemble_forces(bc)
    dpn = mesh.dpn
    nd = mesh.n_dim
    labels = ["x", "y", "z"][:nd]
    rows = []
    for d, lbl in enumerate(labels):
        indices = list(range(d, mesh.n_dof, dpn))
        F_applied = float(np.sum(F_ext[indices]))
        F_reaction = float(np.sum(R[indices]))
        residual = F_applied + F_reaction
        rows.append((lbl, F_applied, F_reaction, residual))
    return rows


# ---------------------------------------------------------------------------
# Affichage statique enrichi
# ---------------------------------------------------------------------------

def _dof_labels(n_dim: int, dpn: int) -> list[str]:
    """Retourne les noms des DDL selon la dimension et le dpn.

    Mapping :
    - (2, 2) → ux, uy             (Bar2D, continuum 2D)
    - (2, 3) → ux, uy, θz         (Beam2D)
    - (3, 3) → ux, uy, uz         (continuum 3D)
    - (3, 6) → ux, uy, uz, θx, θy, θz  (Beam3D)
    """
    trans = ["ux", "uy", "uz"][:n_dim]
    rot   = ["θx", "θy", "θz"]
    if dpn == n_dim:
        return trans
    if n_dim == 2 and dpn == 3:
        return ["ux", "uy", "θz"]
    if n_dim == 3 and dpn == 6:
        return trans + rot
    # Fallback générique
    return (trans + rot)[:dpn]

def _print_static_enriched(
    model: Any,
    u_full: np.ndarray,
    *,
    detailed: bool,
) -> None:
    """Affiche le résumé enrichi de l'analyse statique."""
    mesh = model.mesh
    bc = model.bc
    dpn = mesh.dpn
    nd = mesh.n_dim
    n_nodes = mesh.n_nodes

    # --- Réactions (assemblage complet requis) ---
    R = _compute_reactions(mesh, bc, u_full)

    # 1. Statistiques globales
    _section("Déplacements nodaux")
    mag = np.array([
        float(np.linalg.norm(u_full[i * dpn: i * dpn + nd]))
        for i in range(n_nodes)
    ])
    _row("u_max absolu", f"{mag.max():.4e} m")
    _row("u_moy absolu", f"{mag.mean():.4e} m")
    _row("Norme L2", f"{float(np.linalg.norm(u_full)):.4e} m")

    # 2. Top-5 nœuds les plus déplacés
    top = _top_displaced_nodes(mesh, u_full, n=5)
    coord_labels = ["x", "y", "z"][:nd]
    comp_labels  = _dof_labels(nd, dpn)
    print()
    header = (
        f"    {'Nœud':>5}  "
        + "  ".join(f"{l:>8}" for l in coord_labels)
        + "  "
        + "  ".join(f"{l:>10}" for l in comp_labels)
        + f"  {'|u|':>10}"
    )
    print("  Top 5 nœuds les plus déplacés :")
    print(header)
    print("    " + "─" * (len(header) - 4))
    for node_id, mag_i, coords, u_comp in top:
        coords_str = "  ".join(f"{c:8.3f}" for c in coords)
        u_str = "  ".join(f"{v:10.3e}" for v in u_comp)
        print(f"    {node_id:>5}  {coords_str}  {u_str}  {mag_i:10.3e}")

    # 3. Tableau complet (--detailed)
    if detailed:
        _section("Tableau complet des déplacements")
        print(f"    {'Nœud':>5}  " + "  ".join(f"{l:>8}" for l in coord_labels)
              + "  " + "  ".join(f"{l:>10}" for l in comp_labels) + f"  {'|u|':>10}")
        print("    " + "─" * 60)
        for i in range(n_nodes):
            coords = mesh.nodes[i].tolist()
            u_comp = u_full[i * dpn: i * dpn + dpn].tolist()
            mag_i = float(np.linalg.norm(u_full[i * dpn: i * dpn + nd]))
            coords_str = "  ".join(f"{c:8.3f}" for c in coords)
            u_str = "  ".join(f"{v:10.3e}" for v in u_comp)
            print(f"    {i:>5}  {coords_str}  {u_str}  {mag_i:10.3e}")

    # 4. Réactions d'appui
    _section("Réactions d'appui")
    constrained: list[tuple[int, int, int]] = []
    for node_id, dof_dict in bc.dirichlet.items():
        for local_dof in dof_dict:
            g = node_id * dpn + local_dof
            constrained.append((node_id, local_dof, g))
    constrained.sort()

    if not constrained:
        print("    (aucun DDL contraint)")
    else:
        r_labels = _dof_labels(nd, dpn)
        print(f"    {'Nœud':>5}  {'DDL':>4}  {'Réaction [N]':>14}")
        print("    " + "─" * 28)
        for node_id, local_dof, g in constrained:
            lbl = r_labels[local_dof] if local_dof < len(r_labels) else str(local_dof)
            rv = float(R[g])
            print(f"    {node_id:>5}  {lbl:>4}  {rv:14.4e}")

    # 5. Bilan d'équilibre
    _section("Bilan d'équilibre")
    eq_rows = _equilibrium(mesh, bc, u_full, R)
    print(f"    {'Dir':>4}  {'F_appliquée [N]':>18}  {'F_réaction [N]':>18}  {'Résidu [N]':>14}")
    print("    " + "─" * 60)
    for lbl, F_a, F_r, res in eq_rows:
        ok = "✓" if abs(res) < max(1e-6, 1e-6 * (abs(F_a) + abs(F_r))) else "✗"
        print(f"    {lbl:>4}  {F_a:18.4e}  {F_r:18.4e}  {res:14.4e}  {ok}")

    # 6. Barres (treillis)
    forces = _bar_forces(mesh, u_full)
    if forces is not None:
        _section("Efforts dans les barres (treillis)")
        tension     = sorted([(N, i, nids) for i, N, nids in forces if N >= 0],
                             reverse=True)[:5]
        compression = sorted([(N, i, nids) for i, N, nids in forces if N < 0])[:5]

        if tension:
            print("  Top 5 traction :")
            print(f"    {'Barre':>6}  {'Nœuds':>12}  {'N [N]':>14}")
            print("    " + "─" * 38)
            for N, idx, nids in tension:
                print(f"    {idx:>6}  {str(nids):>12}  {N:14.4e}")
        if compression:
            print()
            print("  Top 5 compression :")
            print(f"    {'Barre':>6}  {'Nœuds':>12}  {'N [N]':>14}")
            print("    " + "─" * 38)
            for N, idx, nids in compression:
                print(f"    {idx:>6}  {str(nids):>12}  {N:14.4e}")


# ---------------------------------------------------------------------------
# Export VTU
# ---------------------------------------------------------------------------

def _export_vtu(export_path: str, model: Any, results: dict, *, quiet: bool = False) -> None:
    """Exporte les résultats en VTU (ParaView).

    Parameters
    ----------
    export_path : str
    model : FEModel
    results : dict
        Dict retourné par solve_model.
    """
    from femsolver.io.mesh_io import write_vtu
    atype = results.get("analysis_type", "static")
    mesh = model.mesh

    u = None
    if atype == "static" and "u" in results:
        u = np.array(results["u"])

    try:
        write_vtu(export_path, mesh, u=u)
        if not quiet:
            _ok(f"Exporté → {export_path}")
    except (ValueError, ImportError) as exc:
        _err(f"Export VTU échoué : {exc}")


# ---------------------------------------------------------------------------
# Diagrammes d'efforts internes (poutres)
# ---------------------------------------------------------------------------

def _plot_diagrams(
    diagrams_path: str,
    model: Any,
    results: dict,
    *,
    quiet: bool = False,
) -> None:
    """Trace les diagrammes d'efforts internes M/V/N des poutres.

    Parameters
    ----------
    diagrams_path : str
        Chemin de sortie PNG (déjà résolu par l'appelant).
    model : FEModel
    results : dict
        Résultats retournés par ``solve_model`` (doit contenir ``"u"``).
    quiet : bool
    """
    atype = results.get("analysis_type", "static")
    if atype != "static" or "u" not in results:
        _warn("--diagrams n'est disponible que pour l'analyse statique.")
        return

    from femsolver.postprocess.beam_diagrams import (
        extract_beam_diagrams,
        plot_beam_diagrams,
    )

    u_full = np.array(results["u"])
    diagrams = extract_beam_diagrams(model.mesh, model.bc, u_full)
    if not diagrams:
        _warn("Aucun élément poutre dans le modèle — pas de diagramme.")
        return

    try:
        plot_beam_diagrams(
            diagrams, title=model.name, show=False, savefig=diagrams_path
        )
        if not quiet:
            _ok(f"Diagrammes M/V/N → {diagrams_path}  "
                f"({len(diagrams)} élément(s) poutre)")
    except (ValueError, ImportError) as exc:
        _err(f"Tracé des diagrammes échoué : {exc}")


# ---------------------------------------------------------------------------
# Sous-commande : validate
# ---------------------------------------------------------------------------

def cmd_validate(args: argparse.Namespace) -> int:
    path = Path(args.path)

    if not path.exists():
        _err(f"Fichier introuvable : {path}")
        return 1

    try:
        with path.open("r", encoding="utf-8") as fh:
            json.load(fh)
    except json.JSONDecodeError as exc:
        _err(f"JSON invalide : {exc}")
        return 1

    _ok("Syntaxe JSON correcte")

    try:
        from femsolver.io.json_model import load_model
        model = load_model(path)
    except (ValueError, KeyError, TypeError) as exc:
        _err(f"Modèle invalide : {exc}")
        return 1

    _ok(f"Modèle '{model.name}' valide — "
        f"{model.mesh.n_nodes} nœuds, "
        f"{len(model.mesh.elements)} éléments, "
        f"analyse '{model.analysis.get('type', 'static')}'")

    if not model.bc.dirichlet:
        _warn("Aucune condition de Dirichlet — système peut être singulier")

    atype = model.analysis.get("type", "static")
    if atype == "static" and not model.bc.neumann and not model.bc.body_force and \
            not model.bc.pressure and not model.bc.distributed:
        _warn("Aucun chargement défini (Neumann, gravité, pression…)")

    return 0


# ---------------------------------------------------------------------------
# Sous-commande : check
# ---------------------------------------------------------------------------

def cmd_check(args: argparse.Namespace) -> int:
    """Vérifie la santé du modèle (orphelins, Jacobien, singularité, qualité…).

    Contrairement à ``validate`` (qui ne contrôle que le JSON), ``check``
    exécute toutes les vérifications de :func:`run_model_checks` et affiche
    erreurs bloquantes et avertissements.  Retourne 1 si au moins une erreur.
    """
    import logging

    path = Path(args.path)
    try:
        from femsolver.io.json_model import load_model
        model = load_model(path)
    except FileNotFoundError:
        _err(f"Fichier introuvable : {path}")
        return 1
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        _err(f"Erreur de chargement : {exc}")
        return 1

    from femsolver.core.model_check import run_model_checks

    atype = model.analysis.get("type", "static")

    # On affiche le rapport nous-mêmes : on coupe le logging interne pour
    # éviter le double affichage des avertissements.
    mc_logger = logging.getLogger("femsolver.model_check")
    prev_level = mc_logger.level
    mc_logger.setLevel(logging.CRITICAL)
    try:
        report = run_model_checks(model.mesh, model.bc, analysis_type=atype)
    finally:
        mc_logger.setLevel(prev_level)

    _header(f"VÉRIFICATION DU MODÈLE : {model.name}")
    _row("Nœuds", model.mesh.n_nodes)
    _row("Éléments", len(model.mesh.elements))
    _row("Type d'analyse", atype)

    if report.errors:
        _section(f"Erreurs bloquantes ({len(report.errors)})")
        for e in report.errors:
            _err(f"[{e.code}] {e.message}")
    if report.warnings:
        _section(f"Avertissements ({len(report.warnings)})")
        for w in report.warnings:
            _warn(f"[{w.code}] {w.message}")

    print()
    if report.errors:
        _err(f"Modèle NON valide — {len(report.errors)} erreur(s), "
             f"{len(report.warnings)} avertissement(s). Corrigez avant de résoudre.")
        return 1
    if report.warnings:
        _warn(f"Modèle résoluble avec {len(report.warnings)} avertissement(s).")
    else:
        _ok("Modèle sain — aucune erreur, aucun avertissement.")
    return 0


# ---------------------------------------------------------------------------
# Sous-commande : info
# ---------------------------------------------------------------------------

def cmd_info(args: argparse.Namespace) -> int:
    path = Path(args.path)
    try:
        from femsolver.io.json_model import load_model
        model = load_model(path)
    except FileNotFoundError:
        _err(f"Fichier introuvable : {path}")
        return 1
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        _err(f"Erreur de chargement : {exc}")
        return 1

    mesh = model.mesh
    bc = model.bc
    analysis = model.analysis

    _header(f"MODÈLE : {model.name}")

    if model.description:
        desc = textwrap.fill(model.description, width=56, initial_indent="  ",
                             subsequent_indent="  ")
        print(desc)

    _section("Géométrie")
    _row("Dimension spatiale", f"{mesh.n_dim}D")
    _row("Nœuds", mesh.n_nodes)
    _row("DDL par nœud", mesh.dpn)
    _row("DDL totaux", mesh.n_dof)

    _section("Éléments")
    type_counts: Counter[str] = Counter(e.etype.__name__ for e in mesh.elements)
    _row("Total éléments", len(mesh.elements))
    for etype_name, count in sorted(type_counts.items()):
        _row(f"  {etype_name}", count)

    mat_set: dict[str, Any] = {}
    for e in mesh.elements:
        mat = e.material
        key = f"E={mat.E:.3g} nu={mat.nu} rho={mat.rho}"
        mat_set[key] = mat
    _section("Matériaux distincts")
    _row("Total", len(mat_set))
    for key in sorted(mat_set):
        print(f"    {key}")

    _section("Conditions aux limites")
    n_dir = sum(len(v) for v in bc.dirichlet.values())
    _row("Dirichlet (DDL bloqués)", n_dir)
    n_neu = sum(len(v) for v in bc.neumann.values())
    _row("Neumann (forces nodales)", n_neu)
    if bc.pressure:
        _row("Pressions surfaciques", len(bc.pressure))
    if bc.body_force:
        _row("Force de volume", f"a = {bc.body_force.acceleration}")
    if bc.distributed:
        _row("Charges linéiques", len(bc.distributed))

    _section("Analyse")
    atype = analysis.get("type", "static")
    _row("Type", atype)
    for key, val in analysis.items():
        if key == "type":
            continue
        if key == "damping" and isinstance(val, dict):
            dtype = val.get("type", "?")
            params = {k: v for k, v in val.items() if k != "type"}
            _row("  amortissement", f"{dtype} {params}")
        elif key == "freqs" and isinstance(val, dict):
            spec_key = next(iter(val))
            spec_val = val[spec_key]
            _row("  fréquences", f"{spec_key}({spec_val[0]}, {spec_val[1]}, n={spec_val[2]})")
        else:
            _row(f"  {key}", val)

    print()
    return 0


# ---------------------------------------------------------------------------
# Sous-commande : run
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> int:
    path = Path(args.path)
    detailed = getattr(args, "detailed", False)
    export   = getattr(args, "export", None)
    diagrams = getattr(args, "diagrams", None)
    if diagrams == "<auto>":
        diagrams = f"{path.stem}_diagrams.png"

    try:
        from femsolver.core.model_check import ModelError
        from femsolver.io.json_model import load_model, solve_model
        model = load_model(path)
        results = solve_model(model, verbose=False)
    except FileNotFoundError:
        _err(f"Fichier introuvable : {path}")
        return 1
    except ModelError as exc:
        _err(f"Modèle invalide — résolution annulée :\n{exc}")
        _warn("Lancez « python -m femsolver check » pour le diagnostic complet.")
        return 1
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        _err(f"Erreur de chargement : {exc}")
        return 1
    except Exception as exc:  # noqa: BLE001
        _err(f"Erreur lors du calcul : {type(exc).__name__}: {exc}")
        return 1

    atype = results["analysis_type"]

    if not args.quiet:
        _header(f"RÉSULTATS : {results['name']}  [{atype}]")
        if atype == "static":
            _print_static_enriched(model, np.array(results["u"]), detailed=detailed)
        else:
            _print_results_generic(results, atype)
        print()

    if export:
        _export_vtu(export, model, results, quiet=args.quiet)

    if diagrams:
        _plot_diagrams(diagrams, model, results, quiet=args.quiet)

    return 0


def _print_results_generic(results: dict[str, Any], atype: str) -> None:
    """Affichage résumé pour les analyses non-statiques."""
    if atype == "modal":
        freqs = results["freqs"]
        _section("Fréquences propres")
        _row("Nombre de modes", len(freqs))
        for i, f in enumerate(freqs):
            _row(f"  f_{i + 1}", f"{f:.4f} Hz  (T = {1.0/f:.4e} s)")

    elif atype == "buckling":
        lams = results["lambda_cr"]
        _section("Charges critiques de flambage")
        _row("Nombre de modes", len(lams))
        for i, lam in enumerate(lams):
            _row(f"  λ_{i + 1}", f"{lam:.6g}  × P_ref")

    elif atype == "harmonic":
        amp   = np.array(results["amplitude"])
        freqs = np.array(results["freqs"])
        _section("Réponse harmonique")
        _row("Plage fréquences", f"[{freqs[0]:.2f}, {freqs[-1]:.2f}] Hz")
        _row("Amplitude max globale", f"{amp.max():.4e} m")
        peak_dof, peak_f_idx = np.unravel_index(amp.argmax(), amp.shape)
        _row("  → DDL", int(peak_dof))
        _row("  → à f", f"{freqs[peak_f_idx]:.4f} Hz")

    elif atype == "transient":
        u     = np.array(results["u"])
        times = np.array(results["times"])
        _section("Réponse transitoire")
        _row("Durée totale", f"{times[-1]:.4g} s  ({len(times)} pas)")
        _row("u_max absolu", f"{np.abs(u).max():.4e} m")
        _row("u_final max", f"{np.abs(u[:, -1]).max():.4e} m")

    elif atype in ("random_force", "random_base"):
        label = "force" if atype == "random_force" else "base"
        rms_u = np.array(results["rms_u"])
        rms_a = np.array(results["rms_a"])
        freqs = np.array(results["freqs"])
        _section(f"Réponse aléatoire ({label})")
        _row("Plage fréquences", f"[{freqs[0]:.2f}, {freqs[-1]:.2f}] Hz")
        _row("RMS déplacement max", f"{rms_u.max():.4e} m")
        _row("  → DDL", int(rms_u.argmax()))
        _row("RMS accélération max", f"{rms_a.max():.4e} m/s²")
        _row("  → DDL", int(rms_a.argmax()))

    else:
        _section("Résultats bruts")
        for key, val in results.items():
            if key in ("name", "analysis_type"):
                continue
            if isinstance(val, list):
                _row(key, f"[liste de {len(val)} éléments]")
            else:
                _row(key, val)


# ---------------------------------------------------------------------------
# Point d'entrée principal
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m femsolver",
        description="CLI pour le solveur éléments finis fem-solver.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Exemples :
              python -m femsolver validate examples/warren_truss.json
              python -m femsolver check    examples/warren_truss.json
              python -m femsolver info     examples/cantilever_beam.json
              python -m femsolver run      examples/warren_truss.json
              python -m femsolver run      examples/cantilever_beam.json --detailed
              python -m femsolver run      examples/warren_truss.json --export warren.vtu
              python -m femsolver run      examples/cantilever_distributed_beam.json --diagrams
              python -m femsolver run      examples/cantilever_modal.json --quiet
        """),
    )

    sub = parser.add_subparsers(dest="command", metavar="commande")
    sub.required = True

    # validate
    p_val = sub.add_parser(
        "validate",
        help="Vérifier le JSON sans lancer le calcul.",
    )
    p_val.add_argument("path", metavar="model.json")

    # check
    p_chk = sub.add_parser(
        "check",
        help="Vérifier la santé du modèle (orphelins, Jacobien, singularité…).",
    )
    p_chk.add_argument("path", metavar="model.json")

    # info
    p_info = sub.add_parser(
        "info",
        help="Afficher un résumé du modèle sans calcul.",
    )
    p_info.add_argument("path", metavar="model.json")

    # run
    p_run = sub.add_parser(
        "run",
        help="Charger, résoudre et afficher les résultats.",
    )
    p_run.add_argument("path", metavar="model.json")
    p_run.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Supprimer la sortie (utile pour le batch).",
    )
    p_run.add_argument(
        "--detailed",
        action="store_true",
        help="Afficher le tableau complet de tous les nœuds (analyse statique).",
    )
    p_run.add_argument(
        "--export",
        metavar="FILE.vtu",
        default=None,
        help="Exporter les résultats en VTK (ParaView) sans affichage superflu.",
    )
    p_run.add_argument(
        "--diagrams",
        metavar="FILE.png",
        nargs="?",
        const="<auto>",
        default=None,
        help="Tracer les diagrammes d'efforts internes M/V/N des poutres "
             "(analyse statique). Sans valeur, enregistre <modèle>_diagrams.png.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Point d'entrée principal du CLI.

    Parameters
    ----------
    argv : list[str] or None
        Arguments. Défaut : sys.argv[1:].

    Returns
    -------
    int
        0 = succès, 1 = erreur.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    dispatch = {"validate": cmd_validate, "check": cmd_check,
                "info": cmd_info, "run": cmd_run}
    return dispatch[args.command](args)


if __name__ == "__main__":
    sys.exit(main())

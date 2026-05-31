"""CLI pour fem-solver.

Usage
-----
    python -m femsolver run      model.json [--quiet]
    python -m femsolver validate model.json
    python -m femsolver info     model.json

Commandes
---------
run
    Charge le modèle JSON, lance le calcul et affiche un résumé des résultats.

validate
    Vérifie que le JSON est syntaxiquement correct et que le modèle peut être
    construit (types d'éléments, matériaux, sections…), sans lancer aucun calcul.

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


# ---------------------------------------------------------------------------
# Sous-commande : validate
# ---------------------------------------------------------------------------

def cmd_validate(args: argparse.Namespace) -> int:
    """Vérifie la syntaxe JSON et la construction du modèle.

    Returns
    -------
    int
        0 si valide, 1 sinon.
    """
    path = Path(args.path)

    # 1. Existence du fichier
    if not path.exists():
        _err(f"Fichier introuvable : {path}")
        return 1

    # 2. Syntaxe JSON
    try:
        with path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except json.JSONDecodeError as exc:
        _err(f"JSON invalide : {exc}")
        return 1

    _ok("Syntaxe JSON correcte")

    # 3. Construction du modèle (sans calcul)
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

    # 4. Avertissements éventuels
    _warn_no_dirichlet(model)
    _warn_no_neumann(model)

    return 0


def _warn_no_dirichlet(model: Any) -> None:
    if not model.bc.dirichlet:
        print("  \033[33m⚠\033[0m  Aucune condition de Dirichlet — système peut être singulier")


def _warn_no_neumann(model: Any) -> None:
    atype = model.analysis.get("type", "static")
    if atype == "static" and not model.bc.neumann and not model.bc.body_force and \
            not model.bc.pressure and not model.bc.distributed:
        print("  \033[33m⚠\033[0m  Aucun chargement défini (Neumann, gravité, pression…)")


# ---------------------------------------------------------------------------
# Sous-commande : info
# ---------------------------------------------------------------------------

def cmd_info(args: argparse.Namespace) -> int:
    """Affiche un résumé complet du modèle sans calcul.

    Returns
    -------
    int
        0 si le modèle est chargé avec succès, 1 sinon.
    """
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

    # --- Géométrie ---
    _section("Géométrie")
    _row("Dimension spatiale", f"{mesh.n_dim}D")
    _row("Nœuds", mesh.n_nodes)
    _row("DDL par nœud", mesh.dpn)
    _row("DDL totaux", mesh.n_dof)

    # --- Éléments ---
    _section("Éléments")
    type_counts: Counter[str] = Counter(
        e.etype.__name__ for e in mesh.elements
    )
    _row("Total éléments", len(mesh.elements))
    for etype_name, count in sorted(type_counts.items()):
        _row(f"  {etype_name}", count)

    # --- Matériaux ---
    mat_set: dict[str, Any] = {}
    for e in mesh.elements:
        mat = e.material
        key = f"E={mat.E:.3g} nu={mat.nu} rho={mat.rho}"
        mat_set[key] = mat
    _section("Matériaux distincts")
    _row("Total", len(mat_set))
    for key in sorted(mat_set):
        print(f"    {key}")

    # --- Conditions aux limites ---
    _section("Conditions aux limites")
    n_dir = sum(len(v) for v in bc.dirichlet.values())
    _row("Dirichlet (DDL bloqués)", n_dir)
    n_neu = sum(len(v) for v in bc.neumann.values())
    _row("Neumann (forces nodales)", n_neu)
    if bc.pressure:
        _row("Pressions surfaciques", len(bc.pressure))
    if bc.body_force:
        acc = bc.body_force.acceleration
        _row("Force de volume", f"a = {acc}")
    if bc.distributed:
        _row("Charges linéiques", len(bc.distributed))

    # --- Analyse ---
    _section("Analyse")
    atype = analysis.get("type", "static")
    _row("Type", atype)
    for key, val in analysis.items():
        if key == "type":
            continue
        if key == "damping" and isinstance(val, dict):
            dtype = val.get("type", "?")
            params = {k: v for k, v in val.items() if k != "type"}
            _row(f"  amortissement", f"{dtype} {params}")
        elif key == "freqs" and isinstance(val, dict):
            spec_key = next(iter(val))
            spec_val = val[spec_key]
            _row(f"  fréquences", f"{spec_key}({spec_val[0]}, {spec_val[1]}, n={spec_val[2]})")
        else:
            _row(f"  {key}", val)

    print()
    return 0


# ---------------------------------------------------------------------------
# Sous-commande : run
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> int:
    """Charge, résout et affiche les résultats.

    Returns
    -------
    int
        0 en cas de succès, 1 en cas d'erreur.
    """
    path = Path(args.path)
    verbose = not args.quiet

    try:
        from femsolver.io.json_model import run_from_json
        results = run_from_json(path, verbose=False)
    except FileNotFoundError:
        _err(f"Fichier introuvable : {path}")
        return 1
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        _err(f"Erreur de chargement : {exc}")
        return 1
    except Exception as exc:  # noqa: BLE001
        _err(f"Erreur lors du calcul : {type(exc).__name__}: {exc}")
        return 1

    atype = results["analysis_type"]
    name = results["name"]

    if not args.quiet:
        _header(f"RÉSULTATS : {name}  [{atype}]")
        _print_results(results, atype)
        print()

    return 0


def _print_results(results: dict[str, Any], atype: str) -> None:
    import numpy as np

    if atype == "static":
        u = np.array(results["u"])
        _section("Déplacements nodaux")
        _row("u_max absolu", f"{np.abs(u).max():.4e} m")
        _row("u_min", f"{u.min():.4e} m")
        _row("u_max", f"{u.max():.4e} m")
        _row("Norme L2", f"{np.linalg.norm(u):.4e} m")

    elif atype == "modal":
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
        amp = np.array(results["amplitude"])
        freqs = np.array(results["freqs"])
        _section("Réponse harmonique")
        _row("Plage fréquences", f"[{freqs[0]:.2f}, {freqs[-1]:.2f}] Hz")
        _row("Amplitude max globale", f"{amp.max():.4e} m")
        peak_dof, peak_f_idx = np.unravel_index(amp.argmax(), amp.shape)
        _row("  → DDL", int(peak_dof))
        _row("  → à f", f"{freqs[peak_f_idx]:.4f} Hz")

    elif atype == "transient":
        u = np.array(results["u"])
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
        idx_u = int(rms_u.argmax())
        _row("  → DDL", idx_u)
        _row("RMS accélération max", f"{rms_a.max():.4e} m/s²")
        idx_a = int(rms_a.argmax())
        _row("  → DDL", idx_a)

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
              python -m femsolver info     examples/cantilever_beam.json
              python -m femsolver run      examples/warren_truss.json
              python -m femsolver run      examples/cantilever_modal.json --quiet
        """),
    )

    sub = parser.add_subparsers(dest="command", metavar="commande")
    sub.required = True

    # --- validate ---
    p_val = sub.add_parser(
        "validate",
        help="Vérifier le JSON sans lancer le calcul.",
        description="Vérifie la syntaxe JSON et la cohérence du modèle (types, matériaux, sections).",
    )
    p_val.add_argument("path", metavar="model.json", help="Chemin vers le fichier JSON.")

    # --- info ---
    p_info = sub.add_parser(
        "info",
        help="Afficher un résumé du modèle sans calcul.",
        description="Affiche les nœuds, éléments, DDL, conditions aux limites et type d'analyse.",
    )
    p_info.add_argument("path", metavar="model.json", help="Chemin vers le fichier JSON.")

    # --- run ---
    p_run = sub.add_parser(
        "run",
        help="Charger, résoudre et afficher les résultats.",
        description="Charge le modèle JSON, lance le calcul et affiche un résumé des résultats.",
    )
    p_run.add_argument("path", metavar="model.json", help="Chemin vers le fichier JSON.")
    p_run.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Supprimer la sortie (utile pour les tests ou le batch).",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Point d'entrée principal du CLI.

    Parameters
    ----------
    argv : list[str] or None
        Arguments de ligne de commande. Par défaut : sys.argv[1:].

    Returns
    -------
    int
        Code de retour (0 = succès, 1 = erreur).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    dispatch = {
        "validate": cmd_validate,
        "info":     cmd_info,
        "run":      cmd_run,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    sys.exit(main())

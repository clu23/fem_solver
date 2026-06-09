"""Tests du CLI femsolver.__main__.

Chaque test vérifie le code de retour et/ou la sortie stdout/stderr.
Les calculs FEM sont validés séparément dans test_json_model.py — ici on
teste uniquement le comportement du CLI (argparse, dispatch, affichage).
"""

from __future__ import annotations

import json
import sys
from io import StringIO
from pathlib import Path

import pytest

from femsolver.__main__ import main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(argv: list[str], capsys) -> tuple[int, str, str]:
    """Appelle main() et retourne (code_retour, stdout, stderr)."""
    rc = main(argv)
    cap = capsys.readouterr()
    return rc, cap.out, cap.err


def _make_json(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "model.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


STATIC_DATA = {
    "name": "Test Bar",
    "materials": {"mat": {"E": 210e9, "nu": 0.3, "rho": 7800}},
    "nodes": [[0.0, 0.0], [1.0, 0.0]],
    "elements": [
        {"type": "Bar2D", "nodes": [0, 1], "material": "mat", "area": 1e-4}
    ],
    "boundary_conditions": {
        "dirichlet": [
            {"node": 0, "dof": 0}, {"node": 0, "dof": 1}, {"node": 1, "dof": 1}
        ],
        "neumann": [{"node": 1, "dof": 0, "value": 10000.0}],
    },
    "analysis": {"type": "static"},
}

CANTILEVER_JSON_DATA = {
    "name": "Cantilever",
    "materials": {"mat": {"E": 210e9, "nu": 0.3, "rho": 7800}},
    "nodes": [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]],
    "elements": [
        {
            "type": "Beam2D", "nodes": [0, 1], "material": "mat",
            "section": {"type": "rectangular", "width": 0.05, "height": 0.10},
        },
        {
            "type": "Beam2D", "nodes": [1, 2], "material": "mat",
            "section": {"type": "rectangular", "width": 0.05, "height": 0.10},
        },
    ],
    "boundary_conditions": {
        "dirichlet": [
            {"node": 0, "dof": 0}, {"node": 0, "dof": 1}, {"node": 0, "dof": 2}
        ],
        "neumann": [{"node": 2, "dof": 1, "value": -5000.0}],
    },
    "analysis": {"type": "static"},
}

MODAL_DATA = {
    "name": "Test Modal",
    "materials": {"mat": {"E": 210e9, "nu": 0.3, "rho": 7800}},
    "nodes": [[i * 0.1, 0.0] for i in range(6)],
    "elements": [
        {
            "type": "Beam2D", "nodes": [i, i + 1], "material": "mat",
            "section": {"type": "rectangular", "width": 0.05, "height": 0.10},
        }
        for i in range(5)
    ],
    "boundary_conditions": {
        "dirichlet": [
            {"node": 0, "dof": 0}, {"node": 0, "dof": 1}, {"node": 0, "dof": 2}
        ]
    },
    "analysis": {"type": "modal", "n_modes": 2},
}


# ---------------------------------------------------------------------------
# Tests — validate
# ---------------------------------------------------------------------------

class TestValidate:
    """Vérifie le comportement de la commande validate."""

    def test_valid_model_returns_0(self, tmp_path, capsys):
        p = _make_json(tmp_path, STATIC_DATA)
        rc, out, _ = _run(["validate", str(p)], capsys)
        assert rc == 0

    def test_valid_model_prints_ok(self, tmp_path, capsys):
        p = _make_json(tmp_path, STATIC_DATA)
        _, out, _ = _run(["validate", str(p)], capsys)
        assert "valide" in out.lower() or "✓" in out

    def test_file_not_found_returns_1(self, capsys):
        rc, _, err = _run(["validate", "does_not_exist.json"], capsys)
        assert rc == 1
        assert "introuvable" in err.lower() or "not found" in err.lower()

    def test_bad_json_returns_1(self, tmp_path, capsys):
        p = tmp_path / "bad.json"
        p.write_text("{not valid json", encoding="utf-8")
        rc, _, err = _run(["validate", str(p)], capsys)
        assert rc == 1
        assert "json" in err.lower() or "invalid" in err.lower()

    def test_unknown_element_returns_1(self, tmp_path, capsys):
        data = {**STATIC_DATA, "elements": [
            {"type": "MagicElem", "nodes": [0, 1], "material": "mat", "area": 1e-4}
        ]}
        p = _make_json(tmp_path, data)
        rc, _, err = _run(["validate", str(p)], capsys)
        assert rc == 1

    def test_example_warren_valid(self, capsys):
        """Le fichier d'exemple warren_truss.json est valide."""
        rc, out, _ = _run(["validate", "examples/warren_truss.json"], capsys)
        assert rc == 0

    def test_example_cantilever_valid(self, capsys):
        rc, out, _ = _run(["validate", "examples/cantilever_beam.json"], capsys)
        assert rc == 0

    def test_warn_no_dirichlet(self, tmp_path, capsys):
        """Un avertissement est émis si aucun Dirichlet n'est défini."""
        data = {**STATIC_DATA}
        data["boundary_conditions"] = {
            "dirichlet": [],
            "neumann": [{"node": 1, "dof": 0, "value": 1.0}],
        }
        p = _make_json(tmp_path, data)
        rc, out, _ = _run(["validate", str(p)], capsys)
        # Le code retour est 0 (avertissement, pas erreur)
        assert rc == 0
        assert "singulier" in out.lower() or "⚠" in out


# ---------------------------------------------------------------------------
# Tests — check
# ---------------------------------------------------------------------------


class TestCheck:
    """Vérifie la commande check (santé du modèle, distincte de validate)."""

    def test_clean_model_returns_0(self, tmp_path, capsys):
        p = _make_json(tmp_path, STATIC_DATA)
        rc, out, _ = _run(["check", str(p)], capsys)
        assert rc == 0
        assert "sain" in out.lower() or "✓" in out

    def test_orphan_node_returns_1(self, tmp_path, capsys):
        # Ajoute un nœud orphelin (indice 2, non référencé).
        data = {**STATIC_DATA, "nodes": [[0.0, 0.0], [1.0, 0.0], [5.0, 5.0]]}
        p = _make_json(tmp_path, data)
        rc, out, err = _run(["check", str(p)], capsys)
        assert rc == 1
        combined = (out + err).lower()
        assert "orphelin" in combined

    def test_singular_model_returns_1_with_dof(self, tmp_path, capsys):
        # Une seule barre horizontale : nœud 1 libre en y → mécanisme.
        data = {
            "name": "Singulier",
            "materials": {"mat": {"E": 210e9, "nu": 0.3, "rho": 7800}},
            "nodes": [[0.0, 0.0], [1.0, 0.0]],
            "elements": [
                {"type": "Bar2D", "nodes": [0, 1], "material": "mat", "area": 1e-4}
            ],
            "boundary_conditions": {
                "dirichlet": [{"node": 0, "dof": 0}, {"node": 0, "dof": 1}],
                "neumann": [{"node": 1, "dof": 0, "value": 1000.0}],
            },
            "analysis": {"type": "static"},
        }
        p = _make_json(tmp_path, data)
        rc, out, err = _run(["check", str(p)], capsys)
        assert rc == 1
        combined = out + err
        assert "Nœud" in combined and "singul" in combined.lower()

    def test_file_not_found_returns_1(self, capsys):
        rc, _, err = _run(["check", "nope.json"], capsys)
        assert rc == 1


# ---------------------------------------------------------------------------
# Tests — info
# ---------------------------------------------------------------------------

class TestInfo:
    """Vérifie le comportement de la commande info."""

    def test_returns_0_on_valid_model(self, tmp_path, capsys):
        p = _make_json(tmp_path, STATIC_DATA)
        rc, _, _ = _run(["info", str(p)], capsys)
        assert rc == 0

    def test_displays_node_count(self, tmp_path, capsys):
        p = _make_json(tmp_path, STATIC_DATA)
        _, out, _ = _run(["info", str(p)], capsys)
        assert "2" in out        # 2 nœuds

    def test_displays_element_type(self, tmp_path, capsys):
        p = _make_json(tmp_path, STATIC_DATA)
        _, out, _ = _run(["info", str(p)], capsys)
        assert "Bar2D" in out

    def test_displays_analysis_type(self, tmp_path, capsys):
        p = _make_json(tmp_path, STATIC_DATA)
        _, out, _ = _run(["info", str(p)], capsys)
        assert "static" in out

    def test_displays_dirichlet_count(self, tmp_path, capsys):
        p = _make_json(tmp_path, STATIC_DATA)
        _, out, _ = _run(["info", str(p)], capsys)
        assert "3" in out        # 3 DDL Dirichlet

    def test_displays_model_name(self, tmp_path, capsys):
        p = _make_json(tmp_path, STATIC_DATA)
        _, out, _ = _run(["info", str(p)], capsys)
        assert "Test Bar" in out

    def test_modal_analysis_params(self, tmp_path, capsys):
        """L'analyse modale affiche n_modes."""
        p = _make_json(tmp_path, MODAL_DATA)
        _, out, _ = _run(["info", str(p)], capsys)
        assert "modal" in out
        assert "2" in out        # n_modes

    def test_file_not_found_returns_1(self, capsys):
        rc, _, err = _run(["info", "does_not_exist.json"], capsys)
        assert rc == 1

    def test_example_warren_info(self, capsys):
        """info sur le fichier d'exemple Warren affiche les bonnes valeurs."""
        rc, out, _ = _run(["info", "examples/warren_truss.json"], capsys)
        assert rc == 0
        assert "Warren" in out
        assert "9" in out        # 9 nœuds
        assert "15" in out       # 15 éléments

    def test_body_force_displayed(self, tmp_path, capsys):
        """La force de volume est affichée."""
        data = {**STATIC_DATA}
        data["boundary_conditions"] = {
            **data["boundary_conditions"],
            "body_force": [0.0, -9.81],
        }
        p = _make_json(tmp_path, data)
        _, out, _ = _run(["info", str(p)], capsys)
        assert "9.81" in out


# ---------------------------------------------------------------------------
# Tests — run
# ---------------------------------------------------------------------------

class TestRun:
    """Vérifie le comportement de la commande run."""

    def test_static_returns_0(self, tmp_path, capsys):
        p = _make_json(tmp_path, STATIC_DATA)
        rc, _, _ = _run(["run", str(p)], capsys)
        assert rc == 0

    def test_static_displays_displacement(self, tmp_path, capsys):
        p = _make_json(tmp_path, STATIC_DATA)
        _, out, _ = _run(["run", str(p)], capsys)
        assert "u_max" in out.lower() or "déplacement" in out.lower()

    def test_static_displays_top5_nodes(self, tmp_path, capsys):
        """Le tableau Top-5 nœuds les plus déplacés est affiché."""
        p = _make_json(tmp_path, STATIC_DATA)
        _, out, _ = _run(["run", str(p)], capsys)
        assert "Top 5" in out

    def test_static_displays_reactions(self, tmp_path, capsys):
        """Les réactions d'appui sont affichées."""
        p = _make_json(tmp_path, STATIC_DATA)
        _, out, _ = _run(["run", str(p)], capsys)
        assert "Réactions" in out or "réaction" in out.lower()

    def test_static_displays_equilibrium(self, tmp_path, capsys):
        """Le bilan d'équilibre est affiché avec ✓."""
        p = _make_json(tmp_path, STATIC_DATA)
        _, out, _ = _run(["run", str(p)], capsys)
        assert "équilibre" in out.lower() or "Bilan" in out
        assert "✓" in out

    def test_truss_displays_bar_forces(self, capsys):
        """Le treillis Warren affiche les efforts de barre Top-5."""
        _, out, _ = _run(["run", "examples/warren_truss.json"], capsys)
        assert "traction" in out.lower() or "compression" in out.lower()

    def test_non_truss_no_bar_forces(self, tmp_path, capsys):
        """Pour une poutre (Beam2D), pas de section 'barres'."""
        p = _make_json(tmp_path, CANTILEVER_JSON_DATA)
        _, out, _ = _run(["run", str(p)], capsys)
        assert "traction" not in out.lower()

    def test_detailed_shows_all_nodes(self, tmp_path, capsys):
        """--detailed affiche tous les nœuds dans le tableau complet."""
        p = _make_json(tmp_path, STATIC_DATA)
        _, out, _ = _run(["run", str(p), "--detailed"], capsys)
        assert "Tableau complet" in out
        # 2 nœuds → ligne 0 et ligne 1 dans le tableau
        assert "     0 " in out
        assert "     1 " in out

    def test_detailed_absent_without_flag(self, tmp_path, capsys):
        """Sans --detailed, le tableau complet n'est pas affiché."""
        p = _make_json(tmp_path, STATIC_DATA)
        _, out, _ = _run(["run", str(p)], capsys)
        assert "Tableau complet" not in out

    def test_export_vtu_creates_file(self, tmp_path, capsys):
        """--export crée un fichier .vtu."""
        p = _make_json(tmp_path, STATIC_DATA)
        vtu = tmp_path / "out.vtu"
        rc, out, _ = _run(["run", str(p), "--export", str(vtu)], capsys)
        assert rc == 0
        assert vtu.exists()
        assert "Exporté" in out

    def test_export_quiet_still_writes_file(self, tmp_path, capsys):
        """--export + --quiet ne produit aucune sortie mais écrit le fichier."""
        p = _make_json(tmp_path, STATIC_DATA)
        vtu = tmp_path / "silent.vtu"
        rc, out, _ = _run(["run", str(p), "--export", str(vtu), "--quiet"], capsys)
        assert rc == 0
        assert out == ""
        assert vtu.exists()

    def test_beam_export_vtu(self, tmp_path, capsys):
        """Export VTU fonctionne pour une poutre Beam2D (dpn != n_dim)."""
        p = _make_json(tmp_path, CANTILEVER_JSON_DATA)
        vtu = tmp_path / "beam.vtu"
        rc, _, _ = _run(["run", str(p), "--export", str(vtu), "--quiet"], capsys)
        assert rc == 0
        assert vtu.exists()

    def test_diagrams_creates_png(self, tmp_path, capsys):
        """--diagrams crée un fichier PNG pour une poutre Beam2D."""
        p = _make_json(tmp_path, CANTILEVER_JSON_DATA)
        png = tmp_path / "diag.png"
        rc, out, _ = _run(["run", str(p), "--diagrams", str(png)], capsys)
        assert rc == 0
        assert png.exists()
        assert "M/V/N" in out

    def test_diagrams_quiet_still_writes_file(self, tmp_path, capsys):
        """--diagrams + --quiet : aucun affichage mais le PNG est écrit."""
        p = _make_json(tmp_path, CANTILEVER_JSON_DATA)
        png = tmp_path / "silent.png"
        rc, out, _ = _run(["run", str(p), "--diagrams", str(png), "--quiet"], capsys)
        assert rc == 0
        assert out == ""
        assert png.exists()

    def test_diagrams_warns_on_non_beam(self, tmp_path, capsys):
        """--diagrams sur un modèle sans poutre émet un avertissement."""
        p = _make_json(tmp_path, STATIC_DATA)
        png = tmp_path / "none.png"
        rc, out, _ = _run(["run", str(p), "--diagrams", str(png)], capsys)
        assert rc == 0
        assert not png.exists()
        assert "poutre" in out.lower()

    def test_modal_displays_frequencies(self, tmp_path, capsys):
        p = _make_json(tmp_path, MODAL_DATA)
        _, out, _ = _run(["run", str(p)], capsys)
        assert "Hz" in out

    def test_quiet_produces_no_output(self, tmp_path, capsys):
        p = _make_json(tmp_path, STATIC_DATA)
        rc, out, _ = _run(["run", str(p), "--quiet"], capsys)
        assert rc == 0
        assert out == ""

    def test_quiet_short_flag(self, tmp_path, capsys):
        p = _make_json(tmp_path, STATIC_DATA)
        rc, out, _ = _run(["run", str(p), "-q"], capsys)
        assert rc == 0
        assert out == ""

    def test_file_not_found_returns_1(self, capsys):
        rc, _, err = _run(["run", "missing.json"], capsys)
        assert rc == 1

    def test_example_warren_runs(self, capsys):
        rc, out, _ = _run(["run", "examples/warren_truss.json"], capsys)
        assert rc == 0
        assert "Warren" in out

    def test_example_cantilever_runs(self, capsys):
        rc, out, _ = _run(["run", "examples/cantilever_beam.json"], capsys)
        assert rc == 0

    def test_example_modal_runs(self, capsys):
        rc, out, _ = _run(["run", "examples/cantilever_modal.json"], capsys)
        assert rc == 0
        assert "Hz" in out

    def test_result_header_contains_name(self, tmp_path, capsys):
        p = _make_json(tmp_path, STATIC_DATA)
        _, out, _ = _run(["run", str(p)], capsys)
        assert "Test Bar" in out

    def test_result_header_contains_analysis_type(self, tmp_path, capsys):
        p = _make_json(tmp_path, STATIC_DATA)
        _, out, _ = _run(["run", str(p)], capsys)
        assert "static" in out

    def test_dof_labels_beam2d(self, tmp_path, capsys):
        """La troisième colonne DDL d'un Beam2D est 'θz', pas 'uz'."""
        p = _make_json(tmp_path, CANTILEVER_JSON_DATA)
        _, out, _ = _run(["run", str(p), "--detailed"], capsys)
        assert "θz" in out
        assert "uz" not in out


# ---------------------------------------------------------------------------
# Tests — argparse
# ---------------------------------------------------------------------------

class TestArgparse:
    """Vérifie le comportement d'argparse (aide, sous-commandes)."""

    def test_help_does_not_crash(self):
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

    def test_no_subcommand_exits(self):
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code != 0

    def test_unknown_subcommand_exits(self):
        with pytest.raises(SystemExit):
            main(["solve", "foo.json"])

    def test_validate_help(self):
        with pytest.raises(SystemExit) as exc_info:
            main(["validate", "--help"])
        assert exc_info.value.code == 0

    def test_run_help(self):
        with pytest.raises(SystemExit) as exc_info:
            main(["run", "--help"])
        assert exc_info.value.code == 0

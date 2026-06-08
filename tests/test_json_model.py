"""Tests du parseur JSON pour les modèles FEM.

Chaque test valide :
- La construction correcte du modèle depuis le JSON
- La résolution analytique connue (ou convergence FEM documentée)

Références analytiques
-----------------------
- Barre : δ = FL/(EA)
- Poutre console : δ = PL³/(3EI), θ = PL²/(2EI)
- Fréquences poutre console : fn = (βₙL)² / (2π L²) × √(EI / ρA)
  avec β₁L=1.8751, β₂L=4.6941
- Flambage colonne d'Euler encastrée-libre : P_cr = π²EI/(4L²)
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import numpy as np
import pytest

from femsolver.dynamics.damping import HystereticDamping
from femsolver.dynamics.rayleigh import RayleighDamping
from femsolver.io.json_model import FEModel, load_model, run_from_json


# ---------------------------------------------------------------------------
# Helpers — modèles JSON inline (pas de fichiers sur disque)
# ---------------------------------------------------------------------------

def _run_inline(json_str: str) -> dict:
    """Écrit un JSON temporaire, appelle run_from_json, retourne les résultats."""
    import tempfile
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as fh:
        fh.write(json_str)
        tmp_path = fh.name
    return run_from_json(tmp_path, verbose=False)


def _load_inline(json_str: str) -> FEModel:
    import tempfile
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as fh:
        fh.write(json_str)
        tmp_path = fh.name
    return load_model(tmp_path)


SINGLE_BAR_JSON = json.dumps({
    "name": "Single Bar",
    "materials": {"steel": {"E": 210e9, "nu": 0.3, "rho": 7800}},
    "nodes": [[0.0, 0.0], [1.0, 0.0]],
    "elements": [
        {"type": "Bar2D", "nodes": [0, 1], "material": "steel", "area": 1e-4}
    ],
    "boundary_conditions": {
        "dirichlet": [
            {"node": 0, "dof": 0, "value": 0.0},
            {"node": 0, "dof": 1, "value": 0.0},
            {"node": 1, "dof": 1, "value": 0.0},
        ],
        "neumann": [{"node": 1, "dof": 0, "value": 10000.0}],
    },
    "analysis": {"type": "static"},
})

CANTILEVER_JSON = json.dumps({
    "name": "Cantilever",
    "materials": {"steel": {"E": 210e9, "nu": 0.3, "rho": 7800}},
    "nodes": [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]],
    "elements": [
        {
            "type": "Beam2D", "nodes": [0, 1], "material": "steel",
            "section": {"type": "rectangular", "width": 0.05, "height": 0.10},
        },
        {
            "type": "Beam2D", "nodes": [1, 2], "material": "steel",
            "section": {"type": "rectangular", "width": 0.05, "height": 0.10},
        },
    ],
    "boundary_conditions": {
        "dirichlet": [
            {"node": 0, "dof": 0, "value": 0.0},
            {"node": 0, "dof": 1, "value": 0.0},
            {"node": 0, "dof": 2, "value": 0.0},
        ],
        "neumann": [{"node": 2, "dof": 1, "value": -5000.0}],
    },
    "analysis": {"type": "static"},
})

MODAL_JSON = json.dumps({
    "name": "Cantilever Modal",
    "materials": {"steel": {"E": 210e9, "nu": 0.3, "rho": 7800}},
    "nodes": [[i * 0.1, 0.0] for i in range(11)],
    "elements": [
        {
            "type": "Beam2D", "nodes": [i, i + 1], "material": "steel",
            "section": {"type": "rectangular", "width": 0.05, "height": 0.10},
        }
        for i in range(10)
    ],
    "boundary_conditions": {
        "dirichlet": [
            {"node": 0, "dof": 0, "value": 0.0},
            {"node": 0, "dof": 1, "value": 0.0},
            {"node": 0, "dof": 2, "value": 0.0},
        ]
    },
    "analysis": {"type": "modal", "n_modes": 2},
})

BUCKLING_JSON = json.dumps({
    "name": "Euler Column Cantilever",
    "description": "Colonne encastrée-libre : P_cr = π²EI/(4L²). Section 10×10 mm, L=1 m.",
    "materials": {"steel": {"E": 210e9, "nu": 0.3, "rho": 7800}},
    "nodes": [[0.0, i * 0.1] for i in range(11)],
    "elements": [
        {
            "type": "Beam2D", "nodes": [i, i + 1], "material": "steel",
            "section": {"type": "rectangular", "width": 0.01, "height": 0.01},
        }
        for i in range(10)
    ],
    "boundary_conditions": {
        "dirichlet": [
            {"node": 0, "dof": 0, "value": 0.0},
            {"node": 0, "dof": 1, "value": 0.0},
            {"node": 0, "dof": 2, "value": 0.0},
        ],
        "neumann": [{"node": 10, "dof": 1, "value": -1.0}],
    },
    "analysis": {"type": "buckling", "n_modes": 1},
})

HARMONIC_JSON = json.dumps({
    "name": "Harmonic SDOF",
    "materials": {"steel": {"E": 210e9, "nu": 0.3, "rho": 7800}},
    "nodes": [[0.0, 0.0], [1.0, 0.0]],
    "elements": [
        {"type": "Bar2D", "nodes": [0, 1], "material": "steel", "area": 1e-4}
    ],
    "boundary_conditions": {
        "dirichlet": [
            {"node": 0, "dof": 0, "value": 0.0},
            {"node": 0, "dof": 1, "value": 0.0},
            {"node": 1, "dof": 1, "value": 0.0},
        ],
    },
    "analysis": {
        "type": "harmonic",
        "freqs": {"linspace": [1.0, 100.0, 200]},
        "F_hat": [{"node": 1, "dof": 0, "value": 1.0}],
        "damping": {"type": "rayleigh", "alpha": 0.0, "beta": 0.005},
    },
})


# ---------------------------------------------------------------------------
# Tests — load_model
# ---------------------------------------------------------------------------

class TestLoadModel:
    """Vérifie la construction du FEModel depuis le JSON."""

    def test_load_example_warren_truss(self):
        """Fichier warren_truss.json existe et se charge correctement."""
        model = load_model("examples/warren_truss.json")
        assert isinstance(model, FEModel)
        assert model.mesh.n_nodes == 9
        assert len(model.mesh.elements) == 15
        assert model.mesh.n_dim == 2
        assert model.analysis["type"] == "static"

    def test_load_example_cantilever(self):
        """Fichier cantilever_beam.json : poutre console 6 nœuds, 5 éléments."""
        model = load_model("examples/cantilever_beam.json")
        assert model.mesh.n_nodes == 6
        assert len(model.mesh.elements) == 5
        assert model.mesh.dpn == 3   # Beam2D

    def test_load_inline_bar(self):
        """Barre unique — mesh 2 nœuds, dof_per_node=2 (Bar2D)."""
        model = _load_inline(SINGLE_BAR_JSON)
        assert model.mesh.n_nodes == 2
        assert model.mesh.n_dim == 2
        assert model.mesh.dpn == 2

    def test_load_inline_beam(self):
        """Poutre Beam2D — dof_per_node=3."""
        model = _load_inline(CANTILEVER_JSON)
        assert model.mesh.dpn == 3
        assert model.bc.dirichlet[0] == {0: 0.0, 1: 0.0, 2: 0.0}

    def test_load_materials(self):
        """Les matériaux sont correctement parsés."""
        model = _load_inline(SINGLE_BAR_JSON)
        # Tous les éléments utilisent le même matériau acier
        mat = model.mesh.elements[0].material
        assert mat.E == pytest.approx(210e9)
        assert mat.nu == pytest.approx(0.3)

    def test_load_section_rectangular(self):
        """Une section rectangulaire est correctement instanciée."""
        from femsolver.core.sections import RectangularSection
        model = _load_inline(CANTILEVER_JSON)
        props = model.mesh.elements[0].properties
        assert "section" in props
        sec = props["section"]
        assert isinstance(sec, RectangularSection)
        assert sec.width == pytest.approx(0.05)
        assert sec.height == pytest.approx(0.10)

    def test_load_neumann_bc(self):
        """Les forces nodales Neumann sont correctement parsées."""
        model = _load_inline(CANTILEVER_JSON)
        assert 2 in model.bc.neumann
        assert model.bc.neumann[2][1] == pytest.approx(-5000.0)

    def test_load_body_force(self):
        """body_force est parsée comme BodyForce."""
        from femsolver.core.mesh import BodyForce
        js = json.dumps({
            "name": "Gravity",
            "materials": {"mat": {"E": 210e9, "nu": 0.3, "rho": 7800}},
            "nodes": [[0.0, 0.0], [1.0, 0.0]],
            "elements": [
                {"type": "Bar2D", "nodes": [0, 1], "material": "mat", "area": 1e-4}
            ],
            "boundary_conditions": {
                "dirichlet": [{"node": 0, "dof": 0}, {"node": 0, "dof": 1}],
                "body_force": [0.0, -9.81],
            },
            "analysis": {"type": "static"},
        })
        model = _load_inline(js)
        assert isinstance(model.bc.body_force, BodyForce)
        assert model.bc.body_force.acceleration == pytest.approx((0.0, -9.81))

    def test_file_not_found(self):
        """FileNotFoundError si le fichier n'existe pas."""
        with pytest.raises(FileNotFoundError):
            load_model("nonexistent_file_xyz.json")

    def test_unknown_element_type(self):
        """ValueError si le type d'élément est inconnu."""
        js = json.dumps({
            "name": "x",
            "materials": {"mat": {"E": 1.0, "nu": 0.3, "rho": 1.0}},
            "nodes": [[0.0, 0.0], [1.0, 0.0]],
            "elements": [{"type": "FooBar", "nodes": [0, 1], "material": "mat", "area": 1e-4}],
            "boundary_conditions": {"dirichlet": [], "neumann": []},
            "analysis": {"type": "static"},
        })
        with pytest.raises(ValueError, match="inconnu"):
            _load_inline(js)

    def test_unknown_material(self):
        """ValueError si le matériau n'est pas défini."""
        js = json.dumps({
            "name": "x",
            "materials": {},
            "nodes": [[0.0, 0.0], [1.0, 0.0]],
            "elements": [{"type": "Bar2D", "nodes": [0, 1], "material": "ghost", "area": 1e-4}],
            "boundary_conditions": {"dirichlet": [], "neumann": []},
            "analysis": {"type": "static"},
        })
        with pytest.raises(ValueError, match="ghost"):
            _load_inline(js)

    def test_unknown_analysis_type(self):
        """ValueError si le type d'analyse est inconnu."""
        model = _load_inline(SINGLE_BAR_JSON)
        from femsolver.io.json_model import _dispatch_analysis
        with pytest.raises(ValueError, match="inconnu"):
            _dispatch_analysis(model.mesh, model.bc, {"type": "foobar"}, "foobar", verbose=False)


# ---------------------------------------------------------------------------
# Tests — analyse statique
# ---------------------------------------------------------------------------

class TestStaticAnalysis:
    """Vérifie les résultats statiques contre des solutions analytiques."""

    def test_bar_tension_displacement(self):
        """δ = FL/(EA) — barre unique en traction axiale.

        Analytique : δ = 10000 × 1.0 / (210e9 × 1e-4) ≈ 4.762e-7 m
        """
        E, A, L, F = 210e9, 1e-4, 1.0, 10000.0
        result = _run_inline(SINGLE_BAR_JSON)
        u = np.array(result["u"])
        # dof global pour ux au nœud 1 : 1*2+0 = 2
        delta_fem = u[2]
        delta_ana = F * L / (E * A)
        np.testing.assert_allclose(delta_fem, delta_ana, rtol=1e-12)

    def test_bar_reaction_force_equilibrium(self):
        """La somme des forces de réaction est nulle (équilibre statique)."""
        result = _run_inline(SINGLE_BAR_JSON)
        u = np.array(result["u"])
        # Axial dépl nœud 1 = 4.762e-7, nœud 0 = 0
        delta = u[2] - u[0]
        E, A, L = 210e9, 1e-4, 1.0
        F_internal = E * A / L * delta
        np.testing.assert_allclose(F_internal, 10000.0, rtol=1e-12)

    def test_cantilever_tip_displacement(self):
        """δ_tip = PL³/(3EI) — poutre console, 2 éléments Beam2D.

        P = -5000 N, L = 1 m, b = 0.05, h = 0.10
        I = b·h³/12 ≈ 4.167e-6 m⁴
        δ_ana = -5000 × 1³ / (3 × 210e9 × 4.167e-6) ≈ -1.905e-3 m
        """
        E = 210e9
        b, h, L, P = 0.05, 0.10, 1.0, -5000.0
        I = b * h ** 3 / 12.0
        delta_ana = P * L ** 3 / (3.0 * E * I)

        result = _run_inline(CANTILEVER_JSON)
        u = np.array(result["u"])
        # nœud 2, dof 1 : index = 2*3+1 = 7
        delta_fem = u[7]
        np.testing.assert_allclose(delta_fem, delta_ana, rtol=1e-10)

    def test_cantilever_tip_rotation(self):
        """θ_tip = PL²/(2EI) — rotation au bout de la poutre console."""
        E = 210e9
        b, h, L, P = 0.05, 0.10, 1.0, -5000.0
        I = b * h ** 3 / 12.0
        theta_ana = P * L ** 2 / (2.0 * E * I)

        result = _run_inline(CANTILEVER_JSON)
        u = np.array(result["u"])
        # nœud 2, dof 2 (rotation θz) : index = 2*3+2 = 8
        theta_fem = u[8]
        np.testing.assert_allclose(theta_fem, theta_ana, rtol=1e-10)

    def test_warren_truss_example_runs(self):
        """Le treillis Warren (fichier JSON) se résout sans erreur."""
        result = run_from_json("examples/warren_truss.json", verbose=False)
        u = np.array(result["u"])
        assert u.shape == (18,)    # 9 nœuds × 2 DDL
        assert np.isfinite(u).all()

    def test_cantilever_example_exact(self):
        """Le fichier cantilever_beam.json donne δ = PL³/(3EI) exactement (5 éléments)."""
        E = 210e9
        b, h, L, P = 0.05, 0.10, 1.0, -5000.0
        I = b * h ** 3 / 12.0
        delta_ana = P * L ** 3 / (3.0 * E * I)

        result = run_from_json("examples/cantilever_beam.json", verbose=False)
        u = np.array(result["u"])
        # nœud 5, dof 1 : index = 5*3+1 = 16
        delta_fem = u[16]
        np.testing.assert_allclose(delta_fem, delta_ana, rtol=1e-10)

    def test_result_contains_name_and_type(self):
        """Le dict de résultats contient 'name' et 'analysis_type'."""
        result = _run_inline(SINGLE_BAR_JSON)
        assert result["name"] == "Single Bar"
        assert result["analysis_type"] == "static"


# ---------------------------------------------------------------------------
# Tests — analyse modale
# ---------------------------------------------------------------------------

class TestModalAnalysis:
    """Vérifie les fréquences propres contre la solution analytique Euler-Bernoulli."""

    # fn = (βₙL)² / (2π L²) × √(EI / ρA)
    # β₁L = 1.8751040593, β₂L = 4.6940911329
    _beta_L = [1.8751040593, 4.6940911329]
    _E, _rho = 210e9, 7800.0
    _b, _h, _L = 0.05, 0.10, 1.0

    @classmethod
    def _fn_exact(cls, n: int) -> float:
        I = cls._b * cls._h ** 3 / 12.0
        A = cls._b * cls._h
        return cls._beta_L[n] ** 2 / (2.0 * np.pi * cls._L ** 2) * np.sqrt(
            cls._E * I / (cls._rho * A)
        )

    def test_first_frequency(self):
        """f₁ FEM ≈ f₁ analytique à 0.1% (10 éléments Beam2D)."""
        result = _run_inline(MODAL_JSON)
        freqs = result["freqs"]
        fn_ana = self._fn_exact(0)
        np.testing.assert_allclose(freqs[0], fn_ana, rtol=0.001)

    def test_second_frequency(self):
        """f₂ FEM ≈ f₂ analytique à 1% (10 éléments Beam2D)."""
        result = _run_inline(MODAL_JSON)
        freqs = result["freqs"]
        fn_ana = self._fn_exact(1)
        np.testing.assert_allclose(freqs[1], fn_ana, rtol=0.01)

    def test_modes_shape(self):
        """phi a la forme (n_dof, n_modes)."""
        result = _run_inline(MODAL_JSON)
        modes = np.array(result["modes"])
        n_nodes = 11
        n_dof = n_nodes * 3   # Beam2D = 3 DDL/nœud
        assert modes.shape == (n_dof, 2)

    def test_freqs_ascending(self):
        """Les fréquences sont triées par ordre croissant."""
        result = _run_inline(MODAL_JSON)
        freqs = result["freqs"]
        assert freqs[0] < freqs[1]


# ---------------------------------------------------------------------------
# Tests — analyse de flambage
# ---------------------------------------------------------------------------

class TestBucklingAnalysis:
    """Vérifie la charge critique de flambage contre Euler encastré-libre.

    Colonne encastrée-libre : P_cr = π²EI / (4L²)
    Section rect. 50×100 mm, L = 1 m, acier E = 210 GPa.
    """

    def test_euler_column_critical_load(self):
        """P_cr FEM ≈ P_cr Euler (encastré-libre) à 1%.

        P_cr = π²EI/(4L²), section 10×10 mm, L=1 m
        """
        E = 210e9
        b = h = 0.01
        L = 1.0
        I = b * h ** 3 / 12.0
        P_cr_ana = np.pi ** 2 * E * I / (4.0 * L ** 2)

        result = _run_inline(BUCKLING_JSON)
        lambda_cr = result["lambda_cr"]
        P_cr_fem = lambda_cr[0]   # P_ref = 1 N → P_cr = lambda_cr × 1
        np.testing.assert_allclose(P_cr_fem, P_cr_ana, rtol=0.01)

    def test_buckling_result_keys(self):
        """Le dict contient 'lambda_cr' et 'phi'."""
        result = _run_inline(BUCKLING_JSON)
        assert "lambda_cr" in result
        assert "phi" in result

    def test_lambda_positive(self):
        """Les multiplicateurs de charge critique sont positifs."""
        result = _run_inline(BUCKLING_JSON)
        assert all(lam > 0 for lam in result["lambda_cr"])


# ---------------------------------------------------------------------------
# Tests — analyse harmonique
# ---------------------------------------------------------------------------

class TestHarmonicAnalysis:
    """Vérifie que run_harmonic via JSON retourne une amplitude FRF cohérente."""

    def test_harmonic_result_keys(self):
        """Le dict contient 'freqs', 'amplitude', 'U_real', 'U_imag'."""
        result = _run_inline(HARMONIC_JSON)
        assert "freqs" in result
        assert "amplitude" in result
        assert "U_real" in result
        assert "U_imag" in result

    def test_harmonic_amplitude_shape(self):
        """amplitude a la forme (n_dof, n_freqs)."""
        result = _run_inline(HARMONIC_JSON)
        amp = np.array(result["amplitude"])
        n_freqs = 200
        assert amp.shape == (4, n_freqs)   # 2 nœuds × 2 DDL

    def test_harmonic_static_limit(self):
        """À basse fréquence, l'amplitude tend vers la valeur statique u_st = F/k.

        F = 1 N, k = EA/L = 210e9 × 1e-4 / 1 = 2.1e7 N/m
        u_st = 1 / 2.1e7 ≈ 4.762e-8 m
        """
        result = _run_inline(HARMONIC_JSON)
        amp = np.array(result["amplitude"])
        # dof 2 = ux du nœud 1, fréquence la plus basse (1 Hz, loin de la résonance)
        u_static_low = amp[2, 0]
        k = 210e9 * 1e-4 / 1.0
        u_st = 1.0 / k
        np.testing.assert_allclose(u_static_low, u_st, rtol=0.05)


# ---------------------------------------------------------------------------
# Tests — parseur de sections
# ---------------------------------------------------------------------------

class TestSectionParsing:
    """Vérifie que chaque type de section JSON est correctement instancié."""

    def _make_beam_json(self, section_dict: dict) -> str:
        return json.dumps({
            "name": "sec_test",
            "materials": {"mat": {"E": 210e9, "nu": 0.3, "rho": 7800}},
            "nodes": [[0.0, 0.0], [1.0, 0.0]],
            "elements": [
                {"type": "Beam2D", "nodes": [0, 1], "material": "mat", "section": section_dict}
            ],
            "boundary_conditions": {
                "dirichlet": [
                    {"node": 0, "dof": 0}, {"node": 0, "dof": 1}, {"node": 0, "dof": 2},
                    {"node": 1, "dof": 0},
                ],
                "neumann": [{"node": 1, "dof": 1, "value": -1000.0}],
            },
            "analysis": {"type": "static"},
        })

    def test_rectangular_section(self):
        from femsolver.core.sections import RectangularSection
        model = _load_inline(self._make_beam_json(
            {"type": "rectangular", "width": 0.05, "height": 0.10}
        ))
        sec = model.mesh.elements[0].properties["section"]
        assert isinstance(sec, RectangularSection)
        assert sec.area == pytest.approx(0.05 * 0.10)

    def test_circular_section(self):
        from femsolver.core.sections import CircularSection
        import math
        model = _load_inline(self._make_beam_json(
            {"type": "circular", "radius": 0.05}
        ))
        sec = model.mesh.elements[0].properties["section"]
        assert isinstance(sec, CircularSection)
        assert sec.area == pytest.approx(math.pi * 0.05 ** 2)

    def test_hollow_circular_section(self):
        from femsolver.core.sections import HollowCircularSection
        model = _load_inline(self._make_beam_json(
            {"type": "hollow_circular", "outer_radius": 0.05, "inner_radius": 0.04}
        ))
        sec = model.mesh.elements[0].properties["section"]
        assert isinstance(sec, HollowCircularSection)

    def test_i_section(self):
        from femsolver.core.sections import ISection
        model = _load_inline(self._make_beam_json({
            "type": "i_section",
            "flange_width": 0.10, "height": 0.20,
            "flange_thickness": 0.0085, "web_thickness": 0.0056,
        }))
        sec = model.mesh.elements[0].properties["section"]
        assert isinstance(sec, ISection)

    def test_unknown_section_raises(self):
        """ValueError pour un type de section inconnu."""
        from femsolver.io.json_model import _parse_section
        with pytest.raises(ValueError, match="inconnu"):
            _parse_section({"type": "trapezoid", "a": 0.1})


# ---------------------------------------------------------------------------
# Tests — parseur de fréquences
# ---------------------------------------------------------------------------

class TestFreqsParsing:
    """Vérifie les différents formats de spécification de fréquences."""

    def _freqs(self, spec):
        from femsolver.io.json_model import _parse_freqs
        return _parse_freqs(spec)

    def test_linspace(self):
        f = self._freqs({"linspace": [1.0, 100.0, 50]})
        assert len(f) == 50
        assert f[0] == pytest.approx(1.0)
        assert f[-1] == pytest.approx(100.0)

    def test_logspace(self):
        f = self._freqs({"logspace": [1.0, 1000.0, 7]})
        assert len(f) == 7
        assert f[0] == pytest.approx(1.0, rel=1e-10)
        assert f[-1] == pytest.approx(1000.0, rel=1e-10)

    def test_list(self):
        f = self._freqs([1.0, 2.5, 5.0, 10.0])
        assert len(f) == 4
        np.testing.assert_allclose(f, [1.0, 2.5, 5.0, 10.0])

    def test_unknown_spec_raises(self):
        from femsolver.io.json_model import _parse_freqs
        with pytest.raises(ValueError):
            _parse_freqs({"geomspace": [1, 10, 5]})


# ---------------------------------------------------------------------------
# Tests — parseur d'amortissement
# ---------------------------------------------------------------------------

class TestDampingParsing:
    """Vérifie la construction des modèles d'amortissement."""

    def _damp(self, d):
        from femsolver.io.json_model import _parse_damping
        return _parse_damping(d)

    def test_none(self):
        assert self._damp(None) is None

    def test_rayleigh(self):
        d = self._damp({"type": "rayleigh", "alpha": 1.0, "beta": 0.002})
        assert isinstance(d, RayleighDamping)
        assert d.alpha == pytest.approx(1.0)
        assert d.beta == pytest.approx(0.002)

    def test_rayleigh_defaults(self):
        d = self._damp({"type": "rayleigh"})
        assert d.alpha == pytest.approx(0.0)
        assert d.beta == pytest.approx(0.0)

    def test_hysteretic(self):
        d = self._damp({"type": "hysteretic", "eta": 0.1})
        assert isinstance(d, HystereticDamping)
        assert d.eta == pytest.approx(0.1)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="inconnu"):
            self._damp({"type": "viscous"})


# ---------------------------------------------------------------------------
# Ressorts et amortisseurs ponctuels (Spring / Damper)
# ---------------------------------------------------------------------------

class TestSpringDamperParsing:
    """Parsing JSON des connecteurs ponctuels : DDL flexible, matériau optionnel."""

    def test_spring_only_infers_dof_per_node_from_stiffness(self):
        """Modèle 100 % ressorts : dpn déduit de la longueur du vecteur stiffness."""
        model = _load_inline(json.dumps({
            "name": "ground springs",
            "nodes": [[0.0, 0.0], [1.0, 0.0]],
            "elements": [
                {"type": "Spring", "nodes": [0], "stiffness": [1.0e5, 2.0e5]},
                {"type": "Spring", "nodes": [1], "stiffness": [3.0e5, 4.0e5]},
            ],
            "boundary_conditions": {"dirichlet": [], "neumann": []},
            "analysis": {"type": "static"},
        }))
        assert model.mesh.dpn == 2
        assert model.mesh.n_dof == 4

    def test_spring_material_optional(self):
        """Aucune clé 'material' n'est requise pour un ressort."""
        model = _load_inline(json.dumps({
            "name": "no material",
            "nodes": [[0.0, 0.0]],
            "elements": [{"type": "Spring", "nodes": [0], "stiffness": [1000.0]}],
            "boundary_conditions": {"dirichlet": [], "neumann": []},
            "analysis": {"type": "static"},
        }))
        assert len(model.mesh.elements) == 1

    def test_mixed_spring_bar_dof_consistency(self):
        """Spring (stiffness longueur 2) compatible avec Bar2D (dpn=2)."""
        model = _load_inline(json.dumps({
            "name": "bar + spring",
            "materials": {"steel": {"E": 210e9, "nu": 0.3, "rho": 7800}},
            "nodes": [[0.0, 0.0], [1.0, 0.0]],
            "elements": [
                {"type": "Bar2D", "nodes": [0, 1], "material": "steel", "area": 1e-4},
                {"type": "Spring", "nodes": [0, 1], "stiffness": [0.0, 5.0e5]},
            ],
            "boundary_conditions": {"dirichlet": [], "neumann": []},
            "analysis": {"type": "static"},
        }))
        assert model.mesh.dpn == 2

    def test_spring_dof_mismatch_raises(self):
        """Spring de longueur 3 incompatible avec Bar2D (dpn=2) → erreur."""
        with pytest.raises(ValueError, match="incompatibles"):
            _load_inline(json.dumps({
                "name": "mismatch",
                "materials": {"steel": {"E": 210e9, "nu": 0.3, "rho": 7800}},
                "nodes": [[0.0, 0.0], [1.0, 0.0]],
                "elements": [
                    {"type": "Bar2D", "nodes": [0, 1], "material": "steel", "area": 1e-4},
                    {"type": "Spring", "nodes": [0], "stiffness": [1.0, 2.0, 3.0]},
                ],
                "boundary_conditions": {"dirichlet": [], "neumann": []},
                "analysis": {"type": "static"},
            }))

    def test_spring_static_u_equals_F_over_k(self):
        """Bar + ressort vertical : u1_x=Fx·L/(EA), u1_y=Fy/k_y (exemple livré)."""
        result = run_from_json("examples/spring_support_static.json", verbose=False)
        u = np.array(result["u"])
        # nœud 1 : indices 2 (ux) et 3 (uy)
        np.testing.assert_allclose(u[2], 21000.0 / 2.1e7, rtol=1e-10)   # 1.0e-3 m
        np.testing.assert_allclose(u[3], -5000.0 / 5.0e5, rtol=1e-10)   # -1.0e-2 m

    def test_damper_optional_material(self):
        """Un amortisseur se parse sans clé 'material'."""
        model = _load_inline(json.dumps({
            "name": "damper",
            "nodes": [[0.0, 0.0]],
            "elements": [{"type": "Damper", "nodes": [0], "damping": [10.0]}],
            "boundary_conditions": {"dirichlet": [], "neumann": []},
            "analysis": {"type": "static"},
        }))
        assert model.mesh.elements[0].properties["damping"] == [10.0]

    def test_damper_harmonic_finite_peak(self):
        """Amortisseur ponctuel seul → pic de résonance fini près de f_n=5.03 Hz."""
        result = run_from_json("examples/damper_harmonic_sdof.json", verbose=False)
        f = np.array(result["freqs"])
        amp = np.array(result["amplitude"])
        ax = amp[2, :]   # DDL x du nœud 1
        i_peak = int(ax.argmax())
        f_n = np.sqrt(1000.0 / 1.0) / (2.0 * np.pi)
        np.testing.assert_allclose(f[i_peak], f_n, rtol=0.02)
        # Pic fini et proche de F/(k·2ζ) = 10/(1000·0.1) = 0.1 m
        np.testing.assert_allclose(ax[i_peak], 0.1, rtol=0.05)

    def test_damper_reduces_response_vs_undamped(self):
        """Le pic amorti (ζ=5%) est bien inférieur au pic quasi non amorti."""
        damped = run_from_json("examples/damper_harmonic_sdof.json", verbose=False)
        peak_damped = np.array(damped["amplitude"])[2, :].max()

        # Même modèle sans amortisseur (amortisseur c≈0) → pic bien plus grand
        undamped = _run_inline(json.dumps({
            "name": "undamped",
            "materials": {"soft": {"E": 1.0e6, "nu": 0.3, "rho": 3000.0}},
            "nodes": [[0.0, 0.0], [1.0, 0.0]],
            "elements": [
                {"type": "Bar2D", "nodes": [0, 1], "material": "soft", "area": 1.0e-3},
                {"type": "Damper", "nodes": [1], "damping": [0.05, 0.0]},
            ],
            "boundary_conditions": {"dirichlet": [
                {"node": 0, "dof": 0, "value": 0.0},
                {"node": 0, "dof": 1, "value": 0.0},
                {"node": 1, "dof": 1, "value": 0.0},
            ], "neumann": []},
            "analysis": {
                "type": "harmonic",
                "freqs": {"linspace": [1.0, 15.0, 400]},
                "F_hat": [{"node": 1, "dof": 0, "value": 10.0}],
                "damping": None,
            },
        }))
        peak_undamped = np.array(undamped["amplitude"])[2, :].max()
        assert peak_undamped > 5.0 * peak_damped

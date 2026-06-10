"""Parseur JSON pour définir et résoudre un modèle FEM complet.

Format JSON
-----------
Voir ``docs/json_schema.md`` pour le schéma complet.

Exemple minimal (treillis statique) ::

    {
        "name": "Warren Truss",
        "materials": {"steel": {"E": 210e9, "nu": 0.3, "rho": 7800}},
        "nodes": [[0,0], [1,0], [2,0]],
        "elements": [
            {"type": "Bar2D", "nodes": [0,1], "material": "steel", "area": 1e-4}
        ],
        "boundary_conditions": {
            "dirichlet": [{"node": 0, "dof": 0, "value": 0.0}],
            "neumann":   [{"node": 2, "dof": 1, "value": -5000.0}]
        },
        "analysis": {"type": "static"}
    }
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from femsolver.core.assembler import Assembler
from femsolver.core.boundary import apply_dirichlet
from femsolver.core.material import ElasticMaterial
from femsolver.core.model_check import run_model_checks
from femsolver.core.mesh import (
    BodyForce,
    BoundaryConditions,
    DistributedLineLoad,
    ElementData,
    Mesh,
    PressureLoad,
)
from femsolver.core.sections import (
    CircularSection,
    CSection,
    HollowCircularSection,
    HollowRectangularSection,
    ISection,
    LSection,
    RectangularSection,
    Section,
)
from femsolver.core.solver import BucklingSolver, StaticSolver
from femsolver.dynamics.damping import HystereticDamping, ModalDampingModel
from femsolver.dynamics.harmonic import run_harmonic
from femsolver.dynamics.modal import run_modal
from femsolver.dynamics.random_response import (
    psd_white,
    run_random_base,
    run_random_force,
)
from femsolver.dynamics.rayleigh import RayleighDamping
from femsolver.dynamics.transient import run_transient
from femsolver.elements.bar2d import Bar2D
from femsolver.elements.beam2d import Beam2D
from femsolver.elements.beam2d_timoshenko import Beam2DTimoshenko
from femsolver.elements.beam3d import Beam3D
from femsolver.elements.damper import DamperElement
from femsolver.elements.hexa20 import Hexa20
from femsolver.elements.hexa8 import Hexa8
from femsolver.elements.quad4 import Quad4
from femsolver.elements.quad8 import Quad8
from femsolver.elements.spring import SpringElement
from femsolver.elements.tetra10 import Tetra10
from femsolver.elements.tetra4 import Tetra4
from femsolver.elements.tri3 import Tri3
from femsolver.elements.tri6 import Tri6

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registres
# ---------------------------------------------------------------------------

_ELEM_REGISTRY: dict[str, type] = {
    "Bar2D": Bar2D,
    "Beam2D": Beam2D,
    "Beam2DTimoshenko": Beam2DTimoshenko,
    "Beam3D": Beam3D,
    "Tri3": Tri3,
    "Tri6": Tri6,
    "Quad4": Quad4,
    "Quad8": Quad8,
    "Tetra4": Tetra4,
    "Tetra10": Tetra10,
    "Hexa8": Hexa8,
    "Hexa20": Hexa20,
    "Spring": SpringElement,
    "Damper": DamperElement,
}

# n_dim and dof_per_node (None = use n_dim)
_ELEM_META: dict[str, tuple[int, int | None]] = {
    "Bar2D":            (2, None),
    "Beam2D":           (2, 3),
    "Beam2DTimoshenko": (2, 3),
    "Beam3D":           (3, 6),
    "Tri3":             (2, None),
    "Tri6":             (2, None),
    "Quad4":            (2, None),
    "Quad8":            (2, None),
    "Tetra4":           (3, None),
    "Tetra10":          (3, None),
    "Hexa8":            (3, None),
    "Hexa20":           (3, None),
}

# Éléments « DDL-flexibles » : ressort et amortisseur ponctuels. Leur nombre de
# DDL par nœud n'est pas fixe — il est déduit de la longueur du vecteur de
# raideur / amortissement et doit coïncider avec le ``dof_per_node`` du maillage.
# Le matériau y est optionnel (un connecteur ponctuel n'a pas de matériau continu).
_FLEX_ELEMS: frozenset[str] = frozenset({"Spring", "Damper"})

# Matériau factice pour les éléments flexibles sans clé "material".
_FLEX_MATERIAL = ElasticMaterial(E=1.0, nu=0.0, rho=1.0)

_SECTION_REGISTRY: dict[str, type] = {
    "rectangular":        RectangularSection,
    "circular":           CircularSection,
    "hollow_circular":    HollowCircularSection,
    "hollow_rectangular": HollowRectangularSection,
    "i_section":          ISection,
    "c_section":          CSection,
    "l_section":          LSection,
}


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------

@dataclass
class FEModel:
    """Modèle FEM complet prêt à résoudre.

    Parameters
    ----------
    name : str
        Nom du modèle.
    mesh : Mesh
        Maillage (nœuds, éléments, matériaux).
    bc : BoundaryConditions
        Conditions aux limites.
    analysis : dict[str, Any]
        Paramètres d'analyse tels que lus dans le JSON.
    description : str
        Description libre.
    """

    name: str
    mesh: Mesh
    bc: BoundaryConditions
    analysis: dict[str, Any]
    description: str = ""


# ---------------------------------------------------------------------------
# Helpers internes
# ---------------------------------------------------------------------------

def _parse_section(sec_dict: dict[str, Any]) -> Section:
    """Construit un objet Section depuis un dict JSON.

    Parameters
    ----------
    sec_dict : dict
        Doit contenir ``"type"`` (ex: ``"rectangular"``) plus les
        paramètres propres à la section.

    Returns
    -------
    Section
        Instance de la section correspondante.

    Raises
    ------
    ValueError
        Si le type de section est inconnu.
    """
    sec_type = sec_dict.get("type", "")
    cls = _SECTION_REGISTRY.get(sec_type)
    if cls is None:
        known = ", ".join(_SECTION_REGISTRY)
        raise ValueError(
            f"Type de section inconnu : '{sec_type}'. Connus : {known}"
        )
    params = {k: v for k, v in sec_dict.items() if k != "type"}
    return cls(**params)


def _parse_material(mat_dict: dict[str, Any]) -> ElasticMaterial:
    """Construit un ElasticMaterial depuis un dict JSON.

    Parameters
    ----------
    mat_dict : dict
        Doit contenir ``"E"``, ``"nu"`` et optionnellement ``"rho"`` et
        ``"alpha"`` (coefficient de dilatation thermique [1/K]).

    Returns
    -------
    ElasticMaterial
    """
    return ElasticMaterial(
        E=float(mat_dict["E"]),
        nu=float(mat_dict["nu"]),
        rho=float(mat_dict.get("rho", 0.0)),
        alpha=float(mat_dict.get("alpha", 0.0)),
    )


def _parse_element_properties(elem_type: str, edict: dict[str, Any]) -> dict:
    """Extrait les propriétés géométriques d'un élément depuis le dict JSON.

    Parameters
    ----------
    elem_type : str
        Nom de l'élément (ex: ``"Bar2D"``).
    edict : dict
        Dictionnaire JSON de l'élément (contient ``"type"``, ``"nodes"``,
        ``"material"`` et les propriétés spécifiques).

    Returns
    -------
    dict
        Dict ``properties`` passé à ``ElementData``.
    """
    props: dict[str, Any] = {}

    if elem_type in ("Bar2D",):
        props["area"] = float(edict["area"])

    elif elem_type in ("Beam2D", "Beam2DTimoshenko"):
        if "section" in edict:
            props["section"] = _parse_section(edict["section"])
        else:
            props["area"] = float(edict["area"])
            props["inertia"] = float(edict["inertia"])

    elif elem_type == "Beam3D":
        if "section" not in edict:
            raise ValueError("Beam3D requiert une clé 'section' dans l'élément.")
        props["section"] = _parse_section(edict["section"])
        if "v_vec" in edict:
            props["v_vec"] = np.asarray(edict["v_vec"], dtype=float)

    elif elem_type in ("Tri3", "Tri6", "Quad4", "Quad8"):
        if "thickness" in edict:
            props["thickness"] = float(edict["thickness"])

    elif elem_type == "Spring":
        props["stiffness"] = _as_float_list(edict["stiffness"])

    elif elem_type == "Damper":
        props["damping"] = _as_float_list(edict["damping"])

    # Tetra4, Tetra10, Hexa8, Hexa20 — pas de propriétés supplémentaires

    return props


def _as_float_list(value: Any) -> list[float]:
    """Normalise une raideur / un amortissement en liste de réels.

    Accepte un scalaire (→ liste à un élément) ou une liste. La longueur du
    résultat fixe le nombre de DDL par nœud du connecteur ponctuel.

    Parameters
    ----------
    value : float or list
        Valeur(s) JSON (``"stiffness"`` ou ``"damping"``).

    Returns
    -------
    list[float]
    """
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    return [float(value)]


def _parse_freqs(freqs_spec: Any) -> np.ndarray:
    """Convertit la spec de fréquences JSON en tableau numpy.

    Formats acceptés :

    - ``{"linspace": [fmin, fmax, n]}``
    - ``{"logspace": [fmin, fmax, n]}``
    - ``[f1, f2, ...]``  (liste de valeurs)

    Parameters
    ----------
    freqs_spec : dict or list
        Spécification des fréquences.

    Returns
    -------
    np.ndarray, shape (n,)
    """
    if isinstance(freqs_spec, dict):
        if "linspace" in freqs_spec:
            fmin, fmax, n = freqs_spec["linspace"]
            return np.linspace(float(fmin), float(fmax), int(n))
        if "logspace" in freqs_spec:
            fmin, fmax, n = freqs_spec["logspace"]
            return np.logspace(np.log10(float(fmin)), np.log10(float(fmax)), int(n))
        raise ValueError(f"Spec fréquences inconnue : {freqs_spec}")
    return np.asarray(freqs_spec, dtype=float)


def _parse_thermal(
    analysis: dict[str, Any], n_nodes: int
) -> float | np.ndarray | None:
    """Construit le champ de température ΔT depuis le bloc ``analysis.thermal``.

    Formats acceptés ::

        "thermal": {"delta_T": 50.0}                  # uniforme (global)
        "thermal": {"delta_T_nodes": [50, 50, 60, …]} # par nœud (interpolé)

    La forme ``delta_T`` accepte aussi directement une liste (équivalente à
    ``delta_T_nodes``).

    Parameters
    ----------
    analysis : dict
        Bloc ``analysis`` du JSON.
    n_nodes : int
        Nombre de nœuds du maillage (pour valider un champ nodal).

    Returns
    -------
    float, np.ndarray or None
        Scalaire (ΔT uniforme), tableau ``(n_nodes,)`` (ΔT nodal) ou ``None``
        si aucun chargement thermique n'est défini.

    Raises
    ------
    ValueError
        Si un champ nodal n'a pas la longueur ``n_nodes``.
    """
    thermal = analysis.get("thermal")
    if thermal is None:
        return None

    if "delta_T_nodes" in thermal:
        field = np.asarray(thermal["delta_T_nodes"], dtype=float)
    elif "delta_T" in thermal:
        val = thermal["delta_T"]
        if isinstance(val, (list, tuple)):
            field = np.asarray(val, dtype=float)
        else:
            return float(val)   # ΔT uniforme global
    else:
        raise ValueError(
            "Le bloc 'thermal' requiert 'delta_T' (scalaire ou liste) ou "
            "'delta_T_nodes' (liste par nœud)."
        )

    if field.shape != (n_nodes,):
        raise ValueError(
            f"Champ ΔT nodal de longueur {field.shape[0]} ≠ {n_nodes} nœuds."
        )
    return field


def _parse_damping(
    damp_dict: dict[str, Any] | None,
) -> RayleighDamping | HystereticDamping | None:
    """Construit un modèle d'amortissement depuis un dict JSON.

    Parameters
    ----------
    damp_dict : dict or None
        ``{"type": "rayleigh", "alpha": 0.0, "beta": 0.01}``
        ``{"type": "hysteretic", "eta": 0.1}``
        ``None`` → pas d'amortissement.

    Returns
    -------
    RayleighDamping, HystereticDamping, or None
    """
    if damp_dict is None:
        return None
    dtype = damp_dict.get("type", "")
    if dtype == "rayleigh":
        return RayleighDamping(
            alpha=float(damp_dict.get("alpha", 0.0)),
            beta=float(damp_dict.get("beta", 0.0)),
        )
    if dtype == "hysteretic":
        return HystereticDamping(eta=float(damp_dict["eta"]))
    raise ValueError(f"Type d'amortissement inconnu : '{dtype}'")


def _build_F_hat(
    f_hat_spec: Any, n_dof: int
) -> np.ndarray:
    """Construit le vecteur force F_hat depuis une spec JSON.

    Formats acceptés :

    - ``[{"node": i, "dof": d, "value": v}, ...]`` — contributions nodales
    - ``null`` → vecteur nul

    Parameters
    ----------
    f_hat_spec : list or None
    n_dof : int

    Returns
    -------
    np.ndarray, shape (n_dof,)
    """
    F = np.zeros(n_dof)
    if f_hat_spec is None:
        return F
    for entry in f_hat_spec:
        node = int(entry["node"])
        dof = int(entry["dof"])
        val = float(entry["value"])
        F[node] = val  # will be overwritten if dpn != 1; use global_idx
    return F


def _build_F_hat_with_dpn(
    f_hat_spec: Any, n_dof: int, dpn: int
) -> np.ndarray:
    """Comme _build_F_hat mais utilise (node * dpn + dof) pour l'indice global.

    Parameters
    ----------
    f_hat_spec : list or None
    n_dof : int
    dpn : int
        Degrés de liberté par nœud.

    Returns
    -------
    np.ndarray, shape (n_dof,)
    """
    F = np.zeros(n_dof)
    if f_hat_spec is None:
        return F
    for entry in f_hat_spec:
        node = int(entry["node"])
        dof = int(entry["dof"])
        val = float(entry["value"])
        F[node * dpn + dof] = val
    return F


# ---------------------------------------------------------------------------
# Construction du maillage
# ---------------------------------------------------------------------------

def _build_mesh_and_bc(data: dict[str, Any]) -> tuple[Mesh, BoundaryConditions]:
    """Parse materials, nodes, elements et boundary_conditions.

    Parameters
    ----------
    data : dict
        Données JSON complètes.

    Returns
    -------
    tuple[Mesh, BoundaryConditions]
    """
    # --- Matériaux ---
    materials: dict[str, ElasticMaterial] = {}
    for mat_name, mat_dict in data.get("materials", {}).items():
        materials[mat_name] = _parse_material(mat_dict)

    # --- Nœuds ---
    nodes = np.asarray(data["nodes"], dtype=float)
    n_dim = nodes.shape[1]  # 2 ou 3

    # --- Éléments ---
    elem_dicts = data.get("elements", [])
    if not elem_dicts:
        raise ValueError("Le modèle ne contient aucun élément.")

    elements_list: list[ElementData] = []
    dpn_common: int | None = None     # DDL/nœud effectif, commun à tout le maillage
    any_explicit_dpn = False          # True si un élément impose un dpn ≠ n_dim

    for edict in elem_dicts:
        etype_name = edict["type"]
        if etype_name not in _ELEM_REGISTRY:
            known = ", ".join(_ELEM_REGISTRY)
            raise ValueError(
                f"Type d'élément inconnu : '{etype_name}'. Connus : {known}"
            )
        elem_cls = _ELEM_REGISTRY[etype_name]

        # DDL par nœud effectif. Pour les éléments flexibles (Spring/Damper) il
        # est déduit de la longueur du vecteur raideur/amortissement ; sinon des
        # métadonnées (None ⇒ n_dim).
        if etype_name in _FLEX_ELEMS:
            vec_key = "stiffness" if etype_name == "Spring" else "damping"
            if vec_key not in edict:
                raise ValueError(
                    f"L'élément '{etype_name}' requiert la clé '{vec_key}'."
                )
            dpn_eff = len(_as_float_list(edict[vec_key]))
            explicit = True
        else:
            _, meta_dpn = _ELEM_META[etype_name]
            dpn_eff = meta_dpn if meta_dpn is not None else n_dim
            explicit = meta_dpn is not None

        # Cohérence du dof_per_node sur tout le maillage
        if dpn_common is None:
            dpn_common = dpn_eff
        elif dpn_common != dpn_eff:
            raise ValueError(
                f"Mélange de dof_per_node incompatibles dans les éléments : "
                f"{dpn_common} vs {dpn_eff} pour '{etype_name}'"
            )
        any_explicit_dpn = any_explicit_dpn or explicit

        # Matériau — optionnel pour les connecteurs ponctuels (Spring/Damper)
        if "material" in edict:
            mat_key = edict["material"]
            if mat_key not in materials:
                raise ValueError(
                    f"Matériau '{mat_key}' non défini. Matériaux disponibles : "
                    f"{list(materials)}"
                )
            material = materials[mat_key]
        elif etype_name in _FLEX_ELEMS:
            material = _FLEX_MATERIAL
        else:
            raise ValueError(
                f"L'élément '{etype_name}' requiert une clé 'material'."
            )

        node_ids = tuple(int(n) for n in edict["nodes"])
        props = _parse_element_properties(etype_name, edict)
        elements_list.append(
            ElementData(etype=elem_cls, node_ids=node_ids, material=material, properties=props)
        )

    # dof_per_node passé au maillage : None si tous les éléments suivent n_dim
    # (comportement historique), sinon la valeur explicite commune.
    dof_per_node_mesh = dpn_common if any_explicit_dpn else None

    mesh = Mesh(
        nodes=nodes,
        elements=tuple(elements_list),
        n_dim=n_dim,
        dof_per_node=dof_per_node_mesh,
    )

    # --- Conditions aux limites ---
    bc_dict = data.get("boundary_conditions", {})
    dpn = mesh.dpn

    # Dirichlet
    dirichlet: dict[int, dict[int, float]] = {}
    for entry in bc_dict.get("dirichlet", []):
        node = int(entry["node"])
        dof = int(entry["dof"])
        val = float(entry.get("value", 0.0))
        dirichlet.setdefault(node, {})[dof] = val

    # Neumann
    neumann: dict[int, dict[int, float]] = {}
    for entry in bc_dict.get("neumann", []):
        node = int(entry["node"])
        dof = int(entry["dof"])
        val = float(entry["value"])
        neumann.setdefault(node, {})[dof] = val

    # Pression
    pressure_loads: list[PressureLoad] = []
    for entry in bc_dict.get("pressure", []):
        pnodes = tuple(int(n) for n in entry["nodes"])
        mag = float(entry["magnitude"])
        pressure_loads.append(PressureLoad(node_ids=pnodes, magnitude=mag))

    # Force de volume
    body_force: BodyForce | None = None
    if "body_force" in bc_dict:
        acc = tuple(float(a) for a in bc_dict["body_force"])
        body_force = BodyForce(acceleration=acc)

    # Charges linéiques distribuées
    distributed_loads: list[DistributedLineLoad] = []
    for entry in bc_dict.get("distributed", []):
        pnodes = tuple(int(n) for n in entry["nodes"])
        qx = float(entry.get("qx", 0.0))
        qy = float(entry.get("qy", 0.0))
        distributed_loads.append(DistributedLineLoad(node_ids=pnodes, qx=qx, qy=qy))

    bc = BoundaryConditions(
        dirichlet=dirichlet,
        neumann=neumann,
        pressure=tuple(pressure_loads),
        body_force=body_force,
        distributed=tuple(distributed_loads),
    )

    return mesh, bc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_model(path: str | Path) -> FEModel:
    """Charge un modèle FEM depuis un fichier JSON.

    Parameters
    ----------
    path : str or Path
        Chemin vers le fichier JSON.

    Returns
    -------
    FEModel
        Modèle prêt à être résolu.

    Raises
    ------
    FileNotFoundError
        Si le fichier n'existe pas.
    ValueError
        Si le JSON est invalide ou contient des types inconnus.

    Examples
    --------
    >>> model = load_model("examples/warren_truss.json")
    >>> model.mesh.n_nodes
    9
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    mesh, bc = _build_mesh_and_bc(data)

    return FEModel(
        name=data.get("name", path.stem),
        mesh=mesh,
        bc=bc,
        analysis=data.get("analysis", {"type": "static"}),
        description=data.get("description", ""),
    )


def run_from_json(
    path: str | Path, *, verbose: bool = True
) -> dict[str, Any]:
    """Charge, résout et retourne les résultats d'un modèle JSON.

    Dispatche automatiquement vers le solveur approprié selon
    ``analysis.type`` :

    - ``"static"``       → StaticSolver, retourne ``u``
    - ``"modal"``        → run_modal, retourne ``freqs`` et ``phi``
    - ``"buckling"``     → BucklingSolver, retourne ``lambda_cr`` et ``phi``
    - ``"harmonic"``     → run_harmonic, retourne ``freqs`` et ``amplitude``
    - ``"transient"``    → run_transient, retourne ``times`` et ``u``
    - ``"random_force"`` → run_random_force, retourne ``rms_u`` et ``rms_a``
    - ``"random_base"``  → run_random_base, retourne ``rms_u`` et ``rms_a``

    Parameters
    ----------
    path : str or Path
        Chemin vers le fichier JSON.
    verbose : bool
        Si ``True``, affiche un résumé des résultats.

    Returns
    -------
    dict[str, Any]
        Résultats sérialisables (numpy arrays convertis en listes Python).

    Examples
    --------
    >>> results = run_from_json("examples/warren_truss.json")
    >>> results["u"]   # déplacements nodaux [m]
    """
    model = load_model(path)
    return solve_model(model, verbose=verbose)


def solve_model(model: FEModel, *, verbose: bool = True) -> dict[str, Any]:
    """Résout un FEModel déjà chargé et retourne les résultats.

    Sépare le chargement (``load_model``) de la résolution, ce qui permet
    au code appelant (p. ex. le CLI) d'accéder au modèle sans double parsing.

    Parameters
    ----------
    model : FEModel
        Modèle chargé par ``load_model``.
    verbose : bool
        Si ``True``, journalise un résumé des résultats via ``logging``.

    Returns
    -------
    dict[str, Any]
        Résultats sérialisables. Mêmes clés que ``run_from_json``.

    Examples
    --------
    >>> model = load_model("examples/warren_truss.json")
    >>> results = solve_model(model)
    >>> results["u"]
    """
    mesh = model.mesh
    bc = model.bc
    analysis = model.analysis
    atype = analysis.get("type", "static")

    if verbose:
        logger.info("Modèle '%s' chargé — %d nœuds, %d éléments, analyse '%s'",
                    model.name, mesh.n_nodes, len(mesh.elements), atype)

    # Vérification de santé du modèle avant résolution (style Abaqus) :
    # erreurs bloquantes (orphelins, Jacobien, singularité) → ModelError ;
    # avertissements (nœuds coïncidents, doublons, qualité, conditionnement)
    # loggés, le calcul continue.
    run_model_checks(mesh, bc, analysis_type=atype).raise_if_errors()

    results = _dispatch_analysis(mesh, bc, analysis, atype, verbose=verbose)
    results["name"] = model.name
    results["analysis_type"] = atype
    return results


# ---------------------------------------------------------------------------
# Dispatch d'analyse
# ---------------------------------------------------------------------------

def _dispatch_analysis(
    mesh: Mesh,
    bc: BoundaryConditions,
    analysis: dict[str, Any],
    atype: str,
    *,
    verbose: bool,
) -> dict[str, Any]:
    """Résout le modèle selon le type d'analyse.

    Parameters
    ----------
    mesh, bc, analysis, atype, verbose
        Voir ``run_from_json``.

    Returns
    -------
    dict[str, Any]
    """
    if atype == "static":
        return _run_static(mesh, bc, analysis, verbose=verbose)
    if atype == "modal":
        return _run_modal(mesh, bc, analysis, verbose=verbose)
    if atype == "buckling":
        return _run_buckling(mesh, bc, analysis, verbose=verbose)
    if atype == "harmonic":
        return _run_harmonic(mesh, bc, analysis, verbose=verbose)
    if atype == "transient":
        return _run_transient(mesh, bc, analysis, verbose=verbose)
    if atype == "random_force":
        return _run_random_force(mesh, bc, analysis, verbose=verbose)
    if atype == "random_base":
        return _run_random_base(mesh, bc, analysis, verbose=verbose)

    known = "static, modal, buckling, harmonic, transient, random_force, random_base"
    raise ValueError(f"Type d'analyse inconnu : '{atype}'. Connus : {known}")


# ---------------------------------------------------------------------------
# Analyses spécifiques
# ---------------------------------------------------------------------------

def _run_static(
    mesh: Mesh,
    bc: BoundaryConditions,
    analysis: dict[str, Any],
    *,
    verbose: bool,
) -> dict[str, Any]:
    assembler = Assembler(mesh)
    K = assembler.assemble_stiffness()
    F = assembler.assemble_forces(bc)

    # Chargement thermomécanique : F_th = ∫ Bᵀ D (α ΔT m) dV s'ajoute au
    # second membre. ΔT uniforme (scalaire) ou nodal (interpolé par N).
    delta_T = _parse_thermal(analysis, mesh.n_nodes)
    if delta_T is not None:
        F = F + assembler.assemble_thermal_forces(delta_T)

    ds = apply_dirichlet(K, F, mesh, bc)
    u = StaticSolver().solve(ds.K_free, ds.F_free)
    u_full = ds.recover(u)
    if verbose:
        u_max = float(np.abs(u_full).max())
        logger.info("  Statique : u_max = %.4e m", u_max)
    return {"u": u_full.tolist()}


def _run_modal(
    mesh: Mesh,
    bc: BoundaryConditions,
    analysis: dict[str, Any],
    *,
    verbose: bool,
) -> dict[str, Any]:
    n_modes = int(analysis.get("n_modes", 5))
    use_lumped = bool(analysis.get("use_lumped", False))
    result = run_modal(mesh, bc, n_modes=n_modes, use_lumped=use_lumped)
    if verbose:
        freq_str = ", ".join(f"{f:.3f}" for f in result.freqs)
        logger.info("  Modal : fréquences [Hz] = [%s]", freq_str)
    return {
        "freqs": result.freqs.tolist(),
        "modes": result.modes.tolist(),
    }


def _run_buckling(
    mesh: Mesh,
    bc: BoundaryConditions,
    analysis: dict[str, Any],
    *,
    verbose: bool,
) -> dict[str, Any]:
    n_modes = int(analysis.get("n_modes", 1))
    assembler = Assembler(mesh)
    K = assembler.assemble_stiffness()
    F = assembler.assemble_forces(bc)
    ds = apply_dirichlet(K, F, mesh, bc)
    u = StaticSolver().solve(ds.K_free, ds.F_free)
    u_full = ds.recover(u)
    K_g = assembler.assemble_geometric_stiffness(u_full)
    K_g_free = ds.reduce(K_g)
    lambda_cr, phi_free = BucklingSolver().solve(ds.K_free, K_g_free, n_modes=n_modes)
    phi = ds.recover_modes(phi_free)
    if verbose:
        lam_str = ", ".join(f"{l:.4f}" for l in lambda_cr)
        logger.info("  Flambage : lambda_cr = [%s]", lam_str)
    return {
        "lambda_cr": lambda_cr.tolist(),
        "phi": phi.tolist(),
    }


def _run_harmonic(
    mesh: Mesh,
    bc: BoundaryConditions,
    analysis: dict[str, Any],
    *,
    verbose: bool,
) -> dict[str, Any]:
    freqs = _parse_freqs(analysis["freqs"])
    dpn = mesh.dpn
    F_hat = _build_F_hat_with_dpn(analysis.get("F_hat"), mesh.n_dof, dpn)
    damping = _parse_damping(analysis.get("damping"))
    result = run_harmonic(mesh, bc, F_hat, freqs, damping)
    amplitude = np.abs(result.U)
    if verbose:
        a_max = float(amplitude.max())
        logger.info("  Harmonique : amplitude_max = %.4e m", a_max)
    return {
        "freqs": freqs.tolist(),
        "amplitude": amplitude.tolist(),
        "U_real": result.U.real.tolist(),
        "U_imag": result.U.imag.tolist(),
    }


def _run_transient(
    mesh: Mesh,
    bc: BoundaryConditions,
    analysis: dict[str, Any],
    *,
    verbose: bool,
) -> dict[str, Any]:
    dt = float(analysis["dt"])
    n_steps = int(analysis["n_steps"])
    dpn = mesh.dpn
    F_hat = _build_F_hat_with_dpn(analysis.get("F_hat"), mesh.n_dof, dpn)
    u0 = np.zeros(mesh.n_dof)
    v0 = np.zeros(mesh.n_dof)
    damping = _parse_damping(analysis.get("damping"))
    result = run_transient(mesh, bc, F_hat, u0, v0, dt, n_steps, damping)
    if verbose:
        u_max = float(np.abs(result.u).max())
        logger.info("  Transitoire : u_max = %.4e m (sur %d pas)", u_max, n_steps)
    return {
        "times": result.t.tolist(),
        "u": result.u.tolist(),
    }


def _run_random_force(
    mesh: Mesh,
    bc: BoundaryConditions,
    analysis: dict[str, Any],
    *,
    verbose: bool,
) -> dict[str, Any]:
    freqs = _parse_freqs(analysis["freqs"])
    G0 = float(analysis["G0"])
    dpn = mesh.dpn
    # Direction de la force : liste de contributions nodales
    F_dir = _build_F_hat_with_dpn(analysis.get("F_dir"), mesh.n_dof, dpn)
    if np.allclose(F_dir, 0.0):
        raise ValueError(
            "random_force requiert 'F_dir' (liste de contributions nodales) non nul."
        )
    G_input = psd_white(G0, freqs)
    damping = _parse_damping(analysis.get("damping"))
    result = run_random_force(mesh, bc, F_dir, G_input, freqs, damping)
    if verbose:
        logger.info("  Aléatoire force : rms_u_max = %.4e m", float(result.rms_u.max()))
    return {
        "rms_u": result.rms_u.tolist(),
        "rms_a": result.rms_a.tolist(),
        "freqs": freqs.tolist(),
    }


def _run_random_base(
    mesh: Mesh,
    bc: BoundaryConditions,
    analysis: dict[str, Any],
    *,
    verbose: bool,
) -> dict[str, Any]:
    freqs = _parse_freqs(analysis["freqs"])
    G0 = float(analysis["G0"])
    direction = int(analysis.get("direction", 1))
    G_input = psd_white(G0, freqs)
    damping = _parse_damping(analysis.get("damping"))
    result = run_random_base(mesh, bc, direction=direction, G_input=G_input,
                              freqs=freqs, damping=damping)
    if verbose:
        logger.info("  Aléatoire base : rms_u_max = %.4e m, rms_a_max = %.4e m/s²",
                    float(result.rms_u.max()), float(result.rms_a.max()))
    return {
        "rms_u": result.rms_u.tolist(),
        "rms_a": result.rms_a.tolist(),
        "freqs": freqs.tolist(),
    }

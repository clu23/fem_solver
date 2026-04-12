"""Estimateur d'erreur a posteriori de Zienkiewicz–Zhu (ZZ).

Principe — du champ discontinu au champ lissé
----------------------------------------------
En FEM, les contraintes brutes σ_h = D·B_e·u_e sont **discontinues** entre
éléments : chaque élément calcule ses déformations à partir de ses seuls
degrés de liberté. Aux interfaces, les valeurs de deux éléments voisins
diffèrent — ce saut est directement lié à l'erreur de discrétisation.

L'idée de ZZ (1987) est de construire un champ σ* **continu** et **plus
précis** que σ_h par lissage, puis de mesurer l'écart σ* − σ_h.

Deux méthodes de lissage
-------------------------
``"sna"`` — Simple Nodal Averaging
    σ*(nœud) = moyenne des contributions élémentaires adjacentes
    (extrapolées aux nœuds via A⁻¹ pour les Quad4).  Simple et rapide.

``"spr"`` — Superconvergent Patch Recovery (Zienkiewicz & Zhu, 1987)
    Pour chaque nœud n, les points de Gauss de tous les éléments adjacents
    (le *patch* de n) sont superconvergents : l'erreur FEM y est d'un ordre
    supérieur.  On ajuste un polynôme linéaire

        σ*(x, y) = a₀ + a₁·(x−xₙ) + a₂·(y−yₙ)

    aux valeurs aux GPs par moindres carrés.  La valeur au nœud est a₀.
    Quand le patch est trop petit (< 3 GPs), repli sur la moyenne nodale.

Indicateur élémentaire
-----------------------
    η_e² = t ∫_Ωe (σ* − σ_h)ᵀ D⁻¹ (σ* − σ_h) dΩ
         ≈ t Σ_p w_p |det Jₚ| Δσₚᵀ D⁻¹ Δσₚ

Norme énergétique du champ FEM
--------------------------------
    ‖σ_h‖² = Σ_e t Σ_p w_p |det Jₚ| σ_h_pᵀ D⁻¹ σ_h_p

Erreur globale et relative
---------------------------
    η      = √(Σ_e η_e²)
    e_rel  = η / ‖σ_h‖   (≤ 1 en pratique)

Signification physique d'un indicateur élevé
--------------------------------------------
η_e élevé → le maillage est trop grossier localement pour résoudre les
gradients de contraintes.  Sur la plaque trouée, le champ de Kirsch varie
en 1/r² près du trou (r = R) : gradients intenses → sauts de contraintes
importants entre éléments → η_e maximal à r ≈ R.  L'indicateur guide
directement le raffinement adaptatif.

Références
----------
Zienkiewicz, O.C. & Zhu, J.Z. (1987). A simple error estimator and
adaptive procedure for practical engineering analysis.
*Int. J. Numer. Meth. Engng* 24, 337–357.

Cook, R.D. et al. (2002). *Concepts and Applications of FEA*, 4th ed.,
§17.4.  Wiley.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from femsolver.core.mesh import Mesh
from femsolver.elements.quad4 import Quad4, _GAUSS_POINTS_2X2
from femsolver.elements.tri3 import Tri3
from femsolver.postprocess.stress import nodal_stresses

# ---------------------------------------------------------------------------
# Matrice d'évaluation des fonctions de forme aux GPs de Gauss pour Quad4
# (direction forward : σ*_GP = _GP_INTERP_Q4 @ σ*_nodes)
# ---------------------------------------------------------------------------
# _GAUSS_POINTS_2X2 = [(-g,-g,1), (+g,-g,1), (+g,+g,1), (-g,+g,1)]
# A[i,j] = N_j(ξ_i, η_i) — évaluation des f. de forme au GP_i
_g = 1.0 / np.sqrt(3.0)
_p = 0.25 * (1.0 + _g) ** 2
_q = 0.25 * (1.0 - _g ** 2)
_r = 0.25 * (1.0 - _g) ** 2

_GP_INTERP_Q4: np.ndarray = np.array(
    [
        [_p, _q, _r, _q],   # GP0 = (-g,-g) : N1=_p, N2=_q, N3=_r, N4=_q
        [_q, _p, _q, _r],   # GP1 = (+g,-g)
        [_r, _q, _p, _q],   # GP2 = (+g,+g)
        [_q, _r, _q, _p],   # GP3 = (-g,+g)
    ]
)


# ---------------------------------------------------------------------------
# Résultat
# ---------------------------------------------------------------------------


@dataclass
class ZZErrorResult:
    """Résultat de l'estimateur d'erreur ZZ.

    Attributes
    ----------
    eta_e : np.ndarray, shape (n_elem,)
        Indicateur d'erreur élémentaire [Pa·√m].
        Norme énergie de (σ* − σ_h) sur chaque élément.
    eta : float
        Erreur globale = √(Σ η_e²).
    norm_sigma_h : float
        Norme énergétique du champ FEM = √(Σ ‖σ_h‖_e²).
    relative_error : float
        Erreur relative = η / ‖σ_h‖ (sans dimension).
    sigma_nodal : np.ndarray, shape (n_nodes, 3)
        Contraintes lissées σ* utilisées pour l'estimation.
    method : str
        Méthode de lissage utilisée (``"sna"`` ou ``"spr"``).
    """

    eta_e: np.ndarray
    eta: float
    norm_sigma_h: float
    relative_error: float
    sigma_nodal: np.ndarray
    method: str


# ---------------------------------------------------------------------------
# Fonction publique principale
# ---------------------------------------------------------------------------


def zz_error_estimate(
    mesh: Mesh,
    u: np.ndarray,
    formulation: str = "plane_stress",
    method: str = "spr",
) -> ZZErrorResult:
    """Estime l'erreur a posteriori par la méthode Zienkiewicz–Zhu.

    Parameters
    ----------
    mesh : Mesh
        Maillage 2D (nœuds + éléments Tri3 ou Quad4).
    u : np.ndarray, shape (n_dof,)
        Vecteur de déplacements global.
    formulation : str
        ``"plane_stress"`` (défaut) ou ``"plane_strain"``.
    method : str
        Méthode de lissage : ``"spr"`` (SPR, défaut) ou ``"sna"``
        (Simple Nodal Averaging).

    Returns
    -------
    ZZErrorResult
        Contient les indicateurs élémentaires, l'erreur globale et relative,
        et le champ σ* lissé.

    Raises
    ------
    ValueError
        Si ``method`` n'est pas ``"spr"`` ou ``"sna"``.
    NotImplementedError
        Si un type d'élément non supporté est rencontré.

    Examples
    --------
    >>> result = zz_error_estimate(mesh, u)
    >>> print(f"Erreur relative : {result.relative_error:.1%}")
    >>> # Éléments les plus erronés
    >>> worst = np.argsort(result.eta_e)[-5:]

    Notes
    -----
    Algorithme :

    1. Lissage → σ* nodal (SNA ou SPR).
    2. Pour chaque élément, aux points de Gauss :
       - σ_h_p = D · B_p · u_e  (contrainte FEM brute)
       - σ*_p  = Σ_i N_i(GP_p) · σ*_{node_i}  (interpolation du champ lissé)
       - Δσ_p  = σ*_p − σ_h_p
    3. η_e² = t · Σ_p w_p |det Jₚ| Δσₚᵀ D⁻¹ Δσₚ
    4. η = √(Σ η_e²),  e_rel = η / ‖σ_h‖

    Références : Zienkiewicz & Zhu (1987), Cook et al. (2002) §17.4.
    """
    if method not in ("spr", "sna"):
        raise ValueError(f"method doit être 'spr' ou 'sna', reçu '{method}'")

    # --- Étape 1 : lissage σ* -----------------------------------------------
    if method == "sna":
        sigma_nodal = nodal_stresses(mesh, u, formulation)
    else:
        sigma_nodal = _spr_nodal_stresses(mesh, u, formulation)

    # --- Étapes 2–4 : indicateurs élémentaires -------------------------------
    n_elem = len(mesh.elements)
    eta_e_sq = np.zeros(n_elem)
    norm_sq = np.zeros(n_elem)

    for idx, elem_data in enumerate(mesh.elements):
        elem = elem_data.get_element()
        node_coords = mesh.node_coords(elem_data.node_ids)
        dofs = mesh.global_dofs(elem_data.node_ids)
        u_e = u[dofs]
        t = elem_data.properties.get("thickness", 1.0)

        if isinstance(elem, Tri3):
            e2, n2 = _element_error_tri3(
                elem, elem_data.material, node_coords, u_e,
                elem_data.node_ids, sigma_nodal, formulation, t,
            )
        elif isinstance(elem, Quad4):
            e2, n2 = _element_error_quad4(
                elem, elem_data.material, node_coords, u_e,
                elem_data.node_ids, sigma_nodal, formulation, t,
            )
        else:
            raise NotImplementedError(
                f"Estimateur ZZ non implémenté pour {type(elem).__name__}"
            )

        eta_e_sq[idx] = e2
        norm_sq[idx] = n2

    eta_e = np.sqrt(np.maximum(eta_e_sq, 0.0))
    eta = float(np.sqrt(np.sum(eta_e_sq)))
    norm_sigma_h = float(np.sqrt(np.sum(norm_sq)))

    if norm_sigma_h > 0.0:
        relative_error = eta / norm_sigma_h
    else:
        relative_error = 0.0

    return ZZErrorResult(
        eta_e=eta_e,
        eta=eta,
        norm_sigma_h=norm_sigma_h,
        relative_error=relative_error,
        sigma_nodal=sigma_nodal,
        method=method,
    )


# ---------------------------------------------------------------------------
# Lissage SPR — Superconvergent Patch Recovery
# ---------------------------------------------------------------------------


def _spr_nodal_stresses(
    mesh: Mesh,
    u: np.ndarray,
    formulation: str,
) -> np.ndarray:
    """Lissage SPR : ajustement polynomial aux points de Gauss du patch.

    Parameters
    ----------
    mesh : Mesh
        Maillage 2D.
    u : np.ndarray, shape (n_dof,)
        Déplacements globaux.
    formulation : str
        ``"plane_stress"`` ou ``"plane_strain"``.

    Returns
    -------
    sigma_spr : np.ndarray, shape (n_nodes, 3)
        Contraintes SPR lissées [σxx, σyy, τxy] à chaque nœud.

    Notes
    -----
    Algorithme (ZZ 1987) :

    1. Pour chaque élément, calculer σ_h aux points de Gauss et mémoriser
       les coordonnées physiques des GPs.
    2. Pour chaque nœud n de coordonnées (xₙ, yₙ) :
       - Rassembler tous les GPs des éléments adjacents.
       - Construire la matrice de base polynomiale :
             P[k] = [1, x_k − xₙ, y_k − yₙ]
       - Résoudre PᵀP a = Pᵀ σ_GP  par pseudo-inverse.
       - σ*(n) = a[0]  (terme constant = valeur au nœud).
    3. Si le patch est trop petit (< 3 GPs), repli sur la moyenne nodale.
    """
    n_nodes = mesh.n_nodes

    # --- Pré-calcul des contraintes et coordonnées aux GPs ------------------
    # gp_data[elem_idx] = list of (x_phys, y_phys, sigma_h : shape(3,))
    gp_data_by_elem: list[list[tuple[float, float, np.ndarray]]] = []

    for elem_data in mesh.elements:
        elem = elem_data.get_element()
        node_coords = mesh.node_coords(elem_data.node_ids)
        dofs = mesh.global_dofs(elem_data.node_ids)
        u_e = u[dofs]
        formul = elem_data.properties.get("formulation", formulation)

        gp_list: list[tuple[float, float, np.ndarray]] = []

        if isinstance(elem, Tri3):
            # Tri3 : 1 seul GP au centroïde (coordonnées physiques moyennées)
            B, _ = elem._strain_displacement_matrix(node_coords)
            D = elem._elasticity_matrix(elem_data.material, formul)
            sigma_h = D @ B @ u_e  # shape (3,)
            x_c = float(node_coords[:, 0].mean())
            y_c = float(node_coords[:, 1].mean())
            gp_list.append((x_c, y_c, sigma_h))

        elif isinstance(elem, Quad4):
            D = elem._elasticity_matrix(elem_data.material, formul)
            for xi, eta, _ in _GAUSS_POINTS_2X2:
                B, _ = elem._strain_displacement_matrix(xi, eta, node_coords)
                sigma_h = D @ B @ u_e  # shape (3,)
                # Coordonnées physiques du GP
                N = Quad4._shape_functions(xi, eta)       # (4,)
                x_gp = float(N @ node_coords[:, 0])
                y_gp = float(N @ node_coords[:, 1])
                gp_list.append((x_gp, y_gp, sigma_h))
        else:
            raise NotImplementedError(
                f"SPR non implémenté pour {type(elem).__name__}"
            )

        gp_data_by_elem.append(gp_list)

    # --- Construire la connectivité nœud → éléments -------------------------
    node_to_elems: list[list[int]] = [[] for _ in range(n_nodes)]
    for e_idx, elem_data in enumerate(mesh.elements):
        for nid in elem_data.node_ids:
            node_to_elems[nid].append(e_idx)

    # --- SPR par nœud -------------------------------------------------------
    sigma_spr = np.zeros((n_nodes, 3))

    for nid in range(n_nodes):
        x_n, y_n = mesh.nodes[nid, 0], mesh.nodes[nid, 1]

        # Rassembler les GPs du patch
        patch_gps: list[tuple[float, float, np.ndarray]] = []
        for e_idx in node_to_elems[nid]:
            patch_gps.extend(gp_data_by_elem[e_idx])

        n_gp = len(patch_gps)

        if n_gp < 3:
            # Patch trop petit — repli sur la moyenne des GPs disponibles
            if n_gp > 0:
                sigma_spr[nid] = np.mean(
                    [gp[2] for gp in patch_gps], axis=0
                )
            continue

        # Matrice de base polynomiale P (n_gp × 3)
        # p(x, y) = a₀ + a₁·(x − xₙ) + a₂·(y − yₙ)
        P = np.ones((n_gp, 3))
        P[:, 1] = [gp[0] - x_n for gp in patch_gps]
        P[:, 2] = [gp[1] - y_n for gp in patch_gps]

        # Valeurs aux GPs : shape (n_gp, 3) — une colonne par composante
        sigma_gp_patch = np.stack([gp[2] for gp in patch_gps])  # (n_gp, 3)

        # Moindres carrés : PᵀP a = Pᵀ σ_GP → a (3, 3)
        # La valeur au nœud est a[0, :] (terme constant, x=xₙ → dx=0)
        PtP = P.T @ P          # (3, 3)
        Pts = P.T @ sigma_gp_patch  # (3, 3) — une colonne par composante

        try:
            a = np.linalg.solve(PtP, Pts)   # (3, 3)
        except np.linalg.LinAlgError:
            # Système singulier (nœuds colinéaires, patch dégénéré) → SNA
            a = None

        if a is not None:
            sigma_spr[nid] = a[0]   # terme constant = valeur au nœud
        else:
            sigma_spr[nid] = np.mean(sigma_gp_patch, axis=0)

    return sigma_spr


# ---------------------------------------------------------------------------
# Indicateurs élémentaires
# ---------------------------------------------------------------------------


def _element_error_tri3(
    elem: Tri3,
    material,
    node_coords: np.ndarray,
    u_e: np.ndarray,
    node_ids: tuple[int, ...],
    sigma_nodal: np.ndarray,
    formulation: str,
    thickness: float,
) -> tuple[float, float]:
    """Indicateur d'erreur ZZ pour un élément Tri3.

    Tri3 : σ_h constant → 1 seul point d'intégration (centroïde, poids = A).

    Parameters
    ----------
    elem : Tri3
        Instance de l'élément.
    material : ElasticMaterial
        Matériau de l'élément.
    node_coords : np.ndarray, shape (3, 2)
        Coordonnées nodales.
    u_e : np.ndarray, shape (6,)
        Déplacements élémentaires.
    node_ids : tuple[int, ...]
        Indices globaux des 3 nœuds.
    sigma_nodal : np.ndarray, shape (n_nodes, 3)
        Contraintes lissées aux nœuds.
    formulation : str
        Formulation d'état plan.
    thickness : float
        Épaisseur [m].

    Returns
    -------
    eta_e_sq : float
        η_e² = t · A · Δσᵀ D⁻¹ Δσ
    norm_sq : float
        ‖σ_h‖_e² = t · A · σ_h_centᵀ D⁻¹ σ_h_cent
    """
    B, area = elem._strain_displacement_matrix(node_coords)
    D = elem._elasticity_matrix(material, formulation)
    D_inv = np.linalg.inv(D)

    # σ_h constant dans l'élément
    sigma_h = D @ B @ u_e  # shape (3,)

    # σ* au centroïde : N_i(centroïde) = 1/3 pour un triangle
    sigma_star = sigma_nodal[list(node_ids), :].mean(axis=0)  # shape (3,)

    delta = sigma_star - sigma_h  # shape (3,)

    weight = thickness * area
    eta_e_sq = float(weight * delta @ D_inv @ delta)
    norm_sq = float(weight * sigma_h @ D_inv @ sigma_h)

    return eta_e_sq, norm_sq


def _element_error_quad4(
    elem: Quad4,
    material,
    node_coords: np.ndarray,
    u_e: np.ndarray,
    node_ids: tuple[int, ...],
    sigma_nodal: np.ndarray,
    formulation: str,
    thickness: float,
) -> tuple[float, float]:
    """Indicateur d'erreur ZZ pour un élément Quad4.

    Quad4 : intégration 2×2 Gauss (4 points), même quadrature que K.

    Parameters
    ----------
    elem : Quad4
        Instance de l'élément.
    material : ElasticMaterial
        Matériau de l'élément.
    node_coords : np.ndarray, shape (4, 2)
        Coordonnées nodales.
    u_e : np.ndarray, shape (8,)
        Déplacements élémentaires.
    node_ids : tuple[int, ...]
        Indices globaux des 4 nœuds.
    sigma_nodal : np.ndarray, shape (n_nodes, 3)
        Contraintes lissées aux nœuds.
    formulation : str
        Formulation d'état plan.
    thickness : float
        Épaisseur [m].

    Returns
    -------
    eta_e_sq : float
        η_e² = t Σ_p w_p |det Jₚ| Δσₚᵀ D⁻¹ Δσₚ
    norm_sq : float
        ‖σ_h‖_e² = t Σ_p w_p |det Jₚ| σ_h_pᵀ D⁻¹ σ_h_p
    """
    D = elem._elasticity_matrix(material, formulation)
    D_inv = np.linalg.inv(D)

    # σ* nodaux de cet élément : shape (4, 3)
    sigma_star_nodes = sigma_nodal[list(node_ids), :]  # (4, 3)

    # σ* aux 4 GPs par interpolation : _GP_INTERP_Q4 @ sigma_star_nodes
    # _GP_INTERP_Q4 : (4, 4),  sigma_star_nodes : (4, 3)  →  (4, 3)
    sigma_star_gp = _GP_INTERP_Q4 @ sigma_star_nodes   # (4, 3)

    eta_e_sq = 0.0
    norm_sq = 0.0

    for k, (xi, eta, w) in enumerate(_GAUSS_POINTS_2X2):
        B, det_J = elem._strain_displacement_matrix(xi, eta, node_coords)
        sigma_h_p = D @ B @ u_e          # shape (3,)
        sigma_star_p = sigma_star_gp[k]  # shape (3,)

        delta_p = sigma_star_p - sigma_h_p  # shape (3,)
        weight = w * abs(det_J) * thickness

        eta_e_sq += weight * float(delta_p @ D_inv @ delta_p)
        norm_sq += weight * float(sigma_h_p @ D_inv @ sigma_h_p)

    return eta_e_sq, norm_sq

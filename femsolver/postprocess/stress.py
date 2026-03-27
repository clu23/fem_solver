"""Calcul, extrapolation et lissage des contraintes aux nœuds.

Stratégie par type d'élément
-----------------------------
Tri3 (CST)
    B est constant → σ est constant dans l'élément.
    On affecte la même valeur σ_elem aux 3 nœuds de l'élément.

Quad4 (Q4 isoparamétrique)
    σ varie sur l'élément. On calcule σ aux 4 points de Gauss,
    puis on extrapole aux 4 coins via la matrice A⁻¹ :

        σ_GP = A · σ_nodes   (A[i,j] = Nj évalué au GPi)
        →  σ_nodes = A⁻¹ · σ_GP

Lissage final
    Chaque nœud reçoit les contributions de tous les éléments adjacents.
    La valeur lissée est la moyenne arithmétique.
"""

from __future__ import annotations

import numpy as np

from femsolver.core.mesh import Mesh
from femsolver.elements.quad4 import Quad4, _GAUSS_POINTS_2X2
from femsolver.elements.tri3 import Tri3


# ---------------------------------------------------------------------------
# Matrice d'extrapolation Gauss → nœuds pour Quad4 2×2
# ---------------------------------------------------------------------------
# Gauss points: GP0=(-g,-g), GP1=(+g,-g), GP2=(+g,+g), GP3=(-g,+g)
# avec g = 1/√3
# Nœuds naturels: N0=(-1,-1), N1=(+1,-1), N2=(+1,+1), N3=(-1,+1)
#
# A[i,j] = Nj(GPi) = shape function j evaluated at Gauss point i
# σ_GP = A · σ_nodes  →  σ_nodes = A⁻¹ · σ_GP

_g = 1.0 / np.sqrt(3.0)
_p = 0.25 * (1.0 + _g) ** 2   # valeur max de Ni à son propre GP
_q = 0.25 * (1.0 - _g ** 2)   # valeur intermédiaire
_r = 0.25 * (1.0 - _g) ** 2   # valeur min (coin le plus éloigné)

# Ligne i = valeurs des 4 fonctions de forme au point de Gauss i
_A = np.array(
    [
        [_p, _q, _r, _q],   # GP0 = (-g,-g)
        [_q, _p, _q, _r],   # GP1 = (+g,-g)
        [_r, _q, _p, _q],   # GP2 = (+g,+g)
        [_q, _r, _q, _p],   # GP3 = (-g,+g)
    ]
)
_EXTRAP_Q4: np.ndarray = np.linalg.inv(_A)  # shape (4, 4) : nœuds × GPs


# ---------------------------------------------------------------------------
# Fonctions publiques
# ---------------------------------------------------------------------------


def nodal_stresses(
    mesh: Mesh,
    u: np.ndarray,
    formulation: str = "plane_stress",
) -> np.ndarray:
    """Calcule les contraintes lissées aux nœuds par moyenne des éléments.

    Parameters
    ----------
    mesh : Mesh
        Maillage (nœuds + éléments).
    u : np.ndarray, shape (n_dof,)
        Vecteur de déplacements global.
    formulation : str
        ``"plane_stress"`` (défaut) ou ``"plane_strain"``.

    Returns
    -------
    sigma : np.ndarray, shape (n_nodes, 3)
        Contraintes lissées ``[σxx, σyy, τxy]`` à chaque nœud [Pa].

    Notes
    -----
    Algorithme (Simple Nodal Averaging) :

    1. Pour chaque élément, calculer σ aux nœuds de l'élément :
       - Tri3 : σ constant dans l'élément → affecter σ_centroïde aux 3 nœuds.
       - Quad4 : calculer σ aux 4 GPs, extrapoler aux 4 coins via A⁻¹.
    2. Accumuler les contributions dans ``sigma_sum[node_id]``.
    3. Diviser par le nombre de contributions (``count[node_id]``).

    References
    ----------
    Cook et al., « Concepts and Applications of FEA », 4th ed., §11.3.
    """
    sigma_sum = np.zeros((mesh.n_nodes, 3))
    count = np.zeros(mesh.n_nodes, dtype=int)

    for elem_data in mesh.elements:
        elem = elem_data.get_element()
        node_coords = mesh.node_coords(elem_data.node_ids)
        dofs = mesh.global_dofs(elem_data.node_ids)
        u_e = u[dofs]

        if isinstance(elem, Tri3):
            sigma_e = _tri3_nodal_stresses(elem, elem_data.material,
                                           node_coords, u_e, formulation)
        elif isinstance(elem, Quad4):
            sigma_e = _quad4_nodal_stresses(elem, elem_data.material,
                                            node_coords, u_e, formulation)
        else:
            raise NotImplementedError(
                f"Extrapolation de contraintes non implémentée pour {type(elem).__name__}"
            )

        for local_i, nid in enumerate(elem_data.node_ids):
            sigma_sum[nid] += sigma_e[local_i]
            count[nid] += 1

    # Moyenne (un nœud isolé garde count=0 → contrainte nulle, pas de division)
    valid = count > 0
    sigma_avg = np.zeros((mesh.n_nodes, 3))
    sigma_avg[valid] = sigma_sum[valid] / count[valid, np.newaxis]
    return sigma_avg


def von_mises_2d(sigma: np.ndarray) -> np.ndarray:
    """Contrainte de Von Mises en état plan.

    Parameters
    ----------
    sigma : np.ndarray, shape (..., 3)
        Contraintes ``[σxx, σyy, τxy]`` — fonctionne sur des tableaux de
        n'importe quelle forme tant que la dernière dimension est 3.

    Returns
    -------
    sigma_vm : np.ndarray, shape (...)
        Contrainte équivalente de Von Mises [Pa].

    Notes
    -----
    En état plan de contraintes (σzz = σxz = σyz = 0) :

        σ_vm = √(σxx² − σxx·σyy + σyy² + 3·τxy²)

    Dérivée du critère de Von Mises 3D en substituant σzz = 0.
    """
    sxx = sigma[..., 0]
    syy = sigma[..., 1]
    txy = sigma[..., 2]
    return np.sqrt(sxx ** 2 - sxx * syy + syy ** 2 + 3.0 * txy ** 2)


def principal_stresses_2d(sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Contraintes principales σ1 ≥ σ2 en état plan.

    Parameters
    ----------
    sigma : np.ndarray, shape (n, 3)
        Contraintes ``[σxx, σyy, τxy]``.

    Returns
    -------
    s1 : np.ndarray, shape (n,)
        Contrainte principale majeure [Pa].
    s2 : np.ndarray, shape (n,)
        Contrainte principale mineure [Pa].

    Notes
    -----
    σ1,2 = (σxx+σyy)/2 ± √(((σxx−σyy)/2)² + τxy²)
    """
    sxx = sigma[:, 0]
    syy = sigma[:, 1]
    txy = sigma[:, 2]
    center = (sxx + syy) / 2.0
    radius = np.sqrt(((sxx - syy) / 2.0) ** 2 + txy ** 2)
    return center + radius, center - radius


# ---------------------------------------------------------------------------
# Fonctions internes
# ---------------------------------------------------------------------------


def _tri3_nodal_stresses(
    elem: Tri3,
    material,
    node_coords: np.ndarray,
    u_e: np.ndarray,
    formulation: str,
) -> np.ndarray:
    """Stress constant du Tri3, répété aux 3 nœuds. Shape (3, 3)."""
    sigma = elem.stress(material, node_coords, u_e, formulation)  # shape (3,)
    return np.tile(sigma, (3, 1))  # shape (3, 3) : même valeur sur les 3 nœuds


def _quad4_nodal_stresses(
    elem: Quad4,
    material,
    node_coords: np.ndarray,
    u_e: np.ndarray,
    formulation: str,
) -> np.ndarray:
    """Extrapolation Gauss → coins pour Quad4. Shape (4, 3).

    1. Calculer σ aux 4 points de Gauss → sigma_gp (4, 3).
    2. Extrapoler aux 4 coins : sigma_nodes = _EXTRAP_Q4 @ sigma_gp.
    """
    D = elem._elasticity_matrix(material, formulation)
    sigma_gp = np.zeros((4, 3))
    for k, (xi, eta, _) in enumerate(_GAUSS_POINTS_2X2):
        B, _ = elem._strain_displacement_matrix(xi, eta, node_coords)
        sigma_gp[k] = D @ B @ u_e   # shape (3,)

    return _EXTRAP_Q4 @ sigma_gp    # (4, 4) @ (4, 3) → (4, 3)

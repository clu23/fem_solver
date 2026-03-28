"""Calcul et lissage des contraintes aux nœuds pour les éléments 3D.

Stratégie par type d'élément
------------------------------
Tetra4 (CST-3D)
    B est constante → σ est constante dans tout l'élément.
    La même valeur σ_elem est assignée aux 4 nœuds de l'élément.

Hexa8
    B varie d'un point de Gauss à l'autre. On calcule σ aux 8 points de
    Gauss 2×2×2, puis on extrapole aux 8 coins via la matrice A⁻¹ :

        σ_GP[i] = Σⱼ A[i,j] · σ_nodes[j]   avec A[i,j] = Nⱼ(GPᵢ)
        →  σ_nodes = A⁻¹ · σ_GP

    La matrice A est calculée une fois et mise en cache à l'import.

Lissage (moyennage nodal)
--------------------------
Chaque nœud peut appartenir à plusieurs éléments. La valeur finale lissée
est la moyenne arithmétique des contributions de chacun des éléments
adjacents (Simple Nodal Averaging).

Conventions
-----------
Vecteur de contrainte en notation de Voigt 3D :
    σ = [σxx, σyy, σzz, τyz, τxz, τxy]

Contrainte de Von Mises 3D :
    σ_vm = √½ · √[(σxx−σyy)² + (σyy−σzz)² + (σzz−σxx)² + 6(τyz²+τxz²+τxy²)]

References
----------
Cook et al., « Concepts and Applications of FEA », 4th ed., §11.3.
Zienkiewicz & Taylor, « The FEM — Solid Mechanics », vol. 2, §14.3.
"""

from __future__ import annotations

import numpy as np

from femsolver.core.mesh import Mesh
from femsolver.elements.hexa8 import Hexa8, _GAUSS_POINTS_2X2X2
from femsolver.elements.tetra4 import Tetra4


# ---------------------------------------------------------------------------
# Matrice d'extrapolation Gauss → nœuds pour Hexa8 2×2×2 (précalculée)
# ---------------------------------------------------------------------------
#
# A[i, j] = Nⱼ(GPᵢ)  — valeur de la fonction de forme j au point de Gauss i
# σ_GP = A · σ_nodes  →  σ_nodes = A⁻¹ · σ_GP = EXTRAP_H8 · σ_GP
#
# Les lignes de A sont les GPs, les colonnes les nœuds.
# La matrice est carrée 8×8 et bien conditionnée (det ≠ 0).

def _build_hexa8_extrapolation_matrix() -> np.ndarray:
    """Construit la matrice d'extrapolation A⁻¹ (8×8) pour Hexa8.

    A[i, j] = Nⱼ(GPᵢ) : évalue la fonction de forme j au point de Gauss i.
    La matrice EXTRAP = A⁻¹ permet de passer des valeurs aux GPs aux
    valeurs aux nœuds : σ_nodes = EXTRAP · σ_GP.
    """
    A = np.zeros((8, 8))
    for i, (xi, eta, zeta, _) in enumerate(_GAUSS_POINTS_2X2X2):
        A[i] = Hexa8._shape_functions(xi, eta, zeta)   # Nⱼ(GPᵢ), j=0..7
    return np.linalg.inv(A)   # shape (8, 8)


# Précalculé une seule fois à l'import du module
_EXTRAP_H8: np.ndarray = _build_hexa8_extrapolation_matrix()


# ---------------------------------------------------------------------------
# API publique
# ---------------------------------------------------------------------------


def nodal_stresses_3d(
    mesh: Mesh,
    u: np.ndarray,
) -> np.ndarray:
    """Contraintes lissées aux nœuds pour un maillage 3D (Tetra4 et Hexa8).

    Parameters
    ----------
    mesh : Mesh
        Maillage 3D (n_dim doit être 3).
    u : np.ndarray, shape (n_dof,)
        Vecteur de déplacements global [m].

    Returns
    -------
    sigma : np.ndarray, shape (n_nodes, 6)
        Contraintes lissées ``[σxx, σyy, σzz, τyz, τxz, τxy]`` à chaque
        nœud [Pa].

    Raises
    ------
    NotImplementedError
        Si un type d'élément non supporté est rencontré.

    Notes
    -----
    Algorithme (Simple Nodal Averaging) :

    1. Pour chaque élément, calculer σ aux nœuds locaux :
       - Tetra4 : σ constante → même valeur aux 4 nœuds.
       - Hexa8 : σ aux 8 GPs → extrapolation aux coins via EXTRAP_H8.
    2. Accumuler ``sigma_sum[node_id] += σ_local[i]``.
    3. Diviser par le nombre de contributions ``count[node_id]``.

    References
    ----------
    Cook et al., § 11.3 — Stress recovery and extrapolation.

    Examples
    --------
    >>> sigma = nodal_stresses_3d(mesh, u)
    >>> vm = von_mises_3d(sigma)
    >>> print(f"σ_VM max = {vm.max() / 1e6:.2f} MPa")
    """
    sigma_sum = np.zeros((mesh.n_nodes, 6))
    count     = np.zeros(mesh.n_nodes, dtype=int)

    for elem_data in mesh.elements:
        elem = elem_data.get_element()
        node_coords = mesh.node_coords(elem_data.node_ids)
        dofs  = mesh.global_dofs(elem_data.node_ids)
        u_e   = u[dofs]

        if isinstance(elem, Tetra4):
            sigma_e = _tetra4_nodal_stresses(elem, elem_data.material, node_coords, u_e)
        elif isinstance(elem, Hexa8):
            sigma_e = _hexa8_nodal_stresses(elem, elem_data.material, node_coords, u_e)
        else:
            raise NotImplementedError(
                f"Extrapolation de contraintes non implémentée pour "
                f"'{type(elem).__name__}'. Supportés : Tetra4, Hexa8."
            )

        for local_i, nid in enumerate(elem_data.node_ids):
            sigma_sum[nid] += sigma_e[local_i]
            count[nid] += 1

    valid = count > 0
    sigma_avg = np.zeros((mesh.n_nodes, 6))
    sigma_avg[valid] = sigma_sum[valid] / count[valid, np.newaxis]
    return sigma_avg


def von_mises_3d(sigma: np.ndarray) -> np.ndarray:
    """Contrainte équivalente de Von Mises (état 3D général).

    Parameters
    ----------
    sigma : np.ndarray, shape (..., 6)
        Contraintes ``[σxx, σyy, σzz, τyz, τxz, τxy]`` [Pa]. Fonctionne
        sur des tableaux de n'importe quelle forme tant que la dernière
        dimension est 6.

    Returns
    -------
    sigma_vm : np.ndarray, shape (...)
        Contrainte équivalente de Von Mises [Pa].

    Notes
    -----
    Formule en notation de Voigt :

        σ_vm = √{ ½ [(σxx−σyy)² + (σyy−σzz)² + (σzz−σxx)²
                      + 6(τyz² + τxz² + τxy²)] }

    Correspond au critère de plasticité de Von Mises :
        σ_vm < σ_yield  ← comportement élastique.

    Examples
    --------
    >>> sigma = np.array([100., 0., 0., 0., 0., 0.])   # traction uniaxiale
    >>> float(von_mises_3d(sigma))
    100.0
    """
    sxx = sigma[..., 0]
    syy = sigma[..., 1]
    szz = sigma[..., 2]
    tyz = sigma[..., 3]
    txz = sigma[..., 4]
    txy = sigma[..., 5]
    return np.sqrt(0.5 * (
        (sxx - syy) ** 2
        + (syy - szz) ** 2
        + (szz - sxx) ** 2
        + 6.0 * (tyz ** 2 + txz ** 2 + txy ** 2)
    ))


def principal_stresses_3d(sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Contraintes principales σ₁ ≥ σ₂ ≥ σ₃ (valeurs propres du tenseur de contrainte).

    Parameters
    ----------
    sigma : np.ndarray, shape (n_nodes, 6)
        Contraintes ``[σxx, σyy, σzz, τyz, τxz, τxy]``.

    Returns
    -------
    s1, s2, s3 : np.ndarray, shape (n_nodes,)
        Contraintes principales (valeurs propres) triées par ordre décroissant.

    Notes
    -----
    Pour chaque nœud, construit le tenseur symétrique 3×3 et calcule ses
    valeurs propres réelles (garanties car tenseur symétrique).
    """
    n = sigma.shape[0]
    s1 = np.empty(n)
    s2 = np.empty(n)
    s3 = np.empty(n)

    for k in range(n):
        sxx, syy, szz, tyz, txz, txy = sigma[k]
        tensor = np.array([
            [sxx, txy, txz],
            [txy, syy, tyz],
            [txz, tyz, szz],
        ])
        vals = np.linalg.eigvalsh(tensor)
        s3[k], s2[k], s1[k] = np.sort(vals)   # croissant → décroissant

    return s1, s2, s3


# ---------------------------------------------------------------------------
# Fonctions internes
# ---------------------------------------------------------------------------


def _tetra4_nodal_stresses(
    elem: Tetra4,
    material: object,
    node_coords: np.ndarray,
    u_e: np.ndarray,
) -> np.ndarray:
    """Contraintes constantes du Tetra4 répétées aux 4 nœuds. Shape (4, 6)."""
    sigma = elem.stress(material, node_coords, u_e)   # shape (6,)
    return np.tile(sigma, (4, 1))                      # (4, 6)


def _hexa8_nodal_stresses(
    elem: Hexa8,
    material: object,
    node_coords: np.ndarray,
    u_e: np.ndarray,
) -> np.ndarray:
    """Extrapolation Gauss → coins pour Hexa8. Shape (8, 6).

    1. Calcul de σ aux 8 points de Gauss → sigma_gp (8, 6).
    2. Extrapolation aux 8 coins : sigma_nodes = EXTRAP_H8 @ sigma_gp.

    Notes
    -----
    EXTRAP_H8 = A⁻¹ avec A[i,j] = Nⱼ(GPᵢ).
    Cette extrapolation est équivalente à une interpolation polynomiale
    trilinéaire des valeurs aux GPs, évaluée aux coins (±1,±1,±1).
    """
    D = material.elasticity_matrix_3d()

    sigma_gp = np.zeros((8, 6))
    for k, (xi, eta, zeta, _) in enumerate(_GAUSS_POINTS_2X2X2):
        B, _ = elem._strain_displacement_matrix(xi, eta, zeta, node_coords)
        sigma_gp[k] = D @ B @ u_e   # shape (6,)

    return _EXTRAP_H8 @ sigma_gp    # (8, 8) @ (8, 6) → (8, 6)

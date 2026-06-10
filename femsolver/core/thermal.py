"""Outils communs pour le chargement thermomécanique (déformations initiales).

Modèle
------
Une variation de température ΔT impose une **déformation libre** (sans
contrainte si la dilatation n'est pas empêchée) :

    ε_th = α · ΔT · m

où ``m`` est le vecteur unitaire de dilatation isotrope (les termes de
cisaillement sont nuls : une dilatation thermique isotrope ne distord pas) :

    - 2D (Voigt [εxx, εyy, γxy])           : m = [1, 1, 0]
    - 3D (Voigt [εxx, εyy, εzz, γyz, γxz, γxy]) : m = [1, 1, 1, 0, 0, 0]

La contrainte totale est calculée sur la **déformation mécanique**
(différence entre déformation totale et déformation thermique) :

    σ = D · (ε − ε_th) = D · (B u − α ΔT m)

et le vecteur de forces nodales équivalentes (terme de précontrainte
thermique passé au second membre K u = F + F_th) est :

    F_th = ∫_V Bᵀ · D · ε_th dV

Champ de température
--------------------
``ΔT`` peut être fourni de deux façons :

- **uniforme par élément** : un scalaire ⇒ ΔT constant dans l'élément ;
- **nodal** : un tableau de longueur ``n_nodes`` (valeurs aux nœuds de
  l'élément) ⇒ ΔT(ξ) = N(ξ) · ΔT_nodes interpolé via les fonctions de forme.

Les helpers ci-dessous factorisent ces deux conventions pour tous les
éléments continus (Tri3, Quad4, Tri6, Quad8, Tetra4, Hexa8, Tetra10, Hexa20).
"""

from __future__ import annotations

import numpy as np

# Vecteurs unitaires de dilatation thermique isotrope (notation de Voigt).
THERMAL_UNIT_2D: np.ndarray = np.array([1.0, 1.0, 0.0])
THERMAL_UNIT_3D: np.ndarray = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])


def thermal_unit_vector(n_strain: int) -> np.ndarray:
    """Vecteur unitaire de dilatation isotrope en notation de Voigt.

    Parameters
    ----------
    n_strain : int
        Nombre de composantes de déformation : 3 (état plan) ou 6 (3D).

    Returns
    -------
    np.ndarray, shape (n_strain,)
        ``[1, 1, 0]`` en 2D, ``[1, 1, 1, 0, 0, 0]`` en 3D.

    Raises
    ------
    ValueError
        Si ``n_strain`` n'est ni 3 ni 6.
    """
    if n_strain == 3:
        return THERMAL_UNIT_2D
    if n_strain == 6:
        return THERMAL_UNIT_3D
    raise ValueError(f"n_strain doit valoir 3 (2D) ou 6 (3D), reçu {n_strain}")


def normalize_delta_T(delta_T: float | np.ndarray, n_nodes: int) -> np.ndarray:
    """Normalise ΔT en un tableau, scalaire ou par nœud, prêt à interpoler.

    Parameters
    ----------
    delta_T : float or array_like
        Variation de température. Un scalaire (uniforme dans l'élément) ou
        un tableau de longueur ``n_nodes`` (valeurs nodales à interpoler).
    n_nodes : int
        Nombre de nœuds de l'élément (longueur attendue d'un champ nodal).

    Returns
    -------
    np.ndarray
        Tableau 0-d (scalaire) ou tableau 1-d de longueur ``n_nodes``.

    Raises
    ------
    ValueError
        Si ``delta_T`` est un tableau de longueur ≠ ``n_nodes``.
    """
    arr = np.asarray(delta_T, dtype=float)
    if arr.ndim == 0:
        return arr
    if arr.shape != (n_nodes,):
        raise ValueError(
            f"Champ ΔT nodal de longueur {arr.shape} incompatible avec "
            f"{n_nodes} nœuds (attendu shape ({n_nodes},))."
        )
    return arr


def delta_T_at(delta_T_norm: np.ndarray, N: np.ndarray) -> float:
    """Interpole ΔT à un point via les fonctions de forme.

    Parameters
    ----------
    delta_T_norm : np.ndarray
        Sortie de :func:`normalize_delta_T` (scalaire 0-d ou nodal 1-d).
    N : np.ndarray, shape (n_nodes,)
        Fonctions de forme évaluées au point d'intégration.

    Returns
    -------
    float
        ΔT au point. Pour un champ scalaire : la valeur constante. Pour un
        champ nodal : ``N · ΔT_nodes``.
    """
    if delta_T_norm.ndim == 0:
        return float(delta_T_norm)
    return float(N @ delta_T_norm)


def mean_delta_T(delta_T_norm: np.ndarray) -> float:
    """ΔT moyen — exact pour les éléments à déformation constante.

    Pour un élément à fonctions de forme linéaires (Tri3, Tetra4),
    ``∫_V ΔT dV = ΔT_moyen · V`` puisque ``∫_V N_i dV`` est identique pour
    chaque nœud. Le vecteur de forces thermiques ne dépend alors que de la
    moyenne des valeurs nodales.

    Parameters
    ----------
    delta_T_norm : np.ndarray
        Sortie de :func:`normalize_delta_T`.

    Returns
    -------
    float
        La valeur scalaire, ou la moyenne arithmétique des valeurs nodales.
    """
    if delta_T_norm.ndim == 0:
        return float(delta_T_norm)
    return float(delta_T_norm.mean())

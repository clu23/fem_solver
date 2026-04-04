"""Amortissement de Rayleigh — C = α·M + β·K.

Théorie
-------
L'amortissement de Rayleigh postule que la matrice d'amortissement C est une
combinaison linéaire de la matrice de masse M et de la matrice de rigidité K :

    C = α·M + β·K

**Propriété dans la base modale** (modes M-orthonormés φₙᵀ M φₘ = δₙₘ) :

    φₙᵀ C φₙ = α·(φₙᵀ M φₙ) + β·(φₙᵀ K φₙ)
             = α + β·ωₙ²
             = 2ξₙωₙ

Ce qui donne le **taux d'amortissement modal** :

    ξₙ = α/(2ωₙ) + β·ωₙ/2

- Le terme α (proportionnel à M) amortit surtout les **basses fréquences** :
  ξₙ ∝ 1/ωₙ → décroissant avec la fréquence.
- Le terme β (proportionnel à K) amortit surtout les **hautes fréquences** :
  ξₙ ∝ ωₙ → croissant avec la fréquence.

La courbe ξ(ω) est une **parabole en U** : elle minimise un taux d'amortissement
entre ω₁ et ω₂ si l'on calibre sur ces deux fréquences.

**Calibration à partir de deux modes**

On impose ξ(ω₁) = ξ₁ et ξ(ω₂) = ξ₂ :

    ┌ 1/(2ω₁)  ω₁/2 ┐ ┌α┐   ┌ξ₁┐
    └ 1/(2ω₂)  ω₂/2 ┘ └β┘ = └ξ₂┘

Résolution par la règle de Cramer :

    det = (ω₂² − ω₁²) / (4ω₁ω₂)

    α = 2ω₁ω₂(ξ₁ω₂ − ξ₂ω₁) / (ω₂² − ω₁²)
    β = 2(ξ₂ω₂ − ξ₁ω₁)     / (ω₂² − ω₁²)

Cas particulier ξ₁ = ξ₂ = ξ (taux uniforme sur toute la bande) :

    α = 2ξω₁ω₂ / (ω₁ + ω₂)
    β = 2ξ     / (ω₁ + ω₂)

**Propriétés de la matrice C**

- C est **symétrique** : héritée de la symétrie de M et K.
- C est **semi-définie positive** si α ≥ 0 et β ≥ 0 (cas physique).
- C est **creuse** avec exactement le même patron non nul que M + K.
- En pratique, C est assemblée à partir des matrices globales et non
  élément par élément.

Références
----------
Bathe K.J., «Finite Element Procedures», 2nd ed., §9.3.
Rayleigh J.W.S., «The Theory of Sound», Vol. 2, §§306–307.
Clough R.W. & Penzien J., «Dynamics of Structures», 3rd ed., §12.2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix


# ---------------------------------------------------------------------------
# Conteneur des coefficients
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RayleighDamping:
    """Coefficients d'amortissement de Rayleigh.

    Attributes
    ----------
    alpha : float
        Coefficient de la partie proportionnelle à M [1/s].
        Amortit surtout les basses fréquences.
    beta : float
        Coefficient de la partie proportionnelle à K [s].
        Amortit surtout les hautes fréquences.

    Notes
    -----
    Taux d'amortissement du mode n :

        ξₙ = alpha/(2·ωₙ) + beta·ωₙ/2

    Pour un amortissement purement proportionnel à M : beta = 0.
    Pour un amortissement purement proportionnel à K : alpha = 0.

    Examples
    --------
    Créer manuellement (α=5, β=0.001) :

    >>> d = RayleighDamping(alpha=5.0, beta=0.001)
    >>> d.modal_damping_ratio(np.array([100.0]))   # ξ(ω=100)
    array([0.075])

    Calibrer sur deux modes à 5% :

    >>> d = rayleigh_from_modes(omega1=10., omega2=100., zeta1=0.05, zeta2=0.05)
    """

    alpha: float
    beta: float

    def __post_init__(self) -> None:
        if self.alpha < 0.0:
            raise ValueError(f"alpha doit être ≥ 0, reçu {self.alpha}")
        if self.beta < 0.0:
            raise ValueError(f"beta doit être ≥ 0, reçu {self.beta}")

    def modal_damping_ratio(self, omega: np.ndarray | float) -> np.ndarray:
        """Taux d'amortissement modal ξₙ pour chaque pulsation ωₙ.

        Parameters
        ----------
        omega : np.ndarray or float
            Pulsations propres ω [rad/s] (strictement positives).

        Returns
        -------
        zeta : np.ndarray
            Taux d'amortissement ξ(ω) = α/(2ω) + β·ω/2 ∈ [0, ∞).

        Examples
        --------
        >>> d = RayleighDamping(alpha=2.0, beta=0.001)
        >>> d.modal_damping_ratio(np.array([10.0, 100.0]))
        array([0.105, 0.06 ])
        """
        omega = np.asarray(omega, dtype=float)
        return self.alpha / (2.0 * omega) + self.beta * omega / 2.0


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def rayleigh_from_modes(
    omega1: float,
    omega2: float,
    zeta1: float,
    zeta2: float,
) -> RayleighDamping:
    """Calibre les coefficients de Rayleigh à partir de deux taux modaux cibles.

    Résout le système linéaire 2×2 :

        ┌ 1/(2ω₁)  ω₁/2 ┐ ┌α┐   ┌ξ₁┐
        └ 1/(2ω₂)  ω₂/2 ┘ └β┘ = └ξ₂┘

    via la règle de Cramer.

    Parameters
    ----------
    omega1 : float
        Première pulsation de référence ω₁ [rad/s].  Doit être > 0.
    omega2 : float
        Deuxième pulsation de référence ω₂ [rad/s].  Doit être > ω₁.
    zeta1 : float
        Taux d'amortissement cible au mode 1 ξ₁ [-].  ∈ [0, 1).
    zeta2 : float
        Taux d'amortissement cible au mode 2 ξ₂ [-].  ∈ [0, 1).

    Returns
    -------
    RayleighDamping
        Coefficients (α, β) tels que ξ(ω₁) = ξ₁ et ξ(ω₂) = ξ₂.

    Raises
    ------
    ValueError
        Si ω₁ ≥ ω₂ ou si ω₁ ≤ 0.

    Notes
    -----
    Formule explicite :

        α = 2·ω₁·ω₂·(ξ₁·ω₂ − ξ₂·ω₁) / (ω₂² − ω₁²)
        β = 2·(ξ₂·ω₂ − ξ₁·ω₁)        / (ω₂² − ω₁²)

    Cas particulier ξ₁ = ξ₂ = ξ :

        α = 2ξ·ω₁·ω₂ / (ω₁ + ω₂)
        β = 2ξ        / (ω₁ + ω₂)

    Examples
    --------
    5% d'amortissement sur les modes à 10 et 100 rad/s :

    >>> d = rayleigh_from_modes(omega1=10., omega2=100., zeta1=0.05, zeta2=0.05)
    >>> d.modal_damping_ratio(np.array([10., 100.]))
    array([0.05, 0.05])

    Taux différents (2% sur mode 1, 5% sur mode 2) :

    >>> d = rayleigh_from_modes(10., 100., 0.02, 0.05)
    >>> round(d.modal_damping_ratio(np.array([10.]))[0], 6)
    0.02
    """
    if omega1 <= 0.0:
        raise ValueError(f"omega1 doit être > 0, reçu {omega1}")
    if omega2 <= omega1:
        raise ValueError(
            f"omega2 doit être > omega1 ({omega1}), reçu omega2={omega2}"
        )
    if not (0.0 <= zeta1 < 1.0):
        raise ValueError(f"zeta1 doit être dans [0, 1), reçu {zeta1}")
    if not (0.0 <= zeta2 < 1.0):
        raise ValueError(f"zeta2 doit être dans [0, 1), reçu {zeta2}")

    denom = omega2 ** 2 - omega1 ** 2   # > 0 car omega2 > omega1

    alpha = 2.0 * omega1 * omega2 * (zeta1 * omega2 - zeta2 * omega1) / denom
    beta  = 2.0 * (zeta2 * omega2 - zeta1 * omega1) / denom

    # α et β peuvent être négatifs pour des taux très différents.
    # On ne les force pas à 0 : l'utilisateur est responsable du sens physique.
    # La validation se fait dans RayleighDamping.__post_init__.
    if alpha < 0.0 or beta < 0.0:
        raise ValueError(
            f"La calibration produit α={alpha:.4g}, β={beta:.4g} < 0, "
            "ce qui n'est pas physique.  Vérifiez les taux ξ₁ et ξ₂ : "
            "des taux très différents peuvent rendre un coefficient négatif."
        )

    return RayleighDamping(alpha=float(alpha), beta=float(beta))


# ---------------------------------------------------------------------------
# Construction de la matrice C
# ---------------------------------------------------------------------------


def build_damping_matrix(
    damping: RayleighDamping,
    M: csr_matrix,
    K: csr_matrix,
) -> csr_matrix:
    """Assemble la matrice d'amortissement de Rayleigh C = α·M + β·K.

    Parameters
    ----------
    damping : RayleighDamping
        Coefficients α (mass-proportional) et β (stiffness-proportional).
    M : csr_matrix, shape (n, n)
        Matrice de masse globale (consistante ou condensée).
    K : csr_matrix, shape (n, n)
        Matrice de rigidité globale (avant application des CL, ou après
        si l'on veut une C cohérente avec K_bc).

    Returns
    -------
    C : csr_matrix, shape (n, n)
        Matrice d'amortissement symétrique, creuse.
        Pattern non nul = union des patterns de M et K.

    Notes
    -----
    C est construite à partir des matrices **globales** (pas des élémentaires).
    Ce choix garantit que C a exactement le même patron creux que M + K.
    La matrice C est ensuite réduite aux DDL libres dans le solveur harmonique
    (C_free = C[free, free]).

    Pour l'amortissement purement proportionnel à M (β=0), C a le même
    patron que M (plus creux que M + K).  Pour β=0, K n'est pas utilisé.

    Examples
    --------
    >>> damping = RayleighDamping(alpha=2.0, beta=0.001)
    >>> C = build_damping_matrix(damping, M, K)
    >>> C.shape == K.shape
    True

    Raises
    ------
    ValueError
        Si les formes de M et K sont incompatibles.
    """
    if M.shape != K.shape:
        raise ValueError(
            f"M.shape {M.shape} ≠ K.shape {K.shape} — matrices incompatibles."
        )

    C = damping.alpha * M + damping.beta * K
    return C.tocsr()

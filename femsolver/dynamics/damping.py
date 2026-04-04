"""Modèles d'amortissement indépendants de la fréquence.

Deux modèles complémentaires à l'amortissement de Rayleigh (rayleigh.py) :

1.  **Amortissement hystérétique** (structural / hysteretic damping)

    Z(Ω) = K(1 + iη) − Ω²M

    Le facteur de perte η est indépendant de la fréquence : la partie imaginaire
    de la rigidité est toujours proportionnelle à K, quelle que soit Ω.

    Propriété clé : à la résonance (K = ωₙ²M), Z = iηK, donc |H(ωₙ)| = 1/(ηK).
    Équivalence avec un taux visqueux : **η = 2ξ** à la résonance.

    Différence avec le visqueux à basse fréquence : même pour Ω → 0,
    Im(H) = −η/K ≠ 0, la phase est −arctan(η) (pas nulle comme pour C·Ω → 0).

2.  **Amortissement modal constant**

    Taux ξₙ imposé directement sur chaque mode — ni croissant ni décroissant
    avec ω.  Impossible à représenter exactement par Rayleigh si les ξₙ varient.

    En réponse harmonique : superposition modale tronquée
    (voir ``solve_harmonic_modal`` dans harmonic.py).

    En transitoire : reconstruction de C physique via
    C = M·Φ·diag(2ξₙωₙ)·Φᵀ·M (voir ``build_C_physical``).

Amplitude à la résonance (tous les modèles avec le même ξ)
-----------------------------------------------------------

    Rayleigh α pur   :  Z = i·αωM   = i·2ξK   →  |H(ωₙ)| = 1/(2ξk)
    Hystérétique η=2ξ:  Z = iηK     = i·2ξK   →  |H(ωₙ)| = 1/(2ξk)
    Modal ξₙ = ξ     :  Z_modal = 2iξωₙ²      →  |H(ωₙ)| = 1/(2ξk)

Références
----------
Ewins D.J., «Modal Testing: Theory, Practice and Application», 2nd ed., §1.3.2.
Nashif A.D. et al., «Vibration Damping», §2.3.
Clough R.W. & Penzien J., «Dynamics of Structures», 3rd ed., §14.2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_matrix

if TYPE_CHECKING:
    from femsolver.dynamics.modal import ModalResult


# ---------------------------------------------------------------------------
# 1. Amortissement hystérétique
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HystereticDamping:
    """Amortissement hystérétique (structural damping) — Z = K(1+iη) − Ω²M.

    L'impédance complexe est construite en remplaçant K par K(1+iη) :
    la partie imaginaire est proportionnelle à la rigidité, indépendante de Ω.

    Attributes
    ----------
    eta : float
        Facteur de perte structural η ≥ 0 [-].
        Règle d'équivalence avec un taux visqueux à la résonance : η = 2ξ.

    Notes
    -----
    À la résonance (Ω = ωₙ) :

        Z = K(1 + iη) − ωₙ²M = (K − ωₙ²M) + iηK = iηK

    car K = ωₙ²M dans la base modale.  Donc |H(ωₙ)| = F/(ηK) = 1/(ηK).

    À basse fréquence (Ω → 0) :

        Z → K(1 + iη)    →    H → 1/(K(1 + iη))

    Im(H(0)) = −η/(K(1+η²)) ≠ 0 : contrairement au modèle visqueux, la phase
    n'est PAS nulle à Ω = 0.  C'est la signature caractéristique de l'hystérétique.

    Examples
    --------
    >>> d = HystereticDamping(eta=0.10)
    >>> d.equivalent_zeta()
    0.05
    """

    eta: float

    def __post_init__(self) -> None:
        if self.eta < 0.0:
            raise ValueError(f"eta doit être ≥ 0, reçu {self.eta}")

    def equivalent_zeta(self) -> float:
        """Taux d'amortissement visqueux équivalent à la résonance : ξ = η/2.

        Returns
        -------
        zeta : float
            ξ = η/2  (amortissement équivalent à la résonance).
        """
        return self.eta / 2.0


# ---------------------------------------------------------------------------
# 2. Amortissement modal constant
# ---------------------------------------------------------------------------


@dataclass
class ModalDampingModel:
    """Amortissement modal constant — ξₙ imposé par mode.

    Chaque mode possède son propre taux d'amortissement, indépendamment
    de la relation ξ(ω) imposée par Rayleigh.

    Attributes
    ----------
    omega_n : np.ndarray, shape (n_modes,)
        Pulsations propres [rad/s] (strictement positives).
    zeta_n : np.ndarray, shape (n_modes,)
        Taux d'amortissement modal ξₙ ∈ [0, 1) par mode [-].
    phi : np.ndarray, shape (n_dof, n_modes)
        Vecteurs propres M-normalisés (φₙᵀ M φₙ = 1), taille **complète**
        (n_dof, pas n_free).  Les DDL contraints ont une valeur nulle exacte
        (convention de ``run_modal``).

    Notes
    -----
    En réponse harmonique : superposition modale tronquée.
    Si n_modes < n_dof, la contribution des modes non retenus est négligée
    (résidu modal non nul pour Ω loin des fréquences propres).

    En transitoire : utiliser ``build_C_physical(M)`` pour reconstruire C.

    Utiliser le constructeur classique ``from_modal_result`` plutôt que
    d'instancier directement (assure la cohérence phi/omega_n/n_modes).

    Examples
    --------
    >>> from femsolver.dynamics.modal import run_modal
    >>> result = run_modal(mesh, bc, n_modes=5)
    >>> d = ModalDampingModel.from_modal_result(result, zeta_n=0.02)
    >>> d.omega_n.shape
    (5,)
    """

    omega_n: np.ndarray    # (n_modes,) [rad/s]
    zeta_n: np.ndarray     # (n_modes,) [-]
    phi: np.ndarray        # (n_dof, n_modes), M-normalisés, taille complète

    def __post_init__(self) -> None:
        self.omega_n = np.asarray(self.omega_n, dtype=float)
        self.zeta_n  = np.asarray(self.zeta_n,  dtype=float)
        self.phi     = np.asarray(self.phi,      dtype=float)

        n_modes = len(self.omega_n)
        if len(self.zeta_n) != n_modes:
            raise ValueError(
                f"omega_n ({n_modes}) et zeta_n ({len(self.zeta_n)}) "
                "doivent avoir la même longueur."
            )
        if self.phi.ndim != 2 or self.phi.shape[1] != n_modes:
            raise ValueError(
                f"phi.shape[1]={self.phi.shape[1]} ≠ n_modes={n_modes} "
                "(phi doit être de forme (n_dof, n_modes))."
            )
        if np.any(self.omega_n <= 0.0):
            raise ValueError("Toutes les pulsations propres doivent être > 0.")
        if np.any(self.zeta_n < 0.0):
            raise ValueError("Tous les taux d'amortissement doivent être ≥ 0.")

    @classmethod
    def from_modal_result(
        cls,
        result: "ModalResult",
        zeta_n: np.ndarray | float,
    ) -> "ModalDampingModel":
        """Construit le modèle depuis un ``ModalResult``.

        Parameters
        ----------
        result : ModalResult
            Résultats d'une analyse modale (``run_modal``).
        zeta_n : float or np.ndarray, shape (n_modes,)
            Taux d'amortissement par mode.  Un scalaire applique le même taux
            à tous les modes.

        Returns
        -------
        ModalDampingModel
            Modèle cohérent avec ``result.omega``, ``result.modes``.

        Examples
        --------
        5% sur tous les modes :

        >>> d = ModalDampingModel.from_modal_result(result, zeta_n=0.05)

        Taux croissants : 2%, 3%, 4% sur les 3 premiers modes :

        >>> d = ModalDampingModel.from_modal_result(result, zeta_n=[0.02, 0.03, 0.04])
        """
        zeta_arr = np.broadcast_to(
            np.asarray(zeta_n, dtype=float), (result.n_modes,)
        ).copy()
        return cls(
            omega_n=result.omega.copy(),
            zeta_n=zeta_arr,
            phi=result.modes.copy(),
        )

    def build_C_physical(self, M: csr_matrix) -> csr_matrix:
        """Reconstruit la matrice d'amortissement physique C = MΦ diag(2ξₙωₙ) ΦᵀM.

        Utilisé pour le transitoire (Newmark) où C doit être une matrice physique.

        Parameters
        ----------
        M : csr_matrix, shape (n_dof, n_dof)
            Matrice de masse globale (taille complète, cohérente avec phi).

        Returns
        -------
        C : csr_matrix, shape (n_dof, n_dof)
            Matrice d'amortissement physique, symétrique, semi-définie positive.

        Notes
        -----
        Formule :

            C = Σₙ 2ξₙωₙ · M·φₙ·φₙᵀ·M = M·Φ·diag(γ)·Φᵀ·M

        où γₙ = 2ξₙωₙ.

        Vérification modale : φₘᵀ·C·φₙ = 2ξₙωₙ·δₘₙ (diagonale dans la base modale).

        Si n_modes < n_dof, C est de rang n_modes (reconstruction tronquée).
        Pour le transitoire, prendre suffisamment de modes pour couvrir la bande
        de fréquences d'excitation.

        Raises
        ------
        ValueError
            Si M.shape[0] ≠ phi.shape[0].
        """
        n = self.phi.shape[0]
        if M.shape != (n, n):
            raise ValueError(
                f"M.shape {M.shape} incompatible avec phi.shape[0]={n}."
            )
        M_arr  = M.toarray()
        MPhi   = M_arr @ self.phi              # (n, n_modes)
        gamma  = 2.0 * self.zeta_n * self.omega_n   # (n_modes,)
        C_arr  = (MPhi * gamma) @ MPhi.T       # (n, n)
        return csr_matrix(C_arr)

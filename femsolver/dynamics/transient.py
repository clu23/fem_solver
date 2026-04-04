"""Intégration temporelle Newmark-β — schéma implicite/explicite.

Théorie
-------

L'équation du mouvement discrétisée :

    M·ü(t) + C·u̇(t) + K·u(t) = F(t)

Le schéma de Newmark prédit l'état au pas n+1 depuis l'état n via deux
paramètres γ et β qui contrôlent la précision et la stabilité.

**Prédicteur (explicite)** :

    ũ_{n+1} = u_n + Δt·v_n + Δt²·(½ − β)·a_n
    ṽ_{n+1} = v_n + Δt·(1 − γ)·a_n

**Système effectif (implicite pour β > 0)** :

    K_eff · u_{n+1} = F_eff

    K_eff = K + a₁·M + a₂·C        a₁ = 1/(β·Δt²)  a₂ = γ/(β·Δt)
    F_eff = F_{n+1} + a₁·M·ũ_{n+1} + a₂·C·ũ_{n+1} − C·ṽ_{n+1}

K_eff est **constant** si Δt est constant → une seule factorisation LU,
réutilisée à chaque pas (via ``scipy.sparse.linalg.factorized``).

**Correcteur** :

    a_{n+1} = (u_{n+1} − ũ_{n+1}) / (β·Δt²)
    v_{n+1} = ṽ_{n+1} + γ·Δt·a_{n+1}

**Cas β = 0 (explicite — différences centrales)** :

    u_{n+1} = ũ_{n+1}                    (pas de correction déplacement)
    (M + γ·Δt·C) · a_{n+1} = F_{n+1} − K·u_{n+1} − C·ṽ_{n+1}
    v_{n+1} = ṽ_{n+1} + γ·Δt·a_{n+1}

Stabilité
---------

Pour γ = ½, la stabilité inconditionnelle requiert β ≥ ¼.
Pour β < ¼ (conditionnel), le pas de temps critique est :

    Δt_crit = Ω_crit / ωₙ_max

    β = 0         :  Ω_crit = 2.000  (central differences, CFL)
    β = 1/12      :  Ω_crit ≈ 2.449  (Fox-Goodwin)
    β = 1/6       :  Ω_crit ≈ 3.464  (accélération linéaire)
    β = 1/4       :  Ω_crit = ∞      (trapèze, inconditionnellement stable)

Précision temporelle : O(Δt²) pour γ = ½ (ordre 2) ; O(Δt) pour γ ≠ ½.
Dissipation numérique : aucune pour γ = ½ ; amorti artificiellement pour γ > ½.

Résumé des choix courants
--------------------------

- **Trapèze** (γ = ½, β = ¼) : recommandé.  Inconditionnellement stable,
  aucune dissipation numérique, ordre 2.  Équivalent à la règle de Simpson.
- **HHT-α** / **Generalized-α** : γ > ½ avec correction, pour contrôler la
  dissipation haute-fréquence sans perte d'ordre (non implémenté ici).
- **Différences centrales** (γ = ½, β = 0) : explicite, bon marché par pas,
  mais Δt ≤ 2/ωₙ_max.  Attractif si M est diagonale et le système est grand.

Références
----------
Newmark N.M., «A Method of Computation for Structural Dynamics», ASCE, 1959.
Hughes T.J.R., «The Finite Element Method», §9.2–9.3, Prentice-Hall, 1987.
Bathe K.J., «Finite Element Procedures», §9.4, 2nd ed., 2014.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import factorized as sp_factorized
from scipy.sparse.linalg import spsolve

from femsolver.core.assembler import Assembler
from femsolver.core.boundary import apply_dirichlet
from femsolver.core.mesh import BoundaryConditions, Mesh
from femsolver.dynamics.damping import HystereticDamping, ModalDampingModel
from femsolver.dynamics.rayleigh import RayleighDamping, build_damping_matrix


# ---------------------------------------------------------------------------
# Paramètres du schéma
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NewmarkBeta:
    """Paramètres γ et β du schéma de Newmark.

    Attributes
    ----------
    gamma : float
        Paramètre γ ∈ [0, 1].  γ = ½ : ordre 2, aucune dissipation numérique.
        γ > ½ : dissipation artificielle des hautes fréquences (1er ordre).
    beta : float
        Paramètre β ∈ [0, ½].  Contrôle l'interpolation d'accélération.
        β = 0 : explicite (pas de correction déplacement).
        β = ¼ : trapèze (average acceleration), inconditionnellement stable.

    Notes
    -----
    Condition d'inconditionnalité (Newmark, 1959) :

        γ ≥ ½  et  β ≥ (γ + ½)² / 4

    Pour γ = ½ : condition simplifiée β ≥ ¼.

    Examples
    --------
    >>> NewmarkBeta()              # trapèze (défaut)
    NewmarkBeta(gamma=0.5, beta=0.25)
    >>> NewmarkBeta(gamma=0.5, beta=0.0)   # central differences
    NewmarkBeta(gamma=0.5, beta=0.0)
    """

    gamma: float = 0.5
    beta: float  = 0.25

    def __post_init__(self) -> None:
        if not (0.0 <= self.gamma <= 1.0):
            raise ValueError(f"gamma doit être dans [0,1], reçu {self.gamma}")
        if not (0.0 <= self.beta <= 0.5):
            raise ValueError(f"beta doit être dans [0,0.5], reçu {self.beta}")

    def is_unconditionally_stable(self) -> bool:
        """True si le schéma est inconditionnellement stable.

        Condition : γ ≥ ½ ET β ≥ (γ + ½)²/4.

        Returns
        -------
        bool
        """
        return self.gamma >= 0.5 and self.beta >= (self.gamma + 0.5) ** 2 / 4.0

    def critical_dt(self, omega_max: float) -> float | None:
        """Pas de temps critique Δt_crit = Ω_crit / ω_max.

        Parameters
        ----------
        omega_max : float
            Pulsation propre maximale du système [rad/s].

        Returns
        -------
        float or None
            Δt_crit en secondes.  None si inconditionnellement stable.

        Raises
        ------
        ValueError
            Si omega_max ≤ 0.
        NotImplementedError
            Si γ ≠ ½ (formule non implémentée pour γ ≠ ½).
        """
        if omega_max <= 0.0:
            raise ValueError(f"omega_max doit être > 0, reçu {omega_max}")
        if self.is_unconditionally_stable():
            return None
        if abs(self.gamma - 0.5) > 1e-14:
            raise NotImplementedError(
                "critical_dt non implémenté pour γ ≠ ½.  "
                "Utiliser γ = ½ avec β ∈ [0, ¼)."
            )
        # Pour γ = ½, 0 ≤ β < ¼ :  Ω_crit = 2 / √(1 − 4β)
        denom = math.sqrt(1.0 - 4.0 * self.beta)
        return 2.0 / (omega_max * denom)


# Schémas nommés prédéfinis
NEWMARK_TRAPEZOIDAL     = NewmarkBeta(gamma=0.5, beta=0.25)   # trapèze (défaut)
NEWMARK_CENTRAL_DIFF    = NewmarkBeta(gamma=0.5, beta=0.0)    # central differences
NEWMARK_FOX_GOODWIN     = NewmarkBeta(gamma=0.5, beta=1/12)   # Fox-Goodwin
NEWMARK_LINEAR_ACCEL    = NewmarkBeta(gamma=0.5, beta=1/6)    # accél. linéaire


# ---------------------------------------------------------------------------
# Conteneur des résultats
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransientResult:
    """Résultats d'une intégration temporelle Newmark-β.

    Attributes
    ----------
    t : np.ndarray, shape (n_steps+1,)
        Instants [s] : t[0] = 0, t[-1] = n_steps·Δt.
    u : np.ndarray, shape (n_dof, n_steps+1)
        Déplacements nodaux.  DDL contraints = 0.
    v : np.ndarray, shape (n_dof, n_steps+1)
        Vitesses nodales.
    a : np.ndarray, shape (n_dof, n_steps+1)
        Accélérations nodales.
    free_dofs : np.ndarray, shape (n_free,)
        Indices des DDL libres.
    params : NewmarkBeta
        Paramètres γ, β utilisés.

    Examples
    --------
    >>> result = run_transient(mesh, bc, F, u0, v0, dt=1e-3, n_steps=1000)
    >>> tip = result.u[-3, :]        # DDL uy du dernier nœud
    >>> plt.plot(result.t, tip)
    """

    t: np.ndarray
    u: np.ndarray
    v: np.ndarray
    a: np.ndarray
    free_dofs: np.ndarray
    params: NewmarkBeta


# ---------------------------------------------------------------------------
# Solveur bas niveau (DDL libres)
# ---------------------------------------------------------------------------


def solve_newmark(
    K_free: csr_matrix,
    M_free: csr_matrix,
    C_free: csr_matrix,
    F_free_fn: Callable[[float], np.ndarray] | None,
    u0: np.ndarray,
    v0: np.ndarray,
    dt: float,
    n_steps: int,
    params: NewmarkBeta = NewmarkBeta(),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Intégration Newmark-β sur les DDL libres.

    Interface bas niveau : travaille sur des matrices déjà réduites.
    Pour l'interface maillage complète, utiliser :func:`run_transient`.

    Parameters
    ----------
    K_free : csr_matrix, shape (n_free, n_free)
        Matrice de rigidité réduite.
    M_free : csr_matrix, shape (n_free, n_free)
        Matrice de masse réduite (définie positive).
    C_free : csr_matrix, shape (n_free, n_free)
        Matrice d'amortissement réduite.  Peut être nulle (matrice zéro).
    F_free_fn : callable or None
        Force en fonction du temps : ``F_free_fn(t)`` → ndarray(n_free,).
        Si ``None``, F = 0 (vibration libre).
    u0 : np.ndarray, shape (n_free,)
        Déplacement initial.
    v0 : np.ndarray, shape (n_free,)
        Vitesse initiale.
    dt : float
        Pas de temps [s].  Doit être ≤ Δt_crit pour β = 0.
    n_steps : int
        Nombre de pas de temps.
    params : NewmarkBeta
        Paramètres γ et β.  Défaut : trapèze (γ=½, β=¼).

    Returns
    -------
    u_hist : np.ndarray, shape (n_free, n_steps+1)
        Déplacements à chaque pas.
    v_hist : np.ndarray, shape (n_free, n_steps+1)
        Vitesses à chaque pas.
    a_hist : np.ndarray, shape (n_free, n_steps+1)
        Accélérations à chaque pas.

    Notes
    -----
    **Accélération initiale** : calculée depuis l'équilibre dynamique
    au pas 0 : ``M·a₀ = F(0) − C·v₀ − K·u₀``.

    **Efficacité** : pour β > 0 (implicite), K_eff est factorisée LU **une
    seule fois** (``scipy.sparse.linalg.factorized``) puis réutilisée à
    chaque pas.  Complexité O(n²) par pas après la factorisation O(n³).

    **Cas β = 0 (explicite)** : K_eff n'est pas utilisée.  On résout
    ``(M + γ·Δt·C)·a_{n+1} = rhs`` à chaque pas.  Même stratégie de
    factorisation pour (M + γ·Δt·C) si C ≠ 0.
    """
    gamma = params.gamma
    beta  = params.beta
    n_free = K_free.shape[0]

    # Force nulle si non fournie
    if F_free_fn is None:
        _F = lambda t: np.zeros(n_free)
    else:
        _F = F_free_fn

    # Allocation des historiques
    u_hist = np.zeros((n_free, n_steps + 1))
    v_hist = np.zeros((n_free, n_steps + 1))
    a_hist = np.zeros((n_free, n_steps + 1))

    u_hist[:, 0] = u0
    v_hist[:, 0] = v0

    # Accélération initiale : M·a₀ = F(0) − C·v₀ − K·u₀
    rhs0 = _F(0.0) - C_free @ v0 - K_free @ u0
    a_hist[:, 0] = spsolve(M_free, rhs0)

    if beta > 0.0:
        # ------------------------------------------------------------------
        # Schéma implicite (β > 0)
        # ------------------------------------------------------------------
        a1 = 1.0 / (beta * dt * dt)    # 1/(β·Δt²)
        a2 = gamma / (beta * dt)        # γ/(β·Δt)

        K_eff = (K_free + a1 * M_free + a2 * C_free).tocsc()
        solve_K_eff = sp_factorized(K_eff)   # factorisation LU une seule fois

        # Partie de K_eff due à M et C (sans K) : a₁M + a₂C
        K_inert = (a1 * M_free + a2 * C_free)

        for n in range(n_steps):
            t_n1 = (n + 1) * dt
            u_n  = u_hist[:, n]
            v_n  = v_hist[:, n]
            a_n  = a_hist[:, n]

            # Prédicteurs
            u_pred = u_n + dt * v_n + dt * dt * (0.5 - beta) * a_n
            v_pred = v_n + dt * (1.0 - gamma) * a_n

            # Second membre effectif
            F_eff = _F(t_n1) + K_inert @ u_pred - C_free @ v_pred

            # Résolution implicite (substitution LU, O(n²))
            u_n1 = solve_K_eff(F_eff)

            # Correcteurs
            a_n1 = (u_n1 - u_pred) * a1   # = (u_{n+1} - ũ)/(β·Δt²)
            v_n1 = v_pred + gamma * dt * a_n1

            u_hist[:, n + 1] = u_n1
            v_hist[:, n + 1] = v_n1
            a_hist[:, n + 1] = a_n1

    else:
        # ------------------------------------------------------------------
        # Schéma explicite β = 0 (différences centrales)
        # ------------------------------------------------------------------
        # Système à factoriser : M_eff = M + γ·Δt·C (constant)
        M_eff = (M_free + gamma * dt * C_free).tocsc()
        solve_M_eff = sp_factorized(M_eff)

        for n in range(n_steps):
            t_n1 = (n + 1) * dt
            u_n  = u_hist[:, n]
            v_n  = v_hist[:, n]
            a_n  = a_hist[:, n]

            # Prédicteurs (β=0 : (½−0) = ½ pour u_pred)
            u_pred = u_n + dt * v_n + 0.5 * dt * dt * a_n
            v_pred = v_n + dt * (1.0 - gamma) * a_n

            # Déplacement explicite (pas de correction pour β=0)
            u_n1 = u_pred

            # Résolution pour l'accélération
            rhs  = _F(t_n1) - K_free @ u_n1 - C_free @ v_pred
            a_n1 = solve_M_eff(rhs)

            # Correction de la vitesse
            v_n1 = v_pred + gamma * dt * a_n1

            u_hist[:, n + 1] = u_n1
            v_hist[:, n + 1] = v_n1
            a_hist[:, n + 1] = a_n1

    return u_hist, v_hist, a_hist


# ---------------------------------------------------------------------------
# Interface de haut niveau (maillage)
# ---------------------------------------------------------------------------


def run_transient(
    mesh: Mesh,
    bc: BoundaryConditions,
    F_hat: np.ndarray | Callable[[float], np.ndarray] | None,
    u0_full: np.ndarray,
    v0_full: np.ndarray,
    dt: float,
    n_steps: int,
    damping: RayleighDamping | ModalDampingModel | None = None,
    params: NewmarkBeta = NewmarkBeta(),
) -> TransientResult:
    """Intégration temporelle Newmark-β complète sur un maillage.

    Assemble K et M, construit C, réduit aux DDL libres, puis intègre
    l'équation du mouvement :

        M·ü + C·u̇ + K·u = F(t)

    Parameters
    ----------
    mesh : Mesh
        Maillage du modèle.
    bc : BoundaryConditions
        Conditions aux limites de Dirichlet (imposées par élimination vraie).
    F_hat : np.ndarray, callable, or None
        Force externe :

        - ``None`` : vibration libre F = 0.
        - ``ndarray`` shape ``(n_dof,)`` : force constante.
        - ``callable`` : ``F_hat(t: float) → ndarray(n_dof,)`` force variable.

    u0_full : np.ndarray, shape (n_dof,)
        Condition initiale déplacement (taille complète).
    v0_full : np.ndarray, shape (n_dof,)
        Condition initiale vitesse (taille complète).
    dt : float
        Pas de temps [s].
    n_steps : int
        Nombre de pas de temps.
    damping : RayleighDamping, ModalDampingModel, or None
        Modèle d'amortissement :

        - ``None`` : C = 0.
        - ``RayleighDamping`` : C = α·M + β·K.
        - ``ModalDampingModel`` : C = M·Φ·diag(2ξₙωₙ)·Φᵀ·M.
        - ``HystereticDamping`` : **non supporté** en transitoire (défini
          uniquement pour les régimes harmoniques permanents).

    params : NewmarkBeta
        Paramètres γ et β.  Défaut : trapèze (γ=½, β=¼).

    Returns
    -------
    TransientResult
        Contient ``t``, ``u``, ``v``, ``a`` (taille n_dof × n_steps+1),
        ``free_dofs`` et ``params``.

    Raises
    ------
    TypeError
        Si ``damping`` est un ``HystereticDamping`` (incompatible avec le
        transitoire) ou un type non reconnu.

    Notes
    -----
    Les DDL contraints (Dirichlet) restent à zéro dans ``u``, ``v``, ``a``.
    Les conditions initiales ``u0_full[constrained]`` sont ignorées.

    Examples
    --------
    Vibration libre amortie d'une console :

    >>> u0 = np.zeros(mesh.n_dof); u0[tip_dof] = 0.01  # 1 cm
    >>> v0 = np.zeros(mesh.n_dof)
    >>> damping = RayleighDamping(alpha=2.0, beta=0.0)
    >>> result = run_transient(mesh, bc, None, u0, v0, dt=1e-4, n_steps=5000,
    ...                        damping=damping)
    """
    if isinstance(damping, HystereticDamping):
        raise TypeError(
            "HystereticDamping n'est pas supporté en analyse transitoire : "
            "la rigidité complexe K(1+iη) est définie uniquement pour le régime "
            "harmonique permanent.  Utiliser RayleighDamping ou ModalDampingModel."
        )
    if damping is not None and not isinstance(damping, (RayleighDamping, ModalDampingModel)):
        raise TypeError(
            f"Type d'amortissement non supporté : {type(damping).__name__}.  "
            "Utiliser RayleighDamping, ModalDampingModel ou None."
        )

    # Assemblage
    assembler = Assembler(mesh)
    K = assembler.assemble_stiffness()
    M = assembler.assemble_mass()

    # Conditions aux limites — élimination vraie
    F_dummy = np.zeros(mesh.n_dof)
    ds   = apply_dirichlet(K, F_dummy, mesh, bc)
    free = ds.free_dofs

    K_free = ds.K_free
    M_free = ds.reduce_mass(M)

    # Matrice d'amortissement
    if damping is None:
        n_free = len(free)
        C_free = csr_matrix((n_free, n_free))
    elif isinstance(damping, RayleighDamping):
        C      = build_damping_matrix(damping, M, K)
        C_free = C[free, :][:, free].tocsr()
    else:  # ModalDampingModel
        C      = damping.build_C_physical(M)
        C_free = C[free, :][:, free].tocsr()

    # Conditions initiales réduites
    u0_free = u0_full[free]
    v0_free = v0_full[free]

    # Encapsulation de la force dans un callable (n_free,)
    if F_hat is None:
        F_free_fn: Callable[[float], np.ndarray] | None = None
    elif callable(F_hat):
        F_free_fn = lambda t, _f=F_hat: _f(t)[free]
    else:
        F_arr = np.asarray(F_hat, dtype=float)
        F_free_const = F_arr[free]
        F_free_fn = lambda t: F_free_const

    # Intégration temporelle
    u_free, v_free, a_free = solve_newmark(
        K_free, M_free, C_free, F_free_fn,
        u0_free, v0_free, dt, n_steps, params,
    )

    # Reconstruction taille complète (DDL contraints = 0)
    n_dof  = mesh.n_dof
    n_time = n_steps + 1
    u_full = np.zeros((n_dof, n_time))
    v_full = np.zeros((n_dof, n_time))
    a_full = np.zeros((n_dof, n_time))

    u_full[free, :] = u_free
    v_full[free, :] = v_free
    a_full[free, :] = a_free

    t_arr = np.arange(n_steps + 1, dtype=float) * dt

    return TransientResult(
        t=t_arr,
        u=u_full,
        v=v_full,
        a=a_full,
        free_dofs=free,
        params=params,
    )

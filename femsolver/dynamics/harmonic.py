"""Réponse harmonique en fréquence — balayage (frequency sweep).

Théorie
-------
Pour une excitation harmonique en régime permanent :

    F(t) = Re[F̂ · e^{iΩt}]

l'équation du mouvement amorti :

    M·ü + C·u̇ + K·u = F(t)

admet une solution de la forme u(t) = Re[U(Ω) · e^{iΩt}] où :

    (K − Ω²·M + i·Ω·C) · U(Ω) = F̂
     ╰─────────────────────────╯
          matrice dynamique Z(Ω)     [complexe, n_free × n_free]

La **Fonction de Réponse en Fréquence (FRF)** est :

    H(Ω) = Z(Ω)⁻¹ = (K − Ω²M + iΩC)⁻¹

Le **balayage en fréquence** résout Z(Ω)·U = F̂ pour chaque Ω = 2πf.

Comportement physique
---------------------

- À **basse fréquence** (Ω → 0) : Z → K, donc U → K⁻¹F̂ (statique).
- À **haute fréquence** (Ω → ∞) : Z ≈ −Ω²M, donc |U| → 0 (inertie domine).
- Aux **résonances** (Ω ≈ ωₙ) : |U(ωₙ)| est maximisé ≈ 1/(2ξₙ·kₙ).
  Sans amortissement ξₙ = 0 → singularité de Z → amplitude infinie.
- **Phase** : à la résonance, le déplacement est en quadrature retard de
  90° par rapport à l'excitation ; en dessous : phase ≈ 0° ; au-dessus : ≈ 180°.

Conditions aux limites
-----------------------
On applique les CL de Dirichlet par **élimination vraie** :

1. ``apply_dirichlet`` → ``DirichletSystem`` ds avec ``free_dofs``, ``K_free``.
2. ``C_free = C[free, free]``,  ``M_free = M[free, free]``.
3. Pour chaque Ω :  ``Z_free = K_free − Ω²·M_free + iΩ·C_free``
4. Résoudre ``Z_free · U_free = F̂[free_dofs]``  (scipy spsolve, complexe)
5. Reconstruire ``U[free_dofs] = U_free``, ``U[constrained] = 0``

Cette stratégie évite de re-former Z de taille n_dof × n_dof et d'y
appliquer les CL à chaque fréquence.

Notes sur la résolution complexe
---------------------------------
scipy.sparse.linalg.spsolve accepte les matrices complexes (CSR avec dtype
complex128).  La matrice dynamique Z est formée à chaque fréquence :

    Z = K_free.astype(complex) - Omega**2 * M_free + 1j * Omega * C_free

Pour un grand nombre de fréquences avec les mêmes CL, une factorisation LU
de Z à chaque fréquence est inévitable (Z change avec Ω).  Pour les systèmes
de petite à moyenne taille (< 10 000 DDL), spsolve est suffisant.

Références
----------
Ewins D.J., «Modal Testing: Theory, Practice and Application», 2nd ed.
Clough R.W. & Penzien J., «Dynamics of Structures», 3rd ed., §12.3.
Cook R.D. et al., «Concepts and Applications of FEA», 4th ed., §17.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from femsolver.core.assembler import Assembler
from femsolver.core.boundary import apply_dirichlet
from femsolver.core.mesh import BoundaryConditions, Mesh
from femsolver.dynamics.damping import HystereticDamping, ModalDampingModel
from femsolver.dynamics.rayleigh import RayleighDamping, build_damping_matrix


# ---------------------------------------------------------------------------
# Conteneur des résultats
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HarmonicResult:
    """Résultats d'un balayage en fréquence.

    Attributes
    ----------
    freqs : np.ndarray, shape (n_freqs,)
        Fréquences d'excitation [Hz].
    U : np.ndarray, shape (n_dof, n_freqs), dtype complex128
        Déplacements complexes nodaux.  ``U[dof, k]`` est le déplacement
        au DDL ``dof`` pour la fréquence ``freqs[k]``.
        Les DDL contraints (Dirichlet = 0) ont U = 0.
    free_dofs : np.ndarray, shape (n_free,)
        Indices des DDL libres utilisés pour la résolution.

    Notes
    -----
    Pour obtenir l'amplitude (FRF) et la phase :

    .. code-block:: python

        amplitude = np.abs(result.U)      # shape (n_dof, n_freqs)
        phase_deg = np.angle(result.U, deg=True)

    Pour extraire la FRF d'un DDL observé (p.ex. DDL 5) :

    .. code-block:: python

        H5 = result.U[5, :]              # (n_freqs,)
    """

    freqs: np.ndarray      # [Hz]
    U: np.ndarray          # complex, (n_dof, n_freqs)
    free_dofs: np.ndarray  # indices des DDL libres


# ---------------------------------------------------------------------------
# Solveur harmonique de bas niveau (matrices déjà réduites)
# ---------------------------------------------------------------------------


def solve_harmonic(
    K_free: csr_matrix,
    M_free: csr_matrix,
    C_free: csr_matrix,
    F_free: np.ndarray,
    freqs: np.ndarray,
) -> np.ndarray:
    """Résoudre Z(Ω)·U_free = F_free pour chaque fréquence.

    Travaille sur des matrices déjà réduites aux DDL libres.
    Interface bas niveau utilisée par ``run_harmonic`` et par les tests.

    Parameters
    ----------
    K_free : csr_matrix, shape (n_free, n_free)
        Matrice de rigidité réduite.
    M_free : csr_matrix, shape (n_free, n_free)
        Matrice de masse réduite.
    C_free : csr_matrix, shape (n_free, n_free)
        Matrice d'amortissement réduite.  Peut être nulle (matrice zéro)
        pour un calcul sans amortissement (cas singulier aux résonances).
    F_free : np.ndarray, shape (n_free,)
        Amplitude de force aux DDL libres (réel).
    freqs : np.ndarray, shape (n_freqs,)
        Fréquences d'excitation [Hz].

    Returns
    -------
    U_mat : np.ndarray, shape (n_free, n_freqs), dtype complex128
        Déplacements complexes aux DDL libres pour chaque fréquence.

    Notes
    -----
    La matrice dynamique est formée pour chaque fréquence :

        Z(Ω) = K_free − Ω²·M_free + iΩ·C_free   (complexe)

    La résolution utilise ``scipy.sparse.linalg.spsolve`` qui supporte
    les matrices CSR complexes (dtype complex128).
    """
    n_free = K_free.shape[0]
    n_freqs = len(freqs)
    U_mat = np.zeros((n_free, n_freqs), dtype=complex)

    # Pré-convertir en complexe pour éviter la recopie à chaque fréquence
    K_c = K_free.astype(complex)
    M_c = M_free.astype(complex)
    C_c = C_free.astype(complex)
    F_c = F_free.astype(complex)

    for k, f in enumerate(freqs):
        Omega = 2.0 * np.pi * f
        Z = K_c - (Omega ** 2) * M_c + (1j * Omega) * C_c
        U_mat[:, k] = spsolve(Z.tocsr(), F_c)

    return U_mat


# ---------------------------------------------------------------------------
# Solveur bas niveau — amortissement hystérétique
# ---------------------------------------------------------------------------


def solve_harmonic_hysteretic(
    K_free: csr_matrix,
    M_free: csr_matrix,
    eta: float,
    F_free: np.ndarray,
    freqs: np.ndarray,
) -> np.ndarray:
    """Résoudre Z(Ω)·U = F̂ pour un amortissement hystérétique (structural).

    Z(Ω) = K(1 + iη) − Ω²M

    La matrice dynamique est construite une fois (K·(1+iη)) puis le terme
    inertiel −Ω²M est soustrait à chaque fréquence.

    Parameters
    ----------
    K_free : csr_matrix, shape (n_free, n_free)
        Matrice de rigidité réduite.
    M_free : csr_matrix, shape (n_free, n_free)
        Matrice de masse réduite.
    eta : float
        Facteur de perte structural η ≥ 0.  Pour η = 0 : calcul sans amortissement.
    F_free : np.ndarray, shape (n_free,)
        Amplitude de force aux DDL libres (réel).
    freqs : np.ndarray, shape (n_freqs,)
        Fréquences d'excitation [Hz].

    Returns
    -------
    U_mat : np.ndarray, shape (n_free, n_freqs), dtype complex128
        Déplacements complexes aux DDL libres.

    Notes
    -----
    Contrairement au visqueux, K·iη ne dépend pas de Ω : la partie complexe
    de Z est constante.  Seul −Ω²M change avec la fréquence.

    À la résonance (K = ωₙ²M dans la base modale) :

        Z = iηK    →    |H(ωₙ)| = 1 / (ηK)

    Équivalence avec ξ = η/2 au sens de l'amplitude à la résonance.
    """
    n_free = K_free.shape[0]
    n_freqs = len(freqs)
    U_mat = np.zeros((n_free, n_freqs), dtype=complex)

    # K(1+iη) est constant — calculé une seule fois
    K_c = K_free.astype(complex)
    M_c = M_free.astype(complex)
    K_complex = K_c * (1.0 + 1j * eta)   # K(1 + iη)
    F_c = F_free.astype(complex)

    for k, f in enumerate(freqs):
        Omega = 2.0 * np.pi * f
        Z = K_complex - (Omega ** 2) * M_c
        U_mat[:, k] = spsolve(Z.tocsr(), F_c)

    return U_mat


# ---------------------------------------------------------------------------
# Solveur bas niveau — superposition modale
# ---------------------------------------------------------------------------


def solve_harmonic_modal(
    omega_n: np.ndarray,
    zeta_n: np.ndarray,
    phi_free: np.ndarray,
    F_free: np.ndarray,
    freqs: np.ndarray,
) -> np.ndarray:
    """Superposition modale pour amortissement modal constant par mode.

    U(Ω) = Σₙ φₙ · Γₙ / (ωₙ² − Ω² + 2iξₙωₙΩ)

    où Γₙ = φₙᵀ·F̂ est le facteur de participation modale.

    Parameters
    ----------
    omega_n : np.ndarray, shape (n_modes,)
        Pulsations propres [rad/s].
    zeta_n : np.ndarray, shape (n_modes,)
        Taux d'amortissement par mode [-].
    phi_free : np.ndarray, shape (n_free, n_modes)
        Vecteurs propres M-normalisés, réduits aux DDL libres.
    F_free : np.ndarray, shape (n_free,)
        Amplitude de force aux DDL libres [N].
    freqs : np.ndarray, shape (n_freqs,)
        Fréquences d'excitation [Hz].

    Returns
    -------
    U_mat : np.ndarray, shape (n_free, n_freqs), dtype complex128
        Déplacements complexes par superposition modale tronquée.

    Notes
    -----
    Pour chaque fréquence Ω = 2πf :

        denom_n = ωₙ² − Ω² + 2iξₙωₙΩ          (n_modes scalaires)
        q_n     = Γₙ / denom_n                  (coordonnées modales)
        U_free  = Φ · q = Σₙ φₙ · q_n           (retour en physique)

    Complexité : O(n_free × n_modes) par fréquence — beaucoup moins coûteux que
    la résolution directe O(n_free³) quand n_modes ≪ n_free.

    Si n_modes < n_dof, la réponse statique n'est pas exactement reproduite
    (erreur de troncature modale).  Pour une limite statique exacte, utiliser
    la correction statique (static residual correction, non implémentée ici).

    À la résonance du mode n (Ω = ωₙ) :

        denom_n = 2iξₙωₙ²    →    q_n = Γₙ / (2iξₙωₙ²)

    Le terme dominant donne |U[dof]| ≈ |φₙ[dof]| · |Γₙ| / (2ξₙωₙ²).
    Pour un SDOF M-normalisé (φ₁ = 1/√m) : |H(ωₙ)| = 1/(2ξωₙ²m) = 1/(2ξk).
    """
    n_freqs = len(freqs)
    n_free  = phi_free.shape[0]

    # Facteurs de participation modale Γₙ = φₙᵀ·F̂  →  (n_modes,)
    Gamma = phi_free.T @ F_free

    U_mat = np.zeros((n_free, n_freqs), dtype=complex)
    for k, f in enumerate(freqs):
        Omega = 2.0 * np.pi * f
        # Dénominateur modal : (n_modes,) — vectorisé sur les modes
        denom = omega_n ** 2 - Omega ** 2 + 2j * zeta_n * omega_n * Omega
        q     = Gamma / denom          # coordonnées modales (n_modes,)
        U_mat[:, k] = phi_free @ q     # retour en physique (n_free,)

    return U_mat


# ---------------------------------------------------------------------------
# Interface de haut niveau
# ---------------------------------------------------------------------------


def run_harmonic(
    mesh: Mesh,
    bc: BoundaryConditions,
    F_hat: np.ndarray,
    freqs: np.ndarray,
    damping: RayleighDamping | HystereticDamping | ModalDampingModel | None = None,
) -> HarmonicResult:
    """Exécute un balayage en fréquence complet.

    Assemble K, M, construit C (Rayleigh), réduit aux DDL libres, puis
    résout Z(Ω)·U_free = F̂_free pour chaque fréquence.

    Parameters
    ----------
    mesh : Mesh
        Maillage du modèle.
    bc : BoundaryConditions
        Conditions aux limites (seule la partie Dirichlet est utilisée).
        Les forces harmoniques sont passées via ``F_hat``.
    F_hat : np.ndarray, shape (n_dof,)
        Amplitude des forces nodales [N] (vecteur réel).
        Les DDL contraints sont ignorés (F_hat[constrained] peut être 0).
    freqs : np.ndarray, shape (n_freqs,)
        Fréquences d'excitation [Hz] pour le balayage.
    damping : RayleighDamping, HystereticDamping, ModalDampingModel, or None
        Modèle d'amortissement.  Dispatch automatique vers le solveur adapté :

        - ``RayleighDamping`` : Z = K − Ω²M + iΩ(αM + βK)
        - ``HystereticDamping`` : Z = K(1+iη) − Ω²M
        - ``ModalDampingModel`` : superposition modale U = Σₙ φₙΓₙ/(ωₙ²−Ω²+2iξₙωₙΩ)
        - ``None`` : C = 0 (système non amorti — quasi-singulier aux résonances)

    Returns
    -------
    HarmonicResult
        Contient ``freqs``, ``U`` (complex, n_dof × n_freqs) et ``free_dofs``.

    Notes
    -----
    La réduction aux DDL libres est faite **une seule fois** avant la boucle
    en fréquences.

    Pour ``ModalDampingModel``, ``damping.phi`` doit être de taille complète
    (n_dof, n_modes) — tel que retourné par ``run_modal``.  La réduction aux
    DDL libres (phi[free, :]) est effectuée ici.

    Examples
    --------
    Rayleigh 2% :

    >>> d = rayleigh_from_modes(omega1, omega2, zeta1=0.02, zeta2=0.02)
    >>> result = run_harmonic(mesh, bc, F_hat, freqs, d)

    Hystérétique η=0.04 (≡ ξ=2%) :

    >>> result = run_harmonic(mesh, bc, F_hat, freqs, HystereticDamping(eta=0.04))

    Modal 2% sur tous les modes :

    >>> modal_result = run_modal(mesh, bc, n_modes=5)
    >>> d = ModalDampingModel.from_modal_result(modal_result, zeta_n=0.02)
    >>> result = run_harmonic(mesh, bc, F_hat, freqs, d)
    """
    assembler = Assembler(mesh)
    K = assembler.assemble_stiffness()
    M = assembler.assemble_mass()

    # Réduction aux DDL libres (élimination vraie, pas de modes parasites)
    F_dummy = np.zeros(mesh.n_dof)
    ds   = apply_dirichlet(K, F_dummy, mesh, bc)
    free = ds.free_dofs

    K_free = ds.K_free
    M_free = ds.reduce_mass(M)
    F_free = F_hat[free].astype(float)

    # Dispatch selon le type d'amortissement
    if damping is None:
        n_free = len(free)
        C_free = csr_matrix((n_free, n_free))
        U_free = solve_harmonic(K_free, M_free, C_free, F_free, freqs)

    elif isinstance(damping, RayleighDamping):
        C      = build_damping_matrix(damping, M, K)
        C_free = C[free, :][:, free].tocsr()
        U_free = solve_harmonic(K_free, M_free, C_free, F_free, freqs)

    elif isinstance(damping, HystereticDamping):
        U_free = solve_harmonic_hysteretic(K_free, M_free, damping.eta, F_free, freqs)

    elif isinstance(damping, ModalDampingModel):
        # phi stocké en taille complète (n_dof, n_modes) — extraire les DDL libres
        phi_free = damping.phi[free, :]
        U_free   = solve_harmonic_modal(
            damping.omega_n, damping.zeta_n, phi_free, F_free, freqs
        )

    else:
        raise TypeError(
            f"Type d'amortissement non supporté : {type(damping).__name__}.  "
            "Utiliser RayleighDamping, HystereticDamping, ModalDampingModel ou None."
        )

    # Reconstruction à la taille n_dof (DDL contraints = 0)
    n_dof  = mesh.n_dof
    n_freq = len(freqs)
    U_full = np.zeros((n_dof, n_freq), dtype=complex)
    U_full[free, :] = U_free

    return HarmonicResult(
        freqs=np.asarray(freqs, dtype=float),
        U=U_full,
        free_dofs=free,
    )

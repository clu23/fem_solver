"""Réponse aléatoire — analyse spectrale PSD (Power Spectral Density).

Principe physique
-----------------
Pour un système linéaire excité par un processus aléatoire stationnaire, la
densité spectrale de puissance de la sortie s'obtient directement à partir de
celle de l'entrée via la FRF au carré :

    G_sortie(f) = |H(f)|² · G_entrée(f)

où :

- G [unit²/Hz] est la PSD unilatérale (one-sided, f ≥ 0),
- H(f) = H(2πf) est la fonction de réponse en fréquence (FRF),
- la FRF est la même qu'en harmonique : H(Ω) = (K − Ω²M + iΩC)⁻¹ · F_dir.

La valeur RMS (root-mean-square, écart-type pour processus centré) est :

    σ = √(∫₀^∞ G_sortie(f) df)

calculée numériquement par intégration trapézoïdale sur [f_min, f_max].

Excitation en force
-------------------
Pour une force aléatoire dans la direction ``F_dir`` (vecteur nodal) :

    H_u(Ω) = (K − Ω²M + iΩC)⁻¹ · F_dir      [m/N]  (ou [m] si F_dir = 1 N)
    G_u(f)  = |H_u(f)|² · G_F(f)              [m²/Hz si G_F en N²/Hz]

Excitation à la base (base excitation)
---------------------------------------
Pour un sol se déplaçant selon la direction ``r`` (vecteur d'influence) :

    F_eff(Ω) = −M · r · Ü_g(Ω)

En PSD d'accélération (G_0 en (m/s²)²/Hz) :

    H_rel(Ω)   = −(K − Ω²M + iΩC)⁻¹ · M_free · r_free   [m/(m/s²)]
    H_a_abs(Ω) = −Ω² · H_rel(Ω) + r_free                 [(m/s²)/(m/s²)]

    G_u_rel(f)  = |H_rel(f)|²  · G_0(f)    [m²/Hz]
    G_a_abs(f)  = |H_a_abs(f)|² · G_0(f)   [(m/s²)²/Hz]

Équation de Miles (SDOF, bruit blanc)
--------------------------------------
Pour un oscillateur 1-DDL sous bruit blanc en accélération de PSD G₀ :

    σ_a ≈ √(π · fₙ · G₀ / (4ξ))   [m/s²]   (Miles' equation, one-sided PSD)

Approxime l'intégrale totale ∫₀^∞ |T(f)|² G₀ df ≈ G₀×πfₙ/(4ξ) (résonance
dominante, ξ petit).  Pour le déplacement relatif :

    σ_u_rel = √(G₀ / (8ξωₙ³))     [m]

Références
----------
Miles J.W. (1954), «On structural fatigue under random loading», J. Aeronaut. Sci.
Wirsching P.H. et al., «Random Vibrations: Theory and Practice», Wiley 1995.
Clough R.W. & Penzien J., «Dynamics of Structures», 3rd ed., §20.
Harris C.M. & Piersol A.G., «Harris' Shock and Vibration Handbook», 5th ed., §22.

Note on Miles equation
----------------------
The one-sided PSD formulation (f ≥ 0) gives:

    σ²_a = G₀ × ∫₀^∞ |T(f)|² df  ≈  G₀ × π fₙ / (4ξ)

because ∫₀^∞ |T(f)|² df = πfₙ/(4ξ) for the transmissibility of a lightly
damped SDOF (resonance approximation).  Some textbooks write this as
σ_a = √(π fₙ Q G₀ / 2) with Q = 1/(2ξ), which is identical.
Do NOT confuse with the two-sided-PSD form: the extra factor of 2 does not
appear here because the one-sided PSD already integrates over [0, ∞).
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
class RandomResult:
    """Résultats d'une analyse de réponse aléatoire.

    Attributes
    ----------
    freqs : np.ndarray, shape (n_freqs,)
        Fréquences d'analyse [Hz].
    G_u : np.ndarray, shape (n_dof, n_freqs)
        PSD unilatérale du déplacement [m²/Hz] (déplacement relatif pour
        base excitation).  Les DDL contraints ont G_u = 0.
    rms_u : np.ndarray, shape (n_dof,)
        Valeur RMS du déplacement [m] = √(∫G_u df).
    free_dofs : np.ndarray, shape (n_free,)
        Indices des DDL libres.
    G_a : np.ndarray or None, shape (n_dof, n_freqs)
        PSD de l'accélération absolue [(m/s²)²/Hz] — uniquement pour
        ``run_random_base``.  None pour excitation en force.
    rms_a : np.ndarray or None, shape (n_dof,)
        Valeur RMS de l'accélération absolue [m/s²] — uniquement pour
        ``run_random_base``.  None pour excitation en force.

    Notes
    -----
    Pour extraire la PSD et la RMS d'un DDL particulier (p.ex. DDL 5) :

    .. code-block:: python

        G5 = result.G_u[5, :]     # PSD, shape (n_freqs,)
        s5 = result.rms_u[5]      # RMS scalaire
    """

    freqs: np.ndarray
    G_u: np.ndarray
    rms_u: np.ndarray
    free_dofs: np.ndarray
    G_a: np.ndarray | None = None
    rms_a: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Utilitaires publics
# ---------------------------------------------------------------------------


def miles_equation(fn: float, zeta: float, G0: float) -> float:
    """Équation de Miles — RMS accélération pour un SDOF sous bruit blanc.

    Valide pour G_input plat (G₀) sur une large plage contenant la résonance
    et un amortissement léger (ξ ≤ 0.20).  Approxime l'intégrale totale
    ∫₀^∞ |T(f)|² G₀ df ≈ G₀ × πfₙ/(4ξ) (résonance dominante).

    Parameters
    ----------
    fn : float
        Fréquence propre [Hz].
    zeta : float
        Taux d'amortissement visqueux [-].
    G0 : float
        Densité spectrale d'accélération du sol (one-sided) [(m/s²)²/Hz].

    Returns
    -------
    float
        σ_a = √(π · fₙ · G₀ / (4ξ))  [m/s²].

    Notes
    -----
    Convention one-sided (f ≥ 0) : ∫₀^∞ |T|² df = πfₙ/(4ξ).
    Forme équivalente : σ_a = √(π fₙ Q G₀ / 2) avec Q = 1/(2ξ).
    """
    return float(np.sqrt(np.pi * fn * G0 / (4.0 * zeta)))


def compute_rms(G: np.ndarray, freqs: np.ndarray) -> np.ndarray | float:
    """Calculer la valeur RMS par intégration trapézoïdale de la PSD.

    Parameters
    ----------
    G : np.ndarray, shape (n_freqs,) ou (n_dof, n_freqs)
        PSD unilatérale.  Si 1D, un scalaire est retourné.
    freqs : np.ndarray, shape (n_freqs,)
        Fréquences associées [Hz].  Doit être croissant.

    Returns
    -------
    float ou np.ndarray, shape (n_dof,)
        σ = √(∫G df) via intégration trapézoïdale.
    """
    if G.ndim == 1:
        return float(np.sqrt(np.trapezoid(G, freqs)))
    return np.sqrt(np.trapezoid(G, freqs, axis=-1))


def psd_white(G0: float, freqs: np.ndarray) -> np.ndarray:
    """Créer un spectre de bruit blanc constant.

    Parameters
    ----------
    G0 : float
        Niveau PSD constant [unit²/Hz].
    freqs : np.ndarray, shape (n_freqs,)
        Fréquences d'analyse [Hz].

    Returns
    -------
    np.ndarray, shape (n_freqs,)
        G_input = G₀ · ones(n_freqs).
    """
    return G0 * np.ones(len(freqs))


def influence_vector(mesh: Mesh, direction: int) -> np.ndarray:
    """Construire le vecteur d'influence pour une excitation à la base.

    Le vecteur d'influence ``r`` a r[i] = 1 pour tout DDL de translation dans
    la direction ``direction``, 0 ailleurs (rotations, autres directions).

    Parameters
    ----------
    mesh : Mesh
        Maillage contenant ``n_dof``, ``dof_per_node``, ``n_dim``.
    direction : int
        Indice de la direction de translation (0=x, 1=y, 2=z).
        Doit être < n_dim.

    Returns
    -------
    np.ndarray, shape (n_dof,)
        Vecteur d'influence r.

    Raises
    ------
    ValueError
        Si ``direction`` est hors des dimensions du maillage.
    """
    if direction >= mesh.n_dim:
        raise ValueError(
            f"direction={direction} hors des dimensions du maillage (n_dim={mesh.n_dim})."
        )
    r = np.zeros(mesh.n_dof)
    dpn = mesh.dpn
    n_nodes = mesh.n_nodes
    for i in range(n_nodes):
        # Les DDL de translation sont les indices 0..n_dim-1 dans le nœud
        r[i * dpn + direction] = 1.0
    return r


# ---------------------------------------------------------------------------
# Solveurs FRF de bas niveau
# ---------------------------------------------------------------------------


def _frf_direct(
    K_free: csr_matrix,
    M_free: csr_matrix,
    C_free: csr_matrix,
    F_dir_free: np.ndarray,
    freqs: np.ndarray,
) -> np.ndarray:
    """Calculer H(f) = Z(2πf)⁻¹ · F_dir pour chaque fréquence.

    Parameters
    ----------
    K_free, M_free, C_free : csr_matrix, shape (n_free, n_free)
        Matrices réduites.
    F_dir_free : np.ndarray, shape (n_free,)
        Vecteur de direction de la force (réel).
    freqs : np.ndarray, shape (n_freqs,)
        Fréquences [Hz].

    Returns
    -------
    H : np.ndarray, shape (n_free, n_freqs), dtype complex128
        FRF pour chaque DDL libre et chaque fréquence.
    """
    n_free = K_free.shape[0]
    n_freqs = len(freqs)
    H = np.zeros((n_free, n_freqs), dtype=complex)

    K_c = K_free.astype(complex)
    M_c = M_free.astype(complex)
    C_c = C_free.astype(complex)
    F_c = F_dir_free.astype(complex)

    for k, f in enumerate(freqs):
        Omega = 2.0 * np.pi * f
        Z = K_c - (Omega ** 2) * M_c + (1j * Omega) * C_c
        H[:, k] = spsolve(Z.tocsr(), F_c)

    return H


def _frf_hysteretic(
    K_free: csr_matrix,
    M_free: csr_matrix,
    eta: float,
    F_dir_free: np.ndarray,
    freqs: np.ndarray,
) -> np.ndarray:
    """FRF avec amortissement hystérétique : Z = K(1+iη) − Ω²M.

    Parameters
    ----------
    K_free, M_free : csr_matrix
        Matrices de rigidité et de masse réduites.
    eta : float
        Facteur de perte η.
    F_dir_free : np.ndarray, shape (n_free,)
        Vecteur de direction de la force.
    freqs : np.ndarray, shape (n_freqs,)
        Fréquences [Hz].

    Returns
    -------
    H : np.ndarray, shape (n_free, n_freqs), dtype complex128
    """
    n_free = K_free.shape[0]
    n_freqs = len(freqs)
    H = np.zeros((n_free, n_freqs), dtype=complex)

    K_complex = K_free.astype(complex) * (1.0 + 1j * eta)
    M_c = M_free.astype(complex)
    F_c = F_dir_free.astype(complex)

    for k, f in enumerate(freqs):
        Omega = 2.0 * np.pi * f
        Z = K_complex - (Omega ** 2) * M_c
        H[:, k] = spsolve(Z.tocsr(), F_c)

    return H


def _frf_modal(
    omega_n: np.ndarray,
    zeta_n: np.ndarray,
    phi_free: np.ndarray,
    F_dir_free: np.ndarray,
    freqs: np.ndarray,
) -> np.ndarray:
    """FRF par superposition modale.

    H(Ω) = Σₙ φₙ · (φₙᵀ F̂) / (ωₙ² − Ω² + 2iξₙωₙΩ)

    Parameters
    ----------
    omega_n : np.ndarray, shape (n_modes,)
        Pulsations propres [rad/s].
    zeta_n : np.ndarray, shape (n_modes,)
        Taux d'amortissement par mode.
    phi_free : np.ndarray, shape (n_free, n_modes)
        Vecteurs propres M-normalisés aux DDL libres.
    F_dir_free : np.ndarray, shape (n_free,)
        Vecteur de direction (réel).
    freqs : np.ndarray, shape (n_freqs,)
        Fréquences [Hz].

    Returns
    -------
    H : np.ndarray, shape (n_free, n_freqs), dtype complex128
    """
    n_freqs = len(freqs)
    Gamma = phi_free.T @ F_dir_free          # facteurs de participation (n_modes,)

    H = np.zeros((phi_free.shape[0], n_freqs), dtype=complex)
    for k, f in enumerate(freqs):
        Omega = 2.0 * np.pi * f
        denom = omega_n ** 2 - Omega ** 2 + 2j * zeta_n * omega_n * Omega
        q = Gamma / denom
        H[:, k] = phi_free @ q

    return H


def _build_C_free(
    damping: RayleighDamping | HystereticDamping | ModalDampingModel | None,
    M: csr_matrix,
    K: csr_matrix,
    free: np.ndarray,
) -> csr_matrix:
    """Construire C_free pour amortissement visqueux (Rayleigh ou zéro)."""
    n_free = len(free)
    if damping is None:
        return csr_matrix((n_free, n_free))
    if isinstance(damping, RayleighDamping):
        C = build_damping_matrix(damping, M, K)
        return C[free, :][:, free].tocsr()
    return csr_matrix((n_free, n_free))


def _dispatch_frf(
    damping: RayleighDamping | HystereticDamping | ModalDampingModel | None,
    K_free: csr_matrix,
    M_free: csr_matrix,
    C_free: csr_matrix,
    F_dir_free: np.ndarray,
    freqs: np.ndarray,
    free: np.ndarray,
) -> np.ndarray:
    """Dispatcher vers le bon solveur FRF selon le type d'amortissement."""
    if isinstance(damping, HystereticDamping):
        return _frf_hysteretic(K_free, M_free, damping.eta, F_dir_free, freqs)
    if isinstance(damping, ModalDampingModel):
        phi_free = damping.phi[free, :]
        return _frf_modal(damping.omega_n, damping.zeta_n, phi_free, F_dir_free, freqs)
    return _frf_direct(K_free, M_free, C_free, F_dir_free, freqs)


# ---------------------------------------------------------------------------
# Solveurs bas niveau publics
# ---------------------------------------------------------------------------


def solve_random_force(
    K_free: csr_matrix,
    M_free: csr_matrix,
    C_free: csr_matrix,
    F_dir_free: np.ndarray,
    G_input: np.ndarray,
    freqs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculer la PSD et la RMS de déplacement sous excitation aléatoire en force.

    Interface bas niveau (matrices déjà réduites).

    Parameters
    ----------
    K_free, M_free, C_free : csr_matrix, shape (n_free, n_free)
        Matrices réduites aux DDL libres.
    F_dir_free : np.ndarray, shape (n_free,)
        Vecteur de direction de la force (réel).  Doit être normalisé selon
        l'usage : p.ex. [0,…,1,…,0] pour une force unitaire sur un DDL.
    G_input : np.ndarray, shape (n_freqs,)
        PSD unilatérale de l'entrée [N²/Hz].
    freqs : np.ndarray, shape (n_freqs,)
        Fréquences [Hz], croissantes.

    Returns
    -------
    G_u_free : np.ndarray, shape (n_free, n_freqs)
        PSD du déplacement aux DDL libres [m²/Hz].
    rms_u_free : np.ndarray, shape (n_free,)
        RMS du déplacement [m].
    """
    H = _frf_direct(K_free, M_free, C_free, F_dir_free, freqs)
    G_u_free = np.abs(H) ** 2 * G_input[np.newaxis, :]
    rms_u_free = compute_rms(G_u_free, freqs)
    return G_u_free, rms_u_free


# ---------------------------------------------------------------------------
# Interface de haut niveau — excitation en force
# ---------------------------------------------------------------------------


def run_random_force(
    mesh: Mesh,
    bc: BoundaryConditions,
    F_direction: np.ndarray,
    G_input: np.ndarray,
    freqs: np.ndarray,
    damping: RayleighDamping | HystereticDamping | ModalDampingModel | None = None,
) -> RandomResult:
    """Analyse de réponse aléatoire sous excitation en force.

    Parameters
    ----------
    mesh : Mesh
        Maillage du modèle.
    bc : BoundaryConditions
        Conditions aux limites de Dirichlet.
    F_direction : np.ndarray, shape (n_dof,)
        Vecteur de direction de la force [N].  Définit la forme spatiale de
        l'excitation.  La magnitude est portée par ``G_input``.
    G_input : np.ndarray, shape (n_freqs,)
        PSD unilatérale de la force d'entrée [N²/Hz].
    freqs : np.ndarray, shape (n_freqs,)
        Fréquences d'analyse [Hz], croissantes.
    damping : RayleighDamping, HystereticDamping, ModalDampingModel or None
        Modèle d'amortissement (même convention que ``run_harmonic``).

    Returns
    -------
    RandomResult
        Champs G_u [m²/Hz], rms_u [m].  G_a et rms_a sont None.

    Notes
    -----
    G_u(f) = |H_u(f)|² · G_F(f) avec H_u = (K−Ω²M+iΩC)⁻¹ · F_dir.
    """
    assembler = Assembler(mesh)
    K = assembler.assemble_stiffness()
    M = assembler.assemble_mass()

    F_dummy = np.zeros(mesh.n_dof)
    ds = apply_dirichlet(K, F_dummy, mesh, bc)
    free = ds.free_dofs

    K_free = ds.K_free
    M_free = ds.reduce_mass(M)
    C_free = _build_C_free(damping, M, K, free)
    F_dir_free = F_direction[free].astype(float)

    H = _dispatch_frf(damping, K_free, M_free, C_free, F_dir_free, freqs, free)
    G_u_free = np.abs(H) ** 2 * G_input[np.newaxis, :]
    rms_u_free = compute_rms(G_u_free, freqs)

    n_dof = mesh.n_dof
    n_freqs = len(freqs)
    G_u_full = np.zeros((n_dof, n_freqs))
    rms_u_full = np.zeros(n_dof)
    G_u_full[free, :] = G_u_free
    rms_u_full[free] = rms_u_free

    return RandomResult(
        freqs=np.asarray(freqs, dtype=float),
        G_u=G_u_full,
        rms_u=rms_u_full,
        free_dofs=free,
        G_a=None,
        rms_a=None,
    )


# ---------------------------------------------------------------------------
# Interface de haut niveau — excitation à la base
# ---------------------------------------------------------------------------


def run_random_base(
    mesh: Mesh,
    bc: BoundaryConditions,
    direction: int,
    G_input: np.ndarray,
    freqs: np.ndarray,
    damping: RayleighDamping | HystereticDamping | ModalDampingModel | None = None,
) -> RandomResult:
    """Analyse de réponse aléatoire sous excitation à la base (base excitation).

    L'accélération du sol est un bruit aléatoire dans la direction ``direction``
    avec PSD G₀(f).  La réponse est calculée en déplacement relatif (sol/structure)
    et en accélération absolue.

    Parameters
    ----------
    mesh : Mesh
        Maillage du modèle.
    bc : BoundaryConditions
        Conditions aux limites de Dirichlet (encastrement de la base).
    direction : int
        Direction de l'excitation (0=x, 1=y, 2=z).
    G_input : np.ndarray, shape (n_freqs,)
        PSD unilatérale de l'accélération du sol [(m/s²)²/Hz].
    freqs : np.ndarray, shape (n_freqs,)
        Fréquences d'analyse [Hz], croissantes.
    damping : RayleighDamping, HystereticDamping, ModalDampingModel or None
        Modèle d'amortissement.

    Returns
    -------
    RandomResult
        - G_u [m²/Hz]  : PSD du déplacement relatif
        - rms_u [m]    : RMS déplacement relatif
        - G_a [(m/s²)²/Hz] : PSD accélération absolue
        - rms_a [m/s²] : RMS accélération absolue

    Notes
    -----
    Avec r = vecteur d'influence (translation dans ``direction``) :

        F_eff(Ω) = −M_free · r_free · Ω²    [N]  (par unité d'accélération sol)

    En divisant par Ω² (entrée accélération → sortie déplacement) :

        H_rel(Ω)   = (K−Ω²M+iΩC)⁻¹ · (−M_free · r_free)   [m/(m/s²)]
        H_a_abs(Ω) = −Ω² · H_rel(Ω) + r_free                [(m/s²)/(m/s²)]

    G_u_rel(f)  = |H_rel(f)|²   · G_0(f)
    G_a_abs(f)  = |H_a_abs(f)|² · G_0(f)

    La FRF relative est calculée à chaque fréquence par spsolve (ou superposition
    modale), avec F_eff = −M_free · r_free comme vecteur de direction constant.
    """
    assembler = Assembler(mesh)
    K = assembler.assemble_stiffness()
    M = assembler.assemble_mass()

    F_dummy = np.zeros(mesh.n_dof)
    ds = apply_dirichlet(K, F_dummy, mesh, bc)
    free = ds.free_dofs

    K_free = ds.K_free
    M_free = ds.reduce_mass(M)
    C_free = _build_C_free(damping, M, K, free)

    # Vecteur d'influence aux DDL libres
    r_full = influence_vector(mesh, direction)
    r_free = r_full[free]

    # Vecteur de force effective : F_eff = −M_free · r_free
    # (par unité d'accélération de sol)
    M_free_dense = M_free.toarray() if hasattr(M_free, "toarray") else np.array(M_free)
    F_eff_free = -(M_free_dense @ r_free)

    # FRF relative : H_rel(Ω) = Z(Ω)⁻¹ · F_eff
    H_rel = _dispatch_frf(damping, K_free, M_free, C_free, F_eff_free, freqs, free)

    # FRF accélération absolue : H_a_abs(Ω) = −Ω² H_rel + r_free
    omegas = (2.0 * np.pi * freqs)[np.newaxis, :]        # (1, n_freqs)
    H_a_abs = -(omegas ** 2) * H_rel + r_free[:, np.newaxis]

    # PSD et RMS
    G_u_free = np.abs(H_rel) ** 2 * G_input[np.newaxis, :]
    G_a_free = np.abs(H_a_abs) ** 2 * G_input[np.newaxis, :]

    rms_u_free = compute_rms(G_u_free, freqs)
    rms_a_free = compute_rms(G_a_free, freqs)

    n_dof = mesh.n_dof
    n_freqs = len(freqs)
    G_u_full = np.zeros((n_dof, n_freqs))
    G_a_full = np.zeros((n_dof, n_freqs))
    rms_u_full = np.zeros(n_dof)
    rms_a_full = np.zeros(n_dof)

    G_u_full[free, :] = G_u_free
    G_a_full[free, :] = G_a_free
    rms_u_full[free] = rms_u_free
    rms_a_full[free] = rms_a_free

    return RandomResult(
        freqs=np.asarray(freqs, dtype=float),
        G_u=G_u_full,
        rms_u=rms_u_full,
        free_dofs=free,
        G_a=G_a_full,
        rms_a=rms_a_full,
    )

"""Solveurs statique et modal (pattern Strategy pour le backend)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh, spsolve


class SolverBackend(ABC):
    """Interface abstraite pour les backends de résolution.

    Permet de substituer scipy par MUMPS ou PETSc sans modifier le code appelant.
    """

    @abstractmethod
    def solve_static(self, K: csr_matrix, F: np.ndarray) -> np.ndarray:
        """Résoudre le système linéaire creux K·u = F.

        Parameters
        ----------
        K : csr_matrix
            Matrice de rigidité (modifiée par conditions aux limites).
        F : np.ndarray
            Vecteur de forces.

        Returns
        -------
        u : np.ndarray, shape (n_dof,)
            Vecteur de déplacements.
        """
        ...

    @abstractmethod
    def solve_buckling(
        self,
        K: csr_matrix,
        K_g: csr_matrix,
        n_modes: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Résoudre le problème de flambage linéaire K·φ = λ·(−K_g)·φ.

        Parameters
        ----------
        K : csr_matrix
            Matrice de rigidité élastique (définie positive, avec CL).
        K_g : csr_matrix
            Matrice de rigidité géométrique (semi-définie négative pour
            un chargement compressif).
        n_modes : int
            Nombre de modes de flambage à extraire.

        Returns
        -------
        lambda_cr : np.ndarray, shape (n_modes,)
            Multiplicateurs de charge critique (triés par ordre croissant).
            La charge critique est P_cr = lambda_cr × P_ref.
        phi : np.ndarray, shape (n_dof, n_modes)
            Modes de flambage (vecteurs propres).
        """
        ...

    @abstractmethod
    def solve_eigen(
        self,
        K: csr_matrix,
        M: csr_matrix,
        n_modes: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Résoudre le problème aux valeurs propres généralisé K·φ = ω²·M·φ.

        Parameters
        ----------
        K : csr_matrix
            Matrice de rigidité.
        M : csr_matrix
            Matrice de masse.
        n_modes : int
            Nombre de modes à extraire.

        Returns
        -------
        omega_sq : np.ndarray, shape (n_modes,)
            Valeurs propres ω² [rad²/s²] triées par ordre croissant.
        phi : np.ndarray, shape (n_dof, n_modes)
            Vecteurs propres (modes) normalisés par rapport à M.
        """
        ...


class ScipyBackend(SolverBackend):
    """Backend scipy (spsolve + eigsh).

    Utilisé par défaut. Efficace jusqu'à ~100 000 DDL.
    """

    def solve_static(self, K: csr_matrix, F: np.ndarray) -> np.ndarray:
        """Résoudre K·u = F avec scipy.sparse.linalg.spsolve.

        Parameters
        ----------
        K : csr_matrix
            Matrice de rigidité symétrique définie positive.
        F : np.ndarray
            Vecteur de forces.

        Returns
        -------
        u : np.ndarray, shape (n_dof,)
            Vecteur de déplacements.
        """
        return spsolve(K, F)

    def solve_buckling(
        self,
        K: csr_matrix,
        K_g: csr_matrix,
        n_modes: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Résoudre K·φ = λ·(−K_g)·φ avec eigsh (shift-invert, σ = 0).

        Reformulation : K·φ = λ·(−K_g)·φ est un problème aux valeurs
        propres généralisé symétrique où K est SPD et (−K_g) est PSD
        pour un chargement de compression pur.

        L'algorithme shift-invert avec σ = 0 factorise K (Cholesky) et
        trouve les plus petits λ (charges critiques les plus basses).

        Parameters
        ----------
        K : csr_matrix
            Matrice de rigidité élastique (SPD, DDL libres seulement).
        K_g : csr_matrix
            Matrice de rigidité géométrique (semi-définie négative pour
            compression, DDL libres seulement).
        n_modes : int
            Nombre de modes de flambage à extraire.

        Returns
        -------
        lambda_cr : np.ndarray, shape (n_modes,)
            Multiplicateurs de charge critique (triés croissants).
        phi : np.ndarray, shape (n_dof, n_modes)
            Modes de flambage.

        Notes
        -----
        Requiert −K_g semi-définie positive (chargement principalement
        compressif). Si K_g a des parties positives (traction), ajouter
        un décalage ε·I à −K_g pour la stabilité numérique.
        """
        K_g_neg = (-K_g).tocsr()
        # eigsh(A, M=B) résout A·v = λ·B·v
        # Avec A=K, B=−K_g, σ=0 → trouve les λ les plus petits (charges critiques)
        lambda_vals, phi = eigsh(K, k=n_modes, M=K_g_neg, sigma=0.0, which="LM")
        idx = np.argsort(lambda_vals)
        return lambda_vals[idx], phi[:, idx]

    def solve_eigen(
        self,
        K: csr_matrix,
        M: csr_matrix,
        n_modes: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Résoudre K·φ = ω²·M·φ avec scipy.sparse.linalg.eigsh (Lanczos).

        Parameters
        ----------
        K : csr_matrix
            Matrice de rigidité (semi-définie positive après CL).
        M : csr_matrix
            Matrice de masse (définie positive).
        n_modes : int
            Nombre de modes à extraire (les plus basses fréquences).

        Returns
        -------
        omega_sq : np.ndarray, shape (n_modes,)
            Valeurs propres ω² triées.
        phi : np.ndarray, shape (n_dof, n_modes)
            Vecteurs propres.
        """
        # Shift-invert (σ=0) : trouve les plus petites valeurs propres de
        # K φ = ω² M φ en résolvant (K - σM)⁻¹ M φ = (1/ω²) φ.
        # Beaucoup plus stable que which="SM" car le conditionnement de
        # l'itération Lanczos ne dépend plus du ratio ω_max/ω_min.
        # Requiert K positif défini (assuré après conditions aux limites).
        omega_sq, phi = eigsh(K, k=n_modes, M=M, sigma=0.0, which="LM")
        idx = np.argsort(omega_sq)
        return omega_sq[idx], phi[:, idx]


class StaticSolver:
    """Solveur pour l'analyse statique linéaire K·u = F.

    Parameters
    ----------
    backend : SolverBackend, optional
        Backend de résolution. Par défaut : ScipyBackend.

    Examples
    --------
    >>> solver = StaticSolver()
    >>> u = solver.solve(K, F)
    """

    def __init__(self, backend: SolverBackend | None = None) -> None:
        self.backend: SolverBackend = backend or ScipyBackend()

    def solve(self, K: csr_matrix, F: np.ndarray) -> np.ndarray:
        """Résoudre le système statique K·u = F.

        Parameters
        ----------
        K : csr_matrix
            Matrice de rigidité (avec conditions aux limites appliquées).
        F : np.ndarray, shape (n_dof,)
            Vecteur de forces (avec conditions aux limites appliquées).

        Returns
        -------
        u : np.ndarray, shape (n_dof,)
            Vecteur de déplacements nodaux [m].
        """
        return self.backend.solve_static(K, F)


class ModalSolver:
    """Solveur pour l'analyse modale — problème aux valeurs propres généralisé.

    Parameters
    ----------
    backend : SolverBackend, optional
        Backend de résolution. Par défaut : ScipyBackend.

    Examples
    --------
    >>> solver = ModalSolver()
    >>> freqs, modes = solver.solve(K, M, n_modes=5)
    """

    def __init__(self, backend: SolverBackend | None = None) -> None:
        self.backend: SolverBackend = backend or ScipyBackend()

    def solve(
        self,
        K: csr_matrix,
        M: csr_matrix,
        n_modes: int = 5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extraire les n_modes premières fréquences propres et modes de vibration.

        Parameters
        ----------
        K : csr_matrix
            Matrice de rigidité (avec conditions aux limites).
        M : csr_matrix
            Matrice de masse.
        n_modes : int
            Nombre de modes à extraire.

        Returns
        -------
        freqs : np.ndarray, shape (n_modes,)
            Fréquences propres [Hz] : f_n = ω_n / (2π).
        modes : np.ndarray, shape (n_dof, n_modes)
            Modes propres normalisés.
        """
        omega_sq, modes = self.backend.solve_eigen(K, M, n_modes)
        omega_sq = np.maximum(omega_sq, 0.0)  # Évite les valeurs propres < 0 (bruit numérique)
        freqs = np.sqrt(omega_sq) / (2.0 * np.pi)
        return freqs, modes


class BucklingSolver:
    """Solveur pour l'analyse de flambage linéaire.

    Résout le problème aux valeurs propres généralisé :

        (K + λ K_g) φ = 0   ⟺   K φ = λ (−K_g) φ

    où :
    - **K** est la matrice de rigidité élastique (définie positive après CL).
    - **K_g** est la matrice de rigidité géométrique calculée à partir de
      l'état pré-flambement (semi-définie négative pour un chargement
      principalement compressif).
    - **λ** (positif) est le multiplicateur de charge critique.
    - **φ** est le mode de flambage.

    La charge critique est : P_cr = λ_min × P_ref, où P_ref est la charge
    de référence utilisée pour calculer K_g.

    Parameters
    ----------
    backend : SolverBackend, optional
        Backend de résolution. Par défaut : ScipyBackend.

    Examples
    --------
    >>> # Workflow complet — colonne d'Euler pince-pincée
    >>> assembler = Assembler(mesh)
    >>> K = assembler.assemble_stiffness()
    >>> F = assembler.assemble_forces(bc)           # charge P_ref = 1 N
    >>> ds = apply_dirichlet(K, F, mesh, bc)
    >>> u = StaticSolver().solve(*ds)               # état pré-flambement
    >>> K_g = assembler.assemble_geometric_stiffness(u)
    >>> lambda_cr, phi = BucklingSolver().solve(ds.K_free, ds.reduce(K_g))
    >>> P_cr = lambda_cr[0]                         # P_ref = 1 → P_cr = λ_cr
    """

    def __init__(self, backend: SolverBackend | None = None) -> None:
        self.backend: SolverBackend = backend or ScipyBackend()

    def solve(
        self,
        K: csr_matrix,
        K_g: csr_matrix,
        n_modes: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculer les multiplicateurs de charge critique et les modes de flambage.

        Parameters
        ----------
        K : csr_matrix, shape (n_free, n_free)
            Matrice de rigidité élastique réduite aux DDL libres
            (sortie de ``DirichletSystem.K_free``).
        K_g : csr_matrix, shape (n_free, n_free)
            Matrice de rigidité géométrique réduite aux DDL libres
            (sortie de ``DirichletSystem.reduce(K_g_global)``).
        n_modes : int
            Nombre de modes de flambage à extraire (par ordre de charge
            critique croissante).

        Returns
        -------
        lambda_cr : np.ndarray, shape (n_modes,)
            Multiplicateurs de charge critique, triés par ordre croissant.
            La i-ième charge critique est ``lambda_cr[i] × P_ref``.
        phi : np.ndarray, shape (n_free, n_modes)
            Modes de flambage (déplacements aux DDL libres uniquement).
            Utiliser ``DirichletSystem.recover_modes(phi)`` pour obtenir
            le vecteur complet de taille n_dof.

        Notes
        -----
        Implémentation : eigsh(K, M=−K_g, sigma=0, which='LM') qui trouve
        les λ les plus petits de K·φ = λ·(−K_g)·φ via shift-invert.

        Requiert −K_g semi-définie positive (chargement compressif pur).

        Examples
        --------
        >>> solver = BucklingSolver()
        >>> lambda_cr, phi_free = solver.solve(K_free, K_g_free, n_modes=3)
        >>> phi = ds.recover_modes(phi_free)   # modes complets
        >>> print(f"1ère charge critique : {lambda_cr[0]:.2f} × P_ref")
        """
        return self.backend.solve_buckling(K, K_g, n_modes)

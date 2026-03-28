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

"""Analyse modale — utilitaires de haut niveau.

Ce module fournit :
- ``run_modal`` : assemblage + CL + solveur eigsh en un appel
- ``lumped_mass`` : conversion masse consistante → diagonale (row-sum)
- ``ModalResult`` : conteneur des résultats

Le solveur sous-jacent est ``ModalSolver`` (femsolver.core.solver),
qui utilise ``scipy.sparse.linalg.eigsh`` (algorithme de Lanczos).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix, diags

from femsolver.core.assembler import Assembler
from femsolver.core.boundary import apply_dirichlet
from femsolver.core.mesh import BoundaryConditions, Mesh
from femsolver.core.solver import ModalSolver


@dataclass(frozen=True)
class ModalResult:
    """Résultats d'une analyse modale.

    Attributes
    ----------
    freqs : np.ndarray, shape (n_modes,)
        Fréquences propres [Hz], triées par ordre croissant.
    omega : np.ndarray, shape (n_modes,)
        Pulsations propres ω_n [rad/s].
    modes : np.ndarray, shape (n_dof, n_modes)
        Vecteurs propres (colonnes), M-normalisés : φᵀ M φ = I.
    n_modes : int
        Nombre de modes extraits.
    """

    freqs: np.ndarray
    omega: np.ndarray
    modes: np.ndarray
    n_modes: int


def lumped_mass(M: csr_matrix) -> csr_matrix:
    """Convertit la matrice de masse consistante en masse condensée diagonale.

    Technique de condensation par somme des lignes (row-sum) :
    chaque DDL reçoit la somme de sa ligne. La masse totale est conservée.

    Parameters
    ----------
    M : csr_matrix
        Matrice de masse consistante symétrique (n_dof × n_dof).

    Returns
    -------
    M_lumped : csr_matrix
        Matrice de masse diagonale (n_dof × n_dof).

    Notes
    -----
    Pour une barre à 2 nœuds avec M_e = (ρAL/6)·[[2,1],[1,2]] :

        somme ligne 1 = ρAL/6·(2+1) = ρAL/2

    Chaque nœud reçoit la moitié de la masse de l'élément — cohérent
    avec une distribution uniforme de masse.

    La masse condensée surestime légèrement les fréquences propres
    (borne supérieure), contrairement à la masse consistante qui les
    sous-estime (borne inférieure).
    """
    row_sums = np.asarray(M.sum(axis=1)).ravel()
    return diags(row_sums, format="csr")


def run_modal(
    mesh: Mesh,
    bc: BoundaryConditions,
    n_modes: int = 5,
    use_lumped: bool = False,
) -> ModalResult:
    """Exécute une analyse modale complète.

    Assemble K et M, applique les conditions de Dirichlet sur K,
    puis résout le problème aux valeurs propres généralisé K φ = ω² M φ.

    Parameters
    ----------
    mesh : Mesh
        Maillage du modèle (nœuds, éléments, matériaux).
    bc : BoundaryConditions
        Conditions aux limites — seule la partie Dirichlet est utilisée.
    n_modes : int
        Nombre de modes propres à extraire (les plus basses fréquences).
    use_lumped : bool
        Si ``True``, utilise la masse condensée diagonale.
        Si ``False`` (défaut), utilise la masse consistante.

    Returns
    -------
    result : ModalResult
        Fréquences [Hz], pulsations [rad/s] et modes propres.

    Notes
    -----
    Les conditions de Dirichlet sont imposées à K par pénalisation :
    K[i,i] += α (α = K_max × 10¹⁵). Les DDL contraints acquièrent
    des fréquences parasites très élevées (f ∝ √α) qui n'affectent pas
    les premiers modes extraits avec ``which="SM"``.

    La matrice de masse M n'est pas modifiée : les DDL contraints ont
    M[i,i] > 0 (masse physique) et K[i,i] ≈ α → ω² ≈ α/M[i,i] ≫ 1,
    ce qui les envoie automatiquement en haut du spectre.
    """
    assembler = Assembler(mesh)
    K = assembler.assemble_stiffness()
    M = assembler.assemble_mass()

    F_dummy = np.zeros(mesh.n_dof)
    # penalty_factor réduit pour limiter le conditionnement de K :
    # α = K_max × 1e8  ≫  ω²_max_physique  (modes parasites à très haute fréquence)
    # mais α ≪ K_max × 1e15  → condition(K_bc) ≈ 1e8 au lieu de 1e23
    K_bc, _ = apply_dirichlet(K, F_dummy, mesh, bc, penalty_factor=1e8)

    if use_lumped:
        M = lumped_mass(M)

    freqs, modes = ModalSolver().solve(K_bc, M, n_modes=n_modes)
    omega = freqs * (2.0 * np.pi)
    return ModalResult(freqs=freqs, omega=omega, modes=modes, n_modes=n_modes)

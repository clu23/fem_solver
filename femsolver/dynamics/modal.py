"""Analyse modale — utilitaires de haut niveau.

Ce module fournit :
- ``run_modal`` : assemblage + CL + solveur eigsh en un appel
- ``lumped_mass`` : conversion masse consistante → diagonale (row-sum)
- ``ModalResult`` : conteneur des résultats

Le solveur sous-jacent est ``ModalSolver`` (femsolver.core.solver),
qui utilise ``scipy.sparse.linalg.eigsh`` (algorithme de Lanczos).

Stratégie d'imposition des conditions aux limites
-------------------------------------------------
On utilise la **méthode d'élimination vraie** :

1. ``apply_dirichlet(K, ...)`` (mode ``"elimination"``) produit un
   ``DirichletSystem`` qui expose ``K_free``, ``reduce_mass()`` et
   ``recover_modes()``.

2. ``K_free = K_bc[free, free]`` et ``M_free = M[free, free]`` sont
   les sous-matrices exactes des DDL libres.

3. On résout ``K_free φ = ω² M_free φ`` avec ``eigsh`` (Lanczos,
   shift-invert σ=0).

4. Les vecteurs propres sont reconstruits à taille n_dof :
   ``φ_full[free] = φ_free``, ``φ_full[constrained] = 0``.

**Pourquoi l'élimination vraie évite les modes parasites**

Avec la pénalisation ou le row-zero partiel (K[s,s]=1, M[s,s]≠0), le
spectre de ``(K, M)`` contient des valeurs propres parasites :

- *Pénalisation* : ω²_parasite = α / M[s,s] → très grande (haute fréquence),
  mais α doit être finement calibré (1e8 × max(K)) pour ne pas dégrader
  le conditionnement ni interférer avec les modes physiques.
- *Row-zero* : ω²_parasite = 1 / M[s,s] → peut tomber parmi les premiers
  modes physiques (basse fréquence ≈ 1 Hz pour l'acier).

Avec l'élimination vraie, K_free et M_free ne contiennent plus les DDL
bloqués : aucun mode parasite, spectre purement physique.
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
        Vecteurs propres (colonnes), M_free-normalisés : φᵀ M φ = I.
        Les DDL contraints ont une valeur nulle exacte.
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

    Pour l'analyse modale avec ``run_modal``, appliquer ``lumped_mass``
    **avant** de réduire M aux DDL libres (``ds.reduce_mass``).
    """
    row_sums = np.asarray(M.sum(axis=1)).ravel()
    return diags(row_sums, format="csr")


def run_modal(
    mesh: Mesh,
    bc: BoundaryConditions,
    n_modes: int = 5,
    use_lumped: bool = False,
) -> ModalResult:
    """Exécute une analyse modale complète avec élimination vraie des CL.

    Assemble K et M, réduit le système aux DDL libres par élimination vraie,
    puis résout le problème aux valeurs propres K_free φ = ω² M_free φ.

    Parameters
    ----------
    mesh : Mesh
        Maillage du modèle (nœuds, éléments, matériaux).
    bc : BoundaryConditions
        Conditions aux limites — seule la partie Dirichlet est utilisée.
    n_modes : int
        Nombre de modes propres à extraire (les plus basses fréquences).
    use_lumped : bool
        Si ``True``, utilise la masse condensée diagonale (row-sum).
        Si ``False`` (défaut), utilise la masse consistante.

    Returns
    -------
    result : ModalResult
        Fréquences [Hz], pulsations [rad/s] et modes propres (taille n_dof,
        zéros aux DDL contraints).

    Notes
    -----
    **Workflow interne** :

    .. code-block:: text

        K, M ← Assembler.assemble_stiffness(), assemble_mass()
        ds   ← apply_dirichlet(K, 0, mesh, bc)   [élimination row-zero]
        K_f  ← ds.K_free   [K_bc[free, free] = K_original[free, free]]
        M_f  ← ds.reduce_mass(M)   [M[free, free]]
        ω², φ_f ← eigsh(K_f, M=M_f, sigma=0, which="LM")   [Lanczos]
        φ   ← ds.recover_modes(φ_f)   [remettre à taille n_dof]

    Le spectre de (K_f, M_f) ne contient aucun mode parasite car les DDL
    contraints ont été supprimés de la matrice — pas d'artifice de pénalisation.

    Examples
    --------
    >>> result = run_modal(mesh, bc, n_modes=5)
    >>> print(result.freqs)   # [f1, f2, f3, f4, f5] Hz
    """
    assembler = Assembler(mesh)
    K = assembler.assemble_stiffness()
    M = assembler.assemble_mass()

    # Masse condensée appliquée avant la réduction (conservation de la masse)
    if use_lumped:
        M = lumped_mass(M)

    F_dummy = np.zeros(mesh.n_dof)
    # Élimination vraie : K_free = K[free, free], M_free = M[free, free]
    # Aucun mode parasite (DDL contraints supprimés du spectre).
    ds = apply_dirichlet(K, F_dummy, mesh, bc)   # method="elimination" par défaut
    K_free = ds.K_free
    M_free = ds.reduce_mass(M)

    freqs_free, phi_free = ModalSolver().solve(K_free, M_free, n_modes=n_modes)

    # Reconstruction des modes à la taille complète (zéros aux DDL contraints)
    phi_full = ds.recover_modes(phi_free)

    omega = freqs_free * (2.0 * np.pi)
    return ModalResult(
        freqs=freqs_free,
        omega=omega,
        modes=phi_full,
        n_modes=n_modes,
    )

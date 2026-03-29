"""Application des conditions aux limites de Dirichlet (élimination directe)."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from femsolver.core.mesh import BoundaryConditions, Mesh


def apply_dirichlet(
    K: csr_matrix,
    F: np.ndarray,
    mesh: Mesh,
    bc: BoundaryConditions,
    *,
    penalty_factor: float = 1e15,
) -> tuple[csr_matrix, np.ndarray]:
    """Applique les conditions de Dirichlet par la méthode de pénalisation.

    La méthode de pénalisation remplace K[i,i] par un grand nombre α
    et F[i] par α·u_i_imposé. Elle préserve la structure creuse et la
    symétrie de la matrice sans réorganiser le système.

    Parameters
    ----------
    K : csr_matrix, shape (n_dof, n_dof)
        Matrice de rigidité globale assemblée.
    F : np.ndarray, shape (n_dof,)
        Vecteur de forces.
    mesh : Mesh
        Maillage (pour calculer les indices de DDL globaux).
    bc : BoundaryConditions
        Conditions aux limites (seule la partie Dirichlet est utilisée).

    Returns
    -------
    K_bc : csr_matrix
        Matrice de rigidité modifiée (même structure creuse).
    F_bc : np.ndarray
        Vecteur de forces modifié.

    Notes
    -----
    Coefficient de pénalisation : α = max(|K|) · ``penalty_factor``.
    La valeur par défaut 1e15 garantit ~1e-15 de précision pour l'analyse
    statique. Pour l'analyse modale, utiliser ``penalty_factor=1e8`` afin
    de limiter le conditionnement de K (critère : α >> ω²_max_physique).

    Attention : la méthode de pénalisation peut dégrader le conditionnement de K.
    Pour des systèmes très mal conditionnés, préférer l'élimination directe.
    L'élimination directe sera implémentée si nécessaire.

    Examples
    --------
    >>> K_bc, F_bc = apply_dirichlet(K, F, mesh, bc)
    >>> u = spsolve(K_bc, F_bc)
    """
    K_bc = K.tolil()
    F_bc = F.copy()

    alpha = float(K.data.max()) * penalty_factor if K.nnz > 0 else penalty_factor

    for node_id, dof_values in bc.dirichlet.items():
        for dof, value in dof_values.items():
            global_dof = mesh.dpn * node_id + dof
            K_bc[global_dof, global_dof] = alpha
            F_bc[global_dof] = alpha * value

    return K_bc.tocsr(), F_bc

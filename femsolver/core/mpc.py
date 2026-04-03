"""Contraintes multi-points (MPC) — deux méthodes d'imposition.

Méthode d'élimination
---------------------
Pour chaque contrainte ``a_s · u_s + a_m · u_m = β`` (1 esclave, 1 maître) :

    u_s = α · u_m + β̃    avec α = −a_m/a_s, β̃ = β/a_s

On construit une matrice de transformation **T** (n_dof × n_réduit) telle que :

    u = T · û + g

puis on résout le système réduit :

    K_red = Tᵀ · K · T
    F_red = Tᵀ · (F − K · g)
    → û = K_red⁻¹ F_red
    → u  = T · û + g

**Avantages** : système plus petit, K_red reste SPD (si K l'est).
**Limites** : 1 maître par esclave, pas de chaînage.

Multiplicateurs de Lagrange
----------------------------
Pour des contraintes générales ``Σ aᵢ·u[node_i, dof_i] = β`` :

    [K   Cᵀ] [u]   [F]
    [C   0 ] [λ] = [g]

**Avantages** : N termes couplés, λ = forces de réaction de contrainte.
**Limites** : système augmenté indéfini (zéros sur la diagonale des λ),
solveur direct requis (spsolve).
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, bmat

from femsolver.core.mesh import MPCConstraint, Mesh


# ---------------------------------------------------------------------------
# Méthode d'élimination
# ---------------------------------------------------------------------------


def apply_mpc_elimination(
    K: csr_matrix,
    F: np.ndarray,
    mesh: Mesh,
    constraints: tuple[MPCConstraint, ...],
) -> tuple[csr_matrix, np.ndarray, csr_matrix, np.ndarray, list[int]]:
    """Applique les MPC par élimination directe.

    Chaque contrainte ``MPCConstraint`` doit avoir **exactement 2 termes**.
    Le **premier terme** est l'esclave (DDL éliminé).

    Construction de la transformation u = T · û + g :

    - T est une matrice creuse (n_dof × n_réduit).
    - Colonne j de T → DDL libre ``free_dofs[j]``.
    - Ligne s de T → α au niveau de la colonne maître (couplage MPC).
    - g[s] = β̃ (décalage constant).

    Parameters
    ----------
    K : csr_matrix, shape (n_dof, n_dof)
        Matrice de rigidité (après application de Dirichlet recommandée).
    F : np.ndarray, shape (n_dof,)
        Vecteur de forces (après application de Dirichlet recommandée).
    mesh : Mesh
        Maillage (pour la correspondance nœud → DDL global).
    constraints : tuple[MPCConstraint, ...]
        Contraintes MPC à 2 termes chacune.

    Returns
    -------
    K_red : csr_matrix, shape (n_réduit, n_réduit)
        Matrice de rigidité réduite ``Tᵀ·K·T``.
    F_red : np.ndarray, shape (n_réduit,)
        Vecteur de forces réduit ``Tᵀ·(F − K·g)``.
    T : csr_matrix, shape (n_dof, n_réduit)
        Matrice de transformation.
    g : np.ndarray, shape (n_dof,)
        Vecteur de décalage (partie constante).
    slave_dofs : list[int]
        Indices globaux des DDL esclaves éliminés.

    Raises
    ------
    ValueError
        Si une contrainte n'a pas exactement 2 termes.
        Si un DDL esclave apparaît comme maître dans une autre contrainte
        (chaînage non supporté).
        Si le maître n'est pas dans les DDL libres (après exclusion des esclaves).

    Examples
    --------
    Liaison rigide u_x(1) = u_x(3) (Bar2D, n_dim=2) :

    >>> c = MPCConstraint(terms=((1, 0, 1.0), (3, 0, -1.0)), rhs=0.0)
    >>> K_red, F_red, T, g, slave_dofs = apply_mpc_elimination(K_bc, F_bc, mesh, (c,))
    >>> u_red = StaticSolver().solve(K_red, F_red)
    >>> u = recover_mpc(u_red, T, g)
    """
    n_dof = K.shape[0]

    slave_dofs: list[int] = []
    master_info: list[tuple[int, float, float]] = []   # (master_global, alpha, beta_tilde)

    slave_set: set[int] = set()

    for c in constraints:
        if len(c.terms) != 2:
            raise ValueError(
                f"apply_mpc_elimination : chaque contrainte doit avoir exactement "
                f"2 termes (esclave + maître), reçu {len(c.terms)} termes."
            )
        s_node, s_ldof, a_s = c.terms[0]
        m_node, m_ldof, a_m = c.terms[1]

        s_global = mesh.dpn * s_node + s_ldof
        m_global = mesh.dpn * m_node + m_ldof

        if a_s == 0.0:
            raise ValueError(
                f"Coefficient esclave nul pour la contrainte sur le nœud "
                f"{s_node}, DDL {s_ldof}."
            )

        alpha = -a_m / a_s          # u_s = alpha * u_m + beta_tilde
        beta_tilde = c.rhs / a_s

        slave_dofs.append(s_global)
        slave_set.add(s_global)
        master_info.append((m_global, alpha, beta_tilde))

    # Vérification : pas de chaînage (un esclave ne peut pas être maître)
    master_set = {m for m, _, _ in master_info}
    chained = slave_set & master_set
    if chained:
        raise ValueError(
            f"apply_mpc_elimination : chaînage détecté — les DDL {chained} "
            "sont à la fois esclaves et maîtres.  Réordonner les contraintes "
            "ou utiliser apply_mpc_lagrange."
        )

    # DDL libres (hors esclaves)
    free_dofs: list[int] = [d for d in range(n_dof) if d not in slave_set]
    free_index: dict[int, int] = {d: j for j, d in enumerate(free_dofs)}
    n_red = len(free_dofs)

    # Construction de T (n_dof × n_red)
    rows_T: list[int] = []
    cols_T: list[int] = []
    vals_T: list[float] = []

    # Partie identité : DDL libres → leur propre colonne
    for j, d in enumerate(free_dofs):
        rows_T.append(d)
        cols_T.append(j)
        vals_T.append(1.0)

    # Couplage MPC : ligne esclave, colonne du maître
    g = np.zeros(n_dof)
    for s_global, (m_global, alpha, beta_tilde) in zip(slave_dofs, master_info):
        g[s_global] = beta_tilde
        if m_global not in free_index:
            raise ValueError(
                f"Le maître (DDL global {m_global}) n'est pas un DDL libre "
                f"(il est peut-être esclave dans une autre contrainte)."
            )
        rows_T.append(s_global)
        cols_T.append(free_index[m_global])
        vals_T.append(alpha)

    T = coo_matrix(
        (vals_T, (rows_T, cols_T)), shape=(n_dof, n_red)
    ).tocsr()

    # Système réduit
    Kg = K @ g
    K_red: csr_matrix = (T.T @ K @ T).tocsr()
    F_red: np.ndarray = T.T @ (F - Kg)

    return K_red, F_red, T, g, slave_dofs


def recover_mpc(
    u_red: np.ndarray,
    T: csr_matrix,
    g: np.ndarray,
) -> np.ndarray:
    """Reconstruit le vecteur de déplacements complet après élimination MPC.

    Parameters
    ----------
    u_red : np.ndarray, shape (n_réduit,)
        Solution du système réduit.
    T : csr_matrix, shape (n_dof, n_réduit)
        Matrice de transformation retournée par :func:`apply_mpc_elimination`.
    g : np.ndarray, shape (n_dof,)
        Vecteur de décalage retourné par :func:`apply_mpc_elimination`.

    Returns
    -------
    u : np.ndarray, shape (n_dof,)
        Déplacements complets incluant les DDL esclaves.

    Notes
    -----
    ``u = T @ u_red + g``.  Les DDL esclaves vérifient exactement la
    contrainte MPC à la précision machine.

    Examples
    --------
    >>> K_red, F_red, T, g, _ = apply_mpc_elimination(K, F, mesh, constraints)
    >>> u_red = StaticSolver().solve(K_red, F_red)
    >>> u = recover_mpc(u_red, T, g)
    """
    return T @ u_red + g


# ---------------------------------------------------------------------------
# Méthode des multiplicateurs de Lagrange
# ---------------------------------------------------------------------------


def apply_mpc_lagrange(
    K: csr_matrix,
    F: np.ndarray,
    mesh: Mesh,
    constraints: tuple[MPCConstraint, ...],
) -> tuple[csr_matrix, np.ndarray]:
    """Applique les MPC par multiplicateurs de Lagrange.

    Construit le système augmenté (selle) :

    .. math::

        \\begin{bmatrix} K & C^T \\\\ C & 0 \\end{bmatrix}
        \\begin{bmatrix} u \\\\ \\lambda \\end{bmatrix}
        = \\begin{bmatrix} F \\\\ g \\end{bmatrix}

    où ``C`` est la matrice de contrainte (n_c × n_dof) et ``g`` le second membre.

    Parameters
    ----------
    K : csr_matrix, shape (n_dof, n_dof)
        Matrice de rigidité (après application de Dirichlet recommandée).
    F : np.ndarray, shape (n_dof,)
        Vecteur de forces (après application de Dirichlet recommandée).
    mesh : Mesh
        Maillage.
    constraints : tuple[MPCConstraint, ...]
        Contraintes MPC (nombre de termes quelconque ≥ 2).

    Returns
    -------
    K_aug : csr_matrix, shape (n_dof + n_c, n_dof + n_c)
        Système augmenté.  Matrice indéfinie (zéros sur la diagonale λ).
    F_aug : np.ndarray, shape (n_dof + n_c,)
        Second membre augmenté ``[F, g]``.

    Notes
    -----
    Extraction de la solution après ``spsolve(K_aug, F_aug)`` :

    .. code-block:: python

        result = spsolve(K_aug, F_aug)
        u       = result[:mesh.n_dof]
        lambdas = result[mesh.n_dof:]

    Les multiplicateurs ``λ_k`` représentent les forces de réaction nécessaires
    pour imposer la k-ième contrainte [N ou N·m selon le type de DDL].

    Examples
    --------
    Raccord 3 DDL entre deux segments de poutre :

    >>> constraints = (
    ...     MPCConstraint(((1, 0, 1.0), (2, 0, -1.0))),  # u_x
    ...     MPCConstraint(((1, 1, 1.0), (2, 1, -1.0))),  # u_y
    ...     MPCConstraint(((1, 2, 1.0), (2, 2, -1.0))),  # θ
    ... )
    >>> K_aug, F_aug = apply_mpc_lagrange(K_bc, F_bc, mesh, constraints)
    >>> result = spsolve(K_aug, F_aug)
    >>> u = result[:mesh.n_dof]
    """
    n_dof = K.shape[0]
    n_c = len(constraints)

    # Construction de C (n_c × n_dof)
    rows_C: list[int] = []
    cols_C: list[int] = []
    vals_C: list[float] = []
    g = np.zeros(n_c)

    for k, c in enumerate(constraints):
        for node_id, local_dof, coeff in c.terms:
            global_dof = mesh.dpn * node_id + local_dof
            rows_C.append(k)
            cols_C.append(global_dof)
            vals_C.append(float(coeff))
        g[k] = c.rhs

    C = coo_matrix(
        (vals_C, (rows_C, cols_C)), shape=(n_c, n_dof)
    ).tocsr()

    # Système augmenté par blocs
    zero_block = coo_matrix((n_c, n_c), dtype=float).tocsr()
    K_aug: csr_matrix = bmat(
        [[K, C.T], [C, zero_block]], format="csr"
    )
    F_aug = np.concatenate([F, g])

    return K_aug, F_aug

"""Application des conditions aux limites de Dirichlet.

``DirichletSystem`` — objet retourné par ``apply_dirichlet``
-------------------------------------------------------------
Encapsule le résultat et fournit trois interfaces :

1. **Rétrocompatibilité** via ``__iter__`` :

   .. code-block:: python

       K_bc, F_bc = apply_dirichlet(K, F, mesh, bc)
       u = StaticSolver().solve(K_bc, F_bc)   # u plein (n_dof)

   K_bc est de taille n_dof × n_dof (méthode row-zero) ; la résolution
   donne directement u de taille n_dof, avec u[s] = ū_s exact.

2. **Élimination vraie** (modal, MPC imbriqué…) :

   .. code-block:: python

       ds = apply_dirichlet(K, F, mesh, bc)
       K_free, F_free = ds.K_free, ds.F_free   # taille n_free
       M_free = ds.reduce_mass(M)
       omega_sq, phi_f = eigsh(K_free, M=M_free, sigma=0, which="LM")
       phi = ds.recover_modes(phi_f)            # taille n_dof
       u   = ds.recover(solve(K_free, F_free))  # taille n_dof

3. **Propriété** ``free_dofs`` : indices des DDL libres (toujours disponible).

Deux méthodes d'imposition
---------------------------
``"elimination"`` (défaut)
    Row-zero : F[j] −= K[j,s]·ū, puis K[s,:]=K[:,s]=0, K[s,s]=1, F[s]=ū.
    Exact (u[s]=ū_s), conditionnement inchangé.  Les propriétés ``K_free``,
    ``F_free``, ``reduce_mass``, ``recover``, ``recover_modes`` sont disponibles.

``"penalty"``
    K[s,s] += α·max(K), F[s] = α·max(K)·ū.  Conservé pour rétrocompatibilité.
    ``K_free`` / ``reduce_mass`` lèvent ``NotImplementedError`` car le terme de
    couplage K[f,s] n'est pas soustrait de F[f].
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from femsolver.core.mesh import BoundaryConditions, Mesh


# ---------------------------------------------------------------------------
# DirichletSystem
# ---------------------------------------------------------------------------


class DirichletSystem:
    """Résultat de l'application des conditions de Dirichlet.

    Produit par :func:`apply_dirichlet`.  Peut être dépaqueté comme un
    2-uplet ``(K_bc, F_bc)`` pour la rétrocompatibilité, ou utilisé via
    ses propriétés pour l'analyse modale avec élimination vraie.

    Parameters
    ----------
    K_bc : csr_matrix, shape (n_dof, n_dof)
        Matrice modifiée par la méthode choisie.
    F_bc : np.ndarray, shape (n_dof,)
        Vecteur modifié.
    free_dofs : np.ndarray, shape (n_free,)
        Indices globaux des DDL libres (non bloqués par Dirichlet),
        dans l'ordre croissant.
    u_prescribed : np.ndarray, shape (n_dof,)
        Vecteur complet avec ū_s aux DDL contraints, 0 ailleurs.
    method : str
        ``"elimination"`` ou ``"penalty"``.

    Attributes
    ----------
    free_dofs : np.ndarray
        Indices des DDL libres.
    K_free : csr_matrix
        Sous-matrice (n_free × n_free) — élimination uniquement.
    F_free : np.ndarray
        Sous-vecteur (n_free,) — élimination uniquement.
    """

    __slots__ = ("_K_bc", "_F_bc", "_free_dofs", "_u_prescribed", "_method")

    def __init__(
        self,
        K_bc: csr_matrix,
        F_bc: np.ndarray,
        free_dofs: np.ndarray,
        u_prescribed: np.ndarray,
        method: str,
    ) -> None:
        self._K_bc = K_bc
        self._F_bc = F_bc
        self._free_dofs = free_dofs
        self._u_prescribed = u_prescribed
        self._method = method

    # ------------------------------------------------------------------
    # Rétrocompatibilité : K_bc, F_bc = apply_dirichlet(...)
    # ------------------------------------------------------------------

    def __iter__(self):
        """Permet le dépaquetage ``K_bc, F_bc = apply_dirichlet(...)``."""
        yield self._K_bc
        yield self._F_bc

    # ------------------------------------------------------------------
    # Accès rapide au système plein (rétrocompatibilité)
    # ------------------------------------------------------------------

    @property
    def free_dofs(self) -> np.ndarray:
        """Indices globaux des DDL libres, triés par ordre croissant."""
        return self._free_dofs

    # ------------------------------------------------------------------
    # Sous-système réduit (méthode d'élimination uniquement)
    # ------------------------------------------------------------------

    def _require_elimination(self) -> None:
        if self._method != "elimination":
            raise NotImplementedError(
                "K_free, F_free et reduce_mass() ne sont disponibles "
                "qu'avec method='elimination'.\n"
                "Pour l'analyse modale, n'utilisez pas method='penalty'."
            )

    @property
    def K_free(self) -> csr_matrix:
        """Sous-matrice de rigidité réduite aux DDL libres.

        .. math:: K_{\\text{free}} = K_{\\text{bc}}[f, f]

        où ``f = free_dofs``.  Valide car la méthode row-zero laisse
        ``K_bc[f, f'] = K_original[f, f']`` inchangé.

        Returns
        -------
        csr_matrix, shape (n_free, n_free)
        """
        self._require_elimination()
        f = self._free_dofs
        return self._K_bc[f, :][:, f].tocsr()

    @property
    def F_free(self) -> np.ndarray:
        """Sous-vecteur de forces réduit aux DDL libres.

        .. math:: F_{\\text{free}} = F_{\\text{bc}}[f]

        Valide car la méthode row-zero a soustrait la contribution des DDL
        bloqués : ``F_bc[f] = F_original[f] − K_{fc} · ū_c``.

        Returns
        -------
        np.ndarray, shape (n_free,)
        """
        self._require_elimination()
        return self._F_bc[self._free_dofs].copy()

    def reduce_mass(self, M: csr_matrix) -> csr_matrix:
        """Extrait la sous-matrice de masse pour les DDL libres.

        .. math:: M_{\\text{free}} = M[f, f]

        Les DDL contraints ont une vitesse nulle (déplacement imposé) :
        leur masse n'intervient pas dans le problème aux valeurs propres.
        La sous-matrice M_free est définie positive si M l'est.

        Parameters
        ----------
        M : csr_matrix, shape (n_dof, n_dof)
            Matrice de masse (consistante ou condensée).

        Returns
        -------
        csr_matrix, shape (n_free, n_free)

        Notes
        -----
        Pour la masse condensée (lumped), appliquer ``lumped_mass(M)``
        **avant** de passer M à cette méthode.

        Raises
        ------
        NotImplementedError
            Si ``method != "elimination"``.
        """
        self._require_elimination()
        f = self._free_dofs
        return M[f, :][:, f].tocsr()

    # ------------------------------------------------------------------
    # Reconstruction de la solution complète
    # ------------------------------------------------------------------

    def recover(self, u_free: np.ndarray) -> np.ndarray:
        """Reconstruit le vecteur de déplacements complet depuis u_free.

        .. math:: u[f] = u_{\\text{free}},\\quad u[s] = \\bar{u}_s

        Parameters
        ----------
        u_free : np.ndarray, shape (n_free,)
            Déplacements aux DDL libres (solution de K_free · u_free = F_free).

        Returns
        -------
        u : np.ndarray, shape (n_dof,)
            Déplacements complets.

        Notes
        -----
        Pour la voie rétrocompatible (``solve(K_bc, F_bc)``), la solution
        est déjà de taille n_dof et u[s] = ū_s exact — pas besoin de
        ``recover()``.  Cette méthode est nécessaire uniquement quand on
        résout sur le sous-système réduit ``K_free · u_free = F_free``.
        """
        u = self._u_prescribed.copy()
        u[self._free_dofs] = u_free
        return u

    def recover_modes(self, phi_free: np.ndarray) -> np.ndarray:
        """Reconstruit les vecteurs propres complets depuis les modes réduits.

        Les DDL contraints ont un déplacement modal nul (encastrement).

        Parameters
        ----------
        phi_free : np.ndarray, shape (n_free, n_modes)
            Vecteurs propres aux DDL libres.

        Returns
        -------
        phi : np.ndarray, shape (n_dof, n_modes)
            Vecteurs propres complets (zéros aux DDL contraints).
        """
        n_dof = len(self._u_prescribed)
        phi = np.zeros((n_dof, phi_free.shape[1]))
        phi[self._free_dofs] = phi_free
        return phi


# ---------------------------------------------------------------------------
# Fonction principale
# ---------------------------------------------------------------------------


def apply_dirichlet(
    K: csr_matrix,
    F: np.ndarray,
    mesh: Mesh,
    bc: BoundaryConditions,
    *,
    method: str = "elimination",
    penalty_factor: float = 1e15,
) -> DirichletSystem:
    """Applique les conditions de Dirichlet sur le système K·u = F.

    Parameters
    ----------
    K : csr_matrix, shape (n_dof, n_dof)
        Matrice de rigidité globale assemblée.
    F : np.ndarray, shape (n_dof,)
        Vecteur de forces nodales.
    mesh : Mesh
        Maillage (pour le calcul des indices de DDL globaux).
    bc : BoundaryConditions
        Conditions aux limites (seule la partie ``dirichlet`` est utilisée).
    method : {"elimination", "penalty"}
        Méthode d'imposition.

        ``"elimination"`` (défaut)
            Row-zero exact.  Fournit ``K_free``, ``reduce_mass``,
            ``recover``, ``recover_modes`` pour l'élimination vraie.

        ``"penalty"``
            Pénalisation approchée.  Rétrocompatibilité et cas où
            la structure creuse de K doit rester inchangée.

    penalty_factor : float
        Coefficient de pénalisation α (ignoré si ``method="elimination"``).

    Returns
    -------
    DirichletSystem
        Objet encapsulant K_bc, F_bc et les utilitaires de réduction.
        Peut être dépaqueté : ``K_bc, F_bc = apply_dirichlet(...)``.

    Raises
    ------
    ValueError
        Si ``method`` n'est pas reconnu.

    Examples
    --------
    Interface rétrocompatible (statique) :

    >>> K_bc, F_bc = apply_dirichlet(K, F, mesh, bc)
    >>> u = StaticSolver().solve(K_bc, F_bc)

    Élimination vraie (modal, sans modes parasites) :

    >>> ds = apply_dirichlet(K, F_dummy, mesh, bc)
    >>> omega_sq, phi_f = eigsh(ds.K_free, M=ds.reduce_mass(M), sigma=0)
    >>> phi = ds.recover_modes(phi_f)
    """
    if method == "elimination":
        return _build_elimination(K, F, mesh, bc)
    elif method == "penalty":
        return _build_penalty(K, F, mesh, bc, penalty_factor)
    else:
        raise ValueError(
            f"Méthode Dirichlet inconnue : {method!r}. "
            "Choisir 'elimination' ou 'penalty'."
        )


# ---------------------------------------------------------------------------
# Implémentations internes
# ---------------------------------------------------------------------------


def _collect_constrained(mesh: Mesh, bc: BoundaryConditions) -> dict[int, float]:
    """Retourne {global_dof: valeur_imposée} depuis bc.dirichlet."""
    constrained: dict[int, float] = {}
    for node_id, dof_values in bc.dirichlet.items():
        for local_dof, value in dof_values.items():
            constrained[mesh.dpn * node_id + local_dof] = float(value)
    return constrained


def _build_elimination(
    K: csr_matrix,
    F: np.ndarray,
    mesh: Mesh,
    bc: BoundaryConditions,
) -> DirichletSystem:
    """Row-zero elimination (méthode exacte, taille inchangée).

    Algorithme
    ----------
    1. Pour chaque DDL bloqué s avec valeur ū (sur K original) :
       ``F[j] −= K[j, s] · ū``  pour tout j — soustrait la contribution.
    2. ``K[s, :] = 0``,  ``K[:, s] = 0``,  ``K[s, s] = 1``,  ``F[s] = ū``.

    **Propriété clé** : après l'étape 2,
    ``K_bc[f, f'] = K_original[f, f']``  (DDL libres f, f' inchangés),
    ``F_bc[f]    = F_f − K_fc · ū_c``   (contribution soustraite à l'étape 1).
    Ces deux invariants permettent d'extraire le sous-système libre
    ``K_free = K_bc[free, free]`` et ``F_free = F_bc[free]`` sans
    re-calculer quoi que ce soit.
    """
    constrained = _collect_constrained(mesh, bc)

    n_dof = K.shape[0]
    u_prescribed = np.zeros(n_dof)
    for s, u_bar in constrained.items():
        u_prescribed[s] = u_bar

    free_dofs = np.array(
        sorted(set(range(n_dof)) - set(constrained.keys())), dtype=int
    )

    if not constrained:
        return DirichletSystem(
            K_bc=K.tocsr(),
            F_bc=F.copy(),
            free_dofs=free_dofs,
            u_prescribed=u_prescribed,
            method="elimination",
        )

    K_csr = K.tocsr()
    F_bc = F.copy()

    # Étape 1 : soustrait la contribution de chaque DDL bloqué de F.
    # On opère sur K_csr original (avant zeroing) pour que l'ordre de
    # traitement des contraintes n'ait pas d'importance.
    for s, u_bar in constrained.items():
        if u_bar != 0.0:
            F_bc -= K_csr.getcol(s).toarray().ravel() * u_bar

    # Étape 2 : zeroing lignes/colonnes + équation triviale K[s,s]=1, F[s]=ū.
    K_lil = K_csr.tolil()
    for s, u_bar in constrained.items():
        K_lil[s, :] = 0.0
        K_lil[:, s] = 0.0
        K_lil[s, s] = 1.0
        F_bc[s] = u_bar          # écrase la modification de l'étape 1

    return DirichletSystem(
        K_bc=K_lil.tocsr(),
        F_bc=F_bc,
        free_dofs=free_dofs,
        u_prescribed=u_prescribed,
        method="elimination",
    )


def _build_penalty(
    K: csr_matrix,
    F: np.ndarray,
    mesh: Mesh,
    bc: BoundaryConditions,
    penalty_factor: float,
) -> DirichletSystem:
    """Pénalisation (rétrocompatibilité).

    Note : ``K_free``, ``F_free`` et ``reduce_mass()`` lèvent
    ``NotImplementedError`` en mode pénalisation car ``F_bc[f]`` ne
    contient pas la soustraction de K_fc·ū_c.
    """
    constrained = _collect_constrained(mesh, bc)

    n_dof = K.shape[0]
    u_prescribed = np.zeros(n_dof)
    for s, u_bar in constrained.items():
        u_prescribed[s] = u_bar

    free_dofs = np.array(
        sorted(set(range(n_dof)) - set(constrained.keys())), dtype=int
    )

    K_lil = K.tolil()
    F_bc = F.copy()
    alpha = float(K.data.max()) * penalty_factor if K.nnz > 0 else penalty_factor

    for s, u_bar in constrained.items():
        K_lil[s, s] = alpha
        F_bc[s] = alpha * u_bar

    return DirichletSystem(
        K_bc=K_lil.tocsr(),
        F_bc=F_bc,
        free_dofs=free_dofs,
        u_prescribed=u_prescribed,
        method="penalty",
    )

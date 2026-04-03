"""Tests unitaires pour l'analyse modale — validation par solution analytique.

Cas de référence : barre en traction longitudinale
--------------------------------------------------
Une barre de longueur L, section A, matériau (E, ρ) est modélisée par N
éléments Bar2D alignés horizontalement.

Barre encastrée–encastrée (fixed–fixed)
    f_n = n · c / (2L),    n = 1, 2, 3, …
    c = √(E/ρ)  [vitesse du son dans le matériau]

Barre encastrée–libre (fixed–free)
    f_n = (2n−1) · c / (4L),    n = 1, 2, 3, …

Note sur la modélisation 2D
---------------------------
Bar2D a 2 DDL par nœud (ux, uy). Pour isoler les modes longitudinaux :
- on impose uy = 0 sur tous les nœuds (CL Dirichlet)
- ces DDL reçoivent une rigidité fictive K_penalisé très grande et
  leurs modes parasites apparaissent à très haute fréquence (hors
  de la plage d'intérêt).
"""

from __future__ import annotations

import numpy as np
import pytest

from femsolver.core.assembler import Assembler
from femsolver.core.boundary import DirichletSystem, apply_dirichlet
from femsolver.core.material import ElasticMaterial
from femsolver.core.mesh import BoundaryConditions, ElementData, Mesh
from femsolver.dynamics.modal import ModalResult, lumped_mass, run_modal
from femsolver.elements.bar2d import Bar2D


# ---------------------------------------------------------------------------
# Données communes
# ---------------------------------------------------------------------------

E = 210e9      # Pa
RHO = 7800.0   # kg/m³
A = 1e-4       # m²
L = 1.0        # m
C = float(np.sqrt(E / RHO))   # vitesse du son ≈ 5189 m/s

MAT = ElasticMaterial(E=E, nu=0.3, rho=RHO)


def _fixed_fixed_bar(n_elem: int) -> tuple[Mesh, BoundaryConditions]:
    """Barre horizontale encastrée–encastrée en n_elem éléments Bar2D.

    DDL contraints :
    - ux = 0 aux nœuds 0 et n_elem (extrémités encastrées)
    - uy = 0 sur tous les nœuds (supprime les modes transversaux parasites)
    """
    n_nodes = n_elem + 1
    nodes = np.column_stack([np.linspace(0.0, L, n_nodes),
                             np.zeros(n_nodes)])
    elements = tuple(
        ElementData(Bar2D, (i, i + 1), MAT, {"area": A})
        for i in range(n_elem)
    )
    mesh = Mesh(nodes=nodes, elements=elements, n_dim=2)

    dirichlet: dict[int, dict[int, float]] = {}
    for i in range(n_nodes):
        dirichlet[i] = {1: 0.0}        # uy = 0 partout
    dirichlet[0][0] = 0.0              # ux = 0 à gauche
    dirichlet[n_elem][0] = 0.0         # ux = 0 à droite

    bc = BoundaryConditions(dirichlet=dirichlet, neumann={})
    return mesh, bc


def _fixed_free_bar(n_elem: int) -> tuple[Mesh, BoundaryConditions]:
    """Barre horizontale encastrée–libre en n_elem éléments Bar2D."""
    n_nodes = n_elem + 1
    nodes = np.column_stack([np.linspace(0.0, L, n_nodes),
                             np.zeros(n_nodes)])
    elements = tuple(
        ElementData(Bar2D, (i, i + 1), MAT, {"area": A})
        for i in range(n_elem)
    )
    mesh = Mesh(nodes=nodes, elements=elements, n_dim=2)

    dirichlet: dict[int, dict[int, float]] = {}
    for i in range(n_nodes):
        dirichlet[i] = {1: 0.0}        # uy = 0 partout
    dirichlet[0][0] = 0.0              # ux = 0 à gauche seulement

    bc = BoundaryConditions(dirichlet=dirichlet, neumann={})
    return mesh, bc


# ---------------------------------------------------------------------------
# Tests de lumped_mass
# ---------------------------------------------------------------------------


class TestLumpedMass:
    """Vérification de la condensation de la matrice de masse."""

    def _assemble_M(self, n_elem: int):
        from femsolver.core.assembler import Assembler
        mesh, _ = _fixed_fixed_bar(n_elem)
        return Assembler(mesh).assemble_mass()

    def test_total_mass_conserved(self) -> None:
        """La masse totale est la même avant et après condensation.

        Masse totale = ρ·A·L.
        """
        M = self._assemble_M(n_elem=10)
        M_lump = lumped_mass(M)
        total_consistent = M.diagonal().sum()  # ≠ masse totale pour consistante
        total_lumped = M_lump.diagonal().sum()
        # Les deux doivent donner la même somme des lignes = masse totale
        np.testing.assert_allclose(total_lumped, RHO * A * L * 2,
                                   rtol=1e-12,
                                   err_msg="Masse totale non conservée après condensation")

    def test_lumped_is_diagonal(self) -> None:
        """M_lumped est diagonale : tous les termes hors-diag sont nuls."""
        M = self._assemble_M(n_elem=5)
        M_lump = lumped_mass(M)
        # Extraire la partie hors-diagonale
        import scipy.sparse as sp
        off_diag = M_lump - sp.diags(M_lump.diagonal())
        assert off_diag.nnz == 0

    def test_lumped_diagonal_equals_row_sums(self) -> None:
        """Diagonale de M_lumped = sommes des lignes de M_consistante."""
        M = self._assemble_M(n_elem=8)
        M_lump = lumped_mass(M)
        row_sums = np.asarray(M.sum(axis=1)).ravel()
        np.testing.assert_allclose(M_lump.diagonal(), row_sums, rtol=1e-14)

    def test_lumped_positive_definite(self) -> None:
        """Tous les termes diagonaux de M_lumped sont strictement positifs."""
        M = self._assemble_M(n_elem=6)
        M_lump = lumped_mass(M)
        assert np.all(M_lump.diagonal() > 0)


# ---------------------------------------------------------------------------
# Tests fréquences propres — barre encastrée–encastrée
# ---------------------------------------------------------------------------


class TestFixedFixedBarModes:
    """Fréquences longitudinales d'une barre encastrée–encastrée.

    Solution analytique : f_n = n · c / (2L),  n = 1, 2, 3, …
    avec c = √(E/ρ) ≈ 5189 m/s pour l'acier.

    Comportement de convergence (barre 1D, éléments linéaires) :
    - Masse consistante : surestime les fréquences — converge par le haut
      ω²_n_FEM = (12c²/h²) · sin²(nπh/(2L)) / (2+cos(nπh/L)) ≥ ω²_n_exact
    - Masse condensée (row-sum) : sous-estime les fréquences — converge par le bas
      ω²_n_FEM = (4c²/h²) · sin²(nπh/(2L)) ≤ ω²_n_exact

    Référence : Meirovitch, « Fundamentals of Vibrations », §7.3 ;
                Hughes, « The FEM », §9.3 (tableaux d'erreur).
    """

    N_ELEM = 40   # 40 éléments → erreur < 0.7 % sur les 5 premiers modes

    @staticmethod
    def _analytical(n_modes: int) -> np.ndarray:
        return np.array([(n * C) / (2.0 * L) for n in range(1, n_modes + 1)])

    def test_consistent_mass_first_5_modes(self) -> None:
        """Masse consistante : erreur < 1 % sur les 5 premières fréquences."""
        mesh, bc = _fixed_fixed_bar(self.N_ELEM)
        result = run_modal(mesh, bc, n_modes=5, use_lumped=False)

        f_ref = self._analytical(5)
        np.testing.assert_allclose(
            result.freqs, f_ref, rtol=0.01,
            err_msg="Masse consistante : erreur > 1 % sur barre encastrée–encastrée",
        )

    def test_lumped_mass_first_5_modes(self) -> None:
        """Masse condensée : erreur < 1 % sur les 5 premières fréquences."""
        mesh, bc = _fixed_fixed_bar(self.N_ELEM)
        result = run_modal(mesh, bc, n_modes=5, use_lumped=True)

        f_ref = self._analytical(5)
        np.testing.assert_allclose(
            result.freqs, f_ref, rtol=0.01,
            err_msg="Masse condensée : erreur > 1 % sur barre encastrée–encastrée",
        )

    def test_consistent_mass_upper_bound(self) -> None:
        """Masse consistante surestime les fréquences (borne supérieure).

        Pour les éléments bar linéaires, ω²_FEM_consistant ≥ ω²_exact.
        Propriété vérifiée analytiquement (Hughes §9.3).
        """
        mesh, bc = _fixed_fixed_bar(self.N_ELEM)
        result = run_modal(mesh, bc, n_modes=5, use_lumped=False)
        f_ref = self._analytical(5)
        assert np.all(result.freqs >= f_ref * 0.999), (
            "La masse consistante devrait surestimer les fréquences pour la barre"
        )

    def test_lumped_mass_lower_bound(self) -> None:
        """Masse condensée sous-estime les fréquences (borne inférieure).

        Pour les éléments bar linéaires (row-sum), ω²_FEM_lumped ≤ ω²_exact.
        """
        mesh, bc = _fixed_fixed_bar(self.N_ELEM)
        result = run_modal(mesh, bc, n_modes=5, use_lumped=True)
        f_ref = self._analytical(5)
        assert np.all(result.freqs <= f_ref * 1.001), (
            "La masse condensée devrait sous-estimer les fréquences pour la barre"
        )

    def test_freqs_are_sorted(self) -> None:
        """Les fréquences sont retournées en ordre croissant."""
        mesh, bc = _fixed_fixed_bar(self.N_ELEM)
        result = run_modal(mesh, bc, n_modes=5)
        assert np.all(np.diff(result.freqs) > 0)

    def test_omega_consistent_with_freqs(self) -> None:
        """ω_n = 2π f_n."""
        mesh, bc = _fixed_fixed_bar(self.N_ELEM)
        result = run_modal(mesh, bc, n_modes=5)
        np.testing.assert_allclose(result.omega, result.freqs * 2.0 * np.pi,
                                   rtol=1e-14)

    def test_refinement_improves_accuracy(self) -> None:
        """Raffiner le maillage réduit l'erreur sur f_1 (convergence)."""
        f_ref_1 = C / (2.0 * L)

        mesh_coarse, bc_coarse = _fixed_fixed_bar(n_elem=5)
        mesh_fine, bc_fine = _fixed_fixed_bar(n_elem=40)

        err_coarse = abs(run_modal(mesh_coarse, bc_coarse, n_modes=1).freqs[0] / f_ref_1 - 1.0)
        err_fine   = abs(run_modal(mesh_fine,   bc_fine,   n_modes=1).freqs[0] / f_ref_1 - 1.0)

        assert err_fine < err_coarse, (
            f"Le raffinage devrait améliorer f_1 : err_coarse={err_coarse:.4f}, "
            f"err_fine={err_fine:.4f}"
        )


# ---------------------------------------------------------------------------
# Tests fréquences propres — barre encastrée–libre
# ---------------------------------------------------------------------------


class TestFixedFreeBarModes:
    """Fréquences longitudinales d'une barre encastrée–libre.

    Solution analytique : f_n = (2n−1) · c / (4L),  n = 1, 2, 3, …

    Référence : Meirovitch, « Fundamentals of Vibrations », §7.3.
    """

    N_ELEM = 40

    @staticmethod
    def _analytical(n_modes: int) -> np.ndarray:
        return np.array([((2 * n - 1) * C) / (4.0 * L) for n in range(1, n_modes + 1)])

    def test_consistent_mass_first_5_modes(self) -> None:
        """Masse consistante : erreur < 1 % sur les 5 premières fréquences."""
        mesh, bc = _fixed_free_bar(self.N_ELEM)
        result = run_modal(mesh, bc, n_modes=5, use_lumped=False)

        f_ref = self._analytical(5)
        np.testing.assert_allclose(
            result.freqs, f_ref, rtol=0.01,
            err_msg="Masse consistante : erreur > 1 % sur barre encastrée–libre",
        )

    def test_lumped_mass_first_5_modes(self) -> None:
        """Masse condensée : erreur < 1 % sur les 5 premières fréquences."""
        mesh, bc = _fixed_free_bar(self.N_ELEM)
        result = run_modal(mesh, bc, n_modes=5, use_lumped=True)

        f_ref = self._analytical(5)
        np.testing.assert_allclose(
            result.freqs, f_ref, rtol=0.01,
            err_msg="Masse condensée : erreur > 1 % sur barre encastrée–libre",
        )


# ===========================================================================
# Tests de l'élimination vraie (DirichletSystem)
# ===========================================================================


class TestDirichletSystemElimination:
    """Vérifie la réduction exacte du système et l'absence de modes parasites.

    La barre encastrée–encastrée est choisie car :
    - La méthode row-zero (K[s,s]=1, M[s,s]≠0) produirait des modes
      parasites à ω²=1/M[s,s] ≈ (10–50 Hz) parmi les modes physiques.
    - L'élimination vraie les supprime complètement.
    """

    N_ELEM = 20

    def setup_method(self):
        self.mesh, self.bc = _fixed_fixed_bar(self.N_ELEM)
        assembler = Assembler(self.mesh)
        self.K = assembler.assemble_stiffness()
        self.M = assembler.assemble_mass()
        F_dummy = np.zeros(self.mesh.n_dof)
        self.ds = apply_dirichlet(self.K, F_dummy, self.mesh, self.bc)

    # -- Propriétés du DirichletSystem --

    def test_returns_dirichlet_system(self):
        """apply_dirichlet retourne un DirichletSystem."""
        assert isinstance(self.ds, DirichletSystem)

    def test_backward_compat_unpack(self):
        """K_bc, F_bc = apply_dirichlet(...) fonctionne toujours."""
        K_bc, F_bc = self.ds
        assert K_bc.shape == (self.mesh.n_dof, self.mesh.n_dof)
        assert F_bc.shape == (self.mesh.n_dof,)

    def test_free_dofs_count(self):
        """n_free = n_dof - n_constrained.

        Bar2D, N=20, n_nodes=21, n_dof=42 :
        Contraintes : ux(0)=0, ux(20)=0, uy(i)=0 pour i=0..20
        → 2 + 21 = 23 DDL globaux bloqués → n_free = 42 - 23 = 19.
        """
        # Comptage des DDL globaux bloqués (pas des local DOF uniques)
        constrained_global = {
            self.mesh.dpn * node + ldof
            for node, dofs in self.bc.dirichlet.items()
            for ldof in dofs
        }
        expected_free = self.mesh.n_dof - len(constrained_global)
        assert len(self.ds.free_dofs) == expected_free

    def test_K_free_shape(self):
        """K_free est carré de taille n_free."""
        n_free = len(self.ds.free_dofs)
        assert self.ds.K_free.shape == (n_free, n_free)

    def test_M_free_shape(self):
        """M_free est carré de taille n_free."""
        n_free = len(self.ds.free_dofs)
        M_free = self.ds.reduce_mass(self.M)
        assert M_free.shape == (n_free, n_free)

    def test_K_free_equals_K_original_submatrix(self):
        """K_free = K_original[free, free] (invariant row-zero)."""
        f = self.ds.free_dofs
        K_free_direct = self.K.tocsr()[f, :][:, f].tocsr()
        np.testing.assert_allclose(
            self.ds.K_free.toarray(),
            K_free_direct.toarray(),
            atol=1e-10,
        )

    def test_F_free_contains_rhs_correction(self):
        """F_free = F[free] - K_fc @ ū_c (force soustraite aux DDL libres).

        Pour ce cas, ū = 0 partout → F_free = F[free] = 0.
        """
        F_free = self.ds.F_free
        np.testing.assert_allclose(F_free, 0.0, atol=1e-15)

    def test_penalty_raises_for_K_free(self):
        """method='penalty' lève NotImplementedError sur K_free."""
        F_dummy = np.zeros(self.mesh.n_dof)
        ds_pen = apply_dirichlet(
            self.K, F_dummy, self.mesh, self.bc, method="penalty"
        )
        with pytest.raises(NotImplementedError):
            _ = ds_pen.K_free

    def test_penalty_raises_for_reduce_mass(self):
        """method='penalty' lève NotImplementedError sur reduce_mass."""
        F_dummy = np.zeros(self.mesh.n_dof)
        ds_pen = apply_dirichlet(
            self.K, F_dummy, self.mesh, self.bc, method="penalty"
        )
        with pytest.raises(NotImplementedError):
            ds_pen.reduce_mass(self.M)

    # -- Absence de modes parasites --

    def test_no_spurious_modes_vs_row_zero(self):
        """L'élimination vraie donne les fréquences physiques sans modes parasites.

        Avec row-zero (K[s,s]=1, M[s,s]≠0), les DDL uy bloqués auraient
        ω²_parasite = 1/M[s,s] << 1 Hz — sous le premier mode physique.
        On vérifie que les 3 premiers modes sont bien les fréquences physiques.
        N_ELEM=20 → erreur de discrétisation ~1 % sur les premiers modes.
        """
        # Maillage plus fin pour avoir rtol=0.01 sur les 3 premiers modes
        mesh, bc = _fixed_fixed_bar(n_elem=100)
        result = run_modal(mesh, bc, n_modes=3)
        f_ref = np.array([n * C / (2.0 * L) for n in range(1, 4)])
        np.testing.assert_allclose(
            result.freqs, f_ref, rtol=0.01,
            err_msg="Modes parasites détectés : fréquences hors de l'analytique.",
        )

    def test_constrained_dofs_zero_in_modes(self):
        """Les DDL contraints ont une valeur modale exactement nulle."""
        result = run_modal(self.mesh, self.bc, n_modes=3)
        constrained_dofs = [
            self.mesh.dpn * node + dof
            for node, dofs in self.bc.dirichlet.items()
            for dof in dofs
        ]
        for k in range(result.n_modes):
            np.testing.assert_allclose(
                result.modes[constrained_dofs, k], 0.0, atol=1e-14,
                err_msg=f"Mode {k+1} : DDL contraint non nul.",
            )

    def test_modes_M_orthogonal(self):
        """φᵀ M φ = I (M-orthogonalité des modes complets)."""
        result = run_modal(self.mesh, self.bc, n_modes=5)
        phi = result.modes    # (n_dof, n_modes)
        gram = phi.T @ self.M @ phi
        np.testing.assert_allclose(gram, np.eye(5), atol=1e-8,
                                   err_msg="Modes non M-orthogonaux.")

    def test_recover_modes_full_size(self):
        """recover_modes retourne des vecteurs de taille n_dof."""
        phi_free = np.random.rand(len(self.ds.free_dofs), 3)
        phi_full = self.ds.recover_modes(phi_free)
        assert phi_full.shape == (self.mesh.n_dof, 3)
        np.testing.assert_array_equal(phi_full[self.ds.free_dofs], phi_free)

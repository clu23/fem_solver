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

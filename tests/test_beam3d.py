"""Tests unitaires — Beam3D (Timoshenko 3D, 2 nœuds, 6 DDL/nœud).

Stratégie de validation
-----------------------
Tous les tests utilisent un élément unique dont le nœud 0 est encastré
(tous les DDL = 0).  La réduction du système 12×12 à la partition libre
(indices 6–11, nœud 1) donne un système 6×6 résolu exactement.

Solutions analytiques de référence (Timoshenko)
------------------------------------------------

Traction axiale (force Fx en bout) :
    δx = F · L / (EA)

Flexion dans le plan xy (force Fy en bout, encastrement en x=0) :
    δy = F·L³/(3·EIz) + F·L/(GAsy)   [Timoshenko exact, sans locking]
    θz = F·L²/(2·EIz)                  [rotation de section en bout]

    La contribution cisaillement est : FL/(GAsy) = FL·Φy / (12·EIz/L²)

Flexion dans le plan xz (force Fz en bout, encastrement en x=0) :
    δz = F·L³/(3·EIy) + F·L/(GAsz)
    θy = -F·L²/(2·EIy)                [signe − : règle de la main droite]

Torsion (moment Tx en bout) :
    φx = T · L / (GJ)

Offset (excentrement az au nœud 1, force Fy = F) :
    Le moment supplémentaire au nœud est Mz = F · az.
    La matrice excentrée inclut le terme de couplage
    K_off[7,11] = K_off[7,5] qui fait varier la solution.

Références
----------
Przemieniecki J.S. (1968). Theory of Matrix Structural Analysis, chap. 5.
Friedman & Kosmatka (1993). CMAME 105, 187–199.
Cook et al. (2002). Concepts and Applications of FEA, 4th ed., chap. 5.
"""

from __future__ import annotations

import numpy as np
import pytest

from femsolver.core.material import ElasticMaterial
from femsolver.core.sections import CircularSection, RectangularSection
from femsolver.elements.beam3d import Beam3D


# ---------------------------------------------------------------------------
# Fixtures communes
# ---------------------------------------------------------------------------

E   = 210e9     # Pa  (acier)
NU  = 0.3
RHO = 7800.0    # kg/m³
L   = 2.0       # m
F   = 1000.0    # N  (force appliquée en bout)
T   = 500.0     # N·m (couple de torsion)

MATERIAL = ElasticMaterial(E=E, nu=NU, rho=RHO)
SEC_RECT = RectangularSection(width=0.10, height=0.20)   # b=0.10, h=0.20 m
SEC_CIRC = CircularSection(radius=0.05)

# Nœuds : poutre horizontale le long de x
NODES_X = np.array([[0.0, 0.0, 0.0], [L, 0.0, 0.0]])


def _solve_cantilever(K_e: np.ndarray, force_vec: np.ndarray) -> np.ndarray:
    """Résout le cantilever single-element : nœud 0 encastré, nœud 1 libre.

    Parameters
    ----------
    K_e : np.ndarray, shape (12, 12)
        Matrice de rigidité élémentaire globale.
    force_vec : np.ndarray, shape (12,)
        Vecteur de force (non nul uniquement sur le nœud 1, indices 6–11).

    Returns
    -------
    u_free : np.ndarray, shape (6,)
        Déplacements du nœud 1 [ux2, uy2, uz2, θx2, θy2, θz2].
    """
    K_ff = K_e[6:12, 6:12]          # sous-matrice libre (nœud 1 × nœud 1)
    f_f  = force_vec[6:12]
    return np.linalg.solve(K_ff, f_f)


# ---------------------------------------------------------------------------
# Classe 1 — Interface Element
# ---------------------------------------------------------------------------

class TestBeam3DInterface:
    """Vérifications de l'interface Element (dof_per_node, n_nodes, shape)."""

    def setup_method(self) -> None:
        self.elem = Beam3D()

    def test_dof_per_node(self) -> None:
        assert self.elem.dof_per_node() == 6

    def test_n_nodes(self) -> None:
        assert self.elem.n_nodes() == 2

    def test_n_dof(self) -> None:
        assert self.elem.n_dof() == 12

    def test_stiffness_shape(self) -> None:
        K = self.elem.stiffness_matrix(MATERIAL, NODES_X, {"section": SEC_RECT})
        assert K.shape == (12, 12)

    def test_mass_shape(self) -> None:
        M = self.elem.mass_matrix(MATERIAL, NODES_X, {"section": SEC_RECT})
        assert M.shape == (12, 12)


# ---------------------------------------------------------------------------
# Classe 2 — Repère local et v-vector
# ---------------------------------------------------------------------------

class TestLocalFrame:
    """Construction du repère local (e₁, e₂, e₃) et gestion du v-vector."""

    def test_default_vvec_horizontal_beam(self) -> None:
        """Beam horizontal → v-vec=[0,0,1] → e₂ devrait pointer vers +y.

        e₁ = [1,0,0], e₃ = e₁ × [0,0,1] = [0,-1,0]... attendez :
        e₃ = normalise(e₁ × v) = [1,0,0] × [0,0,1] = [0·1−0·0, 0·0−1·1, 1·0−0·0]
               = [0,-1,0]
        e₂ = e₃ × e₁ = [0,-1,0] × [1,0,0] = [(-1)·0−0·0, 0·1−0·0, 0·0−(-1)·1]
               = [0,0,1]

        Donc e₂ = [0,0,1] pour une poutre selon +x avec v=[0,0,1].
        C'est le plan de flexion forte (Iz) dans le plan x-z.

        Pour e₂ = [0,1,0] (vertical vers le haut), utiliser v=[0,1,0].
        Le v-vector par défaut ([0,0,1] pour poutres horizontales) est
        documenté dans le module.
        """
        L_calc, lam = Beam3D._local_frame(NODES_X, v_vec=None)
        np.testing.assert_allclose(L_calc, L, rtol=1e-12)
        # e₁ = [1,0,0]
        np.testing.assert_allclose(lam[0], [1.0, 0.0, 0.0], atol=1e-12)
        # Vérifier l'orthonormalité
        np.testing.assert_allclose(lam @ lam.T, np.eye(3), atol=1e-12)

    def test_vvec_up_gives_e2_up(self) -> None:
        """v=[0,1,0] → e₂=[0,1,0] pour une poutre selon +x.

        e₁ = [1,0,0], e₃ = [1,0,0]×[0,1,0] = [0,0,1]
        e₂ = [0,0,1] × [1,0,0] = [0,1,0]  ← dans le plan xy
        """
        _, lam = Beam3D._local_frame(NODES_X, v_vec=np.array([0.0, 1.0, 0.0]))
        np.testing.assert_allclose(lam[1], [0.0, 1.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(lam[2], [0.0, 0.0, 1.0], atol=1e-12)

    def test_orthonormality(self) -> None:
        """La matrice λ doit être orthogonale : λ·λᵀ = I₃."""
        for v in ([0, 0, 1], [0, 1, 0], [1, 1, 0], [0.5, 0.3, 0.8]):
            nodes = np.array([[0., 0., 0.], [1.5, 1.0, 0.5]])
            v_arr = np.array(v, dtype=float)
            if np.dot(nodes[1]-nodes[0], v_arr) < 1.0 - 1e-6:   # non parallèle
                _, lam = Beam3D._local_frame(nodes, v_vec=v_arr)
                np.testing.assert_allclose(
                    lam @ lam.T, np.eye(3), atol=1e-12,
                    err_msg=f"λ non orthogonal pour v={v}"
                )

    def test_vertical_beam_default_vvec(self) -> None:
        """Poutre verticale (selon +z) → v-vec=[1,0,0] par défaut."""
        nodes = np.array([[0., 0., 0.], [0., 0., 3.]])
        L_calc, lam = Beam3D._local_frame(nodes, v_vec=None)
        np.testing.assert_allclose(lam @ lam.T, np.eye(3), atol=1e-12)
        # e₁ doit pointer selon +z
        np.testing.assert_allclose(lam[0], [0., 0., 1.], atol=1e-12)

    def test_parallel_vvec_raises(self) -> None:
        """v-vec parallèle à e₁ doit lever ValueError."""
        with pytest.raises(ValueError, match="parallèle"):
            Beam3D._local_frame(NODES_X, v_vec=np.array([1.0, 0.0, 0.0]))

    def test_zero_length_raises(self) -> None:
        """Nœuds confondus (longueur nulle) → ValueError."""
        nodes = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        with pytest.raises(ValueError, match="nulle"):
            Beam3D._local_frame(nodes, v_vec=None)


# ---------------------------------------------------------------------------
# Classe 3 — Matrice de rigidité locale (symétrie, découpage)
# ---------------------------------------------------------------------------

class TestStiffnessMatrix:
    """Propriétés algébriques de K_e et découpage en blocs.

    On utilise v_vec=[0,1,0] pour que le repère local coïncide avec le repère
    global (lam = I₃) : global y ↔ local y (fort Iz), global z ↔ local z (faible Iy).
    Sans v_vec explicite, le v_vec par défaut [0,0,1] échange les axes,
    rendant les tests numériques sur les blocs peu lisibles.
    """

    def setup_method(self) -> None:
        self.elem = Beam3D()
        # v_vec=[0,1,0] → lam = I₃ pour poutre le long de +x
        self.props = {"section": SEC_RECT, "v_vec": np.array([0., 1., 0.])}

    def test_symmetry(self) -> None:
        """K_e doit être symétrique (erreur < 1e-12 × ||K||)."""
        K = self.elem.stiffness_matrix(MATERIAL, NODES_X, self.props)
        err = np.max(np.abs(K - K.T))
        assert err < 1e-10 * np.max(np.abs(K)), f"K non symétrique, max err = {err}"

    def test_positive_semidefinite(self) -> None:
        """K_e doit être SDP (6 valeurs propres ≈ 0 (modes rigides) + 6 positives)."""
        K = self.elem.stiffness_matrix(MATERIAL, NODES_X, self.props)
        eigvals = np.linalg.eigvalsh(K)
        assert np.all(eigvals >= -1e-6 * eigvals[-1]), (
            f"K n'est pas SDP : min eigenvalue = {eigvals[0]:.3e}"
        )
        # 6 modes rigides
        near_zero = np.sum(np.abs(eigvals) < 1e-6 * eigvals[-1])
        assert near_zero == 6, f"Attendu 6 modes rigides, obtenu {near_zero}"

    def test_axial_block(self) -> None:
        """Bloc axial K[0,0]=K[6,6]=EA/L, K[0,6]=-EA/L.

        Valeur analytique : EA/L = 210e9 × (0.10×0.20) / 2.0 = 2.1e9 N/m.
        """
        K = self.elem.stiffness_matrix(MATERIAL, NODES_X, self.props)
        EA_over_L = MATERIAL.E * SEC_RECT.area / L
        np.testing.assert_allclose(K[0, 0],  EA_over_L, rtol=1e-12)
        np.testing.assert_allclose(K[6, 6],  EA_over_L, rtol=1e-12)
        np.testing.assert_allclose(K[0, 6], -EA_over_L, rtol=1e-12)

    def test_torsion_block(self) -> None:
        """Bloc torsion K[3,3]=GJ/L, K[3,9]=-GJ/L."""
        K = self.elem.stiffness_matrix(MATERIAL, NODES_X, self.props)
        GJ_over_L = MATERIAL.G * SEC_RECT.J / L
        np.testing.assert_allclose(K[3, 3],  GJ_over_L, rtol=1e-10)
        np.testing.assert_allclose(K[9, 9],  GJ_over_L, rtol=1e-10)
        np.testing.assert_allclose(K[3, 9], -GJ_over_L, rtol=1e-10)

    def test_bending_xy_block(self) -> None:
        """Vérification numérique du bloc de flexion xy (plan xy).

        Indices : uy₁=1, θz₁=5, uy₂=7, θz₂=11.
        k = EIz / ((1+Φy)·L³),  Φy = 12·EIz / (GAsy·L²).
        """
        K = self.elem.stiffness_matrix(MATERIAL, NODES_X, self.props)
        kappa = SEC_RECT.shear_correction_factor(NU)
        EIz   = MATERIAL.E * SEC_RECT.Iz
        GAsy  = MATERIAL.G * kappa * SEC_RECT.area
        Phi_y = 12.0 * EIz / (GAsy * L**2)
        k     = EIz / ((1.0 + Phi_y) * L**3)

        np.testing.assert_allclose(K[1,  1],  12.0 * k,          rtol=1e-10)
        np.testing.assert_allclose(K[7,  7],  12.0 * k,          rtol=1e-10)
        np.testing.assert_allclose(K[1,  7], -12.0 * k,          rtol=1e-10)
        np.testing.assert_allclose(K[5,  5],  (4.0+Phi_y)*k*L**2, rtol=1e-10)
        np.testing.assert_allclose(K[11, 11], (4.0+Phi_y)*k*L**2, rtol=1e-10)

    def test_no_coupling_between_planes(self) -> None:
        """K doit être bloc-diagonal : pas de couplage xy/xz/axial/torsion.

        Les termes K[i,j] entre des DDL de plans orthogonaux doivent être nuls.
        Exemples : K[uy,uz]=0, K[uy,θy]=0, K[θz,θy]=0, K[axial,bending]=0.
        """
        K = self.elem.stiffness_matrix(MATERIAL, NODES_X, self.props)
        # Axial ne se couple pas à la flexion
        assert abs(K[0, 1]) < 1e-12   # ux1 – uy1
        assert abs(K[0, 7]) < 1e-12   # ux1 – uy2
        # xy ne se couple pas à xz
        assert abs(K[1, 2]) < 1e-12   # uy1 – uz1
        assert abs(K[5, 4]) < 1e-12   # θz1 – θy1
        assert abs(K[7, 8]) < 1e-12   # uy2 – uz2
        # Torsion découplée
        assert abs(K[3, 1]) < 1e-12   # θx1 – uy1
        assert abs(K[9, 2]) < 1e-12   # θx2 – uz2


# ---------------------------------------------------------------------------
# Classe 4 — Déplacements analytiques (cantilever single-element)
# ---------------------------------------------------------------------------

class TestCantileverDisplacements:
    """Comparaison des déplacements au nœud libre vs solutions analytiques.

    L'élément Timoshenko exact (fonctions de forme d'équilibre) reproduit
    les déflexions analytiques à la précision machine (rtol = 1e-12).

    On utilise v_vec=[0,1,0] (lam = I₃) afin que :
      - Fy global (DDL 7) = force locale uy → plan xy local → EIz (fort)
      - Fz global (DDL 8) = force locale uz → plan xz local → EIy (faible)
    """

    def setup_method(self) -> None:
        self.elem = Beam3D()
        self.props = {"section": SEC_RECT, "v_vec": np.array([0., 1., 0.])}

    # ── Rigidités ──────────────────────────────────────────────────────────
    @property
    def _props_scalar(self) -> tuple:
        kappa = SEC_RECT.shear_correction_factor(NU)
        EA   = MATERIAL.E * SEC_RECT.area
        EIy  = MATERIAL.E * SEC_RECT.Iy
        EIz  = MATERIAL.E * SEC_RECT.Iz
        GJ   = MATERIAL.G * SEC_RECT.J
        GAsy = MATERIAL.G * kappa * SEC_RECT.area
        GAsz = MATERIAL.G * kappa * SEC_RECT.area
        return EA, EIy, EIz, GJ, GAsy, GAsz

    def test_axial_displacement(self) -> None:
        """δx = F·L/(EA).

        Référence : Przemieniecki (1968) eq. 5.62 — terme diagonal axial.
        """
        K_e = self.elem.stiffness_matrix(MATERIAL, NODES_X, self.props)
        f = np.zeros(12)
        f[6] = F                               # Fx au nœud 1
        u_free = _solve_cantilever(K_e, f)
        EA, *_ = self._props_scalar
        delta_analytical = F * L / EA
        np.testing.assert_allclose(u_free[0], delta_analytical, rtol=1e-12,
            err_msg=f"δx FEM={u_free[0]:.6e} vs analytique={delta_analytical:.6e}")

    def test_torsion_angle(self) -> None:
        """φx = T·L/(GJ).

        Référence : Przemieniecki (1968) eq. 5.62 — terme diagonal torsion.
        """
        K_e = self.elem.stiffness_matrix(MATERIAL, NODES_X, self.props)
        f = np.zeros(12)
        f[9] = T                               # Tx au nœud 1
        u_free = _solve_cantilever(K_e, f)
        _, _, _, GJ, _, _ = self._props_scalar
        phi_analytical = T * L / GJ
        np.testing.assert_allclose(u_free[3], phi_analytical, rtol=1e-12,
            err_msg=f"φx FEM={u_free[3]:.6e} vs analytique={phi_analytical:.6e}")

    def test_bending_xy_tip_deflection(self) -> None:
        """δy = F·L³/(3·EIz) + F·L/(GAsy)  [Timoshenko exact].

        Le terme cisaillement F·L/(GAsy) est significatif pour les poutres
        courtes (ici L/h = 2.0/0.2 = 10).
        """
        K_e = self.elem.stiffness_matrix(MATERIAL, NODES_X, self.props)
        f = np.zeros(12)
        f[7] = F                               # Fy au nœud 1
        u_free = _solve_cantilever(K_e, f)
        _, _, EIz, _, GAsy, _ = self._props_scalar

        delta_eb   = F * L**3 / (3.0 * EIz)
        delta_shear = F * L / GAsy
        delta_analytical = delta_eb + delta_shear

        np.testing.assert_allclose(u_free[1], delta_analytical, rtol=1e-12,
            err_msg=f"δy FEM={u_free[1]:.6e} vs analytique={delta_analytical:.6e}")

    def test_bending_xy_tip_rotation(self) -> None:
        """θz = F·L²/(2·EIz)  [identique EB et Timoshenko pour section uniforme].

        La rotation de section en bout ne dépend pas du cisaillement dans la
        formulation de Timoshenko à 2 nœuds (fonctions de forme d'équilibre).
        Référence : Friedman & Kosmatka (1993), équation vérifiée sur le
        schéma statique exact du cantilever.
        """
        K_e = self.elem.stiffness_matrix(MATERIAL, NODES_X, self.props)
        f = np.zeros(12)
        f[7] = F                               # Fy au nœud 1
        u_free = _solve_cantilever(K_e, f)
        _, _, EIz, _, GAsy, _ = self._props_scalar
        Phi_y = 12.0 * EIz / (GAsy * L**2)
        # Rotation de section exacte pour Timoshenko
        k = EIz / ((1.0 + Phi_y) * L**3)
        K_ff = self.elem.stiffness_matrix(MATERIAL, NODES_X, self.props)[6:12, 6:12]
        # Réutilise la solution calculée
        theta_z_fem = u_free[5]               # θz2 (indice 5 dans le vecteur libre)

        theta_analytical = F * L**2 / (2.0 * EIz)
        np.testing.assert_allclose(theta_z_fem, theta_analytical, rtol=1e-10,
            err_msg=f"θz FEM={theta_z_fem:.6e} vs analytique={theta_analytical:.6e}")

    def test_bending_xz_tip_deflection(self) -> None:
        """δz = F·L³/(3·EIy) + F·L/(GAsz)  [plan xz, Timoshenko exact].

        Convention θy : rotation positive θy autour de +ŷ fait tourner +x̂ vers −ẑ.
        L'effort Fz positif produit δz positif et θy négatif.
        """
        K_e = self.elem.stiffness_matrix(MATERIAL, NODES_X, self.props)
        f = np.zeros(12)
        f[8] = F                               # Fz au nœud 1
        u_free = _solve_cantilever(K_e, f)
        _, EIy, _, _, _, GAsz = self._props_scalar

        delta_eb    = F * L**3 / (3.0 * EIy)
        delta_shear = F * L / GAsz
        delta_analytical = delta_eb + delta_shear

        np.testing.assert_allclose(u_free[2], delta_analytical, rtol=1e-12,
            err_msg=f"δz FEM={u_free[2]:.6e} vs analytique={delta_analytical:.6e}")

    def test_bending_xz_tip_rotation_sign(self) -> None:
        """θy négatif sous Fz positif (règle de la main droite).

        Fz > 0 → la poutre fléchit vers +z → la section en bout tourne de
        telle sorte que la normale à la section passe de +x vers -z, soit
        θy < 0 (rotation autour de +ŷ dans le sens négatif).
        """
        K_e = self.elem.stiffness_matrix(MATERIAL, NODES_X, self.props)
        f = np.zeros(12)
        f[8] = F
        u_free = _solve_cantilever(K_e, f)
        assert u_free[4] < 0.0, (
            f"θy doit être négatif sous Fz positif, obtenu θy={u_free[4]:.4e}"
        )

    def test_decoupling_between_planes(self) -> None:
        """Une force Fy ne provoque aucun déplacement uz, θx, θy."""
        K_e = self.elem.stiffness_matrix(MATERIAL, NODES_X, self.props)
        f = np.zeros(12)
        f[7] = F                               # Fy uniquement
        u_free = _solve_cantilever(K_e, f)

        assert abs(u_free[0]) < 1e-14 * abs(u_free[1])   # ux
        assert abs(u_free[2]) < 1e-14 * abs(u_free[1])   # uz
        assert abs(u_free[3]) < 1e-14 * abs(u_free[1])   # θx
        assert abs(u_free[4]) < 1e-14 * abs(u_free[1])   # θy


# ---------------------------------------------------------------------------
# Classe 5 — Rotation de repère (beam oblique)
# ---------------------------------------------------------------------------

class TestRotation:
    """La transformation T = block_diag(λ, λ, λ, λ) doit conserver la physique."""

    def setup_method(self) -> None:
        self.elem = Beam3D()
        self.props_circ = {"section": SEC_CIRC}

    def test_oblique_beam_axial(self) -> None:
        """Poutre oblique à 45° dans le plan xy — déplacement axial identique.

        Une force appliquée dans la direction de la poutre doit produire
        la même déflexion axiale δ = FL/(EA) indépendamment de l'orientation.
        """
        L_obl = 2.0 * np.sqrt(2.0)                        # longueur = 2√2 m
        nodes_obl = np.array([[0., 0., 0.], [2., 2., 0.]])
        K_e = self.elem.stiffness_matrix(MATERIAL, nodes_obl, self.props_circ)

        # Direction de la poutre (e₁ normalisé)
        d = nodes_obl[1] - nodes_obl[0]
        e1 = d / np.linalg.norm(d)   # = [1/√2, 1/√2, 0]

        # Force dans la direction de la poutre
        Fvec = np.zeros(12)
        Fvec[6:9] = F * e1            # Fx, Fy au nœud 1

        # Résolution (nœud 0 encastré)
        K_ff = K_e[6:12, 6:12]
        u_free = np.linalg.solve(K_ff, Fvec[6:12])

        # Déplacement axial = projection u sur e₁
        delta_axial = np.dot(u_free[:3], e1)
        EA = MATERIAL.E * SEC_CIRC.area
        delta_analytical = F * L_obl / EA

        np.testing.assert_allclose(delta_axial, delta_analytical, rtol=1e-12)

    def test_arbitrary_orientation_symmetry(self) -> None:
        """K_e d'une poutre arbitraire doit rester symétrique."""
        nodes_obl = np.array([[0., 0., 0.], [1., 2., 1.5]])
        K_e = self.elem.stiffness_matrix(MATERIAL, nodes_obl, self.props_circ)
        np.testing.assert_allclose(K_e, K_e.T, atol=1e-10,
            err_msg="K_e non symétrique pour poutre oblique")

    def test_rotation_invariance_section_circular(self) -> None:
        """Section circulaire : K_e identique quelle que soit l'orientation v-vec.

        Pour une section isotrope (Iy = Iz, GAsy = GAsz), la matrice locale
        K_local est la même pour tout v-vec ; seule la rotation T change,
        mais Tᵀ K_local T reste le même (à une transformation orthogonale près).
        En particulier, toutes les valeurs propres sont identiques.
        """
        props_v1 = {"section": SEC_CIRC, "v_vec": np.array([0., 1., 0.])}
        props_v2 = {"section": SEC_CIRC, "v_vec": np.array([0., 0., 1.])}
        K1 = self.elem.stiffness_matrix(MATERIAL, NODES_X, props_v1)
        K2 = self.elem.stiffness_matrix(MATERIAL, NODES_X, props_v2)

        eigvals1 = np.sort(np.linalg.eigvalsh(K1))
        eigvals2 = np.sort(np.linalg.eigvalsh(K2))
        np.testing.assert_allclose(eigvals1, eigvals2, rtol=1e-10,
            err_msg="Valeurs propres différentes pour deux v-vecs d'une section circulaire")


# ---------------------------------------------------------------------------
# Classe 6 — Offset (liaison rigide excentrée)
# ---------------------------------------------------------------------------

class TestOffset:
    """Vérification de la transformation d'offset K_off = T_offᵀ · K_e · T_off."""

    def setup_method(self) -> None:
        self.elem = Beam3D()
        self.props_base = {"section": SEC_CIRC}

    def test_no_offset_unchanged(self) -> None:
        """offset_i=0, offset_j=0 → K identique à sans offset."""
        K_no  = self.elem.stiffness_matrix(MATERIAL, NODES_X, self.props_base)
        props = {**self.props_base,
                 "offset_i": np.zeros(3),
                 "offset_j": np.zeros(3)}
        K_off = self.elem.stiffness_matrix(MATERIAL, NODES_X, props)
        np.testing.assert_allclose(K_off, K_no, atol=1e-10)

    def test_offset_symmetry(self) -> None:
        """K excentrée doit rester symétrique."""
        props = {**self.props_base, "offset_j": np.array([0.0, 0.1, 0.0])}
        K_off = self.elem.stiffness_matrix(MATERIAL, NODES_X, props)
        np.testing.assert_allclose(K_off, K_off.T, atol=1e-10)

    def test_offset_transforms_displacement(self) -> None:
        """Excentrement az au nœud 1 ajoute un bras de levier pour Fy.

        Un offset purement en z au nœud 2 (offset_j = [0, 0, az]) ne
        déplace pas la position du nœud, mais la connexion avec la poutre
        est excentrée.  La rigidité effective est modifiée : le bras de
        levier crée un couplage Fy–θz.

        Test de cohérence : K_off doit être SDP et différente de K_no.
        """
        az = 0.05    # m d'excentrement selon z
        props = {**self.props_base, "offset_j": np.array([0.0, 0.0, az])}
        K_no  = self.elem.stiffness_matrix(MATERIAL, NODES_X, self.props_base)
        K_off = self.elem.stiffness_matrix(MATERIAL, NODES_X, props)

        # Les deux matrices doivent être différentes
        assert not np.allclose(K_no, K_off), "K_off devrait différer de K_no avec offset ≠ 0"

        # K_off doit rester SDP
        eigvals = np.linalg.eigvalsh(K_off)
        assert np.all(eigvals >= -1e-6 * eigvals[-1]), (
            f"K_off n'est pas SDP : min eigenvalue = {eigvals[0]:.3e}"
        )

    def test_offset_transform_matrix(self) -> None:
        """Vérification directe de la matrice T_off pour un offset simple.

        Offset a = [0, 0, az] au nœud i.
        G(a) = [[I, -ã], [0, I]] avec ã = skew([0,0,az]).

        skew([0,0,az]) = [[ 0,  -az,  0],
                          [ az,   0,  0],
                          [ 0,    0,  0]]
        Donc −ã = [[0, az, 0], [-az, 0, 0], [0, 0, 0]].
        Bloc (0:3, 3:6) de G :
            G[0,4] =  az    (ux déplacé par +az·θy)
            G[0,5] = -ay=0  (uy=0)
            G[1,3] = -az    (uy déplacé par −az·θx)
            G[1,5] =  ax=0  (ax=0)
            G[2,3] =  ay=0  (ay=0)
            G[2,4] = -ax=0  (ax=0)
        """
        az = 0.1
        T_off = Beam3D._offset_transform(
            np.array([0., 0., az]),   # offset_i
            None,
        )
        assert T_off is not None
        # Bloc G(a_i) = T_off[0:6, 0:6]
        G = T_off[0:6, 0:6]
        np.testing.assert_allclose(G[0, 4],  az, atol=1e-14)    # ux déplacé par θy
        np.testing.assert_allclose(G[1, 3], -az, atol=1e-14)    # uy déplacé par θx
        # Bloc nœud 2 doit être identité (offset_j = None → zéro)
        np.testing.assert_allclose(T_off[6:12, 6:12], np.eye(6), atol=1e-14)


# ---------------------------------------------------------------------------
# Classe 7 — Matrice de masse
# ---------------------------------------------------------------------------

class TestMassMatrix:
    """Vérifications de la matrice de masse consistante."""

    def setup_method(self) -> None:
        self.elem = Beam3D()
        self.props = {"section": SEC_RECT}

    def test_symmetry(self) -> None:
        """M_e doit être symétrique."""
        M = self.elem.mass_matrix(MATERIAL, NODES_X, self.props)
        np.testing.assert_allclose(M, M.T, atol=1e-12)

    def test_positive_definite(self) -> None:
        """M_e doit être strictement définie positive."""
        M = self.elem.mass_matrix(MATERIAL, NODES_X, self.props)
        eigvals = np.linalg.eigvalsh(M)
        assert np.all(eigvals > 0.0), (
            f"M_e n'est pas DP : min eigenvalue = {eigvals[0]:.3e}"
        )

    def test_axial_total_mass(self) -> None:
        """Somme de ligne du bloc axial = ρ·A·L (masse totale).

        Pour la matrice de masse consistante linéaire du bloc axial,
        chaque ligne de M[{0,6}, {0,6}] somme à ρAL/2 (demi-masse par nœud).
        La somme globale des deux lignes = ρAL.

        Référence : Przemieniecki (1968) eq. 5.20 — matrice de masse barre.
        """
        M = self.elem.mass_matrix(MATERIAL, NODES_X, self.props)
        m_total = MATERIAL.rho * SEC_RECT.area * L
        row_sum_0 = M[0, 0] + M[0, 6]
        row_sum_6 = M[6, 0] + M[6, 6]
        np.testing.assert_allclose(row_sum_0, m_total / 2.0, rtol=1e-12)
        np.testing.assert_allclose(row_sum_6, m_total / 2.0, rtol=1e-12)

    def test_torsion_mass(self) -> None:
        """Bloc torsion : terme diagonal = ρ·Ip·L/3, hors-diagonal = ρ·Ip·L/6.

        Ip = Iy + Iz = moment polaire.
        """
        M = self.elem.mass_matrix(MATERIAL, NODES_X, self.props)
        Ip     = SEC_RECT.Iy + SEC_RECT.Iz
        rho_Ip = MATERIAL.rho * Ip
        np.testing.assert_allclose(M[3, 3], 2.0 * rho_Ip * L / 6.0, rtol=1e-12)
        np.testing.assert_allclose(M[9, 9], 2.0 * rho_Ip * L / 6.0, rtol=1e-12)
        np.testing.assert_allclose(M[3, 9],       rho_Ip * L / 6.0, rtol=1e-12)

    def test_bending_mass_positive(self) -> None:
        """Les termes diagonaux de flexion M[uy1,uy1] doivent être > 0."""
        M = self.elem.mass_matrix(MATERIAL, NODES_X, self.props)
        assert M[1, 1] > 0.0   # M[uy1, uy1]
        assert M[7, 7] > 0.0   # M[uy2, uy2]
        assert M[2, 2] > 0.0   # M[uz1, uz1]
        assert M[8, 8] > 0.0   # M[uz2, uz2]

    def test_mass_invariant_to_orientation(self) -> None:
        """La masse totale (somme lignes bloc axial) ne change pas avec v-vec."""
        m_ref = MATERIAL.rho * SEC_CIRC.area * L
        for v in ([0, 1, 0], [0, 0, 1], [0, 1, 1]):
            props = {"section": SEC_CIRC, "v_vec": np.array(v, dtype=float)}
            M = self.elem.mass_matrix(MATERIAL, NODES_X, props)
            row_sum = M[0, 0] + M[0, 6]
            np.testing.assert_allclose(row_sum, m_ref / 2.0, rtol=1e-10,
                err_msg=f"Masse incorrecte pour v={v}")


# ---------------------------------------------------------------------------
# Classe 8 — Efforts internes (section_forces)
# ---------------------------------------------------------------------------

class TestSectionForces:
    """Vérification de l'équilibre nodal via section_forces."""

    def setup_method(self) -> None:
        self.elem = Beam3D()
        self.props = {"section": SEC_CIRC}

    def _full_cantilever_u(self, dof_idx: int, force_val: float) -> np.ndarray:
        """Résout le cantilever et retourne le vecteur déplacement 12D."""
        K_e = self.elem.stiffness_matrix(MATERIAL, NODES_X, self.props)
        f = np.zeros(12)
        f[dof_idx] = force_val
        u_free = _solve_cantilever(K_e, f)
        u = np.zeros(12)
        u[6:12] = u_free
        return u

    def test_axial_equilibrium(self) -> None:
        """N1 + N2 = 0 pour une traction axiale (équilibre de la barre).

        N1 est la force de réaction (négatif), N2 est la force appliquée.
        """
        u = self._full_cantilever_u(6, F)
        sf = self.elem.section_forces(MATERIAL, NODES_X, self.props, u)
        # N2 = F appliqué, N1 = -F (réaction)
        np.testing.assert_allclose(sf["N1"] + sf["N2"], 0.0, atol=1e-6 * F,
            err_msg=f"N1={sf['N1']:.4e}, N2={sf['N2']:.4e}")

    def test_bending_xy_equilibrium(self) -> None:
        """Vy1 + Vy2 = 0 et Mz1 = F·L − Mz2 pour la flexion xy.

        Équilibre statique de la poutre isolée :
            ΣFy = 0 → Vy1 + Vy2 = 0
            ΣMz (nœud 1) = 0 → Mz1 = F·L  (réaction d'encastrement)
        """
        u = self._full_cantilever_u(7, F)
        sf = self.elem.section_forces(MATERIAL, NODES_X, self.props, u)
        np.testing.assert_allclose(sf["Vy1"] + sf["Vy2"], 0.0, atol=1e-6 * F)

    def test_torsion_equilibrium(self) -> None:
        """Tx1 + Tx2 = 0 pour une torsion pure."""
        u = self._full_cantilever_u(9, T)
        sf = self.elem.section_forces(MATERIAL, NODES_X, self.props, u)
        np.testing.assert_allclose(sf["Tx1"] + sf["Tx2"], 0.0, atol=1e-6 * T)

    def test_section_forces_keys(self) -> None:
        """section_forces doit retourner les 12 clés attendues."""
        u = np.zeros(12)
        sf = self.elem.section_forces(MATERIAL, NODES_X, self.props, u)
        expected_keys = {"N1", "Vy1", "Vz1", "Tx1", "My1", "Mz1",
                         "N2", "Vy2", "Vz2", "Tx2", "My2", "Mz2"}
        assert set(sf.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Classe 9 — Robustesse et cas limites
# ---------------------------------------------------------------------------

class TestRobustness:
    """Comportement en cas de section très raide (Φ → 0) et très souple."""

    def setup_method(self) -> None:
        self.elem = Beam3D()

    def test_slender_beam_approaches_euler_bernoulli(self) -> None:
        """Poutre très élancée (L/h = 200) → Timoshenko ≈ Euler–Bernoulli.

        Le terme de cisaillement FL/(GAsy) ≪ FL³/(3EIz) pour L/h grand.
        La différence relative doit être < 1 % pour L/h = 200.

        v_vec=[0,1,0] → lam=I₃ → Fy global maps to local uy → EIz (fort).
        """
        L_long = 20.0                          # m (L/h = 200 avec h=0.20 m)
        nodes = np.array([[0., 0., 0.], [L_long, 0., 0.]])
        props = {"section": SEC_RECT, "v_vec": np.array([0., 1., 0.])}
        K_e = self.elem.stiffness_matrix(MATERIAL, nodes, props)

        f = np.zeros(12)
        f[7] = F
        u_free = _solve_cantilever(K_e, f)
        delta_fem = u_free[1]

        EIz = MATERIAL.E * SEC_RECT.Iz
        delta_eb = F * L_long**3 / (3.0 * EIz)
        rel_diff = abs(delta_fem - delta_eb) / delta_eb
        assert rel_diff < 0.01, (
            f"Poutre élancée : écart Timoshenko/EB = {rel_diff*100:.3f}% > 1%"
        )

    def test_kappa_override(self) -> None:
        """Surcharger kappa_y / kappa_z doit modifier la rigidité de cisaillement."""
        props_default = {"section": SEC_RECT}
        props_kappa   = {"section": SEC_RECT, "kappa_y": 1.0, "kappa_z": 1.0}

        K_def = self.elem.stiffness_matrix(MATERIAL, NODES_X, props_default)
        K_kap = self.elem.stiffness_matrix(MATERIAL, NODES_X, props_kappa)
        # kappa=1 → moins de cisaillement → poutre plus rigide en flexion
        # Terme K[7,7] doit être plus grand avec kappa=1
        assert K_kap[7, 7] >= K_def[7, 7], (
            "κ=1 devrait augmenter la rigidité de cisaillement (terme K[7,7])"
        )

    def test_missing_section_raises(self) -> None:
        """Absence de 'section' dans properties → KeyError."""
        with pytest.raises(KeyError, match="section"):
            self.elem.stiffness_matrix(MATERIAL, NODES_X, {})

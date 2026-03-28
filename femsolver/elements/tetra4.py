"""Élément tétraèdre linéaire isoparamétrique à 4 nœuds (C3D4).

Passage 2D → 3D : principes conservés, dimensions étendues
-----------------------------------------------------------

**Matrice D 3D (6×6)**
En 2D contrainte plane, le vecteur de déformation est [εxx, εyy, γxy] → D est 3×3.
En 3D isotrope, les 6 composantes indépendantes (notation de Voigt) sont :
    ε = [εxx, εyy, εzz, γyz, γxz, γxy]
et D est 6×6 avec les constantes de Lamé λ et μ.

**Matrice B 3D (6×12)**
4 nœuds × 3 DDL (ux, uy, uz) = 12 DDL élémentaires.
B traduit les déplacements en déformations : ε = B · u_e.
Pour le nœud i (colonnes 3i, 3i+1, 3i+2) :

    B[:, 3i:3i+3] =
        [ ∂Ni/∂x,    0,       0      ]   ← εxx = ∂ux/∂x
        [ 0,         ∂Ni/∂y,  0      ]   ← εyy = ∂uy/∂y
        [ 0,         0,       ∂Ni/∂z ]   ← εzz = ∂uz/∂z
        [ 0,         ∂Ni/∂z,  ∂Ni/∂y ]   ← γyz = ∂uy/∂z + ∂uz/∂y
        [ ∂Ni/∂z,   0,       ∂Ni/∂x ]   ← γxz = ∂ux/∂z + ∂uz/∂x
        [ ∂Ni/∂y,   ∂Ni/∂x,  0      ]   ← γxy = ∂ux/∂y + ∂uy/∂x

**Calcul de K_e — même principe qu'en 2D**
Tetra4 a des fonctions de forme *linéaires* → B est *constante* dans l'élément.
L'intégrale ∫ B^T D B dV se réduit à un seul évaluation :

    K_e = B^T · D · B · V        (V = volume du tétraèdre)

C'est l'analogue 3D du CST (Tri3) où K_e = B^T · D · B · A · t.
Aucune quadrature de Gauss n'est nécessaire.

**Volume via le Jacobien**
En coordonnées naturelles (ξ, η, ζ) ∈ tétraèdre de référence :
    N1 = 1-ξ-η-ζ,  N2 = ξ,  N3 = η,  N4 = ζ
Le Jacobien J = dN_nat · nodes (3×3) vérifie :
    V = det(J) / 6

**Matrice de masse consistante (12×12)**
∫_V ρ N^T N dV se calcule analytiquement grâce aux coordonnées volumiques :
    ∫_tet Ni · Nj dV = V/20  (i ≠ j)
    ∫_tet Ni² dV      = V/10

→ M_e = ρV/20 · [[2I, I, I, I],
                   [I, 2I, I, I],
                   [I, I, 2I, I],
                   [I, I, I, 2I]]
où I est la matrice identité 3×3. Implémenté via produit de Kronecker.

Numérotation des nœuds
----------------------
Aucune convention imposée par l'élément, mais det(J) doit être > 0
(orientation trigonométrique du tétraèdre de référence) :

    N4
    |  \\
    |   N3
    |  /
    N1---N2

Le nœud N4 doit être du même côté que la normale sortante à la face N1-N2-N3.

References
----------
Bathe, « Finite Element Procedures », 2nd ed., chap. 5.
Cook et al., « Concepts and Applications of FEA », 4th ed., chap. 6.
Zienkiewicz & Taylor, « The FEM — Solid Mechanics », vol. 2, chap. 9.
"""

from __future__ import annotations

import numpy as np

from femsolver.core.element import Element
from femsolver.core.material import ElasticMaterial


class Tetra4(Element):
    """Tétraèdre linéaire isoparamétrique à 4 nœuds — élasticité 3D.

    4 nœuds, 3 DDL par nœud (ux, uy, uz) → matrices élémentaires 12×12.
    Déformation constante dans l'élément (CST 3D) → 1 point d'intégration.

    Ordre des DDL : [u1, v1, w1, u2, v2, w2, u3, v3, w3, u4, v4, w4].

    Parameters (via ``properties``)
    --------------------------------
    Aucun paramètre requis. Le dict peut être vide ``{}``.

    Raises
    ------
    ValueError
        Si det(J) ≤ 0 (nœuds coplanaires ou orientation incorrecte).
    ValueError
        Si nodes.shape ≠ (4, 3).

    Examples
    --------
    >>> import numpy as np
    >>> from femsolver.core.material import ElasticMaterial
    >>> mat = ElasticMaterial(E=210e9, nu=0.3, rho=7800)
    >>> nodes = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    >>> K_e = Tetra4().stiffness_matrix(mat, nodes, {})
    >>> K_e.shape
    (12, 12)
    """

    # Dérivées des fonctions de forme par rapport aux coordonnées naturelles
    # dN_nat[i, k] = ∂Nk/∂ξi  (shape : 3×4, constante pour Tetra4)
    #
    # N1=1-ξ-η-ζ, N2=ξ, N3=η, N4=ζ
    # ∂N/∂ξ = [-1, 1, 0, 0]
    # ∂N/∂η = [-1, 0, 1, 0]
    # ∂N/∂ζ = [-1, 0, 0, 1]
    _DN_NAT: np.ndarray = np.array(
        [[-1.0, 1.0, 0.0, 0.0],
         [-1.0, 0.0, 1.0, 0.0],
         [-1.0, 0.0, 0.0, 1.0]],
    )

    def dof_per_node(self) -> int:
        """3 DDL par nœud : ux, uy, uz."""
        return 3

    def n_nodes(self) -> int:
        """4 nœuds."""
        return 4

    # ------------------------------------------------------------------
    # Calcul du Jacobien et du volume
    # ------------------------------------------------------------------

    def _jacobian_and_volume(
        self,
        nodes: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Jacobien J (3×3) et volume V du tétraèdre.

        Parameters
        ----------
        nodes : np.ndarray, shape (4, 3)
            Coordonnées physiques [[x1,y1,z1], …, [x4,y4,z4]].

        Returns
        -------
        J : np.ndarray, shape (3, 3)
            Jacobien de la transformation référence → physique.
            J[i,:] = nœud_{i+1} − nœud_0 (vecteurs d'arête).
        volume : float
            Volume du tétraèdre (> 0 si nœuds en orientation positive).

        Raises
        ------
        ValueError
            Si det(J) ≤ 0 (tétraèdre dégénéré ou nœuds mal orientés).

        Notes
        -----
        J = dN_nat · nodes  (3,4) @ (4,3) → (3,3)
        J[i,:] = Σ_k dN_nat[i,k] · nodes[k,:]

        En particulier : J[0] = nodes[1]−nodes[0],
                         J[1] = nodes[2]−nodes[0],
                         J[2] = nodes[3]−nodes[0].

        Volume = det(J) / 6.
        """
        J = self._DN_NAT @ nodes  # (3,4) @ (4,3) = (3,3)
        det_J = np.linalg.det(J)

        if abs(det_J) < 1e-30:
            raise ValueError(
                f"Jacobien singulier (det ≈ 0) — nœuds coplanaires :\n{nodes}"
            )
        if det_J < 0.0:
            raise ValueError(
                f"det(J) = {det_J:.6g} < 0 — vérifier l'ordre des nœuds du "
                f"tétraèdre (orientation positive requise)."
            )

        return J, det_J / 6.0

    # ------------------------------------------------------------------
    # Matrice B (6×12)
    # ------------------------------------------------------------------

    def _strain_displacement_matrix(
        self,
        nodes: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Matrice déformation–déplacement B (6×12) et volume V.

        Parameters
        ----------
        nodes : np.ndarray, shape (4, 3)
            Coordonnées physiques des nœuds.

        Returns
        -------
        B : np.ndarray, shape (6, 12)
            Matrice constante : ε = B · u_e.
        volume : float
            Volume du tétraèdre [m³].

        Notes
        -----
        Dérivées physiques obtenues par transformation Jacobienne :
            dN_phys = J⁻¹ · dN_nat    (3×3 @ 3×4 → 3×4)

        Vecteur de déformation (notation de Voigt) :
            ε = [εxx, εyy, εzz, γyz, γxz, γxy]

        Structure de B pour le nœud i (colonnes 3i, 3i+1, 3i+2) :
            Row 0 (εxx) : [∂Ni/∂x, 0, 0]
            Row 1 (εyy) : [0, ∂Ni/∂y, 0]
            Row 2 (εzz) : [0, 0, ∂Ni/∂z]
            Row 3 (γyz) : [0, ∂Ni/∂z, ∂Ni/∂y]
            Row 4 (γxz) : [∂Ni/∂z, 0, ∂Ni/∂x]
            Row 5 (γxy) : [∂Ni/∂y, ∂Ni/∂x, 0]
        """
        if nodes.shape != (4, 3):
            raise ValueError(
                f"Tetra4 attend nodes.shape == (4, 3), reçu {nodes.shape}"
            )

        J, volume = self._jacobian_and_volume(nodes)
        # dN_phys[i, k] = ∂Nk/∂x_i  (i=0→x, 1→y, 2→z)
        dN_phys = np.linalg.solve(J, self._DN_NAT)  # (3,3)⁻¹ · (3,4) = (3,4)

        B = np.zeros((6, 12))
        for i in range(4):
            c = 3 * i                              # indice de colonne pour ux_i
            dNx = dN_phys[0, i]                    # ∂Ni/∂x
            dNy = dN_phys[1, i]                    # ∂Ni/∂y
            dNz = dN_phys[2, i]                    # ∂Ni/∂z
            B[0, c    ] = dNx                      # εxx = ∂ux/∂x
            B[1, c + 1] = dNy                      # εyy = ∂uy/∂y
            B[2, c + 2] = dNz                      # εzz = ∂uz/∂z
            B[3, c + 1] = dNz                      # γyz : ∂uy/∂z
            B[3, c + 2] = dNy                      # γyz : ∂uz/∂y
            B[4, c    ] = dNz                      # γxz : ∂ux/∂z
            B[4, c + 2] = dNx                      # γxz : ∂uz/∂x
            B[5, c    ] = dNy                      # γxy : ∂ux/∂y
            B[5, c + 1] = dNx                      # γxy : ∂uy/∂x

        return B, volume

    # ------------------------------------------------------------------
    # Matrices élémentaires
    # ------------------------------------------------------------------

    def stiffness_matrix(
        self,
        material: ElasticMaterial,
        nodes: np.ndarray,
        properties: dict,
    ) -> np.ndarray:
        """Matrice de rigidité élémentaire K_e (12×12).

        Parameters
        ----------
        material : ElasticMaterial
            Matériau (E, nu utilisés pour construire D 6×6).
        nodes : np.ndarray, shape (4, 3)
            Coordonnées des 4 nœuds en repère global.
        properties : dict
            Non utilisé pour Tetra4 (peut être ``{}``).

        Returns
        -------
        K_e : np.ndarray, shape (12, 12)
            Matrice de rigidité symétrique.

        Notes
        -----
        B étant constante sur tout l'élément (fonctions de forme linéaires),
        l'intégrale se réduit à une multiplication par le volume :

            K_e = Bᵀ · D · B · V

        C'est l'analogue 3D du Tri3 (CST) pour lequel K_e = Bᵀ D B A t.
        Aucune quadrature de Gauss n'est nécessaire.

        Examples
        --------
        >>> mat = ElasticMaterial(E=1.0, nu=0.0, rho=1.0)
        >>> nodes = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
        >>> K_e = Tetra4().stiffness_matrix(mat, nodes, {})
        >>> K_e.shape
        (12, 12)
        """
        D = material.elasticity_matrix_3d()
        B, volume = self._strain_displacement_matrix(nodes)
        K_e = B.T @ D @ B * volume
        return K_e

    def mass_matrix(
        self,
        material: ElasticMaterial,
        nodes: np.ndarray,
        properties: dict,
    ) -> np.ndarray:
        """Matrice de masse consistante M_e (12×12).

        Parameters
        ----------
        material : ElasticMaterial
            Matériau (rho utilisé).
        nodes : np.ndarray, shape (4, 3)
            Coordonnées des nœuds.
        properties : dict
            Non utilisé pour Tetra4.

        Returns
        -------
        M_e : np.ndarray, shape (12, 12)
            Matrice de masse consistante symétrique.

        Notes
        -----
        En utilisant les coordonnées volumiques (coordonnées barycentriques),
        l'intégrale de deux fonctions de forme s'exprime analytiquement :

            ∫_tet Ni · Nj dV = V/20   (i ≠ j)
            ∫_tet Ni²   dV  = V/10    (i = j)

        La matrice scalaire 4×4 est :
            m_scalar = ρV/20 · [[2,1,1,1], [1,2,1,1], [1,1,2,1], [1,1,1,2]]

        Pour 3 DDL par nœud, on étend par produit de Kronecker :
            M_e = kron(m_scalar, I_3)

        Cela garantit le découplage des directions x, y, z et la conservation
        de la masse : la somme de chaque ligne vaut ρV/4.
        """
        _, volume = self._jacobian_and_volume(nodes)
        m_scalar = (material.rho * volume / 20.0) * (
            np.ones((4, 4)) + np.eye(4)   # diag=2, off-diag=1 → × ρV/20
        )
        return np.kron(m_scalar, np.eye(3))

    # ------------------------------------------------------------------
    # Post-traitement
    # ------------------------------------------------------------------

    def strain(
        self,
        nodes: np.ndarray,
        u_e: np.ndarray,
    ) -> np.ndarray:
        """Vecteur de déformations ε = B · u_e (constant dans l'élément).

        Parameters
        ----------
        nodes : np.ndarray, shape (4, 3)
            Coordonnées nodales.
        u_e : np.ndarray, shape (12,)
            Déplacements élémentaires [u1,v1,w1, u2,v2,w2, …].

        Returns
        -------
        epsilon : np.ndarray, shape (6,)
            [εxx, εyy, εzz, γyz, γxz, γxy].
        """
        B, _ = self._strain_displacement_matrix(nodes)
        return B @ u_e

    def stress(
        self,
        material: ElasticMaterial,
        nodes: np.ndarray,
        u_e: np.ndarray,
    ) -> np.ndarray:
        """Vecteur de contraintes σ = D · B · u_e (constant dans l'élément).

        Parameters
        ----------
        material : ElasticMaterial
            Matériau (E, nu).
        nodes : np.ndarray, shape (4, 3)
            Coordonnées nodales.
        u_e : np.ndarray, shape (12,)
            Déplacements élémentaires.

        Returns
        -------
        sigma : np.ndarray, shape (6,)
            [σxx, σyy, σzz, τyz, τxz, τxy] [Pa].
        """
        D = material.elasticity_matrix_3d()
        return D @ self.strain(nodes, u_e)

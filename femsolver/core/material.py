"""Lois de comportement élastique linéaire isotrope."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ElasticMaterial:
    """Matériau élastique linéaire isotrope.

    Parameters
    ----------
    E : float
        Module d'Young [Pa]. Doit être strictement positif.
    nu : float
        Coefficient de Poisson [-]. Doit vérifier -1 < nu < 0.5.
    rho : float
        Masse volumique [kg/m³]. Doit être strictement positive.

    Examples
    --------
    >>> steel = ElasticMaterial(E=210e9, nu=0.3, rho=7800)
    >>> steel.E
    210000000000.0
    """

    E: float
    nu: float
    rho: float

    def __post_init__(self) -> None:
        if self.E <= 0:
            raise ValueError(f"Le module d'Young doit être > 0, reçu E={self.E}")
        if not (-1.0 < self.nu < 0.5):
            raise ValueError(f"Poisson doit être dans ]-1, 0.5[, reçu nu={self.nu}")
        if self.rho <= 0:
            raise ValueError(f"La densité doit être > 0, reçu rho={self.rho}")

    def elasticity_matrix_plane_stress(self) -> np.ndarray:
        """Matrice de comportement D en contrainte plane (3×3).

        Hypothèse σ_zz = σ_xz = σ_yz = 0 (plaque mince).

        Returns
        -------
        D : np.ndarray, shape (3, 3)
            Matrice reliant déformations et contraintes : σ = D ε.

        Notes
        -----
        D = E/(1-ν²) * [[1, ν, 0],
                         [ν, 1, 0],
                         [0, 0, (1-ν)/2]]
        """
        E, nu = self.E, self.nu
        factor = E / (1.0 - nu**2)
        return factor * np.array(
            [
                [1.0, nu, 0.0],
                [nu, 1.0, 0.0],
                [0.0, 0.0, (1.0 - nu) / 2.0],
            ]
        )

    def elasticity_matrix_3d(self) -> np.ndarray:
        """Matrice de comportement D en élasticité 3D isotrope (6×6).

        Loi de Hooke généralisée en notation de Voigt :
        σ = D ε,  avec ε = [εxx, εyy, εzz, γyz, γxz, γxy].

        Returns
        -------
        D : np.ndarray, shape (6, 6)
            Matrice symétrique définie positive.

        Notes
        -----
        En introduisant les constantes de Lamé λ et μ :

            λ = E·ν / ((1+ν)(1−2ν))   (premier coefficient de Lamé)
            μ = E / (2(1+ν))           (module de cisaillement)

        La matrice prend la forme :

            D = [[λ+2μ, λ,    λ,    0, 0, 0],
                 [λ,    λ+2μ, λ,    0, 0, 0],
                 [λ,    λ,    λ+2μ, 0, 0, 0],
                 [0,    0,    0,    μ, 0, 0],
                 [0,    0,    0,    0, μ, 0],
                 [0,    0,    0,    0, 0, μ]]

        Le cas 3D inclut la contrainte plane (σzz ≠ 0) et la déformation
        transverse (εzz ≠ 0) — aucune hypothèse de réduction dimensionnelle.

        Examples
        --------
        >>> steel = ElasticMaterial(E=210e9, nu=0.3, rho=7800)
        >>> D = steel.elasticity_matrix_3d()
        >>> D.shape
        (6, 6)
        """
        E, nu = self.E, self.nu
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))   # λ de Lamé
        mu  = E / (2.0 * (1.0 + nu))                       # μ = G (cisaillement)
        D = np.zeros((6, 6))
        # Bloc normal (contraintes/déformations directes)
        D[0, 0] = D[1, 1] = D[2, 2] = lam + 2.0 * mu
        D[0, 1] = D[1, 0] = lam
        D[0, 2] = D[2, 0] = lam
        D[1, 2] = D[2, 1] = lam
        # Bloc cisaillement (γ_ij = 2ε_ij, facteur μ)
        D[3, 3] = D[4, 4] = D[5, 5] = mu
        return D

    def elasticity_matrix_plane_strain(self) -> np.ndarray:
        """Matrice de comportement D en déformation plane (3×3).

        Hypothèse ε_zz = ε_xz = ε_yz = 0 (pièce longue).

        Returns
        -------
        D : np.ndarray, shape (3, 3)
            Matrice reliant déformations et contraintes : σ = D ε.

        Notes
        -----
        D = E/((1+ν)(1-2ν)) * [[1-ν, ν,   0         ],
                                 [ν,   1-ν, 0         ],
                                 [0,   0,   (1-2ν)/2  ]]
        """
        E, nu = self.E, self.nu
        factor = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
        return factor * np.array(
            [
                [1.0 - nu, nu, 0.0],
                [nu, 1.0 - nu, 0.0],
                [0.0, 0.0, (1.0 - 2.0 * nu) / 2.0],
            ]
        )

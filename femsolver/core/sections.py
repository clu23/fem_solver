"""Sections transversales de poutres — propriétés géométriques.

Convention des axes (repère centroïdal G)
------------------------------------------

.. code-block:: text

    y ↑  (hauteur, flexion forte dans Beam2D)
      │
    G ┼──── z  (largeur, flexion faible)

- Iz = ∫ y² dA  — flexion dans le plan xy (Beam2D, « flexion forte »)
- Iy = ∫ z² dA  — flexion hors plan (« flexion faible »)
- Iyz = ∫ y·z dA — produit d'inertie (nul pour les sections à un axe de symétrie)
- J              — constante de torsion de Saint-Venant

Axes principaux (Mohr)
-----------------------
- tan(2α) = −2·Iyz / (Iz − Iy)
- I₁ = (Iz+Iy)/2 + √[((Iz−Iy)/2)² + Iyz²]  (max)
- I₂ = (Iz+Iy)/2 − √[((Iz−Iy)/2)² + Iyz²]  (min)

Décomposition en rectangles élémentaires
-----------------------------------------
Les sections composées (I, C, L, RHS) sont décomposées en un ensemble de
rectangles non chevauchants.  Le centroïde G et les moments quadratiques
sont calculés automatiquement par le théorème de Huygens-Steiner.

Références
----------
Pilkey W.D., «Formulas for Stress, Strain, and Structural Matrices», 2nd ed.
Timoshenko S.P., «Strength of Materials», Part 1.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.axes import Axes


# ──────────────────────────────────────────────────────────────────────────────
# Brique élémentaire
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class _Rect:
    """Rectangle élémentaire défini par son centroïde et ses dimensions.

    Paramètres dans un repère de référence arbitraire (pas nécessairement
    centroïdal) — c'est ``_props_from_rects`` qui centre.

    Parameters
    ----------
    y_c, z_c : float
        Coordonnées du centroïde [m].
    h : float
        Dimension selon y [m].
    b : float
        Dimension selon z [m].
    """

    y_c: float
    z_c: float
    h: float
    b: float

    @property
    def area(self) -> float:
        return self.h * self.b

    @property
    def Iz_self(self) -> float:
        """b·h³/12 — Iz centroïdal propre."""
        return self.b * self.h ** 3 / 12.0

    @property
    def Iy_self(self) -> float:
        """h·b³/12 — Iy centroïdal propre."""
        return self.h * self.b ** 3 / 12.0

    @property
    def y_top(self) -> float:
        return self.y_c + self.h / 2.0

    @property
    def y_bot(self) -> float:
        return self.y_c - self.h / 2.0

    @property
    def z_right(self) -> float:
        return self.z_c + self.b / 2.0

    @property
    def z_left(self) -> float:
        return self.z_c - self.b / 2.0

    def vertices_yz(self) -> list[tuple[float, float]]:
        """Coins (y, z) dans le sens antihoraire."""
        return [
            (self.y_bot, self.z_left),
            (self.y_bot, self.z_right),
            (self.y_top, self.z_right),
            (self.y_top, self.z_left),
        ]


# ──────────────────────────────────────────────────────────────────────────────
# Interface abstraite
# ──────────────────────────────────────────────────────────────────────────────

class Section(ABC):
    """Interface pour toute section transversale de poutre.

    Toutes les propriétés sont exprimées dans le **repère centroïdal** G.

    Notes
    -----
    ``Beam2D`` utilise :attr:`Iz` pour la matrice de rigidité de flexion.
    Les autres propriétés (``Iy``, ``Iyz``, ``J``) seront exploitées par
    les éléments Beam3D futurs et le post-traitement des contraintes.
    """

    # ── Propriétés abstraites ─────────────────────────────────────────────────

    @property
    @abstractmethod
    def area(self) -> float:
        """Aire A [m²]."""

    @property
    @abstractmethod
    def Iz(self) -> float:
        """Iz = ∫ y² dA [m⁴] — flexion forte (axe z)."""

    @property
    @abstractmethod
    def Iy(self) -> float:
        """Iy = ∫ z² dA [m⁴] — flexion faible (axe y)."""

    @property
    @abstractmethod
    def Iyz(self) -> float:
        """Iyz = ∫ y·z dA [m⁴] — produit d'inertie centroïdal."""

    @property
    @abstractmethod
    def J(self) -> float:
        """Constante de torsion de Saint-Venant [m⁴]."""

    @property
    @abstractmethod
    def y_max(self) -> float:
        """Fibre extrême supérieure : y_max > 0 [m] depuis G."""

    @property
    @abstractmethod
    def y_min(self) -> float:
        """Fibre extrême inférieure : y_min < 0 [m] depuis G."""

    @property
    @abstractmethod
    def z_max(self) -> float:
        """Fibre extrême droite : z_max > 0 [m] depuis G."""

    @property
    @abstractmethod
    def z_min(self) -> float:
        """Fibre extrême gauche : z_min < 0 [m] depuis G."""

    @abstractmethod
    def shear_correction_factor(self, nu: float) -> float:
        """Facteur de correction de cisaillement de Timoshenko κ [-].

        κ est utilisé pour calculer la rigidité de cisaillement effective :
        GA_s = G·κ·A.  Dépend de la forme de la section et éventuellement
        du coefficient de Poisson ν.

        Parameters
        ----------
        nu : float
            Coefficient de Poisson du matériau [-].

        Returns
        -------
        float
            κ ∈ (0, 1].

        References
        ----------
        Cowper G.R. (1966), *The Shear Coefficient in Timoshenko's Beam
        Theory*, J. Appl. Mech. 33(2), 335–340.
        """
        raise NotImplementedError  # pragma: no cover

    # ── Propriétés dérivées (concrètes) ──────────────────────────────────────

    @property
    def I_principal(self) -> tuple[float, float]:
        """Moments principaux (I₁ ≥ I₂) [m⁴].

        Calculés par le cercle de Mohr :

            I₁, I₂ = (Iz+Iy)/2 ± √[((Iz−Iy)/2)² + Iyz²]

        Returns
        -------
        tuple[float, float]
            (I₁, I₂) avec I₁ ≥ I₂.

        Examples
        --------
        >>> sec = CircularSection(radius=0.05)
        >>> I1, I2 = sec.I_principal
        >>> abs(I1 - I2) / I1 < 1e-12   # section isotrope
        True
        """
        center = (self.Iz + self.Iy) * 0.5
        r = math.sqrt(((self.Iz - self.Iy) * 0.5) ** 2 + self.Iyz ** 2)
        return center + r, center - r

    @property
    def alpha_principal(self) -> float:
        """Angle α [rad] de l'axe I₁ par rapport à l'axe z (sens antihoraire).

        tan(2α) = −2·Iyz / (Iz − Iy).

        Returns
        -------
        float
            α ∈ (−π/2, π/2].  0 si la section est isotrope (Iz = Iy, Iyz = 0).

        Examples
        --------
        >>> import math
        >>> sec = LSection(width=0.1, height=0.1, thickness=0.01)
        >>> abs(sec.alpha_principal) > 0   # cornière non symétrique
        True
        """
        dI = self.Iz - self.Iy
        if abs(dI) < 1e-30 and abs(self.Iyz) < 1e-30:
            return 0.0
        return 0.5 * math.atan2(-2.0 * self.Iyz, dI)

    def sigma_bending(self, Mz: float, y: float) -> float:
        """Contrainte de flexion σ = Mz·y / Iz [Pa].

        Valide pour la flexion plane (Beam2D), y mesuré depuis G.

        Parameters
        ----------
        Mz : float
            Moment fléchissant [N·m].
        y : float
            Coordonnée centroïdale [m].

        Returns
        -------
        float
            Contrainte σ [Pa] (positive = traction).
        """
        return Mz * y / self.Iz

    # ── Visualisation ─────────────────────────────────────────────────────────

    @abstractmethod
    def _draw_patches(self, ax: "Axes", *, color: str, alpha: float) -> None:
        """Dessine le remplissage de la section sur ``ax``.

        Les coordonnées sont dans le repère centroïdal (G à l'origine).
        L'axe horizontal du graphique représente z, l'axe vertical y.
        """

    def plot(
        self,
        ax: "Axes | None" = None,
        *,
        title: str = "",
        show_principal_axes: bool = True,
        color: str = "steelblue",
        fill_alpha: float = 0.45,
    ) -> "Axes":
        """Visualise la section avec le centroïde G et les axes principaux.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes existants ; crée une nouvelle figure si None.
        title : str
            Titre du graphique.
        show_principal_axes : bool
            Afficher les axes principaux d'inertie (défaut True).
        color : str
            Couleur de remplissage.
        fill_alpha : float
            Transparence du remplissage.

        Returns
        -------
        matplotlib.axes.Axes
            L'objet Axes utilisé.

        Notes
        -----
        Axe horizontal du graphique = z, axe vertical = y
        (convention mécanique des structures).
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(5, 5))

        self._draw_patches(ax, color=color, alpha=fill_alpha)

        # Centroïde G
        ax.plot(0.0, 0.0, "k+", markersize=14, markeredgewidth=2.0,
                zorder=6, label="G")

        if show_principal_axes:
            alpha_rad = self.alpha_principal
            I1, I2 = self.I_principal
            span = 1.3 * max(self.y_max, -self.y_min, self.z_max, -self.z_min)
            c_a, s_a = math.cos(alpha_rad), math.sin(alpha_rad)

            # Direction de I₁ dans (z, y) : (cos α, sin α)
            ax.plot(
                [-span * c_a, span * c_a],
                [-span * s_a, span * s_a],
                color="crimson", lw=1.8, ls="--",
                label=f"Axe 1 (I₁ = {I1:.3e} m⁴)",
            )
            # Direction de I₂ dans (z, y) : (−sin α, cos α)
            ax.plot(
                [span * s_a, -span * s_a],
                [-span * c_a, span * c_a],
                color="forestgreen", lw=1.8, ls=":",
                label=f"Axe 2 (I₂ = {I2:.3e} m⁴)",
            )

        ax.set_aspect("equal")
        ax.set_xlabel("z [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(title or type(self).__name__)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)
        return ax


# ──────────────────────────────────────────────────────────────────────────────
# Sections circulaires
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CircularSection(Section):
    """Section circulaire pleine.

    Parameters
    ----------
    radius : float
        Rayon r [m].

    Notes
    -----
    Formules exactes :

    - A = π r²
    - Iz = Iy = π r⁴ / 4
    - J  = π r⁴ / 2  (moment polaire — section circulaire isotrope)
    - Iyz = 0

    Examples
    --------
    >>> import math
    >>> sec = CircularSection(radius=0.05)
    >>> abs(sec.area - math.pi * 0.05**2) < 1e-14
    True
    """

    radius: float

    def __post_init__(self) -> None:
        if self.radius <= 0:
            raise ValueError(f"radius doit être > 0, reçu {self.radius}")

    @property
    def area(self) -> float:
        return math.pi * self.radius ** 2

    @property
    def Iz(self) -> float:
        return math.pi * self.radius ** 4 / 4.0

    @property
    def Iy(self) -> float:
        return self.Iz

    @property
    def Iyz(self) -> float:
        return 0.0

    @property
    def J(self) -> float:
        return math.pi * self.radius ** 4 / 2.0

    @property
    def y_max(self) -> float:
        return self.radius

    @property
    def y_min(self) -> float:
        return -self.radius

    @property
    def z_max(self) -> float:
        return self.radius

    @property
    def z_min(self) -> float:
        return -self.radius

    def shear_correction_factor(self, nu: float) -> float:
        """κ pour section circulaire pleine — Cowper (1966), Eq. 28.

        κ = 6(1+ν) / (7+6ν)
        """
        return 6.0 * (1.0 + nu) / (7.0 + 6.0 * nu)

    def _draw_patches(self, ax: "Axes", *, color: str, alpha: float) -> None:
        import matplotlib.patches as mpatches
        circle = mpatches.Circle(
            (0.0, 0.0), self.radius,
            facecolor=color, alpha=alpha, edgecolor="navy", linewidth=1.4,
        )
        ax.add_patch(circle)
        r = self.radius * 1.25
        ax.set_xlim(-r, r)
        ax.set_ylim(-r, r)


@dataclass(frozen=True)
class HollowCircularSection(Section):
    """Section circulaire creuse (tube).

    Parameters
    ----------
    outer_radius : float
        Rayon extérieur R [m].
    inner_radius : float
        Rayon intérieur r [m] (r < R).

    Notes
    -----
    - A = π(R²−r²)
    - Iz = Iy = π(R⁴−r⁴)/4
    - J = π(R⁴−r⁴)/2
    - Iyz = 0

    Examples
    --------
    >>> sec = HollowCircularSection(outer_radius=0.05, inner_radius=0.04)
    >>> sec.area > 0
    True
    """

    outer_radius: float
    inner_radius: float

    def __post_init__(self) -> None:
        if self.outer_radius <= 0:
            raise ValueError(f"outer_radius doit être > 0, reçu {self.outer_radius}")
        if self.inner_radius <= 0:
            raise ValueError(f"inner_radius doit être > 0, reçu {self.inner_radius}")
        if self.inner_radius >= self.outer_radius:
            raise ValueError(
                f"inner_radius ({self.inner_radius}) doit être < outer_radius ({self.outer_radius})"
            )

    @property
    def area(self) -> float:
        R, r = self.outer_radius, self.inner_radius
        return math.pi * (R ** 2 - r ** 2)

    @property
    def Iz(self) -> float:
        R, r = self.outer_radius, self.inner_radius
        return math.pi * (R ** 4 - r ** 4) / 4.0

    @property
    def Iy(self) -> float:
        return self.Iz

    @property
    def Iyz(self) -> float:
        return 0.0

    @property
    def J(self) -> float:
        R, r = self.outer_radius, self.inner_radius
        return math.pi * (R ** 4 - r ** 4) / 2.0

    @property
    def y_max(self) -> float:
        return self.outer_radius

    @property
    def y_min(self) -> float:
        return -self.outer_radius

    @property
    def z_max(self) -> float:
        return self.outer_radius

    @property
    def z_min(self) -> float:
        return -self.outer_radius

    def shear_correction_factor(self, nu: float) -> float:
        """κ pour section circulaire creuse — Cowper (1966), Eq. 29.

        Avec m = r_i / r_o :

            κ = 6(1+ν)(1+m²)² / [(7+6ν)(1+m²)² + (20+12ν)m²]

        Limite m → 0 (section pleine) : κ → 6(1+ν)/(7+6ν)  (formule circulaire).
        Limite m → 1 (paroi très mince) : κ → 2(1+ν)/(4+3ν).
        """
        m = self.inner_radius / self.outer_radius
        m2 = m * m
        num = 6.0 * (1.0 + nu) * (1.0 + m2) ** 2
        den = (7.0 + 6.0 * nu) * (1.0 + m2) ** 2 + (20.0 + 12.0 * nu) * m2
        return num / den

    def _draw_patches(self, ax: "Axes", *, color: str, alpha: float) -> None:
        import matplotlib.patches as mpatches
        outer = mpatches.Circle(
            (0.0, 0.0), self.outer_radius,
            facecolor=color, alpha=alpha, edgecolor="navy", linewidth=1.4,
        )
        inner = mpatches.Circle(
            (0.0, 0.0), self.inner_radius,
            facecolor="white", alpha=1.0, edgecolor="navy", linewidth=1.4,
        )
        ax.add_patch(outer)
        ax.add_patch(inner)
        R = self.outer_radius * 1.25
        ax.set_xlim(-R, R)
        ax.set_ylim(-R, R)


# ──────────────────────────────────────────────────────────────────────────────
# Sections rectangulaires composées — classe de base
# ──────────────────────────────────────────────────────────────────────────────

class RectangleBasedSection(Section, ABC):
    """Classe de base pour les sections décomposables en rectangles.

    Les sous-classes doivent implémenter :meth:`_rects` qui retourne une
    liste de :class:`_Rect` dans un repère arbitraire.  Le centroïde G et
    les propriétés mécaniques sont calculés automatiquement.

    Torsion par défaut
    ------------------
    Le module J par défaut utilise l'approximation section mince ouverte :

        J ≈ (1/3) Σ bmax_i · tmin_i³

    Les sous-classes (RectangularSection, HollowRectangularSection)
    surchargent cette propriété avec des formules plus précises.
    """

    @abstractmethod
    def _rects(self) -> list[_Rect]:
        """Liste des rectangles constitutifs dans le repère de référence."""

    @staticmethod
    def _props_from_rects(
        rects: list[_Rect],
    ) -> tuple[float, float, float, float, float, float, float, float, float, float]:
        """Calcule (A, y_G, z_G, Iz, Iy, Iyz, y_max, y_min, z_max, z_min)
        depuis une liste de rectangles par Huygens-Steiner.
        """
        A = sum(r.area for r in rects)
        y_G = sum(r.area * r.y_c for r in rects) / A
        z_G = sum(r.area * r.z_c for r in rects) / A

        Iz = sum(r.Iz_self + r.area * (r.y_c - y_G) ** 2 for r in rects)
        Iy = sum(r.Iy_self + r.area * (r.z_c - z_G) ** 2 for r in rects)
        Iyz = sum(r.area * (r.y_c - y_G) * (r.z_c - z_G) for r in rects)

        y_max = max(r.y_top for r in rects) - y_G
        y_min = min(r.y_bot for r in rects) - y_G
        z_max = max(r.z_right for r in rects) - z_G
        z_min = min(r.z_left for r in rects) - z_G

        return A, y_G, z_G, Iz, Iy, Iyz, y_max, y_min, z_max, z_min

    # ── Propriétés issues de la décomposition ────────────────────────────────

    @property
    def _p(self) -> tuple[float, ...]:
        return self._props_from_rects(self._rects())

    @property
    def area(self) -> float:
        return self._p[0]

    @property
    def Iz(self) -> float:
        return self._p[3]

    @property
    def Iy(self) -> float:
        return self._p[4]

    @property
    def Iyz(self) -> float:
        return self._p[5]

    @property
    def y_max(self) -> float:
        return self._p[6]

    @property
    def y_min(self) -> float:
        return self._p[7]

    @property
    def z_max(self) -> float:
        return self._p[8]

    @property
    def z_min(self) -> float:
        return self._p[9]

    @property
    def J(self) -> float:
        """Approximation section mince ouverte : J ≈ (1/3) Σ blong · t³ thin."""
        return (1.0 / 3.0) * sum(
            max(r.h, r.b) * min(r.h, r.b) ** 3 for r in self._rects()
        )

    def _draw_patches(self, ax: "Axes", *, color: str, alpha: float) -> None:
        """Dessine chaque rectangle en repère centroïdal (z, y) → (xplot, yplot)."""
        import matplotlib.patches as mpatches

        rects = self._rects()
        A = sum(r.area for r in rects)
        y_G = sum(r.area * r.y_c for r in rects) / A
        z_G = sum(r.area * r.z_c for r in rects) / A

        for r in rects:
            # Vertices (y, z) → translate to centroidal → swap to (xplot=z, yplot=y)
            verts = [
                (v[1] - z_G, v[0] - y_G)   # (z_centroidal, y_centroidal)
                for v in r.vertices_yz()
            ]
            patch = mpatches.Polygon(
                verts, closed=True,
                facecolor=color, alpha=alpha,
                edgecolor="navy", linewidth=1.2,
            )
            ax.add_patch(patch)

        ax.autoscale_view()


# ──────────────────────────────────────────────────────────────────────────────
# Section rectangulaire pleine
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RectangularSection(RectangleBasedSection):
    """Section rectangulaire pleine b × h.

    Parameters
    ----------
    width : float
        Dimension en z (largeur b) [m].
    height : float
        Dimension en y (hauteur h) [m].

    Notes
    -----
    - A   = b·h
    - Iz  = b·h³/12  (flexion forte)
    - Iy  = h·b³/12  (flexion faible)
    - Iyz = 0
    - J   ≈ (b·h³/3)·(1 − 0.63·h/b + 0.052·(h/b)⁵)  b ≥ h
      [approximation de Saint-Venant, erreur < 1 % pour b/h ≥ 1]

    Examples
    --------
    >>> sec = RectangularSection(width=0.1, height=0.2)
    >>> abs(sec.Iz - 0.1 * 0.2**3 / 12) < 1e-14
    True
    """

    width: float    # b
    height: float   # h

    def __post_init__(self) -> None:
        if self.width <= 0:
            raise ValueError(f"width doit être > 0, reçu {self.width}")
        if self.height <= 0:
            raise ValueError(f"height doit être > 0, reçu {self.height}")

    def _rects(self) -> list[_Rect]:
        return [_Rect(y_c=0.0, z_c=0.0, h=self.height, b=self.width)]

    @property
    def J(self) -> float:
        """Approximation de Saint-Venant pour rectangle plein.

        J ≈ (b·h³/3)·(1 − 0.63·h/b + 0.052·(h/b)⁵)  pour b ≥ h.

        Erreur < 1 % pour b/h ≥ 1 (< 4 % pour b = h).
        """
        b, h = self.width, self.height
        if b < h:
            b, h = h, b   # on veut b = dimension longue
        ratio = h / b
        return (b * h ** 3 / 3.0) * (1.0 - 0.63 * ratio + 0.052 * ratio ** 5)

    def shear_correction_factor(self, nu: float) -> float:
        """κ pour section rectangulaire pleine — Cowper (1966), Eq. 30.

        κ = 10(1+ν) / (12+11ν)

        Indépendant du rapport h/b (valide pour toute section rectangulaire
        pleine soumise à un cisaillement uniforme dans la direction de h).
        """
        return 10.0 * (1.0 + nu) / (12.0 + 11.0 * nu)


# ──────────────────────────────────────────────────────────────────────────────
# Section rectangulaire creuse (RHS)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class HollowRectangularSection(Section):
    """Section rectangulaire creuse à paroi uniforme (RHS / tube carré-rectangle).

    Parameters
    ----------
    outer_width : float
        Largeur extérieure B [m].
    outer_height : float
        Hauteur extérieure H [m].
    thickness : float
        Épaisseur de paroi t [m] (uniforme, t < min(B,H)/2).

    Notes
    -----
    - A   = B·H − (B−2t)·(H−2t)
    - Iz  = (B·H³ − (B−2t)·(H−2t)³) / 12
    - Iy  = (H·B³ − (H−2t)·(B−2t)³) / 12
    - Iyz = 0  (double symétrie)
    - J   = 2·t·(B−t)²·(H−t)² / (B+H−2t)  [formule de Bredt section fermée]

    Examples
    --------
    >>> sec = HollowRectangularSection(outer_width=0.1, outer_height=0.15, thickness=0.005)
    >>> sec.area > 0
    True
    """

    outer_width: float    # B
    outer_height: float   # H
    thickness: float      # t

    def __post_init__(self) -> None:
        if self.outer_width <= 0:
            raise ValueError(f"outer_width doit être > 0, reçu {self.outer_width}")
        if self.outer_height <= 0:
            raise ValueError(f"outer_height doit être > 0, reçu {self.outer_height}")
        if self.thickness <= 0:
            raise ValueError(f"thickness doit être > 0, reçu {self.thickness}")
        if self.thickness >= min(self.outer_width, self.outer_height) / 2.0:
            raise ValueError(
                f"thickness ({self.thickness}) doit être < min(B,H)/2 = "
                f"{min(self.outer_width, self.outer_height)/2}"
            )

    @property
    def _inner_w(self) -> float:
        return self.outer_width - 2.0 * self.thickness

    @property
    def _inner_h(self) -> float:
        return self.outer_height - 2.0 * self.thickness

    @property
    def area(self) -> float:
        B, H, t = self.outer_width, self.outer_height, self.thickness
        return B * H - self._inner_w * self._inner_h

    @property
    def Iz(self) -> float:
        B, H = self.outer_width, self.outer_height
        return (B * H ** 3 - self._inner_w * self._inner_h ** 3) / 12.0

    @property
    def Iy(self) -> float:
        B, H = self.outer_width, self.outer_height
        return (H * B ** 3 - self._inner_h * self._inner_w ** 3) / 12.0

    @property
    def Iyz(self) -> float:
        return 0.0

    @property
    def J(self) -> float:
        """Formule de Bredt pour section rectangulaire fermée à paroi uniforme.

        J = 2·t·(B−t)²·(H−t)² / (B+H−2t)
        """
        B, H, t = self.outer_width, self.outer_height, self.thickness
        return 2.0 * t * (B - t) ** 2 * (H - t) ** 2 / (B + H - 2.0 * t)

    @property
    def y_max(self) -> float:
        return self.outer_height / 2.0

    @property
    def y_min(self) -> float:
        return -self.outer_height / 2.0

    @property
    def z_max(self) -> float:
        return self.outer_width / 2.0

    @property
    def z_min(self) -> float:
        return -self.outer_width / 2.0

    def _draw_patches(self, ax: "Axes", *, color: str, alpha: float) -> None:
        import matplotlib.patches as mpatches
        B, H = self.outer_width, self.outer_height
        outer = mpatches.Rectangle(
            (-B / 2.0, -H / 2.0), B, H,
            facecolor=color, alpha=alpha, edgecolor="navy", linewidth=1.4,
        )
        iw, ih = self._inner_w, self._inner_h
        inner = mpatches.Rectangle(
            (-iw / 2.0, -ih / 2.0), iw, ih,
            facecolor="white", alpha=1.0, edgecolor="navy", linewidth=1.2,
        )
        ax.add_patch(outer)
        ax.add_patch(inner)
        ax.autoscale_view()

    def shear_correction_factor(self, nu: float) -> float:
        """κ pour section rectangulaire creuse (RHS) — méthode de l'âme.

        La force tranchante est reprise par les deux âmes verticales
        (épaisseur t, hauteur H extérieure) :

            A_s = 2 · t · H
            κ = A_s / A

        Cette approximation est conservative et couramment utilisée
        en bureau d'études (Eurocode 3).  Indépendant de ν.
        """
        return 2.0 * self.thickness * self.outer_height / self.area


# ──────────────────────────────────────────────────────────────────────────────
# Profilé en I (IPE, HEA, HEB, …)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ISection(RectangleBasedSection):
    """Profilé en I symétrique (IPE / HEA / HEB sans congés).

    Parameters
    ----------
    flange_width : float
        Largeur des ailes b_f [m].
    height : float
        Hauteur totale h [m].
    flange_thickness : float
        Épaisseur des ailes t_f [m].
    web_thickness : float
        Épaisseur de l'âme t_w [m].

    Notes
    -----
    Décomposition : 2 ailes + âme (non chevauchants).

    - Iz = (b_f·h³ − (b_f−t_w)·h_w³) / 12  avec h_w = h − 2·t_f
    - Iy = (2·t_f·b_f³ + h_w·t_w³) / 12
    - Iyz = 0  (double symétrie)
    - J ≈ (1/3)·(2·b_f·t_f³ + h_w·t_w³)  [paroi mince ouverte]

    Les valeurs de référence pour les profilés IPE/HEA incluent les congés
    (r) qui ajoutent ~ 5 % à Iz et A.

    Examples
    --------
    IPE 200 simplifié (sans congés) :

    >>> sec = ISection(flange_width=0.100, height=0.200,
    ...               flange_thickness=0.0085, web_thickness=0.0056)
    >>> round(sec.Iz * 1e8, 0)   # cm⁴ × 10
    18430.0
    """

    flange_width: float      # b_f
    height: float            # h
    flange_thickness: float  # t_f
    web_thickness: float     # t_w

    def __post_init__(self) -> None:
        for name, val in [
            ("flange_width", self.flange_width),
            ("height", self.height),
            ("flange_thickness", self.flange_thickness),
            ("web_thickness", self.web_thickness),
        ]:
            if val <= 0:
                raise ValueError(f"{name} doit être > 0, reçu {val}")
        if 2.0 * self.flange_thickness >= self.height:
            raise ValueError(
                f"2·t_f ({2*self.flange_thickness}) doit être < h ({self.height})"
            )
        if self.web_thickness >= self.flange_width:
            raise ValueError(
                f"t_w ({self.web_thickness}) doit être < b_f ({self.flange_width})"
            )

    def _rects(self) -> list[_Rect]:
        h, b_f = self.height, self.flange_width
        t_f, t_w = self.flange_thickness, self.web_thickness
        h_w = h - 2.0 * t_f
        return [
            _Rect(y_c=0.0,              z_c=0.0, h=h_w, b=t_w),         # âme
            _Rect(y_c=+(h - t_f) / 2.0, z_c=0.0, h=t_f, b=b_f),        # aile haut
            _Rect(y_c=-(h - t_f) / 2.0, z_c=0.0, h=t_f, b=b_f),        # aile bas
        ]

    def shear_correction_factor(self, nu: float) -> float:
        """κ pour profilé en I — méthode de l'âme.

        La force tranchante verticale est intégralement reprise par l'âme :

            A_web = t_w · h_w   (h_w = h − 2·t_f)
            κ = A_web / A

        Approximation courante (Eurocode 3, AISC).  Indépendant de ν.
        """
        h_w = self.height - 2.0 * self.flange_thickness
        return self.web_thickness * h_w / self.area


# ──────────────────────────────────────────────────────────────────────────────
# Profilé en C / U
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CSection(RectangleBasedSection):
    """Profilé en C (U) — âme verticale, deux ailes horizontales.

    Parameters
    ----------
    flange_width : float
        Largeur des ailes b_f [m] (âme comprise, extension vers +z).
    height : float
        Hauteur totale h [m].
    flange_thickness : float
        Épaisseur des ailes t_f [m].
    web_thickness : float
        Épaisseur de l'âme t_w [m].

    Notes
    -----
    Géométrie (vue de face) :

    .. code-block:: text

        ┌──────────┐  ← aile supérieure
        │
        │ ← âme
        │
        └──────────┘  ← aile inférieure

    L'âme est à gauche (z = 0 à t_w), les ailes s'étendent vers z = b_f.
    Le centroïde G est **décalé vers les ailes** (z_G > 0).

    - Iyz = 0  (symétrie par rapport à l'axe horizontal y = h/2)
    - Iy ≠ moitié de I-section (section ouverte sur un côté)

    Examples
    --------
    >>> sec = CSection(flange_width=0.075, height=0.200,
    ...               flange_thickness=0.012, web_thickness=0.008)
    >>> sec.Iyz == 0.0
    True
    """

    flange_width: float      # b_f
    height: float            # h
    flange_thickness: float  # t_f
    web_thickness: float     # t_w

    def __post_init__(self) -> None:
        for name, val in [
            ("flange_width", self.flange_width),
            ("height", self.height),
            ("flange_thickness", self.flange_thickness),
            ("web_thickness", self.web_thickness),
        ]:
            if val <= 0:
                raise ValueError(f"{name} doit être > 0, reçu {val}")
        if 2.0 * self.flange_thickness >= self.height:
            raise ValueError(
                f"2·t_f ({2*self.flange_thickness}) doit être < h ({self.height})"
            )
        if self.web_thickness >= self.flange_width:
            raise ValueError(
                f"t_w ({self.web_thickness}) doit être < b_f ({self.flange_width})"
            )

    def _rects(self) -> list[_Rect]:
        h, b_f = self.height, self.flange_width
        t_f, t_w = self.flange_thickness, self.web_thickness
        h_w = h - 2.0 * t_f
        # Référence : bas gauche de l'âme à (y=0, z=0)
        return [
            _Rect(y_c=t_f + h_w / 2.0, z_c=t_w / 2.0,  h=h_w, b=t_w),  # âme
            _Rect(y_c=h - t_f / 2.0,   z_c=b_f / 2.0,  h=t_f, b=b_f),  # aile haut
            _Rect(y_c=t_f / 2.0,        z_c=b_f / 2.0,  h=t_f, b=b_f),  # aile bas
        ]

    def shear_correction_factor(self, nu: float) -> float:
        """κ pour profilé en C — méthode de l'âme.

        Comme pour le profilé en I :

            A_web = t_w · h_w   (h_w = h − 2·t_f)
            κ = A_web / A

        Indépendant de ν.
        """
        h_w = self.height - 2.0 * self.flange_thickness
        return self.web_thickness * h_w / self.area


# ──────────────────────────────────────────────────────────────────────────────
# Cornière L
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LSection(RectangleBasedSection):
    """Cornière à ailes égales ou inégales (profilé L).

    Parameters
    ----------
    width : float
        Largeur de l'aile horizontale b [m].
    height : float
        Hauteur de l'aile verticale h [m].
    thickness : float
        Épaisseur t [m] (uniforme).

    Notes
    -----
    Géométrie (coin extérieur en bas à gauche) :

    .. code-block:: text

        │← aile verticale (h, t)
        │
        └──────  ← aile horizontale (b, t)

    Le centroïde G est **au-dessus et à droite** du coin intérieur, à
    distance (y_G, z_G) du coin bas-gauche.

    Pour une cornière à ailes égales (b = h) :

    - Iz = Iy   (symétrie)
    - Iyz ≠ 0   (section non symétrique dans le repère y–z)
    - α = ±45°  (axes principaux à 45°)

    Examples
    --------
    Cornière L100×100×10 (dimensions en m) :

    >>> import math
    >>> sec = LSection(width=0.100, height=0.100, thickness=0.010)
    >>> abs(sec.Iz - sec.Iy) < 1e-10 * sec.Iz   # cornière égale → Iz = Iy
    True
    >>> abs(sec.alpha_principal - (-math.pi/4)) < 1e-10
    True
    """

    width: float      # b
    height: float     # h
    thickness: float  # t

    def __post_init__(self) -> None:
        for name, val in [
            ("width", self.width),
            ("height", self.height),
            ("thickness", self.thickness),
        ]:
            if val <= 0:
                raise ValueError(f"{name} doit être > 0, reçu {val}")
        if self.thickness >= min(self.width, self.height):
            raise ValueError(
                f"thickness ({self.thickness}) doit être < min(b,h) = "
                f"{min(self.width, self.height)}"
            )

    def _rects(self) -> list[_Rect]:
        b, h, t = self.width, self.height, self.thickness
        # Référence : coin bas-gauche extérieur à (y=0, z=0)
        return [
            _Rect(y_c=h / 2.0,    z_c=t / 2.0,            h=h,   b=t),    # aile verticale
            _Rect(y_c=t / 2.0,    z_c=t + (b - t) / 2.0,  h=t,   b=b-t), # aile horizontale
        ]

    def shear_correction_factor(self, nu: float) -> float:
        """κ pour cornière L — formule rectangulaire (Cowper 1966, Eq. 30).

        La cornière L est traitée comme un rectangle équivalent pour le
        calcul de κ.  Pour la flexion dans le plan xy (forte), l'aile
        verticale (h × t) porte l'essentiel du cisaillement.

            κ = 10(1+ν) / (12+11ν)

        C'est une approximation conservative couramment utilisée pour les
        sections ouvertes à paroi mince de forme complexe.
        """
        return 10.0 * (1.0 + nu) / (12.0 + 11.0 * nu)

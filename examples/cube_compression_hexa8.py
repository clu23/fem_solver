"""Exemple : cube en compression uniaxiale avec l'élément Hexa8.

Problème
--------
Un cube d'acier 10 × 10 × 10 cm est soumis à une pression de compression
uniforme P = 100 MPa appliquée sur la face supérieure (z = 0.1 m).

Conditions aux limites "rouleau" (conditions libres en x,y)
------------------------------------------------------------
Pour obtenir un état de compression uniaxiale PURE, il faut laisser libre
l'expansion Poisson en x et y. On utilise des conditions "rouleau" :
  - Tous les nœuds de la base : uz = 0 (appui vertical)
  - Nœud 0 uniquement : ux = uy = 0 (suppression des modes rigides)
  - Nœud 1 uniquement : uy = 0 (suppression de la rotation autour de z)
  → 6 contraintes au total = exactement les 6 modes rigides bloqués.

Avec ces CL, la solution analytique est exactement reproduite par Hexa8
(champ de déplacement linéaire en z → représenté exactement).

Solution analytique
-------------------
    u_z(z) = -P/E · z     → δ = -P·L/E ≈ -47.619 µm
    u_x(x) = +ν·P/E · x  → dilatation transversale en x (Poisson)
    u_y(y) = +ν·P/E · y  → dilatation transversale en y (Poisson)
    σzz = -P, σxx = σyy = τ.. = 0

Résultats affichés
------------------
1. Déplacements aux 8 nœuds (table)
2. Contraintes au centre de l'élément
3. Comparaison calculé / analytique (erreur ~ machine precision)
"""

from __future__ import annotations

import numpy as np

from femsolver.core.assembler import Assembler
from femsolver.core.boundary import apply_dirichlet
from femsolver.core.material import ElasticMaterial
from femsolver.core.mesh import BoundaryConditions, ElementData, Mesh
from femsolver.core.solver import StaticSolver
from femsolver.elements.hexa8 import Hexa8

# ---------------------------------------------------------------------------
# Paramètres du problème
# ---------------------------------------------------------------------------

L = 0.10         # côté du cube [m]
E = 210e9        # module de Young acier [Pa]
nu = 0.3         # coefficient de Poisson
rho = 7800.0     # masse volumique [kg/m³]
P = 100e6        # pression appliquée [Pa] (compression)

# ---------------------------------------------------------------------------
# Maillage — 1 élément Hexa8
# ---------------------------------------------------------------------------
#
#  Numérotation des nœuds :
#
#       7-----6
#      /|    /|       z
#     4-----5 |       |  y
#     | 3---|-2       | /
#     |/    |/        |/
#     0-----1         +---x
#
# Face z=0 (nœuds 0..3) : encastrée
# Face z=L (nœuds 4..7) : pression P appliquée (compression)

nodes = np.array([
    [0.0, 0.0, 0.0],    # 0
    [L,   0.0, 0.0],    # 1
    [L,   L,   0.0],    # 2
    [0.0, L,   0.0],    # 3
    [0.0, 0.0, L  ],    # 4
    [L,   L,   L  ],    # 5  <— noter : non (L,0,L) mais (L,L,L) pour respecter
    [L,   0.0, L  ],    # 6      l'ordre trigonométrique
    [0.0, L,   L  ],    # 7
], dtype=float)

# Correction de la numérotation standard Hexa8 (face z=L : 4,5,6,7 CCW vue de haut)
nodes = np.array([
    [0.0, 0.0, 0.0],    # 0  coin bas-avant-gauche
    [L,   0.0, 0.0],    # 1  coin bas-avant-droit
    [L,   L,   0.0],    # 2  coin bas-arrière-droit
    [0.0, L,   0.0],    # 3  coin bas-arrière-gauche
    [0.0, 0.0, L  ],    # 4  coin haut-avant-gauche
    [L,   0.0, L  ],    # 5  coin haut-avant-droit
    [L,   L,   L  ],    # 6  coin haut-arrière-droit
    [0.0, L,   L  ],    # 7  coin haut-arrière-gauche
], dtype=float)

material = ElasticMaterial(E=E, nu=nu, rho=rho)

elements = (
    ElementData(
        etype=Hexa8,
        node_ids=(0, 1, 2, 3, 4, 5, 6, 7),
        material=material,
        properties={},
    ),
)
mesh = Mesh(nodes=nodes, elements=elements, n_dim=3)

# ---------------------------------------------------------------------------
# Conditions aux limites
# ---------------------------------------------------------------------------

# Conditions "rouleau" : tous les nœuds de la base bloqués en z,
# mais libres en x,y → expansion Poisson libre.
# Suppression des modes rigides restants avec 2 contraintes ponctuelles :
#   - Nœud 0 : ux=0 et uy=0  (bloque translations x et y)
#   - Nœud 1 : uy=0           (bloque rotation autour de z)
dirichlet: dict[int, dict[int, float]] = {
    0: {0: 0.0, 1: 0.0, 2: 0.0},   # ux, uy, uz bloqués
    1: {1: 0.0, 2: 0.0},            # uy, uz bloqués
    2: {2: 0.0},                    # uz bloqué
    3: {2: 0.0},                    # uz bloqué
}

# Face supérieure (z=L) : pression uniforme P
# Chargement cohérent : F_z = -P × aire / 4 nœuds = -P × L² / 4
f_nodal = -P * L * L / 4.0   # < 0 : compression en -z
neumann: dict[int, dict[int, float]] = {
    4: {2: f_nodal},
    5: {2: f_nodal},
    6: {2: f_nodal},
    7: {2: f_nodal},
}
bc = BoundaryConditions(dirichlet=dirichlet, neumann=neumann)

# ---------------------------------------------------------------------------
# Résolution
# ---------------------------------------------------------------------------

assembler = Assembler(mesh)
K = assembler.assemble_stiffness()
F = assembler.assemble_forces(bc)
K_bc, F_bc = apply_dirichlet(K, F, mesh, bc)
u = StaticSolver().solve(K_bc, F_bc)

# ---------------------------------------------------------------------------
# Post-traitement
# ---------------------------------------------------------------------------

print("=" * 65)
print("  CUBE EN COMPRESSION — HEXA8 (1 élément)")
print("=" * 65)
print(f"  Matériau : E = {E/1e9:.0f} GPa, ν = {nu}, ρ = {rho} kg/m³")
print(f"  Géométrie : cube {L*100:.0f} cm × {L*100:.0f} cm × {L*100:.0f} cm")
print(f"  Chargement : P = {P/1e6:.0f} MPa (compression en z)")
print()

# Tableau des déplacements
print("  Déplacements nodaux :")
print(f"  {'Nœud':>5}  {'x [m]':>8}  {'y [m]':>8}  {'z [m]':>8}  "
      f"{'ux [µm]':>9}  {'uy [µm]':>9}  {'uz [µm]':>9}")
print("  " + "-" * 63)
for i in range(8):
    xi, yi, zi = nodes[i]
    ux = u[3 * i    ] * 1e6
    uy = u[3 * i + 1] * 1e6
    uz = u[3 * i + 2] * 1e6
    print(f"  {i:>5}  {xi:>8.3f}  {yi:>8.3f}  {zi:>8.3f}  "
          f"{ux:>9.4f}  {uy:>9.4f}  {uz:>9.4f}")

# Contraintes au centre de l'élément
u_e = u[:24]
sigma = Hexa8().stress(material, nodes, u_e, xi=0.0, eta=0.0, zeta=0.0)
labels = ["σxx", "σyy", "σzz", "τyz", "τxz", "τxy"]

print()
print("  Contraintes au centre de l'élément [MPa] :")
for lbl, s in zip(labels, sigma):
    print(f"    {lbl} = {s/1e6:+.4f} MPa")

# Solution analytique
delta_analytique = -P * L / E     # < 0 car compression
delta_numerique = float(np.mean(u[[3*4+2, 3*5+2, 3*6+2, 3*7+2]]))
erreur_rel = abs((delta_numerique - delta_analytique) / delta_analytique)

print()
print("  Comparaison déplacement axial u_z (face supérieure) :")
print(f"    Analytique : {delta_analytique*1e6:+.6f} µm")
print(f"    Numérique  : {delta_numerique*1e6:+.6f} µm")
print(f"    Erreur rel.: {erreur_rel:.2e}")

# Dilatation transversale (effet Poisson)
# Nœud 5 est à (x=L, y=0, z=L) → u_x attendu = +ν·P/E·L (dilatation)
ux_5_analytique = nu * P * L / E
ux_5_numerique  = float(u[3 * 5])
print()
print("  Effet Poisson — nœud 5 (x=L, y=0, z=L) :")
print(f"    u_x analytique : +ν·P·L/E = {ux_5_analytique*1e6:+.6f} µm")
print(f"    u_x numérique  :            {ux_5_numerique*1e6:+.6f} µm")
print(f"    Erreur rel.    :             {abs((ux_5_numerique-ux_5_analytique)/ux_5_analytique):.2e}")
print()
print("=" * 65)

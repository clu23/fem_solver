"""Équerre en L sous charge ponctuelle — maillage Hexa8 structuré.

Géométrie
---------
Une équerre en acier en forme de L (vue dans le plan XZ) :

        z ↑
          |
     b  +-+
        | |   ← bras vertical (x ∈ [0,t], z ∈ [0,b])
        | |
     t  +-+---+---+---+---+
        |       bras horizontal    (x ∈ [0,a], z ∈ [0,t])
     0  +---+---+---+---+---+--→ x
        0   t                 a

  - a = 100 mm  longueur du bras horizontal
  - b = 100 mm  hauteur du bras vertical (inclut le coin)
  - t =  20 mm  épaisseur (même dans les deux bras)
  - d =  20 mm  profondeur en y (extrusion)

Conditions aux limites
----------------------
- **Encastrement** : face supérieure du bras vertical (z = b), appui mural.
  → ux = uy = uz = 0 sur les 4 nœuds de cette face.

- **Charge** : force verticale F = -5 kN sur la face libre du bras horizontal
  (x = a, z ∈ [0,t]). Répartie équitablement sur les 4 nœuds de cette face.

       ↓ F
  +-+-+-+-|---+
  |       |   | ← face chargée
  +-------+---+
  (Free end at x=a)

Maillage
--------
Maillage Hexa8 structuré, résolution fine autour du coin :

  - Bras horizontal : nx_h éléments en x, 1 en y, 1 en z
  - Coin            : 1×1×1 élément partagé
  - Bras vertical   : 1 en x, 1 en y, nz_v éléments en z

  Total = (nx_h + 1 + nz_v) éléments = 4 + 1 + 4 = 9 éléments

Physique
--------
- E = 210 GPa (acier)
- ν = 0.3
- F = 5 kN (vers le bas, -z)

Résultats attendus
------------------
La concentration de contraintes apparaît au coin (jonction des deux bras).
Le bras horizontal se comporte comme un porte-à-faux :
  - Déflexion à l'extrémité ≈ FL³/(3EI)  (ordre de grandeur)
  - La formule de Barlow donne σ_max au coin

Note sur la précision
---------------------
Ce maillage grossier (9 éléments) est illustratif. Pour des résultats
quantitatifs précis, il faudrait raffiner autour du coin (zone de
concentration de contraintes) avec au moins 4×4×4 éléments par bras.
"""

from __future__ import annotations

import numpy as np

from femsolver.core.assembler import Assembler
from femsolver.core.boundary import apply_dirichlet
from femsolver.core.material import ElasticMaterial
from femsolver.core.mesh import BoundaryConditions, ElementData, Mesh
from femsolver.core.solver import StaticSolver
from femsolver.elements.hexa8 import Hexa8
from femsolver.postprocess.stress3d import nodal_stresses_3d, von_mises_3d

# ---------------------------------------------------------------------------
# Paramètres géométriques et matériau
# ---------------------------------------------------------------------------

a   = 0.100   # longueur bras horizontal [m]
b   = 0.100   # hauteur bras vertical    [m]
t   = 0.020   # épaisseur                [m]
d   = 0.020   # profondeur en y          [m]

E   = 210e9   # module de Young acier    [Pa]
nu  = 0.3     # coefficient de Poisson
rho = 7800.0  # densité                  [kg/m³]
F   = -5000.0  # force appliquée         [N] (négatif = vers le bas)

# Discrétisation
nx_h = 4   # éléments dans le bras horizontal (hors coin)
nz_v = 4   # éléments dans le bras vertical   (hors coin)
ny   = 1   # éléments en profondeur (y)


# ---------------------------------------------------------------------------
# Construction du maillage L
# ---------------------------------------------------------------------------

def _build_l_bracket(
    a: float, b: float, t: float, d: float,
    nx_h: int, nz_v: int, ny: int,
    material: ElasticMaterial,
) -> tuple[Mesh, dict[str, list[int]]]:
    """Construire un maillage Hexa8 structuré pour une équerre en L.

    Parameters
    ----------
    a, b : float  Longueur bras horizontal/hauteur bras vertical [m].
    t, d : float  Épaisseur et profondeur [m].
    nx_h, nz_v : int  Éléments dans chaque bras (hors coin).
    ny : int  Éléments en profondeur.
    material : ElasticMaterial

    Returns
    -------
    mesh : Mesh
    node_groups : dict[str, list[int]]
        ``"top_face"``  : nœuds de la face d'encastrement (z = b).
        ``"free_face"`` : nœuds de la face libre (x = a).
        ``"corner"``    : nœuds du coin (x ∈ [0,t], z ∈ [0,t]).
    """
    # --- Coordonnées de la grille ---
    # x : coin [0, t], puis bras horizontal (t, a]
    x_corner = np.array([0.0, t])
    x_arm    = np.linspace(t, a, nx_h + 1)[1:]      # nx_h intervalles
    x_arr    = np.concatenate([x_corner, x_arm])     # nx_h + 2 valeurs

    # z : bras horizontal [0, t], puis bras vertical (t, b]
    z_arm  = np.array([0.0, t])
    z_vert = np.linspace(t, b, nz_v + 1)[1:]        # nz_v intervalles
    z_arr  = np.concatenate([z_arm, z_vert])         # nz_v + 2 valeurs

    # y : simple extrusion
    y_arr = np.linspace(0.0, d, ny + 1)

    # Indices utiles
    nx_total = len(x_arr) - 1   # nx_h + 1 cellules en x
    nz_total = len(z_arr) - 1   # nz_v + 1 cellules en z

    # --- Nœuds actifs (ceux utilisés par des cellules L-actives) ---
    #
    # Cellule (ix, iy, iz) est active si :
    #   • iz == 0  (bras horizontal, toutes les cellules en x)  ← nœuds z=0 et z=1
    #   • ix == 0  (bras vertical, toutes les cellules en z)    ← nœuds x=0 et x=1
    #
    # Le coin (ix=0, iz=0) est inclus dans les deux, compté une seule fois.

    active_nodes: set[tuple[int, int, int]] = set()

    def _add_cell_nodes(ix_: int, iy_: int, iz_: int) -> None:
        for dix in range(2):
            for diy_ in range(2):
                for diz in range(2):
                    active_nodes.add((ix_ + dix, iy_ + diy_, iz_ + diz))

    # Bras horizontal : iz = 0, toutes les cellules x
    for ix in range(nx_total):
        for iy in range(ny):
            _add_cell_nodes(ix, iy, 0)

    # Bras vertical (inclut le coin) : ix = 0, toutes les cellules z
    for iz in range(nz_total):
        for iy in range(ny):
            _add_cell_nodes(0, iy, iz)

    # Tri reproductible → mappage stable (ix,iy,iz) → indice global
    sorted_nodes = sorted(active_nodes)
    node_map: dict[tuple[int, int, int], int] = {n: i for i, n in enumerate(sorted_nodes)}

    # Coordonnées des nœuds
    nodes = np.array(
        [[x_arr[ix], y_arr[iy], z_arr[iz]] for ix, iy, iz in sorted_nodes],
        dtype=float,
    )

    # --- Éléments ---
    elements: list[ElementData] = []

    def _add_element(ix_: int, iy_: int, iz_: int) -> None:
        """Ajoute un Hexa8 à partir de la cellule (ix_, iy_, iz_).

        Ordre des nœuds : face basse (ζ=-1) numérotée CCW vue de -z,
        puis face haute (ζ=+1) idem.  Le ζ-local croît avec z global,
        assurant det(J) > 0.
        """
        n = node_map
        connectivity = (
            n[(ix_,     iy_,     iz_    )],   # 0
            n[(ix_ + 1, iy_,     iz_    )],   # 1
            n[(ix_ + 1, iy_ + 1, iz_    )],   # 2
            n[(ix_,     iy_ + 1, iz_    )],   # 3
            n[(ix_,     iy_,     iz_ + 1)],   # 4
            n[(ix_ + 1, iy_,     iz_ + 1)],   # 5
            n[(ix_ + 1, iy_ + 1, iz_ + 1)],  # 6
            n[(ix_,     iy_ + 1, iz_ + 1)],  # 7
        )
        elements.append(ElementData(
            etype=Hexa8,
            node_ids=connectivity,
            material=material,
            properties={},
        ))

    # Bras horizontal
    for ix in range(nx_total):
        for iy in range(ny):
            _add_element(ix, iy, 0)

    # Bras vertical (ix=0, iz≥1 : au-dessus du coin)
    for iz in range(1, nz_total):
        for iy in range(ny):
            _add_element(0, iy, iz)

    mesh = Mesh(nodes=nodes, elements=tuple(elements), n_dim=3)

    # --- Groupes de nœuds pour les CL ---
    iz_top  = nz_v + 1   # index z pour z = b
    ix_free = nx_h + 1   # index x pour x = a

    top_face: list[int] = []    # encastrement (z = b)
    free_face: list[int] = []   # chargement (x = a)
    corner: list[int] = []      # coin (x ∈ [0,t], z ∈ [0,t])

    for key in sorted_nodes:
        ix_, iy_, iz_ = key
        nid = node_map[key]
        if iz_ == iz_top and ix_ <= 1:   # face bras vertical (z = b)
            top_face.append(nid)
        if ix_ == ix_free:               # face libre bras horizontal (x = a)
            free_face.append(nid)
        if ix_ <= 1 and iz_ <= 1:        # coin
            corner.append(nid)

    node_groups = {
        "top_face": top_face,
        "free_face": free_face,
        "corner": corner,
    }
    return mesh, node_groups


# ---------------------------------------------------------------------------
# Conditions aux limites
# ---------------------------------------------------------------------------

material = ElasticMaterial(E=E, nu=nu, rho=rho)
mesh, groups = _build_l_bracket(a, b, t, d, nx_h, nz_v, ny, material)

# Encastrement : face supérieure du bras vertical (z = b)
dirichlet: dict[int, dict[int, float]] = {
    nid: {0: 0.0, 1: 0.0, 2: 0.0}
    for nid in groups["top_face"]
}

# Chargement : face libre bras horizontal (x = a)
n_loaded = len(groups["free_face"])
f_nodal = F / n_loaded   # force répartie équitablement [N par nœud]
neumann: dict[int, dict[int, float]] = {
    nid: {2: f_nodal}
    for nid in groups["free_face"]
}

bc = BoundaryConditions(dirichlet=dirichlet, neumann=neumann)

# ---------------------------------------------------------------------------
# Résolution statique
# ---------------------------------------------------------------------------

assembler = Assembler(mesh)
K = assembler.assemble_stiffness()
F_ext = assembler.assemble_forces(bc)
K_bc, F_bc = apply_dirichlet(K, F_ext, mesh, bc)
u = StaticSolver().solve(K_bc, F_bc)

# ---------------------------------------------------------------------------
# Post-traitement
# ---------------------------------------------------------------------------

sigma  = nodal_stresses_3d(mesh, u)
vm     = von_mises_3d(sigma)

# Déplacements en mm pour l'affichage
u_nodes_mm = u.reshape(-1, 3) * 1e3

print("=" * 65)
print("  ÉQUERRE EN L — HEXA8 (9 éléments Hexa8)")
print("=" * 65)
print(f"  Matériau  : E = {E/1e9:.0f} GPa, ν = {nu}, ρ = {rho} kg/m³")
print(f"  Géométrie : a={a*1e3:.0f} mm × b={b*1e3:.0f} mm × t={t*1e3:.0f} mm × d={d*1e3:.0f} mm")
print(f"  Charge    : F = {F/1e3:.1f} kN (−z) sur la face libre (x = {a*1e3:.0f} mm)")
print(f"  Maillage  : {len(mesh.elements)} éléments Hexa8, {mesh.n_nodes} nœuds, {mesh.n_dof} DDL")
print()

# Déplacement maximum
i_umax = np.argmax(np.linalg.norm(u_nodes_mm, axis=1))
print(f"  Déplacement max   : {np.linalg.norm(u_nodes_mm[i_umax]):.4f} mm  (nœud {i_umax})")
print(f"  uz max (vers bas) : {u_nodes_mm[:, 2].min():.4f} mm")
print()

# Déflexion analytique de poutre cantilever (approximation bras horizontal)
L_eff = a - t / 2.0                       # longueur effective
I = d * t ** 3 / 12.0                     # moment quadratique [m⁴]
delta_beam = abs(F) * L_eff ** 3 / (3.0 * E * I)
print(f"  Déflexion analytique poutre (approx.) : {delta_beam*1e3:.4f} mm")
print(f"  Déflexion FEM uz à la face libre       : {abs(u_nodes_mm[:, 2].min()):.4f} mm")
print(f"  (Écart attendu : le coin introduit une rigidité supplémentaire)")
print()

# Contraintes de Von Mises
print(f"  Von Mises σ_VM [MPa] :")
print(f"    Maximum global     : {vm.max()/1e6:.2f} MPa")
print(f"    Au coin (moy)      : {vm[groups['corner']].mean()/1e6:.2f} MPa")
print(f"    À l'encastrement   : {vm[groups['top_face']].mean()/1e6:.2f} MPa")
print()

# Contrainte de flexion analytique à la base du bras horizontal
M_base = abs(F) * (a - t)
sigma_beam_max = M_base * (t / 2.0) / I
print(f"  σ_max flexion analytique (base bras)   : {sigma_beam_max/1e6:.2f} MPa")
print(f"  σ_xx max FEM                            : {abs(sigma[:, 0]).max()/1e6:.2f} MPa")
print("=" * 65)

# ---------------------------------------------------------------------------
# Export VTK pour ParaView (optionnel)
# ---------------------------------------------------------------------------

try:
    from femsolver.io.mesh_io import write_vtu
    import os
    out_dir = os.path.dirname(os.path.abspath(__file__))
    vtu_path = os.path.join(out_dir, "l_bracket_results.vtu")
    write_vtu(vtu_path, mesh, u=u, sigma=sigma, sigma_vm=vm)
    print(f"\n  Résultats exportés → {vtu_path}  (ouvrir avec ParaView)")
except Exception as exc:
    print(f"\n  Export VTK ignoré : {exc}")

# ---------------------------------------------------------------------------
# Visualisation PyVista (screenshot hors-écran)
# ---------------------------------------------------------------------------

try:
    from femsolver.postprocess.plotter3d import plot_deformed_3d, plot_mesh_3d

    import os
    out_dir = os.path.dirname(os.path.abspath(__file__))

    # Maillage non déformé
    mesh_png = os.path.join(out_dir, "l_bracket_mesh.png")
    plot_mesh_3d(mesh, title="Équerre en L — Maillage", show=False, screenshot=mesh_png)
    print(f"\n  Maillage        → {mesh_png}")

    # Déformée + Von Mises
    vm_png = os.path.join(out_dir, "l_bracket_vm.png")
    plot_deformed_3d(
        mesh, u=u, sigma_vm=vm, scale=None,
        title="Équerre en L — Von Mises [MPa]", show=False, screenshot=vm_png,
    )
    print(f"  Von Mises       → {vm_png}")

    # Déformée + σxx (contrainte normale)
    sxx_png = os.path.join(out_dir, "l_bracket_sxx.png")
    plot_deformed_3d(
        mesh, u=u, sigma_component=sigma[:, 0], component_label="σxx [MPa]",
        title="Équerre en L — σxx [MPa]", show=False, screenshot=sxx_png,
    )
    print(f"  σxx             → {sxx_png}")

except ImportError as exc:
    print(f"\n  Visualisation PyVista ignorée : {exc}")
except Exception as exc:
    print(f"\n  Erreur visualisation : {exc}")

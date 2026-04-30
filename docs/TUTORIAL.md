# Tutoriel — Comment utiliser fem-solver

Ce guide explique comment créer un modèle, lancer une analyse et exploiter
les résultats avec fem-solver, de A à Z.

---

## 1. Structure d'un calcul FEM

Tout calcul suit le même enchaînement :

```
Matériau → Maillage → Assemblage → Conditions aux limites → Résolution → Post-traitement
```

En code, ça se traduit par 6 imports principaux :

```python
from femsolver.core.material import ElasticMaterial
from femsolver.core.mesh import Mesh, BoundaryConditions
from femsolver.core.assembler import Assembler
from femsolver.core.boundary import apply_dirichlet
from femsolver.core.solver import StaticSolver
from femsolver.core.diagnostics import run_diagnostics
```

---

## 2. Définir un matériau

```python
# Acier standard (unités SI : Pa, m, kg/m³)
steel = ElasticMaterial(E=210e9, nu=0.3, rho=7850)

# Aluminium
aluminium = ElasticMaterial(E=70e9, nu=0.33, rho=2700)

# Contrainte plane (défaut) vs déformation plane
steel_plane_strain = ElasticMaterial(E=210e9, nu=0.3, rho=7850, plane_stress=False)
```

Le matériau fournit automatiquement les matrices de comportement :
- `steel.D_plane_stress` → matrice D 3×3 (éléments 2D, contrainte plane)
- `steel.D_plane_strain` → matrice D 3×3 (éléments 2D, déformation plane)
- `steel.elasticity_matrix_3d()` → matrice D 6×6 (éléments 3D)

---

## 3. Définir les éléments

### 3.1 Treillis (Bar2D)

```python
from femsolver.elements.bar2d import Bar2D
import numpy as np

# Nœuds : array (n_nodes, 2)
nodes = np.array([
    [0.0, 0.0],   # nœud 0
    [1.0, 0.0],   # nœud 1
    [0.5, 0.866],  # nœud 2
])

# Éléments : chaque Bar2D prend les indices des nœuds et les propriétés
elements = [
    Bar2D(node_ids=[0, 1], material=steel, properties={"area": 1e-3}),
    Bar2D(node_ids=[1, 2], material=steel, properties={"area": 1e-3}),
    Bar2D(node_ids=[0, 2], material=steel, properties={"area": 1e-3}),
]
```

### 3.2 Poutres (Beam2D / Beam3D)

```python
from femsolver.elements.beam2d_timoshenko import Beam2DTimoshenko
from femsolver.elements.beam3d import Beam3D
from femsolver.core.sections import (
    RectangularSection, CircularSection, ISection
)

# Définir une section
rect_section = RectangularSection(b=0.1, h=0.2)  # 100mm × 200mm
ipe200 = ISection(h=0.2, b=0.1, t_f=0.0085, t_w=0.0056)  # IPE 200
tube = CircularSection(r=0.05)  # tube plein Ø100mm

# Beam2D (Timoshenko, dans le plan)
beam2d = Beam2DTimoshenko(
    node_ids=[0, 1],
    material=steel,
    properties={"section": rect_section}
)

# Beam3D (Timoshenko 3D, 6 ddl/nœud)
beam3d = Beam3D(
    node_ids=[0, 1],
    material=steel,
    properties={
        "section": ipe200,
        "v_vec": [0, 0, 1],           # vecteur d'orientation (optionnel)
        "offset_i": [0, 0, 0.1],      # offset au nœud i (optionnel)
        "offset_j": [0, 0, 0],        # offset au nœud j (optionnel)
    }
)
```

### 3.3 Éléments 2D (Tri3, Quad4, Tri6)

```python
from femsolver.elements.tri3 import Tri3
from femsolver.elements.quad4 import Quad4
from femsolver.elements.tri6 import Tri6

# Triangle 3 nœuds (contrainte plane)
tri = Tri3(
    node_ids=[0, 1, 2],
    material=steel,
    properties={"thickness": 0.01}  # épaisseur 10mm
)

# Quadrilatère 4 nœuds
quad = Quad4(
    node_ids=[0, 1, 2, 3],
    material=steel,
    properties={"thickness": 0.01}
)

# Quad4 avec intégration réduite sélective (anti shear locking)
# → utiliser quad.stiffness_matrix_sri(nodes) au lieu de quad.stiffness_matrix(nodes)

# Triangle quadratique 6 nœuds (plus précis)
tri6 = Tri6(
    node_ids=[0, 1, 2, 3, 4, 5],  # 3 coins + 3 milieux d'arêtes
    material=steel,
    properties={"thickness": 0.01}
)
```

### 3.4 Éléments 3D (Tetra4, Hexa8, Tetra10)

```python
from femsolver.elements.tetra4 import Tetra4
from femsolver.elements.hexa8 import Hexa8
from femsolver.elements.tetra10 import Tetra10

# Tétraèdre 4 nœuds
tet = Tetra4(node_ids=[0, 1, 2, 3], material=steel, properties={})

# Hexaèdre 8 nœuds
hex_elem = Hexa8(node_ids=[0,1,2,3,4,5,6,7], material=steel, properties={})

# Tétraèdre quadratique 10 nœuds
tet10 = Tetra10(
    node_ids=[0,1,2,3,4,5,6,7,8,9],
    material=steel,
    properties={}
)
```

---

## 4. Créer le maillage

```python
mesh = Mesh(nodes=nodes, elements=elements, n_dim=2)
# n_dim = 2 pour les problèmes 2D (Bar2D, Tri3, Quad4)
# n_dim = 3 pour les problèmes 3D (Tetra4, Hexa8, Beam3D)
```

Pour les maillages structurés, tu peux les générer programmatiquement :

```python
def mesh_rectangle_quad4(Lx, Ly, nx, ny, material, thickness):
    """Maille un rectangle en Quad4."""
    nodes = []
    for j in range(ny + 1):
        for i in range(nx + 1):
            nodes.append([i * Lx / nx, j * Ly / ny])
    nodes = np.array(nodes)

    elements = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            n1 = n0 + 1
            n2 = n1 + (nx + 1)
            n3 = n0 + (nx + 1)
            elements.append(Quad4(
                node_ids=[n0, n1, n2, n3],
                material=material,
                properties={"thickness": thickness}
            ))
    return Mesh(nodes=nodes, elements=elements, n_dim=2)

mesh = mesh_rectangle_quad4(Lx=2.0, Ly=0.5, nx=40, ny=10,
                             material=steel, thickness=0.01)
```

Pour importer un maillage Gmsh :

```python
from femsolver.io.mesh_io import read_mesh
mesh, groups = read_mesh("model.msh", material=steel)
# groups contient les groupes physiques Gmsh → utile pour les BCs
```

---

## 5. Conditions aux limites

### 5.1 Dirichlet (déplacements imposés)

```python
# Format : {dof_global: valeur}
# Pour un problème 2D (2 ddl/nœud) :
#   nœud i → dof ux = 2*i, dof uy = 2*i + 1
# Pour un problème 3D (3 ddl/nœud) :
#   nœud i → dof ux = 3*i, dof uy = 3*i + 1, dof uz = 3*i + 2
# Pour les poutres (6 ddl/nœud) :
#   nœud i → ux=6i, uy=6i+1, uz=6i+2, θx=6i+3, θy=6i+4, θz=6i+5

bc = BoundaryConditions()

# Encastrement du nœud 0 (2D, 2 ddl)
bc.dirichlet[0] = 0.0   # ux = 0
bc.dirichlet[1] = 0.0   # uy = 0

# Appui simple (rouleau) du nœud 4 : bloque uy seulement
bc.dirichlet[2 * 4 + 1] = 0.0  # uy = 0

# Déplacement imposé non nul
bc.dirichlet[2 * 3 + 0] = 0.001  # ux du nœud 3 = 1 mm

# Encastrement complet nœud 0 en 3D poutre (6 ddl)
for dof in range(6):
    bc.dirichlet[6 * 0 + dof] = 0.0
```

### 5.2 Neumann (forces)

```python
# Forces ponctuelles : {dof_global: valeur en Newtons}
bc.neumann[2 * 2 + 1] = -50000.0  # Fy = -50 kN sur nœud 2
bc.neumann[2 * 3 + 1] = -50000.0  # Fy = -50 kN sur nœud 3
```

### 5.3 Charges avancées

```python
from femsolver.core.mesh import PressureLoad, BodyForce, DistributedLineLoad

# Pression sur une face (nœuds de la face, pression en Pa)
pressure = PressureLoad(
    face_nodes=[10, 11, 12, 13],  # 4 nœuds de la face
    pressure=1e6                   # 1 MPa, direction = normale sortante
)

# Gravité (force volumique)
gravity = BodyForce(direction=[0, -1, 0], magnitude=9.81)  # g vers -y

# Charge distribuée sur une poutre (N/m)
line_load = DistributedLineLoad(
    element_index=0,   # indice de l'élément poutre
    qx=0.0,            # charge axiale
    qy=-1000.0         # charge transversale : 1 kN/m vers le bas
)
```

### 5.4 MPC (Multi-Point Constraints)

```python
from femsolver.core.mpc import MPCConstraint, apply_mpc_elimination

# Contrainte : u_esclave = α * u_maître + β
# Exemple : le ddl 10 est esclave du ddl 4
constraint = MPCConstraint(
    terms=[(10, 1.0), (4, -1.0)],  # 1.0*u10 - 1.0*u4 = 0
    rhs=0.0
)

# Application
K_red, F_red, T, g, slaves = apply_mpc_elimination(K, F, mesh, [constraint])
```

---

## 6. Assemblage et résolution

### 6.1 Analyse statique

```python
from femsolver.core.assembler import Assembler
from femsolver.core.boundary import apply_dirichlet
from femsolver.core.solver import StaticSolver

# Assembler les matrices
assembler = Assembler(mesh)
K = assembler.assemble_stiffness()       # matrice de rigidité globale
F = assembler.assemble_force_vector(bc)  # vecteur de forces

# Appliquer les conditions de Dirichlet
ds = apply_dirichlet(K, F, mesh, bc)

# Résoudre
solver = StaticSolver()
u = solver.solve(ds.K_bc, ds.F_bc)  # vecteur de déplacements
```

### 6.2 Analyse modale

```python
from femsolver.dynamics.modal import run_modal

# Calcul des 10 premiers modes propres
result = run_modal(mesh, bc, n_modes=10)

print(f"Fréquences propres : {result.frequencies} Hz")
# result.mode_shapes : (n_dof, n_modes) — déformées modales
```

### 6.3 Réponse harmonique

```python
from femsolver.dynamics.harmonic import run_harmonic
from femsolver.dynamics.damping import RayleighDamping, HystereticDamping

# Avec amortissement de Rayleigh
damping = RayleighDamping.from_modes(
    omega1=result.frequencies[0] * 2 * np.pi,
    omega2=result.frequencies[2] * 2 * np.pi,
    zeta1=0.02, zeta2=0.02  # 2% d'amortissement
)

# Balayage en fréquence
freqs = np.linspace(1, 200, 500)  # 1 à 200 Hz, 500 points
H = run_harmonic(mesh, bc, freqs, damping=damping)
# H : (n_freq, n_dof) — déplacements complexes à chaque fréquence

# Avec amortissement hystérétique (indépendant de la fréquence)
damping_h = HystereticDamping(eta=0.04)  # facteur de perte 4%
H = run_harmonic(mesh, bc, freqs, damping=damping_h)
```

### 6.4 Analyse transitoire

```python
from femsolver.dynamics.transient import run_transient, NewmarkBeta

# Définir le chargement en fonction du temps
def force_func(t):
    """Impact : force constante pendant 0.01s puis rien."""
    F = np.zeros(mesh.n_dof)
    if t < 0.01:
        F[2 * 2 + 1] = -50000.0  # 50 kN sur nœud 2
    return F

# Paramètres de Newmark (trapézoïdal, inconditionnellement stable)
scheme = NewmarkBeta(gamma=0.5, beta=0.25)

# Intégration temporelle
dt = 0.001  # pas de temps 1 ms
t_end = 0.5  # durée 500 ms
result = run_transient(
    mesh, bc,
    force_func=force_func,
    dt=dt, t_end=t_end,
    scheme=scheme,
    damping=damping
)
# result.times : (n_steps,)
# result.displacements : (n_steps, n_dof)
# result.velocities : (n_steps, n_dof)
```

---

## 7. Post-traitement

### 7.1 Contraintes et Von Mises (2D)

```python
from femsolver.postprocess.stress import nodal_stresses, von_mises

# Contraintes lissées aux nœuds
sigma = nodal_stresses(mesh, u)  # (n_nodes, 3) : [σxx, σyy, τxy]

# Von Mises
sigma_vm = von_mises(sigma)  # (n_nodes,)
```

### 7.2 Contraintes 3D

```python
from femsolver.postprocess.stress3d import nodal_stresses_3d, von_mises_3d

sigma = nodal_stresses_3d(mesh, u)  # (n_nodes, 6) : [σxx, σyy, σzz, τyz, τxz, τxy]
sigma_vm = von_mises_3d(sigma)
```

### 7.3 Estimateur d'erreur ZZ

```python
from femsolver.postprocess.error_estimator import estimate_error_zz

result = estimate_error_zz(mesh, u)
# result.element_errors : erreur par élément
# result.global_error : erreur globale relative
# Les éléments avec les plus grandes erreurs → à raffiner
```

### 7.4 Diagnostics (masse, réactions, équilibre)

```python
from femsolver.core.diagnostics import run_diagnostics

diag = run_diagnostics(mesh, K, u, F, bc)
# Affiche automatiquement :
# - Masse totale et CG
# - Réactions d'appui (SPCFORCE)
# - Bilan d'équilibre par direction
```

### 7.5 Visualisation 2D

```python
from femsolver.postprocess.plotter2d import plot_deformed

plot_deformed(mesh, u, scale=100, forces=bc)
```

### 7.6 Visualisation 3D et export

```python
from femsolver.postprocess.plotter3d import plot_deformed_3d
from femsolver.io.mesh_io import write_vtu

# Visualisation PyVista
plot_deformed_3d(mesh, u, sigma_vm=sigma_vm, scale=50)

# Export pour ParaView
write_vtu("results.vtu", mesh, u, sigma=sigma, sigma_vm=sigma_vm)
```

---

## 8. Exemple complet : poutre console en flexion

```python
import numpy as np
from femsolver.core.material import ElasticMaterial
from femsolver.core.mesh import Mesh, BoundaryConditions
from femsolver.core.sections import RectangularSection
from femsolver.core.assembler import Assembler
from femsolver.core.boundary import apply_dirichlet
from femsolver.core.solver import StaticSolver
from femsolver.core.diagnostics import run_diagnostics
from femsolver.elements.beam2d_timoshenko import Beam2DTimoshenko

# --- Paramètres ---
L = 2.0          # longueur 2 m
n_elem = 10      # nombre d'éléments
F_tip = -10000   # 10 kN vers le bas au bout

steel = ElasticMaterial(E=210e9, nu=0.3, rho=7850)
section = RectangularSection(b=0.1, h=0.2)  # 100×200 mm

# --- Maillage ---
nodes = np.array([[i * L / n_elem, 0.0] for i in range(n_elem + 1)])
elements = [
    Beam2DTimoshenko(
        node_ids=[i, i + 1],
        material=steel,
        properties={"section": section}
    )
    for i in range(n_elem)
]
mesh = Mesh(nodes=nodes, elements=elements, n_dim=2)

# --- Conditions aux limites ---
bc = BoundaryConditions()
# Encastrement nœud 0 (3 ddl pour Beam2D : ux, uy, θz)
bc.dirichlet[0] = 0.0  # ux
bc.dirichlet[1] = 0.0  # uy
bc.dirichlet[2] = 0.0  # θz
# Force au bout
dof_tip_y = 3 * n_elem + 1  # uy du dernier nœud
bc.neumann[dof_tip_y] = F_tip

# --- Résolution ---
assembler = Assembler(mesh)
K = assembler.assemble_stiffness()
F = assembler.assemble_force_vector(bc)
ds = apply_dirichlet(K, F, mesh, bc)
solver = StaticSolver()
u = solver.solve(ds.K_bc, ds.F_bc)

# --- Vérification ---
delta_tip = u[dof_tip_y]
I = section.Iz
delta_EB = F_tip * L**3 / (3 * steel.E * I)
print(f"Flèche FEM     : {delta_tip*1000:.4f} mm")
print(f"Flèche Euler-B : {delta_EB*1000:.4f} mm")

# --- Diagnostics ---
run_diagnostics(mesh, K, u, F, bc)
```

---

## 9. Astuces et pièges courants

**Unités** : tout en SI (Pa, m, kg, N, Hz). Ne pas mélanger mm et m —
une erreur d'un facteur 1000 sur E donne des résultats absurdes
mais sans erreur de calcul.

**Numérotation des ddl** : en interne, le ddl j du nœud i est à
l'indice `i * dof_per_node + j`. Pour Bar2D/Tri3/Quad4 : dof_per_node=2.
Pour Beam2D : dof_per_node=3. Pour les éléments 3D et Beam3D : dof_per_node=3 ou 6.

**Sens des éléments 2D** : les nœuds des Tri3/Quad4 doivent tourner
en sens anti-horaire (CCW). Si l'aire calculée est négative, l'élément
corrige automatiquement mais c'est mieux de le faire dès le maillage.

**Shear locking** : sur des maillages grossiers en flexion, utiliser
`stiffness_matrix_sri()` au lieu de `stiffness_matrix()` pour Quad4 et Hexa8,
ou passer aux éléments quadratiques (Tri6, Tetra10).

**Conditionnement en modal** : utiliser la méthode d'élimination
(par défaut) plutôt que la pénalisation pour éviter les modes parasites.

**V-vector pour Beam3D** : ne doit jamais être parallèle à l'axe
de la poutre, sinon le repère local est indéterminé.

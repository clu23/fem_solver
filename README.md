# fem-solver

Solveur éléments finis en Python pour la mécanique des solides.

## Fonctionnalités

- **Statique** : contraintes et déformations (2D / 3D)
- **Dynamique** : analyse modale, réponse harmonique, transitoire
- **Éléments** : barres, triangles (CST, T6), quadrilatères (Q4), tétraèdres, hexaèdres
- **Solveurs** : SciPy (défaut), MUMPS et PETSc en option pour les gros modèles

## Installation

```bash
git clone git@github.com:clu23/fem_solver.git
cd fem_solver
pip install -e ".[dev]"
```

Pour la visualisation 3D et les solveurs haute performance :

```bash
pip install -e ".[all]"
```

## Utilisation rapide

```python
from femsolver.core.material import ElasticMaterial
from femsolver.core.mesh import Mesh
from femsolver.elements.bar2d import Bar2D
from femsolver.core.assembler import assemble_stiffness
from femsolver.core.boundary import apply_dirichlet
from femsolver.core.solver import solve_static

# Définir matériau (acier)
steel = ElasticMaterial(E=210e9, nu=0.3, rho=7850)

# Créer maillage, assembler, résoudre...
# Voir examples/ pour des cas complets
```

## Tests

```bash
pytest
```

## Structure du projet

```
femsolver/
├── core/          # Maillage, assemblage, solveurs, conditions aux limites
├── elements/      # Bibliothèque d'éléments (bar2d, tri3, quad4, tetra4, hexa8)
├── dynamics/      # Analyse modale et transitoire
├── io/            # Import/export maillages (Gmsh, VTK)
└── postprocess/   # Calcul de contraintes, visualisation
```

## Documentation

- [SPECS.md](SPECS.md) — Spécifications techniques
- [CLAUDE.md](CLAUDE.md) — Instructions pour Claude Code
- [fem_solver_roadmap.md](fem_solver_roadmap.md) — Roadmap et architecture

## Licence

MIT

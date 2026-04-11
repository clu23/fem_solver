# CLAUDE.md — Instructions pour Claude Code

## Identité du projet

Tu travailles sur **fem-solver**, un solveur mécanique par éléments finis écrit en Python. Le projet suit une roadmap en 5 phases (voir `fem_solver_roadmap.md`). Les spécifications techniques détaillées sont dans `SPECS.md`.

---

## Stack technique

- **Python 3.11+** — langage principal
- **NumPy** — calcul matriciel, algèbre linéaire dense
- **SciPy** — matrices creuses (`scipy.sparse`), solveurs (`spsolve`, `eigsh`)
- **Matplotlib** — visualisation 2D
- **PyVista** — visualisation 3D (Phase 4+)
- **meshio** — import/export maillages (Gmsh, VTK, Abaqus)
- **pytest** — tests unitaires

---

## Environnement Python
Toujours utiliser le venv du projet :
- Python : .venv/bin/python3
- Tests : .venv/bin/python3 -m pytest tests/ -v
- Ne jamais chercher d'autre interpréteur Python.

## Conventions de code

### Style général
- **Type hints** sur toutes les fonctions et méthodes (paramètres + retour)
- **Docstrings NumPy** sur toute fonction publique (Parameters, Returns, Examples, Raises)
- **Pas de `print()` dans le code bibliothèque** — utiliser `logging` si nécessaire
- **Pas de `from module import *`** — imports explicites uniquement
- Lignes max : 99 caractères
- Formatage compatible ruff/black

### Nommage

| Concept | Convention | Exemple |
|---------|-----------|---------|
| Matrice de rigidité élémentaire | `K_e` | `K_e = bar.stiffness_matrix(...)` |
| Matrice de rigidité globale | `K` | `K = assembler.assemble_stiffness(...)` |
| Matrice de masse | `M_e` / `M` | élémentaire / globale |
| Vecteur forces | `F` | `F = assembler.assemble_forces(...)` |
| Déplacements | `u` | `u = solver.solve_static(K, F)` |
| Contraintes | `sigma` | `sigma = stress.compute(...)` |
| Déformations | `epsilon` | `epsilon = B @ u_e` |
| Points de Gauss | `gp` ou `gauss_pts` | |
| Fonctions de forme | `N` | `N = shape_functions(xi, eta)` |
| Matrice B (défo-dépl) | `B` | `B = strain_displacement_matrix(...)` |
| Matrice D (comportement) | `D` | `D = material.elasticity_matrix(...)` |
| Nombre de DDL | `n_dof` | |
| Nombre de nœuds | `n_nodes` | |
| Dimension spatiale | `n_dim` | 2 ou 3 |

### Matrices creuses

**Règle absolue** : les matrices globales K et M ne sont JAMAIS denses.

```python
# Pattern d'assemblage standard
from scipy.sparse import coo_matrix

rows, cols, vals = [], [], []
for elem in mesh.elements:
    K_e = elem.etype.stiffness_matrix(elem.material, node_coords)
    dofs = global_dofs(elem)
    for i, di in enumerate(dofs):
        for j, dj in enumerate(dofs):
            rows.append(di)
            cols.append(dj)
            vals.append(K_e[i, j])

K = coo_matrix((vals, (rows, cols)), shape=(n_dof, n_dof)).tocsr()
```

Les matrices élémentaires (K_e, M_e) restent en dense (`np.ndarray`) car elles sont petites (4×4 à 24×24).

---

## Architecture — Règles à respecter

### Séparation des responsabilités

| Module | Responsabilité | Ne fait PAS |
|--------|---------------|-------------|
| `core/element.py` | Classe abstraite Element | Pas de calcul concret |
| `core/material.py` | Propriétés matériau + matrice D | Pas de calcul d'élément |
| `core/mesh.py` | Stockage nœuds/éléments/BCs | Pas de calcul |
| `core/assembler.py` | Assemblage K, M, F globales | Pas de résolution |
| `core/boundary.py` | Application conditions limites | Pas de modification du maillage |
| `core/solver.py` | Résolution K u = F et eigen | Pas d'assemblage |
| `elements/*.py` | Matrices élémentaires concrètes | Pas d'assemblage global |
| `postprocess/stress.py` | Calcul contraintes à partir de u | Pas de visualisation |
| `postprocess/plotter*.py` | Affichage uniquement | Pas de calcul |

### Pattern Strategy pour le solveur

Le solveur utilise un backend interchangeable :

```python
# Ne PAS coder en dur scipy dans solver.py
# Utiliser l'interface SolverBackend

class StaticSolver:
    def __init__(self, backend: SolverBackend = None):
        self.backend = backend or ScipyBackend()

    def solve(self, K, F):
        return self.backend.solve_static(K, F)
```

Cela permettra d'ajouter MUMPS ou PETSc plus tard sans toucher au code appelant.

### Immutabilité

- `Mesh`, `ElasticMaterial`, `ElementData` sont des dataclasses (de préférence `frozen=True`)
- Les fonctions d'assemblage et de résolution retournent de nouveaux objets, jamais de mutation in-place
- Exception : les listes COO d'assemblage (rows, cols, vals) sont mutées pendant la construction, puis gelées en CSR

---

## Tests

### Philosophie

Chaque module a son fichier test. Chaque test compare à une **solution analytique connue**, pas à un résultat "attendu" arbitraire.

### Structure d'un test

```python
import numpy as np
import pytest
from femsolver.core.material import ElasticMaterial
from femsolver.elements.bar2d import Bar2D

class TestBar2DSingleElement:
    """Barre unique en traction — solution analytique δ = FL/(EA)."""

    def setup_method(self):
        self.E = 210e9      # Pa (acier)
        self.A = 1e-4       # m² (section)
        self.L = 1.0        # m (longueur)
        self.F = 10000.0    # N (force)
        self.material = ElasticMaterial(E=self.E, nu=0.3, rho=7800)

    def test_displacement_matches_analytical(self):
        """δ = FL/(EA) = 10000 * 1.0 / (210e9 * 1e-4) ≈ 4.762e-7 m"""
        # ... setup mesh, assemble, solve ...
        delta_analytical = self.F * self.L / (self.E * self.A)
        np.testing.assert_allclose(u_tip, delta_analytical, rtol=1e-12)

    def test_reaction_force_equilibrium(self):
        """Somme des forces = 0 (équilibre statique)."""
        # ... vérifier que R + F = 0 ...
```

### Commande de test standard

```bash
pytest tests/ -v --tb=short
```

### Tolérances

- Résultats exacts (barre unique, patch test) : `rtol=1e-12`
- Convergence maillage (poutre, plaque) : `rtol=0.05` (5%) avec note sur le maillage utilisé
- Fréquences propres : `rtol=0.01` (1%) avec maillage suffisamment fin

---

## Workflow de développement

### Ordre d'implémentation par phase

Toujours suivre cet ordre au sein d'une phase :
1. **Structures de données** (material, mesh) — les fondations
2. **Élément(s)** — matrices élémentaires
3. **Assemblage** — matrice globale
4. **Conditions aux limites** — modification du système
5. **Solveur** — résolution
6. **Post-traitement** — contraintes, visualisation
7. **Tests** — validation analytique
8. **Exemple** — cas d'usage documenté

### Quand tu crées un nouvel élément

1. Hériter de `Element` dans `core/element.py`
2. Implémenter `stiffness_matrix()`, `mass_matrix()`, `dof_per_node()`, `n_nodes()`
3. Écrire le test unitaire avec solution analytique
4. Vérifier le patch test si applicable (éléments 2D/3D continus)
5. Ajouter l'élément dans `elements/__init__.py`

### Quand tu modifies le solveur ou l'assembleur

1. **Toujours relancer tous les tests** après modification : `pytest tests/ -v`
2. Ne jamais casser la compatibilité avec les éléments existants
3. Documenter tout changement d'interface dans les docstrings

---

## Erreurs courantes à éviter

### ❌ Ne PAS faire

```python
# Dense global matrix — INTERDIT au-delà de 100 DDL
K = np.zeros((n_dof, n_dof))

# Modifier le maillage en place
mesh.nodes[3] = [1.0, 2.0]  # Non ! Créer un nouveau Mesh

# Oublier la rotation repère local → global pour Bar2D
K_e = (E * A / L) * np.array([[1, -1], [-1, 1]])  # C'est en LOCAL !

# Hardcoder le nombre de dimensions
if n_dim == 2:  # Préférer des boucles sur n_dim

# Matrice de masse identité ou diagonale comme placeholder
M = scipy.sparse.eye(n_dof)  # Non ! Implémenter la vraie masse consistante
```

### ✅ Faire

```python
# Matrices creuses dès l'assemblage
K = coo_matrix((vals, (rows, cols)), shape=(n_dof, n_dof)).tocsr()

# Rotation explicite pour Bar2D
c, s = cos(theta), sin(theta)
T = np.array([[c, s, 0, 0], [-s, c, 0, 0],
              [0, 0, c, s], [0, 0, -s, c]])
K_e_global = T.T @ K_e_local @ T

# Validation des entrées
assert nodes.shape[1] == n_dim, f"Expected {n_dim}D nodes, got {nodes.shape[1]}D"

# Tests avec solutions analytiques documentées
# Toujours citer la formule de référence dans le docstring du test
```

---

## Gestion des dépendances optionnelles

```python
# Pattern pour les dépendances optionnelles (MUMPS, PyVista)
try:
    import mumps
    HAS_MUMPS = True
except ImportError:
    HAS_MUMPS = False

class MUMPSBackend(SolverBackend):
    def __init__(self):
        if not HAS_MUMPS:
            raise ImportError(
                "python-mumps is required for MUMPSBackend. "
                "Install with: pip install python-mumps"
            )
```

---

## Checklist avant commit

- [ ] Tous les tests passent : `pytest tests/ -v`
- [ ] Type hints sur toutes les fonctions publiques
- [ ] Docstrings NumPy sur toutes les fonctions publiques
- [ ] Pas de matrice dense globale
- [ ] Pas de `print()` dans le code bibliothèque
- [ ] Nouvel élément : test unitaire + patch test inclus
- [ ] Solution analytique citée dans chaque test

# SPECS.md — Spécifications techniques du solveur FEM

## 1. Périmètre fonctionnel

### 1.1 Analyses supportées

| Analyse | Équation gouvernante | Phase | Priorité |
|---------|---------------------|-------|----------|
| Statique linéaire | Ku = F | 1-2 | Critique |
| Analyse modale | Kφ = ω²Mφ | 3 | Haute |
| Réponse harmonique | (K - ω²M + iωC)u = F | 5 | Moyenne |
| Transitoire (Newmark) | Mü + Cu̇ + Ku = F(t) | 5 | Moyenne |

### 1.2 Éléments supportés

| Élément | Dim | Nœuds | DDL/nœud | Intégration | Phase |
|---------|-----|-------|----------|-------------|-------|
| Bar2D | 2D | 2 | 2 (ux, uy) | Analytique | 1 |
| Beam2D | 2D | 2 | 3 (ux, uy, θz) | Analytique | 2 |
| Tri3 (CST) | 2D | 3 | 2 (ux, uy) | Analytique | 2 |
| Quad4 | 2D | 4 | 2 (ux, uy) | Gauss 2×2 | 2 |
| Tri6 | 2D | 6 | 2 (ux, uy) | Gauss 3 pts | 2+ |
| Tetra4 | 3D | 4 | 3 (ux, uy, uz) | 1 pt | 4 |
| Hexa8 | 3D | 8 | 3 (ux, uy, uz) | Gauss 2×2×2 | 4 |

### 1.3 Lois de comportement

Phase 1-4 : **Élasticité linéaire isotrope** uniquement.

```
Contrainte plane :
D = E/(1-ν²) × [[1, ν, 0], [ν, 1, 0], [0, 0, (1-ν)/2]]

Déformation plane :
D = E/((1+ν)(1-2ν)) × [[1-ν, ν, 0], [ν, 1-ν, 0], [0, 0, (1-2ν)/2]]

3D isotrope :
D = matrice 6×6 de Hooke complète
```

Extensions futures possibles : élastoplasticité (Von Mises), hyperélasticité.

---

## 2. Architecture logicielle

### 2.1 Principes de conception

- **Séparation des responsabilités** : chaque module a un rôle unique
- **Pattern Strategy** pour le solveur : interface abstraite, backends interchangeables
- **Immutabilité des données d'entrée** : le maillage et les matériaux ne sont jamais modifiés en place
- **Matrices creuses partout** : jamais de matrice dense pour K ou M globales
- **Validation explicite** : chaque entrée est vérifiée (dimensions, connectivité, matériau)

### 2.2 Interfaces clés

#### Element (classe abstraite)

```python
from abc import ABC, abstractmethod
import numpy as np

class Element(ABC):
    """Interface pour tous les éléments finis."""

    @abstractmethod
    def stiffness_matrix(self, material, nodes) -> np.ndarray:
        """Matrice de rigidité élémentaire en repère global.

        Parameters
        ----------
        material : ElasticMaterial
            Propriétés du matériau.
        nodes : np.ndarray, shape (n_elem_nodes, n_dim)
            Coordonnées des nœuds de l'élément.

        Returns
        -------
        K_e : np.ndarray, shape (n_dof_elem, n_dof_elem)
            Matrice de rigidité élémentaire.
        """
        ...

    @abstractmethod
    def mass_matrix(self, material, nodes) -> np.ndarray:
        """Matrice de masse consistante élémentaire."""
        ...

    @abstractmethod
    def dof_per_node(self) -> int:
        """Nombre de DDL par nœud."""
        ...

    @abstractmethod
    def n_nodes(self) -> int:
        """Nombre de nœuds de l'élément."""
        ...

    def n_dof(self) -> int:
        """Nombre total de DDL de l'élément."""
        return self.n_nodes() * self.dof_per_node()
```

#### SolverBackend (pattern Strategy)

```python
from abc import ABC, abstractmethod
from scipy.sparse import spmatrix

class SolverBackend(ABC):
    """Interface pour les backends de résolution."""

    @abstractmethod
    def solve_static(self, K: spmatrix, F: np.ndarray) -> np.ndarray:
        """Résout Ku = F. Retourne u."""
        ...

    @abstractmethod
    def solve_eigen(self, K: spmatrix, M: spmatrix, n_modes: int
                    ) -> tuple[np.ndarray, np.ndarray]:
        """Résout Kφ = ω²Mφ. Retourne (eigenvalues, eigenvectors)."""
        ...

class ScipyBackend(SolverBackend):
    """Backend par défaut : scipy.sparse.linalg."""
    ...

class MUMPSBackend(SolverBackend):
    """Backend optionnel : python-mumps (Phase 4+)."""
    ...
```

### 2.3 Flux de données principal

```
[Définition modèle]          JSON / Python API
        │
        ▼
[Mesh]                        nœuds (n,d) + connectivité + BCs
        │
        ▼
[Éléments]                    K_e, M_e pour chaque élément
        │
        ▼
[Assembler]                   K, M, F globales (COO → CSR)
        │
        ▼
[Boundary]                    Application Dirichlet (pénalisation)
        │
        ▼
[Solver]                      u = K⁻¹F  ou  (ω², φ) = eigen(K, M)
        │
        ▼
[Stress]                      σ = D·B·u aux points de Gauss
        │
        ▼
[Postprocess / Export]        Matplotlib, PyVista, VTK
```

---

## 3. Structures de données

### 3.1 Mesh

```python
@dataclass
class Mesh:
    nodes: np.ndarray              # (n_nodes, n_dim) — coordonnées
    elements: list[ElementData]    # connectivité + type
    bc_dirichlet: dict[int, dict[int, float]]  # {node: {dof: value}}
    bc_neumann: dict[int, dict[int, float]]    # {node: {dof: force}}
    n_dim: int                     # 2 ou 3

@dataclass
class ElementData:
    etype: type[Element]           # classe de l'élément (Bar2D, Tri3, ...)
    node_ids: list[int]            # indices des nœuds
    material: ElasticMaterial      # matériau associé
    properties: dict               # section, épaisseur, etc.
```

### 3.2 Résultats

```python
@dataclass
class StaticResult:
    displacements: np.ndarray      # (n_nodes, n_dim)
    reactions: np.ndarray          # (n_nodes, n_dim) — forces de réaction aux appuis
    element_stresses: list[np.ndarray]   # σ par élément aux points de Gauss
    element_strains: list[np.ndarray]    # ε par élément
    von_mises: np.ndarray          # σ_VM par nœud (lissé)

@dataclass
class ModalResult:
    frequencies: np.ndarray        # (n_modes,) en Hz
    mode_shapes: np.ndarray        # (n_modes, n_dof) — vecteurs propres normalisés
    effective_mass: np.ndarray     # masse effective modale (optionnel)
```

---

## 4. Stratégie de résolution

### 4.1 Assemblage

- Format COO (triplets row, col, val) pendant l'assemblage
- Conversion en CSR pour la résolution (`coo_matrix.tocsr()`)
- La matrice COO gère nativement les doublons (somme automatique)

### 4.2 Conditions aux limites — Méthode de pénalisation

```python
# Pour chaque DDL imposé i avec valeur u_i :
penalty = 1e20 * max(abs(K.diagonal()))
K[i, i] += penalty
F[i] += penalty * u_i
```

Avantages : ne modifie pas la taille du système, simple à implémenter.
Inconvénient : conditionnement dégradé (acceptable pour solveur direct).

Alternative Phase 4+ : élimination des DDL (réduction du système).

### 4.3 Solveurs

| Backend | Méthode | Usage | Taille max recommandée |
|---------|---------|-------|----------------------|
| SciPy (défaut) | SuperLU / UMFPACK | Statique directe | ~50k DDL |
| SciPy eigsh | Lanczos (ARPACK) | Modale | ~50k DDL |
| MUMPS (optionnel) | Multifrontal direct | Statique/modale grands systèmes | ~500k DDL |
| PETSc (futur) | Krylov itératif | Très grands systèmes | >1M DDL |

### 4.4 Post-traitement des contraintes

1. Calcul aux points de Gauss : `σ_gp = D · B(ξ_gp) · u_e`
2. Extrapolation aux nœuds (pour Quad4 : extrapolation depuis 2×2 Gauss)
3. Lissage nodal : moyenne pondérée des contributions de chaque élément
4. Von Mises : `σ_VM = √(σ_xx² + σ_yy² - σ_xx·σ_yy + 3·τ_xy²)` (2D)

---

## 5. Formats d'entrée/sortie

### 5.1 Modèle JSON (format natif)

```json
{
  "metadata": { "name": "Treillis Warren", "dimension": 2 },
  "materials": [
    { "id": "steel", "E": 210e9, "nu": 0.3, "rho": 7800 }
  ],
  "nodes": [
    [0.0, 0.0],
    [1.0, 0.0],
    [0.5, 0.866]
  ],
  "elements": [
    { "type": "Bar2D", "nodes": [0, 1], "material": "steel", "section": 1e-4 },
    { "type": "Bar2D", "nodes": [1, 2], "material": "steel", "section": 1e-4 }
  ],
  "boundary_conditions": {
    "dirichlet": { "0": { "ux": 0, "uy": 0 }, "1": { "uy": 0 } },
    "neumann": { "2": { "fy": -10000 } }
  },
  "analysis": { "type": "static" }
}
```

### 5.2 Import maillages externes

- **Gmsh** (.msh) via meshio — maillages 2D/3D avec groupes physiques
- **VTK** (.vtk/.vtu) — import/export pour ParaView
- **Abaqus** (.inp) — lecture basique (nœuds + éléments) via meshio

### 5.3 Export résultats

- **VTK** : champs scalaires/vectoriels sur le maillage (déplacements, contraintes)
- **JSON** : résultats numériques bruts (déplacements, fréquences, réactions)
- **PNG/SVG** : visualisations 2D (matplotlib)
- **HTML** : rapport auto-généré (optionnel Phase 5)

---

## 6. Validation & tests

### 6.1 Stratégie de test

Chaque élément et chaque module est validé contre des solutions analytiques connues.

| Test | Solution de référence | Tolérance |
|------|----------------------|-----------|
| Barre 1 élément en traction | δ = FL/(EA) | < 1e-12 (exact) |
| Treillis Warren | Méthode des nœuds (RdM) | < 1e-10 |
| Patch test T3/Q4 | Champ linéaire reproduit exactement | < 1e-10 |
| Poutre console (T3/Q4) | σ_max = M·y/I | < 5% (convergence maillage) |
| Plaque trouée | Kt ≈ 3.0 (Kirsch) | < 10% (maillage modéré) |
| Fréquences barre | f_n = n·√(E/ρ) / (2L) | < 1% (maillage fin) |
| Fréquences poutre EB | f_n analytiques | < 1% |
| Cube compression 3D | σ = F/A uniforme | < 1e-10 |

### 6.2 Framework de test

```bash
# Lancer tous les tests
pytest tests/ -v

# Lancer les tests d'une phase
pytest tests/ -v -k "bar2d or assembly"

# Avec couverture
pytest tests/ --cov=femsolver --cov-report=html
```

---

## 7. Performance cible

| Métrique | Phase 1-3 (2D) | Phase 4 (3D) |
|----------|----------------|--------------|
| Taille max maillage | 10k nœuds | 100k nœuds |
| Temps assemblage | < 1s | < 10s |
| Temps résolution statique | < 2s | < 30s |
| Temps analyse modale (10 modes) | < 5s | < 60s |
| Mémoire pic | < 500 MB | < 4 GB |

Ces cibles sont indicatives et correspondent au backend SciPy sur une machine standard.

---

## 8. Limites connues et hors périmètre

### Hors périmètre (v1)
- Non-linéarité géométrique (grands déplacements)
- Non-linéarité matérielle (plasticité, endommagement)
- Contact et frottement
- Éléments coques et plaques
- Parallélisme MPI
- Maillage adaptatif (raffinement h/p)

### Limitations techniques
- Solveur direct uniquement (pas d'itératif en v1)
- Pas de préconditionnement
- Pas de renumérotation automatique (Cuthill-McKee) en v1
- Conditions aux limites par pénalisation (précision limitée par le facteur de pénalité)

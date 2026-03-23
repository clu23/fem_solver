# FEM Solver — Architecture & Roadmap

## Vue d'ensemble du projet

**Objectif** : Développer un solveur mécanique par éléments finis (FEM) en Python, couvrant l'analyse statique (contraintes/déformations) puis la dynamique des structures (vibrations, modes propres). Le projet démarre en 2D et s'étend progressivement au 3D.

**Outil de développement** : Claude Code  
**Langage** : Python 3.11+  
**Dépendances principales** : NumPy, SciPy, Matplotlib, PyVista (3D), meshio (import/export maillages)

---

## Arborescence du projet

```
fem-solver/
├── README.md                     # Description, installation, exemples rapides
├── pyproject.toml                # Config projet (dependencies, metadata)
├── requirements.txt              # numpy, scipy, matplotlib, pyvista, meshio, pytest
│
├── femsolver/                    # Package principal
│   ├── __init__.py
│   │
│   ├── core/                     # Noyau mathématique
│   │   ├── __init__.py
│   │   ├── element.py            # Classes abstraites d'éléments
│   │   ├── material.py           # Lois de comportement (élastique linéaire, etc.)
│   │   ├── mesh.py               # Structure de données maillage (nœuds, connectivité)
│   │   ├── assembler.py          # Assemblage matrices globales K, M, F
│   │   ├── boundary.py           # Conditions aux limites (Dirichlet, Neumann)
│   │   └── solver.py             # Solveurs (statique, modal, dynamique)
│   │
│   ├── elements/                 # Bibliothèque d'éléments
│   │   ├── __init__.py
│   │   ├── bar2d.py              # Barre/treillis 2D (2 nœuds, 2 ddl/nœud)
│   │   ├── beam2d.py             # Poutre Euler-Bernoulli 2D (2 nœuds, 3 ddl/nœud)
│   │   ├── tri3.py               # Triangle à 3 nœuds (contrainte plane / défo plane)
│   │   ├── quad4.py              # Quadrilatère à 4 nœuds (isoparamétrique)
│   │   ├── tri6.py               # Triangle quadratique (6 nœuds)
│   │   ├── tetra4.py             # Tétraèdre linéaire (3D)
│   │   └── hexa8.py              # Hexaèdre linéaire (3D)
│   │
│   ├── dynamics/                 # Module dynamique (Phase 3+)
│   │   ├── __init__.py
│   │   ├── modal.py              # Analyse modale (valeurs/vecteurs propres)
│   │   ├── harmonic.py           # Réponse harmonique
│   │   └── transient.py          # Intégration temporelle (Newmark, etc.)
│   │
│   ├── io/                       # Entrées/sorties
│   │   ├── __init__.py
│   │   ├── mesh_io.py            # Lecture/écriture maillages (via meshio)
│   │   ├── vtk_export.py         # Export VTK pour ParaView
│   │   └── json_model.py         # Format JSON pour définir un modèle complet
│   │
│   └── postprocess/              # Post-traitement & visualisation
│       ├── __init__.py
│       ├── stress.py             # Calcul contraintes/déformations aux points de Gauss
│       ├── plotter2d.py          # Visualisation 2D (matplotlib)
│       └── plotter3d.py          # Visualisation 3D (pyvista)
│
├── examples/                     # Cas-tests documentés
│   ├── truss_bridge.py           # Treillis pont 2D
│   ├── cantilever_beam.py        # Poutre encastrée
│   ├── plate_with_hole.py        # Plaque trouée (concentration de contraintes)
│   ├── modal_beam.py             # Modes propres poutre
│   └── 3d_bracket.py             # Équerre 3D
│
├── tests/                        # Tests unitaires & validation
│   ├── test_bar2d.py             # Vérification barre élémentaire
│   ├── test_tri3.py              # Patch test triangle
│   ├── test_assembly.py          # Assemblage multi-éléments
│   ├── test_boundary.py          # Conditions aux limites
│   ├── test_solver.py            # Convergence solveur
│   └── test_modal.py             # Fréquences propres vs analytique
│
└── docs/                         # Documentation technique
    ├── theory.md                 # Rappels théoriques (formulations faibles, etc.)
    ├── elements_catalog.md       # Catalogue d'éléments avec matrices
    └── validation_cases.md       # Cas de validation et solutions analytiques
```

---

## Roadmap détaillée

### Phase 1 — Treillis 2D (Fondations)
**Durée estimée** : 1-2 sessions Claude Code  
**Objectif** : Valider la chaîne complète assemblage → résolution → post-traitement

#### Concepts clés
- Matrice de rigidité élémentaire barre 2D (rotation repère local → global)
- Assemblage par table de connectivité (scatter dans matrice globale)
- Application conditions aux limites par pénalisation ou élimination
- Résolution Ku = F (système creux)

#### Livrables
1. `material.py` : classe `ElasticMaterial(E, nu, rho)` — module d'Young, Poisson, densité
2. `bar2d.py` : matrice de rigidité 4×4 en repère global
3. `mesh.py` : structure nœuds + connectivité + conditions aux limites
4. `assembler.py` : assemblage K globale (format COO → CSR)
5. `boundary.py` : imposition déplacements imposés
6. `solver.py` : résolution statique `scipy.sparse.linalg.spsolve`
7. `plotter2d.py` : affichage maillage déformé + efforts dans les barres
8. **Cas-test** : treillis Warren (6 barres, 4 nœuds) — comparaison solution analytique

#### Validation
- Treillis à 1 barre : solution analytique δ = FL/EA
- Treillis Warren : réactions d'appui, efforts normaux dans chaque barre

---

### Phase 2 — Éléments 2D continus (CST, Q4)
**Durée estimée** : 2-3 sessions Claude Code  
**Objectif** : Résoudre des problèmes de contrainte plane / déformation plane

#### Concepts clés
- Fonctions de forme (linéaires pour T3, bilinéaires pour Q4)
- Matrice B (déformation-déplacement) et matrice D (loi de comportement)
- Intégration numérique (Gauss) pour Q4
- Contrainte plane vs déformation plane (choix de D)
- Calcul des contraintes σ = D·B·u aux points de Gauss

#### Livrables
1. `tri3.py` : triangle CST — matrice de rigidité analytique (pas de Gauss nécessaire)
2. `quad4.py` : quadrilatère isoparamétrique — intégration 2×2 Gauss
3. `stress.py` : calcul et lissage des contraintes (Von Mises, principales)
4. `plotter2d.py` (enrichi) : carte de couleurs des contraintes/déplacements
5. **Cas-tests** :
   - Patch test (vérification convergence éléments)
   - Poutre console en flexion : σ_xx vs solution RdM
   - Plaque trouée : concentration de contraintes (Kt ≈ 3)

#### Validation
- Patch test : champ de déplacement linéaire reproduit exactement
- Convergence : raffiner le maillage → solution analytique

---

### Phase 3 — Analyse modale 2D
**Durée estimée** : 1-2 sessions Claude Code  
**Objectif** : Calculer fréquences propres et modes de vibration

#### Concepts clés
- Matrice de masse consistante M (même fonctions de forme que K)
- Matrice de masse condensée (lumped) — alternative simplifiée
- Problème aux valeurs propres généralisé : Kφ = ω²Mφ
- Solveur `scipy.sparse.linalg.eigsh` (Lanczos, matrices symétriques)
- Fréquences propres f_n = ω_n / (2π)

#### Livrables
1. Matrices de masse pour bar2d, tri3, quad4
2. `modal.py` : extraction des n premiers modes propres
3. Visualisation des déformées modales animées
4. **Cas-tests** :
   - Barre en traction : f_n = n·c/(2L) avec c = √(E/ρ)
   - Poutre Euler-Bernoulli : f_n analytiques

#### Validation
- Erreur < 1% sur les 5 premières fréquences vs analytique (maillage fin)

---

### Phase 4 — Extension 3D
**Durée estimée** : 2-3 sessions Claude Code  
**Objectif** : Passer aux éléments volumiques 3D

#### Concepts clés
- 3 ddl par nœud (u, v, w)
- Matrice D 6×6 (élasticité 3D isotrope)
- Matrice B 6×(3×nb_noeuds)
- Intégration de Gauss 3D (tétraèdres : 1 ou 4 points ; hexaèdres : 2×2×2)
- Import de maillages depuis Gmsh (format .msh via meshio)

#### Livrables
1. `tetra4.py` : tétraèdre linéaire (matrice analytique, 1 point de Gauss)
2. `hexa8.py` : hexaèdre trilinéaire (intégration 2×2×2)
3. `mesh_io.py` : lecture fichiers Gmsh
4. `plotter3d.py` : visualisation PyVista (maillage, déformées, contraintes)
5. `vtk_export.py` : export pour ParaView
6. **Cas-test** : cube en compression, équerre 3D

---

### Phase 5 — Dynamique avancée (optionnel)
**Durée estimée** : 2+ sessions Claude Code

#### Livrables
1. `harmonic.py` : réponse fréquentielle (balayage en fréquence)
2. `transient.py` : intégration temporelle Newmark-β
3. Amortissement de Rayleigh (C = αM + βK)
4. Visualisation temporelle animée

---

## Conventions de code

### Pour Claude Code
- **Docstrings** : format NumPy (Parameters, Returns, Examples)
- **Type hints** : partout (np.ndarray, float, etc.)
- **Tests** : pytest, un fichier test par module, comparaison à des solutions analytiques
- **Nommage** :
  - Matrices : `K_e` (rigidité élémentaire), `K` (globale), `M` (masse), `F` (forces)
  - Déplacements : `u` (vecteur global)
  - Contraintes : `sigma`, déformations : `epsilon`

### Formats de données internes
```python
# Nœuds : array (n_nodes, n_dim)
nodes = np.array([[0, 0], [1, 0], [0.5, 0.866]])

# Connectivité : liste de tuples (type_element, [node_ids])
elements = [(BarElement, [0, 1]), (BarElement, [1, 2])]

# Conditions aux limites
# Dirichlet : {node_id: {dof: value}}  (dof = 0 pour ux, 1 pour uy, 2 pour uz)
bc_dirichlet = {0: {0: 0.0, 1: 0.0}, 3: {1: 0.0}}

# Neumann : {node_id: {dof: value}}
bc_neumann = {2: {1: -1000.0}}  # Force de -1000 N en y sur nœud 2
```

### Matrices creuses
```python
# Assemblage en format COO puis conversion CSR
from scipy.sparse import coo_matrix, csr_matrix

rows, cols, vals = [], [], []
for elem in elements:
    ke = elem.stiffness_matrix()
    dofs = elem.global_dofs()
    for i, di in enumerate(dofs):
        for j, dj in enumerate(dofs):
            rows.append(di)
            cols.append(dj)
            vals.append(ke[i, j])

K = coo_matrix((vals, (rows, cols)), shape=(n_dof, n_dof)).tocsr()
```

---

## Ressources théoriques de référence

| Sujet | Source recommandée |
|-------|-------------------|
| Formulation éléments finis | « FEA Concepts » — NPTEL (gratuit, vidéos) |
| Matrices élémentaires barres | Toute ref RdM : K = (EA/L) × [1,-1;-1,1] en local |
| Éléments T3/Q4 | Logan, « A First Course in FEM » — chapitres 6-8 |
| Intégration de Gauss | Quadratures 1D/2D/3D : points + poids tabulés |
| Analyse modale | Kφ = ω²Mφ, solveur Lanczos (eigsh) |
| Newmark-β | Hughes, « The FEM — Linear Static and Dynamic Analysis » |

---

## Prompt d'initialisation pour Claude Code

```
Initialise le projet fem-solver selon l'arborescence définie dans
fem_solver_roadmap.md. Commence par :
1. Créer la structure de dossiers et les __init__.py
2. Implémenter pyproject.toml avec les dépendances
3. Implémenter Phase 1 (treillis 2D) dans cet ordre :
   a. material.py (ElasticMaterial)
   b. mesh.py (Mesh : nœuds, connectivité, BCs)
   c. bar2d.py (Bar2D : matrice de rigidité locale → globale)
   d. assembler.py (assemblage K globale en COO→CSR)
   e. boundary.py (application Dirichlet par pénalisation)
   f. solver.py (résolution statique Ku=F)
   g. plotter2d.py (maillage déformé + efforts)
4. Écrire un exemple truss_bridge.py (treillis Warren)
5. Écrire les tests avec pytest (test_bar2d.py, test_assembly.py)
6. Valider vs solution analytique

Conventions : type hints, docstrings NumPy, matrices creuses scipy.
```

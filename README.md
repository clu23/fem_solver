# fem-solver

Solveur éléments finis pour la mécanique des solides, écrit en Python. Couvre l'analyse statique, le flambage, la dynamique modale et transitoire, et la réponse aléatoire (PSD). Les modèles se définissent en JSON et se lancent en une commande.

**917 tests · 14 types d'éléments · 7 types d'analyse · Python 3.11+**

---

## Quickstart

```bash
# 1. Installation
git clone https://github.com/clu23/fem_solver.git
cd fem_solver
pip install -e ".[dev]"

# 2. Vérifier un modèle JSON
python -m femsolver validate examples/warren_truss.json

# 3. Résoudre et afficher les résultats
python -m femsolver run examples/warren_truss.json
```

---

## Fonctionnalités

### Éléments finis

| Élément | Dim | DDL/nœud | Formulation |
|---------|-----|----------|-------------|
| `Bar2D` | 2D | 2 | Barre/treillis, masse consistante |
| `Beam2D` | 2D | 3 | Euler-Bernoulli, intégration exacte |
| `Beam2DTimoshenko` | 2D | 3 | Timoshenko, réduction du blocage en cisaillement |
| `Beam3D` | 3D | 6 | Timoshenko 3D, torsion de Saint-Venant, offset de section |
| `Tri3` | 2D | 2 | CST — contrainte plane / déformation plane |
| `Tri6` | 2D | 2 | Triangle quadratique, 3 points de Gauss |
| `Quad4` | 2D | 2 | Isoparamétrique, 2×2 Gauss |
| `Quad8` | 2D | 2 | Sérendipité, 3×3 Gauss |
| `Tetra4` | 3D | 3 | Tétraèdre linéaire |
| `Tetra10` | 3D | 3 | Tétraèdre quadratique |
| `Hexa8` | 3D | 3 | Hexaèdre linéaire, 2×2×2 Gauss |
| `Hexa20` | 3D | 3 | Hexaèdre sérendipité, 3×3×3 Gauss |
| `Spring` | 2D/3D | flexible | Ressort ponctuel (CBUSH) — 2 nœuds ou sol, translation/rotation |
| `Damper` | 2D/3D | flexible | Amortisseur visqueux ponctuel (CBUSH) — 2 nœuds ou sol |

Les connecteurs ponctuels `Spring` / `Damper` sont alignés sur les axes globaux et sans matériau : la raideur (`stiffness`) ou le coefficient visqueux (`damping`) est fourni DDL par DDL. L'amortisseur s'assemble dans une matrice C globale qui s'ajoute à l'amortissement de Rayleigh dans les analyses dynamiques visqueuses.

Toutes les matrices globales (K, M) sont creuses (`scipy.sparse.csr_matrix`). Les matrices élémentaires restent denses (4×4 à 60×60).

### Sections de poutre

Sept profils paramétriques utilisables pour `Beam2D`, `Beam2DTimoshenko` et `Beam3D` : rectangle plein, circulaire plein, tube circulaire, tube rectangulaire (RHS), profilé en I, profilé en C (U), cornière en L. Chaque section calcule A, Iz, Iy, J et le facteur de correction de cisaillement κ de Cowper.

### Types d'analyse

| Type | Description |
|------|-------------|
| `static` | Résolution K·u = F, élimination vraie des conditions de Dirichlet |
| `modal` | Problème aux valeurs propres généralisé, Lanczos (`eigsh`) |
| `buckling` | Flambage linéaire K·φ = λ(−K_g)·φ, rigidité géométrique K_g |
| `harmonic` | Balayage en fréquence Z(Ω)·U = F̂ |
| `transient` | Intégration Newmark-β (trapèze inconditionnel γ=½, β=¼) |
| `random_force` | Réponse PSD à une excitation en force, G_u(f) = \|H(f)\|²·G_F(f) |
| `random_base` | Réponse PSD à une excitation de base, équation de Miles incluse |

Modèles d'amortissement : Rayleigh (αM + βK), hystérétique (K·iη), modal (ξₙ par mode).

**Vérification du modèle avant résolution** (style Abaqus : vérifier, avertir, bloquer si grave). Avant chaque résolution, `run_model_checks` détecte les défauts de modélisation :

- *Erreurs bloquantes* (lèvent `ModelError`) : nœuds orphelins, Jacobien négatif/nul aux points de Gauss (élément retourné/dégénéré), singularité après BCs (factorisation LDLᵀ de K_free, avec identification du DDL — par ex. `Nœud 7 (rotation θz)`).
- *Avertissements* (loggés, le calcul continue) : nœuds coïncidents, éléments dupliqués, qualité (aspect ratio > 10, angles < 10° ou > 170°), conditionnement de K > 10¹².

Pas d'AUTOSPC, pas de correction automatique — on bloque ou on avertit, c'est à l'utilisateur de corriger.

### Post-traitement

- Contraintes et déformations aux points de Gauss, lissage nodal (moyenne pondérée)
- Contrainte de Von Mises 2D et 3D
- Diagrammes d'efforts internes M/V/N (et torsion T en 3D) pour les poutres, récupérés par équilibre (moment parabolique / tranchant linéaire exacts sous charge répartie)
- Export VTU (ParaView) : déplacements, contraintes, Von Mises — tous types d'éléments
- Visualisation 2D (Matplotlib) et 3D (PyVista)
- Estimateur d'erreur par élément pour piloter le raffinement de maillage

### Interface en ligne de commande

```
python -m femsolver run      model.json [--detailed] [--export out.vtu]
                                         [--diagrams out.png] [--quiet]
python -m femsolver validate model.json
python -m femsolver check    model.json
python -m femsolver info     model.json
```

- **`validate`** — vérifie la syntaxe JSON, les types d'éléments, les matériaux et les sections sans aucun calcul ; signale un système potentiellement singulier ou un modèle sans chargement
- **`check`** — vérifie la **santé du modèle** (style Abaqus) : nœuds orphelins, Jacobien négatif/nul, singularité après BCs (erreurs bloquantes) ; nœuds coïncidents, éléments dupliqués, mauvaise qualité, conditionnement (avertissements). Ces contrôles tournent aussi automatiquement avant chaque résolution
- **`info`** — affiche nœuds, DDL, éléments, matériaux distincts, conditions aux limites et paramètres d'analyse
- **`run`** — résout et affiche les résultats ; pour les analyses statiques : top-5 nœuds les plus déplacés avec coordonnées, réactions d'appui, bilan d'équilibre global, et efforts de barre top-5 traction/compression (treillis uniquement)
- **`--detailed`** — tableau complet nœud par nœud avec toutes les composantes DDL
- **`--export out.vtu`** — écrit le champ de déplacement dans un fichier VTU sans encombrer le terminal
- **`--diagrams [out.png]`** — trace les diagrammes d'efforts internes M/V/N des poutres (analyse statique) ; sans valeur, enregistre `<modèle>_diagrams.png`

### API Python

```python
from femsolver.io import load_model, solve_model

model = load_model("examples/warren_truss.json")
results = solve_model(model)   # dict Python sérialisable
print(results["u"])            # déplacements nodaux [m]
```

---

## Exemple de sortie CLI

```
$ python -m femsolver run examples/warren_truss.json
==============================================================
  RÉSULTATS : Warren Truss  [static]
==============================================================

  ────────────────────────────────────────────────────────
  Déplacements nodaux
    u_max absolu                 2.6831e-03 m
    u_moy absolu                 1.5031e-03 m
    Norme L2                     5.1025e-03 m

  Top 5 nœuds les plus déplacés :
     Nœud         x         y          ux          uy         |u|
    ─────────────────────────────────────────────────────────────
        2     2.000     0.000   4.762e-04  -2.641e-03   2.683e-03
        6     1.500     1.000   7.143e-04  -2.189e-03   2.302e-03
        7     2.500     1.000   2.381e-04  -2.189e-03   2.202e-03
        3     3.000     0.000   8.333e-04  -1.558e-03   1.767e-03
        1     1.000     0.000   1.190e-04  -1.558e-03   1.563e-03

  ────────────────────────────────────────────────────────
  Réactions d'appui
     Nœud   DDL    Réaction [N]
    ────────────────────────────
        0    ux     -1.0914e-11
        0    uy      5.0000e+03
        4    uy      5.0000e+03

  ────────────────────────────────────────────────────────
  Bilan d'équilibre
     Dir     F_appliquée [N]      F_réaction [N]      Résidu [N]
    ────────────────────────────────────────────────────────────
       x          0.0000e+00         -1.0465e-11     -1.0465e-11  ✓
       y         -1.0000e+04          1.0000e+04     -7.2760e-12  ✓

  ────────────────────────────────────────────────────────
  Efforts dans les barres (treillis)
  Top 5 traction :
     Barre         Nœuds           N [N]
    ──────────────────────────────────────
         1        (1, 2)      7.5000e+03
         2        (2, 3)      7.5000e+03
        11        (2, 7)      5.5902e+03

  Top 5 compression :
     Barre         Nœuds           N [N]
    ──────────────────────────────────────
         5        (6, 7)     -1.0000e+04
        12        (3, 7)     -5.5902e+03
         7        (0, 5)     -5.5902e+03
```

---

## Format JSON

Un modèle complet se définit dans un fichier JSON. Aucun Python requis pour les cas courants.

```json
{
  "name": "Warren Truss",
  "materials": {
    "steel": { "E": 210e9, "nu": 0.3, "rho": 7800 }
  },
  "nodes": [[0,0], [1,0], [2,0], [0.5,1], [1.5,1]],
  "elements": [
    { "type": "Bar2D", "nodes": [0,1], "material": "steel", "area": 1e-4 },
    { "type": "Beam2D", "nodes": [0,1], "material": "steel",
      "section": {"type": "rectangular", "width": 0.05, "height": 0.10} }
  ],
  "boundary_conditions": {
    "dirichlet": [{"node": 0, "dof": 0}, {"node": 0, "dof": 1}],
    "neumann":   [{"node": 2, "dof": 1, "value": -5000.0}],
    "body_force": [0.0, -9.81]
  },
  "analysis": {
    "type": "harmonic",
    "freqs": {"linspace": [1.0, 300.0, 600]},
    "F_hat": [{"node": 5, "dof": 1, "value": 1.0}],
    "damping": {"type": "rayleigh", "alpha": 0.0, "beta": 1.9e-4}
  }
}
```

Le schéma complet (sections, pressions, charges distribuées, amortissement hystérétique, spécification logspace…) est documenté dans [`docs/json_schema.md`](docs/json_schema.md).

---

## Galerie d'exemples

Chaque fichier est exécutable directement : `python -m femsolver run examples/<nom>.json`.

| Fichier | Analyse | Description |
|---------|---------|-------------|
| `warren_truss.json` | Statique | Treillis Warren 9 nœuds × 15 Bar2D, charge centrale 10 kN. Réactions, bilan d'équilibre, efforts de barre top-5. |
| `cantilever_beam.json` | Statique | Console Beam2D 6 nœuds, F=−5 kN en bout. Résultat exact : δ = PL³/(3EI) = −1.905 mm à rtol=1×10⁻¹⁰. |
| `cantilever_distributed_beam.json` | Statique | Console Beam2D 4 éléments, charge répartie q=−8 kN/m. Diagrammes `--diagrams` : V linéaire (\|V\|max=q·L=16 kN), M parabolique (\|M\|max=q·L²/2=16 kN·m), exacts. |
| `cantilever_modal.json` | Modale | Console 11 nœuds, 3 modes. f₁=83.8 Hz, f₂=525 Hz — comparaison avec les βₙL d'Euler-Bernoulli. |
| `euler_column_buckling.json` | Flambage | Colonne encastrée-libre 20 Beam2D, P_ref=1 N. λ_cr=431.80 contre P_cr=431.80 N analytique (err < 0.01 %). |
| `plate_hole_quad4.json` | Statique | Quart de plaque trouée, maillage polaire 12 Quad4, σ_y=1 MPa. Concentration de contraintes → Kt=3 de Kirsch au raffinement. |
| `harmonic_cantilever.json` | Harmonique | Console 10 éléments, balayage 1–300 Hz, amortissement Rayleigh 5 % (β=1.9×10⁻⁴ s). Pic de résonance à f₁. |
| `transient_impact_beam.json` | Transitoire | Force échelon 1 kN sur console, Newmark 2000 pas Δt=50 µs. DAF=1.83 (borne théorique 2 sans amortissement). |
| `random_base_sdof.json` | Aléatoire base | SDOF Bar2D, k=1 kN/m, m=1 kg, G₀=0.01 (m/s²)²/Hz. σ_u=0.39 mm et σ_a=1.17 m/s² — Miles err < 0.4 %. |
| `portal_frame_3d.json` | Statique 3D | Portique Beam3D 4 nœuds, F_x=10 kN. Réactions d'encastrement 3D, bilan d'équilibre toutes directions. |
| `spring_support_static.json` | Statique | Barre Bar2D + ressort ponctuel `Spring` (appui élastique). Cohabitation élément structural / connecteur. Exact : u₁ₓ=FₓL/(EA), u₁ᵧ=Fᵧ/kᵧ, équilibre vérifié. |
| `damper_harmonic_sdof.json` | Harmonique | SDOF Bar2D amorti par un amortisseur ponctuel `Damper` seul (ζ=5 %). Pic de résonance fini à f≈5.03 Hz, amplitude≈F/(k·2ζ)=0.1 m. |

---

## Tests

```bash
# Suite complète (917 tests, ~15 s)
python -m pytest tests/ -v

# Fichier unique
python -m pytest tests/test_beam2d.py -v

# Avec couverture
python -m pytest tests/ --cov=femsolver --cov-report=term-missing
```

Chaque test compare à une **solution analytique documentée** — pas de valeur arbitraire « attendue ». Les tolérances sont explicites : `rtol=1e-12` pour les cas exacts (barre unique, patch test), `rtol=0.01` pour la convergence maillage, `rtol=0.01` pour les fréquences propres.

---

## Structure du projet

```
fem-solver/
├── femsolver/
│   ├── __main__.py              # CLI : run / validate / info
│   ├── core/
│   │   ├── element.py           # Classe abstraite Element
│   │   ├── material.py          # ElasticMaterial, matrice de comportement D
│   │   ├── mesh.py              # Mesh, ElementData, BoundaryConditions,
│   │   │                        #   PressureLoad, BodyForce, DistributedLineLoad
│   │   ├── sections.py          # 7 profils : RectangularSection, CircularSection,
│   │   │                        #   HollowCircular, HollowRectangular, ISection,
│   │   │                        #   CSection, LSection
│   │   ├── assembler.py         # Assemblage COO→CSR : K, M, F, K_g (rigidité géo.)
│   │   ├── boundary.py          # apply_dirichlet (élimination vraie), DirichletSystem
│   │   ├── solver.py            # StaticSolver, ModalSolver, BucklingSolver
│   │   ├── mpc.py               # Contraintes multi-points (élimination / Lagrange)
│   │   ├── diagnostics.py       # Masse/réactions/équilibre + détection de mécanismes
│   │   └── model_check.py       # Santé du modèle pré-résolution (orphelins, J, singularité…)
│   ├── elements/
│   │   ├── bar2d.py             # axial_force, geometric_stiffness_matrix
│   │   ├── beam2d.py            # Euler-Bernoulli, section obj. ou scalaires
│   │   ├── beam2d_timoshenko.py # Timoshenko, réduction du blocage
│   │   ├── beam3d.py            # Timoshenko 3D, Wagner torsion, v-vector
│   │   ├── tri3.py / tri6.py
│   │   ├── quad4.py / quad8.py
│   │   ├── tetra4.py / tetra10.py
│   │   ├── hexa8.py / hexa20.py
│   │   ├── spring.py            # SpringElement — ressort ponctuel (CBUSH)
│   │   └── damper.py            # DamperElement — amortisseur visqueux ponctuel
│   ├── dynamics/
│   │   ├── modal.py             # run_modal → ModalResult (freqs, omega, modes)
│   │   ├── harmonic.py          # run_harmonic (Rayleigh / hystérétique / modal)
│   │   ├── transient.py         # run_transient, NewmarkBeta, TransientResult
│   │   ├── random_response.py   # run_random_force, run_random_base, miles_equation
│   │   ├── damping.py           # HystereticDamping, ModalDampingModel
│   │   └── rayleigh.py          # RayleighDamping, rayleigh_from_modes
│   ├── io/
│   │   ├── json_model.py        # load_model, solve_model, run_from_json, FEModel
│   │   └── mesh_io.py           # read_mesh (Gmsh/Abaqus), write_vtu (tous éléments)
│   └── postprocess/
│       ├── stress.py            # nodal_stresses, von_mises_2d
│       ├── stress3d.py          # contraintes 3D
│       ├── plotter2d.py         # Matplotlib
│       ├── plotter3d.py         # PyVista
│       ├── beam_diagrams.py     # diagrammes d'efforts internes M/V/N (poutres)
│       └── error_estimator.py   # indicateur ZZ par élément
├── tests/                       # 32 fichiers, 917 tests
├── examples/                    # 12 modèles JSON + scripts Python annotés
├── docs/
│   └── json_schema.md           # Schéma JSON complet avec exemples
├── SPECS.md                     # Spécifications techniques détaillées
└── pyproject.toml
```

---

## Dépendances

| Paquet | Version min. | Rôle |
|--------|-------------|------|
| NumPy | 1.24 | Algèbre linéaire dense, matrices élémentaires |
| SciPy | 1.11 | Matrices creuses CSR, `eigsh`, `spsolve` |
| Matplotlib | 3.7 | Visualisation 2D |
| meshio | 5.3 | Export VTU, import Gmsh / Abaqus / VTK |
| PyVista ≥ 0.42 | *(optionnel)* | Visualisation 3D interactive |

```bash
# Visualisation 3D incluse
pip install -e ".[all]"
```

---

## Licence

MIT

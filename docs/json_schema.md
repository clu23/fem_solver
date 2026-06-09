# Schéma JSON pour les modèles FEM

Ce document décrit le format JSON utilisé par `femsolver.io.json_model` pour définir
un modèle FEM complet sans écrire de Python.

**Chargement :**

```python
from femsolver.io import load_model, run_from_json

model = load_model("mon_modele.json")   # → FEModel
results = run_from_json("mon_modele.json")  # → dict
```

---

## Structure de premier niveau

```json
{
  "name": "Nom du modèle",
  "description": "Description libre (optionnel)",
  "materials": { ... },
  "nodes": [ ... ],
  "elements": [ ... ],
  "boundary_conditions": { ... },
  "analysis": { ... }
}
```

| Clé | Requis | Type | Description |
|-----|--------|------|-------------|
| `name` | non | string | Nom du modèle (défaut : nom du fichier sans extension) |
| `description` | non | string | Description libre |
| `materials` | oui | objet | Dictionnaire de matériaux (clé → propriétés) |
| `nodes` | oui | tableau | Liste des coordonnées nodales |
| `elements` | oui | tableau | Liste des éléments |
| `boundary_conditions` | oui | objet | Conditions aux limites |
| `analysis` | non | objet | Type d'analyse (défaut : `{"type": "static"}`) |

---

## Matériaux

```json
"materials": {
    "acier": { "E": 210e9, "nu": 0.3, "rho": 7800 },
    "alu":   { "E": 70e9,  "nu": 0.33, "rho": 2700 }
}
```

| Paramètre | Requis | Unité | Description |
|-----------|--------|-------|-------------|
| `E` | oui | Pa | Module d'Young |
| `nu` | oui | — | Coefficient de Poisson |
| `rho` | non | kg/m³ | Masse volumique (défaut : 0) |

---

## Nœuds

Tableau de coordonnées. Chaque ligne est un nœud `[x, y]` (2D) ou `[x, y, z]` (3D).

```json
"nodes": [
    [0.0, 0.0],
    [1.0, 0.0],
    [2.0, 0.0]
]
```

La dimension spatiale (`n_dim`) est déduite automatiquement de la longueur des
vecteurs (2 → 2D, 3 → 3D).

---

## Éléments

Chaque élément est un objet avec au minimum `type`, `nodes` et `material`.

```json
"elements": [
    {
        "type": "Bar2D",
        "nodes": [0, 1],
        "material": "acier",
        "area": 1e-4
    }
]
```

### Types d'éléments supportés

| `type` | n_dim | DDL/nœud | Propriétés supplémentaires requises |
|--------|-------|----------|--------------------------------------|
| `Bar2D` | 2 | 2 | `area` [m²] |
| `Beam2D` | 2 | 3 | `section` ou (`area` + `inertia`) |
| `Beam2DTimoshenko` | 2 | 3 | `section` ou (`area` + `inertia`) |
| `Beam3D` | 3 | 6 | `section` (obligatoire) + `v_vec` (optionnel) |
| `Tri3` | 2 | 2 | `thickness` [m] (optionnel) |
| `Tri6` | 2 | 2 | `thickness` [m] (optionnel) |
| `Quad4` | 2 | 2 | `thickness` [m] (optionnel) |
| `Quad8` | 2 | 2 | `thickness` [m] (optionnel) |
| `Tetra4` | 3 | 3 | — |
| `Tetra10` | 3 | 3 | — |
| `Hexa8` | 3 | 3 | — |
| `Hexa20` | 3 | 3 | — |
| `Spring` | 2/3 | flexible | `stiffness` (réel ou liste par DDL) |
| `Damper` | 2/3 | flexible | `damping` (réel ou liste par DDL) |

**Règle :** tous les éléments d'un même maillage doivent avoir le même nombre
de DDL par nœud. On ne peut pas mélanger `Bar2D` (2 DDL) et `Beam2D` (3 DDL).

### Connecteurs ponctuels : `Spring` et `Damper`

Ressort et amortisseur ponctuels (type CBUSH de Nastran), alignés sur les axes
globaux. Ils relient **deux nœuds** (`"nodes": [i, j]`) ou **un nœud et le sol**
(`"nodes": [i]`), en 1D, 2D ou 3D :

- ressort sol  : `K_e = diag(k)` ;
- ressort 2 nœuds : `K_e = [[diag(k), −diag(k)], [−diag(k), diag(k)]]`
  (idem pour l'amortisseur avec `damping`).

Spécificités :

- **Pas de `material`** : la clé est optionnelle (un connecteur ponctuel n'a pas
  de matériau continu). Le coefficient est donné directement.
- **DDL flexible** : la longueur du vecteur `stiffness` / `damping` fixe le
  nombre de DDL par nœud et **doit égaler** le `dof_per_node` du maillage. Une
  valeur nulle désactive le DDL correspondant.
- `Spring` agit en statique et dans toutes les analyses (sa raideur entre dans K).
- `Damper` n'apporte qu'un amortissement visqueux, assemblé dans la matrice C.
  Il agit dans les analyses dynamiques **visqueuses** (harmonique, transitoire,
  réponse aléatoire) avec amortissement `None` ou `rayleigh`, et s'**ajoute** à
  l'amortissement de Rayleigh. Il est incompatible avec un amortissement
  hystérétique ou modal (le solveur lève alors une erreur explicite).

```jsonc
"elements": [
    // ressort au sol, modèle 2D : kx=1e5, ky=2e5 [N/m]
    {"type": "Spring", "nodes": [0], "stiffness": [1.0e5, 2.0e5]},
    // ressort entre 2 nœuds, raideur en y seulement
    {"type": "Spring", "nodes": [0, 1], "stiffness": [0.0, 5.0e5]},
    // amortisseur au sol, c_x=3.16 [N·s/m], c_y=0
    {"type": "Damper", "nodes": [1], "damping": [3.162, 0.0]}
]
```

### Propriétés de section pour Beam2D / Beam2DTimoshenko

**Option A — objet Section (recommandée) :**

```json
{
    "type": "Beam2D", "nodes": [0, 1], "material": "acier",
    "section": {"type": "rectangular", "width": 0.05, "height": 0.10}
}
```

**Option B — scalaires directs :**

```json
{
    "type": "Beam2D", "nodes": [0, 1], "material": "acier",
    "area": 5e-3, "inertia": 4.167e-6
}
```

### Propriétés de section pour Beam3D

```json
{
    "type": "Beam3D", "nodes": [0, 1], "material": "acier",
    "section": {"type": "circular", "radius": 0.05},
    "v_vec": [0.0, 1.0, 0.0]
}
```

`v_vec` est le vecteur d'orientation du plan de flexion forte (plan x-y de l'élément).
Défaut : calculé automatiquement si omis.

---

## Sections transversales

Les sections sont définies inline dans l'élément avec un objet `section`.

### `rectangular` — Section rectangulaire pleine

```json
{"type": "rectangular", "width": 0.05, "height": 0.10}
```

| Paramètre | Unité | Description |
|-----------|-------|-------------|
| `width` | m | Dimension en z (largeur b) |
| `height` | m | Dimension en y (hauteur h) |

### `circular` — Section circulaire pleine

```json
{"type": "circular", "radius": 0.05}
```

### `hollow_circular` — Section circulaire creuse (tube)

```json
{"type": "hollow_circular", "outer_radius": 0.05, "inner_radius": 0.04}
```

### `hollow_rectangular` — Section rectangulaire creuse (RHS)

```json
{"type": "hollow_rectangular", "outer_width": 0.10, "outer_height": 0.15, "thickness": 0.005}
```

### `i_section` — Profilé en I

```json
{
    "type": "i_section",
    "flange_width": 0.10,
    "height": 0.20,
    "flange_thickness": 0.0085,
    "web_thickness": 0.0056
}
```

### `c_section` — Profilé en C (U)

Mêmes paramètres que `i_section`.

### `l_section` — Profilé en L (cornière)

```json
{
    "type": "l_section",
    "flange_width": 0.08,
    "height": 0.08,
    "flange_thickness": 0.008,
    "web_thickness": 0.008
}
```

---

## Conditions aux limites

```json
"boundary_conditions": {
    "dirichlet": [ ... ],
    "neumann":   [ ... ],
    "pressure":  [ ... ],
    "body_force": [0.0, -9.81],
    "distributed": [ ... ]
}
```

### Dirichlet — déplacements imposés

```json
"dirichlet": [
    {"node": 0, "dof": 0, "value": 0.0},
    {"node": 0, "dof": 1, "value": 0.0},
    {"node": 0, "dof": 2, "value": 0.0}
]
```

`dof` est l'indice **local** au nœud (0 = ux, 1 = uy, 2 = uz ou θz pour poutre).
`value` est le déplacement imposé en mètres (défaut : 0.0 si omis).

### Neumann — forces nodales

```json
"neumann": [
    {"node": 5, "dof": 1, "value": -5000.0}
]
```

`value` est la force en Newtons.

### Pression surfacique

```json
"pressure": [
    {"nodes": [1, 2],       "magnitude": 10000.0},
    {"nodes": [3, 4, 5, 6], "magnitude": 5000.0}
]
```

`nodes` liste les nœuds de la face/arête dans l'ordre antihoraire (CCW).
`magnitude` est la pression en Pascals (positif = compression).

### Force de volume (gravité)

```json
"body_force": [0.0, -9.81]
```

Vecteur d'accélération en m/s². `b = ρ × acceleration`.

### Charges linéiques distribuées (Bar2D, Beam2D)

```json
"distributed": [
    {"nodes": [0, 1], "qx": 0.0, "qy": -5000.0}
]
```

`qx` : charge axiale [N/m], `qy` : charge transverse [N/m], dans le repère local de l'élément.

---

## Types d'analyse

### `static` — Analyse statique linéaire

```json
"analysis": {"type": "static"}
```

**Résultats :** `u` (liste de longueur n_dof, déplacements en m).

Pour les modèles contenant des poutres (`Beam2D`, `Beam2DTimoshenko`, `Beam3D`),
l'option CLI `--diagrams [out.png]` trace les diagrammes d'efforts internes
M / V / N (et la torsion T en 3D). Les efforts sont reconstruits par équilibre,
de sorte qu'une charge répartie donne un moment **parabolique** et un effort
tranchant **linéaire** exacts :

```bash
python -m femsolver run examples/cantilever_distributed_beam.json --diagrams
```

### `modal` — Analyse modale

```json
"analysis": {
    "type": "modal",
    "n_modes": 5,
    "use_lumped": false
}
```

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `n_modes` | 5 | Nombre de modes à extraire |
| `use_lumped` | false | Masse condensée (true) ou consistante (false) |

**Résultats :** `freqs` [Hz], `modes` [shape: n_dof × n_modes].

### `buckling` — Flambage linéaire

```json
"analysis": {
    "type": "buckling",
    "n_modes": 1
}
```

Le chargement de référence est défini par `neumann` dans `boundary_conditions`.

**Résultats :** `lambda_cr` (multiplicateurs de charge critique), `phi` (modes de flambage).

La charge critique est `P_cr = lambda_cr[i] × P_ref`.

### `harmonic` — Balayage en fréquence

```json
"analysis": {
    "type": "harmonic",
    "freqs": {"linspace": [1.0, 100.0, 500]},
    "F_hat": [{"node": 5, "dof": 1, "value": 1000.0}],
    "damping": {"type": "rayleigh", "alpha": 0.0, "beta": 0.002}
}
```

**Résultats :** `freqs` [Hz], `amplitude` [m, shape: n_dof × n_freqs], `U_real`, `U_imag`.

### `transient` — Intégration temporelle Newmark-β

```json
"analysis": {
    "type": "transient",
    "dt": 1e-3,
    "n_steps": 1000,
    "F_hat": [{"node": 5, "dof": 1, "value": -5000.0}],
    "damping": {"type": "rayleigh", "alpha": 1.0, "beta": 0.002}
}
```

Conditions initiales : zéro (u₀ = 0, v₀ = 0). `F_hat = null` → vibration libre.

**Résultats :** `times` [s, longueur n_steps+1], `u` [m, shape: n_dof × (n_steps+1)].

### `random_force` — Réponse aléatoire à une excitation en force

```json
"analysis": {
    "type": "random_force",
    "freqs": {"linspace": [0.5, 200.0, 2000]},
    "G0": 1.0,
    "F_dir": [{"node": 5, "dof": 1, "value": 1.0}],
    "damping": {"type": "rayleigh", "alpha": 0.0, "beta": 0.005}
}
```

`G0` : niveau PSD bruit blanc [N²/Hz].
`F_dir` : direction de la force (vecteur normalisé).

**Résultats :** `rms_u` [m, longueur n_dof], `rms_a` [m/s², longueur n_dof], `freqs`.

### `random_base` — Réponse aléatoire à une excitation de base

```json
"analysis": {
    "type": "random_base",
    "freqs": {"linspace": [1.0, 200.0, 2000]},
    "G0": 0.01,
    "direction": 1,
    "damping": {"type": "rayleigh", "alpha": 0.0, "beta": 0.005}
}
```

`G0` : niveau PSD accélération sol [(m/s²)²/Hz].
`direction` : axe d'excitation (0=x, 1=y, 2=z).

**Résultats :** `rms_u` [m, déplacement relatif], `rms_a` [m/s², accélération absolue], `freqs`.

---

## Modèles d'amortissement

```json
"damping": {"type": "rayleigh", "alpha": 0.0, "beta": 0.002}
"damping": {"type": "hysteretic", "eta": 0.05}
```

| Type | Paramètres | Formule |
|------|-----------|---------|
| `rayleigh` | `alpha`, `beta` (défaut 0) | C = αM + βK |
| `hysteretic` | `eta` | K_eff = K(1 + iη) |

`null` ou absent → pas d'amortissement (C = 0).

---

## Spécification des fréquences

Trois formats sont acceptés pour les clés `freqs` :

```json
{"linspace": [fmin, fmax, n]}   // n points équidistants
{"logspace": [fmin, fmax, n]}   // n points en échelle log
[f1, f2, f3, ...]               // liste directe
```

---

## Exemples complets

### Treillis Warren (statique)

```json
{
  "name": "Warren Truss",
  "materials": {"steel": {"E": 210e9, "nu": 0.3, "rho": 7800}},
  "nodes": [
    [0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0],
    [0.5, 1.0], [1.5, 1.0], [2.5, 1.0], [3.5, 1.0]
  ],
  "elements": [
    {"type": "Bar2D", "nodes": [0, 1], "material": "steel", "area": 1e-4},
    ...
  ],
  "boundary_conditions": {
    "dirichlet": [
      {"node": 0, "dof": 0}, {"node": 0, "dof": 1}, {"node": 4, "dof": 1}
    ],
    "neumann": [{"node": 2, "dof": 1, "value": -10000.0}]
  },
  "analysis": {"type": "static"}
}
```

Voir `examples/warren_truss.json` pour l'exemple complet.

### Poutre console (statique)

Voir `examples/cantilever_beam.json`.

Solution analytique : δ = PL³/(3EI), θ = PL²/(2EI).

---

## Conventions DDL

| Élément | DDL par nœud | Signification |
|---------|-------------|---------------|
| Bar2D, continuum 2D | 2 | ux, uy |
| Beam2D, Beam2DTimoshenko | 3 | ux, uy, θz |
| Continuum 3D | 3 | ux, uy, uz |
| Beam3D | 6 | ux, uy, uz, θx, θy, θz |

L'indice global du DDL `d` du nœud `i` est `i × dpn + d`.

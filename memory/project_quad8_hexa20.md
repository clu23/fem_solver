---
name: Phase — Quad8 and Hexa20 serendipity elements
description: Quad8 (2D 8-node serendipity, 3×3 Gauss) and Hexa20 (3D 20-node serendipity, 3×3×3 Gauss) implemented with full test suite and convergence benchmark
type: project
---

Quad8 and Hexa20 serendipity elements implemented and tested.

**Files added:**
- `femsolver/elements/quad8.py` — Quad8 (16×16 K_e/M_e, 3×3 Gauss, batch)
- `femsolver/elements/hexa20.py` — Hexa20 (60×60 K_e/M_e, 3×3×3 Gauss, batch)
- `tests/test_quad8.py` — 28 tests including cantilever convergence vs Quad4/Tri6
- `tests/test_hexa20.py` — 23 tests including traction with consistent nodal forces

**Key implementation details:**
- Serendipity shape functions (no center node): corner formula N_i = ¼(1+ξᵢξ)(1+ηᵢη)(ξᵢξ+ηᵢη-1)
- Mid-side nodes use half-order formula: ½(1-ξ²)(1+ηᵢη) etc.
- 3×3 Gauss mandatory (2×2 causes rank deficiency / spurious hourglass modes)
- Consistent nodal forces for Quad8 face: F_corner = -P/12, F_midside = +P/3 (corners are negative!)
- Quad4 has severe shear locking (~38% error) on cantilever vs Quad8 (~1%)

**Why:** 3×3 Gauss because B has degree-2 terms → B^T D B is degree 4; rank argument: 2×2=4 pts gives rank≤12 < 13 needed for 16-DOF element.

How to apply: Hexa20 face forces need consistent distribution (not uniform); full clamping valid for ν=0 tests.

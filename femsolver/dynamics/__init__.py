"""Module dynamique — analyse modale, harmonique et transitoire."""

from femsolver.dynamics.modal import ModalResult, lumped_mass, run_modal
from femsolver.dynamics.rayleigh import (
    RayleighDamping,
    build_damping_matrix,
    rayleigh_from_modes,
)
from femsolver.dynamics.damping import HystereticDamping, ModalDampingModel
from femsolver.dynamics.harmonic import (
    HarmonicResult,
    run_harmonic,
    solve_harmonic,
    solve_harmonic_hysteretic,
    solve_harmonic_modal,
)
from femsolver.dynamics.transient import (
    NEWMARK_CENTRAL_DIFF,
    NEWMARK_FOX_GOODWIN,
    NEWMARK_LINEAR_ACCEL,
    NEWMARK_TRAPEZOIDAL,
    NewmarkBeta,
    TransientResult,
    run_transient,
    solve_newmark,
)

__all__ = [
    "ModalResult",
    "lumped_mass",
    "run_modal",
    "RayleighDamping",
    "build_damping_matrix",
    "rayleigh_from_modes",
    "HystereticDamping",
    "ModalDampingModel",
    "HarmonicResult",
    "run_harmonic",
    "solve_harmonic",
    "solve_harmonic_hysteretic",
    "solve_harmonic_modal",
    "NewmarkBeta",
    "NEWMARK_TRAPEZOIDAL",
    "NEWMARK_CENTRAL_DIFF",
    "NEWMARK_FOX_GOODWIN",
    "NEWMARK_LINEAR_ACCEL",
    "TransientResult",
    "solve_newmark",
    "run_transient",
]

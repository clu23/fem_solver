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
]

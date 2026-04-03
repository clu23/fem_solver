"""Noyau mathématique du solveur FEM."""

from femsolver.core.assembler import Assembler
from femsolver.core.boundary import apply_dirichlet, DirichletSystem
from femsolver.core.element import Element
from femsolver.core.material import ElasticMaterial
from femsolver.core.mesh import (
    BodyForce,
    BoundaryConditions,
    DistributedLineLoad,
    ElementData,
    Mesh,
    MPCConstraint,
    PressureLoad,
)
from femsolver.core.mpc import apply_mpc_elimination, apply_mpc_lagrange, recover_mpc
from femsolver.core.sections import (
    CircularSection,
    CSection,
    HollowCircularSection,
    HollowRectangularSection,
    ISection,
    LSection,
    RectangularSection,
    Section,
)
from femsolver.core.solver import ModalSolver, ScipyBackend, SolverBackend, StaticSolver

__all__ = [
    "Assembler",
    "apply_dirichlet",
    "DirichletSystem",
    "apply_mpc_elimination",
    "apply_mpc_lagrange",
    "recover_mpc",
    "Element",
    "ElasticMaterial",
    "BodyForce",
    "BoundaryConditions",
    "DistributedLineLoad",
    "ElementData",
    "Mesh",
    "MPCConstraint",
    "PressureLoad",
    "CircularSection",
    "CSection",
    "HollowCircularSection",
    "HollowRectangularSection",
    "ISection",
    "LSection",
    "RectangularSection",
    "Section",
    "ModalSolver",
    "ScipyBackend",
    "SolverBackend",
    "StaticSolver",
]

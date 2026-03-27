"""Noyau mathématique du solveur FEM."""

from femsolver.core.assembler import Assembler
from femsolver.core.boundary import apply_dirichlet
from femsolver.core.element import Element
from femsolver.core.material import ElasticMaterial
from femsolver.core.mesh import BoundaryConditions, ElementData, Mesh
from femsolver.core.solver import ModalSolver, ScipyBackend, SolverBackend, StaticSolver

__all__ = [
    "Assembler",
    "apply_dirichlet",
    "Element",
    "ElasticMaterial",
    "BoundaryConditions",
    "ElementData",
    "Mesh",
    "ModalSolver",
    "ScipyBackend",
    "SolverBackend",
    "StaticSolver",
]

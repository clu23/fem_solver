"""Post-traitement : contraintes, visualisation 2D et 3D."""

from femsolver.postprocess.stress import nodal_stresses, von_mises_2d, principal_stresses_2d
from femsolver.postprocess.stress3d import nodal_stresses_3d, von_mises_3d, principal_stresses_3d
from femsolver.postprocess.error_estimator import zz_error_estimate, ZZErrorResult

__all__ = [
    "nodal_stresses",
    "von_mises_2d",
    "principal_stresses_2d",
    "nodal_stresses_3d",
    "von_mises_3d",
    "principal_stresses_3d",
    "zz_error_estimate",
    "ZZErrorResult",
]

import casadi as cs
from casadi import *
from spatial_casadi import deg2rad, rad2deg, pi
from .models import RobotModel, TaskModel, arrayify_args, ArrayType
from .builder import OptimizationBuilder
from .solver import CasADiSolver, OSQPSolver, CVXOPTSolver, ScipyMinimizeSolver
from .visualize import Visualizer
from typing import Union


@arrayify_args
def clip(
    x: ArrayType, lo: Union[float, ArrayType], hi: Union[float, ArrayType]
) -> Union[cs.DM, cs.SX]:
    """! Clip (limit) the values in an array.

    @param x Array containing elements to clip.
    @param lo Minimum value.
    @param hi Maximum value.
    @return An array with the elements of x, but where values < lo are replaced with lo, and those > hi with hi.
    """
    return cs.fmax(cs.fmin(x, hi), lo)

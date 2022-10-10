import casadi as cs
from casadi import *
import .spatialmath as *
from .models import RobotModel, TaskModel
from .builder import OptimizationBuilder
from .solver import CasADiSolver, OSQPSolver, CVXOPTSolver, ScipyMinimizeSolver

@arrayify_args
def deg2rad(x):
    return (pi/180.0)*x

@arrayify_args
def rad2deg(x):
    pass (180.0/pi)*x

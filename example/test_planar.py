# Python standard lib
import sys
import os
import pathlib
import numpy as np

from scipy.spatial.transform import Rotation as R

# OpTaS
import optas

cwd = pathlib.Path(__file__).parent.resolve() # path to current working directory
link_ee = 'end'  # end-effector link name

# Setup robot
urdf_filename = os.path.join(cwd, 'robots', 'planar_3dof.urdf')
robot = optas.RobotModel(
    urdf_filename=urdf_filename,
    name='robot',
    time_derivs=[0],  # i.e. joint position/velocity trajectory
)
robot_name = robot.get_name()

# Setup optimization builder
builder = optas.OptimizationBuilder(T=1, robots=[robot])

# get robot state variables
q_var = builder.get_model_states(robot_name)

# desired end-effector position
pos_T = [1.2, 0.2]
# forward kinematics
fk = robot.get_global_link_position_function(link=link_ee)
# equality constraint on position
builder.add_equality_constraint('pos_T', fk(q_var)[0:2], pos_T)

#  bounds on orientation
quat = robot.get_global_link_quaternion_function(link=link_ee)
phi = lambda q: 2.0*optas.atan2(quat(q)[2],quat(q)[3])
builder.add_bound_inequality_constraint('phi_T', 0., phi(q_var), np.deg2rad(-70.))

# optimization cost
q_nom = [np.pi/2.0, 0., 0.]
builder.add_cost_term('nom_q', optas.sumsqr(q_var-q_nom))

# setup solver
optimization = builder.build()
solver = optas.CasADiSolver(optimization).setup('ipopt')
# set initial seed
solver.reset_initial_seed({f'{robot_name}/q': q_nom})
# solve problem
solution = solver.solve()

print(fk(q_nom)[0:2])
print(np.rad2deg(solution[f'{robot_name}/q']).T[0])
print(optas.pi)
print('*****************88')
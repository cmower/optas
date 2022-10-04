# Python standard lib
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
# q_nom = optas.DM.zeros(robot.ndof)
q_nom = np.asarray([0., np.pi/2.0, 0.])

# forward kinematics
fk_pos = robot.get_global_link_position_function(link=link_ee)
fk_ori = robot.get_global_link_quaternion_function(link=link_ee)
print(q_nom)
print(fk_pos(q_nom))
print(fk_ori(q_nom))
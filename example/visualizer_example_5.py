import os
import sys
import pathlib

import optas
from optas.visualize import Visualizer

cwd = pathlib.Path(__file__).parent.resolve()  # path to current working directory

if "lbr" in sys.argv:
    model = "lbr"
elif "lwr" in sys.argv:
    model = "lwr"
else:
    model = "lwr"

if model == "lwr":
    urdf_filename = os.path.join(cwd, "robots", "kuka_lwr", "kuka_lwr.urdf")
    robot_model = optas.RobotModel(urdf_filename=urdf_filename)

elif model == "lbr":
    xacro_filename = os.path.join(cwd, "robots", "kuka_lbr", "med7.urdf.xacro")
    robot_model = optas.RobotModel(xacro_filename=xacro_filename)

n = 100
Q = optas.np.zeros((robot_model.ndof, n))
qS = robot_model.get_random_joint_positions().toarray().flatten()
qG = robot_model.get_random_joint_positions().toarray().flatten()
for i in range(n):
    alpha = float(i) / float(n - 1)
    Q[:, i] = (1 - alpha) * qS + alpha * qG

vis = Visualizer(camera_position=[3, 3, 3])
vis.grid_floor()
vis.robot_traj(robot_model, Q, animate=True, duration=4.0)
vis.start()

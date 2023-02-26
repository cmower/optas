import os
import sys
import pathlib

import optas
from optas.visualize import Visualizer, robot, grid_floor

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

vis = Visualizer(camera_position=[3, 3, 3])
vis.append_actors(
    grid_floor(),
    robot(robot_model, q=robot_model.get_random_joint_positions()),
)
vis.start()

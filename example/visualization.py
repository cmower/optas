import os
import pathlib
import optas

cwd = pathlib.Path(__file__).parent.resolve() # path to current working directory
urdf_filename = os.path.join(cwd, 'robots', 'kuka_lwr', 'kuka_lwr.urdf')
robot = optas.RobotModel(urdf_filename=urdf_filename)

q = optas.deg2rad([0, 45, 0, -90, 0, -45, 0])
params = {
    'alpha': 0.5,
    'show_link_names': True,
    'show_links': True,
    'link_axis_scale': 0.3,
}
optas.RobotVisualizer(robot, q=q, params=params).start()

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
vis = optas.RobotVisualizer(robot, q=q, params=params)
T = optas.np.array([
    [1, 0, 0, -1],
    [0, 1, 0, -1],
    [0, 0, 1, 0.25],
    [0, 0, 0, 1],
])
vis.draw_box(0.5, 0.5, 0.5, rgba=[0, 1, 0, 1], T=T)
vis.draw_sphere(0.25, rgba=[1, 0, 0, 1], position=[0, -1, 0.25])
vis.start()


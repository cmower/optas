import os
import sys
import pathlib
import optas

cwd = pathlib.Path(__file__).parent.resolve()  # path to current working directory

if "lbr" in sys.argv:
    model = "lbr"
elif "lwr" in sys.argv:
    model = "lwr"
else:
    model = "lwr"

if model == "lwr":
    urdf_filename = os.path.join(cwd, "robots", "kuka_lwr", "kuka_lwr.urdf")
    robot = optas.RobotModel(urdf_filename=urdf_filename)

elif model == "lbr":
    xacro_filename = os.path.join(cwd, "robots", "kuka_lbr", "med7.urdf.xacro")
    robot = optas.RobotModel(xacro_filename=xacro_filename)

q = optas.deg2rad([0, 45, 0, -90, 0, -45, 0])
params = {
    "alpha": 0.5,
    "show_link_names": True,
    "show_links": True,
    "link_axis_scale": 0.3,
}
vis = optas.RobotVisualizer(robot, q=q, params=params)
T = optas.np.array(
    [
        [1, 0, 0, -1],
        [0, 1, 0, -1],
        [0, 0, 1, 0.25],
        [0, 0, 0, 1],
    ]
)
T_cyl = optas.np.array(
    [
        [1, 0, 0, 1],
        [0, 1, 0, -1],
        [0, 0, 1, 0.25],
        [0, 0, 0, 1],
    ]
)
vis.draw_box(0.5, 0.5, 0.5, rgba=[0, 1, 0, 1], T=T)
vis.draw_sphere(0.25, rgba=[1, 0, 0, 1], position=[0, -1, 0.25])
vis.draw_cylinder(0.25, 0.5, rgba=[0, 0, 1, 1], T=T_cyl)
vis.start()

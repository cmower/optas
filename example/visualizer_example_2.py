from optas.visualize import Visualizer

vis = Visualizer(camera_position=[8, 8, 8])
vis.box(position=[0, 2, -1], orientation=[0, 0, 20], euler_degrees=True, rgb=[0, 1, 1])
vis.cylinder(radius=0.1),
vis.cylinder_urdf(
    position=[0, 0, 1],
    # orientation=[0, 90, 0],
    euler_degrees=True,
    radius=0.4,
    height=1.0,
    alpha=0.8,
    rgb=[1, 0, 0],
)
vis.link(axis_scale=2.0, axis_linewidth=1.0)
vis.obj(
    "duck.obj",
    png_texture_filename="duckCM.png",
    position=[2, 0, 0],
    orientation=[90, 0, -90],
    euler_degrees=True,
)
vis.start()

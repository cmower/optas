from optas.visualize import Visualizer, sphere, grid_floor, link, text

vis = Visualizer()
vis.append_actors(
    text(vis.camera, position=[0.0, 0.0, 0.5], scale=[0.01, 0.01, 0.01]),
    grid_floor(stride=0.1, euler=[0, 45, 0]),
    sphere(position=[0, 0, 0], radius=0.3, alpha=0.8, rgb=[1, 0, 0]),
    link(axis_scale=1.0),
)
vis.start()

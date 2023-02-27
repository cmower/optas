import numpy as np
from optas.visualize import Visualizer

n = 10
traj = np.zeros((3, n))
start = np.random.uniform(-1, 1, size=(3,))
end = np.random.uniform(-1, 1, size=(3,))
for i in range(n):
    alpha = float(i) / float(n - 1)
    traj[:, i] = (1 - alpha) * start + alpha * end

vis = Visualizer()
vis.text(position=[0.0, 0.0, 0.5], scale=[0.01, 0.01, 0.01])
vis.grid_floor(stride=0.1, euler=[0, 45, 0])
vis.sphere(position=[0, 0, 0], radius=0.3, alpha=0.8, rgb=[1, 0, 0])
vis.link(axis_scale=1.0)
vis.sphere_traj(traj, radius=0.1, rgb=[0, 0, 1])
vis.start()

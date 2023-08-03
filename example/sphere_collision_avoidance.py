import os
import pathlib

import optas
from optas.visualize import Visualizer

# Setup robot model
cwd = pathlib.Path(__file__).parent.resolve()  # path to current working directory
urdf_filename = os.path.join(cwd, "robots", "kuka_lwr", "kuka_lwr.urdf")
robot_model = optas.RobotModel(urdf_filename=urdf_filename, time_derivs=[0, 1, 2])
name = robot_model.get_name()
qnom = optas.deg2rad([0, -45, 0, 90, 0, 45, 0])
ee_link = "end_effector_ball"

T0 = robot_model.get_global_link_transform(ee_link, qnom)
z0 = T0[:3, 2]


def compute_initial_configuration():
    # Compute an intiail configuration
    start_eff_position = optas.DM([0.825, -0.35, 0.2])
    builder = optas.OptimizationBuilder(1, robots=robot_model, derivs_align=True)
    q = builder.get_model_state(name, 0)
    T = robot_model.get_global_link_transform(ee_link, q)
    p = T[:3, 3]
    z = T[:3, 2]

    builder.enforce_model_limits(name)

    builder.add_equality_constraint("eff_pos", p, start_eff_position)
    builder.add_equality_constraint("eff_ori", z, z0)

    builder.initial_configuration(name, time_deriv=1)
    builder.initial_configuration(name, time_deriv=2)

    builder.add_cost_term("nominal", optas.sumsqr(q - qnom))

    solver = optas.CasADiSolver(builder.build()).setup("ipopt")
    solver.reset_initial_seed({f"{name}/q": qnom})
    solution = solver.solve()

    return solution[f"{name}/q"]


obs = optas.DM.zeros(3, 6)
obs[0, :] = 0.55
obs[1, :] = 0.0
obs[2, :] = optas.DM([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).T

obsrad = 0.1 * optas.DM.ones(6)

linkrad = 0.15

q0 = compute_initial_configuration()

duration = 10.

T = 20
builder = optas.OptimizationBuilder(T, robots=robot_model, derivs_align=True)

dt = duration / float(T - 1)

builder.enforce_model_limits(name)
builder.integrate_model_states(name, 2, dt)  # acc->vel
builder.integrate_model_states(name, 1, dt)  # vel->pos

builder.initial_configuration(name, init=q0)
builder.initial_configuration(name, time_deriv=1)
builder.initial_configuration(name, time_deriv=2)

goal_eff_position = optas.DM([0.825, 0.35, 0.2])
qf = builder.get_model_state(name, t=-1)
Tf = robot_model.get_global_link_transform(ee_link, qf)
pf = Tf[:3, 3]
zf = Tf[:3, 2]

builder.add_equality_constraint(
    "eff_pos", pf, goal_eff_position, reduce_constraint=True
)
builder.add_equality_constraint("eff_ori", zf, z0, reduce_constraint=True)

builder.add_cost_term(
    "min_vel", optas.sumsqr(builder.get_model_states(name, time_deriv=1))
)
builder.add_cost_term(
    "min_acc", 100*optas.sumsqr(builder.get_model_states(name, time_deriv=2))
)

builder.add_cost_term("nominal", 1e2 * optas.sumsqr(qf - qnom))

obstacle_names = [f"obs{i}" for i in range(6)]


link_names = ['end_effector_ball', 'lwr_arm_7_link', 'lwr_arm_5_link', 'lwr_arm_6_link']

builder.sphere_collision_constraints(name, obstacle_names, link_names=link_names)

solver = optas.CasADiSolver(builder.build()).setup("ipopt")

params = {}
for link_name in link_names:
    params[link_name + "_radii"] = linkrad

for i, obstacle_name in enumerate(obstacle_names):
    params[obstacle_name + "_position"] = obs[:, i]
    params[obstacle_name + "_radii"] = obsrad[i]

solver.reset_parameters(params)

print("Solver ready")
solution = solver.solve()

Q = solution[f"{name}/q"]

vis = Visualizer()
vis.robot_traj(robot_model, Q, animate=True, duration=duration)
vis.robot(robot_model, q0, alpha=0.5)
vis.grid_floor()
# vis.link(optas.DM.eye(4), axis_scale=1, axis_linewidth=8)
for i in range(6):
    vis.sphere(radius=obsrad[i], position=obs[:, i], rgb=[1.0, 0.0, 0.0])

for link_name in link_names:
    postraj = optas.DM.zeros(3, T)
    for t in range(T):
        postraj[:, t] = robot_model.get_global_link_position(link_name, Q[:, t])

    vis.sphere_traj(
        radius=linkrad,
        position_traj=postraj,
        rgb=[1, 0.64705882352, 0.0],
        alpha_spec={'style': 'const', 'alpha': 0.25},
        duration=duration,
        animate=True,
    )
    
vis.start()

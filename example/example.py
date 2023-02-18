import os
import pathlib

import optas

# Specify URDF filename
cwd = pathlib.Path(__file__).parent.resolve()  # path to current working directory
urdf_filename = os.path.join(
    cwd, "robots", "kuka_lwr", "kuka_lwr.urdf"
)  # KUKA LWR, 7-DoF

# Setup robot model
robot = optas.RobotModel(urdf_filename=urdf_filename)
name = robot.get_name()

# Setup optimization builder
T = 1
builder = optas.OptimizationBuilder(T, robots=robot)

# Setup parameters
qn = builder.add_parameter("q_nominal", robot.ndof)
pg = builder.add_parameter("p_goal", 3)

# Constraint: end goal
q = builder.get_model_state(name, 0)
end_effector_name = "end_effector_ball"
p = robot.get_global_link_position(end_effector_name, q)
builder.add_equality_constraint("end_goal", p, pg)

# Cost: nominal configuration
builder.add_cost_term("nominal", optas.sumsqr(q - qn))

# Constraint: joint position limits
builder.enforce_model_limits(name)  # joint limits extracted from URDF

# Build optimization problem
optimization = builder.build()

# Interface optimization problem with a solver
solver = optas.CasADiSolver(optimization).setup("ipopt")
# solver = optas.ScipyMinimizeSolver(optimization).setup("SLSQP")

# Specify a nominal configuration
q_nominal = optas.deg2rad([0, 45, 0, -90, 0, -45, 0])

# Get end-effector position in nominal configuration
p_nominal = robot.get_global_link_position(end_effector_name, q_nominal)

# Specify a goal end-effector position
p_goal = p_nominal + optas.DM([0.0, 0.3, -0.2])

# Reset solver parameters
solver.reset_parameters({"q_nominal": q_nominal, "p_goal": p_goal})

# Reset initial seed
solver.reset_initial_seed({f"{name}/q": q_nominal})

# Compute a solution
solution = solver.solve()
q_solution = solution[f"{name}/q"]

# Visualize the robot
params = {"link_axis_scale": 0.5}
# vis = optas.RobotVisualizer(robot, q=q_nominal, params={})  # nominal
vis = optas.RobotVisualizer(robot, q=q_solution, params=params)  # solution

# Draw goal position and start visualizer
vis.draw_sphere(0.05, rgba=[0, 1, 0, 0.5], position=p_goal.toarray().flatten().tolist())
vis.start()

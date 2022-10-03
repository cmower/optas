# Python standard lib
import os
import pathlib
import numpy as np

# OpTaS
import optas

cwd = pathlib.Path(__file__).parent.resolve() # path to current working directory
link_ee = 'end_effector_ball'  # end-effector link name

# Setup robot
urdf_filename = os.path.join(cwd, 'robots', 'kuka_lwr.urdf')
robot = optas.RobotModel(
    urdf_filename=urdf_filename,
    time_derivs=[0],  # i.e. joint position/velocity trajectory
)
robot_name = robot.get_name()
print(robot_name)

# Setup optimization builder
builder = optas.OptimizationBuilder(T=1, robots=[robot])

# get robot state variables
q_var = builder.get_model_states(robot_name)

# desired end-effector position
pos_T = optas.DM([-0.5, 0.0, 0.5])#np.asarray([-0.5, 0.0, 0.5])

# equality constraint on position
pos_fnc = robot.get_global_link_position_function(link=link_ee)
p = pos_fnc(q_var)
print(f"{q_var.shape=}")
print(f"{p.shape=}")
print(f"{pos_T.shape=}")
builder.add_equality_constraint('final_pos', p, pos_T)
# builder.add_cost_term('final_pos', 1e4*optas.sumsqr(p - pos_T))

# optimization cost: close to nominal config
q_nom = optas.DM.zeros(robot.ndof)
builder.add_cost_term('nom_config', 0.001*optas.sumsqr(q_var-q_nom))

# setup solver
optimization = builder.build()
print(f"{optimization.nv=}")
print(f"{optimization.lbv=}")
print(f"{optimization.ubv=}")
solver = optas.CasADiSolver(optimization=optimization).setup('ipopt')
print(f"{optimization.lbg=}")
print(f"{optimization.ubg=}")
print(f"{optimization.lbh=}")
print(f"{optimization.ubh=}")
print(f"{solver._lbg=}")
print(f"{solver._ubg=}")
# solver = optas.ScipyMinimizeSolver(optimization).setup('COBYLA')
# set initial seed
solver.reset_initial_seed({f'{robot_name}/q': q_nom})
# solve problem
solution = solver.solve()

qT_array = solution[f'{robot_name}/q']
print('********************************')
print("p(qSol)=", pos_fnc(qT_array))
print(f'{solver.opt_type=}')
print(f"{q_nom=}")
print(f"{qT_array=}")
print('********************************')

# Example of an Inverse Kinematic (IK) solver applied to a 3 dof planar robot

# Python standard lib
import os
import pathlib

# OpTaS
import optas

cwd = pathlib.Path(__file__).parent.resolve() # path to current working directory

urdf_filename = os.path.join(cwd, 'robots', 'planar_3dof.urdf')
# Setup robot
robot = optas.RobotModel(urdf_filename)
robot_name = robot.get_name(); link_ee = 'end'  # end-effector link name

# Setup optimization builder
builder = optas.OptimizationBuilder(T=1, robots=[robot])

# get robot state variables
q_T = builder.get_model_states(robot_name)

x_T = [1.2, 0.2] # target end-effector position
q_0 = [optas.pi/2.0, 0., 0.] # nominal config
lim = 160.*optas.pi
q_min = [0., -lim, -lim]
q_max = [optas.pi, lim, lim]

# forward kinematics
fk = robot.get_global_link_position_function(link=link_ee)
quat = robot.get_global_link_quaternion_function(link=link_ee)
phi = lambda q: 2.0*optas.atan2(quat(q)[2],quat(q)[3])

# Setting optimization - cost term and constraints
builder.add_cost_term('cost', optas.sumsqr(q_T-q_0))

builder.add_equality_constraint('FK', fk(q_T)[0:2], x_T)

builder.add_bound_inequality_constraint('joint', q_min, q_T, q_max)

#  bounds on orientation
builder.add_bound_inequality_constraint('task', -70.*(optas.pi/180.), phi(q_T), 0.)

# setup solver
# solver = optas.CasADiSolver(builder.build()).setup('ipopt')
# solver = optas.ScipyMinimizeSolver(builder.build()).setup('SLSQP')
solver = optas.ScipyMinimizeSolver(builder.build()).setup('COBYLA')
# set initial seed
solver.reset_initial_seed({f'{robot_name}/q': q_0})
# solve problem
solution = solver.solve()
print(solution[f'{robot_name}/q']*(180./optas.pi))
# Example of an Inverse Differential Kinematic (IK) solver
# written as an Quadratic Program (QP) applied to a 3 dof planar robot

# Python standard lib
import sys
import os
import pathlib

# OpTaS
import optas

cwd = pathlib.Path(__file__).parent.resolve() # path to current working directory

urdf_filename = os.path.join(cwd, 'robots', 'planar_3dof.urdf')
# Setup robot
robot = optas.RobotModel(urdf_filename, time_derivs=[1])
robot_name = robot.get_name(); link_ee = 'end'  # end-effector link name

# Setup optimization builder
builder = optas.OptimizationBuilder(T=1, robots=[robot], derivs_align=True)

# get robot state variables
dq = builder.get_model_states(robot_name, time_deriv=1)
q = builder.add_parameter('q', robot.ndof)

# Forward Differential Kinematics
J = robot.get_global_linear_jacobian_function(link_ee)
quat = robot.get_global_link_quaternion_function(link_ee)
# End-effector orientation
phi = lambda q: 2.0*optas.atan2(quat(q)[2],quat(q)[3])
# Jacobian of the end-effector orientation
J_phi = robot.get_global_angular_geometric_jacobian_function(link=link_ee)

q_t = [2.39, -2.55, -0.46]
dx = [0.01, 0.] # target end-effector displacement/velocity
dt = 0.01; lim = 0.1
dq_min = [-lim, -lim, -lim]
dq_max = [lim, lim, lim]

# Setting optimization - cost term and constraints
builder.add_cost_term('cost', optas.sumsqr(dq))

builder.add_equality_constraint('FDK', (J(q)[0:2,:])@dq, dx)

builder.add_bound_inequality_constraint('joint', dq_min, dq, dq_max)

#  bounds on orientation
builder.add_bound_inequality_constraint('task', -70*(optas.pi/180.), phi(q)+dt*(J_phi(q)[2,:])@dq, 0.)

# setup solver
optimization = builder.build()
# solver = optas.CasADiSolver(optimization).setup('qpoases')
solver = optas.CVXOPTSolver(optimization).setup()
# set initial seed
solver.reset_initial_seed({f'{robot_name}/dq': [0., 0., 0.]})
solver.reset_parameters({'q': q_t})
# solve problem
solution = solver.solve()
print(solution[f'{robot_name}/dq'])

# solution using the pseudo-inverse approach
# it should give the same result as the QP for the case that the
# solution is completely inside all the boundary constraints
print(optas.pinv(J(q_t)[0:2,:])@dx)

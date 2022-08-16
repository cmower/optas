import pyinvk
pyinvk.cs.np.set_printoptions(precision=3, suppress=True)

"""


qd*  =  arg min  || J(qc) qd - vg ||^2 + lambd*||qd||^2
           qd

qd*: computed joint velocity
qd: decision variable (joint velocity)
qc: current joint state
J: linear geometric jacobian
vg: linear velocity goal
||.||^2: squared Euclidean norm
lambd: scalar weighting term


"""

def main():

    # Constants
    T = 1  # time steps
    lambd = 0.1  # weighting term
    time_derivs=[1]  # time derivatives required in the robot model
    urdf_filename = 'med7.urdf'  # filename for the robot urdf
    eff_link_name = 'lbr_link_ee'  # link name for the end-effector
    derivs_align = True  # time derivatives align, False will throw an error

    # Create robot model
    robot = pyinvk.RobotModel(urdf_filename, time_derivs=time_derivs)

    # Create optimization builder
    builder = pyinvk.OptimizationBuilder(T, robots= {'kuka': robot}, derivs_align=derivs_align)

    # Add parameters to problem
    qcurr = builder.add_parameter('qcurr', robot.ndof)  # current joint state
    vg = builder.add_parameter('vg', 3)  # linear velocity goal

    # Get joint states
    qd = builder.get_joint_state('kuka', 0, time_deriv=1)

    # Cost: match end-effector velocity
    J = robot.get_linear_geometric_jacobian(eff_link_name, qcurr)
    builder.add_cost_term('match_eff_vel', pyinvk.cs.sumsqr(J@qd - vg))

    # Cost: minimize joint velocity
    builder.add_cost_term('min_qd', lambd*pyinvk.cs.sumsqr(qd))

    # Build optimization and setup solver
    solver = pyinvk.OSQPSolver(builder.build()).setup()

    # Reset problem
    solver.reset_parameters({'qcurr': robot.get_random_joint_positions(), 'vg': [0.05, 0.025, 0.02]})

    # Solve problem
    solution = solver.solve()
    print("solution =", solution['kuka/dq'].toarray().flatten())

if __name__ == '__main__':
    main()
    
    

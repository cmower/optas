import pathlib
import pyinvk
pyinvk.cs.np.set_printoptions(precision=3, suppress=True)

"""


qd*  =  arg min  || J(qc) qd - vg ||^2 + lambd*||qd||^2
           qd

         q-     <=    qc+dt*qd <= q+
         -qdlim <=          qd <= qdlim
          plim- <= F(qc+dt+qd) <= plim+

qd*: computed joint velocity
qd: decision variable (joint velocity)
qc: current joint state
J: linear geometric jacobian
vg: linear velocity goal
||.||^2: squared Euclidean norm
lambd: scalar weighting term
q-,q+: lower/upper joint position limits
qdlim: joint velocity limits
dt: time horizon
F: forward kinematics, position of end-effector
plim-,plim+: lower/upper position bounds for end-effector

"""

def main():

    # Constants
    dt = 0.1  # time horizon
    T = 1  # time steps
    lambd = 0.1  # weighting term
    time_derivs=[1]  # time derivatives required in the robot model
    urdf_filename = str(pathlib.Path(__file__).parent.absolute())+'/robots/med7.urdf'  # filename for the robot urdf
    eff_link_name = 'lbr_link_ee'  # link name for the end-effector
    derivs_align = True  # time derivatives align, False will throw an error
    plim_minus = [-0.5]*3  # lower position bound for end-effector
    plim_plus = [0.5]*3 # upper position bound for end-effector

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

    # Constraint: joint position limits
    qminus = robot.lower_actuated_joint_limits
    qplus = robot.upper_actuated_joint_limits
    qnext = qcurr+dt*qd
    builder.add_ineq_bound_constraint('joint_pos_lim', qminus, qnext, qplus)

    # Constraint: joint velocity limits
    qdlim = robot.velocity_actuated_joint_limits
    builder.add_ineq_bound_constraint('joint_vel_lim', -qdlim, qd, qdlim)

    # Constraint: end-effector position bound
    pnext = robot.get_global_link_position(eff_link_name, qnext)
    builder.add_ineq_bound_constraint('eff_pos_bnd', plim_minus, pnext, plim_plus)

    # Build optimization and setup solver
    solver = pyinvk.CasADiSolver(builder.build()).setup('ipopt')

    # Reset problem
    solver.reset_parameters({'qcurr': robot.get_random_joint_positions(), 'vg': pyinvk.cs.np.random.rand(3)})

    # Solve problem
    solution = solver.solve()
    print("solution =", solution['kuka/dq'].toarray().flatten())


if __name__ == '__main__':
    main()

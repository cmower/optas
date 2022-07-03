import time
import casadi as cs
from pyinvk.robot_model import RobotModel
from pyinvk.builder import OptimizationBuilder
from pyinvk.solver import OSQPSolver, CasADiSolver, CVXOPTSolver
from pyinvk.ros import RosNode

def setup_casadi_solver(optimization):
    return CasADiSolver(optimization).setup('qpoases')

def setup_osqp_solver(optimization):
    return OSQPSolver(optimization).setup()

def setup_cvxqp_solver(optimization):
    return CVXOPTSolver(optimization).setup()

def main():

    # Setup PyInvK
    urdf_filename = 'kuka_lwr.urdf'
    robot = RobotModel(urdf_filename)

    robots = {'kuka_lwr': robot}  # multiple robots can be defined, see ex2.py
    builder = OptimizationBuilder(robots=robots, T=2, qderivs=[1])
    qcurr = builder.add_parameter('qcurr', robot.ndof)
    fk = robot.fk('baselink', 'lwr_arm_7_link')
    pos_jac = fk['pos_jac']
    vel_goal = builder.add_parameter('vel_goal', 3)
    J = pos_jac(qcurr)
    dq = builder.get_qstate('kuka_lwr', 0, qderiv=1)
    builder.add_cost_term('goal', cs.sumsqr(J@dq - vel_goal))
    builder.add_cost_term('min_vel', cs.sumsqr(dq))
    optimization = builder.build()

    # solver = setup_casadi_solver(optimization)
    # solver = setup_osqp_solver(optimization)
    solver = setup_cvxqp_solver(optimization)

    # Setup ROS
    node = RosNode(robots, 'pyinvk_ex_unconstrained_qp_node')
    qcurr = node.wait_for_joint_state('kuka_lwr')

    # Reset problem
    vel_goal = [0.0, -0.05, -0.1]
    solver.reset_parameters({
        'vel_goal': vel_goal,
        'qcurr': qcurr,
    })

    # Solve problem
    for i in range(30):
        t0 = time.time()
        solution = solver.solve()
        t1 = time.time()
        ms = 1000*(t1-t0)
        s_ = 1.0/(t1-t0)
        print("Solver took", ms, f"milli-secs, 1/s={s_:.5f}")

        # Compute next joint state
        dt = 0.1
        qnext = qcurr + dt*solution['kuka_lwr/dq']

        node.move_robot_to_joint_state('kuka_lwr', qnext, dt)

        # print("Solution:")
        # print(solution)

        solver.reset_parameters({
            'vel_goal': vel_goal,
            'qcurr': qnext,
        })
        qcurr = qnext
        time.sleep(dt)

    print(f"{type(optimization)=}")        

    print("Goodbye")

if __name__ == '__main__':
    main()

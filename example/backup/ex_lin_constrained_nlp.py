import time
import casadi as cs
from pyinvk.robot_model import RobotModel
from pyinvk.builder import OptimizationBuilder
from pyinvk.solver import CasADiSolver
from pyinvk.ros import RosNode

def main():

    # Setup PyInvK
    urdf_filename = 'kuka_lwr.urdf'
    robot = RobotModel(urdf_filename)

    robots = {'kuka_lwr': robot}  # multiple robots can be defined, see ex2.py
    builder = OptimizationBuilder(robots=robots, T=1, qderivs=[0])
    qnext = builder.get_qstate('kuka_lwr', 0)
    fk = robot.fk('baselink', 'lwr_arm_7_link')
    pos = fk['pos']
    pos_goal = builder.add_parameter('pos_goal', 3)
    builder.add_cost_term('goal', cs.sumsqr(pos(qnext) - pos_goal))

    builder.add_ineq_constraint(
        'pos_lim_lo',
        robot.lower_actuated_joint_limits,
        qnext,
    )

    builder.add_ineq_constraint(
        'pos_lim_hi',
        qnext,
        robot.upper_actuated_joint_limits,
    )

    optimization = builder.build()

    solver = CasADiSolver(optimization).setup('ipopt')

    # Setup ROS
    node = RosNode(robots, 'pyinvk_ex_lin_constrained_nlp_node')
    qcurr = node.wait_for_joint_state('kuka_lwr')

    # Reset problem
    pos_goal = [0.5, -0.5, 0.5]
    solver.reset_parameters({
        'pos_goal': pos_goal,
    })

    # Solve problem
    t0 = time.time()
    solution = solver.solve()
    t1 = time.time()
    ms = 1000*(t1-t0)
    s_ = 1.0/(t1-t0)
    print("Solver took", ms, f"milli-secs, 1/s={s_:.5f}")

    qnext=solution['kuka_lwr/q']

    dt = 0.5
    node.move_robot_to_joint_state('kuka_lwr', qnext, dt)

    print("Goodbye")

if __name__ == '__main__':
    main()

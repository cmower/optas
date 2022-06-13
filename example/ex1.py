import sys
import casadi as cs
from pyinvk.robot_model import RobotModel
from pyinvk.builder import OptimizationBuilder
from pyinvk.solver import CasadiNLPSolver, ScipyMinimizeSolver
from pyinvk.common import RosNode

"""

In this example, we demonstrate the libraries ability to interface
with ROS and the ROS-PyBullet Interface - these are requirements to
run this example. 

Before running this example, in another terminal execute the
following.

$ roslaunch rpbi_examples run_kuka.launch

"""

def main():

    print("Starting pyinvk example 1")

    # Setup/build optimization problem
    urdf_filename = 'kuka_lwr.urdf'
    robot = RobotModel(urdf_filename)
    robots = {'kuka_lwr': robot}
    builder = OptimizationBuilder(robots)
    qnext = builder.get_state('kuka_lwr', 0)
    fk = robot.fk('baselink', 'lwr_arm_7_link')
    pos = fk['pos']
    pos_goal = builder.add_parameter('pos_goal', 3)
    builder.add_cost_term('eff_pos_goal', cs.sumsqr(pos(qnext) - pos_goal))

    qcurr = builder.add_parameter('qcurr', robot.ndof)
    builder.add_cost_term('nominal', 0.1*cs.sumsqr(qnext - qcurr))
    
    lbc = robot.lower_actuated_joint_limits
    ubc = robot.upper_actuated_joint_limits
    builder.add_ineq_constraint('joint_position_limits', qnext, lbc=lbc, ubc=ubc)
    optimization = builder.build()

    # Setup solver
    use_scipy = True
    if not use_scipy:
        solver = CasadiNLPSolver(optimization)
        solver.setup('ipopt')
    else:
        solver = ScipyMinimizeSolver(optimization)
        solver.setup(
            # method='Nelder-Mead',
            method='SLSQP',
            # method='trust-constr',
        )

    # Setup ROS
    node = RosNode(robots, 'pyinvk_ex1_node')
    x0 = node.wait_for_joint_state('kuka_lwr')

    # Get eff position goal from command line
    def get_arg(idx, default):
        try:
            out = float(sys.argv[idx])
        except IndexError:
            out = default
        return out
    xg = get_arg(1, 0.4)
    yg = get_arg(2, 0.4)
    zg = get_arg(3, 0.4)

    # Solve problem
    solver.reset_initial_seed(x0)
    solver.reset_parameters({'pos_goal': [xg, yg, zg], 'qcurr': x0})
    solution = solver.solve()

    # Move robot to solution
    node.move_robot_to_joint_state('kuka_lwr', solution['kuka_lwr/q'], 1.0)

    print("Solution:")
    print(solution['kuka_lwr/q'])

    print("Goodbye")

if __name__ == '__main__':
    main()

import sys
import casadi as cs
from pyinvk.robot_model import RobotModel
from pyinvk.builder import OptimizationBuilder
from pyinvk.solver import CasADiSolver, ScipyMinimizeSolver
from pyinvk.ros import RosNode

"""

In this example, we demonstrate the libraries ability to interface
with ROS and the ROS-PyBullet Interface - these are requirements to
run this example. Furthermore, the example demonstrates how to
setup/solve problems with more than one robot.

Before running this example, in another terminal execute the
following.

$ roslaunch rpbi_examples run_dual_kukas.launch

"""

def main():

    print("Starting pyinvk example 2")

    # Setup/build optimization problem
    urdf_filename = 'kuka_lwr.urdf'
    robot0 = RobotModel(urdf_filename, base_xyz=[0, 1, 0])
    robot1 = RobotModel(urdf_filename)
    robots = {'kuka_lwr_0': robot0, 'kuka_lwr_1': robot1}
    builder = OptimizationBuilder(robots)

    qnext0 = builder.get_state('kuka_lwr_0', 0)
    fk0 = robot0.fk('baselink', 'lwr_arm_7_link')
    pos0 = fk0['pos']
    pos0_goal = builder.add_parameter('pos0_goal', 3)
    builder.add_cost_term('eff_pos_goal0', cs.sumsqr(pos0(qnext0) - pos0_goal))
    lbc0 = robot0.lower_actuated_joint_limits
    ubc0 = robot0.upper_actuated_joint_limits
    builder.add_ineq_constraint('joint_position_limits0_lo', lbc0, qnext0)
    builder.add_ineq_constraint('joint_position_limits0_hi', qnext0, ubc0)

    qnext1 = builder.get_state('kuka_lwr_1', 0)
    fk1 = robot1.fk('baselink', 'lwr_arm_7_link')
    pos1 = fk1['pos']
    pos1_goal = builder.add_parameter('pos1_goal', 3)
    builder.add_cost_term('eff_pos_goal1', cs.sumsqr(pos1(qnext1) - pos1_goal))
    lbc1 = robot1.lower_actuated_joint_limits
    ubc1 = robot1.upper_actuated_joint_limits
    builder.add_ineq_constraint('joint_position_limits1_lo', lbc1, qnext1)
    builder.add_ineq_constraint('joint_position_limits1_hi', qnext1, ubc1)

    optimization = builder.build()

    # Setup solver
    use_scipy = False
    if not use_scipy:
        solver = CasADiSolver(optimization)
        solver.setup('ipopt')
    else:
        solver = ScipyMinimizeSolver(optimization)
        solver.setup(
            # method='Nelder-Mead',
            method='SLSQP',
            # method='trust-constr',
        )

    # Setup ROS
    node = RosNode(robots, 'pyinvk_ex2_node')
    x0_0 = node.wait_for_joint_state('kuka_lwr_0')
    x0_1 = node.wait_for_joint_state('kuka_lwr_1')

    # Solve problem
    solver.reset_initial_seed({'kuka_lwr_0/q': x0_0, 'kuka_lwr_1/q': x0_1})
    solver.reset_parameters({'pos0_goal': [0.5, 0.5, 0.5], 'pos1_goal': [0.5, 0.5, 0.1]})
    solution = solver.solve()

    # Move robot to solution
    node.move_robot_to_joint_state('kuka_lwr_0', solution['kuka_lwr_0/q'], 1.0)
    node.move_robot_to_joint_state('kuka_lwr_1', solution['kuka_lwr_1/q'], 1.0)

    print("Solution:")
    print(solution['kuka_lwr_0/q'])
    print(solution['kuka_lwr_1/q'])

    print("Goodbye")

if __name__ == '__main__':
    main()

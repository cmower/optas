import sys
import rospy
import casadi as cs
import numpy as np
from math import radians
from pprint import pprint
from sensor_msgs.msg import JointState
from ros_pybullet_interface.srv import ResetJointState
from custom_ros_tools.ros_comm import get_srv_handler
from custom_ros_tools.robot import resolve_joint_order

from pyinvk.builder import OptimizationBuilder
from pyinvk.robot_model import RobotModel
from pyinvk.solver import CasadiSolver, ScipySolver

def main():

    # Setup ros
    rospy.init_node('test_pyinvk')
    pub = rospy.Publisher('rpbi/kuka_lwr/joint_states/target', JointState, queue_size=1)

    # Setup constants
    urdf = 'kuka_lwr.urdf'
    tip = 'lwr_arm_7_link'
    root = 'base'
    qnom = cs.DM([0, radians(30), 0, -radians(90), 0, radians(60), 0])
    N = int(sys.argv[4])

    # Setup robot model and optimization builder
    robot_model = RobotModel(urdf, root, tip)
    builder = OptimizationBuilder(robot_model, N)

    # Get required casadi variables/parameters
    qcurr = builder.add_parameter('qcurr', robot_model.ndof)
    qstart = builder.get_q(0)
    qfinal = builder.get_q(N-1)
    eff_pos = robot_model.get_end_effector_position(qfinal)
    eff_goal = builder.add_parameter('eff_goal', 3)

    # Setup cost function
    cost_term1 = cs.sumsqr(eff_pos - eff_goal)
    builder.add_cost_term('eff_goal', cost_term1)

    if N > 1:
        cost_term2 = 0.0
        for k in range(N-1):
            qc = builder.get_q(k)
            qn = builder.get_q(k+1)
            cost_term2 += cs.sumsqr(qn - qc)
        builder.add_cost_term('min_dist', cost_term2)

    # Setup constriants
    builder.enforce_joint_limits()
    if N > 1:
        builder.add_eq_constraint('start_state', qcurr - qstart)

    if sys.argv[3] == 'snopt':
        # qstart[0] +big_number >=  0
        builder.add_ineq_constraint('dummy_constraint_for_snopt', qstart[0] + 1e5)

    optimization = builder.build()

    if sys.argv[2] == 'scipy':
        solver = ScipySolver(optimization)
        solver.setup(sys.argv[3], options={'disp': True})

    elif sys.argv[2] == 'casadi':
        solver = CasadiSolver(optimization)
        solver.setup(sys.argv[3])

    init_seed = cs.DM.ones(robot_model.ndof, N)
    for i in range(N):
        init_seed[:, i] = qnom

    params = {
        'eff_goal': np.array([-0.5, float(sys.argv[1])*0.4, 0.5]),
        'qcurr': cs.DM(resolve_joint_order(rospy.wait_for_message('rpbi/kuka_lwr/joint_states', JointState), robot_model.joint_names).position),
    }    

    solver.reset(init_seed, params)
    
    solution = solver.solve()

    jointmsgs = solver.solution_to_ros_joint_state_msgs(solution)

    rate = rospy.Rate(10)
    for i in range(N):
        js = jointmsgs[i]
        js.header.stamp = rospy.Time.now()
        pub.publish(js)
        print("Published", i+1, "of", N)
        rate.sleep()
        
    print("cost=", optimization.cost(cs.vec(solution), optimization.parameters.dict2vec(params)))


if __name__ == '__main__':
    main()

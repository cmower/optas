import rospy
import casadi as cs
import numpy as np
from math import radians
from pprint import pprint
from ros_pybullet_interface.srv import ResetJointState
from custom_ros_tools.ros_comm import get_srv_handler
from pyinvk.builder import SolverBuilder

def main():

    # Setup ros
    rospy.init_node('test_pyinvk')
    srv = get_srv_handler('/rpbi/kuka_lwr/move_to_joint_state', ResetJointState)

    # Setup
    urdf = 'kuka_lwr.urdf'
    tip = 'lwr_arm_7_link'
    root = 'base'
    qnom = cs.DM([0, radians(30), 0, -radians(90), 0, radians(60), 0])
    N = 20

    # Build solver
    builder = SolverBuilder(urdf, root, tip, N)

    q = builder.get_q() # symbolic q
    eff_pos = builder.get_end_effector_position()  # function of q

    eff_goal = builder.add_parameter('eff_goal', 3)

    cost_term1 = cs.sumsqr(eff_pos - eff_goal)
    builder.add_cost_term(cost_term1)

    cost_term2 = 0.1*cs.sumsqr(q - qnom)/7.
    builder.add_cost_term(cost_term2)


    for i in range(N-1):
        old_eff = builder.get_end_effector_position(i)
        new_eff = builder.get_end_effector_position(i+1)
        builder.add_cost_term(cs.sumsqr(old_eff-new_eff))

    builder.enforce_joint_limits()
    builder.enforce_start_state(qnom)

    solver = builder.build_solver('ipopt')

    # Use solver
    goal = np.array([-0.5, -0.5, 0.5])

    solver.set_parameter('eff_goal', goal)

    init_seed = cs.DM.ones(builder.ndof, N)
    for i in range(N):
        init_seed[:, i] = qnom
    solver.set_initial_seed(init_seed)

    solver.reset()
    solution = solver.solve()

    pprint(solver.stats())

    pprint(solution)

    for i in range(N):
        js = solver.solution2msg(solution, i)
        srv(js, 0.01)
        print("Finished", i+1, "of", N)

    qsol = cs.reshape(solution['x'], builder.ndof, N)
    print("cost = ", solver.cost(qsol, solver.params))


if __name__ == '__main__':
    main()

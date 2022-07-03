import rospy
import numpy as np
import casadi as cs
from pyinvk.robot_model import RobotModel
from pyinvk.builder import OptimizationBuilder
from pyinvk.solver import CasADiSolver
from custom_ros_tools.tf import TfInterface
from pyinvk.ros import RosNode

"""

This is a re-implementation for the controller in

M. Rubagotti, T. Taunyazov, B. Omarali and A. Shintemirov,
"Semi-Autonomous Robot Teleoperation With Obstacle Avoidance via Model
Predictive Control," in IEEE Robotics and Automation Letters, vol. 4,
no. 3, pp. 2746-2753, July 2019, doi: 10.1109/LRA.2019.2917707.

"""

class HumanInterface:

    def __init__(self, hz, T, dt):
        self.T = T
        self.dt = dt
        self.tf = TfInterface()
        self.prev_human_pose = None
        self.human_pose = np.concatenate(self.tf.msg_to_pos_quat(self.tf.wait_for_tf_msg('rpbi/kuka_lwr/base', 'teleop_target')))
        self.timer_dt = 1.0/float(hz)
        rospy.Timer(rospy.Duration(self.timer_dt), self.update_human_pose)

    def _get_human_pose(self):
        tf = self.tf.get_tf_msg('rpbi/kuka_lwr/base', 'teleop_target')
        return np.concatenate(self.tf.msg_to_pos_quat(tf))

    def update_human_pose(self, event):
        self.prev_human_pose = self.human_pose.copy()
        self.human_pose = self._get_human_pose()

    def get_human_pose(self):

        prev_human_pose = self.prev_human_pose.copy()
        human_pose = self.human_pose.copy()
        lin_vel = (human_pose[:3] - prev_human_pose[:3])/self.timer_dt

        out = np.zeros((7, self.T))
        out[:3, 0] = human_pose[:3]
        for k in range(self.T-1):
            out[:3,k+1] = out[:3,k] + self.dt*lin_vel
            out[3:,k] = human_pose[3:]
        out[3:,-1] = human_pose[3:]

        # return out[:3,:]
        return out

def main():

    # Setup PyInvK
    urdf_filename = 'kuka_lwr.urdf'
    robot = RobotModel(urdf_filename)
    T = 2 # num time steps
    dt = 0.1  # time step
    hz = int(1.0/dt)
    solver_method = 'ipopt'

    # Create mpc
    robots = {'kuka_lwr': robot}  # multiple robots can be defined, see ex2.py
    builder = OptimizationBuilder(robots, T=T, qderivs=[0, 1, 2], derivs_align=True)

    # Dynamic constraints
    builder.add_dynamic_integr_constraints('kuka_lwr', 2, dt)  # integr: qdd -> qd
    builder.add_dynamic_integr_constraints('kuka_lwr', 1, dt)  # integr: qd -> q

    # Initial configuration
    qcurr = builder.add_parameter('qcurr', robot.ndof)
    q0 = builder.get_state('kuka_lwr', 0)
    builder.add_eq_constraint('initial_config', q0, qcurr)

    # Joint limits
    builder.add_joint_position_limit_constraints('kuka_lwr')

    # Forward kinematics
    fk = robot.fk('baselink', 'lwr_arm_7_link')

    # Singularity avoidance
    pos_jac = fk['pos_jac']
    eul_jac = fk['eul_jac']
    wmin = builder.add_parameter('omega_min')
    w = cs.SX.zeros(T)
    for t in range(T):
        q = builder.get_state('kuka_lwr', t)
        # J = cs.vertcat(pos_jac(q), eul_jac(q))
        J = pos_jac(q)
        w[t] = cs.sqrt(cs.det(J@J.T))
    builder.add_ineq_constraint('sing_avoid', wmin, w)

    # Human hand pose prediction and min joint speed

    # >> pos/quat <<
    pos_quat = fk['pos_quat'].map(T)
    pose_human = builder.add_parameter('pose_human', 7, T)  # prediction
    pose_robot = pos_quat(builder.get_states('kuka_lwr'))
    xdiff = pose_robot - pose_human
    y = cs.vertcat(xdiff, builder.get_states('kuka_lwr', 1))
    Q = builder.add_parameter('Q', 7+robot.ndof)
    builder.add_cost_term('min_y', cs.trace(y.T@cs.diag(Q)@y))

    # >> pos << 
    # pos = fk['pos'].map(T)
    # pose_human = builder.add_parameter('pose_human', 3, T)  # prediction
    # pose_robot = pos(builder.get_states('kuka_lwr'))
    # xdiff = pose_robot - pose_human
    # y = cs.vertcat(xdiff, builder.get_states('kuka_lwr', 1))
    # Q = builder.add_parameter('Q', 3+robot.ndof)
    # builder.add_cost_term('min_y', cs.trace(y.T@cs.diag(Q)@y))    

    # Minimize controls
    R = builder.add_parameter('R', robot.ndof)
    u = builder.get_states('kuka_lwr', 2)
    builder.add_cost_term('min_u', cs.trace(u.T@cs.diag(R)@u))

    # Obstacle avoidance
    pos = fk['pos'].map(T)
    zpos = pos(builder.get_states('kuka_lwr'))[2,:]
    builder.add_ineq_constraint('floor', zpos)

    # Build optimization
    print("\nOptimization summary:\n")
    builder.print_desc()
    optimization = builder.build()

    # Setup solver
    solver = CasADiSolver(optimization).setup(solver_method)

    # Setup ros
    node = RosNode(robots, 'mpc_for_shared_autonomy_node')
    human_interface = HumanInterface(50, T, dt)
    node.setup_rpbi_joint_state_subscriber('kuka_lwr')
    node.setup_rpbi_joint_state_target_publisher('kuka_lwr')

    qcurr = node.wait_for_joint_state('kuka_lwr')

    states = {
        'kuka_lwr/q': cs.diag(qcurr) @ cs.DM.ones(7, T),
    }

    parameters = {
        'qcurr': None,
        'omega_min': 0.01,
        'pose_human': None,
        'Q': [10.0]*3 + [1.]*4 + [0.001]*robot.ndof,
        'R': [0.001]*robot.ndof,
    }

    rospy.sleep(1.0)
    rate = rospy.Rate(hz)
    while not rospy.is_shutdown():
        parameters['qcurr'] = node.get_joint_state_from_msg('kuka_lwr')
        parameters['pose_human'] = human_interface.get_human_pose()
        solver.reset_initial_seed(states)
        solver.reset_parameters(parameters)
        states = solver.solve()
        node.publish_target_joint_state_msg('kuka_lwr', states['kuka_lwr/q'][:,1])
        rate.sleep()

if __name__ == '__main__':
    main()

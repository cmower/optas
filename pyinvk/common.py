import casadi as cs
try:
    import rospy
    from sensor_msgs.msg import JointState
    from custom_ros_tools.ros_comm import get_srv_handler
    from custom_ros_tools.tf import TfInterface
    from custom_ros_tools.robot import resolve_joint_order
    from ros_pybullet_interface.srv import ResetJointState
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False

class RosNode:

    def __init__(self, robots, node_name, anonymous=False):
        assert ROS_AVAILABLE, f"{name} requires ROS"
        rospy.init_node(node_name, anonymous=anonymous)
        self.tf = TfInterface()
        self.robots = robots
        self.pubs = {}
        self.subs = {}
        self.msgs = {}
        self._rate = None

    def move_robot_to_joint_state(self, robot_name, q, duration):
        self._assert_robot(robot_name)
        handler = get_srv_handler(f'rpbi/{robot_name}/move_to_joint_state', ResetJointState)
        position = cs.DM(q).toarray().flatten().tolist()
        msg = JointState(name=self.robots[robot_name].actuated_joint_names, position=position)
        handler(msg, duration)

    def _assert_robot(self, robot_name):
        assert robot_name in self.robots, f"did not recognize robot with name '{robot_name}'"

    def setup_rpbi_joint_state_subscriber(self, robot_name):
        self._assert_robot(robot_name)
        topic = f'rpbi/{robot_name}/joint_states'
        self.subs[robot_name + '_joint_states'] = rospy.Subscriber(topic, JointState, self._joint_state_callback,  callback_args=robot_name)

    def _joint_state_callback(self, msg, robot_name):
        self.msgs[robot_name + '_joint_state'] = msg

    def get_joint_state(self, robot_name):
        self._assert_robot(robot_name)
        robot_model = self.robots[robot_name]
        msg = resolve_joint_order(
            self.msgs.get(robot_name + '_joint_state'),
            robot_model.actuated_joint_names
        )
        return cs.DM(msg.position)

    def wait_for_joint_state(self, robot_name, timeout=None):
        self._assert_robot(robot_name)
        topic = f'rpbi/{robot_name}/joint_states'
        robot_model = self.robots[robot_name]
        msg = resolve_joint_order(
            rospy.wait_for_message(topic, JointState, timeout=timeout),
            robot_model.actuated_joint_names
        )
        return cs.DM(msg.position)

    def setup_rpbi_joint_state_target_publisher(self, robot_name, queue_size=10):
        self._assert_robot(robot_name)
        topic = f'rpbi/{robot_name}/joint_states/target'
        self.pubs[robot_name + '_joint_state_target'] = rospy.Publisher(
            topic, JointState, queue_size=queue_size
        )

    def pack_joint_state_msg(self, robot_name, position, velocity=None, effort=None):
        self._assert_robot(robot_name)
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = self.robots[robot_name].actuated_joint_names
        msg.position = cs.DM(q).toarray().flatten().tolist()
        if velocity is not None:
            msg.velocity = velocity
        if effort is not None:
            msg.effort = effort
        return msg

    def publish_target_joint_state_msg(self, robot_name, position, velocity=None, effort=None):
        self._assert_robot(robot_name)
        msg = self.pack_joint_state_msg(robot_name, position, velocity=velocity, effort=effort)
        self.pubs[robot_name + '_joint_state_target'].publish(msg)

    def init_rate(self, hz):
        self._rate = rospy.Rate(hz)

    def sleep(self):
        assert self._rate is not None, "you must call init_rate"
        self._rate.sleep()

    def spin(self):
        rospy.spin()

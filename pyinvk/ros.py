import casadi as cs
from typing import Dict
from .robot_model import RobotModel
try:
    import rospy
    from sensor_msgs.msg import JointState
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False

try:
    from custom_ros_tools.ros_comm import get_srv_handler
    from custom_ros_tools.tf import TfInterface
    from custom_ros_tools.robot import resolve_joint_order
    CUSTOM_ROS_TOOLS_AVAILABLE = True
except:
    CUSTOM_ROS_TOOLS_AVAILABLE = False

try:
    from ros_pybullet_interface.srv import ResetJointState
    ROS_PYBULLET_INTERFACE_AVAILABLE = True
except:
    ROS_PYBULLET_INTERFACE_AVAILABLE = False

class RosNode:

    def __init__(
            self,
            robots: Dict[str, RobotModel],
            node_name: str,
            anonymous: bool=False):
        """Constructor for the RosNode class.

        Note, if ROS is not installed then an AssertionError is thrown.

        Parameters
        ----------

        robots : dict[str:pyinvk.robot_model.RobotModel]
            A dictionary containing the robot models in the
            scene. Each model should be indexed by a unique name for
            the robot. If you are interfacing with the ROS-PyBullet
            Interface then the robot name should be set to the same
            name as is given in the interface configuration file.

        """
        assert ROS_AVAILABLE, f"{node_name} requires ROS and the ROS-PyBullet Interface"
        rospy.init_node(node_name, anonymous=anonymous)
        rospy.loginfo(f'initialized ROS node: {node_name}')

        self.tf = None
        if CUSTOM_ROS_TOOLS_AVAILABLE:
            self.tf = TfInterface()
        else:
            rospy.logwarn(f'custom_ros_tools is not installed, did not instantiate TfInterface')

        # Setup class attributes
        self.robots = robots
        self.pubs = {}
        self.subs = {}
        self.msgs = {}

        rospy.loginfo('Initialized RosNode object.')

    def move_robot_to_joint_state(self, robot_name, q, duration):
        self._assert_robot(robot_name)
        assert ROS_PYBULLET_INTERFACE_AVAILABLE, "this method requires the ROS-PyBullet Interface"
        handler = get_srv_handler(f'rpbi/{robot_name}/move_to_joint_state', ResetJointState)
        position = cs.DM(q).toarray().flatten().tolist()
        msg = JointState(name=self.robots[robot_name].actuated_joint_names, position=position)
        handler(msg, duration)
        rospy.loginfo(f'Moved robot to new configuration: {position}')

    def _assert_robot(self, robot_name):
        assert robot_name in self.robots, f"did not recognize robot with name '{robot_name}'"

    def setup_rpbi_joint_state_subscriber(self, robot_name):
        self._assert_robot(robot_name)
        assert ROS_PYBULLET_INTERFACE_AVAILABLE, "this method rquires the ROS-PyBullet Interface"
        topic = f'rpbi/{robot_name}/joint_states'
        self.start_subscriber(topic, JointState)

    def start_subscriber(self, topic, topic_type, callback=None):
        if callback is None:
            callback = self._callback
        self.subs[topic] = rospy.Subscriber(topic, topic_type, callback, callback_args=topic)
        rospy.loginfo('Started subscriber for the topic {topic}.')

    def _callback(self, msg, callback_arg):
        self.msgs[callback_arg] = msg

    def get_joint_state_from_msg(self, robot_name):
        self._assert_robot(robot_name)
        assert ROS_PYBULLET_INTERFACE_AVAILABLE, "this method rquires the ROS-PyBullet Interface"
        robot_model = self.robots[robot_name]
        topic = f'rpbi/{robot_name}/joint_states'
        joint_state = self.msgs.get(topic)
        msg = resolve_joint_order(
            joint_state,
            robot_model.actuated_joint_names
        )
        return cs.DM(msg.position)

    def wait_for_joint_state(self, robot_name, timeout=None):
        self._assert_robot(robot_name)
        assert ROS_PYBULLET_INTERFACE_AVAILABLE, "this method rquires the ROS-PyBullet Interface"
        topic = f'rpbi/{robot_name}/joint_states'
        robot_model = self.robots[robot_name]
        msg = resolve_joint_order(
            rospy.wait_for_message(topic, JointState, timeout=timeout),
            robot_model.actuated_joint_names
        )
        return cs.DM(msg.position)

    def setup_rpbi_joint_state_target_publisher(self, robot_name, queue_size=10):
        self._assert_robot(robot_name)
        assert ROS_PYBULLET_INTERFACE_AVAILABLE, "this method rquires the ROS-PyBullet Interface"
        topic = f'rpbi/{robot_name}/joint_states/target'
        self.pubs[topic] = rospy.Publisher(
            topic, JointState, queue_size=queue_size
        )
        rospy.loginfo(f'Setup publisher for topic: {topic}')

    def pack_joint_state_msg(self, robot_name, position, velocity=None, effort=None):
        self._assert_robot(robot_name)
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = self.robots[robot_name].actuated_joint_names
        msg.position = cs.DM(position).toarray().flatten().tolist()
        if velocity is not None:
            msg.velocity = cs.DM(velocity).toarray().flatten().tolist()
        if effort is not None:
            msg.effort = cs.DM(effort).toarray().flatten().tolist()
        return msg

    def publish_target_joint_state_msg(self, robot_name, position, velocity=None, effort=None):
        self._assert_robot(robot_name)
        assert ROS_PYBULLET_INTERFACE_AVAILABLE, "this method rquires the ROS-PyBullet Interface"
        msg = self.pack_joint_state_msg(robot_name, position, velocity=velocity, effort=effort)
        topic = f'rpbi/{robot_name}/joint_states/target'
        self.pubs[topic].publish(msg)

    def spin(self):
        rospy.loginfo('Started spinnning...')
        rospy.spin()

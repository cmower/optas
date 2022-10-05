
try:
    import rospy
    from sensor_msgs.msg import JointState
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False

try:
    from custom_ros_tools.ros_comm import get_srv_handler
    from custom_ros_tools.tf import TfInterface
    from custom_ros_tools.robot import JointStatePublisher, JointStateSubscriber

    CUSTOM_ROS_TOOLS_AVAILABLE = True
except:
    CUSTOM_ROS_TOOLS_AVAILABLE = False

try:
    from ros_pybullet_interface.srv import ResetJointState
    ROS_PYBULLET_INTERFACE_AVAILABLE = True
except:
    ROS_PYBULLET_INTERFACE_AVAILABLE = False

class RosNode:

    """Class for communicating with ROS."""

    def __init__(self, robots, node_name='optas_node'):
        """RosNode constructor

        Parameters
        ----------

        robots (optas.RobotModel)
            List of robot models.

        node_name (string)
            Name of the ROS node.

        """

        assert ROS_AVAILABLE, "RosNode requires ROS"
        rospy.init_node(node_name)

        self.tf = None
        if CUSTOM_ROS_TOOLS_AVAILABLE:
            self.tf = TfInterface()
        else:
            rospy.logwarn(f'custom_ros_tools is not installed, did not instantiate TfInterface in RosNode')

        self.robots = robots
        self.pubs = {}
        self.subs = {}

    def move_robot_to_joint_state(self, robot_name, q, duration):
        """Move the robot in the ROS-PyBullet Interface to a given joint state."""
        assert ROS_PYBULLET_INTERFACE_AVAILABLE, "this method requires the ROS-PyBullet Interface"
        handler = get_srv_handler(f'rpbi/{robot_name}/move_to_joint_state', ResetJointState)
        position = cs.DM(q).toarray().flatten().tolist()
        js = JointState(name=self.robots[robot_name].actuated_joint_names, position=position)
        handler(js, duration)

    def start_joint_state_subscriber(self, robot_name, topic, callback):
        """Start a subscriber to a joint state topic."""
        robot = self.robots[robot_name]
        self.subs[robot_name] = JointStateSubscriber(topic, robot.actuated_joint_names, callback=callback)

    def start_rpbi_joint_state_subscriber(self, robot_name, callback=None):
        """Start a subscriber to a joint state topic for a robot that is loaded in the ROS-PyBullet Interface."""
        assert ROS_PYBULLET_INTERFACE_AVAILABLE, "this method requires the ROS-PyBullet Interface"
        topic = f'rpbi/{robot_name}/joint_states'
        self.start_joint_state_subscriber(robot_name, topic, callback=callback)

    def get_joint_position(self, robot_name):
        """Get the joint position for a given robot."""
        return cs.DM(self.subs[robot_name].get_position())

    def get_joint_velocity(self, robot_name):
        """Get the joint velocity for a given robot."""        
        return cs.DM(self.subs[robot_name].get_velocity())

    def setup_joint_state_publisher(self, robot_name, topic, queue_size=10):
        """Setup a joint state publisher for a given robot."""
        robot = self.robots[robot_name]
        self.pubs[robot_name] = JointStatePublisher(topic, robot.actuated_joint_names, queue_size=queue_size)

    def setup_rpbi_joint_state_publisher(self, robot_name, queue_size=10):
        """Setup a joint state publisher for a given robot loaded in the ROS-PyBullet Interface."""        
        assert ROS_PYBULLET_INTERFACE_AVAILABLE, "this method requires the ROS-PyBullet Interface"
        topic = f'rpbi/{robot_name}/joint_states/target'
        self.setup_joint_state_publisher(robot_name, topic, queue_size=10)

    def publish_joint_state(self, robot_name, q=None, qd=None):
        """Publisher a joint state for a given robot."""
        q = cs.vec(q).toarray().flatten().tolist()
        if qd is not None:
            qd = cs.vec(qd).toarray().flatten().tolist()
        self.pubs[robot_name].publish(q=q, qd=qd)

    def spin(self):
        """Wrapper method for the ROS spin method."""
        rospy.spin()

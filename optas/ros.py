
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

    def __init__(self, robots, node_name='optas_node', anonymous=False):
        
        assert ROS_AVAILABLE, "RosNode requires ROS"
        rospy.init_node(node_name, anonymous=anonymous)
        
        self.tf = None
        if CUSTOM_ROS_TOOLS_AVAILABLE:
            self.tf = TfInterface()
        else:
            rospy.logwarn(f'custom_ros_tools is not installed, did not instantiate TfInterface in RosNode')

        self.robots = robots
        self.pubs = {}
        self.subs = {}

    def move_robot_to_joint_state(self, robot_name, q, duration):
        assert ROS_PYBULLET_INTERFACE_AVAILABLE, "this method requires the ROS-PyBullet Interface"
        handler = get_srv_handler(f'rpbi/{robot_name}/move_to_joint_state', ResetJointState)
        position = cs.DM(q).toarray().flatten().tolist()
        js = JointState(name=self.robots[robot_name].actuated_joint_names, position=position)
        handler(js, duration)

    def start_joint_state_subscriber(self, robot_name, topic, callback):
        robot = self.robots[robot_name]
        self.subs[robot_name] = JointStateSubscriber(topic, robot.actuated_joint_names, callback=callback)
        
    def start_rpbi_joint_state_subscriber(self, robot_name, callback=None):
        assert ROS_PYBULLET_INTERFACE_AVAILABLE, "this method requires the ROS-PyBullet Interface"
        topic = f'rpbi/{robot_name}/joint_states'
        self.start_joint_state_subscriber(robot_name, topic, callback=callback)        

    def get_joint_position(self, robot_name):
        return cs.DM(self.subs[robot_name].get_position())

    def get_joint_velocity(self, robot_name):
        return cs.DM(self.subs[robot_name].get_velocity())

    def setup_joint_state_publisher(self, robot_name, topic, queue_size=10):
        robot = self.robots[robot_name]
        self.pubs[robot_name] = JointStatePublisher(topic, robot.actuated_joint_names, queue_size=queue_size)

    def setup_rpbi_joint_state_publisher(self, robot_name, queue_size=10):
        assert ROS_PYBULLET_INTERFACE_AVAILABLE, "this method requires the ROS-PyBullet Interface"
        topic = f'rpbi/{robot_name}/joint_states/target'
        self.setup_joint_state_publisher(robot_name, topic, queue_size=10)

    def publish_joint_state(self, robot_name, q=None, qd=None):
        q = cs.vec(q).toarray().flatten().tolist()
        if qd is not None:
            qd = cs.vec(qd).toarray().flatten().tolist()
        self.pubs[robot_name].publish(q=q, qd=qd)

    def spin(self):
        rospy.spin()

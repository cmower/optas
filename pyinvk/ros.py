import casadi as cs
import numpy as np
from typing import Dict, List, Union, Callable, Optional, Callable
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

        node_name : str
            The name for the ROS node.

        anonymous : bool (default is False)
            The anonymous keyword argument is mainly used for nodes
            where you normally expect many of them to be running and
            don't care about their names (e.g. tools, GUIs). It adds a
            random number to the end of your node's name, to make it
            unique. Unique names are more important for nodes like
            drivers, where it is an error if more than one is
            running. If two nodes with the same name are detected on a
            ROS graph, the older node is shutdown.

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

    def move_robot_to_joint_state(
            self,
            robot_name: str,
            q: Union[cs.casadi.DM, List, np.ndarray],
            duration: float) -> None:
        """Moves a robot in the ROS-PyBullet Interface to a given joint state.

        Parameters
        ----------

        robot_name : str
            The name of the robot. This must match the name of the
            robot in the ROS-PyBullet Interface and be an index in the
            robots keys given in to the constructor of the RosNode
            class.

        q : Union[cs.casadi.DM, List, np.ndarray]
            Goal joint state configuration.

        duration : float
            The duration in seconds it should take to move the robot
            from it's current configuration to the given goal
            configuration.

        """
        self._assert_robot(robot_name)
        assert ROS_PYBULLET_INTERFACE_AVAILABLE, "this method requires the ROS-PyBullet Interface"
        handler = get_srv_handler(f'rpbi/{robot_name}/move_to_joint_state', ResetJointState)
        position = cs.DM(q).toarray().flatten().tolist()
        msg = JointState(name=self.robots[robot_name].actuated_joint_names, position=position)
        handler(msg, duration)
        rospy.loginfo(f'Moved robot to new configuration: {position}')

    def _assert_robot(self, robot_name: str) -> None:
        """Private method for ensuring the robot_name exists in the given robots models."""
        assert robot_name in self.robots, f"did not recognize robot with name '{robot_name}'"

    def setup_rpbi_joint_state_subscriber(self, robot_name: str, callback: Optional[Callable]=None) -> None:
        """Starts a subscriber for the joint states for a given robot.

        The callback simply logs the recieved messages in the msgs
        class attribute. Use the get_joint_state_from_msg method to
        retrieve messages.

        Parameters
        ----------

        robot_name : str
            The name of the robot. This must match the name of the
            robot in the ROS-PyBullet Interface and be an index in the
            robots keys given in to the constructor of the RosNode
            class.

        callback : Callable (default is None)
            Callback method, if this is None then the callback used
            simply logs recieved messages in the msgs class attribute
            indexed by the topic name. However, it is suggested that
            you use the get_joint_state_from_msg method to retrieve
            messages when callback=None.

        """
        self._assert_robot(robot_name)
        assert ROS_PYBULLET_INTERFACE_AVAILABLE, "this method requires the ROS-PyBullet Interface"
        topic = f'rpbi/{robot_name}/joint_states'
        self.start_subscriber(topic, JointState, callback=callback)

    def start_subscriber(self, topic: str, topic_type: type, callback: Optional[Callable]=None) -> None:
        """Starts a subscriber.

        Parameters
        ----------

        topic : str
            The topic to subscribe.

        topic_type : type
            The type of the topic.

        callback : Callable (default is None)
            Callback method, if this is None then the callback used
            simply logs recieved messages in the msgs class attribute
            indexed by the topic name.

        """
        if callback is None:
            callback = self._callback
        self.subs[topic] = rospy.Subscriber(topic, topic_type, callback, callback_args=topic)
        rospy.loginfo('Started subscriber for the topic {topic}.')

    def _callback(self, msg, callback_arg):
        """Private callback method used by the ROS subscribers."""
        self.msgs[callback_arg] = msg

    def get_joint_state_from_msg(self, robot_name: str) -> cs.casadi.DM:
        """Return the joint state as a casadi.casadi.DM variable.

        Parameters
        ----------

        robot_name : str
            The name of the robot. This must match the name of the
            robot in the ROS-PyBullet Interface and be an index in the
            robots keys given in to the constructor of the RosNode
            class.

        Returns
        -------

        joint_state : casadi.casadi.DM
            The most recent robot joint state configuration. The order
            of the joints are resolved with respect to the
            actuated_joint_names attribute for the corresponding robot
            model.

        """
        self._assert_robot(robot_name)
        assert ROS_PYBULLET_INTERFACE_AVAILABLE, "this method requires the ROS-PyBullet Interface"
        robot_model = self.robots[robot_name]
        topic = f'rpbi/{robot_name}/joint_states'
        joint_state = self.msgs.get(topic)
        if joint_state:
            msg = resolve_joint_order(
                joint_state,
                robot_model.actuated_joint_names
            )
            return cs.DM(msg.position)

    def wait_for_joint_state(self, robot_name: str, timeout: Optional[float]=None) -> cs.casadi.DM:
        """Receive one message from JointState topic for a given robot in the ROS-PyBullet Interface.

        Parameters
        ----------

        robot_name : str
            The name of the robot. This must match the name of the
            robot in the ROS-PyBullet Interface and be an index in the
            robots keys given in to the constructor of the RosNode
            class.

        timeout : float (default is None)
            Timeout time in seconds.

        """
        self._assert_robot(robot_name)
        assert ROS_PYBULLET_INTERFACE_AVAILABLE, "this method requires the ROS-PyBullet Interface"
        topic = f'rpbi/{robot_name}/joint_states'
        robot_model = self.robots[robot_name]
        msg = resolve_joint_order(
            rospy.wait_for_message(topic, JointState, timeout=timeout),
            robot_model.actuated_joint_names
        )
        return cs.DM(msg.position)

    def setup_rpbi_joint_state_target_publisher(self, robot_name: str, queue_size: Optional[int]=10) -> rospy.Publisher:
        """Instantiate a joint state target publisher.

        This method requires the ROS-PyBullet Interface.

        Parameters
        ----------

        robot_name : str
            The name of the robot. This must match the name of the
            robot in the ROS-PyBullet Interface and be an index in the
            robots keys given in to the constructor of the RosNode
            class.

        queue_size : int (default is 10)
            The queue size used for asynchronously publishing messages
            from different threads. A size of zero means an infinite
            queue, which can be dangerous.

        Returns
        -------

        pub : rospy.Publisher
            The created publisher.

        """
        self._assert_robot(robot_name)
        assert ROS_PYBULLET_INTERFACE_AVAILABLE, "this method requires the ROS-PyBullet Interface"
        topic = f'rpbi/{robot_name}/joint_states/target'
        pub = rospy.Publisher(topic, JointState, queue_size=queue_size)
        self.pubs[topic] = pub
        rospy.loginfo(f'Setup publisher for topic: {topic}')
        return pub

    def pack_joint_state_msg(
            self, robot_name: str,
            position: Union[cs.casadi.DM, List, np.ndarray],
            velocity: Optional[Union[cs.casadi.DM, List, np.ndarray]]=None,
            effort: Optional[Union[cs.casadi.DM, List, np.ndarray]]=None) -> JointState:
        """Packs a joint state message.

        The order of the joint state is assumed to be in the same
        order as given in the robot model.

        Parameters
        ----------

        robot_name : str
            The name of the robot. This must match the name of the
            robot in the ROS-PyBullet Interface and be an index in the
            robots keys given in to the constructor of the RosNode
            class.

        position : Union[cs.casadi.DM, List, np.ndarray]
            Joint state configuration.

        velocity : Union[cs.casadi.DM, List, np.ndarray] (default is None)
            Joint state velocity. Only packed in message when
            non-None.

        effort : Union[cs.casadi.DM, List, np.ndarray] (default is None)
            Goal/Applied effort in the joint. Only packed in message
            when non-None.

        Returns
        -------

        joint_state : rospy.JointState
            Joint state ROS message, the method additionally packs the
            current time stamp and joint names from the robot model.

        """
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

    def publish_target_joint_state_msg(
            self,
            robot_name: str,
            position: Union[cs.casadi.DM, List, np.ndarray],
            velocity: Optional[Union[cs.casadi.DM, List, np.ndarray]]=None,
            effort: Optional[Union[cs.casadi.DM, List, np.ndarray]]=None) -> None:
        """Publishes a joint state message to a robot in the ROS-PyBullet Interface.

        Parameters
        ----------

        robot_name : str
            The name of the robot. This must match the name of the
            robot in the ROS-PyBullet Interface and be an index in the
            robots keys given in to the constructor of the RosNode
            class.

        position : Union[cs.casadi.DM, List, np.ndarray]
            Joint state configuration.

        velocity : Union[cs.casadi.DM, List, np.ndarray] (default is None)
            Joint state velocity. Only packed in message when
            non-None.

        effort : Union[cs.casadi.DM, List, np.ndarray] (default is None)
            Goal/Applied effort in the joint. Only packed in message
            when non-None.
        """
        self._assert_robot(robot_name)
        assert ROS_PYBULLET_INTERFACE_AVAILABLE, "this method requires the ROS-PyBullet Interface"
        msg = self.pack_joint_state_msg(robot_name, position, velocity=velocity, effort=effort)
        topic = f'rpbi/{robot_name}/joint_states/target'
        self.pubs[topic].publish(msg)

    def spin(self) -> None:
        """Simply keeps python from exiting until this node is stopped."""
        rospy.loginfo('Started spinnning...')
        rospy.spin()

class ROSRobot:

    def __init__(self, urdf_filename, parent, child, joint_states_topic='joint_states'):
        assert ROS_AVAILABLE, f"ROSRobot requires ROS"
        self._robot_model = RobotModel(urdf_filename)
        self._joint_names = self._robot_model.actuated_joint_names
        self._fk = self._robot_model.fk(parent, child)
        self._msg = None

        rospy.Subscriber(joint_states_topic, JointState, self._callback)

    def _callback(self, msg):
        self._msg = resolve_joint_order(msg, self._joint_names)

    def recieved_joint_state(self):
        return self._msg is not None

    def get_joint_position(self):
        if self.recieved_joint_state():
            return cs.DM(self._msg.position)

    def get_joint_velocity(self):
        if self.recieved_joint_state():
            return cs.DM(self._msg.velocity)

    def get_joint_effort(self):
        if self.recieved_joint_state():
            return cs.DM(self._msg.effort)

    def get(self, label):
        q = self.get_joint_position()
        if q is None: return
        fun = self._fk[label]
        return fun(q)

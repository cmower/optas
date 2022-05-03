import casadi as cs
import numpy as np
from urdf2casadi import urdfparser as u2c
ROS_AVAILABLE = True
try:
    import rospy
    from sensor_msgs.msg import JointState
except ImportError:
    ROS_AVAILABLE = False

class RobotModel:

    """Robot model class"""

    def __init__(self, urdf_file_name, root, tip):

        # Setup parser
        self.__parser = u2c.URDFparser(use_jit=False)
        self.__parser.from_file(urdf_file_name)

        # FK dict from parser, contains:
        # - joint_names
        # - upper
        # - lower
        # - joint_list
        # - q
        # - quaternion_fk
        # - dual_quaternion_fk
        # - T_fk
        self.__fk_dict = self.__parser.get_forward_kinematics(root, tip)

    @property
    def ndof(self):
        """Number of Degrees Of Freedom"""
        return self.__fk_dict['q'].shape[0]

    @property
    def joint_names(self):
        """Joint names"""
        return self.__fk_dict['joint_names']

    @property
    def lower_joint_limits(self):
        """Lower joint limits"""
        return self.__fk_dict['lower']

    @property
    def upper_joint_limits(self):
        """Upper joint limits"""
        return self.__fk_dict['upper']

    def __check_q(self, q):
        """Check q is correct type/shape"""
        if not isinstance(q, (cs.casadi.SX, cs.casadi.DM, list, tuple, np.ndarray)): raise TypeError(f"q must be casadi.casadi.SX/DM not {type(q)}")
        q = cs.SX(q)  # ensure q is symbolic
        if q.shape != (self.ndof, 1): raise ValueError(f"q is incorrect shape, expected {self.ndof}-by-1, got {q.shape[0]}-by-{q.shape[1]}")

    def get_end_effector_transformation_matrix(self, q):
        """Return the symbolic end-effector transformation matrix"""
        self.__check_q(q)
        T = self.__fk_dict['T_fk']
        return T(q)

    def get_end_effector_position(self, q):
        """Return the symbolic end-effector position"""
        T = self.get_end_effector_transformation_matrix(q)
        return T[:3, 3]

    def get_end_effector_quaternion(self, q):
        """Return the symbolic end-effector quaternion"""
        self.__check_q(q)
        quat = self.__fk_dict['quaternion_fk']
        return quat(q)

    def get_end_effector_euler(self, q):
        quat = self.get_end_effector_quaternion(q)
        qw = quat[3]
        qx = quat[0]
        qy = quat[1]
        qz = quat[2]
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = cs.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (qw * qy - qz * qx)
        pitch = cs.if_else(
            sinp**2 >= 1.0,
            0.5*cs.sign(sinp)*cs.np.pi,
            cs.arcsin(sinp),
        )

        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = cs.arctan2(siny_cosp, cosy_cosp)

        return cs.vertcat(roll, pitch, yaw)

    def to_ros_joint_state_msg(self, q):
        """Convert joint angles to a list of ROS sensor_msgs/JointState messages"""
        assert ROS_AVAILABLE, "ROS is not installed, you can not call to_ros_joint_state_msg"
        assert isinstance(q, (cs.casadi.DM, np.ndarray)), f"cannot parse q of type {type(q)} as joint state message"
        q = cs.DM(q).toarray().flatten()
        assert q.shape[0] == self.ndof, "q is incorrect shape"
        msg = JointState(name=self.joint_names, position=q)
        msg.header.stamp = rospy.Time.now()
        return msg

import casadi as cs
from urdf2casadi import urdfparser as u2c

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
        if not isinstance(q, cs.casadi.SX): raise TypeError("q must be casadi.casadi.SX")
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
        quat = self.fk_dict['quaternion_fk']
        return quat(q)

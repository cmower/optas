import casadi as cs
from typing import Dict
from urdf_parser_py.urdf import URDF, Joint, Link, Pose
from .tf import transformation_matrix_fixed, \
    transformation_matrix_prismatic, \
    transformation_matrix_revolute, \
    euler_from_transformation_matrix, \
    quaternion_product, \
    quaternion_fixed, \
    quaternion_revolute

class RobotModel:

    """Robot model class"""

    def __init__(self, urdf_filename, base_link_name='baselink', base_xyz=[0.0, 0.0, 0.0], base_rpy=[0.0, 0.0, 0.0], base_joint_name='basejoint'):
        """Constructor for the RobotModel class.

        Parameters
        ----------

        urdf_filename : str
            Filename for the URDF file.

        base_link_name : str (default is 'baselink')
            Name for the base link of the URDF. This allows you to
            position several robots with respect to a common frame.

        base_xyz : list[float] (default is [0., 0., 0.])
            The xyz position of the base link (must be a list of three elements).

        base_rpy : list[float] (default is [0., 0., 0.])
            The rpy orientation of the base link (must be a list of
            three elements). The orientation is defined by the
            roll-pitch-yaw Euler angles.

        base_joint_name : str (default is 'basejoint')
            The name for the base joint.

        """

        # Load robot from urdf
        self.robot = URDF.from_xml_file(urdf_filename)

        # Ensure base names are not already defined
        assert base_link_name not in self.robot.link_map, f"given base link name '{base_link_name}' already exists"
        assert base_joint_name not in self.robot.joint_map, f"given base joint name '{base_joint_name}' already exists"

        # Add base
        self.robot.add_link(Link(name=base_link_name))
        self.robot.add_joint(Joint(name=base_joint_name, parent=base_link_name, child=self.robot.links[0].name, joint_type='fixed', origin=Pose(xyz=base_xyz, rpy=base_rpy)))

    @property
    def ndof(self):
        """Number of Degrees Of Freedom"""
        return sum([jnt.type != 'fixed' for jnt in self.robot.joints])

    @property
    def actuated_joint_names(self):
        """Names of actuated joints"""
        return [jnt.name for jnt in self.robot.joints if jnt.type != 'fixed']

    @property
    def lower_actuated_joint_limits(self):
        """Lower position limits for actuated joints"""
        return [jnt.limit.lower for jnt in self.robot.joints if jnt.type != 'fixed']

    @property
    def upper_actuated_joint_limits(self):
        """Upper position limits for actuated joints"""
        return [jnt.limit.upper for jnt in self.robot.joints if jnt.type != 'fixed']

    def _get_joint_chain(self, parent, child):
        """Private method for returning the joint chain."""
        return [
            self.robot.joint_map[name]
            for name in self.robot.get_chain(parent, child)
            if name in self.robot.joint_map
        ]

    @staticmethod
    def _get_joint_origin(joint):
        """Private/static method for returning the joint origin."""
        if joint.origin is not None:
            xyz = cs.DM(joint.origin.xyz)
            rpy = cs.DM(joint.origin.rpy)
        else:
            xyz = cs.DM.zeros(3)
            rpy = cs.DM.zeros(3)
        return xyz, rpy

    @staticmethod
    def _get_joint_axis(joint):
        """Private/state method for returning the joint axis."""
        if joint.axis is not None:
            axis = cs.DM(joint.axis)
        else:
            axis = cs.DM([1.0, 0.0, 0.0])
        return axis

    def fk(self, parent: str, child: str) -> Dict[str, cs.casadi.Function]:
        """Forward kinematics.

        Parameters
        ----------

        parent : str
            The parent link name.

        child : str
            The child link name.

        Returns
        -------

        fk : Dict[str, casadi.casadi.Function]

        The foward kinematics containing the following functions. Each
        function is defined with respect to the joint configuration q
        with ndof elements.
        - 'T': 4-by-4 array defining the transformation frame between
          the parent and child links.
        - 'pos': position of the child link.
        - 'pos_jac': the jacobian array of the position.
        - 'eul': Euler angles defined the orientation between the
          parent and child links.
        - 'eul_jac': Jacobian of the Euler angles.
        - 'quat': Quaternion defining the orientation between the
          parent and child link.

        """

        # Initialize variables
        q = cs.SX.sym('q', self.ndof)
        T = cs.SX.eye(4)
        quat = cs.SX([0.0, 0.0, 0.0, 1.0])

        # Iterate over joints
        for joint in self._get_joint_chain(parent, child):

            # Handle fixed joints
            if joint.type == 'fixed':
                xyz, rpy = RobotModel._get_joint_origin(joint)
                T = T @ transformation_matrix_fixed(xyz, rpy)
                quat = quaternion_product(quat, quaternion_fixed(rpy))
                continue

            # Get actuated joint variable
            joint_index = self.actuated_joint_names.index(joint.name)
            qi = q[joint_index]

            # Handle actuated joints
            if joint.type == 'prismatic':
                xyz, rpy = RobotModel._get_joint_origin(joint)
                axis = RobotModel._get_joint_axis(joint)
                T = T @ transformation_matrix_prismatic(xyz, rpy, axis, qi)
                quat = quaternion_product(quat, quaternion_fixed(rpy))

            elif joint.type in {'revolute', 'continuous'}:
                xyz, rpy = RobotModel._get_joint_origin(joint)
                axis = RobotModel._get_joint_axis(joint)
                axis /= cs.norm_fro(axis)  # ensure axis is normalized
                T = T @ transformation_matrix_revolute(xyz, rpy, axis, qi)
                quat = quaternion_product(quat, quaternion_revolute(xyz, rpy, axis, qi))

        # Get Euler angles and position
        eul = euler_from_transformation_matrix(T)
        pos = T[:3, 3]

        return {
            'T': cs.Function('T', [q], [T]),
            'pos': cs.Function('pos', [q], [pos]),
            'pos_jac': cs.Function('pos_jac', [q], [cs.jacobian(pos, q)]),
            'eul': cs.Function('eul', [q], [eul]),
            'eul_jac': cs.Function('eul_jac', [q], [cs.jacobian(eul, q)]),
            'quat': cs.Function('quat', [q], [quat]),
        }

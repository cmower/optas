import casadi as cs

# https://pypi.org/project/urdf-parser-py
from urdf_parser_py.urdf import URDF, Joint, Link, Pose

from .spatialmath import *

class RobotModel:

    def __init__(self, urdf_filename):
        self._urdf = URDF.from_xml_file(urdf_filename)

    def _add_fixed_link(self, parent_link_name, child_link_name, xyz=None, rpy=None, joint_name=None):

        if xyz is None:
            xyz=[0.0]*3

        if rpy is None:
            rpy=[0.0]*3

        if joint_name is None:
            joint_name = parent_link_name + '_and_' + child_link_name + '_joint'

        self._urdf.add_link(Link(name=child_link_name))

        origin = Pose(xyz=xyz, rpy=rpy)
        self._urdf.add_joint(
            Joint(
                name=joint_name,
                parent=parent_link_name,
                child=child_link_name,
                joint_type='fixed',
                origin=origin,
            )
        )

    def add_base_frame(self, base_link_name, xyz=None, rpy=None, joint_name=None):
        """Add new base frame, note this changes the root link."""
        assert base_link_name not in self.link_names, f"'{base_link_name}' already exists"
        self._add_fixed_link(base_link_name, self._urdf.get_root(), xyz=xyz, rpy=rpy, joint_name=joint_name)

    def add_fixed_link(self, link_name, parent_link_name, xyz=None, rpy=None, joint_name=None):
        """Add a fixed link"""
        assert link_name not in self.link_names, f"'{link_name}' already exists"
        assert parent_link_name in self.link_names, f"{parent_link_name=}, does not appear in link names"
        self._add_fixed_link(parent_link_name, link_name, xyz=xyz, rpy=rpy, joint_name=joint_name)

    def get_root_link(self):
        """Return the root link"""
        return self._urdf.get_root()

    @property
    def joint_names(self):
        """All joint names"""
        return [jnt.name for jnt in self._urdf.joints]

    @property
    def link_names(self):
        """All link names"""
        return [lnk.name for lnk in self._urdf.links]

    @property
    def actuated_joint_names(self):
        """Names of actuated joints"""
        return [jnt.name for jnt in self._urdf.joints if jnt.type != 'fixed']

    @property
    def lower_actuated_joint_limits(self):
        """Lower position limits for actuated joints"""
        return [jnt.limit.lower for jnt in self.robot.joints if jnt.type != 'fixed']

    @property
    def upper_actuated_joint_limits(self):
        """Upper position limits for actuated joints"""
        return [jnt.limit.upper for jnt in self.robot.joints if jnt.type != 'fixed']

    @property
    def ndof(self):
        """Number of Degrees Of Freedom"""
        return len(self.actuated_joint_names)

    @staticmethod
    def _get_joint_origin(joint):
        """Get the origin for the joint"""
        xyz, rpy = cs.DM.zeros(3), cs.DM.zeros(3)
        if joint.origin is not None:
            xyz, rpy = cs.DM(joint.origin.xyz), cs.DM(joint.origin.rpy)
        return xyz, rpy

    @staticmethod
    def _get_joint_axis(joint):
        """Get the axis of joint, the axis is normalized for revolute/continuous joints"""
        axis = cs.DM(joint.axis) if joint.axis is not None else cs.DM([1., 0., 0.])
        if joint.type in {'revolute', 'continuous'}:
            axis = unit(axis)
        return axis

    @vectorize_args
    def get_global_link_transform(self, link_name, q):
        """Get the link transform in the global frame for a given joint state q"""

        assert link_name in self._urdf.link_map.keys(), "given link_name does not appear in URDF"

        T = I4()
        if link_name == self._urdf.get_root(): return T

        for joint in self._urdf.joints:

            xyz, rpy = self._get_joint_origin(joint)

            if joint.type == 'fixed':
                T  = T @ rt2tr(rpy2r(rpy), xyz)
                continue

            joint_index = self.actuated_joint_names.index(joint.name)
            qi = q[joint_index]

            if joint.type in {'revolute', 'continuous'}:
                T = T @ rt2tr(rpy2r(rpy), xyz)
                T = T @ r2t(angvec2r(qi, self._get_joint_axis(joint)))

            else:
                raise NotImplementedError(f"{joint.type} joints are currently not supported")

            if joint.child == link_name:
                break

        return T

    def get_global_link_position(self, link_name, q):
        """Get the link position in the global frame for a given joint state q"""
        return transl(self.get_global_link_transform(link_name, q))

    @vectorize_args
    def get_global_link_quaternion(self, link_name, q):
        """Get the link orientation as a quaternion in the global frame for a given joint state q"""

        assert link_name in self._urdf.link_map.keys(), "given link_name does not appear in URDF"

        quat = Quaternion(0., 0., 0., 1.)
        if link_name == self._urdf.get_root(): return quat

        for joint in self._urdf.joints:

            xyz, rpy = self._get_joint_origin(joint)

            if joint.type == 'fixed':
                quat = quat * Quaternion.fromrpy(rpy)
                continue

            joint_index = self.actuated_joint_names.index(joint.name)
            qi = q[joint_index]

            if joint.type in {'revolute', 'continuous'}:
                quat = quat * Quaternion.fromrpy(rpy)
                quat = quat * Quaternion.fromangvec(qi, self._get_joint_axis(joint))

            else:
                raise NotImplementedError(f"{joint.type} joints are currently not supported")

            if joint.child == link_name:
                break

        return quat.getquat()

    @vectorize_args
    def get_geometric_jacobian(self, link_name, q):
        """Get the geometric jacobian matrix for a given link and joint state q"""

        #
        # TODO: allow user to compute jacobian in any frame (i.e. not
        # only the root link).
        #

        e = self.get_global_link_position(link_name, q)

        w = cs.DM.zeros(3)
        pdot = cs.DM.zeros(3)

        R = I3()

        joint_index_order = []
        jacobian_columns = []

        for joint in self._urdf.joints:

            xyz, rpy = self._get_joint_origin(joint)

            if joint.type == 'fixed':
                R = R @ rpy2r(rpy)
                continue

            joint_index = self.actuated_joint_names.index(joint.name)
            joint_index_order.append(joint_index)
            qi = q[joint_index]

            if joint.type in {'revolute', 'continuous'}:

                axis = self._get_joint_axis(joint)

                R = R @ rpy2r(rpy)
                R = R @ angvec2r(qi, axis)
                p = self.get_global_link_position(joint.child, q)

                z = R @ axis
                pdot = cs.cross(z, e - p)

                jcol = cs.vertcat(pdot, z)
                jacobian_columns.append(jcol)

            # elif joint.type == 'prismatic':  # TODO

            else:
                raise NotImplementedError(f"{joint.type} joints are currently not supported")

        # Sort columns of jacobian
        jacobian_columns_ordered = [jacobian_columns[idx] for idx in joint_index_order]

        # Build jacoian array
        J = jacobian_columns_ordered.pop(0)
        while jacobian_columns_ordered:
            J = cs.horzcat(J, jacobian_columns_ordered.pop(0))

        return J

    def get_manipulability(self, link_index, q):
        """Get the manipulability measure"""
        J = self.get_geometric_jacobian(link_index, q)
        return cs.sqrt(cs.det(J @ J.T))

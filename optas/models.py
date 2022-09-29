import casadi as cs

# https://pypi.org/project/urdf-parser-py
from urdf_parser_py.urdf import URDF, Joint, Link, Pose

from .spatialmath import *

class Model:

    """

    Model base class

    name (str):
        name of model

    dim (int):
        model dimension (for robots this is ndof)

    time_derivs (list[int]):
        time derivatives required for model, 0 means not time
        derivative, 1 means first derivative wrt to time is required,
        etc

    symbol (str):
        a short symbol to represent the model

    dlim (dict[ int, tuple[list[float]] ]):
        limits on each time derivative, index should correspond to a
        time derivative (i.e. 0, 1, ...) and the value should be a
        tuple of two lists containing the lower and upper bounds

    """


    def __init__(self, name, dim, time_derivs, symbol, dlim):
        self.name = name
        self.dim = dim
        self.time_derivs = time_derivs
        self.symbol = symbol
        self.dlim = dlim


    def get_name(self):
        return self.name


    def state_name(self, time_deriv):
        assert time_deriv in self.time_derivs, f"Given time derivative {time_deriv=} is not recognized, only allowed {self.time_derivs}"
        return self.name + '/' + 'd'*time_deriv + self.symbol


    def get_limits(self, time_deriv):
        assert time_deriv in self.time_derivs, f"Given time derivative {time_deriv=} is not recognized, only allowed {self.time_derivs}"
        assert time_deriv in self.dlim.keys(), f"Limit for time derivative {time_deriv=} has not been given"
        return self.dlim[time_deriv]


    def in_limit(self, x, time_deriv):
        lo, up = self.get_limits(time_deriv)
        return cs.logic_all(cs.logical_and(lo <= x, x <= up))


    def in_limits(self, states):
        in_limits = []
        for time_deriv, state in states.items():
            in_limits.append(self.in_limit(state, time_deriv))
        return cs.logical_all(cs.vertcat(*in_limits))


class TaskModel(Model):


    def __init__(self, name, dim, time_derivs=[0], symbol='x', dlim={}):
        super().__init__(name, dim, time_derivs, symbol, dlim)


class JointTypeNotSupported(NotImplementedError):

    def __init__(self, joint_type):
        msg = f'{joint_type} joints are currently not supported\n'
        msg += 'if you require this joint type please raise an issue at '
        msg += 'https://github.com/cmower/optas/issues'
        super().__init__(msg)


class RobotModel(Model):


    def __init__(self, urdf_filename=None, urdf_string=None, name=None, time_derivs=[0], qddlim=None):

        # Load URDF
        self._urdf = None
        if(urdf_filename!=None):
            self._urdf = URDF.from_xml_file(urdf_filename)
        if(urdf_string!=None):
            self._urdf = URDF.from_xml_string(urdf_string)
        assert self._urdf is not None, "You need to supply a urdf, either through filename or as a string"

        # Setup joint limits, joint position/velocity limits
        dlim = {
            0: (self.lower_actuated_joint_limits, self.upper_actuated_joint_limits),
            1: (-self.velocity_actuated_joint_limits, self.velocity_actuated_joint_limits),
        }

        # Handle potential acceleration limit
        if qddlim:
            qddlim = vec(qddlim)
            if qddlim.shape[0] == 1:
                qddlim = qddlim*cs.DM.ones(self.ndof)
            assert qddlim.shape[0] == self.ndof, f"expected ddlim to have {self.ndof} elements"
            dlim[2] = -qddlim, qddlim

        # If user did not supply name for the model then use the one in the URDF
        if name is None:
            name = self._urdf.name

        super().__init__(name, self.ndof, time_derivs, 'q', dlim)

    @property
    def joint_names(self):
        return [jnt.name for jnt in self._urdf.joints]

    @property
    def link_names(self):
        return [lnk.name for lnk in self._urdf.links]

    @property
    def actuated_joint_names(self):
        return [jnt.name for jnt in self._urdf.joints if jnt.type != 'fixed']

    @property
    def ndof(self):
        return len(self.actuated_joint_names)

    @property
    def lower_actuated_joint_limits(self):
        return cs.DM([jnt.limit.lower for jnt in self._urdf.joints if jnt.type != 'fixed'])

    @property
    def upper_actuated_joint_limits(self):
        return cs.DM([jnt.limit.upper for jnt in self._urdf.joints if jnt.type != 'fixed'])

    @property
    def velocity_actuated_joint_limits(self):
        return cs.DM([jnt.limit.velocity for jnt in self._urdf.joints if jnt.type != 'fixed'])

    def add_base_frame(self, base_link, xyz=None, rpy=None, joint_name=None):
        """Add new base frame, note this changes the root link."""

        parent_link = base_link
        child_link = self._urdf.get_root()  # i.e. current root

        if xyz is None:
            xyz=[0.0]*3

        if rpy is None:
            rpy=[0.0]*3

        if not isinstance(joint_name, str):
            joint_name = parent_link + '_and_' + child_link + '_joint'

        self._urdf.add_link(Link(name=parent_link))

        joint = Joint(
            name=joint_name,
            parent=parent_link,
            child=child_link,
            joint_type='fixed',
            origin=Pose(xyz=xyz, rpy=rpy),
        )
        self._urdf.add_joint(joint)


    def add_fixed_link(self, link, parent_link, xyz=None, rpy=None, joint_name=None):
        """Add a fixed link"""

        assert link not in self.link_names, f"'{link}' already exists"
        assert parent_link in self.link_names, f"{parent_link=}, does not appear in link names"

        child_link = link

        if xyz is None:
            xyz=[0.0]*3

        if rpy is None:
            rpy=[0.0]*3

        if not isinstance(joint_name, str):
            joint_name = parent_link + '_and_' + child_link + '_joint'

        self._urdf.add_link(Link(name=child_link))

        origin = Pose(xyz=xyz, rpy=rpy)
        self._urdf.add_joint(
            Joint(
                name=joint_name,
                parent=parent_link,
                child=child_link,
                joint_type='fixed',
                origin=origin,
            )
        )


    def get_root_link(self):
        """Return the root link"""
        return self._urdf.get_root()


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

    def _get_actuated_joint_index(self, joint_name):
        return self.actuated_joint_names.index(joint_name)



    def get_random_joint_positions(self):
        lo = self.lower_actuated_joint_limits.toarray()
        hi = self.upper_actuated_joint_limits.toarray()
        return cs.vec(cs.np.random.uniform(lo, hi))


    def get_random_pose_in_global_link(self, link):
        q = self.get_random_joint_positions()
        return self.get_global_link_transform(link, q)


    def _make_function(self, label, link, method, n=1, base_link=None):
        q = cs.SX.sym('q', self.ndof)
        args = (link, q)
        kwargs = {}
        if base_link is not None:
            kwargs['base_link'] = base_link
        out = method(*args, **kwargs)
        F = cs.Function(label, [q], [out])
        if n > 1:
            F = F.map(n)
        return F


    @vectorize_args
    def get_global_link_transform(self, link, q):
        """Get the link transform in the global frame for a given joint state q"""

        # Setup
        assert link in self._urdf.link_map.keys(), "given link does not appear in URDF"
        root = self._urdf.get_root()
        T = I4()
        if link == root: return T

        # Iterate over joints in chain
        for joint_name in self._urdf.get_chain(root, link, links=False):

            joint = self._urdf.joint_map[joint_name]
            xyz, rpy = self._get_joint_origin(joint)

            if joint.type == 'fixed':
                T  = T @ rt2tr(rpy2r(rpy), xyz)
                continue

            joint_index = self._get_actuated_joint_index(joint.name)
            qi = q[joint_index]

            if joint.type in {'revolute', 'continuous'}:
                T = T @ rt2tr(rpy2r(rpy), xyz)
                T = T @ r2t(angvec2r(qi, self._get_joint_axis(joint)))

            else:
                raise JointTypeNotSupported(joint.type)

        return T


    def get_global_link_transform_function(self, link):
        return self._make_function('T', link, self.get_global_link_transform)


    @vectorize_args
    def get_link_transform(self, link, q, base_link):
        T_L_W = self.get_global_link_transform(link, q)
        T_B_W = self.get_global_link_transform(base_link, q)
        return T_L_W @ invt(T_B_W)


    def get_link_transform_function(self, link, base_link):
        return self._make_function('T', link, self.get_link_transform, base_link=base_link)


    def get_global_link_position(self, link, q):
        """Get the link position in the global frame for a given joint state q"""
        return transl(self.get_global_link_transform(link, q))


    def get_global_link_position_function(self, link, n=1):
        return self._make_function('p', link, self.get_global_link_position, n=n)


    def get_link_position(self, link, q, base_link):
        return transl(self.get_link_transform(link, q, base_link))


    def get_link_position_function(self, link, base_link, n=1):
        return self._make_function('p', link, self.get_link_position, n=n, base_link=base_link)


    def get_global_link_rotation(self, link, q):
        """Get the link rotation in the global frame for a given joint state q"""
        return t2r(self.get_global_link_transform(link, q))


    def get_global_link_rotation_function(self, link):
        return self._make_function('R', link, self.get_global_link_rotation)


    def get_link_rotation(self, link, q, base_link):
        return t2r(self.get_link_transform(link, q, base_link))


    def get_link_rotation_function(self, link, base_link):
        return self._make_function('R', link, self.get_link_rotation, base_link=base_link)


    @vectorize_args
    def get_global_link_quaternion(self, link, q):

        # Setup
        assert link in self._urdf.link_map.keys(), "given link does not appear in URDF"
        root = self._urdf.get_root()
        quat = Quaternion(0., 0., 0., 1.)
        if link == root: return quat

        # Iterate over joints in chain
        for joint_name in self._urdf.get_chain(root, link, links=False):

            joint = self._urdf.joint_map[joint_name]
            xyz, rpy = self._get_joint_origin(joint)

            if joint.type == 'fixed':
                quat = Quaternion.fromrpy(rpy) * quat
                continue

            joint_index = self._get_actuated_joint_index(joint.name)
            qi = q[joint_index]

            if joint.type in {'revolute', 'continuous'}:
                quat = Quaternion.fromrpy(rpy) * quat
                quat = Quaternion.fromangvec(qi, self._get_joint_axis(joint)) * quat

            else:
                raise JointTypeNotSupported(joint.type)

        return quat.getquat()


    def get_global_link_quaternion_function(self, link, n=1):
        return self._make_function('quat', link, self.get_global_link_quaternion, n=n)


    def get_link_quaternion(self, link, q, base_link):
        quat_L_W = Quaternion(self.get_global_link_quaternion(link, q))
        quat_B_W = Quaternion(self.get_global_link_quaternion(base_link, q))
        return (quat_L_W * quat_B_W.inv()).getquat()


    def get_link_quaternion_function(self, link, base_link, n=1):
        return self._make_function('quat', link, self.get_link_quaternion, n=n, base_link=base_link)


    def get_global_geometric_jacobian(self, link, q):

        e = self.get_global_link_position(link, q)

        w = cs.DM.zeros(3)
        pdot = cs.DM.zeros(3)

        joint_index_order = []
        jacobian_columns = []

        past_in_chain = False

        for joint in self._urdf.joints:

            if joint.type == 'fixed':
                continue

            if joint.child == link:
                past_in_chain = True

            joint_index = self._get_actuated_joint_index(joint.name)
            joint_index_order.append(joint_index)
            qi = q[joint_index]

            if past_in_chain:
                jcol = cs.DM.zeros(6)
                jacobian_columns.append(jcol)

            elif joint.type in {'revolute', 'continuous'}:

                axis = self._get_joint_axis(joint)
                R = self.get_global_link_rotation(joint.child, q)
                R = R @ angvec2r(qi, axis)
                p = self.get_global_link_position(joint.child, q)

                z = R @ axis
                pdot = cs.cross(z, e - p)

                jcol = cs.vertcat(pdot, z)
                jacobian_columns.append(jcol)

            else:
                raise JointTypeNotSupported(joint.type)

        # Sort columns of jacobian
        jacobian_columns_ordered = [jacobian_columns[idx] for idx in joint_index_order]

        # Build jacoian array
        J = jacobian_columns_ordered.pop(0)
        while jacobian_columns_ordered:
            J = cs.horzcat(J, jacobian_columns_ordered.pop(0))

        return J


    def get_global_geometric_jacobian_function(self, link):
        return self._make_function('J', link, self.get_global_geometric_jacobian)


    @vectorize_args
    def get_geometric_jacobian(self, link, q, base_link):

        J = self.get_global_geometric_jacobian(link, q)

        # Transform jacobian to given base link
        R = self.get_global_link_rotation(base_link, q).T
        O = cs.DM.zeros(3, 3)
        K = cs.vertcat(
            cs.horzcat(R, O),
            cs.horzcat(O, R),
        )
        J = K @ J

        return J


    def get_geometric_jacobian_function(self, link, base_link):
        return self._make_function('J', link, self.get_geometric_jacobian, base_link=base_link)


    def get_global_linear_geometric_jacobian(self, link, q):
        J = self.get_global_geometric_jacobian(link, q)
        return J[:3, :]


    def get_global_linear_geometric_jacobian_function(self, link):
        return self._make_function('Jl', link, self.get_global_linear_geometric_jacobian)


    def get_linear_geometric_jacobian(self, link, q, base_link):
        J = self.get_geometric_jacobian(link, q, base_link)
        return J[:3, :]


    def get_linear_geometric_jacobian_function(self, link, base_link):
        return self._make_function('Jl', link, self.get_linear_geometric_jacobian, base_link=base_link)


    def get_global_angular_geometric_jacobian(self, link, q):
        J = self.get_global_geometric_jacobian(link, q)
        return J[3:, :]


    def get_global_angular_geometric_jacobian_function(self, link):
        return self._make_function('Ja', link, self.get_global_angular_geometric_jacobian)


    def get_angular_geometric_jacobian(self, link, q, base_link):
        J = self.get_geometric_jacobian(link, q, base_link)
        return J[3:, :]


    def get_angular_geometric_jacobian_function(self, link, base_link):
        return self._make_function('Ja', link, self.get_angular_geometric_jacobian, base_link=base_link)


    def _manipulability(self, J):
        return cs.sqrt(cs.det(J @ J.T))


    def get_global_manipulability(self, link, q):
        J = self.get_global_geometric_jacobian(link, q)
        return self._manipulability(J)


    def get_global_manipulability_function(self, link, n=1):
        return self._make_function('m', link, self.get_global_manipulability, n=n)


    def get_manipulability(self, link, q, base_link):
        """Get the manipulability measure"""
        J = self.get_geometric_jacobian(link, q, base_link)
        return self._manipulability(J)


    def get_manipulability_function(self, link, base_link, n=1):
        return self._make_function('m', link, self.get_manipulability, n=n, base_link=base_link)


    def get_global_linear_manipulability(self, link, q):
        Jl = self.get_global_linear_geometric_jacobian(link, q)
        return self._manipulability(Jl)


    def get_global_linear_manipulability_function(self, link, n=1):
        return self._make_function('ml', link, self.get_global_linear_manipulability, n=n)


    def get_linear_manipulability(self, link, q, base_link):
        Jl = self.get_linear_geometric_jacobian(link, q, base_link)
        return self._manipulability(Jl)


    def get_linear_manipulability_function(self, link, base_link, n=1):
        return self._make_function('Jl', link, self.get_linear_manipulability, n=n, base_link=base_link)


    def get_global_angular_manipulability(self, link, q):
        Ja = self.get_global_angular_geometric_jacobian(link, q)
        return self._manipulability(Ja)


    def get_global_angular_manipulability_function(self, link, n=1):
        return self._make_function('ma', link, self.get_global_angular_manipulability, n=n)


    def get_angular_manipulability(self, link, q, base_link):
        Ja = self.get_angular_geometric_jacobian(link, q, base_link)
        return self._manipulability(Ja)


    def get_angular_manipulability_function(self, link, base_link, n=1):
        return self._make_function('Ja', link, self.get_angular_manipulability, n=n, base_link=base_link)

import casadi as cs
from .joint_model import JointModel, JointType

class RevoluteJointModel(JointModel):

    def __init__(self, name):
        super().__init__(name)
        self._type = JointType.REVOLUTE
        self._axis = None
        self._x2 = None
        self._y2 = None
        self._z2 = None
        self._xy = None
        self._xz = None
        self._yz = None

    def setAxis(self, axis):
        a = cs.vec(axis)
        a_ = a[:3]
        self._axis = a_/cs.norm_fro(a_)
        self._x2 = self._axis[0]*self._axis[0]
        self._y2 = self._axis[1]*self._axis[1]
        self._z2 = self._axis[2]*self._axis[2]
        self._xy = self._axis[0]*self._axis[1]
        self._xz = self._axis[0]*self._axis[2]
        self._yz = self._axis[1]*self._axis[2]

    def getAxis(self, axis):
        return self._axis

    def computeTransform(self, joint_values):
        joint_values = cs.vec(joint_values)
        c, s = cs.cos(joint_values[0]), cs.sin(joint_values[0])
        t = 1. - c
        txy = t * self._xy
        txz = t * self._xz
        tyz = t * self._yz
        
        zs = self._axis[2]*s
        ys = self._axis[1]*s
        xs = self._axis[0]*s

        c1 = cs.vertcat(
            t * self._x2 + c,
            txy + zs,
            txz - ys,
            0.0,
        )

        c2 = cs.vertcat(
            txy - zs,
            t * self._y2 + c,
            tyz + xs,
            0.0,
        )

        c3 = cs.vertcat(
            txz + ys,
            tyz - xs,
            t * self.z2 + c,
            0.0,
        )

        c4 = cs.vertcat(0., 0., 0., 1.)

        return cs.horzcat(c1, c2, c3, c4)

    def computeVariablePositions(self, T):
        raise NotImplementedError("see http://docs.ros.org/en/jade/api/moveit_core/html/revolute__joint__model_8cpp_source.html#l00244")

    def distance(self, value1, value2):
        v1 = cs.vec(value1)[0]
        v2 = cs.vec(value2)[0]
        return cs.fabs(v1 - v2)

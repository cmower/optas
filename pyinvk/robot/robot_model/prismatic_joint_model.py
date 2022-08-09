import casadi as cs
from .joint_model import JointModel, JointType

class PrismaticJointModel(JointModel):

    def __init__(self, name):
        super().__init__(name)
        self._type = JointType.PRISMATIC
        self._axis = None

    def setAxis(self, axis):
        a = cs.vec(axis)
        self._axis = a[:3]

    def getAxis(self, axis):
        return self._axis

    def computeTransform(self, joint_values):
        joint_values = cs.vec(joint_values)
        return cs.vertcat(
            cs.DM.eye(3), self._axis*joint_values[0],
            cs.DM([[0., 0., 0., 1.]]),
        )

    def computeVariablePositions(self, T):
        return cs.dot(T[:3, 3], self._axis)

    def distance(self, value1, value2):
        v1 = cs.vec(value1)[0]
        v2 = cs.vec(value2)[0]
        return cs.fabs(v1 - v2)        
        

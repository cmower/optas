import casadi as cs
from .joint_model import JointModel, JointType

class FixedJointModel(JointModel):

    def __init__(self, name):
        super().__init__(name)
        self._type = JointType.FIXED

    def computeTransform(self, joint_values):
        return cs.DM.eye(3)

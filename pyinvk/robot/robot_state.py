
class RobotState:

    def __init__(self):
        pass

    def getJacobian(self):
        root_joint_model = self._group.getJointModels()[0]
        root_link_model = root_joint_model.getParentLinkModel()

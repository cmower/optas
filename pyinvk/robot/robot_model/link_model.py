
class LinkModel:

    def __init__(self, name):
        self._name = name
        self._link_index = None
        self._parent_joint_model = None
        self._child_joint_model = None
        self._parent_link_model = None
        
    def getName(self):
        return self._name

    def setLinkIndex(self, index):
        self._link_index = index

    def getLinkIndex(self):
        return self._link_index

    def setParentJointModel(self, joint_model):
        self._parent_joint_model = joint_model
        
    def setChildJointModel(self, joint_model):
        self._child_joint_model = joint_model

    def setParentJointModel(self):
        return self._parent_joint_model
        
    def setChildJointModel(self):
        return self._child_joint_model

    def getParentLinkModel(self):
        return self._parent_link_model

    def setParentLinkModel(self, link_model):
        self._parent_link_model = link_model

    def setJointOriginTransform(self, T):
        self._joint_origin_transform = cs.horzcat(T)  # horzcat ensures T is a DM/SX casadi array

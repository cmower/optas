import abc
import enum


class JointType(enum.Enum):
    UNKNOWN   = 0
    REVOLUTE  = 1
    PRISMATIC = 2
    PLANAR    = 3
    FLOATING  = 4
    FIXED     = 5


class JointModel(abc.ABC):


    def __init__(self, name):
        self._name = name
        self._type = None
        self._child_link_model = None
        self._parent_link_model = None

    def getName(self):
        return self._name

    def getType(self):
        return self._type

    def getTypeName(self):
        return str(self._type)

    def getChildLinkModel(self):
        return self._child_link_model

    def getParentLinkModel(self):
        return self._parent_link_model

    def setChildLinkModel(self, link_model):
        self._child_link_model = link_model

    def setParentLinkModel(self, link_model):
        self._parent_link_model = link_model

    @abc.abstractmethod
    def computeTransform(self, joint_values):
        pass

    @abc.abstractmethod
    def computeVariablePositions(self, T):
        pass

    @abc.abstractmethod
    def distance(self, value1, value2):
        pass

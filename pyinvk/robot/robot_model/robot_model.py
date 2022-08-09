
class RobotModel:

    def __init__(self, urdf_model):
        self._urdf_model = urdf_model
        self._joint_model_map = {}

    def getJointModel(self, name):
        return self._joint_model_map[name]

    

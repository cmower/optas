import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot
from roboticstoolbox.robot.ERobot import ERobot

try:
    # roboticstoolbox >= 1.5: URDF_read is a module-level function
    from roboticstoolbox.models.URDF.URDFRobot import URDF_read
except ImportError:
    # older roboticstoolbox: URDF_read is a method of ERobot
    URDF_read = None


class RobotModelTester(ERobot):
    def __init__(self, filename=None):
        if filename is None:
            cwd = pathlib.Path(
                __file__
            ).parent.resolve()  # path to current working directory
            filename = os.path.join(cwd, "tester_robot.urdf")

        if URDF_read is not None:
            links, name, _ = URDF_read(filename)
            super().__init__(links, name=name.upper())
        else:
            links, name, urdf_string, urdf_filepath = self.URDF_read(filename)
            super().__init__(
                links,
                name=name.upper(),
                urdf_string=urdf_string,
                urdf_filepath=urdf_filepath,
            )

        self.qz = np.zeros(self.n)


def main(q=None):
    robot = RobotModelTester()
    print(robot)
    if q is None:
        q = robot.qz
    robot.plot(q, backend="pyplot")
    plt.show()


if __name__ == "__main__":
    main()

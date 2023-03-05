import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot
from roboticstoolbox.robot.ERobot import ERobot


class RobotModelTester(ERobot):
    def __init__(self):
        cwd = pathlib.Path(
            __file__
        ).parent.resolve()  # path to current working directory
        filename = os.path.join(cwd, "tester_robot.urdf")
        links, name, urdf_string, urdf_filepath = self.URDF_read(filename)

        super().__init__(
            links,
            name=name.upper(),
            urdf_string=urdf_string,
            urdf_filepath=urdf_filepath,
        )

        self.qz = np.zeros(3)


def main(q=None):
    robot = RobotModelTester()
    print(robot)
    if q is None:
        q = robot.qz
    robot.plot(q, backend="pyplot")
    plt.show()


if __name__ == "__main__":
    main()

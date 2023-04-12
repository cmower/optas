import os
import pytest
import optas
import pathlib
import numpy as np
import scipy.linalg as linalg
import urdf_parser_py.urdf as urdf
from scipy.spatial.transform import Rotation as Rot

# from test_models import NUM_RANDOM, isclose

import pybullet as pb


def isclose(A: np.ndarray, B: np.ndarray):
    return np.isclose(A, B, atol=8.0e-2).all()


NUM_RANDOM = 100


class TestRnea:
    # Setup path to tester robot URDF
    cwd = pathlib.Path(__file__).parent.resolve()  # path to current working directory
    urdf_filename = "tester_robot_revolute.urdf"
    urdf_path = os.path.join(cwd, urdf_filename)
    # print(cwd)

    # Setup model (with no parameterized joints)
    model = optas.RobotModel(urdf_filename=urdf_path)

    pb.connect(pb.DIRECT)

    # pb.setAdditionalSearchPath(cwd)

    gravz = -9.81
    pb.setGravity(0, 0, gravz)
    id = pb.loadURDF(
        urdf_path,
        basePosition=[0, 0, 0],
    )

    def test_calculate_inverse_dynamics_symbol(self):
        q = optas.SX.sym("q", self.model.ndof)
        qd = optas.SX.sym("qd", self.model.ndof)
        qdd = optas.SX.sym("qdd", self.model.ndof)
        tau1 = self.model.rnea(q, qd, qdd)
        assert isinstance(tau1, optas.SX)

    def test_calculate_inverse_dynamics(self):
        for _ in range(NUM_RANDOM):
            q = self.model.get_random_joint_positions().toarray().flatten()
            qd = self.model.get_random_joint_positions().toarray().flatten()
            qdd = self.model.get_random_joint_positions().toarray().flatten()
            tau1 = self.model.rnea(q, qd, qdd)

            tau2 = np.array(
                pb.calculateInverseDynamics(
                    self.id, q.tolist(), qd.tolist(), qdd.tolist()
                )
            )
            assert isclose(tau1.toarray().flatten(), tau2)


def main(q=None):
    test = TestRnea()
    test.test_calculate_inverse_dynamics_symbol()
    test.test_calculate_inverse_dynamics()


if __name__ == "__main__":
    main()

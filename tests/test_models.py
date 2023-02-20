import os
import pytest
import optas
import pathlib
import numpy as np
import urdf_parser_py.urdf as urdf
from .tester_robot_model import RobotModelTester

NUM_RANDOM = 100

tester_robot_model = RobotModelTester()

###########################################################################
# Helper methods
#


def random_vector(lo=-1, hi=1, n=3):
    return np.random.uniform(-lo, hi, size=(n,))


def isclose(A: np.ndarray, B: np.ndarray):
    return np.isclose(A, B).all()


###########################################################################
# Tests
#


def test_listify_output():
    class _test_cls:
        @optas.models.listify_output
        def method(self, link, q):
            return 2.0 * q

    tc = _test_cls()

    q = optas.DM([1, 2, 3])
    q_exp = optas.DM([2, 4, 6])
    assert isclose(q_exp.toarray(), tc.method("test", q).toarray())

    q = optas.DM(
        [
            [1, 5],
            [2, 4],
            [3, 3],
        ]
    )
    q_exp = 2.0 * q

    assert isclose(q_exp.toarray(), tc.method("test", q).toarray())


class Test_Model:
    def test_get_name(self):
        name = "test"
        dim = 3
        time_deriv = [0, 1]
        symbol = "x"
        dlim = {
            0: ([-1, -1, -1], [1, 1, 1]),
            1: ([0, 0, 0], [1, 1, 2]),
        }
        T = 2

        model = optas.models.Model(name, dim, time_deriv, symbol, dlim, T)

        assert model.get_name() == name

    def test_state_name(self):
        name = "test"
        dim = 3
        time_deriv = [0, 1]
        symbol = "x"
        dlim = {
            0: ([-1, -1, -1], [1, 1, 1]),
            1: ([0, 0, 0], [1, 1, 2]),
        }
        T = 2

        model = optas.models.Model(name, dim, time_deriv, symbol, dlim, T)

        assert model.state_name(0) == "test/x"
        assert model.state_name(1) == "test/dx"

        for _ in range(NUM_RANDOM):
            with pytest.raises(AssertionError):
                n = np.random.randint(2, 10)
                model.state_name(n)

    def test_state_parameter_name(self):
        name = "test"
        dim = 3
        time_deriv = [0, 1]
        symbol = "x"
        dlim = {
            0: ([-1, -1, -1], [1, 1, 1]),
            1: ([0, 0, 0], [1, 1, 2]),
        }
        T = 2

        model = optas.models.Model(name, dim, time_deriv, symbol, dlim, T)

        assert model.state_parameter_name(0) == "test/x/p"
        assert model.state_parameter_name(1) == "test/dx/p"

        for _ in range(NUM_RANDOM):
            with pytest.raises(AssertionError):
                n = np.random.randint(2, 10)
                model.state_parameter_name(n)

    def test_get_limits(self):
        name = "test"
        dim = 3
        time_deriv = [0, 1]
        symbol = "x"
        dlim = {
            0: ([-1, -1, -1], [1, 1, 1]),
            1: ([0, 0, 0], [1, 1, 2]),
        }
        T = 2

        model = optas.models.Model(name, dim, time_deriv, symbol, dlim, T)

        for td, (lo_exp, hi_exp) in dlim.items():
            lo, hi = model.get_limits(td)
            assert isclose(lo, lo_exp)
            assert isclose(hi, hi_exp)

        for _ in range(NUM_RANDOM):
            with pytest.raises(AssertionError):
                n = np.random.randint(2, 10)
                model.get_limits(n)

    def test_in_limit(self):
        name = "test"
        dim = 3
        time_deriv = [0, 1]
        symbol = "x"
        dlim = {
            0: ([-1, -1, -1], [1, 1, 1]),
            1: ([0, 0, 0], [1, 1, 2]),
        }
        T = 2

        model = optas.models.Model(name, dim, time_deriv, symbol, dlim, T)

        def in_limit_known(x, td):
            lo, hi = dlim[td]
            return np.all(np.logical_and(lo <= x, x <= hi))

        for td in dlim.keys():
            for _ in range(NUM_RANDOM):
                x = np.random.uniform(-1.5, 1.5, size=(dim,))
                print(in_limit_known(x, td))
                print(model.in_limit(x, td).toarray().flatten())
                assert (
                    in_limit_known(x, td) == model.in_limit(x, td).toarray().flatten()
                )


def test_TaskModel():
    name = "test"
    dim = 3
    dlim = {
        0: ([-1, -1, -1], [1, 1, 1]),
    }
    T = 2

    model = optas.TaskModel(name, dim, dlim=dlim, T=T)
    assert model.state_name(0) == "test/y"
    assert model.state_parameter_name(0) == "test/y/p"
    assert model.T == T


class TestRobotModel:
    expected_joint_names = [
        "joint0",
        "joint1",
        "joint3",
        "eff_joint",
    ]

    expected_link_names = ["world", "link1", "link2", "link3", "eff"]

    cwd = pathlib.Path(__file__).parent.resolve()  # path to current working directory
    urdf_filename = "tester_robot.urdf"
    model = optas.RobotModel(urdf_filename=os.path.join(cwd, urdf_filename))

    def test_init(self):
        with pytest.raises(AssertionError):
            model = optas.RobotModel()

    def test_get_urdf(self):
        assert isinstance(self.model.get_urdf(), urdf.Robot)

    def test_get_urdf_dirname(self):
        urdf_dirname = self.model.get_urdf_dirname()
        assert isinstance(urdf_dirname, pathlib.Path)
        assert urdf_dirname == self.cwd

    def test_joint_names(self):
        for joint_name, expected_joint_name in zip(
            self.model.joint_names, self.expected_joint_names
        ):
            assert joint_name == expected_joint_name

    def test_link_names(self):
        for link_name, expected_link_name in zip(
            self.model.link_names, self.expected_link_names
        ):
            assert link_name == expected_link_name

    def test_actuated_joint_names(self):        
        for joint_name, expected_joint_name in zip(
            self.model.joint_names, self.expected_joint_names
        ):
            if expected_joint_name == 'eff_joint':
                # i.e. eff_joint is fixed
                continue
            assert joint_name == expected_joint_name

    

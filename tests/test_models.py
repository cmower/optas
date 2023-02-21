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
    # Setup path to tester robot URDF
    cwd = pathlib.Path(__file__).parent.resolve()  # path to current working directory
    urdf_filename = "tester_robot.urdf"
    urdf_path = os.path.join(cwd, urdf_filename)

    # Setup model (with no parameterized joints)
    model = optas.RobotModel(urdf_filename=urdf_path)

    model_expected_joint_names = [
        "joint0",
        "joint1",
        "joint2",
        "eff_joint",
    ]
    model_expected_link_names = ["world", "link1", "link2", "link3", "eff"]
    model_expected_actuated_joint_names = ["joint0", "joint1", "joint2"]
    model_expected_parameter_joint_names = []
    model_expected_optimized_joint_indexes = [0, 1, 2]
    model_expected_optimized_joint_names = ["joint0", "joint1", "joint2"]
    model_expected_parameter_joint_indexes = []

    model_expected_lower_actuated_joint_limits = [-1e9, -1, 0]
    model_expected_upper_actuated_joint_limits = [1e9, 1, 1]
    model_expected_velocity_actuated_joint_limits = [1e9, 1, 1]
    model_expected_lower_optimized_joint_limits = [-1e9, -1, 0]
    model_expected_upper_optimized_joint_limits = [1e9, 1, 1]
    model_expected_velocity_optimized_joint_limits = [1e9, 1, 1]

    # Setup model (with parameterized joints)
    model_p = optas.RobotModel(
        urdf_filename=os.path.join(cwd, urdf_filename),
        param_joints=["joint0"],
    )

    model_p_expected_joint_names = [
        "joint0",
        "joint1",
        "joint2",
        "eff_joint",
    ]
    model_p_expected_link_names = ["world", "link1", "link2", "link3", "eff"]
    model_p_expected_actuated_joint_names = ["joint0", "joint1", "joint2"]
    model_p_expected_parameter_joint_names = ["joint0"]
    model_p_expected_optimized_joint_indexes = [1, 2]
    model_p_expected_optimized_joint_names = ["joint1", "joint2"]
    model_p_expected_parameter_joint_indexes = [0]

    model_p_expected_lower_actuated_joint_limits = [-1e9, -1, 0]
    model_p_expected_upper_actuated_joint_limits = [1e9, 1, 1]
    model_p_expected_velocity_actuated_joint_limits = [1e9, 1, 1]
    model_p_expected_lower_optimized_joint_limits = [-1, 0]
    model_p_expected_upper_optimized_joint_limits = [1, 1]
    model_p_expected_velocity_optimized_joint_limits = [1, 1]

    def test_init(self):
        with pytest.raises(AssertionError):
            model = optas.RobotModel()

    def test_get_urdf(self):
        assert isinstance(self.model.get_urdf(), urdf.Robot)

    def test_get_urdf_dirname(self):
        urdf_dirname = self.model.get_urdf_dirname()
        assert isinstance(urdf_dirname, pathlib.Path)
        assert urdf_dirname == self.cwd

    def check_items(self, items_label):
        for model_name in ("model", "model_p"):
            model = getattr(self, model_name)
            items = getattr(model, items_label)

            expected_items_label = model_name + "_expected_" + items_label
            expected_items = getattr(self, expected_items_label)

            if isinstance(items, list):
                assert len(items) == len(expected_items)
                for item, expected_item in zip(items, expected_items):
                    assert item == expected_item
            elif isinstance(items, optas.DM):
                assert items.shape[0] == len(expected_items)
                for i in range(len(expected_items)):
                    assert isclose(items[i].toarray().flatten(), expected_items[i])

    def test_joint_names(self):
        self.check_items("joint_names")

    def test_link_names(self):
        self.check_items("link_names")

    def test_actuated_joint_names(self):
        self.check_items("actuated_joint_names")

    def test_parameter_joint_names(self):
        self.check_items("parameter_joint_names")

    def test_optimized_joint_indexes(self):
        self.check_items("optimized_joint_indexes")

    def test_optimized_joint_names(self):
        self.check_items("optimized_joint_names")

    def test_parameter_joint_indexes(self):
        self.check_items("parameter_joint_indexes")

    def test_extract_parameter_dimensions(self):
        q = optas.DM([1, 2, 3])
        qp = self.model_p.extract_parameter_dimensions(q)
        assert isclose(q[0].toarray().flatten(), qp.toarray().flatten())

    def test_extract_optimized_dimensions(self):
        q = optas.DM([1, 2, 3])
        qopt = self.model_p.extract_optimized_dimensions(q)
        assert isclose(q[1:].toarray().flatten(), qopt.toarray().flatten())

    def check_num(self, label, expected_nums):
        for model_name, expected_num in zip(("model", "model_p"), expected_nums):
            model = getattr(self, model_name)
            assert getattr(model, label) == expected_num

    def test_ndof(self):
        self.check_num("ndof", [3, 3])

    def test_num_opt_joints(self):
        self.check_num("num_opt_joints", [3, 2])

    def test_num_param_joints(self):
        self.check_num("num_param_joints", [0, 1])

    def test_get_joint_lower_limit(self):
        joint_null = urdf.Joint()
        assert isclose(self.model.get_joint_lower_limit(joint_null), -1e9)

        expected_limits = (-1e9, -1, 0, -1e9)
        for joint, expected_limit in zip(self.model.get_urdf().joints, expected_limits):
            limit = self.model.get_joint_lower_limit(joint)
            assert isclose(limit, expected_limit)

    def test_get_joint_lower_limit(self):
        joint_null = urdf.Joint()
        assert isclose(self.model.get_joint_upper_limit(joint_null), 1e9)

        expected_limits = (1e9, 1, 1, 1e9)
        for joint, expected_limit in zip(self.model.get_urdf().joints, expected_limits):
            limit = self.model.get_joint_upper_limit(joint)
            assert isclose(limit, expected_limit)

    def test_get_velocity_joint_limit(self):
        joint_null = urdf.Joint()
        assert isclose(self.model.get_velocity_joint_limit(joint_null), 1e9)

        expected_limits = (1e9, 1, 1, 1e9)
        for joint, expected_limit in zip(self.model.get_urdf().joints, expected_limits):
            limit = self.model.get_velocity_joint_limit(joint)
            assert isclose(limit, expected_limit)

    def test_lower_actuated_joint_limits(self):
        self.check_items("lower_actuated_joint_limits")

    def test_upper_actuated_joint_limits(self):
        self.check_items("upper_actuated_joint_limits")

    def test_velocity_actuated_joint_limits(self):
        self.check_items("velocity_actuated_joint_limits")

    def test_lower_optimized_joint_limits(self):
        self.check_items("lower_optimized_joint_limits")

    def test_upper_optimized_joint_limits(self):
        self.check_items("upper_optimized_joint_limits")

    def test_velocity_optimized_joint_limits(self):
        self.check_items("velocity_optimized_joint_limits")

    def test_add_base_frame(self):
        model = optas.RobotModel(urdf_filename=self.urdf_path)
        assert model.get_root_link() == 'world'
        
        model.add_base_frame(
            "test_world",
            xyz=[1, 2, 3],
            rpy=[0, 0, 0.5 * np.pi],
            joint_name="test_joint_name",
        )
        assert model.get_root_link() == 'test_world'

    def test_

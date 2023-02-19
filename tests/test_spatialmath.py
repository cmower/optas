import optas
import functools
import numpy as np
from scipy.spatial.transform import Rotation as Rot

NUM_RANDOM = 100

###########################################################################
# Helper methods
#


def random_angle():
    return np.random.uniform(-np.pi, np.pi)


def random_vector(lo=-1, hi=1, n=3):
    return np.random.uniform(-lo, hi, size=(3,))


def random_rotation_matrix():
    return Rot.random().as_matrix()


def normalize(v):
    return v / np.linalg.norm(v)


def isclose(A: np.ndarray, B: np.ndarray):
    return np.isclose(A, B).all()


###########################################################################
# Tests
#


def _test_method(a, b, c="test", d=[1, 2, 3], e=(3, 2, 1), f=1, g=0.5):
    Test_arrayify_args._assert_correct_type(a)
    Test_arrayify_args._assert_correct_type(b)
    Test_arrayify_args._assert_correct_type(c, correct_type=str)
    Test_arrayify_args._assert_correct_type(d)
    Test_arrayify_args._assert_correct_type(e)
    Test_arrayify_args._assert_correct_type(f)
    Test_arrayify_args._assert_correct_type(g)


class Test_arrayify_args:
    @staticmethod
    def _assert_correct_type(value, correct_type=(optas.SX, optas.DM)):
        assert isinstance(
            value, correct_type
        ), f"got {type(value)=}, expected {correct_type}"

    def _test_cls_method(self, a, b, c="test", d=[1, 2, 3], e=(3, 2, 1), f=1, g=0.5):
        self._assert_correct_type(a)
        self._assert_correct_type(b)
        self._assert_correct_type(c, correct_type=str)
        self._assert_correct_type(d)
        self._assert_correct_type(e)
        self._assert_correct_type(f)
        self._assert_correct_type(g)

    def test_cls_method(self):
        decorated_method = optas.arrayify_args(self._test_cls_method)
        decorated_method(optas.DM([1, 2]), np.array([3, 4]))

    def test_method(self):
        decorated_method = optas.arrayify_args(_test_method)
        decorated_method(optas.DM([1, 2]), np.array([3, 4]))


class Test_I3:
    def test_output_type(self):
        assert isinstance(optas.I3(), optas.DM)

    def test_correct_output(self):
        assert isclose(optas.I3().toarray(), np.eye(3))


class Test_I4:
    def test_output_type(self):
        assert isinstance(optas.I4(), optas.DM)

    def test_correct_output(self):
        assert isclose(optas.I4().toarray(), np.eye(4))


class Test_angvec2r:
    def _random_theta(self):
        return random_angle()

    def _random_v(self):
        return random_vector()

    @staticmethod
    def _optas_result(theta, v):
        return optas.angvec2r(theta, v)

    @staticmethod
    def _lib_result(theta, v):
        vn = normalize(v)
        return Rot.from_rotvec(vn * theta).as_matrix()

    def test_symbolic_output(self):
        theta = optas.SX.sym("theta")
        v = optas.SX.sym("v", 3)
        output = self._optas_result(theta, v)
        assert isinstance(output, optas.SX)

    def test_numerical_output(self):
        for _ in range(NUM_RANDOM):
            theta = self._random_theta()
            v = self._random_v()
            result = self._optas_result(theta, v)
            assert isinstance(result, optas.DM)

    def test_against_external_lib(self):
        for _ in range(NUM_RANDOM):
            theta = self._random_theta()
            v = self._random_v()
            optas_result = self._optas_result(theta, v)
            lib_result = self._lib_result(theta, v)
            assert isclose(optas_result.toarray(), lib_result)


class Test_r2t:
    def _random_R(self):
        return random_rotation_matrix()

    def test_numerical_output(self):
        for _ in range(NUM_RANDOM):
            R = self._random_R()
            T = optas.r2t(R)
            assert isinstance(T, optas.DM)

    def test_correct_numerical_output(self):
        for _ in range(NUM_RANDOM):
            R = self._random_R()
            T = optas.r2t(R)
            assert isclose(T[:3, :3].toarray(), R)
            assert isclose(T[:3, 3].toarray().flatten(), np.zeros(3))
            assert isclose(T[3, :3].toarray().flatten(), np.zeros(3))
            assert np.isclose(T[3, 3].toarray(), 1)

    def test_symbolic_output(self):
        R = optas.SX.sym("R", 3, 3)
        assert isinstance(optas.r2t(R), optas.SX)


class _Test_rot:
    _optas_result = None  # must call staticmethod on method handle

    def _random_theta(self):
        return random_angle()

    @staticmethod
    def _lib_result(theta, dim):
        return Rot.from_euler(dim, theta).as_matrix()

    def test_numerical_output(self):
        for _ in range(NUM_RANDOM):
            theta = self._random_theta()
            R = self._optas_result(theta)
            assert isinstance(R, optas.DM)

    def test_against_external_lib(self):
        dim = self._optas_result.__name__[-1]
        for _ in range(NUM_RANDOM):
            theta = self._random_theta()
            _optas_result = self._optas_result(theta)
            _lib_result = self._lib_result(theta, dim)
            assert isclose(_optas_result.toarray(), _lib_result)

    def test_symbolic_output(self):
        theta = optas.SX.sym("theta")
        R = self._optas_result(theta)
        assert isinstance(R, optas.SX)


class Test_rotx(_Test_rot):
    _optas_result = staticmethod(optas.rotx)


class Test_roty(_Test_rot):
    _optas_result = staticmethod(optas.roty)


class Test_rotz(_Test_rot):
    _optas_result = staticmethod(optas.rotz)


class Test_rpy2r:
    pass


class Test_rt2tr:
    pass


class Test_skew:
    pass


class Test_t2r:
    pass


class Test_invt:
    def _random_T(self):
        T = optas.DM.eye(4)
        T[:3, :3] = random_rotation_matrix()
        T[:3, 3] = random_vector()
        return T

    @staticmethod
    def _optas_result(T):
        return optas.invt(T)

    @staticmethod
    def _lib_result(T):
        return np.linalg.inv(T)

    def test_numerical_output(self):
        for _ in range(NUM_RANDOM):
            T = self._random_T()
            assert isinstance(self._optas_result(T), optas.DM)

    def test_against_external_lib(self):
        for _ in range(NUM_RANDOM):
            T = self._random_T()
            _optas_result = self._optas_result(T)
            _lib_result = self._lib_result(T)
            assert isclose(_optas_result.toarray(), _lib_result)

    def test_symbolic_output(self):
        T = optas.SX.sym("T", 4, 4)
        assert isinstance(self._optas_result(T), optas.SX)


class Test_transl:
    def _random_T(self):
        T = optas.DM.eye(4)
        T[:3, :3] = random_rotation_matrix()
        T[:3, 3] = random_vector()
        return T

    def test_numerical_output(self):
        for _ in range(NUM_RANDOM):
            T = self._random_T()
            assert isinstance(optas.transl(T), optas.DM)

    def test_correct_output(self):
        for _ in range(NUM_RANDOM):
            T = self._random_T()
            t = optas.transl(T)
            assert isclose(t.toarray().flatten(), T[:3, 3].toarray().flatten())

    def test_symbolic_output(self):
        T = optas.SX.sym("T", 4, 4)
        assert isinstance(optas.transl(T), optas.SX)


class Test_unit:
    pass


class Test_Quaternion:
    pass

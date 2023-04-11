import pytest
import optas
import numpy as np
from scipy.spatial.transform import Rotation as Rot

NUM_RANDOM = 100

###########################################################################
# Helper methods
#


def random_number(lo=-1, hi=1):
    return np.random.uniform(lo, hi)


def random_angle():
    return random_number(lo=-np.pi, hi=np.pi)


def random_vector(lo=-1, hi=1, n=3):
    return np.random.uniform(-lo, hi, size=(n,))


def random_rotation_matrix():
    return Rot.random().as_matrix()


def random_T():
    T = optas.DM.eye(4)
    T[:3, :3] = random_rotation_matrix()
    T[:3, 3] = random_vector()
    return T


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
        ), f"got type(value)={type(value)}, expected {correct_type}"

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
            theta = random_angle()
            v = random_vector()
            result = self._optas_result(theta, v)
            assert isinstance(result, optas.DM)

    def test_against_external_lib(self):
        for _ in range(NUM_RANDOM):
            theta = random_angle()
            v = random_vector()
            optas_result = self._optas_result(theta, v)
            lib_result = self._lib_result(theta, v)
            assert isclose(optas_result.toarray(), lib_result)


class Test_r2t:
    def test_numerical_output(self):
        for _ in range(NUM_RANDOM):
            R = random_rotation_matrix()
            T = optas.r2t(R)
            assert isinstance(T, optas.DM)

    def test_correct_numerical_output(self):
        for _ in range(NUM_RANDOM):
            R = random_rotation_matrix()
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

    @staticmethod
    def _lib_result(theta, dim):
        return Rot.from_euler(dim, theta).as_matrix()

    def test_numerical_output(self):
        for _ in range(NUM_RANDOM):
            theta = random_angle()
            R = self._optas_result(theta)
            assert isinstance(R, optas.DM)

    def test_against_external_lib(self):
        dim = self._optas_result.__name__[-1]
        for _ in range(NUM_RANDOM):
            theta = random_angle()
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
    orders = ["zyx", "xyz", "yxz", "arm", "vehicle", "camera"]

    @staticmethod
    def _optas_result(rpy, opt):
        return optas.rpy2r(rpy, opt=opt)

    @staticmethod
    def _lib_result(rpy, opt):
        if opt == "arm":
            opt_ = "xyz"
        elif opt == "vehicle":
            opt_ = "zyx"
        elif opt == "camera":
            opt_ = "yxz"
        else:
            opt_ = opt
        return Rot.from_euler(opt_, rpy).as_matrix()

    def test_numerical_output(self):
        for _ in range(NUM_RANDOM):
            for opt in self.orders:
                rpy = random_vector(lo=-np.pi, hi=np.pi)
                assert isinstance(self._optas_result(rpy, opt), optas.DM)

    def test_correct_output(self):
        for _ in range(NUM_RANDOM):
            for opt in self.orders:
                rpy = random_vector(lo=-np.pi, hi=np.pi)
                _optas_result = self._optas_result(rpy, opt)
                _lib_result = self._lib_result(rpy, opt)
                assert isclose(_optas_result.toarray(), _lib_result)

    def test_symbolic_output(self):
        for opt in self.orders:
            rpy = optas.SX.sym("rpy", 3)
            assert isinstance(rpy, optas.SX)


class Test_rt2tr:
    def test_numerical_output(self):
        for _ in range(NUM_RANDOM):
            R = random_rotation_matrix()
            t = random_vector()
            T = optas.rt2tr(R, t)
            assert isinstance(T, optas.DM)

    def test_correct_output(self):
        for _ in range(NUM_RANDOM):
            R = random_rotation_matrix()
            t = random_vector()
            T = optas.rt2tr(R, t)
            assert isclose(T[:3, :3].toarray(), R)
            assert isclose(T[:3, 3].toarray(), t)
            assert isclose(T[3, :].toarray(), [0, 0, 0, 1])

    def test_symbolic_output(self):
        R = optas.SX.sym("R", 3, 3)
        t = optas.SX.sym("t", 3)
        T = optas.rt2tr(R, t)
        assert isinstance(T, optas.SX)


class Test_skew:
    def test_numerical_output(self):
        for _ in range(NUM_RANDOM):
            v = random_vector()
            assert isinstance(optas.skew(v), optas.DM)

    def test_skew_symmetry(self):
        for _ in range(NUM_RANDOM):
            v = random_number()
            S = optas.skew(v)
            assert isclose(S, -S.T)

            v = random_vector()
            S = optas.skew(v)
            assert isclose(S, -S.T)

    def test_error_raised(self):
        for _ in range(NUM_RANDOM):
            v = random_vector(n=np.random.randint(4, 100))
            with pytest.raises(ValueError):
                S = optas.skew(v)

    def test_correct_output(self):
        for _ in range(NUM_RANDOM):
            v1 = random_vector()
            v2 = random_vector()
            S = optas.skew(v1)
            A = S.toarray() @ v2
            B = np.cross(v1, v2)
            assert isclose(A, B)

    def test_symbolic_output(self):
        v = optas.SX.sym("v", 1)
        assert isinstance(optas.skew(v), optas.SX)

        v = optas.SX.sym("v", 3)
        assert isinstance(optas.skew(v), optas.SX)


class Test_t2r:
    def test_numerical_output(self):
        for _ in range(NUM_RANDOM):
            T = random_T()
            R = optas.t2r(T)
            assert isinstance(R, optas.DM)

    def test_correct_numerical_output(self):
        for _ in range(NUM_RANDOM):
            T = random_T()
            R = optas.t2r(T)
            assert isclose(T[:3, :3], R.toarray())

    def test_symbolic_output(self):
        T = optas.SX.sym("T", 4, 4)
        assert isinstance(optas.t2r(T), optas.SX)


class Test_invt:
    @staticmethod
    def _optas_result(T):
        return optas.invt(T)

    @staticmethod
    def _lib_result(T):
        return np.linalg.inv(T)

    def test_numerical_output(self):
        for _ in range(NUM_RANDOM):
            T = random_T()
            assert isinstance(self._optas_result(T), optas.DM)

    def test_against_external_lib(self):
        for _ in range(NUM_RANDOM):
            T = random_T()
            _optas_result = self._optas_result(T)
            _lib_result = self._lib_result(T)
            assert isclose(_optas_result.toarray(), _lib_result)

    def test_symbolic_output(self):
        T = optas.SX.sym("T", 4, 4)
        assert isinstance(self._optas_result(T), optas.SX)


class Test_transl:
    def test_numerical_output(self):
        for _ in range(NUM_RANDOM):
            T = random_T()
            assert isinstance(optas.transl(T), optas.DM)

    def test_correct_output(self):
        for _ in range(NUM_RANDOM):
            T = random_T()
            t = optas.transl(T)
            assert isclose(t.toarray(), T[:3, 3].toarray())

    def test_symbolic_output(self):
        T = optas.SX.sym("T", 4, 4)
        assert isinstance(optas.transl(T), optas.SX)


class Test_unit:
    @staticmethod
    def _optas_result(v):
        return optas.unit(v)

    @staticmethod
    def _lib_result(v):
        return v / np.linalg.norm(v)

    def test_numerical_output(self):
        for _ in range(NUM_RANDOM):
            v = random_vector()
            assert isinstance(self._optas_result(v), optas.DM)

    def test_correct_output(self):
        for _ in range(NUM_RANDOM):
            v = random_vector()
            _optas_result = self._optas_result(v)
            _lib_result = self._lib_result(v)
            assert isclose(_optas_result, _lib_result)

    def test_symbolic_output(self):
        v = optas.SX.sym("v", 3)
        assert isinstance(self._optas_result(v), optas.SX)


class Test_Quaternion:
    def _random_quat(self):
        return Rot.random().as_quat()

    def _random_quaternion(self):
        qr = self._random_quat()
        return optas.Quaternion(qr[0], qr[1], qr[2], qr[3])

    def _symbolic_quat(self):
        return optas.SX.sym("q", 4)

    def _symbolic_quaternion(self):
        q = self._symbolic_quat()
        return optas.Quaternion(q[0], q[1], q[2], q[3])

    # split

    def test_split(self):
        for _ in range(NUM_RANDOM):
            qr = self._random_quat()
            quat = optas.Quaternion(qr[0], qr[1], qr[2], qr[3])
            qr_optas = np.array(quat.split()).flatten()
            assert isclose(qr, qr_optas)

    # __mul__

    def test_mul_numerical_output(self):
        for _ in range(NUM_RANDOM):
            quat0 = self._random_quaternion()
            quat1 = self._random_quaternion()
            quat = quat0 * quat1
            assert isinstance(quat, optas.Quaternion)
            assert isinstance(quat.getquat(), optas.DM)

    def test_mul_correct_output(self):
        for _ in range(NUM_RANDOM):
            quat0 = self._random_quaternion()
            quat1 = self._random_quaternion()
            quat_optas = quat0 * quat1

            q0 = quat0.getquat().toarray().flatten()
            q1 = quat1.getquat().toarray().flatten()
            quat_lib = (Rot.from_quat(q1) * Rot.from_quat(q0)).as_quat()

            assert isclose(quat_optas.getquat().toarray().flatten(), quat_lib)

    def test_mul_symbolic_output(self):
        quat0 = self._symbolic_quaternion()
        quat1 = self._symbolic_quaternion()

        quat = quat0 * quat1
        assert isinstance(quat, optas.Quaternion)
        assert isinstance(quat.getquat(), optas.SX)

    # sumsqr

    def test_sumsqr_numerical_output(self):
        for _ in range(NUM_RANDOM):
            assert isinstance(self._random_quaternion().sumsqr(), optas.DM)

    def test_sumsqr_correct_output(self):
        for _ in range(NUM_RANDOM):
            quat = self._random_quaternion()
            optas_result = quat.sumsqr().toarray().flatten()
            lib_result = np.linalg.norm(quat.getquat().toarray().flatten()) ** 2
            assert isclose(optas_result, lib_result)

    def test_sumsqr_symbolic_output(self):
        assert isinstance(self._symbolic_quaternion().sumsqr(), optas.SX)

    # inv

    def test_inv_numerical_output(self):
        for _ in range(NUM_RANDOM):
            quat = self._random_quaternion()
            quat_inv = quat.inv()
            assert isinstance(quat_inv, optas.Quaternion)
            assert isinstance(quat_inv.getquat(), optas.DM)

    def test_inv_correct_output(self):
        for _ in range(NUM_RANDOM):
            quat = self._random_quaternion()
            quat_inv = quat.inv()
            I = Rot.from_quat(
                (quat * quat_inv).getquat().toarray().flatten()
            ).as_matrix()
            assert isclose(I, np.eye(3))

    def test_inv_symbolic_output(self):
        quat = self._symbolic_quaternion()
        quat_inv = quat.inv()
        assert isinstance(quat_inv, optas.Quaternion)
        assert isinstance(quat_inv.getquat(), optas.SX)

    # fromrpy

    def test_fromrpy_numerical_output(self):
        for _ in range(NUM_RANDOM):
            rpy = random_vector(lo=-np.pi, hi=np.pi)
            quat = optas.Quaternion.fromrpy(rpy)
            assert isinstance(quat, optas.Quaternion)
            assert isinstance(quat.getquat(), optas.DM)

    def test_fromrpy_correct_output(self):
        for _ in range(NUM_RANDOM):
            rpy = random_vector(lo=-np.pi, hi=np.pi)
            quat_optas = optas.Quaternion.fromrpy(rpy).getquat().toarray().flatten()
            quat_lib = Rot.from_euler("xyz", rpy).as_quat()
            assert isclose(quat_optas, quat_lib)

    def test_fromrpy_symbolic_output(self):
        rpy = optas.SX.sym("rpy", 3)
        quat = optas.Quaternion.fromrpy(rpy)
        assert isinstance(quat, optas.Quaternion)
        assert isinstance(quat.getquat(), optas.SX)

    # fromangvec

    def test_fromangvec_numerical_output(self):
        for _ in range(NUM_RANDOM):
            theta = random_angle()
            v = random_vector()
            quat = optas.Quaternion.fromangvec(theta, v)
            assert isinstance(quat, optas.Quaternion)
            assert isinstance(quat.getquat(), optas.DM)

    def test_fromangvec_correct_output(self):
        for _ in range(NUM_RANDOM):
            theta = random_angle()
            v = random_vector()
            quat = optas.Quaternion.fromangvec(theta, v)
            quat_optas = quat.getquat().toarray().flatten()

            vn = normalize(v)
            quat_lib = Rot.from_rotvec(vn * theta).as_quat()
            assert isclose(quat_optas, quat_lib)

    def test_fromangvec_symbolic_output(self):
        theta = optas.SX.sym("theta")
        v = optas.SX.sym("v", 3)
        quat = optas.Quaternion.fromangvec(theta, v)
        assert isinstance(quat, optas.Quaternion)
        assert isinstance(quat.getquat(), optas.SX)

    # getquat

    def test_getquat_numerical_output(self):
        for _ in range(NUM_RANDOM):
            assert isinstance(self._random_quaternion().getquat(), optas.DM)

    def test_getquat_correct_output(self):
        for _ in range(NUM_RANDOM):
            qr = self._random_quat()
            quat = optas.Quaternion(qr[0], qr[1], qr[2], qr[3])
            assert isclose(quat.getquat().toarray().flatten(), qr)

    def test_getquat_symbolic_output(self):
        quat = self._symbolic_quaternion()
        assert isinstance(quat.getquat(), optas.SX)

    # getrpy

    def test_getrpy_numerical_output(self):
        for _ in range(NUM_RANDOM):
            quat = self._random_quaternion()
            assert isinstance(quat.getrpy(), optas.DM)

    def test_getrpy_correct_output(self):
        for _ in range(NUM_RANDOM):
            quat = self._random_quaternion()
            result_optas = quat.getrpy().toarray().flatten()
            result_lib = Rot.from_quat(quat.getquat().toarray().flatten()).as_euler(
                "xyz"
            )
            assert isclose(result_optas, result_lib)

    def test_getrpy_symbolic_output(self):
        quat = self._symbolic_quaternion()
        assert isinstance(quat.getrpy(), optas.SX)

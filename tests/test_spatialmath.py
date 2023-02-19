import optas
import functools
import casadi as cs
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
# Test cases
#

# def test_arrayify_args():
#     pass


def test_I3():
    assert isinstance(optas.I3(), cs.DM)
    assert isclose(optas.I3().toarray(), np.eye(3))


def test_I4():
    assert isinstance(optas.I4(), cs.DM)
    assert isclose(optas.I4().toarray(), np.eye(4))


def test_angvec2r():
    for _ in range(NUM_RANDOM):
        theta = random_angle()
        v = random_vector()

        optas_result = optas.angvec2r(theta, v)

        vn = normalize(v)
        numpy_result = Rot.from_rotvec(vn * theta).as_matrix()

        assert isinstance(optas_result, cs.DM)
        assert isclose(optas_result.toarray(), numpy_result)

    theta = cs.SX.sym("theta")
    v = cs.SX.sym("v", 3)
    assert isinstance(optas.angvec2r(theta, v), cs.SX)


def test_r2t():
    for _ in range(NUM_RANDOM):
        R = random_rotation_matrix()
        T = optas.r2t(R)

        assert isinstance(T, cs.DM)

        T = T.toarray()

        assert isclose(T[:3, :3], R)
        assert isclose(T[:3, 3].flatten(), np.zeros(3))
        assert isclose(T[3, :3].flatten(), np.zeros(3))
        assert np.isclose(T[3, 3], 1)

    R = cs.SX.sym("R", 3, 3)
    assert isinstance(optas.r2t(R), cs.SX)


def test_rotx():
    theta = cs.SX.sym("theta")
    assert isinstance(optas.rotx(theta), cs.SX)

import optas
import numpy as np

NUM_RANDOM = 100

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


class _Test_angle_conv:
    _optas_result = None  # must call staticmethod on method handle
    _lib_result = None  # must call staticmethod on method handle

    def test_numerical_output(self):
        for _ in range(NUM_RANDOM):
            n = np.random.randint(3, 100)
            x = random_vector(n=n)
            assert isinstance(self._optas_result(x), optas.DM)

    def test_correct_output(self):
        for _ in range(NUM_RANDOM):
            n = np.random.randint(3, 100)
            x = random_vector(n=n)
            result_optas = self._optas_result(x)
            result_lib = self._lib_result(x)
            assert isclose(result_optas.toarray().flatten(), result_lib)

    def test_symbolic_output(self):
        for _ in range(NUM_RANDOM):
            n = np.random.randint(3, 100)
            x = optas.SX.sym("x", n)
            assert isinstance(self._optas_result(x), optas.SX)


class Test_deg2rad(_Test_angle_conv):
    _optas_result = staticmethod(optas.deg2rad)
    _lib_result = staticmethod(np.deg2rad)


class Test_rad2deg(_Test_angle_conv):
    _optas_result = staticmethod(optas.rad2deg)
    _lib_result = staticmethod(np.rad2deg)


class Test_clip:
    def test_numerical_output_scalar_bounds(self):
        for _ in range(NUM_RANDOM):
            n = np.random.randint(1, 10)
            x = random_vector(lo=0, hi=3, n=n)
            lo = np.random.uniform(0, 1)
            hi = np.random.uniform(2, 3)
            assert isinstance(optas.clip(x, lo, hi), optas.DM)

    def test_numerical_output_scalar_lower_bound_vector_upper_bound(self):
        for _ in range(NUM_RANDOM):
            n = np.random.randint(1, 10)
            x = random_vector(lo=0, hi=3, n=n)
            lo = np.random.uniform(0, 1)
            hi = np.random.uniform(2, 3, size=(n,))
            assert isinstance(optas.clip(x, lo, hi), optas.DM)

    def test_numerical_output_vector_lower_bound_scalar_upper_bound(self):
        for _ in range(NUM_RANDOM):
            n = np.random.randint(1, 10)
            x = random_vector(lo=0, hi=3, n=n)
            lo = np.random.uniform(0, 1, size=(n,))
            hi = np.random.uniform(2, 3)
            assert isinstance(optas.clip(x, lo, hi), optas.DM)

    def test_numerical_output_vector_bounds(self):
        for _ in range(NUM_RANDOM):
            n = np.random.randint(1, 10)
            x = random_vector(lo=0, hi=3, n=n)
            lo = np.random.uniform(0, 1, size=(n,))
            hi = np.random.uniform(2, 3, size=(n,))
            assert isinstance(optas.clip(x, lo, hi), optas.DM)

    def test_correct_output_scalar_bounds(self):
        for _ in range(NUM_RANDOM):
            n = np.random.randint(1, 10)
            x = random_vector(lo=0, hi=3, n=n)
            lo = np.random.uniform(0, 1)
            hi = np.random.uniform(2, 3)
            assert isclose(
                np.clip(x, lo, hi), optas.clip(x, lo, hi).toarray().flatten()
            )

    def test_correct_output_scalar_lower_bound_vector_upper_bound(self):
        for _ in range(NUM_RANDOM):
            n = np.random.randint(1, 10)
            x = random_vector(lo=0, hi=3, n=n)
            lo = np.random.uniform(0, 1)
            hi = np.random.uniform(2, 3, size=(n,))
            assert isclose(
                np.clip(x, lo, hi), optas.clip(x, lo, hi).toarray().flatten()
            )

    def test_correct_output_vector_lower_bound_scalar_upper_bound(self):
        for _ in range(NUM_RANDOM):
            n = np.random.randint(1, 10)
            x = random_vector(lo=0, hi=3, n=n)
            lo = np.random.uniform(0, 1, size=(n,))
            hi = np.random.uniform(2, 3)
            assert isclose(
                np.clip(x, lo, hi), optas.clip(x, lo, hi).toarray().flatten()
            )

    def test_correct_output_vector_bounds(self):
        for _ in range(NUM_RANDOM):
            n = np.random.randint(1, 10)
            x = random_vector(lo=0, hi=3, n=n)
            lo = np.random.uniform(0, 1, size=(n,))
            hi = np.random.uniform(2, 3, size=(n,))
            assert isclose(
                np.clip(x, lo, hi), optas.clip(x, lo, hi).toarray().flatten()
            )

    def test_symbolic_output_scalar_bounds(self):
        for _ in range(NUM_RANDOM):
            n = np.random.randint(1, 10)
            x = optas.SX.sym("x", n)
            lo = np.random.uniform(0, 1)
            hi = np.random.uniform(2, 3)
            assert isinstance(optas.clip(x, lo, hi), optas.SX)

    def test_symbolic_output_scalar_lower_bound_vector_upper_bound(self):
        for _ in range(NUM_RANDOM):
            n = np.random.randint(1, 10)
            x = optas.SX.sym("x", n)
            lo = np.random.uniform(0, 1)
            hi = np.random.uniform(2, 3, size=(n,))
            assert isinstance(optas.clip(x, lo, hi), optas.SX)

    def test_symbolic_output_vector_lower_bound_scalar_upper_bound(self):
        for _ in range(NUM_RANDOM):
            n = np.random.randint(1, 10)
            x = optas.SX.sym("x", n)
            lo = np.random.uniform(0, 1, size=(n,))
            hi = np.random.uniform(2, 3)
            assert isinstance(optas.clip(x, lo, hi), optas.SX)

    def test_symbolic_output_vector_bounds(self):
        for _ in range(NUM_RANDOM):
            n = np.random.randint(1, 10)
            x = optas.SX.sym("x", n)
            lo = np.random.uniform(0, 1, size=(n,))
            hi = np.random.uniform(2, 3, size=(n,))
            assert isinstance(optas.clip(x, lo, hi), optas.SX)

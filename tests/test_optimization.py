import optas
import numpy as np

NUM_RANDOM = 100

###########################################################################
# Helper methods
#


def isclose(A: np.ndarray, B: np.ndarray):
    return np.isclose(A, B).all()


###########################################################################
# Tests
#


def test_derive_jacobian_and_hessian_functions():
    name = "test"
    x = optas.SX.sym("x", 2)
    p = optas.SX.sym("p", 2)

    f = p[0] * x[0] ** 2 + p[1] * x[1] ** 3
    fun = optas.Function("fun", [x, p], [f])

    Jac, Hes = optas.optimization._derive_jacobian_and_hessian_functions(
        name, fun, x, p
    )

    jac_known = optas.horzcat(2.0 * p[0] * x[0], 3.0 * p[1] * x[1] ** 2)
    Jac_known = optas.Function("test_jac_known", [x, p], [jac_known])

    hes_known = optas.diag(optas.vertcat(2.0 * p[0], 6.0 * p[1] * x[1]))
    Hes_known = optas.Function("test_hes_known", [x, p], [hes_known])

    for _ in range(NUM_RANDOM):
        x = np.random.uniform(-10, 10, size=(2,))
        p = np.random.uniform(-10, 10, size=(2,))
        assert isclose(Jac(x, p).toarray(), Jac_known(x, p).toarray())
        assert isclose(Hes(x, p).toarray(), Hes_known(x, p).toarray())

def test_vertcon():
    pass

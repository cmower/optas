import optas
from optas.sx_container import SXContainer
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
    x = optas.SX.sym("x", 2)
    p = optas.SX.sym("p", 2)

    ineq = [
        optas.Function("i1", [x, p], [p[0] * x[0]]),
        optas.Function("i2", [x, p], [-p[1] * x[1] ** 2]),
    ]

    eq = [
        optas.Function("e", [x, p], [2 * x[0] - x[1]]),
    ]

    fun = optas.optimization._vertcon(x, p, ineq=ineq, eq=eq)

    f_known = optas.vertcat(
        p[0] * x[0],
        -p[1] * x[1] ** 2,
        2 * x[0] - x[1],
        -2 * x[0] + x[1],
    )
    fun_known = optas.Function("v_known", [x, p], [f_known])

    for _ in range(NUM_RANDOM):
        x = np.random.uniform(-10, 10, size=(2,))
        p = np.random.uniform(-10, 10, size=(2,))
        assert isclose(fun_known(x, p).toarray(), fun(x, p).toarray())


def test_optimization():
    dv = SXContainer()
    dv["x"] = optas.SX.sym("x", 2, 3)
    x = dv.vec()

    pr = SXContainer()
    pr["p"] = optas.SX.sym("p")
    p = pr.vec()

    ct = SXContainer()
    ct["c"] = p * optas.sumsqr(x)

    models = [
        optas.TaskModel("test_model_1", 2),
        optas.TaskModel("test_model_2", 3),
    ]

    opt = optas.optimization.Optimization(dv, pr, ct)
    opt.set_models(models)

    assert opt.models == models
    assert opt.nx == 6
    assert opt.np == 1

    assert opt.f([1, 2, 3, 4, 5, 6], [3]) == 273

def test_QuadraticCostUnconstrained():
    pass

def test_QuadraticCostLinearConstraints():
    pass

def test_QuadraticCostNonlinearConstraints():
    pass

def test_NonlinearCostUnconstrained():
    pass

def test_NonlinearCostLinearConstraints():
    pass

def test_NonlinearCostNonlinearConstraints():
    pass

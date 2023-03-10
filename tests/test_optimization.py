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

    Jac, Hes = optas.optimization.derive_jacobian_and_hessian_functions(
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

    fun = optas.optimization.vertcon(x, p, ineq=ineq, eq=eq)

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

    def fun_known(x, p):
        return p * np.sum(x.flatten() ** 2)

    for _ in range(NUM_RANDOM):
        x = np.random.uniform(-10, 10, size=(6,))
        p = np.random.uniform(-10, 10)
        assert isclose(opt.f(x, p), fun_known(x, p))

    attr_exp_value_map = {
        "nx": 6,
        "np": 1,
        "nk": 0,
        "na": 0,
        "ng": 0,
        "nh": 0,
        "nv": 0,
    }
    for attr, exp_value in attr_exp_value_map.items():
        assert getattr(opt, attr) == exp_value


def test_QuadraticCostUnconstrained():
    dv = SXContainer()
    dv["x"] = optas.SX.sym("x", 2, 3)
    x = dv.vec()

    pr = SXContainer()
    pr["p"] = optas.SX.sym("p")
    p = pr.vec()

    ct = SXContainer()
    ct["c"] = p * optas.sumsqr(x)

    opt = optas.optimization.QuadraticCostUnconstrained(dv, pr, ct)

    def fun_known(x, p):
        return p * np.sum(x.flatten() ** 2)

    for _ in range(NUM_RANDOM):
        x = np.random.uniform(-10, 10, size=(6,))
        p = np.random.uniform(-10, 10)
        assert isclose(opt.f(x, p), fun_known(x, p))

        P = opt.P(p).toarray()
        q = opt.q(p).toarray().flatten()

        f = x.T @ P @ x + np.dot(q, x)

        assert isclose(f, fun_known(x, p))

    attr_exp_value_map = {
        "nx": 6,
        "np": 1,
        "nk": 0,
        "na": 0,
        "ng": 0,
        "nh": 0,
        "nv": 0,
    }
    for attr, exp_value in attr_exp_value_map.items():
        assert getattr(opt, attr) == exp_value


def test_QuadraticCostLinearConstraints():
    dv = SXContainer()
    dv["x"] = optas.SX.sym("x", 2, 3)
    x = dv.vec()

    pr = SXContainer()
    pr["p"] = optas.SX.sym("p")
    p = pr.vec()

    ct = SXContainer()
    ct["c"] = p * optas.sumsqr(x)

    lec = SXContainer()
    lec["lec"] = x[0] - 2.0 * x[1]

    lic = SXContainer()
    lic["lic1"] = x[2] + 3.0 * x[4]
    lic["lic2"] = -2.0 * x[5] + x[2]

    opt = optas.optimization.QuadraticCostLinearConstraints(dv, pr, ct, lec, lic)

    def fun_known(x, p):
        return p * np.sum(x.flatten() ** 2)

    def k_known(x, p):
        return np.array([x[2] + 3.0 * x[4], -2.0 * x[5] + x[2]])

    def a_known(x, p):
        return np.array(
            [
                x[0] - 2.0 * x[1],
            ]
        )

    def v_known(x, p):
        a = a_known(x, p)
        return np.concatenate([k_known(x, p), a, -a])

    for _ in range(NUM_RANDOM):
        x = np.random.uniform(-10, 10, size=(6,))
        p = np.random.uniform(-10, 10)

        assert isclose(opt.f(x, p), fun_known(x, p))

        P = opt.P(p).toarray()
        q = opt.q(p).toarray().flatten()

        f = x.T @ P @ x + np.dot(q, x)

        assert isclose(f, fun_known(x, p))

        M = opt.M(p).toarray()
        c = opt.c(p).toarray()

        assert isclose(M @ x + c, k_known(x, p))

        A = opt.A(p).toarray()
        b = opt.b(p).toarray()

        assert isclose(A @ x + b, a_known(x, p))

        v = opt.v(x, p).toarray().flatten()
        assert isclose(v, v_known(x, p))

    attr_exp_value_map = {
        "nx": 6,
        "np": 1,
        "nk": 2,
        "na": 1,
        "ng": 0,
        "nh": 0,
        "nv": 4,
    }
    for attr, exp_value in attr_exp_value_map.items():
        assert getattr(opt, attr) == exp_value


def test_QuadraticCostNonlinearConstraints():
    dv = SXContainer()
    dv["x"] = optas.SX.sym("x", 2, 3)
    x = dv.vec()

    pr = SXContainer()
    pr["p"] = optas.SX.sym("p")
    p = pr.vec()

    ct = SXContainer()
    ct["c"] = p * optas.sumsqr(x)

    lec = SXContainer()
    lec["lec"] = x[0] - 2.0 * x[1]

    lic = SXContainer()
    lic["lic1"] = x[2] + 3.0 * x[4]
    lic["lic2"] = -2.0 * x[5] + x[2]

    ec = SXContainer()
    ec["e1"] = x[3] ** 2 - x[1] * x[2]
    ec["e2"] = x[0] - 2.0 * x[1] * x[4]

    ic = SXContainer()
    ic["ic"] = x[1] + x[2] * x[0]

    opt = optas.optimization.QuadraticCostNonlinearConstraints(
        dv, pr, ct, lec, lic, ec, ic
    )

    def fun_known(x, p):
        return p * np.sum(x.flatten() ** 2)

    def k_known(x, p):
        return np.array([x[2] + 3.0 * x[4], -2.0 * x[5] + x[2]])

    def a_known(x, p):
        return np.array(
            [
                x[0] - 2.0 * x[1],
            ]
        )

    def g_known(x, p):
        return np.array([x[1] + x[2] * x[0]])

    def h_known(x, p):
        return np.array(
            [
                x[3] ** 2 - x[1] * x[2],
                x[0] - 2.0 * x[1] * x[4],
            ]
        )

    def v_known(x, p):
        a = a_known(x, p)
        h = h_known(x, p)
        return np.concatenate([k_known(x, p), g_known(x, p), a, -a, h, -h])

    for _ in range(NUM_RANDOM):
        x = np.random.uniform(-10, 10, size=(6,))
        p = np.random.uniform(-10, 10)

        assert isclose(opt.f(x, p), fun_known(x, p))

        P = opt.P(p).toarray()
        q = opt.q(p).toarray().flatten()

        f = x.T @ P @ x + np.dot(q, x)

        assert isclose(f, fun_known(x, p))

        M = opt.M(p).toarray()
        c = opt.c(p).toarray()

        assert isclose(M @ x + c, k_known(x, p))

        A = opt.A(p).toarray()
        b = opt.b(p).toarray()

        assert isclose(A @ x + b, a_known(x, p))

        v = opt.v(x, p).toarray().flatten()
        assert isclose(v, v_known(x, p))

    attr_exp_value_map = {
        "nx": 6,
        "np": 1,
        "nk": 2,
        "na": 1,
        "ng": 1,
        "nh": 2,
        "nv": 9,
    }
    for attr, exp_value in attr_exp_value_map.items():
        assert getattr(opt, attr) == exp_value


def test_NonlinearCostUnconstrained():
    dv = SXContainer()
    dv["x"] = optas.SX.sym("x", 2, 3)
    x = dv.vec()

    pr = SXContainer()
    pr["p"] = optas.SX.sym("p")
    p = pr.vec()

    ct = SXContainer()
    ct["c"] = p * optas.sumsqr(x) + x[0] * x[1] + 2.0 * (optas.cos(x[3]) - 1.0) ** 2

    opt = optas.optimization.NonlinearCostUnconstrained(dv, pr, ct)

    def fun_known(x, p):
        return (
            p * np.sum(x.flatten() ** 2) + x[0] * x[1] + 2.0 * (np.cos(x[3]) - 1.0) ** 2
        )

    for _ in range(NUM_RANDOM):
        x = np.random.uniform(-10, 10, size=(6,))
        p = np.random.uniform(-10, 10)
        assert isclose(opt.f(x, p), fun_known(x, p))

    attr_exp_value_map = {
        "nx": 6,
        "np": 1,
        "nk": 0,
        "na": 0,
        "ng": 0,
        "nh": 0,
        "nv": 0,
    }
    for attr, exp_value in attr_exp_value_map.items():
        assert getattr(opt, attr) == exp_value


def test_NonlinearCostLinearConstraints():
    dv = SXContainer()
    dv["x"] = optas.SX.sym("x", 2, 3)
    x = dv.vec()

    pr = SXContainer()
    pr["p"] = optas.SX.sym("p")
    p = pr.vec()

    ct = SXContainer()
    ct["c"] = p * optas.sumsqr(x) + x[0] * x[1] + 2.0 * (optas.cos(x[3]) - 1.0) ** 2

    lec = SXContainer()
    lec["lec"] = x[0] - 2.0 * x[1]

    lic = SXContainer()
    lic["lic1"] = x[2] + 3.0 * x[4]
    lic["lic2"] = -2.0 * x[5] + x[2]

    opt = optas.optimization.NonlinearCostLinearConstraints(dv, pr, ct, lec, lic)

    def fun_known(x, p):
        return (
            p * np.sum(x.flatten() ** 2) + x[0] * x[1] + 2.0 * (np.cos(x[3]) - 1.0) ** 2
        )

    def k_known(x, p):
        return np.array([x[2] + 3.0 * x[4], -2.0 * x[5] + x[2]])

    def a_known(x, p):
        return np.array(
            [
                x[0] - 2.0 * x[1],
            ]
        )

    def v_known(x, p):
        a = a_known(x, p)
        return np.concatenate([k_known(x, p), a, -a])

    for _ in range(NUM_RANDOM):
        x = np.random.uniform(-10, 10, size=(6,))
        p = np.random.uniform(-10, 10)

        assert isclose(opt.f(x, p), fun_known(x, p))

        M = opt.M(p).toarray()
        c = opt.c(p).toarray()

        assert isclose(M @ x + c, k_known(x, p))

        A = opt.A(p).toarray()
        b = opt.b(p).toarray()

        assert isclose(A @ x + b, a_known(x, p))

        v = opt.v(x, p).toarray().flatten()
        assert isclose(v, v_known(x, p))

    attr_exp_value_map = {
        "nx": 6,
        "np": 1,
        "nk": 2,
        "na": 1,
        "ng": 0,
        "nh": 0,
        "nv": 4,
    }
    for attr, exp_value in attr_exp_value_map.items():
        assert getattr(opt, attr) == exp_value


def test_NonlinearCostNonlinearConstraints():
    dv = SXContainer()
    dv["x"] = optas.SX.sym("x", 2, 3)
    x = dv.vec()

    pr = SXContainer()
    pr["p"] = optas.SX.sym("p")
    p = pr.vec()

    ct = SXContainer()
    ct["c"] = p * optas.sumsqr(x) + x[0] * x[1] + 2.0 * (optas.cos(x[3]) - 1.0) ** 2

    lec = SXContainer()
    lec["lec"] = x[0] - 2.0 * x[1]

    lic = SXContainer()
    lic["lic1"] = x[2] + 3.0 * x[4]
    lic["lic2"] = -2.0 * x[5] + x[2]

    ec = SXContainer()
    ec["e1"] = x[3] ** 2 - x[1] * x[2]
    ec["e2"] = x[0] - 2.0 * x[1] * x[4]

    ic = SXContainer()
    ic["ic"] = x[1] + x[2] * x[0]

    opt = optas.optimization.NonlinearCostNonlinearConstraints(
        dv, pr, ct, lec, lic, ec, ic
    )

    def fun_known(x, p):
        return (
            p * np.sum(x.flatten() ** 2) + x[0] * x[1] + 2.0 * (np.cos(x[3]) - 1.0) ** 2
        )

    def k_known(x, p):
        return np.array([x[2] + 3.0 * x[4], -2.0 * x[5] + x[2]])

    def a_known(x, p):
        return np.array(
            [
                x[0] - 2.0 * x[1],
            ]
        )

    def g_known(x, p):
        return np.array([x[1] + x[2] * x[0]])

    def h_known(x, p):
        return np.array(
            [
                x[3] ** 2 - x[1] * x[2],
                x[0] - 2.0 * x[1] * x[4],
            ]
        )

    def v_known(x, p):
        a = a_known(x, p)
        h = h_known(x, p)
        return np.concatenate([k_known(x, p), g_known(x, p), a, -a, h, -h])

    for _ in range(NUM_RANDOM):
        x = np.random.uniform(-10, 10, size=(6,))
        p = np.random.uniform(-10, 10)

        assert isclose(opt.f(x, p), fun_known(x, p))

        M = opt.M(p).toarray()
        c = opt.c(p).toarray()

        assert isclose(M @ x + c, k_known(x, p))

        A = opt.A(p).toarray()
        b = opt.b(p).toarray()

        assert isclose(A @ x + b, a_known(x, p))

        v = opt.v(x, p).toarray().flatten()
        assert isclose(v, v_known(x, p))

    attr_exp_value_map = {
        "nx": 6,
        "np": 1,
        "nk": 2,
        "na": 1,
        "ng": 1,
        "nh": 2,
        "nv": 9,
    }
    for attr, exp_value in attr_exp_value_map.items():
        assert getattr(opt, attr) == exp_value

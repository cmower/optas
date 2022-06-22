import casadi as cs

big_number = 1.0e11  # replacement for inf

"""
Letters available for use as variables:
A, B, C, D, E, F, G, H, I, J, K, L, _, N, O, _, Q, R, S, T, U, V, W, X, Y, Z
a, b, _, d, e, _, _, _, i, j, _, l, m, n, o, _, _, _, s, t, _, _, w, _, y, z
"""

############################################################################
# Private optimization classes for specific cost functions and
# constraint formulations.

class _QuadraticCost:

    """

    Classes inheriting from this assume a quadratic cost function f(x, p) = x'P(p)x + x'q(p).

    """

    def P(self, p):

        # Create x SX variables
        x = self.decision_variables.vec()

        # Compute P
        ddf = self.ddf(x, p)  # ddf(x) = 2P for all x
        return 0.5*cs.DM(ddf)

    def q(self, p):

        # Create x SX variables
        x = self.decision_variables.vec()

        # Compute q
        x_zero = cs.DM.zeros(self.nx)
        df = self.df(x_zero, p)  # df(0) = q
        return cs.vec(cs.DM(df))  # ensure q is column vector

class _LinearConstraints:

    """

    Classes inheriting from this assume linear constraints lbk=0 <= k(x, p) = M(p)x + c <= ubk=big_number

    Additional methods:

    lbr(p)=-c(p) <= r(x, p) <= ubr(p)=big_number
                    where r(x, p) = M(p)x

    """

    def __init__(self):
        self.k = None
        self.dk = None
        self.lin_constraints = None

    @property
    def nk(self):
        return self.k.numel_out()

    @property
    def lbk(self):
        return cs.DM.zeros(self.nk)

    @property
    def ubk(self):
        return big_number*cs.DM.ones(self.nk)

    def M(self, p):

        # Create x SX variables
        x = self.decision_variables.vec()

        # Compute M
        dk = self.dk(x, p)  # dk(p) = M
        return cs.DM(dk)

    def c(self, p):

        # Create zeros x
        x_zero = cs.DM.zeros(self.nx)

        # Compute c
        k_zero = self.k(x_zero, p)
        return cs.vec(cs.DM(k_zero))  # ensure c is column vector

    @property
    def nr(self):
        return self.nk

    def r(self, x, p):
        return self.M(p) @ cs.vec(x)

    def dr(self, x, p):
        return self.M(p)

    def lbr(self, p):
        return -self.c(p)

    def ubr(self, p):
        return self.ubk

class _NonlinearEqualityConstraints:

    """

    Classes inheriting from this assume non-linear equality constraints lbg=0 <= g(x, p) <= ubg=0.

    """

    def __init__(self):
        self.g = None
        self.dg = None
        self.ddg = None
        self.eq_constraints = None

    @property
    def ng(self):
        return self.g.numel_out()

    @property
    def lbg(self):
        return cs.DM.zeros(self.ng)

    @property
    def ubg(self):
        return self.lbg

class _NonlinearInequalityConstraints:

    """

    Classes inheriting from this assume non-linear equality constraints lbh=0 <= h(x, p) <= ubh=big_number.

    """

    def __init__(self):
        self.h = None
        self.dh = None
        self.ddh = None
        self.ineq_constraints = None

    @property
    def nh(self):
        return self.h.numel_out()

    @property
    def lbh(self):
        return cs.DM.zeros(self.nh)

    @property
    def ubh(self):
        return big_number*cs.DM.ones(self.nh)

############################################################################
# Optimization classes

class _Optimization:

    """Base optimization class, assumes decision variables, parameters, and cost function"""

    def __init__(self):
        self.f = None
        self.df = None
        self.ddf = None
        self.decision_variables = None
        self.parameters = None
        self.cost_terms = None

    @property
    def nx(self):
        return self.decision_variables.numel()

    @property
    def np(self):
        return self.parameters.numel()

class UnconstrainedQP(
        _Optimization,
        _QuadraticCost):

    def __init__(self):
        _Optimization.__init__(self)
        _QuadraticCost.__init__(self)

class LinearConstrainedQP(
        UnconstrainedQP,
        _LinearConstraints):

    def __init__(self):
        UnconstrainedQP.__init__(self)
        _LinearConstraints.__init__(self)

class NonlinearConstrainedQP(
        UnconstrainedQP,
        _LinearConstraints,
        _NonlinearEqualityConstraints,
        _NonlinearInequalityConstraints):

    def __init__(self):
        UnconstrainedQP.__init__(self)
        _LinearConstraints.__init__(self)
        _NonlinearEqualityConstraints.__init__(self)
        _NonlinearInequalityConstraints.__init__(self)

class UnconstrainedOptimization(_Optimization):
    pass

class LinearConstrainedOptimization(
        UnconstrainedOptimization,
        _LinearConstraints):

    def __init__(self):
        UnconstrainedOptimization.__init__(self)
        _LinearConstraints.__init__(self)

class NonlinearConstrainedOptimization(
        LinearConstrainedOptimization,
        _NonlinearEqualityConstraints,
        _NonlinearInequalityConstraints):

    def __init__(self):
        LinearConstrainedOptimization.__init__(self)
        _NonlinearEqualityConstraints.__init__(self)
        _NonlinearInequalityConstraints.__init__(self)

    # Additional methods

    def u(self, x, p):
        return cs.vertcat(self.k(x, p), self.g(x, p), self.h(x, p))

    def du(self, x, p):
        return cs.vertcat(self.dk(x, p), self.dg(x, p), self.dh(x, p))

    def ddu(self, x, p):
        return cs.vertcat(self.ddk(x, p), self.ddg(x, p), self.ddh(x, p))

    @property
    def nu(self):
        return self.nk + self.ng + self.nh

    @property
    def lbu(self):
        return cs.vertcat(self.lbk, self.lbg, self.lbh)

    @property
    def ubu(self):
        return cs.vertcat(self.ubk, self.ubg, self.ubh)

    def v(self, x, p):
        return cs.vertcat(self.g(x, p), self.h(x, p), -self.h(x, p))

    def dv(self, x, p):
        return cs.vertcat(self.dg(x, p), self.dh(x, p), -self.dh(x, p))

    def ddv(self, x, p):
        return cs.vertcat(self.ddg(x, p), self.ddh(x, p), -self.ddh(x, p))

    @property
    def nv(self):
        return self.ng + 2*self.nh

    @property
    def lbv(self):
        return cs.DM.zeros(self.nv)

    @property
    def ubv(self):
        return cs.vertcat(self.ubg, cs.DM.zeros(2*self.nh))

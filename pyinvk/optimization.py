import casadi as cs
import numpy as np
from typing import Union

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

    def P(self, p: Union[cs.casadi.SX, cs.casadi.DM, np.ndarray]):
        """The P(p) array in a quadratic cost term where p are parameters."""

        # Create x SX variables
        x = self.decision_variables.vec()

        # Compute P
        ddf = self.ddf(x, p)  # ddf(x) = 2P for all x
        return 0.5*cs.DM(ddf)

    def q(self, p: Union[cs.casadi.SX, cs.casadi.DM, np.ndarray]):
        """The q(p) array in a quadratic cost term where p are parameters."""

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
        """Constructor for the _LinearConstraints class."""
        self.k = None
        self.dk = None
        self.ddk = None
        self.lin_constraints = None

    @property
    def nk(self):
        """The number of linear constraints."""
        return self.k.numel_out()

    @property
    def lbk(self):
        """Lower bound for the linear constraints (i.e. the zero array)."""
        return cs.DM.zeros(self.nk)

    @property
    def ubk(self):
        """Upper bound for the linear constraints (i.e. an array with large numbers used instead of infinity)."""
        return big_number*cs.DM.ones(self.nk)

    def M(self, p: Union[cs.casadi.SX, cs.casadi.DM, np.ndarray]):
        """The M(p) array in the linear constraints where p are parameters."""

        # Create x SX variables
        x = self.decision_variables.vec()

        # Compute M
        dk = self.dk(x, p)  # dk(p) = M
        return cs.DM(dk)

    def c(self, p: Union[cs.casadi.SX, cs.casadi.DM, np.ndarray]):
        """The c(p) array in the linear constraints where p are parameters."""

        # Create zeros x
        x_zero = cs.DM.zeros(self.nx)

        # Compute c
        k_zero = self.k(x_zero, p)
        return cs.vec(cs.DM(k_zero))  # ensure c is column vector

    @property
    def nr(self):
        """The number of linear constraints for r(x, p)."""
        return self.nk

    def r(self, x: Union[cs.casadi.SX, cs.casadi.DM, np.ndarray], p: Union[cs.casadi.SX, cs.casadi.DM, np.ndarray]):
        """The constraint vector r(x, p)."""
        return self.M(p) @ cs.vec(x)

    def dr(self, x: Union[cs.casadi.SX, cs.casadi.DM, np.ndarray], p: Union[cs.casadi.SX, cs.casadi.DM, np.ndarray]):
        """The first deriviative for the constraint vector r(x, p) with respect to x."""
        return self.M(p)

    def lbr(self, p: Union[cs.casadi.SX, cs.casadi.DM, np.ndarray]):
        """Lower bound for the constraint vector r(x, p), i.e. it is defined as -c(p)."""
        return -self.c(p)

    def ubr(self, p: Union[cs.casadi.SX, cs.casadi.DM, np.ndarray]):
        """Upper bound for the constraint vector r(x, p), i.e. it is an array with large numbers used instead of infinity."""
        return self.ubk

class _NonlinearEqualityConstraints:

    """

    Classes inheriting from this assume non-linear equality constraints lbg=0 <= g(x, p) <= ubg=0.

    """

    def __init__(self):
        """Constructor for the _NonlinearEqualityConstraint class."""
        self.g = None
        self.dg = None
        self.ddg = None
        self.eq_constraints = None

    @property
    def ng(self):
        """Number of constraints."""
        return self.g.numel_out()

    @property
    def lbg(self):
        """Lower bound for the constraints, i.e. an array of zeros."""
        return cs.DM.zeros(self.ng)

    @property
    def ubg(self):
        """Upper bound for the constraints, i.e. an array of zeros."""
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
        """Number of constraints."""
        return self.h.numel_out()

    @property
    def lbh(self):
        """Lower bound for the constraints, i.e. an array of zeros."""
        return cs.DM.zeros(self.nh)

    @property
    def ubh(self):
        """Upper bound for the constraints, i.e. an array of large numbers instead of infinity."""
        return big_number*cs.DM.ones(self.nh)

class _NonlinearConstraints:

    """Additional methods u, v for (in)equality nonlinear constraints."""

    def u(self, x: Union[cs.casadi.SX, cs.casadi.DM, np.ndarray], p: Union[cs.casadi.SX, cs.casadi.DM, np.ndarray]):
        """Constraint array where u=[k', g', h']'."""
        return cs.vertcat(self.k(x, p), self.g(x, p), self.h(x, p))

    def du(self, x: Union[cs.casadi.SX, cs.casadi.DM, np.ndarray], p: Union[cs.casadi.SX, cs.casadi.DM, np.ndarray]):
        """The first derivative of u with respect to x."""
        return cs.vertcat(self.dk(x, p), self.dg(x, p), self.dh(x, p))

    def ddu(self, x: Union[cs.casadi.SX, cs.casadi.DM, np.ndarray], p: Union[cs.casadi.SX, cs.casadi.DM, np.ndarray]):
        """The second derivative of u with respect to x."""
        return cs.vertcat(self.ddk(x, p), self.ddg(x, p), self.ddh(x, p))

    @property
    def nu(self):
        """The number of constraints in the u array."""
        return self.nk + self.ng + self.nh

    @property
    def lbu(self):
        """The lower bound for the u array."""
        return cs.vertcat(self.lbk, self.lbg, self.lbh)

    @property
    def ubu(self):
        """The upper bound for the u array."""
        return cs.vertcat(self.ubk, self.ubg, self.ubh)


    def v(self, x: Union[cs.casadi.SX, cs.casadi.DM, np.ndarray], p: Union[cs.casadi.SX, cs.casadi.DM, np.ndarray]):
        """Constraint array where v=[k', g', -g', h']' >= 0."""
        return cs.vertcat(
            self.k(x, p),
            self.g(x, p),
            -self.g(x, p),
            self.h(x, p),
        )

    def dv(self, x: Union[cs.casadi.SX, cs.casadi.DM, np.ndarray], p: Union[cs.casadi.SX, cs.casadi.DM, np.ndarray]):
        """The first derivative of the constraints v with respect to x."""
        return cs.vertcat(
            self.dk(x, p),
            self.dg(x, p),
            -self.dg(x, p),
            self.dh(x, p),
        )

    def ddv(self, x: Union[cs.casadi.SX, cs.casadi.DM, np.ndarray], p: Union[cs.casadi.SX, cs.casadi.DM, np.ndarray]):
        """The second derivative of the constraints v with respect to x."""
        return cs.vertcat(
            self.ddk(x, p),
            self.ddg(x, p),
            -self.ddg(x, p),
            self.ddh(x, p),
        )

    @property
    def nv(self):
        """The number of constraints in the vector v."""
        return self.nk + 2*self.ng + self.nh

    @property
    def lbv(self):
        """Lower bound for the constraints v, i.e. the zero array."""
        return cs.DM.zeros(self.nv)

    @property
    def ubv(self):
        """Upper bound for the constraints v, i.e. an array of large numbers instead of infinity."""
        return cs.vertcat(self.ubk, cs.DM.zeros(2*self.ng), self.ubh)

############################################################################
# Optimization classes

class _Optimization:

    """Base optimization class, assumes decision variables (x), parameters (p), and cost function (f)."""

    def __init__(self):
        self.f = None
        self.df = None
        self.ddf = None
        self.decision_variables = None
        self.parameters = None
        self.cost_terms = None

    @property
    def nx(self):
        """Number of decision variables."""
        return self.decision_variables.numel()

    @property
    def np(self):
        """Number of parameters."""
        return self.parameters.numel()

class UnconstrainedQP(
        _Optimization,
        _QuadraticCost):

    """Unconstrained Quadratic Program.

            min cost(x, p) where cost(x, p) = x'.P(p).x + x'.q(p)
             x

    The problem is unconstrained, and has quadratic cost function -
    note, P and q are derived from the given cost function (you don't
    have to explicitly state P/q).

    """

    def __init__(self):
        """Constructor for the UnconstrainedQP problem."""
        _Optimization.__init__(self)
        _QuadraticCost.__init__(self)

class LinearConstrainedQP(
        UnconstrainedQP,
        _LinearConstraints):

    """Linear constrained Quadratic Program.

            min cost(x, p) where cost(x, p) = x'.P(p).x + x'.q(p)
                 x

                subject to k(x, p) = M(p).x + c(p) >= 0

    The problem is constrained by only linear constraints and has a
    quadratic cost function - note, P/M and q/c are derived from the
    given cost function and constraints (you don't have to explicitly
    state P/q/M/c).

    """

    def __init__(self):
        """Constructor for the LinearConstrainedQP problem."""
        UnconstrainedQP.__init__(self)
        _LinearConstraints.__init__(self)

class NonlinearConstrainedQP(
        UnconstrainedQP,
        _LinearConstraints,
        _NonlinearEqualityConstraints,
        _NonlinearInequalityConstraints,
        _NonlinearConstraints):

    """Nonlinear constrained Quadratic Program.

            min cost(x, p) where cost(x) = x'.P(p).x + x'.q
             x

                subject to

                    k(x, p) = M(p).x + c(p) >= 0,
                    g(x) == 0, and
                    h(x) >= 0

    The problem is constrained by nonlinear constraints and has a
    quadratic cost function - note, P/M and q/c are derived from the
    given cost function and constraints (you don't have to explicitly
    state P/q/M/c).

    """

    def __init__(self):
        """Constructor for the NonlinearConstrainedQP problem."""
        UnconstrainedQP.__init__(self)
        _LinearConstraints.__init__(self)
        _NonlinearEqualityConstraints.__init__(self)
        _NonlinearInequalityConstraints.__init__(self)

class UnconstrainedOptimization(_Optimization):
    """Unconstrained optimization problem.

            min cost(x, p)
             x

    The problem is unconstrained and the cost function is nonlinear in
    x.

    """
    pass

class LinearConstrainedOptimization(
        UnconstrainedOptimization,
        _LinearConstraints):
    """Linear constrained optimization problem.


        min cost(x, p)
         x

            subject to k(x, p) = M(p).x + c(p) >= 0

    The problem is constrained with linear constraints and has a
    nonlinear cost function in x.

    """

    def __init__(self):
        """Constructor for the LinearConstrainedOptimization problem."""
        UnconstrainedOptimization.__init__(self)
        _LinearConstraints.__init__(self)

    def u(self, x: Union[cs.casadi.SX, cs.casadi.DM, np.ndarray], p: Union[cs.casadi.SX, cs.casadi.DM, np.ndarray]):
        """Constraint array equivalent to k."""
        return self.k(x, p)

    def du(self, x: Union[cs.casadi.SX, cs.casadi.DM, np.ndarray], p: Union[cs.casadi.SX, cs.casadi.DM, np.ndarray]):
        """The first derivative of the constraint array u with respect to x."""
        return self.dk(x, p)

    def ddu(self, x: Union[cs.casadi.SX, cs.casadi.DM, np.ndarray], p: Union[cs.casadi.SX, cs.casadi.DM, np.ndarray]):
        """The second derivative of the constraint array u with respect to x."""
        return self.ddk(x, p)

    @property
    def nu(self):
        """The number of constraints in the array u."""
        return self.k.numel_out()

    @property
    def lbu(self):
        """Lower bound for the constraints u, i.e. the zero array."""
        return cs.DM.zeros(self.nu)

    @property
    def ubu(self):
        """Upper bound for the constraints u, i.e. an array of large numbers instead of infinity."""
        return big_number*cs.DM(self.nu)

class NonlinearConstrainedOptimization(
        UnconstrainedOptimization,
        _LinearConstraints,
        _NonlinearEqualityConstraints,
        _NonlinearInequalityConstraints,
        _NonlinearConstraints):
    """Nonlinear constrained optimization problem.

        min cost(x, p)
         x

            subject to

                k(x, p) = M(p).x + c(p) >= 0,
                g(x) == 0, and
                h(x) >= 0

    The problem is constrained by nonlinear constraints and has a
    nonlinear cost function.

    """

    def __init__(self):
        """Constructor for the NonlinearConstrainedOptimization problem."""
        UnconstrainedOptimization.__init__(self)
        _LinearConstraints.__init__(self)
        _NonlinearEqualityConstraints.__init__(self)
        _NonlinearInequalityConstraints.__init__(self)

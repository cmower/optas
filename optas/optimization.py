import casadi as cs
from typing import Tuple
from .sx_container import SXContainer

def derive_jacobian_and_hessian_functions(name, fun, x, p):
    fun_input = [x, p]
    jac = cs.jacobian(fun(x, p), x)
    hes = cs.jacobian(jac, x)
    return cs.Function('d'+name, fun_input, [jac]), cs.Function('dd'+name, fun_input, [hes])

def vertcon(x, p, ineq=[], eq=[]):
    con = [i(x, p) for i in ineq]
    for e in eq:
        con.append(e(x, p))
        con.append(-e(x, p))
    return cs.Function('v', [x, p], [cs.vertcat(*con)])


class _Optimization:
    """Base optimization class.

    This class provides basic functions for the following optimization.

      min f(x, p)
       x

    Classes derived from this one will add additional methods
    regarding the cost function, and constraints.

    """

    inf = 1.0e10  # big number rather than np.inf

    def __init__(
            self,
            decision_variables: SXContainer, # SXContainer for decision variables
            parameters: SXContainer, # SXContainer for parameters
            cost_terms: SXContainer, # SXContainer for cost terms
    ):

        # Set class attributes
        self.decision_variables = decision_variables
        self.parameters = parameters
        self.cost_terms = cost_terms

        # Get symbolic variables
        x = decision_variables.vec() # symbolic decision variables
        p = parameters.vec() # symbolic parameters

        # Setup objective function
        f = cs.sum1(cost_terms.vec())
        self.f = cs.Function('f', [x, p], [f])

        # Derive jocobian/hessian of objective function
        self.df, self.ddf = derive_jacobian_and_hessian_functions('f', self.f, x, p)

        self.nx = decision_variables.numel()
        self.np = parameters.numel()
        self.nk = 0
        self.na = 0
        self.ng = 0
        self.nh = 0

class QuadraticCostUnconstrained(_Optimization):
    """Unconstrained Quadratic Program.

            min f(x, p) where f(x, p) = x'.P(p).x + x'.q(p)
             x

    The problem is unconstrained, and has quadratic cost function.

    """

    def __init__(
            self,
            decision_variables: SXContainer, # SXContainer for decision variables
            parameters: SXContainer, # SXContainer for parameters
            cost_terms: SXContainer, # SXContainer for cost terms
    ):
        super().__init__(decision_variables, parameters, cost_terms)

        # Ensure cost function is quadratic
        x = decision_variables.vec() # symbolic decision variables
        p = parameters.vec() # symbolic parameters

        # Initialize function for P
        # Note, since f is quadratic, the hessian is 2M(p)
        self.P = cs.Function('P', [p], [0.5*self.ddf(x, p)])

        # Initialize function for q
        x_zero = cs.DM.zeros(self.nx)
        self.q = cs.Function('q', [p], [cs.vec(self.df(x_zero, p))])

class QuadraticCostLinearConstraints(QuadraticCostUnconstrained):
    """Linear constrained Quadratic Program.

            min f(x, p) where f(x, p) = x'.P(p).x + x'.q(p)
                 x

                subject to
                            k(x, p) = M(p).x + c(p) >= 0
                            a(x, p) = A(p).x + b(p) == 0

    The problem is constrained by only linear constraints and has a
    quadratic cost function. Other methods defined are as follows



    """

    def __init__(
            self,
            decision_variables: SXContainer, # SXContainer for decision variables
            parameters: SXContainer, # SXContainer for parameters
            cost_terms: SXContainer, # SXContainer for cost terms
            lin_eq_constraints: SXContainer, # SXContainer for linear equality constraints
            lin_ineq_constraints: SXContainer, # SXContainer for linear inequality constraints
    ):
        super().__init__(decision_variables, parameters, cost_terms)

        x = decision_variables.vec() # symbolic decision variables
        p = parameters.vec() # symbolic parameters

        self.lin_eq_constraints = lin_eq_constraints
        self.lin_ineq_constraints = lin_ineq_constraints

        # Setup k
        self.k = cs.Function('k', [x, p], [lin_ineq_constraints.vec()])
        self.nk = lin_ineq_constraints.numel()
        self.lbk = cs.DM.zeros(self.nk)
        self.ubk = self.inf*cs.DM.ones(self.nk)

        # Setup M and c
        x_zero = cs.DM.zeros(self.nx)
        self.M = cs.Function('M', [p], [cs.jacobian(self.k(x, p), x)])
        self.c = cs.Function('c', [p], [self.k(x_zero, p)])

        # Setup a
        self.a = cs.Function('a', [x, p], [lin_eq_constraints.vec()])
        self.na = lin_eq_constraints.numel()
        self.lba = cs.DM.zeros(self.na)
        self.uba = cs.DM.zeros(self.na)

        # Setup A and b
        self.A = cs.Function('A', [p], [cs.jacobian(self.a(x, p), x)])
        self.b = cs.Function('b', [p], [self.a(x_zero, p)])

        # Setup v, i.e. v(x, p) >= 0
        self.v = vertcon(x, p, ineq=[self.k], eq=[self.a])
        dv = cs.vertcat(self.M(p), self.A(p), -self.A(p))
        self.dv = cs.Function('dv', [x, p], [dv])
        self.nv = v.shape[0]
        self.lbv = cs.DM.zeros(self.nv)
        self.ubv = self.inf*cs.DM.ones(self.nv)

class QuadraticCostNonlinearConstraints(QuadraticCostLinearConstraints):

    def __init__(
            self,
            decision_variables: SXContainer, # SXContainer for decision variables
            parameters: SXContainer, # SXContainer for parameters
            cost_terms: SXContainer, # SXContainer for cost terms
            lin_eq_constraints: SXContainer, # SXContainer for linear equality constraints
            lin_ineq_constraints: SXContainer, # SXContainer for linear inequality constraints
            eq_constraints: SXContainer, # SXContainer for equality constraints
            ineq_constraints: SXContainer, # SXContainer for inequality constraints
    ):
        super().__init__(decision_variables, parameters, cost_terms, lin_eq_constraints, lin_ineq_constraints)

        x = decision_variables.vec() # symbolic decision variables
        p = parameters.vec() # symbolic parameters

        # Setup g
        self.g = cs.Function('g', [x, p], [ineq_constraints.vec()])
        self.ng = ineq_constraints.numel()
        self.lbg = cs.DM.zeros(self.ng)
        self.ubg = self.inf*cs.DM.ones(self.ng)
        self.dg, self.ddg = derive_jacobian_and_hessian_functions('g', self.g, x, p)

        # Setup h
        self.h = cs.Function('g', [x, p], [eq_constraints.vec()])
        self.nh = eq_constraints.numel()
        self.lbh = cs.DM.zeros(self.nh)
        self.ubh = cs.DM.zeros(self.nh)
        self.dh, self.ddh = derive_jacobian_and_hessian_functions('h', self.h, x, p)

        # Setup v, i.e. v(x, p) >= 0
        self.v = vertcon(x, p, ineq=[self.k, self.g], eq=[self.a, self.h])
        self.nv = v.shape[0]
        self.lbv = cs.DM.zeros(self.nv)
        self.ubv = self.inf*cs.DM.ones(self.nv)
        self.dv, self.ddv = derive_jacobian_and_hessian_functions('v', self.v, x, p)


class NonlinearCostUnconstrained(_Optimization):
    """Unconstrained optimization problem.

            min f(x, p)
             x

    The problem is unconstrained and the cost function is nonlinear.

    """
    pass

class NonlinearCostLinearConstraints(NonlinearCostUnconstrained):
    """Linear constrained optimization problem.


            min f(x, p)
                 x

                subject to
                            k(x, p) = M(p).x + c(p) >= 0
                            a(x, p) = A(p).x + b(p) == 0

    The problem is constrained by only linear constraints and has a
    quadratic cost function.


    The problem is constrained with linear constraints and has a
    nonlinear cost function in x.

    """

    def __init__(
            self,
            decision_variables: SXContainer, # SXContainer for decision variables
            parameters: SXContainer, # SXContainer for parameters
            cost_terms: SXContainer, # SXContainer for cost terms
            lin_eq_constraints: SXContainer, # SXContainer for linear equality constraints
            lin_ineq_constraints: SXContainer, # SXContainer for linear inequality constraints
    ):

        super().__init__(decision_variables, parameters, cost_terms)

        x = decision_variables.vec() # symbolic decision variables
        p = parameters.vec() # symbolic parameters

        self.lin_eq_constraints = lin_eq_constraints
        self.lin_ineq_constraints = lin_ineq_constraints

        # Setup k
        self.k = cs.Function('k', [x, p], [lin_ineq_constraints.vec()])
        self.nk = lin_ineq_constraints.numel()
        self.lbk = cs.DM.zeros(self.nk)
        self.ubk = self.inf*cs.DM.ones(self.nk)

        # Setup M and c
        x_zero = cs.DM.zeros(self.nx)
        self.M = cs.Function('M', [p], [cs.jacobian(self.k(x, p), x)])
        self.c = cs.Function('c', [p], [self.k(x_zero, p)])

        # Setup a
        self.a = cs.Function('a', [x, p], [lin_eq_constraints.vec()])
        self.na = lin_eq_constraints.numel()
        self.lba = cs.DM.zeros(self.na)
        self.uba = cs.DM.zeros(self.na)

        # Setup A and b
        self.A = cs.Function('A', [p], [cs.jacobian(self.a(x, p), x)])
        self.b = cs.Function('b', [p], [self.a(x_zero, p)])

        # Setup v, i.e. v(x, p) >= 0
        self.v = vertcon(x, p, ineq=[self.k], eq=[self.a])
        self.nv = v.shape[0]
        self.lbv = cs.DM.zeros(self.nv)
        self.ubv = self.inf*cs.DM.ones(self.nv)
        self.dv, self.ddv = derive_jacobian_and_hessian_functions('v', self.v, x, p)

class NonlinearCostNonlinearConstraints(NonlinearCostLinearConstraints):
    """Nonlinear constrained optimization problem.

        min f(x, p)
         x

            subject to

                k(x, p) = M(p).x + c(p) >= 0
                a(x, p) = A(p).x + b(p) == 0
                g(x) >= 0, and
                h(x) == 0

    The problem is constrained by nonlinear constraints and has a
    nonlinear cost function.

    """

    def __init__(
            self,
            decision_variables: SXContainer, # SXContainer for decision variables
            parameters: SXContainer, # SXContainer for parameters
            cost_terms: SXContainer, # SXContainer for cost terms
            lin_eq_constraints: SXContainer, # SXContainer for linear equality constraints
            lin_ineq_constraints: SXContainer, # SXContainer for linear inequality constraints
            eq_constraints: SXContainer, # SXContainer for equality constraints
            ineq_constraints: SXContainer, # SXContainer for inequality constraints
    ):
        super().__init__(decision_variables, parameters, cost_terms, lin_eq_constraints, lin_ineq_constraints)

        x = decision_variables.vec() # symbolic decision variables
        p = parameters.vec() # symbolic parameters

        # Setup g
        self.g = cs.Function('g', [x, p], [ineq_constraints.vec()])
        self.ng = ineq_constraints.numel()
        self.lbg = cs.DM.zeros(self.ng)
        self.ubg = self.inf*cs.DM.ones(self.ng)
        self.dg, self.ddg = derive_jacobian_and_hessian_functions('g', self.g, x, p)

        # Setup h
        self.h = cs.Function('g', [x, p], [eq_constraints.vec()])
        self.nh = eq_constraints.numel()
        self.lbh = cs.DM.zeros(self.nh)
        self.ubh = cs.DM.zeros(self.nh)
        self.dh, self.ddh = derive_jacobian_and_hessian_functions('h', self.h, x, p)

        # Setup v, i.e. v(x, p) >= 0
        self.v = vertcon(x, p, ineq=[self.k, self.g], eq=[self.a, self.h])
        self.nv = v.shape[0]
        self.lbv = cs.DM.zeros(self.nv)
        self.ubv = self.inf*cs.DM.ones(self.nv)
        self.dv, self.ddv = derive_jacobian_and_hessian_functions('v', self.v, x, p)

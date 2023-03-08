import casadi as cs
from .models import Model
from .sx_container import SXContainer
from .spatialmath import CasADiArrayType
from typing import List, Tuple


def derive_jacobian_and_hessian_functions(
    name: str, fun: cs.Function, x: CasADiArrayType, p: CasADiArrayType
) -> Tuple[cs.Function]:
    """! Compute the Jacobian and Hessian for a given function using automatic differentiation.

    @param name The function name.
    @param fun The CasADi function.
    @param x The variables of the function.
    @param p The parameters of the function.
    @return The Jacobian and Hessian that are the derivatives of the function wrt the variables x.
    """
    fun_input = [x, p]
    jac = cs.jacobian(fun(x, p), x)
    hes = cs.jacobian(jac, x)
    Jac = cs.Function("d" + name, fun_input, [jac])
    Hes = cs.Function("dd" + name, fun_input, [hes])
    return Jac, Hes


def vertcon(
    x: CasADiArrayType,
    p: CasADiArrayType,
    ineq: List[cs.Function] = [],
    eq: List[cs.Function] = [],
):
    """! Align inequality and equality constraints vertically.

    Given an inequality constraint ineq(x, p) and equality constraint eq(x, p). The method that is returned evaluates the following array.

                  [ ineq(x, p) ]
        v(x, p) = [   eq(x, p) ]
                  [  -eq(x, p) ]

    @param x The variables of the functions.
    @param p The parameters of the functions.
    @param ineq A list of inequality constraints.
    @param eq A list of equality constraints.
    @return A CasADi function that evaluates the constraints in the form v(x, p) >= 0 (see above).
    """
    con = [i(x, p) for i in ineq]
    for e in eq:
        con.append(e(x, p))
        con.append(-e(x, p))
    return cs.Function("v", [x, p], [cs.vertcat(*con)])


class Optimization:
    """! Base optimization class."""

    ## big number rather than np.inf
    inf = 1.0e10

    def __init__(
        self,
        decision_variables: SXContainer,
        parameters: SXContainer,
        cost_terms: SXContainer,
    ):
        """! Initializer for the Optimization class.

        @param decision_variables SXContainer containing decision variables.
        @param parameters SXContainer containing parameters.
        @param cost_terms SXContainer containing cost terms.
        @return Instance of the Optimization class.
        """
        # Set class attributes

        ## SXContainer containing decision variables.
        self.decision_variables = decision_variables

        ## SXContainer containing parameters.
        self.parameters = parameters

        ## SXContainer containing cost terms.
        self.cost_terms = cost_terms

        ## SXContainer containing linear equality constraints.
        self.lin_eq_constraints = {}

        ## SXContainer containing linear inequality constraints.
        self.lin_ineq_constraints = {}

        ## SXContainer containing equality constraints.
        self.eq_constraints = {}

        ## SXContainer containing inequality constraints.
        self.ineq_constraints = {}

        ## A list of the task and robot models (set during build method in the OptimizationBuilder class)
        self.models = None

        # Get symbolic variables
        x = decision_variables.vec()  # symbolic decision variables
        p = parameters.vec()  # symbolic parameters

        ## Vectorized decision variables.
        self.x = x

        ## Vectorized parameters.
        self.p = p

        # Setup objective function.
        f = cs.sum1(cost_terms.vec())

        ## CasADi function that evaluates the objective function.
        self.f = cs.Function("f", [x, p], [f])

        # Derive jocobian/hessian of objective function
        df, ddf = derive_jacobian_and_hessian_functions("f", self.f, x, p)

        ## Jacobian of the objective function.
        self.df = df

        ## Hessian of the objective function.
        self.ddf = ddf

        ## Number of decision variables.
        self.nx = decision_variables.numel()

        ## Number of parameters.
        self.np = parameters.numel()

        ## Number of linear inequality constraints.
        self.nk = 0

        ## Number of linear equality constraints.
        self.na = 0

        ## Number of inequality constraints.
        self.ng = 0

        ## Number of equality constraints.
        self.nh = 0

        ## Number of vectorized constraints (see vertcon).
        self.nv = 0

    def set_models(self, models: List[Model]) -> None:
        """! Specify the models in the optimization problem.

        @param models A list of models (i.e. instance of a sub-class of Model).
        """
        self.models = models

    def specify_v(self, ineq=[], eq=[]):
        self.v = vertcon(self.x, self.p, ineq=ineq, eq=eq)
        self.nv = self.v.numel_out()
        self.lbv = cs.DM.zeros(self.nv)
        self.ubv = self.inf * cs.DM.ones(self.nv)
        self.dv, self.ddv = derive_jacobian_and_hessian_functions(
            "v", self.v, self.x, self.p
        )


class QuadraticCostUnconstrained(Optimization):
    """! Unconstrained Quadratic Program.

            min f(x, p) where f(x, p) = x'.P(p).x + x'.q(p)
             x

    The problem is unconstrained, and has quadratic cost function.
    """

    def __init__(
        self,
        decision_variables: SXContainer,  # SXContainer for decision variables
        parameters: SXContainer,  # SXContainer for parameters
        cost_terms: SXContainer,  # SXContainer for cost terms
    ):
        """! Initializer for the QuadraticCostUnconstrained class.

        @param decision_variables SXContainer containing decision variables.
        @param parameters SXContainer containing parameters.
        @param cost_terms SXContainer containing cost terms.
        @return Instance of the QuadraticCostUnconstrained class.
        """
        super().__init__(decision_variables, parameters, cost_terms)

        # Ensure cost function is quadratic
        x = decision_variables.vec()  # symbolic decision variables
        p = parameters.vec()  # symbolic parameters

        # Initialize function for P
        # Note, since f is quadratic, the hessian is 2M(p)

        ## CasADi function that returns the P term in the cost function.
        self.P = cs.Function("P", [p], [0.5 * self.ddf(x, p)])

        # Initialize function for q
        x_zero = cs.DM.zeros(self.nx)

        ## CasADi function that returns the q term in the cost function.
        self.q = cs.Function("q", [p], [cs.vec(self.df(x_zero, p))])


class QuadraticCostLinearConstraints(QuadraticCostUnconstrained):
    """! Linear constrained Quadratic Program.

            min f(x, p) where f(x, p) = x'.P(p).x + x'.q(p)
                 x

                subject to
                            k(x, p) = M(p).x + c(p) >= 0
                            a(x, p) = A(p).x + b(p) == 0

    The problem is constrained by only linear constraints and has a
    quadratic cost function.
    """

    def __init__(
        self,
        decision_variables: SXContainer,  # SXContainer for decision variables
        parameters: SXContainer,  # SXContainer for parameters
        cost_terms: SXContainer,  # SXContainer for cost terms
        lin_eq_constraints: SXContainer,  # SXContainer for linear equality constraints
        lin_ineq_constraints: SXContainer,  # SXContainer for linear inequality constraints
    ):
        """! Initializer for the QuadraticCostLinearConstraints class.

        @param decision_variables SXContainer containing decision variables.
        @param parameters SXContainer containing parameters.
        @param cost_terms SXContainer containing cost terms.
        @param lin_eq_constraints SXContainer containing the linear equality constraints.
        @param lin_ineq_constraints SXContainer containing the linear inequality constraints.
        @return Instance of the QuadraticCostLinearConstraints class.
        """
        super().__init__(decision_variables, parameters, cost_terms)

        x = decision_variables.vec()  # symbolic decision variables
        p = parameters.vec()  # symbolic parameters

        ## SXContainer containing the linear equality constraints.
        self.lin_eq_constraints = lin_eq_constraints

        ## SXContainer containing the linear inequality constraints.
        self.lin_ineq_constraints = lin_ineq_constraints

        # Setup k

        ## CasADi function that evaluates the linear inequality constraints.
        self.k = cs.Function("k", [x, p], [lin_ineq_constraints.vec()])

        ## Number of linear inequality constraints.
        self.nk = lin_ineq_constraints.numel()

        ## Lower bound for the linear inequality constraints (i.e. zeros).
        self.lbk = cs.DM.zeros(self.nk)

        ## Upper bound for the linear inequality constraints (i.e. inf).
        self.ubk = self.inf * cs.DM.ones(self.nk)

        # Setup M and c
        x_zero = cs.DM.zeros(self.nx)

        ## CasADi function that evaluates the M term in the linear inequality constraints function.
        self.M = cs.Function("M", [p], [cs.jacobian(self.k(x, p), x)])

        ## CasADi function that evaluates the c term in the linear inequality constraints function.
        self.c = cs.Function("c", [p], [self.k(x_zero, p)])

        # Setup a

        ## CasADi function that evaluates the linear equality cosntraints.
        self.a = cs.Function("a", [x, p], [lin_eq_constraints.vec()])

        ## Number of linear equality constraints.
        self.na = lin_eq_constraints.numel()

        ## Lower bound for the linear equality constraints (i.e. zeros).
        self.lba = cs.DM.zeros(self.na)

        ## Upper bound for the linear equality constraints (i.e. zeros).
        self.uba = cs.DM.zeros(self.na)

        # Setup A and b

        ## CasADi function that evaluates the A term in the linear equality constraints.
        self.A = cs.Function("A", [p], [cs.jacobian(self.a(x, p), x)])

        ## CasADi function that evaluates the b term in the linear equality constraints.
        self.b = cs.Function("b", [p], [self.a(x_zero, p)])

        # Setup v, i.e. v(x, p) >= 0
        self.specify_v(ineq=[self.k], eq=[self.a])


class QuadraticCostNonlinearConstraints(QuadraticCostLinearConstraints):

    """Nonlinear constrained optimization problem with quadratic cost function.

            min f(x, p) where f(x, p) = x'.P(p).x + x'.q(p)
                 x

            subject to

                k(x, p) = M(p).x + c(p) >= 0
                a(x, p) = A(p).x + b(p) == 0
                g(x) >= 0, and
                h(x) == 0

    The problem is constrained by nonlinear constraints and has a
    quadratic cost function.

    """

    def __init__(
        self,
        decision_variables: SXContainer,  # SXContainer for decision variables
        parameters: SXContainer,  # SXContainer for parameters
        cost_terms: SXContainer,  # SXContainer for cost terms
        lin_eq_constraints: SXContainer,  # SXContainer for linear equality constraints
        lin_ineq_constraints: SXContainer,  # SXContainer for linear inequality constraints
        eq_constraints: SXContainer,  # SXContainer for equality constraints
        ineq_constraints: SXContainer,  # SXContainer for inequality constraints
    ):
        super().__init__(
            decision_variables,
            parameters,
            cost_terms,
            lin_eq_constraints,
            lin_ineq_constraints,
        )

        x = decision_variables.vec()  # symbolic decision variables
        p = parameters.vec()  # symbolic parameters

        # Setup g
        self.g = cs.Function("g", [x, p], [ineq_constraints.vec()])
        self.ng = ineq_constraints.numel()
        self.lbg = cs.DM.zeros(self.ng)
        self.ubg = self.inf * cs.DM.ones(self.ng)
        self.dg, self.ddg = derive_jacobian_and_hessian_functions("g", self.g, x, p)

        # Setup h
        self.h = cs.Function("h", [x, p], [eq_constraints.vec()])
        self.nh = eq_constraints.numel()
        self.lbh = cs.DM.zeros(self.nh)
        self.ubh = cs.DM.zeros(self.nh)
        self.dh, self.ddh = derive_jacobian_and_hessian_functions("h", self.h, x, p)

        # Setup v, i.e. v(x, p) >= 0        
        self.specify_v(ineq=[self.k, self.g], eq=[self.a, self.h])


class NonlinearCostUnconstrained(Optimization):
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
        decision_variables: SXContainer,  # SXContainer for decision variables
        parameters: SXContainer,  # SXContainer for parameters
        cost_terms: SXContainer,  # SXContainer for cost terms
        lin_eq_constraints: SXContainer,  # SXContainer for linear equality constraints
        lin_ineq_constraints: SXContainer,  # SXContainer for linear inequality constraints
    ):
        super().__init__(decision_variables, parameters, cost_terms)

        x = decision_variables.vec()  # symbolic decision variables
        p = parameters.vec()  # symbolic parameters

        self.lin_eq_constraints = lin_eq_constraints
        self.lin_ineq_constraints = lin_ineq_constraints

        # Setup k
        self.k = cs.Function("k", [x, p], [lin_ineq_constraints.vec()])
        self.nk = lin_ineq_constraints.numel()
        self.lbk = cs.DM.zeros(self.nk)
        self.ubk = self.inf * cs.DM.ones(self.nk)

        # Setup M and c
        x_zero = cs.DM.zeros(self.nx)
        self.M = cs.Function("M", [p], [cs.jacobian(self.k(x, p), x)])
        self.c = cs.Function("c", [p], [self.k(x_zero, p)])

        # Setup a
        self.a = cs.Function("a", [x, p], [lin_eq_constraints.vec()])
        self.na = lin_eq_constraints.numel()
        self.lba = cs.DM.zeros(self.na)
        self.uba = cs.DM.zeros(self.na)

        # Setup A and b
        self.A = cs.Function("A", [p], [cs.jacobian(self.a(x, p), x)])
        self.b = cs.Function("b", [p], [self.a(x_zero, p)])

        # Setup v, i.e. v(x, p) >= 0
        self.specify_v(ineq=[self.k], eq=[self.a])


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
        decision_variables: SXContainer,  # SXContainer for decision variables
        parameters: SXContainer,  # SXContainer for parameters
        cost_terms: SXContainer,  # SXContainer for cost terms
        lin_eq_constraints: SXContainer,  # SXContainer for linear equality constraints
        lin_ineq_constraints: SXContainer,  # SXContainer for linear inequality constraints
        eq_constraints: SXContainer,  # SXContainer for equality constraints
        ineq_constraints: SXContainer,  # SXContainer for inequality constraints
    ):
        super().__init__(
            decision_variables,
            parameters,
            cost_terms,
            lin_eq_constraints,
            lin_ineq_constraints,
        )

        x = decision_variables.vec()  # symbolic decision variables
        p = parameters.vec()  # symbolic parameters

        self.eq_constraints = eq_constraints
        self.ineq_constraints = ineq_constraints

        # Setup g
        self.g = cs.Function("g", [x, p], [ineq_constraints.vec()])
        self.ng = ineq_constraints.numel()
        self.lbg = cs.DM.zeros(self.ng)
        self.ubg = self.inf * cs.DM.ones(self.ng)
        self.dg, self.ddg = derive_jacobian_and_hessian_functions("g", self.g, x, p)

        # Setup h
        self.h = cs.Function("g", [x, p], [eq_constraints.vec()])
        self.nh = eq_constraints.numel()
        self.lbh = cs.DM.zeros(self.nh)
        self.ubh = cs.DM.zeros(self.nh)
        self.dh, self.ddh = derive_jacobian_and_hessian_functions("h", self.h, x, p)

        # Setup v, i.e. v(x, p) >= 0
        self.specify_v(ineq=[self.k, self.g], eq=[self.a, self.h])

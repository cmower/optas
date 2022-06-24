import casadi as cs
from .sx_container import SXContainer
from .optimization import UnconstrainedQP,\
    LinearConstrainedQP, \
    NonlinearConstrainedQP, \
    UnconstrainedOptimization, \
    LinearConstrainedOptimization, \
    NonlinearConstrainedOptimization

class OptimizationBuilder:

    """Class that builds an optimization problem"""

    def __init__(self, robots, T=1, qderivs=[0]):

        # Input check
        assert min(qderivs) >= 0, "All values in qderivs should be positive or zero"
        dorder = max(qderivs)
        assert T >= 1, "T must be strictly positive"
        assert dorder >= 0, "dorder must be non-negative"
        assert T > dorder, f"T must be greater than {dorder}"

        # Set class attributes
        self.qderivs = qderivs
        self.dorder = dorder
        self.robots = robots

        # Setup decision variables
        self.decision_variables = SXContainer()
        for robot_name, robot in robots.items():
            for qderiv in qderivs:
                n = self.statename(robot_name, qderiv)
                self.decision_variables[n] = cs.SX.sym(n, robot.ndof, T-qderiv)

        # Setup containers for parameters, cost terms, ineq/eq constraints
        self.parameters = SXContainer()
        self.cost_terms = SXContainer()
        self.lin_constraints = SXContainer()
        self.ineq_constraints = SXContainer()
        self.eq_constraints = SXContainer()

    @staticmethod
    def statename(robot_name, qderiv):
        return robot_name + '/' + 'd'*qderiv + 'q'

    def get_state(self, robot_name, t, qderiv=0):
        assert qderiv in self.qderivs, f"{qderiv=}, qderiv must be in {self.qderivs}"
        states = self.decision_variables[self.statename(robot_name, qderiv)]
        return states[:, t]

    def add_decision_variables(self, name, m=1, n=1):
        x = cs.SX.sym(name, m, n)
        self.decision_variables[name] = x
        return x

    def add_cost_term(self, name, cost_term):
        assert cost_term.shape == (1, 1), "cost terms must be scalars"
        self.cost_terms[name] = cost_term

    def add_parameter(self, name, m=1, n=1):
        p = cs.SX.sym(name, m, n)
        self.parameters[name] = p
        return p

    def add_lin_constraint(self, name, lbc, c, ubc):
        x = self.decision_variables.vec()
        lb = c - lbc
        ub = ubc - c
        assert cs.is_linear(lb, x) and cs.is_linear(ub, x), "constraint not linear"
        self.lin_constraints[name+'_lb'] = lb
        self.lin_constraints[name+'_ub'] = ub

    def add_ineq_constraint(self, name, lbc, c, ubc):
        x = self.decision_variables.vec()
        lb = c - lbc
        ub = ubc - c
        if cs.is_linear(lb, x) and cs.is_linear(ub, x):
            print(f"[WARN] given constraint '{name}' is linear in x, adding as linear contraint")
            self.add_lin_constraint(name, lbc, c, ubc)
            return
        self.ineq_constraints[name+'_lb'] = lb
        self.ineq_constraints[name+'_ub'] = ub

    def add_eq_constraint(self, name, c, eqc):
        x = self.decision_variables.vec()
        eq = c - eqc
        if cs.is_linear(eq, x):
            print(f"[WARN] given constraint '{name}' is linear in x, adding as linear contraint")
            self.add_lin_constraint(name, eqc, c, eqc)
            return
        self.eq_constraints[name] = eq

    def build(self):

        # Get decision variables and parameters as SX column vectors
        x = self.decision_variables.vec()
        p = self.parameters.vec()

        # Helpful method
        def functionize(name, fun):

            # Setup function input
            fun_input = [x, p]

            # Function
            Fun = cs.Function(name, fun_input, [fun])

            # Jacobian
            jac = cs.jacobian(fun, x)
            Jac = cs.Function('d'+name, fun_input, [jac])

            # Hessian
            hess = cs.jacobian(jac, x)
            Hess = cs.Function('dd'+name, fun_input, [hess])

            return Fun, Jac, Hess

        # Get forward functions
        f = cs.sum1(self.cost_terms.vec())
        k = self.lin_constraints.vec()
        g = self.eq_constraints.vec()
        h = self.ineq_constraints.vec()

        # Setup optimization
        nlin = self.lin_constraints.numel()  # no. linear constraints
        nnlin = self.eq_constraints.numel() + self.ineq_constraints.numel()  # no. nonlin constraints
        if cs.is_quadratic(f, x):
            # True -> use QP formulation
            if nnlin > 0:
                opt = NonlinearConstrainedQP()
            elif nlin > 0:
                opt = LinearConstrainedQP()
            else:
                opt = UnconstrainedQP()
        else:
            # False -> use (nonlinear) Optimization formulation
            if nnlin > 0:
                opt = NonlinearConstrainedOptimization()
            elif nlin > 0:
                opt = LinearConstrainedOptimization()
            else:
                opt = UnconstrainedOptimization()

        # Setup constraints
        if nnlin > 0:
            opt.k, opt.dk, opt.ddk = functionize('k', k)
            opt.g, opt.dg, opt.ddg = functionize('g', g)
            opt.h, opt.dh, opt.ddh = functionize('h', h)
        if nlin > 0:
            opt.k, opt.dk, opt.ddk = functionize('k', k)

        # Setup cost function and other variables
        opt.f, opt.df, opt.ddf = functionize('f', f)
        opt.decision_variables = self.decision_variables
        opt.parameters = self.parameters
        opt.cost_terms = self.cost_terms

        return opt

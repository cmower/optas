import casadi as cs
from .optimization import Optimization
from .sx_container import SXContainer

class OptimizationBuilder:

    """Class that builds an optimization problem"""

    def __init__(self, robots, T=1, dorder=0):

        # Input check
        assert T >= 1, "T must be strictly positive"
        assert dorder >= 0, "dorder must be non-negative"
        assert T > dorder, f"T must be greater than {dorder=}"

        # Set class attributes
        self.dorder = dorder
        self.robots = robots

        # Setup decision variables
        self.decision_variables = SXContainer()
        for robot_name, robot in robots.items():
            for deriv in range(dorder+1):
                n = self.statename(robot_name, deriv)
                self.decision_variables[n] = cs.SX.sym(n, robot.ndof, T-d)

        # Setup containers for parameters, cost terms, ineq/eq constraints
        self.parameters = SXContainer()
        self.cost_terms = SXContainer()
        self.ineq_constraints = SXContainer()
        self.eq_constraints = SXContainer()

        # Create optimization object
        self.optimization = Optimization()

    @staticmethod
    def statename(robot_name, deriv):
        return robot_name + '/' + 'd'*deriv + 'q'

    def get_state(self, robot_name, t, deriv=0):
        assert 0 <= deriv <= self.dorder, f"{deriv=}, deriv must be in [0, {self.dorder}]"
        states = self.decision_variables[self.statename(robot_name, deriv)]
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

    def add_ineq_constraint(self, name, c, lbc=None, ubc=None):

        c_ = cs.vec(c)
        constraint = c_

        if lbc is not None:
            constraint -= cs.vec(lbc)

        if ubc is not None:
            constraint = cs.vertcat(constraint, cs.vec(ubc) - c_)

        self.ineq_constraints[name] = constraint

    def add_eq_constraint(self, name, c, eq=None):
        constraint = cs.vec(c)
        if eq is not None:
            constraint -= cs.vec(eq)
        self.eq_constraints[name] = constraint

    def build(self):

        x = self.decision_variables.vec()
        p = self.parameters.vec()

        f = cs.sum1(self.cost_terms.vec())
        df = cs.jacobian(f, x)
        ddf = cs.jacobian(df, x)
        self.optimization.f = cs.Function('f', [x, p], [f])
        self.optimization.df = cs.Function('df', [x, p], [df])
        self.optimization.ddf = cs.Function('ddf', [x, p], [ddf])

        g = self.ineq_constraints.vec()
        dg = cs.jacobian(g, x)
        ddg = cs.jacobian(dg, x)
        self.optimization.g = cs.Function('g', [x, p], [g])
        self.optimization.dg = cs.Function('dg', [x, p], [dg])
        self.optimization.ddg = cs.Function('ddg', [x, p], [ddg])

        h = self.eq_constraints.vec()
        dh = cs.jacobian(h, x)
        ddh = cs.jacobian(dh, x)
        self.optimization.h = cs.Function('h', [x, p], [h])
        self.optimization.dh = cs.Function('dh', [x, p], [dh])
        self.optimization.ddh = cs.Function('ddh', [x, p], [ddh])

        self.optimization.decision_variables = self.decision_variables
        self.optimization.parameters = self.parameters
        self.optimization.cost_terms = self.cost_terms
        self.optimization.ineq_constraints = self.ineq_constraints
        self.optimization.eq_constraints = self.eq_constraints

        return self.optimization

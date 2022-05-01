import casadi as cs
from .optimization import Optimization


class OptimizationBuilder:

    """Class that builds an optimization problem"""

    def __init__(self, robot_model, N=1):
        self.__robot_model = robot_model
        self.__optimization = Optimization(robot_model, N)

    def get_q(self, i=-1):
        """Get symbolic joint angles"""
        return self.__optimization.q[:, i]

    # Update problem spec

    def add_cost_term(self, name, cost_term):
        """Add cost term"""
        if isinstance(cost_term, (cs.casadi.SX, cs.casadi.DM)):
            assert cost_term.shape == (1, 1), "cost term must have shape 1-by-1"
        self.__optimization.cost_terms[name] = cost_term

    def add_parameter(self, name, m=1, n=1):
        """Add parameter"""
        p = cs.SX.sym(name, m, n)
        self.__optimization.parameters[name] = p
        return p

    def add_ineq_constraint(self, name, constraint):
        """Add inequality constraint g(q) >= 0"""
        self.__optimization.ineq_constraints[name] = constraint  # must be constraint >= 0

    def add_eq_constraint(self, name, constraint):
        """Add equality constraint g(q) == 0"""
        self.__optimization.eq_constraints[name] = constraint  # must be constraint == 0

    # Common constraints

    def enforce_joint_limits(self):
        """Enforce joint limit constraints"""

        lower = cs.DM.zeros(self.__robot_model.ndof, self.__optimization.N)
        upper = cs.DM.zeros(self.__robot_model.ndof, self.__optimization.N)
        for i in range(self.__optimization.N):
            lower[:, i] = self.__robot_model.lower_joint_limits
            upper[:, i] = self.__robot_model.upper_joint_limits

        self.__optimization.lbq = cs.vec(lower)
        self.__optimization.ubq = cs.vec(upper)

    # Main build method

    def build(self):
        return self.__optimization.finalize()

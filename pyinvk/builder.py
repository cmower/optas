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

        # Ensure problem is not already built
        if self.__optimization.optimization_problem_is_finalized:
            raise RuntimeError("build should only be called once")

        # Setup vectorized symbolic variables
        self.__optimization.sx_q    = cs.vec(self.__optimization.q)                  # decision variables, joint angles
        self.__optimization.sx_p    = self.__optimization.parameters.vec()           # problem parameters
        self.__optimization.sx_cost = cs.sum1(self.__optimization.cost_terms.vec())  # cost function
        self.__optimization.sx_g    = self.__optimization.ineq_constraints.vec()     # inequality constraints
        self.__optimization.sx_h    = self.__optimization.eq_constraints.vec()       # equality constraints

        self.__optimization.Nq = self.__optimization.sx_q.numel()  # number of decision variables

        # Method that sets up forward symbolic function and its jacobian/hessian
        fin = [self.__optimization.sx_q, self.__optimization.sx_p]  # all these function have same input

        def setup_funs(label, fun_sx):
            funj_sx = cs.jacobian(fun_sx, self.__optimization.sx_q)
            fun = cs.Function(label, fin, [fun_sx])
            funj = cs.Function(label+'_jacobian', fin, [funj_sx])
            funh = cs.Function(label+'_hessian', fin, [cs.jacobian(funj_sx, self.__optimization.sx_q)])
            return fun, funj, funh

        # Setup cost function, and inequality/equality constraint functions
        self.__optimization.cost, self.__optimization.cost_jacobian, self.__optimization.cost_hessian = setup_funs('cost', self.__optimization.sx_cost)
        self.__optimization.g,    self.__optimization.g_jacobian,    self.__optimization.g_hessian    = setup_funs('g', self.__optimization.sx_g)
        self.__optimization.h,    self.__optimization.h_jacobian,    self.__optimization.h_hessian    = setup_funs('h', self.__optimization.sx_h)

        self.__optimization.Ng = self.__optimization.sx_g.numel()  # number of inequality constraints
        self.__optimization.Nh = self.__optimization.sx_h.numel()  # number of equality constraints

        self.__optimization.lbg = cs.DM.zeros(self.__optimization.ineq_constraints.numel())                               # inequality constraint lower bound, i.e. 0
        self.__optimization.ubg = self.__optimization.BIG_NUMBER*cs.DM.ones(self.__optimization.ineq_constraints.numel()) # inequality constraint upper bound, i.e. inf (big number)

        self.__optimization.lbh = cs.DM.zeros(self.__optimization.eq_constraints.numel())  # equality constraint lower bound, i.e. 0
        self.__optimization.ubh = cs.DM.zeros(self.__optimization.eq_constraints.numel())  # equality constraint upper bound, i.e. 0

        # Finalize optimization problem and return
        self.__optimization.optimization_problem_is_finalized = True  # optimization problem is now finalized

        return self.__optimization

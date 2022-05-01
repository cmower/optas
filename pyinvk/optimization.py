import casadi as cs
from .sx_container import SXContainer


class Optimization:

    """

    Optimization
    ============

    Scalars/vectors:
    ----------------

    q [R^(ndof x N) ]
      joint angles

    p [R^Np]
      parameters

    BIG_NUMBER [float]
      use big number instead of inf

    methods:
    --------

    cost [R^(ndof x N) -> R]
      cost function

    g [R^(ndof x N) -> R^Ng]
      inequality constraints

    h [R^(ndof x N) -> R^Nh]
      equality constraints

    Problem formulation:
    --------------------

    min  cost(q, p)
     q

      st

          lbq     <=   q         <= ubq
          lbg=0   <=   g(q, p)   <= ubg=BIG_NUMBER
          lbh=0   <=   h(q, p)   <= ubh=0

    """

    BIG_NUMBER = 1.0e9

    def __init__(self, robot_model, N):

        # Set class attributes
        self.robot_model = robot_model
        self.ndof = robot_model.ndof
        self.N = N

        # Setup decision variables
        self.q = cs.SX.sym('q', self.ndof, N)

        # Setup optimization attributes
        self.parameters = SXContainer()
        self.cost_terms = SXContainer()
        self.ineq_constraints = SXContainer()
        self.eq_constraints = SXContainer()

        self.lbq = -self.BIG_NUMBER*cs.DM.ones(self.q.numel())
        self.ubq = self.BIG_NUMBER*cs.DM.ones(self.q.numel())

        self.optimization_problem_is_finalized = False

        # Setup class attributes that are set when finalize is called
        self.Nq = None
        self.sx_q = None
        self.sx_p = None
        self.sx_cost = None
        self.sx_g = None
        self.sx_h = None

        self.cost = None
        self.cost_jacobian = None
        self.cost_hessian = None

        self.Ng = None
        self.g = None
        self.g_jacobian = None
        self.g_hessian = None

        self.Nh = None        
        self.h = None
        self.h_jacobian = None
        self.h_hessian = None

        self.lbg = None
        self.ubg = None

        self.lbh = None
        self.ubh = None

    def finalize(self):

        if self.optimization_problem_is_finalized:
            raise RuntimeError("finalize should only be called once")

        self.sx_q = cs.vec(self.q)
        self.sx_p = self.parameters.vec()
        self.sx_cost = cs.sum1(self.cost_terms.vec())
        self.sx_g = self.ineq_constraints.vec()
        self.sx_h = self.eq_constraints.vec()

        self.Nq = self.sx_q.numel()

        fin = [self.sx_q, self.sx_p]

        def setup_funs(label, fun_sx):
            funj_sx = cs.jacobian(fun_sx, self.sx_q)
            fun = cs.Function(label, fin, [fun_sx])
            funj = cs.Function(label+'_jacobian', fin, [funj_sx])
            funh = cs.Function(label+'_hessian', fin, [cs.jacobian(funj_sx, self.sx_q)])
            return fun, funj, funh

        self.cost, self.cost_jacobian, self.cost_hessian = setup_funs('cost', self.sx_cost)
        self.g, self.g_jacobian, self.g_hessian = setup_funs('g', self.sx_g)
        self.h, self.h_jacobian, self.h_hessian = setup_funs('h', self.sx_h)

        self.Ng = self.sx_g.numel()
        self.Nh = self.sx_h.numel()

        self.lbg = cs.DM.zeros(self.ineq_constraints.numel())
        self.ubg = self.BIG_NUMBER*cs.DM.ones(self.ineq_constraints.numel())

        self.lbh = cs.DM.zeros(self.eq_constraints.numel())
        self.ubh = cs.DM.zeros(self.eq_constraints.numel())

        self.optimization_problem_is_finalized = True

        return self

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

        # Setup class attributes that are set by optimization builder
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

import casadi as cs

class Optimization:

    """

    min  f(x, p)
     x

      st

          lbg=0   <=   g(x, p)   <= ubg=big_number
          lbh=0   <=   h(x, p)   <= ubh=0


    Additional defined methods, including
    - 1st/2nd derivatives
    - upper/lower bounds
    - dimensions

    u(x, p) = [  g(x, p) ]
              [  h(x, p) ]

              [  g(x, p) ]    [  u(x, p) ]
    v(x, p) = [  h(x, p) ]  = [ -h(x, p) ]
              [ -h(x, p) ]

    """

    big_number = 1.0e9

    def __init__(self):
        self.f = None
        self.g = None
        self.h = None
        self.df = None
        self.dg = None
        self.dh = None
        self.ddf = None
        self.ddg = None
        self.ddh = None
        self.decision_variables = None
        self.parameters = None
        self.cost_terms = None
        self.ineq_constraints = None
        self.eq_constraints = None

    @property
    def nx(self):
        return self.decision_variables.numel()

    @property
    def np(self):
        return self.parameters.numel()

    @property
    def ng(self):
        return self.g.numel_out()

    @property
    def nh(self):
        return self.h.numel_out()

    @property
    def lbg(self):
        return cs.DM.zeros(self.ng)

    @property
    def ubg(self):
        return self.big_number*cs.DM.ones(self.ng)

    @property
    def lbh(self):
        return cs.DM.zeros(self.nh)

    @property
    def ubh(self):
        return cs.DM.zeros(self.nh)

    # Additional methods

    def u(self, x, p):
        return cs.vertcat(self.g(x, p), self.h(x, p))

    def du(self, x, p):
        return cs.vertcat(self.dg(x, p), self.dh(x, p))

    def ddu(self, x, p):
        return cs.vertcat(self.ddg(x, p), self.ddh(x, p))

    @property
    def nu(self):
        return self.ng + self.nh

    @property
    def lbu(self):
        return cs.vertcat(self.lbg, self.lbh)

    @property
    def ubu(self):
        return cs.vertcat(self.ubg, self.ubh)

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

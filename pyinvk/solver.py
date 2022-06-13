import casadi as cs
from abc import ABC, abstractmethod
from scipy.optimize import minimize, NonlinearConstraint

class Solver(ABC):

    """Base solver class"""

    def __init__(self, optimization):
        self.optimization = optimization
        self.x0 = cs.DM.zeros(optimization.nx)
        self.p = self.optimization.parameters.dict2vec({})

    @abstractmethod
    def setup(self, *args, **kwargs):
        """Initialize solver, called in __init__"""
        pass

    def reset_initial_seed(self, x0):
        if isinstance(x0, dict):
            self.x0 = self.optimization.decision_variables.dict2vec(x0)
        else:
            self.x0 = cs.DM(x0)

    def reset_parameters(self, p):
        if isinstance(p, dict):
            self.p = self.optimization.parameters.dict2vec(p)
        else:
            self.p = cs.DM(p)

    @abstractmethod
    def solve(self):
        """Solve the optimization problem"""
        pass

    @abstractmethod
    def stats(self):
        """Return stats from solver"""
        pass

class CasadiNLPSolver(Solver):

    """Casadi solver"""

    def setup(self, solver_name, solver_options={}):
        """Setup casadi nlp solver"""

        x = cs.SX.sym('x', self.optimization.nx)
        p = cs.SX.sym('p', self.optimization.np)

        problem = {
            'x': x,
            'p': p,
            'f': self.optimization.f(x, p),
            'g': self.optimization.u(x, p),
        }
        self.solver = cs.nlpsol('solver', solver_name, problem, solver_options)

        self.lbg = self.optimization.lbu
        self.ubg = self.optimization.ubu

    def solve(self):
        sol = self.solver(x0=self.x0, p=self.p, lbg=self.lbg, ubg=self.ubg)
        return self.optimization.decision_variables.vec2dict(sol['x'])

    def stats(self):
        return self.solver.stats()

class ScipyMinimizeSolver(Solver):

    """Scipy optimize.minimize solver"""

    methods_req_jac = {'CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'TNC',
                       'SLSQP', 'dogleg', 'trust-ncg', 'trust-krylov',
                       'trust-exact', 'trust-constr'}

    methods_req_hess = {'Newton-CG', 'dogleg', 'trust-ncg', 'trust-krylov',
                        'trust-exact', 'trust-constr'}

    methods_handle_constraints = {'COBYLA', 'SLSQP', 'trust-constr'}

    def setup(self, method='SLSQP', tol=None, options=None):

        # Input check
        if (self.optimization.nu > 0) and (method not in ScipyMinimizeSolver.methods_handle_constraints):
            raise ValueError(f"optimization problem has constraints, the method '{method}' is not suitable")

        # Setup class attributes
        self._stats = None
        self.method = method

        # Setup minimize input parameters
        self.minimize_input = {
            'fun': lambda x, p: float(self.optimization.f(x, p).toarray().flatten()[0]),
            'method': method
        }

        if tol is not None:
            self.minimize_input['tol'] = tol

        if options is not None:
            self.minimize_input['options'] = options

        if method in ScipyMinimizeSolver.methods_req_jac:
            jac = lambda x, p: self.optimization.df(x, p).toarray().flatten()
            self.minimize_input['jac'] = jac

        if method in ScipyMinimizeSolver.methods_req_hess:
            hess = lambda x, p: self.optimization.ddf(x, p).toarray()
            self.minimize_input['hess'] = hess

    def solve(self):

        if (self.method in ScipyMinimizeSolver.methods_handle_constraints):

            if self.method != 'trust-constr':
                constraints = [{
                    'type': 'ineq',
                    'fun': lambda x: self.optimization.v(x, self.p.toarray().flatten()).toarray().flatten(),
                    'jac': lambda x: self.optimization.dv(x, self.p.toarray().flatten()).toarray(),
                }]
            else:
                # Setup constraints for trust-constr
                constraints = NonlinearConstraint(
                    fun=lambda x: self.optimization.u(x, self.p).toarray().flatten(),
                    lb=self.optimization.lbu.toarray().flatten(),
                    ub=self.optimization.ubu.toarray().flatten(),
                    # jac=lambda x: self.optimization.du(x, self.p).toarray(),
                    # hess=lambda x: self.optimization.ddu(x, self.p).toarray(),
                )

            self.minimize_input['constraints'] = constraints

        self.minimize_input['args'] = self.p.toarray().flatten()
        self.minimize_input['x0'] = self.x0.toarray().flatten()

        res = minimize(**self.minimize_input)
        self._stats = res

        return self.optimization.decision_variables.vec2dict(res.x)

    def stats(self):
        return self._stats

import numpy as np
import casadi as cs
from abc import ABC, abstractmethod
from scipy.optimize import minimize, NonlinearConstraint, LinearConstraint
from scipy.sparse import csr_matrix
from .optimization import UnconstrainedQP,\
    LinearConstrainedQP, \
    NonlinearConstrainedQP, \
    UnconstrainedOptimization, \
    LinearConstrainedOptimization, \
    NonlinearConstrainedOptimization

################################################################
# Solver base class

class Solver(ABC):

    """Base solver class"""

    def __init__(self, optimization):
        self.optimization = optimization
        self.x0 = cs.DM.zeros(optimization.nx)
        self.p = cs.DM.zeros(optimization.np)

    @abstractmethod
    def setup(self, *args, **kwargs):
        """Setup solver, must return self"""
        pass

    def reset_initial_seed(self, x0):
        if isinstance(x0, dict):
            self.x0 = self.optimization.decision_variables.dict2vec(x0)
        else:
            self.x0 = cs.vec(cs.DM(x0))

    def reset_parameters(self, p):
        if isinstance(p, dict):
            self.p = self.optimization.parameters.dict2vec(p)
        else:
            self.p = cs.vec(cs.DM(p))

    @abstractmethod
    def solve(self):
        """Solve the optimization problem"""
        pass

    @abstractmethod
    def stats(self):
        """Return stats from solver"""
        pass

################################################################
# CasADi solvers (https://web.casadi.org/)

class CasADiSolver(Solver):

    """

    This is a base class for CasADi solvers. The following
    attributes are expected to be set in the setup method:
    - is_constrained
    - _solver
    - _lbg/_ubg (when is_constrained is True)

    """

    def solve(self):
        solver_input = {
            'x0': self.x0,
            'p': self.p,
        }
        if self.is_constrained:
            solver_input['lbg'] = self._lbg
            solver_input['ubg'] = self._ubg
        solution = self._solver(**solver_input)
        return self.optimization.decision_variables.vec2dict(solution['x'])

    def stats(self):
        return self._solver.stats()

class CasADiQPSolver(CasADiSolver):

    """CasADi QP solver"""

    def setup(self, solver_name, solver_options={}):

        err_msg = "CasadiQPSolver cannot solve this problem\n\nSee https://web.casadi.org/docs/#quadratic-programming"
        opt_type = type(self.optimization)
        assert opt_type in {UnconstrainedQP, LinearConstrainedQP}, err_msg

        x = self.optimization.decision_variables.vec()
        p = self.optimization.parameters.vec()

        problem = {
            'x': x,
            'p': p,
            'f': self.optimization.f(x, p),
        }
        self._lbg = None
        self._ubg = None
        self.is_constrained = opt_type == LinearConstrainedQP
        if self.is_constrained:
            problem['g'] = self.optimization.k(x, p)
            self._lbg = self.optimization.lbk
            self._ubg = self.optimization.ubk

        self._solver = cs.qpsol('solver', solver_name, problem, solver_options)

        return self

class CasADiNLPSolver(CasADiSolver):

    """Casadi NLP solver"""

    def setup(self, solver_name, solver_options={}):
        """Setup casadi nlp solver"""

        x = self.optimization.decision_variables.vec()
        p = self.optimization.parameters.vec()

        problem = {
            'x': x,
            'p': p,
            'f': self.optimization.f(x, p),
        }

        opt_type = type(self.optimization)
        self.is_constrained = opt_type not in {UnconstrainedQP, UnconstrainedOptimization}
        self._lbg = None
        self._ubg = None
        if self.is_constrained:
            problem['g'] = self.optimization.u(x, p)
            self._lbg = self.optimization.lbu
            self._ubg = self.optimization.ubu

        self._solver = cs.nlpsol('solver', solver_name, problem, solver_options)

        return self

################################################################
# OSQP solver (https://osqp.org/)

class OSQPSolver(Solver):

    def setup(self, use_warm_start=True, settings={}):

        opt_type = type(self.optimization)
        assert opt_type in {UnconstrainedQP, LinearConstrainedQP}, "OSQP cannot solve this type of problem, see https://osqp.org/docs/solver/index.html"

        osqp_installed = True
        try:
            import osqp
        except ImportError:
            osqp_installed = False

        assert osqp_installed, "could not import osqp, is it installed? Use $ pip install osqp"

        self.osqp = osqp
        self.use_warm_start = use_warm_start
        self.is_constrained = opt_type == LinearConstrainedQP

        return self

    def solve(self):

        # Setup problem
        setup_input = {
            'P': csr_matrix(self.optimization.P(self.p).toarray()),
            'q': self.optimization.q(self.p).toarray().flatten(),
        }
        if self.is_constrained:
            setup_input['l'] = self.optimization.lbr(self.p).toarray().flatten()
            setup_input['u'] = np.inf*np.ones(self.optimization.nr)
            setup_input['A'] = csr_matrix(self.optimization.M(self.p).toarray())
        self.m = self.osqp.OSQP()
        self.m.setup(**setup_input)

        # Warm start optimization
        if self.use_warm_start:
            self.m.warm_start(x=self.x0.toarray().flatten())

        # Solve problem
        self._stats = self.m.solve()
        solution = self._stats.x

        self.first_solve = False
        return self.optimization.decision_variables.vec2dict(solution)

    def stats(self):
        return self._stats

################################################################
# Scipy Minimize solvers (https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)

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

        self.constraints_are_linear = None
        if method == 'trust-constr':
            x = cs.SX.sym('x', self.optimization.nx)
            p = cs.SX.sym('p', self.optimization.np)
            u = self.optimization.u(x, p)
            self.constraints_are_linear = cs.is_linear(u, x)

    def solve(self):

        if (self.method in ScipyMinimizeSolver.methods_handle_constraints):

            if self.method != 'trust-constr':
                constraints = [{
                    'type': 'ineq',
                    'fun': lambda x: self.optimization.v(x, self.p).toarray().flatten(),
                    'jac': lambda x: self.optimization.dv(x, self.p).toarray(),
                }]
            else:
                # Setup constraints for trust-constr
                # Note, if constraints are linear in x then use
                # LinearConstraint, otherwise use NonlinearConstraint

                if self.constraints_are_linear:
                    x = cs.SX.sym('x', self.optimization.nx)
                    u = self.optimization.u(x, self.p)
                    A = csr_matrix(cs.DM(cs.jacobian(u, x)).toarray())
                    c = self.optimization.u(cs.DM.zeros(self.optimization.nx), self.p).toarray().flatten()
                    lb = -c
                    ub = self.optimization.big_number*cs.np.ones(self.optimization.nu)
                    constraints = LinearConstraint(A=A, lb=lb, ub=ub)
                else:
                    constraints = NonlinearConstraint(
                        fun=lambda x: self.optimization.u(x, self.p).toarray().flatten(),
                        lb=self.optimization.lbu.toarray().flatten(),
                        ub=self.optimization.ubu.toarray().flatten(),
                        jac=lambda x: self.optimization.du(x, self.p).toarray(),
                    )

            self.minimize_input['constraints'] = constraints

        self.minimize_input['args'] = self.p.toarray().flatten()
        self.minimize_input['x0'] = self.x0.toarray().flatten()

        res = minimize(**self.minimize_input)
        self._stats = res

        return self.optimization.decision_variables.vec2dict(res.x)

    def stats(self):
        return self._stats

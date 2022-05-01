import abc
import casadi as cs
from scipy.optimize import minimize, NonlinearConstraint
from sensor_msgs.msg import JointState

class Solver(abc.ABC):

    def __init__(self, optimization):
        assert optimization.optimization_problem_is_finalized, "optimization problem is not finalized"
        self._optimization = optimization
        self.__called_reset = False
        self.__called_solve = False
        self.__called_setup = False
        self._q_init = None

    @abc.abstractmethod
    def _setup(self, *args, **kwargs):
        pass

    def setup(self, *args, **kwargs):
        self._setup(*args, **kwargs)
        self.__called_setup = True

    def reset(self, init_seed, parameters):
        assert self.__called_setup, "you must call setup before reset"
        self._q_init = cs.vec(init_seed)
        self._p = self._optimization.parameters.dict2vec(parameters)
        self.__called_solve = False
        self.__called_reset = True

    @abc.abstractmethod
    def _solve(self):
        pass

    def solve(self):
        assert self.__called_setup, "you must call setup before solve"
        assert self.__called_reset, "you must call reset before solve"
        sol = self._solve()
        self.__called_solve = True
        return sol

    @abc.abstractmethod
    def _stats(self):
        pass

    def stats(self):
        assert self.__called_setup, "you must call setup before setup"
        assert self.__called_solve, "you must call solve before stats"
        return self._stats()

    def solution2msgs(self, solution):
        return [
            JointState(
                name=self._optimization.robot_model.joint_names,
                position=solution[:,i].toarray().flatten().tolist(),
            )
            for i in range(solution.shape[1])

        ]

class CasadiSolver(Solver):

    def _setup(self, solver_name, solver_options={}):
        assert self._optimization.optimization_problem_is_finalized, "optimization must be finalized"

        constraints = self._optimization.ineq_constraints+self._optimization.eq_constraints
        problem = {
            'x': self._optimization.sx_q,
            'p': self._optimization.sx_p,
            'f': self._optimization.sx_cost,
            'g': constraints.vec(),
        }
        self.__lbx = self._optimization.lbq
        self.__ubx = self._optimization.ubq
        self.__lbg = cs.vertcat(self._optimization.lbg, self._optimization.lbh)
        self.__ubg = cs.vertcat(self._optimization.ubg, self._optimization.ubh)
        self.__solver = solver = cs.nlpsol('solver', solver_name, problem, solver_options)

    def _solve(self):
        sol = self.__solver(
            x0=self._q_init, p=self._p,
            lbx=self.__lbx, ubx=self.__ubx,
            lbg=self.__lbg, ubg=self.__ubg,
        )
        self.__stats = self.__solver.stats()
        self.__stats['solution'] = sol
        return cs.reshape(sol['x'], self._optimization.ndof, self._optimization.N)

    def _stats(self):
        return self.__stats


class ScipySolver(Solver):

    def _setup(self, method=None, tol=None, options=None):
        self.__minimize_input = dict(
            fun=lambda q: self._optimization.cost(q, self._p).toarray().flatten()[0],
            method=method,
            jac=lambda q: self._optimization.cost_jacobian(q, self._p).toarray().flatten(),
            hess=lambda q: self._optimization.cost_hessian(q, self._p).toarray(),
            bounds=[(minq, maxq) for minq, maxq in zip(self._optimization.lbq.toarray().flatten(), self._optimization.ubq.toarray().flatten())],
            tol=tol,
            options=options,
        )
        # x0 set during solve

        # Setup constraints
        constraints = None
        if method == 'trust-constr':
            constraints = self.__setup_constraints_trust_constr()
        elif method == 'SLSQP':
            constraints = self.__setup_constraints_SLSQP()
        elif method == 'COBYLA':
            constraints = self.__setup_constraints_COBYLA()
        self.__minimize_input['constraints'] = constraints

        self.__stats = None

    def __setup_constraints_trust_constr(self):

        # Setup
        if (self._optimization.Ng+self._optimization.Nh) == 0: return None
        c = []

        # Inequality constraints
        if self._optimization.Ng > 0:
            c.append(
                NonlinearConstraint(
                    fun=lambda q: self._optimization.g(q, self._p).toarray().flatten(),
                    lb=self._optimization.lbg.toarray().flatten(),
                    ub=self._optimization.ubg.toarray().flatten(),
                    jac=lambda q: self._optimization.g_jacobian(q, self._p).toarray(),
                )
            )

        # Equality constraints
        if self._optimization.Nh > 0:
            c.append(
                NonlinearConstraint(
                    fun=lambda q: self._optimization.h(q, self._p).toarray().flatten(),
                    lb=self._optimization.lbh.toarray().flatten(),
                    ub=self._optimization.ubh.toarray().flatten(),
                    jac=lambda q: self._optimization.h_jacobian(q, self._p).toarray(),
                )
            )

        return c

    def __setup_constraints_SLSQP(self):

        # Setup
        if (self._optimization.Ng+self._optimization.Nh) == 0: return None
        c = []        

        # Inequality constraints
        if self._optimization.Ng > 0:
            c.append({
                'type': 'ineq',
                'fun': lambda q: self._optimization.g(q, self._p).toarray().flatten(),
                'jac': lambda q: self._optimization.g_jacobian(q, self._p).toarray(),
            })

        # Equality constraints
        if self._optimization.Nh > 0:
            c.append({
                'type': 'eq',
                'fun': lambda q: self._optimization.h(q, self._p).toarray().flatten(),
                'jac': lambda q: self._optimization.h_jacobian(q, self._p).toarray(),
            })

        return c

    def __setup_constraints_COBYLA(self):

        # Setup
        if (self._optimization.Ng+self._optimization.Nh) == 0: return None
        c = []

        # Inequality constraints
        if self._optimization.Ng > 0:
            c.append({
                'type': 'ineq',
                'fun': lambda q: self._optimization.g(q, self._p).toarray().flatten(),
                'jac': lambda q: self._optimization.g_jacobian(q, self._p).toarray(),
            })

        # Equality constraints
        if self._optimization.Nh > 0:
            c.append({
                'type': 'ineq',
                'fun': lambda q: self._optimization.h(q, self._p).toarray().flatten(),
                'jac': lambda q: self._optimization.h_jacobian(q, self._p).toarray(),
            })
            c.append({
                'type': 'ineq',
                'fun': lambda q: -self._optimization.h(q, self._p).toarray().flatten(),
                'jac': lambda q: -self._optimization.h_jacobian(q, self._p).toarray(),
            })

        return c

    def _solve(self):


        class Callback:
            def __init__(self):
                self.i = 0

            def __call__(self, *args, **kwargs):
                print("iter:", self.i)
                self.i += 1

        self.__minimize_input['callback'] = Callback()
        
        self.__minimize_input['x0'] = self._q_init.toarray().flatten()
        self.__stats = minimize(**self.__minimize_input)
        return cs.reshape(self.__stats.x, self._optimization.ndof, self._optimization.N)

    def _stats(self):
        return self.__stats

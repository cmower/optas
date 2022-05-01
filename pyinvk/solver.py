import abc
import casadi as cs
from scipy.optimize import minimize, NonlinearConstraint
from sensor_msgs.msg import JointState

class Solver(abc.ABC):

    """Base solver class"""

    def __init__(self, optimization):
        assert optimization.optimization_problem_is_finalized, "optimization problem is not finalized"
        self._optimization = optimization
        self.__called_reset = False
        self.__called_solve = False
        self.__called_setup = False
        self._q_init = None

    @abc.abstractmethod
    def _setup(self, *args, **kwargs):
        """Setup the solver, must be implemented in child-class"""
        pass

    def setup(self, *args, **kwargs):
        """Setup the solver class"""
        self._setup(*args, **kwargs)
        self.__called_setup = True

    def reset(self, init_seed, parameters):
        """Reset the optimization problem"""
        assert self.__called_setup, "you must call setup before reset"
        self._q_init = cs.vec(init_seed)
        self._p = self._optimization.parameters.dict2vec(parameters)
        self.__called_solve = False
        self.__called_reset = True

    @abc.abstractmethod
    def _solve(self):
        """Call solver to solve the optimization problem, must be implemented in child-class"""
        pass

    def solve(self):
        """Solve the optimization problem"""
        assert self.__called_setup, "you must call setup before solve"
        assert self.__called_reset, "you must call reset before solve"
        sol = self._solve()
        self.__called_solve = True
        return sol

    @abc.abstractmethod
    def _stats(self):
        """Return stats from the previous call to solve, must be implemented in child-class"""
        pass

    def stats(self):
        """Return stats from solver"""
        assert self.__called_setup, "you must call setup before setup"
        assert self.__called_solve, "you must call solve before stats"
        return self._stats()

    def solution_to_ros_joint_state_msgs(self, solution):
        """Convert a solution to a list of ROS sensor_msgs/JointState messages"""
        return [
            JointState(
                name=self._optimization.robot_model.joint_names,
                position=solution[:,i].toarray().flatten().tolist(),
            )
            for i in range(solution.shape[1])

        ]

class CasadiSolver(Solver):

    """Casadi solver"""

    def _setup(self, solver_name, solver_options={}):
        """Setup casadi solver"""
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
        """Solve the problem using casadi"""
        sol = self.__solver(
            x0=self._q_init, p=self._p,
            lbx=self.__lbx, ubx=self.__ubx,
            lbg=self.__lbg, ubg=self.__ubg,
        )
        self.__stats = self.__solver.stats()
        self.__stats['solution'] = sol
        return cs.reshape(sol['x'], self._optimization.ndof, self._optimization.N)

    def _stats(self):
        """Return the stats from the casadi solver"""
        return self.__stats


class ScipySolver(Solver):

    """Scipy solver"""

    def _setup(self, method=None, tol=None, options=None):
        """Setup the scipy solver"""
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
        method_handles_constraints = method in {'trust-constr', 'SLSQP', 'COBYLA'}
        n_constraints = self._optimization.Ng+self._optimization.Nh
        if (not method_handles_constraints) and (n_constraints > 0):
            raise RuntimeError(f"you have defined constraints but the method '{method}' does not handle constraints")

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
        """Return constraints in format accepted by trust-constr"""

        # Setup
        if (self._optimization.Ng+self._optimization.Nh) == 0: return []
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
        """Return constraints in format accepted by SLSQP"""        

        # Setup
        if (self._optimization.Ng+self._optimization.Nh) == 0: return []
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
        """Return constraints in format accepted by COBYLA"""        

        # Setup
        if (self._optimization.Ng+self._optimization.Nh) == 0: return []
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
        """Solve the optimization problem using scipy"""

        # class Callback:
        #     def __init__(self):
        #         self.i = 0

        #     def __call__(self, *args, **kwargs):
        #         print("iter:", self.i)
        #         self.i += 1

        # self.__minimize_input['callback'] = Callback()

        self.__minimize_input['x0'] = self._q_init.toarray().flatten()
        self.__stats = minimize(**self.__minimize_input)
        return cs.reshape(self.__stats.x, self._optimization.ndof, self._optimization.N)

    def _stats(self):
        """Return stats from scipy solver"""
        return self.__stats

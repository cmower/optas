import casadi as cs
from .sx_container import SXContainer
from .spatialmath import vectorize_args, arrayify_args

class OptimizationBuilder:

    def __init__(self, T, robots={}, tasks={}, optimize_time=False, derivs_align=False):

        # Input check
        optimize_time = 1 if not optimize_time else 2
        assert T > Tmin, f"T must be greater than {Tmin}"
        assert all(d >= 0 for d in derivs), "derivs must be greater than or equal to zero"
        assert all('dim' in value for tasks.values()), "each task must contain 'dim'"

        # Class attributes
        self.T = T
        self.robots = robots
        self.derivs = derivs
        self.tasks = tasks
        self.optimize_time = optimize_time
        self.derivs_align = derivs_align

        # Setup decision variables
        self._decision_variables = SXContainer()

        for name, robot in robots.items():
            for d in robot.time_derivs:
                n = self.joint_state_name(name, d)
                t = T-d if not derivs_align else T
                self.add_decision_variables(n, robot.ndof, t)

        for name, task in tasks.items():
            dim = task['dim']
            for d in self._get_task_time_derivs(task):
                n = self.task_state_name(name, d)
                t = T-d if not derivs_align else T
                self.add_decision_variables(n, dim, t)

        if optimize_time:
            self.add_decision_variables('dt', T-1)

        # Setup containers for parameters, cost terms, ineq/eq constraints
        self._parameters = SXContainer()
        self._cost_terms = SXContainer()
        self._lin_eq_constraints = SXContainer()
        self._lin_ineq_constraints = SXContainer()
        self._ineq_constraints = SXContainer()
        self._eq_constraints = SXContainer()

    #
    # Joint/task state name generation
    #

    def joint_state_name(self, robot_name, time_deriv):
        assert robot_name in self.robots.keys(), "robot name does not exist"
        return robot_name + '/' + 'd'*time_deriv + 'q'

    def task_state_name(self, task_name, time_deriv):
        assert task_name in self.tasks.keys(), "task name does not exist"
        return 'd'*time_deriv + name

    #
    # Get joint/task states
    #

    def get_joint_state(self, robot_name, t, time_deriv=0):
        joint_states = self.get_joint_states(robot_name, time_deriv)
        return joint_states[:, t]

    def get_joint_states(self, robot_name, time_deriv=0):
        assert time_deriv in self.robots[robot_name].time_derivs, f"robot called {robot_name} was not specified with time derivatives to order {time_deriv}"
        n = self.joint_state_name(robot_name, time_deriv)
        return self._decision_variables[n]

    def get_task_state(self, task_name, t, time_deriv=0):
        task_states = self.get_task_states(task_name, time_deriv)
        return task_states[:, t]

    def get_task_states(self, task_name, time_deriv=0):
        task = self.tasks[task_name]
        time_derivs = self._get_task_time_derivs(task)
        assert time_deriv in time_derivs, f"task called {task_name} was not specified with time derivatives to order {time_deriv}"
        n = self.task_state_name(task_name, time_deriv)
        return self._decision_variables[n]

    #
    # Helper methods
    #

    @staticmethod
    def _get_task_time_derivs(task):
        return task.get('time_derivs', [0])

    def get_dt(self):
        assert self.optimize_time, "optimize_time should be True in the OptimizationBuilder interface"
        return self._decision_variables['dt']

    def _x(self):
        return self._decision_variables.vec()

    def _p(self):
        return self._parameters.vec()

    def _is_linear(self, y):
        return cs.is_linear(y, self._x())

    def _cost(self):
        return cs.sum1(self.cost_terms.vec())

    def is_cost_quadratic(self):
        return cs.is_quadratic(self._cost(), self._x())

    #
    # Upate optimization problem
    #

    def add_decision_variables(self, name, m=1, n=1):
        x = cs.SX.sym(name, m, n)
        self._decision_variables[name] = x
        return x

    def add_parameter(self, name, m=1, n=1):
        p = cs.SX.sym(name, m, n)
        self._parameters[name] = p
        return p

    @vectorize_args
    def add_cost_term(self, name, cost_term):
        m, n = cost_term.shape
        assert m==1 and n==1, "cost term must be scalar"
        self._cost_terms[name] = cost_term

    @arrayify_args
    def add_geq_ineq_constraint(self, name, lhs, rhs=None):
        """lhs >= rhs"""
        if rhs is None:
            rhs = cs.DM.zeros(*lhs.shape)
        self.add_leq_ineq_constraint(name, rhs, lhs):

    @arrayify_args
    def add_leq_ineq_constraint(self, name, lhs, rhs=None):
        """lhs <= rhs"""
        if rhs is None:
            rhs = cs.DM.zeros(*lhs.shape)
        diff = rhs - lhs  # diff >= 0
        if self._is_linear(diff):
            self._lin_ineq_constraints[name] = diff
        else:
            self._ineq_constraints[name] = diff

    @arrayify_args
    def add_eq_constraint(self, name, lhs, rhs=None):
        if rhs is None:
            rhs = cs.DM.zeros(*lhs.shape)
        diff = rhs - lhs  # diff >= 0
        if self._is_linear(diff):
            self._lin_eq_constraints[name] = diff
        else:
            self._eq_constraints[name] = diff

    #
    # Common cost terms
    #

    def add_nominal_configuration_cost_term(self, cost_term_name, robot_name, qnom=None, w=1.):
        robot = self.robots[robot_name]
        if qnom is None:
            lo = robot.lower_actuated_joint_limits
            up = robot.upper_actuated_joint_limits
            qnom = 0.5*(lo + up)

        qnom = cs.vec(qnom)

        w = cs.vec(w)
        if w.shape[0] == 1 and w.shape[1] == 1:
            w = cs.DM.ones(robot.ndof)
        else:
            assert w.shape[0] == robot.ndof, f"w must be scalar or have {robot.ndof} elements"

        # Create nominal function
        W = cs.diag(w)
        q_ = cs.SX.sym('q', robot.ndof)
        qdiff_ = q_ - qnom
        cost_term = cs.Function('nominal_cost', [q_], [qdiff_.T @ W @ qdiff_]).map(self.T)

        # Compute cost term
        Q = self.get_joint_states(robot_name, 0)
        c = cost_term(Q)

        # Add cost term
        self.add_cost_term(cost_term_name, c)

    #
    # Common constraints
    #

    def ensure_positive_dt(self, constraint_name='__ensure_positive_dt__'):
        """dt >= 0"""
        assert self.optimize_time, "optimize_time should be True in the OptimizationBuilder interface"
        self.add_geq_ineq_constraint(constaint_name, self.get_dt())

    def _integr(self, m, n):
        xd = cs.SX.sym('xd', m)
        x0 = cs.SX.sym('x0', m)
        x1 = cs.SX.sym('x1', m)
        dt = cs.SX.sym('dt')
        integr = cs.Function('integr', [x0, x1, xd, dt], [x0 + dt*xd - x1])
        return integr.map(n)

    def integrate_joint_state(self, robot_name, time_deriv, dt=None):

        if self.optimize_time and dt is not None:
            raise ValueError("dt is given but user specified optimize_time as True")
        if not self.optimize_time and dt is None:
            raise ValueError("dt is not given")

        robot = self.robots[robot_name]
        qd = self.get_joint_states(robot_name, time_deriv)
        q = self.get_joint_states(robot_name, time_deriv-1)
        n = qd.shape[1]
        if self.derivs_align:
            n -= 1
            qd = qd[:, :-1]

        if dt is None:
            dt = self.get_dt()[:n]
        else:
            dt = cs.vec(dt)
            assert dt.shape[0] in {1, n}, f"dt should be scalar or have {n} elements"
            if dt.shape[0] == 1:
                dt = dt*cs.DM.ones(n)
        dt = cs.vec(dt).T  # ensure 1-by-n

        integr = self._integr(robot.ndof, n)
        name = f'__integrate_joint_state_{robot_name}_{time_deriv}__'
        self.add_eq_constraint(name, integr(q[:, :-1], q[:, 1:], qd, dt))

    def integrate_task_state(self, task_name, time_deriv, dt=None):

        if self.optimize_time and dt is not None:
            raise ValueError("dt is given but user specified optimize_time as True")
        if not self.optimize_time and dt is None:
            raise ValueError("dt is not given")

        task = self.tasks[task_name]
        assert time_deriv in _get_task_time_derivs(task), f"{time_deriv=}, does not exist"
        yd = self.get_task_states(label, time_deriv)
        y = self.get_task_states(label, time_deriv-1)
        n = yd.shape[1]
        if self.derivs_align:
            n -= 1
            yd = yd[:, :-1]

        if self.optimize_time:
            dt = self.get_dt()[:n]
        else:
            dt = cs.vec(dt)
            assert dt.shape[0] in {1, n}, f"dt should be scalar or have {n} elements"
            if dt.shape[0] == 1:
                dt = dt*cs.DM.ones(n)
        dt = cs.vec(dt).T  # ensure 1-by-n

        integr = self._integr(task['dim'], n)
        name = f'__integrate_task_state_{task_name}_{time_deriv}__'
        self.add_eq_constraint(name, integr(y[:, :-1], y[:, 1:], yd, dt))

    def enforce_joint_position_limit_constraints(self, robot_name):
        robot = self.robots[robot_name]
        qlo = robot.lower_actuated_joint_limits
        qup = robot.upper_actuated_joint_limits
        q = self.get_joint_states(robot_name)
        self.add_leq_ineq_constraint(f'__{robot_name}_joint_limit_lower__', qlo, q)
        self.add_leq_ineq_constraint(f'__{robot_name}_joint_limit_upper__', q, qup)

    def enforce_joint_velocity_limit_constraints(self, robot_name):
        qd = self.get_joint_states(robot_name, 1)
        qdlim = self.robots[robot_name].velocity_actuated_joint_limits
        self.add_leq_ineq_constraint(f'__{robot_name}_joint_velocity_limit_lower__', -qdlim, qd)
        self.add_leq_ineq_constraint(f'__{robot_name}_joint_velocity_limit_upper__',  qd, qdlim)

    #
    # Main build method
    #

    def build(self):

        # Setup optimization
        nlin = self._lin_ineq_constraints.numel()+self._lin_eq_constraints.numel() # total no. linear constraints
        nnlin = self._ineq_constraints.numel()+self._eq_constraints.numel() # total no. nonlinear constraints

        if self.is_cost_quadratic():
            # True -> use QP formulation
            if nnlin > 0:
                opt = QuadraticCostNonlinearConstraints(
                    self._decision_variables,
                    self._parameters,
                    self._cost_terms,
                    self._lin_eq_constraints,
                    self._lin_ineq_constraints,
                    self._eq_constraints,
                    self._ineq_constraints,
                )
            elif nlin > 0:
                opt = QuadraticCostLinearConstraints(
                    self._decision_variables,
                    self._parameters,
                    self._cost_terms,
                    self._lin_eq_constraints,
                    self._lin_ineq_constraints,
                )
            else:
                opt = QuadraticCostUnconstrained(
                    self._decision_variables,
                    self._parameters,
                    self._cost_terms,
                )
        else:
            # False -> use (nonlinear) Optimization formulation
            if nnlin > 0:
                opt = NonlinearCostNonlinearConstraints(
                    self._decision_variables,
                    self._parameters,
                    self._cost_terms,
                    self._lin_eq_constraints,
                    self._lin_ineq_constraints,
                    self._eq_constraints,
                    self._ineq_constraints,
                )
            elif nlin > 0:
                opt = NonlinearCostLinearConstraints(
                    self._decision_variables,
                    self._parameters,
                    self._cost_terms,
                    self._lin_eq_constraints,
                    self._lin_ineq_constraints,
                )
            else:
                opt = NonlinearCostUnconstrained(
                    self._decision_variables,
                    self._parameters,
                    self._cost_terms,
                )

        return opt

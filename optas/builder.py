import casadi as cs
from .sx_container import SXContainer
from .spatialmath import vectorize_args, arrayify_args
from .optimization import *

class OptimizationBuilder:

    def __init__(self, T, robots=[], tasks=[], optimize_time=False, derivs_align=False):

        # Input check
        assert T > 0, f"T must be strictly positive"

        # Class attributes
        self.T = T
        self._models = robots + tasks
        self.optimize_time = optimize_time
        self.derivs_align = derivs_align

        # Setup decision variables
        self._decision_variables = SXContainer()
        for model in self._models:
            for d in model.time_derivs:
                n = model.state_name(d)
                t = T-d if not derivs_align else T
                self.add_decision_variables(n, model.dim, t)

        if optimize_time:
            self.add_decision_variables('dt', T-1)

        # Setup containers for parameters, cost terms, ineq/eq constraints
        self._parameters = SXContainer()
        self._cost_terms = SXContainer()
        self._lin_eq_constraints = SXContainer()
        self._lin_ineq_constraints = SXContainer()
        self._ineq_constraints = SXContainer()
        self._eq_constraints = SXContainer()


    def get_model_names(self):
        return [model.name for model in self._models]


    def get_model_index(self, name):
        return self.get_model_names().index(name)


    def get_model(self, name):
        idx = self.get_model_index(name)
        return self._models[idx]


    def get_model_state(self, name, t, time_deriv=0):
        states = self.get_model_states(name, time_deriv)
        return states[:, t]


    def get_model_states(self, name, time_deriv=0):
        model = self.get_model(name)
        assert time_deriv in model.time_derivs, f"model '{name}', was not specified with time derivative to order {time_deriv}"
        name = model.state_name(time_deriv)
        return self._decision_variables[name]


    def get_dt(self):
        assert self.optimize_time, "to call get_dt(..), optimize_time should be True in the OptimizationBuilder interface"
        return self._decision_variables['dt']


    def _x(self):
        return self._decision_variables.vec()


    def _p(self):
        return self._parameters.vec()


    def _is_linear(self, y):
        return cs.is_linear(y, self._x())


    def _cost(self):
        return cs.sum1(self._cost_terms.vec())


    def is_cost_quadratic(self):
        return cs.is_quadratic(self._cost(), self._x())

    #
    # Upate optimization problem
    #

    def add_decision_variables(self, name, m=1, n=1, is_discrete=False):
        x = cs.SX.sym(name, m, n)
        self._decision_variables[name] = x
        if is_discrete:
            self._decision_variables.variable_is_discrete(name)
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
    def add_geq_inequality_constraint(self, name, lhs, rhs=None):
        """lhs >= rhs"""
        if rhs is None:
            rhs = cs.DM.zeros(*lhs.shape)
        self.add_leq_ineq_constraint(name, rhs, lhs)


    @arrayify_args
    def add_leq_inequality_constraint(self, name, lhs, rhs=None):
        """lhs <= rhs"""
        if rhs is None:
            rhs = cs.DM.zeros(*lhs.shape)
        diff = rhs - lhs  # diff >= 0
        if self._is_linear(diff):
            self._lin_ineq_constraints[name] = diff
        else:
            self._ineq_constraints[name] = diff


    @arrayify_args
    def add_bound_inequality_constraint(self, name, lhs, mid, rhs):
        """lhs <= mid <= rhs"""
        self.add_leq_ineq_constraint(name+'_l', lhs, mid)
        self.add_leq_ineq_constraint(name+'_r', mid, rhs)


    @arrayify_args
    def add_equality_constraint(self, name, lhs, rhs=None):
        """lhs == rhs"""
        if rhs is None:
            rhs = cs.DM.zeros(*lhs.shape)
        diff = rhs - lhs  # diff == 0
        if self._is_linear(diff):
            self._lin_eq_constraints[name] = diff
        else:
            self._eq_constraints[name] = diff


    #
    # Common cost terms
    #


    def add_nominal_configuration_cost_term(self, cost_term_name, robot_name, qnom=None, w=1.):
        robot = self.get_model(robot_name)
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


    def integrate_model_states(self, name, time_deriv, dt=None):

        if self.optimize_time and dt is not None:
            raise ValueError("dt is given but user specified optimize_time as True")
        if not self.optimize_time and dt is None:
            raise ValueError("dt is not given")

        model = self.get_model(name)
        xd = self.get_model_states(name, time_deriv)
        x = self.get_model_states(name, time_deriv)
        n = xd.shape[1]
        if self.derivs_align:
            n -= 1
            qd = qd[:, :-1]

        if self.optimize_time:
            dt = self.get_dt()[:n]
        else:
            dt = cs.vec(dt)
            assert dt.shape[0] in {1, n}, f"dt should be scalar or have {n} elements"
            if dt.shape[0] == 1:
                dt = dt*cs.DM.ones(n)
        dt = cs.vec(dt).T  # ensure 1-by-n

        integr = self._integr(model.dim, n)
        name = f'__integrate_model_states_{name}_{time_deriv}__'

        self.add_eq_constraint(name, integr(x[:, :-1], x[:, 1:], xd, dt))


    def enforce_model_limits(self, name, time_deriv=0):
        x = self.get_model_states(name, time_deriv)
        xlo, xup = self.get_model(name).get_limits(time_deriv)
        n = f'__{name}_model_limit_{time_deriv}__'
        self.add_ineq_bound_constraint(n, xlo, x, xup)


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

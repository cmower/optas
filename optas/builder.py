import casadi as cs
from .sx_container import SXContainer
from .spatialmath import arrayify_args
from .optimization import *

class OptimizationBuilder:

    """

    OptimizationBuilder allows you to build/specify an optimization problem.

    """

    def __init__(self, T, robots=[], tasks=[], optimize_time=False, derivs_align=False):
        """OptimizationBuilder constructor.

        Syntax
        ------

        builder = optas.OptimizationBuilder(T, robots, tasks, optimize_time, derivs_align)

        Parameters
        ----------

        T (int):
          Number of time steps for the trajectory.

        robots (list):
          A list of robot models.

        tasks (list):
          A list of task models.

        optimize_time (bool):
          When true, a trajectory of dt variables are included in the
          decision variables. Default is False.

        derivs_align (bool):
          When true, the time derivatives align for each time
          step. Default is False.

        """

        # Input check
        assert T > 0, f"T must be strictly positive"

        if not isinstance(robots, list):
            robots = [robots] # allow user to pass a single robot

        if not isinstance(tasks, list):
            tasks = [tasks] # all user to pass a single task

        # Class attributes
        self.T = T
        self._models = robots + tasks
        self.optimize_time = optimize_time
        self.derivs_align = derivs_align

        # Ensure T is sufficiently large
        if not derivs_align:

            # Get max time deriv
            all_time_derivs = []
            for m in self._models:
                all_time_derivs += m.time_derivs
            max_time_deriv = max(all_time_derivs)

            # Check T is large enough
            Tmin = max_time_deriv+1
            assert T >= Tmin, f"{T=} is too low, it should be at least {Tmin}"

        model_names = [m.get_name() for m in self._models]
        is_unique_names = len(model_names) == len(set(model_names))
        assert is_unique_names, "each model should have a unique name"

        # Setup decision variables
        self._decision_variables = SXContainer()
        for model in self._models:
            for d in model.time_derivs:
                n = model.state_name(d)
                if model.T is None:
                    t = T-d if not derivs_align else T
                else:
                    t = model.T
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
        """Return the names of each model."""
        return [model.name for model in self._models]


    def get_model_index(self, name):
        """Return the index of the model in the list of models.

        Syntax
        ------

        idx = builder.get_model_index(name)

        Parameters
        ----------

        name (string)
            Name of the model.

        Returns
        -------

        idx (int)
            Index of the model in the list of models.

        """
        return self.get_model_names().index(name)


    def get_model(self, name):
        """Return the model with given name.

        Syntax
        ------

        model = builder.get_model(name)

        Parameters
        ----------

        name (string)
            Name of the model.

        Returns
        -------

        model (optas.models.Model)
            A task or robot model.

        """
        return self._models[self.get_model_index(name)]


    def get_model_state(self, name, t, time_deriv=0):
        """Get the model state at a given time.

        Syntax
        ------

        state = builder.get_model_state(name, t, time_deriv=0)

        Parameters
        ----------

        name (string)
            Name of the model.

        t (int)
            Index of the desired state.

        time_deriv (int)
            The time-deriviative required (i.e. position is 0, velocity is 1, etc.)

        Returns
        -------

        state (casadi.SX, with shape dim-by-1)
            The state vector where dim is the model dimension.

        """
        states = self.get_model_states(name, time_deriv)
        return states[:, t]


    def get_model_states(self, name, time_deriv=0):
        """Get the full state trajectory for a given model.

        Syntax
        ------

        states = builder.get_model_states(name, time_deriv=0)

        Parameters
        ----------

        name (string)
            Name of the model.

        time_deriv (int)
            The time-deriviative required (i.e. position is 0, velocity is 1, etc.)

        Returns
        -------

        states (casadi.SX, with shape dim-by-T)
            The state vector where dim is the model dimension, and T is the number of time-steps in the trajectory.

        """
        model = self.get_model(name)
        assert time_deriv in model.time_derivs, f"model '{name}', was not specified with time derivative to order {time_deriv}"
        name = model.state_name(time_deriv)
        return self._decision_variables[name]


    def get_dt(self):
        """When optimizing time, then this method returns the trajectory of dt variables."""
        assert self.optimize_time, "to call get_dt(..), optimize_time should be True in the OptimizationBuilder interface"
        return self._decision_variables['dt']


    def _x(self):
        """Return the decision variables as a casadi.SX vector."""
        return self._decision_variables.vec()


    def _p(self):
        """Return the parameters as a casadi.SX vector."""
        return self._parameters.vec()


    def _is_linear(self, y):
        """Returns true if y is a linear function of the decision variables."""
        return cs.is_linear(y, self._x())


    def _cost(self):
        """Returns the cost function."""
        return cs.sum1(self._cost_terms.vec())


    def is_cost_quadratic(self):
        """True when cost function is quadratic in the decision variables, False otherwise."""
        return cs.is_quadratic(self._cost(), self._x())

    #
    # Upate optimization problem
    #

    def add_decision_variables(self, name, m=1, n=1, is_discrete=False):
        """Add decision variables to the optimization problem.

        Syntax
        ------

        d = builder.add_decision_variables(name, m=1, n=1, is_discrete=False)

        Parameters
        ----------

        name (string)
            Name of decision variable array.

        m (int)
            Number of rows in decision variable array.

        n (int)
            Number of columns in decision variable array.

        is_discret (bool)
            If true, then the decision variables are treated as discrete variables.

        Return
        ------

        d (casadi.SX)
            Array of the decision variables.

        """
        x = cs.SX.sym(name, m, n)
        self._decision_variables[name] = x
        if is_discrete:
            self._decision_variables.variable_is_discrete(name)
        return x


    def add_parameter(self, name, m=1, n=1):
        """Add a parameter to the optimization problem.

        Syntax
        ------

        p = builder.add_parameter(name, m=1, n=1)

        Parameters
        ----------

        name (string)
            Name of parameter array.

        m (int)
            Number of rows in parameter array.

        n (int)
            Number of columns in parameter array.

        Return
        ------

        p (casadi.SX)
            Array of the parameters.

        """
        p = cs.SX.sym(name, m, n)
        self._parameters[name] = p
        return p


    @arrayify_args
    def add_cost_term(self, name, cost_term):
        """Add cost term to the optimization problem.

        Syntax
        ------

        builder.add_cost_term(name, cost_term)

        Parameters
        ----------

        name (string)
            Name for cost function.

        cost_term (casadi.SX)
            Cost term, must be an array with shape 1-by-1.

        """
        cost_term = cs.vec(cost_term)
        m, n = cost_term.shape
        assert m==1 and n==1, "cost term must be scalar"
        self._cost_terms[name] = cost_term


    @arrayify_args
    def add_geq_inequality_constraint(self, name, lhs, rhs=None):
        """Add the inequality constraint lhs >= rhs to the optimization problem.

        Syntax
        ------

        builder.add_geq_inequality_constraint(name, lhs, rhs=None)

        Parameters
        ----------

        name (string)
            Name for the constraint.

        lhs (array-like: casadi.SX, casadi.DM, or list or numpy.ndarray)
            Left-hand side for the inequality constraint.

        rhs (None, or array-like: casadi.SX, casadi.DM, or list or numpy.ndarray)
            Right-hand side for the inequality constraint. If None, then it is replaced with the zero array with the same shape as lhs.

        """
        if rhs is None:
            rhs = cs.DM.zeros(*lhs.shape)
        self.add_leq_inequality_constraint(name, rhs, lhs)


    @arrayify_args
    def add_leq_inequality_constraint(self, name, lhs, rhs=None):
        """Add the inequality constraint lhs <= rhs to the optimization problem.

        Syntax
        ------

        builder.add_leq_inequality_constraint(name, lhs, rhs=None)

        Parameters
        ----------

        name (string)
            Name for the constraint.

        lhs (array-like: casadi.SX, casadi.DM, or list or numpy.ndarray)
            Left-hand side for the inequality constraint.

        rhs (None, or array-like: casadi.SX, casadi.DM, or list or numpy.ndarray)
            Right-hand side for the inequality constraint. If None, then it is replaced with the zero array with the same shape as lhs.

        """
        if rhs is None:
            rhs = cs.DM.zeros(*lhs.shape)
        diff = rhs - lhs  # diff >= 0
        if self._is_linear(diff):
            self._lin_ineq_constraints[name] = diff
        else:
            self._ineq_constraints[name] = diff


    @arrayify_args
    def add_bound_inequality_constraint(self, name, lhs, mid, rhs):
        """Add the inequality constraint lhs <= mid <= rhs to the optimization problem.

        Syntax
        ------

        builder.add_bound_inequality_constraint(name, lhs, mid, rhs)

        Parameters
        ----------

        name (string)
            Name for the constraint.

        lhs (array-like: casadi.SX, casadi.DM, or list or numpy.ndarray)
            Left-hand side for the inequality constraint.

        mid (array-like: casadi.SX, casadi.DM, or list or numpy.ndarray)
            Middle part of the inequality constraint.

        rhs (None, or array-like: casadi.SX, casadi.DM, or list or numpy.ndarray)
            Right-hand side for the inequality constraint.

        """
        self.add_leq_inequality_constraint(name+'_l', lhs, mid)
        self.add_leq_inequality_constraint(name+'_r', mid, rhs)


    @arrayify_args
    def add_equality_constraint(self, name, lhs, rhs=None):
        """Add the equality constraint lhs == rhs to the optimization problem.

        Syntax
        ------

        builder.add_equality_constraint(name, lhs, rhs=None)

        Parameters
        ----------

        name (string)
            Name for the constraint.

        lhs (array-like: casadi.SX, casadi.DM, or list or numpy.ndarray)
            Left-hand side for the inequality constraint.

        rhs (None, or array-like: casadi.SX, casadi.DM, or list or numpy.ndarray)
            Right-hand side for the inequality constraint. If None, then it is replaced with the zero array with the same shape as lhs.

        """
        if rhs is None:
            rhs = cs.DM.zeros(*lhs.shape)
        diff = rhs - lhs  # diff == 0
        if self._is_linear(diff):
            self._lin_eq_constraints[name] = diff
        else:
            self._eq_constraints[name] = diff

    #
    # Common constraints
    #


    def ensure_positive_dt(self):
        """Specifies the constraint dt >= 0 when optimize_time=True."""
        assert self.optimize_time, "optimize_time should be True in the OptimizationBuilder interface"
        self.add_geq_inequality_constraint('__ensure_positive_dt__', self.get_dt())


    def _integr(self, m, n):
        """Returns an integration function where m is the state dimension, and n is the number of trajectory points."""
        xd = cs.SX.sym('xd', m)
        x0 = cs.SX.sym('x0', m)
        x1 = cs.SX.sym('x1', m)
        dt = cs.SX.sym('dt')
        integr = cs.Function('integr', [x0, x1, xd, dt], [x0 + dt*xd - x1])
        return integr.map(n)


    def integrate_model_states(self, name, time_deriv, dt=None):
        """Integrates the model states over time.

        Syntax
        ------

        builder.integrate_model_states(name, time_deriv, dt=None)

        Parameters
        ----------

        name (string)
            Name of the model.

        time_deriv (int)
            The time-deriviative required (i.e. position is 0, velocity is 1, etc.).

        dt (None, float, or array-like: casadi.SX, casadi.DM, or list or numpy.ndarray)
            Integration time step.

        """

        if self.optimize_time and dt is not None:
            raise ValueError("dt is given but user specified optimize_time as True")
        if not self.optimize_time and dt is None:
            raise ValueError("dt is not given")

        model = self.get_model(name)
        xd = self.get_model_states(name, time_deriv)
        x = self.get_model_states(name, time_deriv-1)
        n = x.shape[1]
        if self.derivs_align:
            xd = xd[:, :-1]

        if self.optimize_time:
            dt = self.get_dt()[:n]
        else:
            dt = cs.vec(dt)
            assert dt.shape[0] in {1, n-1}, f"dt should be scalar or have {n-1} elements"
            if dt.shape[0] == 1:
                dt = dt*cs.DM.ones(n-1)
        dt = cs.vec(dt).T  # ensure dt is 1-by-(n-1) array

        integr = self._integr(model.dim, n-1)
        name = f'__integrate_model_states_{name}_{time_deriv}__'
        self.add_equality_constraint(name, integr(x[:, :-1], x[:, 1:], xd, dt))


    def enforce_model_limits(self, name, time_deriv=0, lo=None, up=None):
        """Enforce model limits.

        Syntax
        ------

        builder.enforce_model_limits(name, time_deriv=0)

        Parameters
        ----------

        name (string)
            Name of model.

        time_deriv (int)
            The time-deriviative required (i.e. position is 0, velocity is 1, etc.)

        lo (None, or array-like: casadi.SX, casadi.DM, or list or numpy.ndarray)
            Lower limits, if None then model limits specified in the model class are used.

        up (None, or array-like: casadi.SX, casadi.DM, or list or numpy.ndarray)
            Upper limits, if None then model limits specified in the model class are used.

        """
        x = self.get_model_states(name, time_deriv)
        xlo = lo
        xup = up
        if (xlo is None) or (xup is None):
            mlo, mup = self.get_model(name).get_limits(time_deriv)
            if xlo is None:
                xlo = mlo
            if xup is None:
                xup = mup
        n = f'__{name}_model_limit_{time_deriv}__'
        self.add_bound_inequality_constraint(n, xlo, x, xup)

    def initial_configuration(self, name, init=None, time_deriv=0):
        """Set initial configuration.

        Syntax
        ------

        builder.initial_configuration(name, init=None, time_deriv=0)

        Parameters
        ----------

        name (string)
            Name of model.

        init (array-like: casadi.SX, casadi.DM, or list or numpy.ndarray)
            Initial configuration.

        time_deriv (int)
            The time-deriviative required (i.e. position is 0, velocity is 1, etc.)

        """
        t0 = 0
        x0 = self.get_model_state(name, t0, time_deriv=time_deriv)
        n = f'__{name}_initial_configuration_{time_deriv}__'
        self.add_equality_constraint(n, lhs=x0, rhs=init)  # init will be zero when None


    def fix_configuration(self, name, config=None, time_deriv=0, t=0):
        """Fix configuration.

        Syntax
        ------

        builder.fix_configuration(name, config=None, time_deriv=0, t=0)

        Parameters
        ----------

        name (string)
            Name of model.

        config (array-like: casadi.SX, casadi.DM, or list or numpy.ndarray)
            The configuration.

        time_deriv (int)
            The time-deriviative required (i.e. position is 0, velocity is 1, etc.)

        t (int)
            Index for the configuration in trajectory (by default this is the first element but it could also be the last for example in moving horizon estimation).

        """
        x0 = self.get_model_state(name, t, time_deriv=time_deriv)
        n = f'__{name}_fix_configuration_{time_deriv}_{t}__'
        self.add_equality_constraint(n, lhs=x0, rhs=config)  # config will be zero when None


    #
    # Main build method
    #

    def build(self):
        """Build the optimization problem."""

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

import casadi as cs
import numpy as np
from typing import Dict, List, Optional, Union
from .robot_model import RobotModel
from .sx_container import SXContainer
from .optimization import QuadraticCostUnconstrained,\
    QuadraticCostLinearConstraints,\
    QuadraticCostNonlinearConstraints,\
    NonlinearCostUnconstrained,\
    NonlinearCostLinearConstraints,\
    NonlinearCostNonlinearConstraints

class OptimizationBuilder:

    """The OptimizationBuilder class is used to build an optimization problem."""

    def __init__(
            self,
            T: Optional[int]=1,
            robots: Optional[Dict[str, RobotModel]]={},
            qderivs: Optional[List[int]]=[0],
            ylabels: Optional[List[str]]=[],
            ydims: Optional[Union[int,List[int]]]=None,
            yderivs: Optional[List[int]]=None,
            derivs_align: Optional[bool]=False,
            optimize_time: Optional[bool]=False,
    ):

        # Input check
        if optimize_time:
            assert T>=2, "T must be greater than 1"
        else:
            assert T >= 1, "T must be strictly positive"
        if robots:
            assert qderivs, "when robots are given, you must supply qderivs"
        maxqderiv = None
        if len(qderivs):
            assert min(qderivs) >= 0, "all values in qderivs should be non-negative"
            maxqderiv = max(qderivs)
            assert maxqderiv >= 0, "maximum qderiv must be non-negative"
        if ylabels:
            assert yderivs, "when ylabels is given, you must supply yderivs"
        maxyderiv = None
        if yderivs:
            assert min(yderivs) >= 0, "all values in yderivs should be non-negative"
            maxyderiv = max(yderivs)
            assert maxyderiv >= 0, "maximum yderiv must be non-negative"

        if not derivs_align and (maxqderiv or maxyderiv):

            derivs = []
            if isinstance(maxqderiv, int):
                derivs.append(maxqderiv)
            if isinstance(maxyderiv, int):
                derivs.append(maxyderiv)

            maxderiv = max(derivs)
            assert T > maxderiv, f"{T=} must be greater than {maxderiv}"

        if isinstance(ydims, int):
            assert ydims > 0, f"{ydims=} must strictly positive"
            ydims = [ydims]*len(ylabels)
        elif isinstance(ydims, int):
            assert len(ydims) == len(ylabels), f"incorrect length for ydims, expected {len(ylabels)} got {len(ydims)}"
        elif ydims is None:
            pass
        else:
            raise TypeError(f"did not recognize {type(ydims)=}")

        # Set class attributes
        self.T = T
        self.robots = robots
        self.qderivs = qderivs
        self.ylabels = ylabels
        self.ydims = ydims
        self.yderivs = yderivs
        self.derivs_align = derivs_align
        self.optimize_time = optimize_time

        # Setup decision variables
        self.decision_variables = SXContainer()
        for robot_name, robot in robots.items():
            for qderiv in qderivs:
                n = self.qstatename(robot_name, qderiv)
                self.decision_variables[n] = cs.SX.sym(n, robot.ndof, T-qderiv if not derivs_align else T)

        if ylabels:
            for label, ndim in zip(ylabels, ydims):
                for yderiv in yderivs:
                    n = self.ystatename(label, yderiv)
                    self.decision_variables[n] = cs.SX.sym(n, ndim, T-yderiv if not derivs_align else T)

        if optimize_time:
            self.decision_variables['dt'] = cs.SX.sym('dt', T-1)

        # Setup containers for parameters, cost terms, ineq/eq constraints
        self.parameters = SXContainer()
        self.cost_terms = SXContainer()
        self.lin_eq_constraints = SXContainer()
        self.lin_ineq_constraints = SXContainer()
        self.ineq_constraints = SXContainer()
        self.eq_constraints = SXContainer()

    def print_desc(self):

        def print_(name, d):
            n = d.numel()
            print(name.capitalize()+f' ({n}):')
            for label, value in d.items():
                print(f"  {label} {value.shape}")

        if self.is_cost_function_quadratic():
            print("Cost function is quadratic.")
        else:
            print("Cost function is nonlinear.")
        if self.decision_variables.numel() > 0:
            print_("decision variables", self.decision_variables)
        if self.parameters.numel() > 0:
            print_("parameters", self.parameters)
        if self.cost_terms.numel() > 0:
            print_("cost terms", self.cost_terms)
        if self.lin_eq_constraints.numel() > 0:
            print_("linear equality constraints", self.lin_eq_constraints)
        if self.lin_ineq_constraints.numel() > 0:
            print_("linear inequality constraints", self.lin_ineq_constraints)
        if self.eq_constraints.numel() > 0:
            print_("equality constraints", self.eq_constraints)
        if self.ineq_constraints.numel() > 0:
            print_("inequality constraints", self.ineq_constraints)

    @staticmethod
    def qstatename(robot_name: str, qderiv: int) -> str:
        """Returns the state name for a given time derivative of q.

        The state name is used to index the decision variables.

        Parameters
        ----------

        robot_name : str
            Name of the robot, as given in the robots parameter in the
            class constructor.

        qderiv : int
            The derivative order.

        Returns
        -------

        state_name : str
            A string in the format "{robot_name}/[D]q" where "[D]" is
            qderiv-number of "d"'s, e.g. if robot_name="robot" and
            qderiv=2, then the state name will be "robot/qdd".

        """
        return robot_name + '/' + 'd'*qderiv + 'q'

    @staticmethod
    def ystatename(label: str, yderiv: int) -> str:
        return 'd'*yderiv + label

    def _is_linear(self, y):
        x = self.decision_variables.vec()
        return cs.is_linear(y, x)

    def get_qstate(
            self,
            robot_name: str,
            t: int,
            qderiv: Optional[int]=0) -> cs.casadi.SX:
        """Return the configuration state for a given time deriviative of q.

        Parameters
        ----------

        robot_name : str
            Name of the robot, as given in the robots parameter in the
            class constructor.

        t : int
            Time step in the trajectory.

        qderiv : int
            The derivative order.

        Returns
        -------

        state : casadi.casadi.SX
            The state for a given robot, at a given time derivative,
            for a given time-step.

        """
        return self.get_qstates(robot_name, qderiv)[:, t]

    def get_qstates(self, robot_name: str, qderiv: Optional[int]=0):
        assert qderiv in self.qderivs, f"{qderiv=}, qderiv must be in {self.qderivs}"
        return self.decision_variables[self.qstatename(robot_name, qderiv)]

    def get_ystate(self, label: str, t: int, yderiv: int):
        self.get_ystates(label, yderiv)[:, t]

    def get_ystates(self, label: str, yderiv: int):
        assert yderiv in self.yderivs, f"{yderiv=}, yderiv myst be in {self.yderivs}"
        return self.decision_variables[self.ystatename(label, yderiv)]

    def add_decision_variables(
            self,
            name: str,
            m: Optional[int]=1,
            n: Optional[int]=1) -> cs.casadi.SX:
        """Add a decision variable to the optimization problem.

        Parameters
        ----------

        name : str
            Name for the decision variable.

        m : int (default is 1)
            Number of rows in the decision variable array.

        n : int (default is 1)
            Number of columns in the decision variable array.

        Returns
        -------

        decision_variable : casadi.casadi.SX
            The decision variable SX array.
        """
        x = cs.SX.sym(name, m, n)
        self.decision_variables[name] = x
        return x

    def add_cost_term(self, name: str, cost_term: cs.casadi.SX) -> None:
        """Add a cost term to the optimization problem.

        When the optimization problem is built, the cost function is
        given by a sum of cost terms.

        Parameters
        ----------

        name : str
            Name for the cost term.

        cost_term : casadi.casadi.SX
            The cost term as a SX variable. Note, this must be scalar,
            i.e. with shape (1, 1).

        """
        assert cost_term.shape == (1, 1), "cost terms must be scalars"
        self.cost_terms[name] = cost_term

    def add_nominal_configuration_cost_term(self, robot_name, qnominal, w=1.0):

        ndof = self.robots[robot_name].ndof
        qnominal = cs.SX(qnominal)
        assert qnominal.shape[0] == ndof, f"qnominal incorrect length, expected {ndof} got {qnominal.shape[0]}"

        # Handle weight
        w = cs.DM(w)
        nw = w.shape[0]
        if nw == 1:
            W = w*cs.SX.eye(ndof)
        else:
            assert nw == ndof, f"w incorrect length, expected {ndof}, got {nw}"
            W = cs.diag(w)

        # Create nominal function
        q_ = cs.SX.sym('q', ndof)  # joint state
        qn_ = cs.SX.sym('qn', ndof)  # nominal joint state
        qdiff_ = q_ - qn_
        cost = cs.Function('nominal_cost', [q_, qn_], [qdiff_.T @ W @ qdiff_]).map(self.T)

        # Handler
        Q = self.get_joint_states(robot_name, qderiv=0)
        QN = cs.diag(qnominal) @ cs.DM.ones(ndof, self.T)

        self.add_cost_term('__nominal_configuration_cost__', cs.sum2(cost(Q, QN)))


    def add_parameter(self, name: str, m: Optional[int]=1, n : Optional[int]=1) -> cs.casadi.SX:
        """Add a parameter to the optimization problem.

        Parameters
        ----------

        name : str
            Name for the parameter. Note, this name is used when
            referencing the parameter for the reset_parameters method
            in the Solver class.

        m : int (default is 1)
            Number of rows in the parameter array.

        n : int (default is 1)
            Number of columns in the parameter array.

        Returns
        -------

        param : casadi.casadi.SX
            The SX parameter array.

        """
        p = cs.SX.sym(name, m, n)
        self.parameters[name] = p
        return p

    def add_ineq_constraint(
            self,
            name: str,
            c1: Union[cs.casadi.SX, cs.casadi.DM],
            c2: Optional[cs.casadi.SX]=None,) -> None:
        """Adds an inequality constraint to the optimization problem.

        Note, the constraint is defined as

            lbc <= c <= ubc.

        This is included in the linear constraint variable container
        as two constraints, i.e.

            c - lbc >= 0, and
            ubc - c >= 0.

        If the constraint is linear then it is added as a linear
        constraint (i.e. it will be added to the lin_constraints
        attribute, otherwise it will be logged in the ineq_constraints
        attribute.

        Parameters
        ----------

        name : str
            Name for the constraint.

        lbc : Union[cs.casadi.SX, cs.casadi.DM]
            Lower bound for constraint.

        c : Union[cs.casadi.SX, cs.casadi.DM]
            The constraint array.

        ubc : Union[cs.casadi.SX, cs.casadi.DM]
            Upper bound for the constraint.

        """

        if c2 is None:
            # c1 >= 0
            LBC = cs.DM.zeros(*c1.shape)
            UBC = c1
        else:
            LBC = c1
            UBC = c2

        diff = UBC - LBC  # LBC <= UBC   =>   diff = UBC - LBC >= 0
        if self._is_linear(diff):
            self.lin_ineq_constraints[name] = diff
        else:
            self.ineq_constraints[name] = diff

    def add_eq_constraint(
            self,
            name: str,
            c1: Union[cs.casadi.SX, cs.casadi.DM],
            c2: Optional[Union[cs.casadi.SX, cs.casadi.DM]]=None) -> None:
        """Adds an equality constraint to the optimization problem.

        Note, the constraint is defined as lhsc == rhsc.

        If the constraint is linear then it is added as a linear
        constraint (i.e. it will be added to the lin_constraints
        attribute, otherwise it will be logged in the eq_constraints
        attribute.

        Parameters
        ----------

        name : str
            Name for the constraint.

        lhsc : Union[cs.casadi.SX, cs.casadi.DM]
            Left hand side of the equality constraint.

        rhsc : Union[cs.casadi.SX, cs.casadi.DM] (default is None)
            Right hand side of the equality constraint. This is
            optional, if it is None then it is assumed to be the zero
            array with the same shape as lhsc.

        """

        if c2 is None:
            LHS = c1
            RHS = cs.DM.zeros(*c1.shape)
        else:
            LHS = c1
            RHS = c2

        eq = LHS - RHS
        if self._is_linear(eq):
            self.lin_eq_constraints[name] = eq
        else:
            self.eq_constraints[name] = eq

    def _create_integration_function(self, ndim, n):
        xd = cs.SX.sym('xd', ndim)
        x0 = cs.SX.sym('x0', ndim)
        x1 = cs.SX.sym('x1', ndim)
        dt = cs.SX.sym('dt')
        integr = cs.Function('integr', [x0, x1, xd, dt], [x0 + dt*xd - x1])
        return integr.map(n)

    def add_dynamic_q_integr_constraints(self, robot_name, qderiv, dt=None):
        assert qderiv > 0, "qderiv must be greater than 0"

        qd = self.get_qstates(robot_name, qderiv)
        q = self.get_qstates(robot_name, qderiv-1)
        n = qd.shape[1]
        if self.derivs_align:
            n -= 1
            qd = qd[:, :-1]


        if self.optimize_time:
            assert dt is None, "optimize_time was given as true and also dt is given"
            dt = self.decision_variables['dt'][:n]
        else:
            assert dt is not None, "dt is required when optimize_time is False"

            dt = cs.vec(dt)
            if dt.shape[0] == 1:
                dt = dt*cs.DM.ones(n)
            else:
                assert df.shape[0]==n, f"incorrect number of elements in dt array, expected {n} got {df.shape[0]}"
            dt = cs.vec(dt).T

        integr = self._create_integration_function(self.robots[robot_name].ndof, n)
        self.add_eq_constraint(
            f'__dynamic_q_integr_{robot_name}_{qderiv}__',
            integr(q[:,:-1], q[:,1:], qd, dt),
        )

    def add_dynamic_y_integr_constraints(self, label, yderiv, dt=None):
        assert yderiv > 0, "yderiv must be greater than 0"

        yd = self.get_ystates(label, yderiv)
        y = self.get_ystates(label, yderiv-1)
        n = yd.shape[1]
        if self.derivs_align:
            n -= 1
            yd = yd[:, :-1]

        if self.optimize_time:
            assert dt is None, "optimize_time was given as true and also dt is given"
            dt = self.decision_variables['dt'][:n]
        else:
            assert dt is not None, "dt is required when optimize_time is False"

            dt = cs.vec(dt)
            if dt.shape[0] == 1:
                dt = dt*cs.DM.ones(n)
            else:
                assert df.shape[0]==n, f"incorrect number of elements in dt array, expected {n} got {df.shape[0]}"
            dt = cs.vec(dt).T

        integr = self._create_integration_function(self.ydims[self.ylabels.index(label)], n)
        self.add_eq_constraint(
            f'__dynamic_y_integr_{label}_{yderiv}__',
            integr(y[:,:-1], y[:,1:], yd, dt),
        )

    def add_fk_constraint(self, robot_name, label, parent, child):
        y = self.get_ystates(label, 0)
        q = self.get_qstates(robot_name)

        forward_kinematics = self.robots[robot_name].fk(parent, child)
        ydim = y.shape[0]
        if ydim == 3:
            fk = forward_kinematics['pos'].map(self.T)
        elif ydim == 6:
            fk = forward_kinematics['pos_eul'].map(self.T)
        elif ydim == 7:
            fk = forward_kinematics['pos_quat'].map(self.T)
        else:
            raise ValueError(f"dimension for y is not recognized, got {ydim} expected either 3, 6, or 7")

        self.add_eq_constraint(f'__fk_constraint_{robot_name}_{label}_{parent}_{child}__', fk(q), y)

    def add_joint_position_limit_constraints(self, robot_name):
        assert robot_name in self.robots, f"did not recognize robot '{robot_name}'"
        robot = self.robots[robot_name]
        qlo = robot.lower_actuated_joint_limits
        qup = robot.upper_actuated_joint_limits
        q = self.get_qstates(robot_name)
        self.add_ineq_constraint(f'{robot_name}_joint_limit_lower', qlo, q)
        self.add_ineq_constraint(f'{robot_name}_joint_limit_upper', q, qup)

    def is_cost_function_quadratic(self):

        # Get decision variables and parameters as SX column vectors
        x = self.decision_variables.vec()
        p = self.parameters.vec()

        # Get forward functions
        f = cs.sum1(self.cost_terms.vec())

        return cs.is_quadratic(f, x)

    def build(self):
        """Build the optimization problem.

        For the given

        - decision variables (x),
        - parameters (p),
        - cost function (cost), and
        - constraints (k/g/h)

        the approrpriate optimization problem is built. The type of
        the optimization problem is chosen depending on the cost
        function and constraints. The available problem types are as
        follows. Note,

        - dash "'" means transpose and full-stop "." means the dot
          product, or matrix-matrix/matrix-vector multiplication,
        - 0 is used to denote the zero array with an appropriate
          dimension,
        - equality constraints can be represented by inequality
          constraints, i.e. lhs == rhs is equivalent to lhs <= rhs and
          lhs >= rhs, and
        - the problem type determines the solvers that are available
          to solve the problem.

        UnconstrainedQP:

                min cost(x, p) where cost(x, p) = x'.P(p).x + x'.q(p)
                 x

            The problem is unconstrained, and has quadratic cost
            function - note, P and q are derived from the given cost
            function (you don't have to explicitly state P/q).

        LinearConstrainedQP:

                min cost(x, p) where cost(x, p) = x'.P(p).x + x'.q(p)
                 x

                subject to k(x, p) = M(p).x + c(p) >= 0

            The problem is constrained by only linear constraints and
            has a quadratic cost function - note, P/M and q/c are
            derived from the given cost function and constraints (you
            don't have to explicitly state P/q/M/c).

        NonlinearConstrainedQP:

                min cost(x, p) where cost(x) = x'.P(p).x + x'.q
                 x

                subject to

                    k(x, p) = M(p).x + c(p) >= 0,
                    g(x) == 0, and
                    h(x) >= 0

            The problem is constrained by nonlinear constraints and
            has a quadratic cost function - note, P/M and q/c are
            derived from the given cost function and constraints (you
            don't have to explicitly state P/q/M/c).

        UnconstrainedOptimization:

                min cost(x, p)
                 x

            The problem is unconstrained and the cost function is
            nonlinear in x.

        LinearConstrainedOptimization:

                min cost(x, p)
                 x

                subject to k(x, p) = M(p).x + c(p) >= 0

            The problem is constrained with linear constraints and has
            a nonlinear cost function in x.

        NonlinearConstrainedOptimization:

                min cost(x, p)
                 x

                subject to

                    k(x, p) = M(p).x + c(p) >= 0,
                    g(x) == 0, and
                    h(x) >= 0

            The problem is constrained by nonlinear constraints and
            has a nonlinear cost function.

        Returns
        -------

        opt_problem : Union[UnconstrainedQP,
                            LinearConstrainedQP,
                            NonlinearConstrainedQP,
                            UnconstrainedOptimization,
                            LinearConstrainedOptimization,
                            NonlinearConstrainedOptimization]
            The optimization problem of either one of the above
            types. The problem type determines what costraints (and
            their type) are available and also the structure of the
            cost function.

        """

        # Setup optimization
        nlin = self.lin_ineq_constraints.numel()+self.lin_eq_constraints.numel()
        nnlin = self.ineq_constraints.numel()+self.eq_constraints.numel()

        if self.is_cost_function_quadratic():
            # True -> use QP formulation
            if nnlin > 0:
                opt = QuadraticCostNonlinearConstraints(
                    self.decision_variables,
                    self.parameters,
                    self.cost_terms,
                    self.lin_eq_constraints,
                    self.lin_ineq_constraints,
                    self.eq_constraints,
                    self.ineq_constraints,
                )
            elif nlin > 0:
                opt = QuadraticCostLinearConstraints(
                    self.decision_variables,
                    self.parameters,
                    self.cost_terms,
                    self.lin_eq_constraints,
                    self.lin_ineq_constraints,
                )
            else:
                opt = QuadraticCostUnconstrained(
                    self.decision_variables,
                    self.parameters,
                    self.cost_terms,
                )
        else:
            # False -> use (nonlinear) Optimization formulation
            if nnlin > 0:
                opt = NonlinearCostNonlinearConstraints(
                    self.decision_variables,
                    self.parameters,
                    self.cost_terms,
                    self.lin_eq_constraints,
                    self.lin_ineq_constraints,
                    self.eq_constraints,
                    self.ineq_constraints,
                )
            elif nlin > 0:
                opt = NonlinearCostLinearConstraints(
                    self.decision_variables,
                    self.parameters,
                    self.cost_terms,
                    self.lin_eq_constraints,
                    self.lin_ineq_constraints,
                )
            else:
                opt = NonlinearCostUnconstrained(
                    self.decision_variables,
                    self.parameters,
                    self.cost_terms,
                )

        return opt

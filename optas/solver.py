import osqp
import cvxopt
import numpy as np
import casadi as cs
from dataclasses import dataclass
from abc import ABC, abstractmethod
from scipy.interpolate import interp1d
from scipy.optimize import minimize, NonlinearConstraint, LinearConstraint
from scipy.sparse import csc_matrix
from .optimization import (
    Optimization,
    QuadraticCostUnconstrained,
    QuadraticCostLinearConstraints,
    QuadraticCostNonlinearConstraints,
    NonlinearCostUnconstrained,
    NonlinearCostLinearConstraints,
    NonlinearCostNonlinearConstraints,
)
from .models import RobotModel

## Optimization problem types with quadratic cost.
QP_COST = {
    QuadraticCostUnconstrained,
    QuadraticCostLinearConstraints,
    QuadraticCostNonlinearConstraints,
}


## Optimization problem types with nonlinear cost.
NL_COST = {
    NonlinearCostUnconstrained,
    NonlinearCostLinearConstraints,
    NonlinearCostNonlinearConstraints,
}

## Optimization problem types that are unconstrained.
UNCONSTRAINED_OPT = {QuadraticCostUnconstrained, NonlinearCostUnconstrained}


## Optimization problem types that are constrained.
CONSTRAINED_OPT = {
    QuadraticCostLinearConstraints,
    QuadraticCostNonlinearConstraints,
    NonlinearCostLinearConstraints,
    NonlinearCostNonlinearConstraints,
}

################################################################
# Solver base class


class Solver(ABC):
    """! Base solver interface class"""

    def __init__(self, optimization: Optimization, error_on_fail: bool = False):
        """! Constructor for the base Solver class.

        @param optimization The optimization problem created by calling the build method of the OptimizationBuilder class.
        @param error_on_fail When True, after solve() is called, if the solver did not converge then a RuntimeError is thrown. Default is False.
        @return An instance of the Solver class.
        """

        ## Instance of the optimization problem.
        self.opt = optimization

        ## Initial guess for the optimization problem (set using reset_initial_seed).
        self.x0 = cs.DM.zeros(optimization.nx)

        ## Parameter vector.
        self.p = cs.DM.zeros(optimization.np)

        ## Parameter dictionary.
        self._p_dict = {}

        ## When True, after solve() is called, if the solver did not converge then a RuntimeError is thrown.
        self._error_on_fail = error_on_fail

    @property
    def opt_type(self) -> type:
        """Optimization type."""
        return type(self.opt)

    @abstractmethod
    def setup(self, *args, **kwargs):
        """Setup solver, note this method must return self."""
        pass

    def reset_initial_seed(self, x0):
        """Reset initial seed for the optimization problem.

        Parameters
        ----------

        x0 : Dict[str, ArrayLike]
            The initial seed.

        """
        self.x0 = self.opt.decision_variables.dict2vec(x0)

    def reset_parameters(self, p):
        """Reset the parameters for the optimization problem.

        Parameters
        ----------

        p : Union[Dict[str, ArrayLike], ArrayLike]
            Specifies the parameters.

        """
        self.p = self.opt.parameters.dict2vec(p)
        self._p_dict = self.opt.parameters.vec2dict(self.p)

    @abstractmethod
    def _solve(self):
        """Solve the optimization problem and return the optimal decision variables as an array."""
        pass

    def solve(self):
        """Solve the optimization problem."""
        solution = self.opt.decision_variables.vec2dict(self._solve())

        if self._error_on_fail and (not self.did_solve()):
            raise RuntimeError("Solver failed!")

        # Add full model state to the solution dictionary
        for model in self.opt.models:
            for d in model.time_derivs:
                n_s = model.state_name(d)
                n_s_x = model.state_optimized_name(d)
                if isinstance(model, RobotModel):
                    if model.num_param_joints > 0:
                        n_s_p = model.state_parameter_name(d)
                        t = solution[n_s_x].shape[1]
                        solution[n_s] = cs.DM.zeros(model.dim, t)
                        solution[n_s][model.optimized_joint_indexes, :] = solution[
                            n_s_x
                        ]
                        solution[n_s][model.parameter_joint_indexes, :] = self._p_dict[
                            n_s_p
                        ]
                    else:
                        solution[n_s] = solution[n_s_x]
                else:
                    solution[n_s] = solution[n_s_x]

        return solution

    @abstractmethod
    def stats(self):
        """Return stats from solver. The return type is specifc to the solver used."""
        pass

    def violated_constraints(self, x, p):
        x = self.opt.decision_variables.dict2vec(x)
        p = self.opt.parameters.dict2vec(p)

        @dataclass
        class ViolatedConstraint:
            label: str
            ctype: str
            diff: cs.DM
            pattern: cs.DM

            def __str__(self):
                return f"\n{self.label} [{self.ctype}]:\n{self.pattern}\n"

            def __repr__(self):
                info = str(self)
                max_width = max(len(line) for line in info.split("\n"))
                return "=" * max_width + info + "-" * max_width + "\n"

            @property
            def verbose_info(self):
                info = str(self)
                info += f"{self.diff}\n"
                return info

        lin_eq_violated_constraints = []
        for label, sx_var in self.opt.lin_eq_constraints.items():
            fun = cs.Function("fun", [self.opt.x, self.opt.p], [sx_var])
            diff = fun(x, p)
            lin_eq_violated_constraints.append(
                ViolatedConstraint(label, "lin_eq", diff, diff >= 0.0)
            )

        eq_violated_constraints = []
        for label, sx_var in self.opt.eq_constraints.items():
            fun = cs.Function("fun", [self.opt.x, self.opt.p], [sx_var])
            diff = fun(x, p)
            eq_violated_constraints.append(
                ViolatedConstraint(label, "eq", diff, diff >= 0.0)
            )

        lin_ineq_violated_constraints = []
        for label, sx_var in self.opt.lin_ineq_constraints.items():
            fun = cs.Function("fun", [self.opt.x, self.opt.p], [sx_var])
            diff = fun(x, p)
            lin_ineq_violated_constraints.append(
                ViolatedConstraint(label, "lin_ineq", diff, diff >= 0.0)
            )

        ineq_violated_constraints = []
        for label, sx_var in self.opt.ineq_constraints.items():
            fun = cs.Function("fun", [self.opt.x, self.opt.p], [sx_var])
            diff = fun(x, p)
            ineq_violated_constraints.append(
                ViolatedConstraint(label, "ineq", diff, diff >= 0.0)
            )

        return (
            lin_eq_violated_constraints,
            eq_violated_constraints,
            lin_ineq_violated_constraints,
            ineq_violated_constraints,
        )

    @staticmethod
    def interpolate(traj, T, **interp_args):
        """Interpolate a trajectory where T is the time duration of the trajectory."""
        assert isinstance(
            traj, cs.casadi.DM
        ), f"traj is incorrect type, got '{type(traj)}', expected casadi.DM'"
        t = np.linspace(0, T, traj.shape[1])
        return interp1d(t, traj.toarray(), **interp_args)

    @abstractmethod
    def did_solve(self):
        """Returns true when the solver solved the previous problem, false otherwise."""
        pass

    @abstractmethod
    def number_of_iterations(self):
        """Returns the number of iterations required to solve the problem."""
        pass

    def evaluate_cost(self, x, p):
        """Evaluates the cost function for given decision variables x and parameters p."""
        x = self.opt.decision_variables.dict2vec(x)
        p = self.opt.parameters.dict2vec(p)
        return self.opt.f(x, p)

    def evaluate_cost_terms(self, x, p):
        """Evaluates each cost term for given decision variables and parameters."""

        x = self.opt.decision_variables.dict2vec(x)
        p = self.opt.parameters.dict2vec(p)

        @dataclass
        class CostTerm:
            label: str
            cost: float

            def __str__(self):
                return f"\n{self.label}: {self.cost}"

            def __repr__(self):
                info = str(self)
                max_width = max(len(line) for line in info.split("\n"))
                return "=" * max_width + info + "-" * max_width + "\n"

        cost_terms = []
        for label, sx_var in self.opt.cost_terms.items():
            fun = cs.Function("fun", [self.opt.x, self.opt.p], [sx_var])
            c = fun(x, p)
            cost_terms.append(c)

        return cost_terms


################################################################
# CasADi solvers (https://web.casadi.org/)


class CasADiSolver(Solver):

    """This is a base class for CasADi solver interfaces."""

    nlp_solvers = {"ipopt", "knitro", "snopt", "worhp", "scpgen", "sqpmethod"}
    qp_solvers = {"cplex", "gurobi", "ooqp", "qpoases", "sqic", "nlp"}

    def setup(self, solver_name, solver_options={}):
        # Setup problem
        x = self.opt.decision_variables.vec()
        p = self.opt.parameters.vec()

        problem = {
            "x": x,
            "p": p,
            "f": self.opt.f(x, p),
        }

        # Setup constraints
        self._lbg = None
        self._ubg = None

        if self.opt_type in CONSTRAINED_OPT:
            problem["g"] = self.opt.v(x, p)
            self._lbg = self.opt.lbv
            self._ubg = self.opt.ubv

        # Get solver interface
        if solver_name in self.qp_solvers:
            sol = cs.qpsol
        elif solver_name in self.nlp_solvers:
            sol = cs.nlpsol
        else:
            raise ValueError(f"did not recognize {solver_name=}")

        # Check for discrete variables
        if self.opt.decision_variables.has_discrete_variables():
            solver_options["discrete"] = self.opt.decision_variables.discrete()

        # Initialize solver
        self._solver = sol("solver", solver_name, problem, solver_options)

        return self

    def _solve(self):
        solver_input = {"x0": self.x0, "p": self.p}
        if self.opt_type in CONSTRAINED_OPT:
            solver_input["lbg"] = self._lbg
            solver_input["ubg"] = self._ubg
        self._solution = self._solver(**solver_input)
        self._stats = self._solver.stats()
        self._stats["solution"] = self._solution
        return self._solution["x"]

    def stats(self):
        return self._stats

    def did_solve(self):
        return self._stats["success"]

    def number_of_iterations(self):
        return self._stats["iter_count"]


################################################################
# OSQP solver (https://osqp.org/)


class OSQPSolver(Solver):

    """OSQP solver interface."""

    OSQP_SOLVED = osqp.constant("OSQP_SOLVED")

    def setup(self, use_warm_start, settings={}):
        """Setup solver.

        Parameters
        ----------

        use_warm_start : bool (default is True)
            When true, the initial seed x0 is used as a warm start.

        settings : Dict
            Settings that are passed to OSQP.

        Returns
        -------

        solver : OSQPSolver
            The instance of the solve (i.e. self).

        """
        assert self.opt_type in QP_COST, "OSQP cannot solve this type of problem"
        self.use_warm_start = use_warm_start
        self._setup_input = settings
        if self.opt_type in CONSTRAINED_OPT:
            self._setup_input["u"] = np.inf * np.ones(self.opt.nk + self.opt.na)
        self._reset_parameters()
        return self

    def reset_parameters(self, p):
        super().reset_parameters(p)
        self._reset_parameters()

    def _reset_parameters(self):
        self._setup_input = {
            "P": csc_matrix(2.0 * self.opt.P(self.p).toarray()),
            "q": self.opt.q(self.p).toarray().flatten(),
        }
        if self.opt_type in CONSTRAINED_OPT:
            A = self.opt.A(self.p)
            b = self.opt.b(self.p)
            self._setup_input["A"] = csc_matrix(
                cs.vertcat(self.opt.M(self.p), A, -A).toarray()
            )
            self._setup_input["l"] = (
                cs.vertcat(-self.opt.c(self.p), -b, b).toarray().flatten()
            )

    def _solve(self):
        # Setup solver
        self.m = osqp.OSQP()
        self.m.setup(**self._setup_input)

        # Warm start optimization
        if self.use_warm_start:
            self.m.warm_start(x=self.x0.toarray().flatten())

        # Solve problem
        self._solution = self.m.solve()

        return self._solution.x

    def stats(self):
        return self._solution

    def did_solve(self):
        return self._solution.info.status == self.OSQP_SOLVED

    def number_of_iterations(self):
        return self._solution.info.iter


################################################################
# CVXOPT QP solver (https://cvxopt.org/)


class CVXOPTSolver(Solver):

    """CVXOPT solver interface."""

    def setup(self, solver_settings={}):
        """Setup the cvxopt solver interface.

        Parameters
        ----------

        solver_settings : Dict
            Settings passed to the CVXOPT solver.

        Returns
        -------

        solver : CVXOPTSolver
            The instance of the solve (i.e. self).

        """
        assert self.opt_type in QP_COST, "CVXOPT cannot solve this problem"
        self._solver_input = solver_settings
        self._reset_parameters()
        return self

    def reset_parameters(self, p):
        super().reset_parameters(p)
        self._reset_parameters()

    def _reset_parameters(self):
        self._solver_input["P"] = cvxopt.matrix(2.0 * self.opt.P(self.p).toarray())
        self._solver_input["q"] = cvxopt.matrix(self.opt.q(self.p).toarray().flatten())
        if self.opt_type in CONSTRAINED_OPT:
            if self.opt.nk > 0:
                self._solver_input["G"] = cvxopt.matrix(-self.opt.M(self.p).toarray())
                self._solver_input["h"] = cvxopt.matrix(
                    self.opt.c(self.p).toarray().flatten()
                )
            if self.opt.na > 0:
                self._solver_input["A"] = cvxopt.matrix(self.opt.A(self.p).toarray())
                self._solver_input["b"] = cvxopt.matrix(-self.opt.b(self.p).toarray())

    def _solve(self):
        self._solution = cvxopt.solvers.qp(**self._solver_input)
        return self._solution["x"]

    def stats(self):
        return self._solution

    def did_solve(self):
        return self._solution["status"] == "optimal"

    def number_of_iterations(self):
        return self._solution["iterations"]


################################################################
# Scipy Minimize solvers (https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)


class ScipyMinimizeSolver(Solver):

    """Scipy solver (scipy.optimize.minimize) interface."""

    methods_req_jac = {
        "CG",
        "BFGS",
        "Newton-CG",
        "L-BFGS-B",
        "TNC",
        "SLSQP",
        "dogleg",
        "trust-ncg",
        "trust-krylov",
        "trust-exact",
        "trust-constr",
    }

    methods_req_hess = {
        "Newton-CG",
        "dogleg",
        "trust-ncg",
        "trust-krylov",
        "trust-exact",
        "trust-constr",
    }

    methods_handle_constraints = {"COBYLA", "SLSQP", "trust-constr"}

    def setup(self, method="SLSQP", tol=None, options=None):
        """Setup the Scipy solver.

        Parameters
        ----------

        method : str
            Type of solver.

        tol : Optional[float] (default is None)
            Tolerance for termination. When tol is specified, the
            selected minimization algorithm sets some relevant
            solver-specific tolerance(s) equal to tol. For detailed
            control, use solver-specific options.

        options : Optional[Dict]
            A dictionary of solver options.

        Returns
        -------

        solver : ScipyMinimizeSolver
            The instance of the solve (i.e. self).

        """

        # Input check
        if self.opt_type in CONSTRAINED_OPT and (
            method not in ScipyMinimizeSolver.methods_handle_constraints
        ):
            raise TypeError(
                f"optimization problem has constraints, the method '{method}' is not suitable"
            )

        # Setup class attributes
        self._stats = None
        self.method = method

        # Setup minimize input parameters
        self.minimize_input = {
            "fun": self.f,
            "method": method,
            "x0": self.x0.toarray().flatten(),
        }

        if tol is not None:
            self.minimize_input["tol"] = tol

        if options is not None:
            self.minimize_input["options"] = options

        if method in ScipyMinimizeSolver.methods_req_jac:
            self.minimize_input["jac"] = self.jac

        if method in ScipyMinimizeSolver.methods_req_hess:
            self.minimize_input["hess"] = self.hess

        self._constraints = {}
        if method in ScipyMinimizeSolver.methods_handle_constraints:
            if method != "trust-constr":
                if self.opt_type in CONSTRAINED_OPT:
                    self._constraints["constr"] = {
                        "type": "ineq",
                        "fun": self.v,
                        "jac": self.dv,
                    }
            else:
                if self.opt.nk:
                    self._constraints["k"] = LinearConstraint(
                        A=csc_matrix(self.opt.M(self.p).toarray()),
                        lb=-self.opt.c(self.p).toarray.flatten(),
                        ub=self.opt.inf * np.ones(self.opt.nk),
                    )

                if self.opt.na:
                    eq = -self.opt.b(self.p).toarray().flatten()
                    self._constraints["a"] = LinearConstraint(
                        A=csc_matrix(self.opt.A(self.p).toarray()),
                        lb=eq,
                        ub=eq,
                    )

                if self.opt.ng:
                    self._constraints["g"] = NonlinearConstraint(
                        fun=self.g,
                        lb=np.zeros(self.opt.ng),
                        ub=self.opt.inf * np.ones(self.opt.ng),
                        jac=self.dg,
                        hess=self.ddg,
                    )

                if self.opt.nh:
                    self._constraints["h"] = NonlinearConstraint(
                        fun=self.h,
                        lb=np.zeros(self.opt.nh),
                        ub=np.zeros(self.opt.nh),
                        jac=self.dh,
                        hess=self.ddh,
                    )

        return self

    def f(self, x):
        return float(self.opt.f(x, self.p).toarray().flatten()[0])

    def jac(self, x):
        return self.opt.df(x, self.p).toarray().flatten()

    def hess(self, x):
        return self.opt.ddf(x, self.p).toarray()

    def v(self, x):
        return self.opt.v(x, self.p).toarray().flatten()

    def dv(self, x):
        return self.opt.dv(x, self.p).toarray()

    def g(self, x):
        return self.opt.g(x, self.p).toarray().flatten()

    def dg(self, x):
        return self.opt.dg(x, self.p).toarray()

    def ddg(self, x):
        return self.opt.ddg(x, self.p).toarray()

    def h(self, x):
        return self.opt.h(x, self.p).toarray().flatten()

    def dh(self, x):
        return self.opt.dh(x, self.p).toarray()

    def ddh(self, x):
        return self.opt.ddh(x, self.p).toarray()

    def reset_initial_seed(self, x0):
        super().reset_initial_seed(x0)
        self.minimize_input["x0"] = self.x0.toarray().flatten()

    def reset_parameters(self, p):
        super().reset_parameters(p)
        if self.method == "trust-constr":
            if self.opt.nk:
                self._constraints["k"].A = csc_matrix(self.opt.M(self.p).toarray())
                self._constraints["k"].lb = -self.opt.c(self.p).toarray.flatten()
            if self.opt.na:
                eq = -self.opt.b(self.p).toarray().flatten()
                self._constraints["a"].A = csc_matrix(self.opt.A(self.p).toarray())
                self._constraints["a"].lb = eq
                self._constraints["a"].ub = eq
        if self._constraints:
            self.minimize_input["constraints"] = list(self._constraints.values())

    def _solve(self):
        self._solution = minimize(**self.minimize_input)
        return self._solution.x

    def stats(self):
        return self._solution

    def did_solve(self):
        return self._solution.success

    def number_of_iterations(self):
        return self._solution.nit

import optas
import numpy as np


###########################################################################
# Helper methods
#


def isclose(A: np.ndarray, B: np.ndarray):
    return np.isclose(A, B).all()


###########################################################################
# Tests
#


class _SolverTester:
    @staticmethod
    def setup_builder(constraint=False):
        # Booth function
        # https://www.sfu.ca/~ssurjano/booth.html
        # https://en.wikipedia.org/wiki/Test_functions_for_optimization
        T = 1
        builder = optas.OptimizationBuilder(T)
        x = builder.add_decision_variables("x")
        y = builder.add_decision_variables("y")
        a = builder.add_parameter("a")
        b = builder.add_parameter("b")
        f = (x + a * y - b) ** 2 + (2.0 * x + y - 5.0) ** 2
        builder.add_cost_term("f", f)
        if constraint:
            builder.add_bound_inequality_constraint("bnd", -1e9, x, 1e9)
        return builder

    @staticmethod
    def setup_optimization(constraint=False):
        builder = _SolverTester.setup_builder(constraint=constraint)
        return builder.build()


class TestSolverInterface(_SolverTester):
    @staticmethod
    def solve_and_check_solution(solver, solver_name=""):
        solver.reset_initial_seed({"x": 0, "y": 0})
        solver.reset_parameters({"a": 2.0, "b": 7.0})
        name = ""
        if solver_name:
            name += solver_name + " "
        result = solver.solve()
        assert isclose(result["x"].toarray().flatten(), 1.0), name + "solver failed"
        assert isclose(result["y"].toarray().flatten(), 3.0), name + "solver failed"

    def test_casadi_interface(self):
        qp_solver_names = ["qpoases"]
        nlp_solver_names = ["ipopt", "sqpmethod"]
        for solver_name in qp_solver_names + nlp_solver_names:
            opt = self.setup_optimization()
            solver = optas.CasADiSolver(opt).setup(solver_name)
            self.solve_and_check_solution(solver)

    def test_osqp_interface(self):
        opt = self.setup_optimization(True)  # OSQP fails for unconstrained
        solver = optas.OSQPSolver(opt).setup(True)
        self.solve_and_check_solution(solver)

    def test_cvxopt_interface(self):
        opt = self.setup_optimization()
        solver = optas.CVXOPTSolver(opt).setup()
        self.solve_and_check_solution(solver)

    def test_scipy_minimize_interface(self):
        solver_names = [
            "Nelder-Mead",
            "Powell",
            "CG",
            "BFGS",
            "Newton-CG",
            "L-BFGS-B",
            "TNC",
            "COBYLA",
            "SLSQP",
            "trust-constr",
            "dogleg",
            "trust-ncg",
            "trust-exact",
            "trust-krylov",
        ]

        for solver_name in solver_names:
            opt = self.setup_optimization()
            solver = optas.ScipyMinimizeSolver(opt).setup(method=solver_name, tol=1e-6)
            self.solve_and_check_solution(solver, solver_name)

import abc
import time
import optas
from typing import List


class Controller(abc.ABC):

    """Controller base class. This provides a structure for implementing controllers."""

    def __init__(self):
        self.solution = None
        self._solver_duration = None
        self.solver = None  # this must be created during initialization

    @abc.abstractmethod
    def is_ready(self) -> bool:
        """True when control loop is ready to run."""
        pass

    def is_first_solve(self):
        """True when the solver is being run for the first time."""
        return self.solution is None

    @abc.abstractmethod
    def reset(self):
        """Reset the optimization problem."""
        pass

    def solve(self):
        """Solves the optimization problem."""
        t0 = time.perf_counter()
        self.solution = self.solver.solve()
        t1 = time.perf_counter()
        self._solver_duration = t1 - t0

    def get_solver_duration(self):
        """Returns the duration of the solver."""
        return self._solver_duration

    @abc.abstractmethod
    def get_joint_names(self) -> List[str]:
        """Returns the list of joint names. Note, this defines the ordering of the joint states in the solution."""
        pass

    @abc.abstractmethod
    def get_joint_state_solution(self) -> optas.DM:
        """Returns the joint state solution for the previous call to solve."""
        pass

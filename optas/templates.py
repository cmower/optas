import abc
import time
import optas


class DifferentialIK:


    def __init__(self, robot, eff_link, base_link=None):
        T = 1
        assert 1 in robot.time_derivs, "robot time_derivs must contain 1"
        self.robot = robot
        self.eff_link = eff_link
        self.builder = optas.OptimizationBuilder(T, robots=robot, derivs_align=True)
        self.robot_name = robot.get_name()
        self.qd = self.builder.get_model_state(self.robot_name, 0, time_deriv=1)
        self.qc = self.builder.add_parameter('qc', robot.ndof)
        if base_link is None:
            self.J = robot.get_global_geometric_jacobian(eff_link, self.qc)
        else:
            self.J = robot.get_geometric_jacobian(eff_link, self.qc, base_link)
        self.veff = self.J @ self.qd
        self._solver_duration = None


    @abc.abstractmethod
    def _setup_problem(self, *args):
        pass


    def setup_problem(self, *args):
        self._setup_problem(*args)
        return self


    def _solver_interface(self):
        return optas.CasADiSolver


    def _solver_setup_args(self):
        return ('ipopt',)


    def _solver_setup_kwargs(self):
        return {}


    def setup_solver(self):
        solver_interface = self._solver_interface()
        args = self._solver_setup_args()
        kwargs = self._solver_setup_kwargs()
        self._solver = solver_interface(self.builder.build()).setup(*args, **kwargs)
        return self


    def reset(self, qc, qd_init=None):
        if qd_init is not None:
            self._solver.reset_initial_seed(
                {f'{self.robot_name}/dq': optas.vec(qd_init)},
            )
        self._params = {'qc': qc}
        self._solver.reset_parameters(self._params)


    def solve(self):
        t0 = time.perf_counter()
        self._solution = self._solver.solve()
        t1 = time.perf_counter()
        self._solver_duration = t1 - t0


    def get_target_dq(self):
        return self._solution[f'{self.robot_name}/dq']


    def get_target_q(self, dt=1.):
        qc = self._params['qc']
        qd = self.get_target_dq()
        return qc + dt*qd

    def get_solver_duration(self):
        return self._solver_duration

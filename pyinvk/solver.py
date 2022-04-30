import casadi as cs
from sensor_msgs.msg import JointState

class Solver:

    """

    min  cost(x)
     x

      st
          lbg             ubg
          0   <=   g   <= BIG_NUMBER   (inequality constraints)
          0   <=   h   <= 0            (  equality constraints)


    """

    BIG_NUMBER = 1e9  # used as upper bound for inequality constraints g

    def __init__(self, casadi_solver, builder):
        self.casadi_solver = casadi_solver
        self.start_state = cs.DM.zeros(builder.q.numel())
        self.params = builder.params.zero()  # initialize with zeros
        self.builder = builder
        self.p = None
        self.lbg = cs.DM.zeros(self.builder.constraints.numel())
        self.ubg = cs.vertcat(
            self.BIG_NUMBER*cs.DM.ones(self.builder.ineq_constraints.numel()),
            cs.DM.zeros(self.builder.eq_constraints.numel()),
        )
        self.__called_reset = False
        self.__called_solve = False

    def set_parameter(self, name, value):
        self.params[name] = cs.DM(value)

    def set_initial_seed(self, init_seed):
        self.start_state = cs.vec(cs.DM(init_seed))

    def reset(self):
        self.p = self.builder.params.dict2vec(self.params)
        self.__called_reset = True

    def solve(self):
        assert self.__called_reset, "you must call reset before solve"
        sol = self.casadi_solver(
            x0=self.start_state,
            p=self.p,
            lbg=self.lbg,
            ubg=self.ubg,
        )
        self.__called_reset = False
        self.__called_solve = True
        return sol

    def stats(self):
        assert self.__called_solve, "you must call solve before stats"
        stats = self.casadi_solver.stats()
        self.__called_solver = False
        return stats        

    def solution2msg(self, solution, i=-1):
        q = cs.reshape(solution['x'], self.builder.ndof, self.builder.N).toarray()
        return JointState(
            name=self.builder.get_joint_names(),
            position=q[:, i].flatten().tolist(),
        )

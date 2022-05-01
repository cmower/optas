import abc
import casadi as cs

@abc.ABC
class Interface:

    def __init__(self, optimization):
        self.__optimization = optimization
        self.__init_seed = None

    def set_inital_seed(self, init_seed):
        self.__init_seed = cs.vec(init_seed)

    @abc.abstractmethod
    def solve_optimization(self):
        pass

    @abc.abstractmethod    
    def get_solution(self):
        pass

class CasadiInterface(Interface):

    def __init__(self, optimization, solver_name, solver_opts={}):
        super().__init__(optimization)
        problem = self.__optimization.get_nlpsol_problem()
        self.__solver = cs.nlpsol('solver', solver_name, problem, solver_opts)
        
    def solve_optimization(self):
        pass

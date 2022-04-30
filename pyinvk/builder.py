import casadi as cs
from urdf2casadi import urdfparser as u2c

from .sx_container import SXContainer, SX
from .solver import Solver

class SolverBuilder:

    def __init__(
            self,
            urdf_file_name,
            root,
            tip,
            N=1,
    ):

        self.urdf_file_name = urdf_file_name
        self.root = root
        self.tip = tip
        self.N = N

        self.robot_parser = u2c.URDFparser(use_jit=False)
        self.robot_parser.from_file(urdf_file_name)

        # FK dict from parser, contains:
        # - joint_names
        # - upper
        # - lower
        # - joint_list
        # - q
        # - quaternion_fk
        # - dual_quaternion_fk
        # - T_fk
        self.fk_dict = self.robot_parser.get_forward_kinematics(root, tip)
        self.ndof = self.fk_dict['q'].shape[0]
        self.q = cs.SX.sym('q', self.ndof, N)

        self.cost = 0.0
        self.params = SXContainer()
        self.ineq_constraints = SXContainer()
        self.eq_constraints = SXContainer()
        self.constraints = None  # set when solver is built

    def add_parameter(self, name: str, m: int=1, n: int=1) -> SX:
        """Add parameter"""
        p = cs.SX.sym(name, m, n)
        self.params[name] = p
        return p

    def add_cost_term(self, term: SX):
        """Add cost term"""
        self.cost += term

    def add_ineq_constraint(self, name, constraint):
        """Add inequality constraint g(q) >= 0"""
        self.ineq_constraints[name] = constraint  # must be constraint >= 0

    def add_eq_constraint(self, name, constraint):
        """Add equality constraint g(q) == 0"""
        self.eq_constraints[name] = constraint  # must be constraint == 0        

    def get_q(self, i=-1):
        """Get symbolic joint angles"""
        return self.q[:, i]

    def get_joint_names(self):
        return self.fk_dict['joint_names']

    def get_lower_q_limits(self):
        """Get lower joint limits"""
        return self.fk_dict['lower']

    def get_upper_q_limits(self):
        """Get upper joint limits"""        
        return self.fk_dict['upper']

    def get_end_effector_position(self, i=-1):
        """Get end effector position (function of joint angles)"""
        q = self.get_q(i)
        T = self.fk_dict['T_fk']
        tf = T(q)
        return tf[:3, 3]

    def get_end_effector_quaternion(self, i=-1):
        q = self.get_q(i)
        quat = self.fk_dict['quaternion_fk']
        return quat(q)

    # Common constraints
    def enforce_joint_limits(self):
        """Enforce joint limit (inequality) constraints"""

        lo = self.get_lower_q_limits()
        up = self.get_upper_q_limits()
        q = self.fk_dict['q']

        # lo <= q <= up
        #
        # q - lo >= 0    lower_g_ineq
        # up - q >= 0    upper_g_ineq         
        joint_limit = cs.Function('jlim', [q], [cs.vertcat(q-lo, up-q)])
        joint_limit_ = joint_limit.map(self.N)
        self.ineq_constraints['__joint_limit_constraint__'] = joint_limit_(self.q)

    def enforce_start_state(self, qstart):
        """Enforce initial state constraint"""
        if self.N < 2: raise ValueError(f"{self.N=} must be 2 or greater")
        self.eq_constraints['__start_state_constraint__'] = self.get_q(0) - cs.DM(qstart)
        
    # Build solver
    def build_solver(self, solver_name):
        """Build and retrn solver"""

        self.constraints = self.ineq_constraints + self.eq_constraints

        problem = {
            'x': cs.vec(self.q),
            'f': self.cost,
            'p': self.params.vec(),
            'g': self.constraints.vec(),
        }

        casadi_solver = cs.nlpsol('solver', solver_name,  problem)

        return Solver(casadi_solver, self)

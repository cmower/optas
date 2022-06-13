import casadi as cs
from pyinvk.robot_model import RobotModel
from pyinvk.builder import OptimizationBuilder
from pyinvk.solver import CasadiNLPSolver, ScipyMinimizeSolver

"""

In this example, we demonstrate how to load a robot, and setup/solve
an IK problem.

"""

def main():

    print("Starting pyinvk example 0")

    # Setup/build optimization problem
    urdf_filename = 'kuka_lwr.urdf'
    robot = RobotModel(urdf_filename)
    robots = {'kuka': robot}
    builder = OptimizationBuilder(robots)
    qnext = builder.get_state('kuka', 0)
    fk = robot.fk('baselink', 'lwr_arm_7_link')
    pos = fk['pos']
    pos_goal = builder.add_parameter('pos_goal', 3)
    builder.add_cost_term('eff_pos_goal', cs.sumsqr(pos(qnext) - pos_goal))
    lbc = robot.lower_actuated_joint_limits
    ubc = robot.upper_actuated_joint_limits
    builder.add_ineq_constraint('joint_position_limits', qnext, lbc=lbc, ubc=ubc)
    optimization = builder.build()

    # Setup solver
    use_scipy = True
    if not use_scipy:
        solver = CasadiNLPSolver(optimization)
        solver.setup('ipopt')
    else:
        solver = ScipyMinimizeSolver(optimization)
        solver.setup(
            # method='Nelder-Mead',
            method='SLSQP',
            # method='trust-constr',
        )

    # Solve problem
    solver.reset_parameters({'pos_goal': [0.3, 0.4, 0.3]})
    solution = solver.solve()

    print("Solution:")
    print(solution['kuka/q'])
    

    print("Goodbye")

if __name__ == '__main__':
    main()

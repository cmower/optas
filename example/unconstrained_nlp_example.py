import pathlib
import pyinvk
pyinvk.cs.np.set_printoptions(precision=3, suppress=True)

"""


q*  =  arg min  || F(q) - fg ||^2
           q

q*: computed joint position
q: decision variable (joint position)
F: forward kinematics, position of end-effector
fg: linear position goal
||.||^2: squared Euclidean norm


"""

def main():

    # Constants
    T = 1  # time steps
    urdf_filename = str(pathlib.Path(__file__).parent.absolute())+'/robots/med7.urdf'  # filename for the robot urdf        
    eff_link_name = 'lbr_link_ee'  # link name for the end-effector

    # Create robot model
    robot = pyinvk.RobotModel(urdf_filename)

    # Create optimization builder
    builder = pyinvk.OptimizationBuilder(T, robots={'kuka': robot})

    # Add parameters to problem
    fg = builder.add_parameter('fg', 3)  # linear position goal

    # Get joint states
    q = builder.get_joint_state('kuka', 0)

    # Cost: match end-effector position
    F = robot.get_global_link_position(eff_link_name, q)
    builder.add_cost_term('match_eff_pos', pyinvk.cs.sumsqr(F - fg))

    # Build optimization and setup solver
    solver = pyinvk.CasADiSolver(builder.build()).setup('ipopt')

    # Reset problem
    fg = pyinvk.cs.np.random.rand(3)
    solver.reset_parameters({'fg': fg})

    # Solve problem
    solution = solver.solve()
    qstar = solution['kuka/q'].toarray().flatten()
    print("solution =", qstar)
    pstar = robot.get_global_link_position(eff_link_name, qstar).toarray().flatten()
    print("fg =", fg.flatten())
    print("pstar =", pstar)

if __name__ == '__main__':
    main()
    
    

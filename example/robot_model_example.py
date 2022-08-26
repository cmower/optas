import pathlib
import pyinvk
pyinvk.cs.np.set_printoptions(precision=3, suppress=True)

def main():

    urdf_filename = str(pathlib.Path(__file__).parent.absolute())+'/robots/med7.urdf'  # filename for the robot urdf        
    print("Loading model ...")
    robot = pyinvk.RobotModel(urdf_filename)
    print("\nCompleted loading model", end='\n\n')

    print("ndof =", robot.ndof, end='\n\n')

    print("joint names:")
    for name in robot.actuated_joint_names:
        print(" ", name)

    qrand = robot.get_random_joint_positions()
    print("\nRandom joint position state:")
    print(qrand.toarray().flatten())

    print(f"\nEnd-effector transform in global link ({robot.get_root_link()}):")
    print(robot.get_global_link_transform('lbr_link_ee', qrand).toarray(), end='\n\n')

    print("Geometric jacobian:")
    print(robot.get_geometric_jacobian('lbr_link_ee', qrand).toarray(), end='\n\n')

if __name__ == '__main__':
    main()

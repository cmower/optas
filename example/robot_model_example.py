import pyinvk
pyinvk.cs.np.set_printoptions(precision=3, suppress=True)

print("Loading model ...")
robot = pyinvk.RobotModel('med7.urdf')
print("\nCompleted loading model", end='\n\n')

print("ndof =", robot.ndof, end='\n\n')

print("joint names:")
for name in robot.actuated_joint_names:
    print(" ", name)

qrand = robot.get_random_q()
print("\nRandom joint position state:")
print(qrand.toarray().flatten())

print(f"\nEnd-effector transform in global link ({robot.get_root_link()}):")
print(robot.get_global_link_transform('lbr_link_ee', qrand).toarray(), end='\n\n')

print("Geometric jacobian:")
print(robot.get_geometric_jacobian('lbr_link_ee', qrand).toarray(), end='\n\n')

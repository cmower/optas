from test_utils import *

h = 1e-8  # used for finite-differences

class Test_Model_models_py(unittest.TestCase):

    def test_get_name(self):
        pass

class Test_TaskModel_models_py(unittest.TestCase):

    def test_init(self):
        pass

class Test_RobotModel_models_py(unittest.TestCase):

    def test_joint_names(self):
        pass

    def test_link_names(self):
        pass

    def _estimate_jacobian(self, fun, q):
        n = fun(q).shape[0]
        ndof = q.shape[0]
        J = np.zeros((n, ndof))
        for j in range(ndof):
            qn = q.copy()
            qn[j] += h
            J[:, j] = (fun(qn) - fun(q)/h).toarray().flatten()
        return J

    def test_get_global_linear_jacobian(self):
        robot = optas.RobotModel(urdf_string=urdf_string)
        q = robot.get_random_joint_positions().toarray().flatten()
        J = robot.get_global_linear_geometric_jacobian('gripper', q)
        p = robot.get_global_link_position_function('gripper')
        J_fd = self._estimate_jacobian(p, q)
        self.assertTrue(isclose(J_fd, J))


if __name__ == '__main__':
    unittest.main()

from test_utils import *

class Test_builder_py(unittest.TestCase):

    def test_get_model_names(self):
        task_1 = optas.TaskModel('task_1', 10)
        task_2 = optas.TaskModel('task_2', 20)
        robot_1 = optas.RobotModel(urdf_string=urdf_string, name="robot_1")
        planar_3dof = optas.RobotModel(urdf_string=urdf_string)
        builder = optas.OptimizationBuilder(T=10,
                                            robots=[robot_1, planar_3dof],
                                            tasks=[task_1, task_2],
        )

        names_cmp = builder.get_model_names()
        names_exp = {'task_1', 'task_2', 'robot_1', 'planar_3dof'}
        self.assertEqual(set(names_exp), names_exp)

if __name__ == '__main__':
    unittest.main()

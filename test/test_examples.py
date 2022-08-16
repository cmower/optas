import os
import sys
import pathlib
import unittest

examples_dir = os.path.join(pathlib.Path(__file__).parent.absolute().parent.absolute(), 'example')
sys.path.append(examples_dir)

class TestExamples(unittest.TestCase):

    def _run_main(self, path):
        module = __import__(path)
        try:
            module.main()
        except:
            self.fail(f"{path} raised an exception unexpectedly")

    def test_linear_constrained_qd_example_py(self):
        self._run_main('linear_constrained_qd_example')

    def test_nonlinear_constrained_qp_py(self):
        self._run_main('nonlinear_constrained_qp_example')

    def test_robot_model_example_py(self):
        self._run_main('robot_model_example')

    def test_spatialmath_example_py(self):
        self._run_main('spatialmath_example')

    def test_unconstrained_qp_example_py(self):
        self._run_main('unconstrained_qp_example')

if __name__ == '__main__':
    unittest.main()

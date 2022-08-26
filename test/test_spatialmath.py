import unittest
import casadi as cs
import pyinvk.spatialmath as sm

def isclose(a, b):
    return cs.np.isclose(a.toarray(), b.toarray())


class TestSpatialMath(unittest.TestCase):
    
    def test_is_2x2(self):

        M1 = cs.DM.eye(2)
        self.assertTrue(sm.is_2x2(M1))

        M2 = cs.DM.eye(3)
        self.assertFalse(sm.is_2x2(M2))

    def test_is_3x3(self):

        M1 = cs.DM.eye(2)
        self.assertFalse(sm.is_3x3(M1))

        M2 = cs.DM.eye(3)
        self.assertTrue(sm.is_3x3(M2))

    def test_I3(self):
        self.assertTrue(isclose(sm.I3(), cs.DM.eye(3)))

    def test_I4(self):
        self.assertTrue(isclose(sm.I4(), cs.DM.
        self.assertTrue((sm.I4() == cs.DM.eye(4)).toarray().sum() == 16)

    def test_angvec2r(self):
        pass
        
if __name__ == '__main__':
    unittest.main()

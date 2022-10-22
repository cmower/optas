import unittest
import optas
import numpy as np
from scipy.spatial.transform import Rotation as R


np.random.seed(10)  # ensure random numbers are consistent
pi = np.pi

@optas.arrayify_args
def isclose(A, B):
    """Returns a boolean array where two arrays are element-wise equal within a tolerance."""
    A = A.toarray().flatten()
    B = B.toarray().flatten()
    return np.allclose(A, B)


class TestSpatialMath(unittest.TestCase):


    def _rand_ang(self):
        return np.random.uniform(-2*pi, 2*pi, size=(3,))


    def test_is_2x2(self):

        M1 = optas.DM.eye(2)
        self.assertTrue(optas.is_2x2(M1))

        M2 = optas.DM.eye(3)
        self.assertFalse(optas.is_2x2(M2))


    def test_is_3x3(self):

        M1 = optas.DM.eye(2)
        self.assertFalse(optas.is_3x3(M1))

        M2 = optas.DM.eye(3)
        self.assertTrue(optas.is_3x3(M2))


    def test_I3(self):
        self.assertTrue(isclose(optas.I3(), np.eye(3)))


    def test_I4(self):
        self.assertTrue(isclose(optas.I4(), np.eye(4)))


    def test_angvec2r(self):
        pass

    def test_angvec2tr(self):
        pass

    def test_delta2tr(self):
        pass

    def test_e2h(self):
        pass

    def test_eul2jac(self):
        pass

    def test_eul2r(self):
        num_test = 20
        for _ in range(num_test):
            eul = self._rand_ang()
            R_exp = R.from_euler('ZYZ', eul).as_matrix()
            R_cmp = optas.eul2r(eul[0], eul[1], eul[2])
            self.assertTrue(isclose(R_cmp, R_exp))

    def test_eul2tr(self):
        num_test = 20
        for _ in range(num_test):
            eul = self._rand_ang()
            Rot = R.from_euler('ZYZ', eul).as_matrix()
            T_exp = self._homogeneous_transform(R=Rot)
            T_cmp = optas.eul2tr(eul[0], eul[1], eul[2])
            self.assertTrue(isclose(T_cmp, T_exp))

    def test_h2e(self):
        pass

    def test_oa2r(self):
        pass

    def test_oa2tr(self):
        pass

    def test_r2t(self):
        num_test = 20
        for _ in range(num_test):
            eul = self._rand_ang()
            Rot = R.from_euler('ZYZ', eul).as_matrix()
            T_exp = self._homogeneous_transform(R=Rot)
            T_cmp = optas.r2t(Rot)
            self.assertTrue(isclose(T_cmp, T_exp))

    def test_rot2(self):
        n_test = 20
        test_theta = np.random.uniform(-2*pi, 2*pi, size=(n_test,)).tolist()
        for theta in test_theta:
            R_exp = R.from_euler('Z', theta).as_matrix()[:2, :2]
            R_cmp = optas.rot2(theta)
            self.assertTrue(isclose(R_cmp, R_exp))

    def _test_rotd(self, dim_label, num_test=20):
        optas_rotd = getattr(optas, f'rot{dim_label}')
        theta_test = np.random.uniform(-2*pi, 2*pi, size=(num_test,)).tolist()
        for theta in theta_test:
            R_exp = R.from_euler(dim_label.upper(), theta).as_matrix()
            R_cmp = optas_rotd(theta)
            self.assertTrue(isclose(R_cmp, R_exp))

    def test_rotx(self):
        self._test_rotd('x')

    def test_roty(self):
        self._test_rotd('y')

    def test_rotz(self):
        self._test_rotd('z')

    def test_rpy2jac(self):
        pass

    def test_rpy2r(self):
        num_test = 20
        opt_test = ['zyx', 'xyz', 'yxz']
        for opt in opt_test:
            for _ in range(num_test):
                rpy = self._rand_ang()
                R_exp = R.from_euler(opt.upper(), rpy[::-1]).as_matrix()
                R_cmp = optas.rpy2r(rpy, opt=opt)
                self.assertTrue(isclose(R_cmp, R_exp))


    def test_rpy2tr(self):
        num_test = 20
        opt_test = ['zyx', 'xyz', 'yxz']
        for opt in opt_test:
            for _ in range(num_test):
                rpy = self._rand_ang()
                Rot = R.from_euler(opt.upper(), rpy[::-1]).as_matrix()
                T_exp = self._homogeneous_transform(R=Rot)
                T_cmp = optas.rpy2tr(rpy, opt=opt)
                self.assertTrue(isclose(T_cmp, T_exp))

    def test_rt2tr(self):
        num_test = 20
        for _ in range(num_test):
            eul = self._rand_ang()
            Rot = R.from_euler('ZYZ', eul).as_matrix()
            t = np.random.uniform(-10, 10, size=(3,))
            T_exp = self._homogeneous_transform(R=Rot, t=t)
            T_cmp = optas.rt2tr(Rot, t)
            self.assertTrue(isclose(T_cmp, T_exp))

    def test_skew(self):
        pass

    def skewa(self):
        pass

    def test_t2r(self):
        num_test = 20
        for _ in range(num_test):
            eul = self._rand_ang()
            R_exp = R.from_euler('ZYZ', eul).as_matrix()
            T = self._homogeneous_transform(R=R_exp)
            R_cmp = optas.t2r(T)
            self.assertTrue(isclose(R_cmp, R_exp))

    def test_invt(self):
        num_test = 20
        for _ in range(num_test):
            eul = self._rand_ang()
            Rot = R.from_euler('ZYZ', eul).as_matrix()
            t = np.random.uniform(-10, 10, size=(3,))
            T = self._homogeneous_transform(R=Rot, t=t)
            T_exp = np.linalg.inv(T)
            T_cmp = optas.invt(T)
            self.assertTrue(isclose(T_cmp, T_exp))

    def test_tr2angvec(self):
        pass

    def test_tr2delta(self):
        pass

    def test_tr2eul(self):
        pass

    def test_tr2jac(self):
        pass

    def test_tr2rt(self):
        pass

    def test_transl(self):
        num_test = 20
        for _ in range(num_test):
            t_exp = np.random.uniform(-10, 10, size=(3,))
            T = self._homogeneous_transform(t=t_exp)
            t_cmp = optas.transl(T)
            self.assertTrue(isclose(t_exp, t_cmp))

    def test_transl2(self):
        num_test = 20
        for _ in range(num_test):
            t_exp = np.random.uniform(-10, 10, size=(2,))
            T = self._homogeneous_transform2(t=t_exp)
            t_cmp = optas.transl2(T)
            self.assertTrue(isclose(t_exp, t_cmp))


    def test_trlog(self):
        pass

    def _homogeneous_transform(self, R=None, t=None):
        T = np.eye(4)
        if R is not None:
            T[:3,:3] = R
        if t is not None:
            T[:3, 3] = t
        return T

    def _homogeneous_transform2(self, R=None, t=None):
        T = np.eye(3)
        if R is not None:
            T[:2,:2] = R
        if t is not None:
            T[:2, 2] = t
        return T

    def _test_trotd(self, dim_label):
        optas_trotd = getattr(optas, f'trot{dim_label}')
        num_test = 20
        for _ in range(num_test):
            theta = np.random.uniform(-2*pi, 2*pi)
            T_exp = self._homogeneous_transform(R=R.from_euler(dim_label.upper(), theta).as_matrix())
            T_cmp = optas_trotd(theta)
            self.assertTrue(isclose(T_cmp, T_exp))

    def test_trotx(self):
        self._test_trotd('x')

    def test_troty(self):
        self._test_trotd('y')

    def test_trotz(self):
        self._test_trotd('z')

    def test_unit(self):
        num_test = 20
        for _ in range(num_test):
            v = np.random.uniform(-1, 1, size=(2,))
            v_exp = v/np.linalg.norm(v)
            v_cmp = optas.unit(v)
            self.assertTrue(isclose(v_cmp, v_exp))
            self.assertTrue(isclose(np.linalg.norm(v_cmp.toarray().flatten()), 1))

    def test_vex(self):
        pass

    def test_vexa(self):
        pass

    def test_Quaternion(self):
        pass

if __name__ == '__main__':
    unittest.main()

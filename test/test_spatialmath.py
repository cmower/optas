from test_utils import *
from scipy.linalg import logm

class Test_spatialmath_py(unittest.TestCase):


    num_test = 30


    def _rand_ang(self):
        return np.random.uniform(-pi, pi, size=(3,))


    def _rand_unit_vec(self):
        v = np.random.uniform(-1, 1, size=(3,))
        v /= np.linalg.norm(v)
        return v


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


    @optas.arrayify_args
    def _isrot(self, R):

        # Check: square
        n = R.shape[0]
        if n != R.shape[1]:
            return False

        # Check: orthogonality
        if not isclose(R.T @ R, optas.DM.eye(n)):
            return False

        # Check: unit determinant
        if not isclose(optas.det(R), 1):
            return False

        return True


    @optas.arrayify_args
    def _ishomog(self, T):

        # Check: size
        n = T.shape[0]
        if n != T.shape[1]:
            return False

        if not (n == 3 or n == 4):
            return False

        # Check: rotation
        if n == 3:
            R = T[:2, :2]
            bottom_row = optas.DM([0, 0, 1])
        else:
            R = T[:3, :3]
            bottom_row = optas.DM([0, 0, 0, 1])

        if not self._isrot(R):
            return False

        # Check bottom row
        if not isclose(T[-1, :], bottom_row):
            return False

        return True


    def _assertIsRot(self, R):
        self.assertTrue(self._isrot(R))

    def _assertIsHomog(self, T):
        self.assertTrue(self._ishomog(T))

    def _assertIsDM(self, obj):
        self.assertIsInstance(obj, optas.DM)


    def _assertIsSX(self, obj):
        self.assertIsInstance(obj, optas.SX)


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
        for _ in range(self.num_test):
            theta = np.random.uniform(-pi, pi)
            v = self._rand_unit_vec()
            R_exp = R.from_rotvec(theta*v).as_matrix()
            R_cmp = optas.angvec2r(theta, v)
            self.assertTrue(isclose(R_cmp, R_exp))
            self._assertIsDM(R_cmp)
            self._assertIsRot(R_cmp)
        theta_sym = optas.SX.sym('theta')
        v_sym = optas.SX.sym('v', 3)
        self._assertIsSX(optas.angvec2r(theta_sym, v_sym))


    def test_angvec2tr(self):
        for _ in range(self.num_test):
            theta = np.random.uniform(-pi, pi)
            v = self._rand_unit_vec()
            T_exp = self._homogeneous_transform(R=R.from_rotvec(theta*v).as_matrix())
            T_cmp = optas.angvec2tr(theta, v)
            self.assertTrue(isclose(T_cmp, T_exp))
            self._assertIsDM(T_cmp)
            self._assertIsHomog(T_cmp)
        theta_sym = optas.SX.sym('theta')
        v_sym = optas.SX.sym('v', 3)
        self._assertIsSX(optas.angvec2tr(theta_sym, v_sym))


    def test_eul2r(self):
        for _ in range(self.num_test):
            eul = self._rand_ang()
            R_exp = R.from_euler('ZYZ', eul).as_matrix()
            R_cmp = optas.eul2r(eul[0], eul[1], eul[2])
            self.assertTrue(isclose(R_cmp, R_exp))
            self._assertIsDM(R_cmp)
            self._assertIsRot(R_cmp)
        eul_sym = optas.SX.sym('eul', 3)
        self._assertIsSX(optas.eul2r(eul_sym[0], eul_sym[1], eul_sym[2]))


    def test_eul2tr(self):
        for _ in range(self.num_test):
            eul = self._rand_ang()
            Rot = R.from_euler('ZYZ', eul).as_matrix()
            T_exp = self._homogeneous_transform(R=Rot)
            T_cmp = optas.eul2tr(eul[0], eul[1], eul[2])
            self.assertTrue(isclose(T_cmp, T_exp))
            self._assertIsDM(T_cmp)
            self._assertIsHomog(T_cmp)
        eul_sym = optas.SX.sym('eul', 3)
        self._assertIsSX(optas.eul2tr(eul_sym[0], eul_sym[1], eul_sym[2]))


    def test_oa2r(self):
        o_sym = optas.SX.sym('o', 3)
        a_sym = optas.SX.sym('a', 3)
        self._assertIsSX(optas.oa2r(o_sym, a_sym))

    def test_oa2tr(self):
        o_sym = optas.SX.sym('o', 3)
        a_sym = optas.SX.sym('a', 3)
        self._assertIsSX(optas.oa2tr(o_sym, a_sym))


    def test_r2t(self):
        for _ in range(self.num_test):
            eul = self._rand_ang()
            Rot = R.from_euler('ZYZ', eul).as_matrix()
            T_exp = self._homogeneous_transform(R=Rot)
            T_cmp = optas.r2t(Rot)
            self.assertTrue(isclose(T_cmp, T_exp))
            self._assertIsDM(T_cmp)
            self._assertIsHomog(T_cmp)
        Rot_sym = optas.SX.sym('R', 3, 3)
        self._assertIsSX(optas.r2t(Rot_sym))


    def test_rot2(self):
        test_theta = np.random.uniform(-2*pi, 2*pi, size=(self.num_test,)).tolist()
        for theta in test_theta:
            R_exp = R.from_euler('Z', theta).as_matrix()[:2, :2]
            R_cmp = optas.rot2(theta)
            self.assertTrue(isclose(R_cmp, R_exp))
            self._assertIsDM(R_cmp)
            self._assertIsRot(R_cmp)
        theta = optas.SX.sym('theta')
        self._assertIsSX(optas.rot2(theta))


    def _test_rotd(self, dim_label):
        optas_rotd = getattr(optas, f'rot{dim_label}')
        theta_test = np.random.uniform(-2*pi, 2*pi, size=(self.num_test,)).tolist()
        for theta in theta_test:
            R_exp = R.from_euler(dim_label.upper(), theta).as_matrix()
            R_cmp = optas_rotd(theta)
            self.assertTrue(isclose(R_cmp, R_exp))
            self._assertIsDM(R_cmp)
            self._assertIsRot(R_cmp)
        theta = optas.SX.sym('theta')
        self._assertIsSX(optas_rotd(theta))


    def test_rotx(self):
        self._test_rotd('x')


    def test_roty(self):
        self._test_rotd('y')


    def test_rotz(self):
        self._test_rotd('z')


    def test_rpy2r(self):
        opt_test = ['zyx', 'xyz', 'yxz']
        rpy_sym = optas.SX.sym('rpy', 3)
        for opt in opt_test:
            for _ in range(self.num_test):
                rpy = self._rand_ang()
                R_exp = R.from_euler(opt.upper(), rpy[::-1]).as_matrix()
                R_cmp = optas.rpy2r(rpy, opt=opt)
                self.assertTrue(isclose(R_cmp, R_exp))
                self._assertIsDM(R_cmp)
                self._assertIsRot(R_cmp)
            self._assertIsSX(optas.rpy2r(rpy_sym, opt=opt))


    def test_rpy2tr(self):
        opt_test = ['zyx', 'xyz', 'yxz']
        rpy_sym = optas.SX.sym('rpy', 3)
        for opt in opt_test:
            for _ in range(self.num_test):
                rpy = self._rand_ang()
                Rot = R.from_euler(opt.upper(), rpy[::-1]).as_matrix()
                T_exp = self._homogeneous_transform(R=Rot)
                T_cmp = optas.rpy2tr(rpy, opt=opt)
                self.assertTrue(isclose(T_cmp, T_exp))
                self._assertIsDM(T_cmp)
                self._assertIsDM(optas.rpy2tr(optas.DM(rpy), opt=opt))
                self._assertIsHomog(T_cmp)
            self._assertIsSX(optas.rpy2tr(rpy_sym, opt=opt))


    def test_rt2tr(self):
        for _ in range(self.num_test):
            eul = self._rand_ang()
            Rot = R.from_euler('ZYZ', eul).as_matrix()
            t = np.random.uniform(-10, 10, size=(3,))
            T_exp = self._homogeneous_transform(R=Rot, t=t)
            T_cmp = optas.rt2tr(Rot, t)
            self.assertTrue(isclose(T_cmp, T_exp))
            self._assertIsDM(T_cmp)
            self._assertIsDM(optas.rt2tr(optas.DM(Rot), optas.DM(t)))
            self._assertIsDM(optas.rt2tr(optas.DM(Rot), optas.DM(t)))
            self._assertIsDM(optas.rt2tr(Rot, t.tolist()))
            self._assertIsHomog(T_cmp)
        Rot_sym = optas.SX.sym('R', 3, 3)
        t_sym = optas.SX.sym('t', 3)
        self._assertIsSX(optas.rt2tr(Rot_sym, t_sym))


    @staticmethod
    def _skew(x):
        """https://stackoverflow.com/a/36916261"""
        return np.array([[0, -x[2], x[1]],
                         [x[2], 0, -x[0]],
                         [-x[1], x[0], 0]])


    def test_skew(self):
        for _ in range(self.num_test):

            # Check 2-vector
            v = np.random.uniform(-10, 10)
            S = optas.skew(v)
            self.assertTrue(S.shape == (2, 2))
            self.assertTrue(isclose(np.diag(S.toarray()), np.zeros(2)))
            self.assertTrue(isclose(S, -S.T))
            self._assertIsDM(S)

            # Check 3-vector
            v = np.random.uniform(-10, 10, size=(3,))
            S_exp = self._skew(v)
            S_cmp = optas.skew(v)
            self.assertTrue(S_cmp.shape == (3, 3))
            self.assertTrue(isclose(S_cmp, S_exp))
            self.assertTrue(isclose(S_cmp, -S_cmp.T))
            self._assertIsDM(S_cmp)

            # Check raises Value error
            n = np.random.randint(4, 100)
            v = np.random.uniform(-10, 10, size=(n,))
            self.assertRaises(ValueError, optas.skew, v)

        v = optas.SX.sym('v')
        self._assertIsSX(optas.skew(v))

        v = optas.SX.sym('v', 3)
        self._assertIsSX(optas.skew(v))


    def test_t2r(self):
        for _ in range(self.num_test):
            eul = self._rand_ang()
            R_exp = R.from_euler('ZYZ', eul).as_matrix()
            T = self._homogeneous_transform(R=R_exp)
            R_cmp = optas.t2r(T)
            self.assertTrue(isclose(R_cmp, R_exp))
            self._assertIsDM(R_cmp)
            self._assertIsDM(optas.t2r(optas.DM(T)))
            self._assertIsRot(R_cmp)
        T_sym = optas.SX.sym('T', 4, 4)
        self._assertIsSX(optas.t2r(T_sym))


    def test_invt(self):
        for _ in range(self.num_test):
            eul = self._rand_ang()
            Rot = R.from_euler('ZYZ', eul).as_matrix()
            t = np.random.uniform(-10, 10, size=(3,))
            T = self._homogeneous_transform(R=Rot, t=t)
            T_exp = np.linalg.inv(T)
            T_cmp = optas.invt(T)
            self.assertTrue(isclose(T_cmp, T_exp))
            self._assertIsDM(T_cmp)
            self._assertIsDM(optas.invt(optas.DM(T)))
            self._assertIsHomog(T_cmp)
        T = optas.SX.sym('T', 4, 4)
        self._assertIsSX(optas.invt(T))


    def test_tr2eul(self):
        pass


    def test_tr2rt(self):
        for _ in range(self.num_test):
            R_exp = R.from_euler('ZYZ', self._rand_ang()).as_matrix()
            t_exp = np.random.uniform(-10, 10, size=(3,))
            T = self._homogeneous_transform(R=R_exp, t=t_exp)
            R_cmp, t_cmp = optas.tr2rt(T)
            self.assertTrue(isclose(R_cmp, R_exp))
            self.assertTrue(isclose(t_cmp, t_exp))
            self._assertIsDM(R_cmp)
            self._assertIsDM(t_cmp)
            self._assertIsRot(R_cmp)
            R_cmp, t_cmp = optas.tr2rt(optas.DM(T))
            self._assertIsDM(R_cmp)
            self._assertIsDM(t_cmp)
            self._assertIsRot(R_cmp)
        R_sym, t_sym = optas.tr2rt(optas.SX.sym('T', 4, 4))
        self._assertIsSX(R_sym)
        self._assertIsSX(t_sym)


    def test_transl(self):
        for _ in range(self.num_test):
            t_exp = np.random.uniform(-10, 10, size=(3,))
            T = self._homogeneous_transform(t=t_exp)
            t_cmp = optas.transl(T)
            self.assertTrue(isclose(t_exp, t_cmp))
            # TODO: check output types

    def test_transl2(self):
        for _ in range(self.num_test):
            t_exp = np.random.uniform(-10, 10, size=(2,))
            T = self._homogeneous_transform2(t=t_exp)
            t_cmp = optas.transl2(T)
            self.assertTrue(isclose(t_exp, t_cmp))
            # TODO: check output types

    def test_trlog(self):
        for _ in range(self.num_test):
            eul = self._rand_ang()
            Rot = R.from_euler('ZYZ', eul).as_matrix()
            LR_exp = logm(Rot)
            LR_cmp = optas.trlog(Rot, rmat=True)
            self.assertTrue(isclose(LR_cmp, LR_exp))
            # TODO: check output types


    def _test_trotd(self, dim_label):
        optas_trotd = getattr(optas, f'trot{dim_label}')
        for _ in range(self.num_test):
            theta = np.random.uniform(-2*pi, 2*pi)
            T_exp = self._homogeneous_transform(R=R.from_euler(dim_label.upper(), theta).as_matrix())
            T_cmp = optas_trotd(theta)
            self.assertTrue(isclose(T_cmp, T_exp))
            self._assertIsHomog(T_cmp)
            # TODO: check output types

    def test_trotx(self):
        self._test_trotd('x')

    def test_troty(self):
        self._test_trotd('y')

    def test_trotz(self):
        self._test_trotd('z')

    def test_unit(self):
        for _ in range(self.num_test):
            v = np.random.uniform(-1, 1, size=(2,))
            v_exp = v/np.linalg.norm(v)
            v_cmp = optas.unit(v)
            self.assertTrue(isclose(v_cmp, v_exp))
            self.assertTrue(isclose(np.linalg.norm(v_cmp.toarray().flatten()), 1))
            # TODO: check output types

    def test_vex(self):
        for _ in range(self.num_test):

            # Check 2-by-2
            v = np.random.uniform(-10, 10)
            S = np.array([[0, -v], [v, 0]])
            self.assertTrue(isclose(optas.vex(S), v))

            # Check 3-by-3
            v = np.random.uniform(-10, 10, size=(3,))
            S = self._skew(v)
            self.assertTrue(isclose(optas.vex(S), v))

            # Check ValueError raised
            n = np.random.randint(4, 100)
            S = np.random.uniform(-10, 10, size=(n,))
            self.assertRaises(ValueError, optas.vex, S)
            # TODO: check output types

    def test_Quaternion(self):
        for _ in range(self.num_test):

            # Setup
            eul = self._rand_ang()
            rot = R.from_euler('ZYZ', eul)
            quat_exp = rot.as_quat()
            quat = optas.Quaternion(quat_exp)

            # Check Quaternion.split
            for q, qe in zip(quat.split(), quat_exp):
                self.assertEqual(q, qe)

            # Check Quaternion.sumsqr
            self.assertTrue(isclose(np.linalg.norm(quat_exp)**2, quat.sumsqr()))

            # Check Quaternion.inv
            rot_inv = R.from_quat(quat.inv().getquat().toarray().flatten())
            A = (rot_inv * rot).as_matrix()
            self.assertTrue(isclose(A, np.eye(3)))

            # Check Quaternion.fromrpy

            # Check Quaternion.fromangvec

            # Check Quaternion.getquat
            self.assertTrue(isclose(optas.Quaternion(quat_exp).getquat(), quat_exp))
            self.assertTrue(isclose(
                optas.Quaternion(quat_exp[0], quat_exp[1], quat_exp[2], quat_exp[3]).getquat(),
                quat_exp,
            ))

            # Check Quaternion.getrpy

            # TODO: check output types


if __name__ == '__main__':
    unittest.main()

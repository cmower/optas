import os
import abc
import time

import pybullet_api

import optas
from optas.spatialmath import arrayify_args
import tf_conversions

import exotica_scipy_solver
import pyexotica as exo
import exotica_core_task_maps_py

from trac_ik_python.trac_ik_wrap import TRAC_IK

import matplotlib.pyplot as plt

path = os.path.dirname(os.path.realpath(__file__))
datadir = os.path.join(path, 'data')
if not os.path.exists(datadir):
    os.mkdir(datadir)

exprdir = os.path.join(datadir, 'expr2')
if not os.path.exists(exprdir):
    os.mkdir(exprdir)

pi = optas.np.pi
urdf = os.path.join(path, 'robots', 'kuka_lwr', 'kuka_lwr.urdf')
ee_link = "end_effector_ball"
robot = optas.RobotModel(urdf_filename=urdf, time_derivs=[0, 1])
pos = robot.get_global_link_position_function(ee_link)
q0 = optas.np.deg2rad([0, 30, 0, -90, 0, -30, 0])
Tr = robot.get_global_link_transform_function(ee_link)
quat0 = tf_conversions.transformations.quaternion_from_matrix(Tr(q0).toarray())
quat0 /= optas.np.linalg.norm(quat0) # ensure normalized
eul0 = tf_conversions.transformations.euler_from_matrix(Tr(q0).toarray())

p0 = pos(q0)
N = 2000
dt = 0.01
use_pb = True


def figure_eight(t):
    return p0 + 0.2*optas.DM([0., optas.sin(t), optas.sin(0.5*t)])


class ExprIK(abc.ABC):

    def __init__(self):
        self._solver_dur = None

    @abc.abstractmethod
    def reset(self, t, qc):
        pass

    @abc.abstractmethod
    def solve(self):
        pass

    def get_solver_duration(self):
        return self._solver_dur

class OpTaSIK(ExprIK):

    def __init__(self):

        super().__init__()

        # Setup
        robot = optas.RobotModel(urdf_filename=urdf, time_derivs=[0])
        self.name = robot.get_name()
        builder = optas.OptimizationBuilder(T=1, robots=[robot])
        qc = builder.add_parameter('qc', robot.ndof)
        qF = builder.get_model_state(self.name, t=0)
        qd = (qF - qc) / dt
        pg = builder.add_parameter('pg', 3)  # goal position
        Jp = robot.get_global_linear_jacobian(ee_link, qc)
        pc = pos(qc)
        p = pc + dt*Jp@qd
        Ja = robot.get_global_angular_geometric_jacobian(ee_link, qc)
        va = Ja@qd
        manip = robot.get_global_manipulability(ee_link, qF)

        # Cost term
        builder.add_cost_term('min_qd', optas.sumsqr(qc - qF))
        builder.add_cost_term('manip', -manip)

        # End-effector goal
        tol = 0.001
        for i in range(3):
            builder.add_leq_inequality_constraint(
                f'achieve_pos_{i}', optas.sumsqr(p[i] - pg[i]), tol**2)
            builder.add_leq_inequality_constraint(
                f'no_ee_ang_vel_{i}', optas.sumsqr(va[i]), tol**2)

        # Set joint limit constraints
        lo = robot.lower_actuated_joint_limits
        up = robot.upper_actuated_joint_limits
        builder.add_leq_inequality_constraint('lower_qlim', lo, qF)
        builder.add_leq_inequality_constraint('upper_qlim', qF, up)

        # Setup solver
        self._solver = optas.ScipyMinimizeSolver(builder.build()).setup('SLSQP')

    @arrayify_args
    def reset(self, t, qc):
        self._solver.reset_initial_seed({f'{self.name}/q': qc})
        self._solver.reset_parameters({'pg': figure_eight(t), 'qc': qc})

    def solve(self):
        t0 = time.perf_counter()
        solution = self._solver.solve()
        t1 = time.perf_counter()
        self._solver_dur = t1 - t0
        return solution[f'{self.name}/q'].toarray().flatten()

class EXOTicaIK(ExprIK):

    def __init__(self):

        super().__init__()
        xml = os.path.join(path, 'robots', 'kuka_lwr', 'exo.xml')
        self.problem = exo.Setup.load_problem(xml)
        self.solver = exotica_scipy_solver.end_pose_solver.SciPyEndPoseSolver(
            problem=self.problem, method='SLSQP', debug=False,
        )
        self.scene = self.problem.get_scene()
        self.task_maps = self.problem.get_task_maps()

    def reset(self, t, qc):
        p = figure_eight(t).toarray().flatten().tolist()
        q = quat0.tolist()
        self.scene.attach_object_local('Target', '', p+q)
        self.problem.start_state = qc
        jp = self.task_maps['JointPose']
        jp.joint_ref = qc

    def solve(self):
        t0 = time.perf_counter()
        q = self.solver.solve()[0]
        t1 = time.perf_counter()
        self._solver_dur = t1 - t0
        return q


class TracIK(ExprIK):

    tol = 0.001

    def __init__(self):
        super().__init__()
        with open(urdf, 'r') as urdf_file:
            urdf_str = urdf_file.read()
            self._solver = TRAC_IK('lwr_arm_0_link', ee_link, urdf_str, 0.005, 1e-5, 'Speed')

    def reset(self, t, qc):
        self._pg = figure_eight(t).toarray().flatten().tolist()
        self._qc = optas.DM(qc).toarray().flatten().tolist()

    def solve(self):
        t0  = time.perf_counter()
        solution = self._solver.CartToJnt(
            self._qc,
            self._pg[0], self._pg[1], self._pg[2],
            quat0[0], quat0[1], quat0[2], quat0[3],
            self.tol, self.tol, self.tol,
            self.tol, self.tol, self.tol,
        )
        t1 = time.perf_counter()
        self._solver_dur = t1 - t0
        return optas.np.array(solution)


class Experiment:

    def __init__(self):
        self.optas_ik = OpTaSIK()
        self.trac_ik = TracIK()
        self.exo_ik = EXOTicaIK()

        if use_pb:
            self.setup_pb()

    def setup_pb(self):
        self.pb = pybullet_api.PyBullet(dt)
        self.kuka = pybullet_api.KukaLWR()
        self.pb.start()                

    def reset_pb(self):
        self.kuka.reset(q0)
        time.sleep(1.)

    def update_pb(self, q):
        self.kuka.cmd(q)
        # time.sleep(dt)

    def _run(self, ik):

        if use_pb:
            self.reset_pb()

        t = 0.0
        q = optas.vec(q0)

        T = optas.DM.zeros(N)
        solver_duration = optas.DM.zeros(N)
        err_pos = optas.DM.zeros(N)
        # err_rot = optas.DM.zeros(N)

        for i in range(N):

            T[i] = t

            ik.reset(t, q)
            q = ik.solve()

            solver_duration[i] = ik.get_solver_duration()
            err_pos[i] = optas.np.linalg.norm(pos(q) - figure_eight(t))
            # eul = tf_conversions.transformations.euler_from_matrix(Tr(q).toarray())
            # err_rot[i] = optas.np.linalg.norm(optas.np.array(eul) - optas.np.array(eul0))
            
            t += dt

            if use_pb:
                self.update_pb(q)

        return [
            T.toarray().flatten(),
            solver_duration.toarray().flatten(),
            err_pos.toarray().flatten(),
            # err_rot.toarray().flatten(),
        ]

    def run(self):
        # self.data_trac_ik = self._run(self.trac_ik)
        self.data_optas_ik = self._run(self.optas_ik)        
        self.data_exo_ik = self._run(self.exo_ik)        



if __name__ == '__main__':

    # Setup and run experiment
    expr = Experiment()
    expr.run()

    # Plot err
    fig, ax = plt.subplots(tight_layout=True)
    ax.plot(expr.data_optas_ik[0], expr.data_optas_ik[2], '_g', label='OpTaS')
    # ax.plot(expr.data_optas_ik[0], expr.data_trac_ik[2], '_r', label='TracIK')
    ax.plot(expr.data_optas_ik[0], expr.data_exo_ik[2], '_b', label='EXOTica')

    # Plot time
    fig, ax = plt.subplots(tight_layout=True, figsize=(7.7, 3.75))
    # ax[0].plot(expr.data_optas_ik[0], expr.data_optas_ik[2], '_g')
    # ax[0].plot(expr.data_optas_ik[0], expr.data_trac_ik[2], '_r')
    # ax[0].plot(expr.data_optas_ik[0], expr.data_exo_ik[2], '_b')
    print("OpTaS:", optas.np.mean(1000*expr.data_optas_ik[1]), optas.np.std(1000*expr.data_optas_ik[1]))
    # print("TracIK:", optas.np.mean(1000*expr.data_trac_ik[1]), optas.np.std(1000*expr.data_trac_ik[1]))
    print("EXOTica:", optas.np.mean(1000*expr.data_exo_ik[1]), optas.np.std(1000*expr.data_exo_ik[1]))        
    ax.plot(expr.data_optas_ik[0], 1000*expr.data_optas_ik[1], '_g', label='OpTaS', markersize=5)
    # ax.plot(expr.data_trac_ik[0], 1000*expr.data_trac_ik[1], '_r', label='TracIK', markersize=5)
    ax.plot(expr.data_exo_ik[0], 1000*expr.data_exo_ik[1], '_b', label='EXOTica', markersize=5)
    ax.legend(loc='upper right', ncol=3, fontsize=16)
    ax.grid()
    # ax.set_ylim(0, 6)
    ax.set_xlim(0, 20)
    ax.set_xlabel('Time (s)', fontsize=27)
    ax.set_ylabel('CPU Time (ms)', fontsize=27)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)            

    fig_filename = os.path.join(exprdir, 'optas_cmp.pdf')
    fig.savefig(fig_filename)    

    # Show plot
    plt.show()

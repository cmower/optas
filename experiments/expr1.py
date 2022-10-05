import os
import abc
import time
import datetime
from typing import List
import matplotlib.pyplot as plt
from dataclasses import dataclass

import optas
from optas.spatialmath import vectorize_args
from optas.templates import DifferentialIK

path = os.path.dirname(os.path.realpath(__file__))
pi = optas.np.pi
angle = -0.25*pi
R = optas.np.array([
    [optas.np.cos(angle), -optas.np.sin(angle)],
    [optas.np.sin(angle), optas.np.cos(angle)]
])

datadir = os.path.join(path, 'data')
if not os.path.exists(datadir):
    os.mkdir(datadir)

exprdir = os.path.join(datadir, 'expr1')
if not os.path.exists(exprdir):
    os.mkdir(exprdir)

class ExprIK(DifferentialIK):


    @vectorize_args
    def _setup_problem(self, xydir, dt):

        # Cost function
        w = 1.
        self.builder.add_cost_term('min_qd', w*optas.sumsqr(self.qd))

        # Optimize eff motion
        w = 1000.
        maxvel = 0.1
        desvel = maxvel*xydir
        v = self._get_desired_vel_difference(desvel)
        self.builder.add_cost_term('eff_motion', w*optas.sumsqr(v))

        # Set joint limit constraints
        lo = self.robot.lower_actuated_joint_limits
        up = self.robot.upper_actuated_joint_limits
        qn = self.qc + dt*self.qd
        self.builder.add_leq_inequality_constraint('lower_qlim', lo, qn)
        self.builder.add_leq_inequality_constraint('upper_qlim', qn, up)

        # Z-bounds
        pc = self.robot.get_global_link_position(self.eff_link, self.qc)
        pn = pc + dt*self.veff[:3]
        zlo, zup = 0.025, 0.15
        self.builder.add_leq_inequality_constraint('lower_zlim', zlo, pn[2])
        self.builder.add_leq_inequality_constraint('upper_zlim', pn[2], zup)


    @abc.abstractmethod
    def _get_desired_vel_difference(self, xydir):
        pass


    def _solver_interface(self):
        return optas.OSQPSolver


    def _solver_setup_args(self):
        return []


class IK1(ExprIK):


    def _get_desired_vel_difference(self, desvel):
        vg = optas.vertcat(desvel, optas.DM.zeros(4))  # optimize 6D
        return self.veff - vg


class IK2(ExprIK):


    def _get_desired_vel_difference(self, desvel):
        return self.veff[:2] - desvel # optimize 2D


@dataclass
class Data:
    T: optas.DM     # time stamps
    Q: optas.DM     # joint space position trajectory
    DQ: optas.DM    # joint space velocity trajectory
    P: optas.DM     # task space position trajectory
    sdur: optas.DM  #  solver duration


    def save(self, filename):
        data = optas.vertcat(
            self.T.T,
            self.Q,
            self.DQ,
            self.P,
        ).toarray()
        full_filename = os.path.join(exprdir, filename)
        optas.np.savetxt(full_filename, data, delimiter=',')

        return self


    def plot_distance_from_start(self, ax, fmt='-k', **kwargs):
        diff = self.P - self.P[:, 0]
        d = optas.np.linalg.norm(diff.toarray(), axis=0)
        ax.plot(self.T.toarray().flatten(), d, fmt, **kwargs)


    def plot_position_path(self, ax, fmt='-k', **kwargs):
        p = R@self.P.toarray()[:2,:]
        p -= p[:, 0].reshape(2, 1)
        ax.plot(p[0,:], p[1, :], fmt, **kwargs)

    def get_initial_position(self):
        return self.P.toarray()[:, 0].flatten()


@dataclass
class Config:
    urdf: str
    eff_link: str
    q0: List[float]
    xydir: List[float]
    rpbi: bool


class Experiment:


    N = 2*1400 + 200  # number of iterations of experiment
    dt = 0.01*0.5  # time step for each iteration of experiment


    def __init__(self, config):
        self.robot = optas.RobotModel(urdf_filename=config.urdf, time_derivs=[1])
        self.eff_pos = self.robot.get_global_link_position_function(config.eff_link, n=self.N+1)
        self.q0 = optas.vec(optas.np.deg2rad(config.q0))
        self.ik1 = IK1(self.robot, config.eff_link).setup_problem(config.xydir, self.dt).setup_solver()
        self.ik2 = IK2(self.robot, config.eff_link).setup_problem(config.xydir, self.dt).setup_solver()
        self.use_rpbi = config.rpbi
        if self.use_rpbi:
            self.setup_rpbi()


    def setup_rpbi(self):
        import rospy
        from sensor_msgs.msg import JointState
        self.rospy = rospy
        self.JointState = JointState
        rospy.init_node('optas_expr_node')
        self._pub = rospy.Publisher('rpbi/kuka_lwr/joint_states/target', JointState, queue_size=10)

    def reset_rpbi(self):
        self.publish_rpbi(self.q0)
        time.sleep(1.)


    def publish_rpbi(self, q):
        msg = self.JointState(
            name=self.robot.actuated_joint_names, position=q.toarray().flatten().tolist()
        )
        msg.header.stamp = self.rospy.Time.now()
        self._pub.publish(msg)


    def update_rpbi(self, q):
        self.publish_rpbi(q)
        time.sleep(self.dt)


    def _run(self, label, ik):

        if self.use_rpbi:
            self.reset_rpbi()

        t = 0.0
        q = optas.vec(self.q0)
        dq = optas.DM.zeros(self.robot.ndof)

        T = optas.DM.zeros(self.N+1)
        Q = optas.horzcat(q, optas.DM.zeros(self.robot.ndof, self.N))
        DQ = optas.horzcat(dq, optas.DM.zeros(self.robot.ndof, self.N))
        solver_duration = optas.DM.zeros(self.N)

        for i in range(self.N):

            ik.reset(q, dq)
            ik.solve()

            solver_duration[i] = ik.get_solver_duration()

            t += self.dt
            q = ik.get_target_q(self.dt)
            dq = ik.get_target_dq()

            T[i+1] = t
            Q[:, i+1] = q
            DQ[:, i+1] = dq

            if self.use_rpbi:
                self.update_rpbi(q)

        stamp = datetime.datetime.now().strftime('%d_%m_%Y_%H_%M_%S_')+str(time.time_ns())
        return Data(T, Q, DQ, self.eff_pos(Q), solver_duration).save(
            f'expr1_{label}_{stamp}.csv')


    def run(self):
        self.data1 = self._run('ik1', self.ik1)
        self.data2 = self._run('ik2', self.ik2)


if __name__ == '__main__':

    rpbi = True

    # Setup config
    xydir = optas.np.array([0.70710678, 0.70710678])
    config = Config(
        os.path.join(path, 'robots', 'kuka_lwr', 'kuka_lwr.urdf'),
        "end_effector_ball",
        [0, 30, 0, -90, 0, 60, 0],
        xydir,
        rpbi,
    )

    # Setup and run experiment
    expr = Experiment(config)
    expr.run()

    # Plot distance from start results
    # fig, ax = plt.subplots(tight_layout=True)
    # fig.suptitle('Distance from start')
    # expr.data1.plot_distance_from_start(ax, fmt='-r', label='Optimize 6D', linewidth=3)
    # expr.data2.plot_distance_from_start(ax, fmt='-g', label='Optimize 2D', linewidth=3)
    # ax.legend()

    # Plot position trajectory
    fig, ax = plt.subplots(tight_layout=True, figsize=(3, 3))
    # fig.suptitle('Position path')
    expr.data1.plot_position_path(ax, fmt='-r', label='Optimize 6D', linewidth=5)
    expr.data2.plot_position_path(ax, fmt='-g', label='Optimize 2D', linewidth=3)
    p0 = optas.np.zeros(2)
    pN = R@(p0 + 1.425*xydir)
    ax.plot([p0[0], pN[0]], [p0[1], pN[1]], '--k', linewidth=1, label='Ideal path')
    ax.plot([0], [0], 'ob', label='Start', markersize=10)
    ax.set_aspect('equal')
    ax.grid()
    ax.set_xlim(-0.05, 1.45)
    ax.set_ylim(-0.175, 0.175)
    ax.legend(ncol=4)
    ax.set_xlabel('$X$ (m)', fontsize=20)
    ax.set_ylabel('$Y$ (m)', fontsize=20)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)        

    fig_filename = os.path.join(exprdir, 'plot_position_trajectory.pdf')
    fig.savefig(fig_filename)

    expr.reset_rpbi()


    # Show plot
    plt.show()

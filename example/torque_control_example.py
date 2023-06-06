# Python standard lib
import os
import sys
import math
import pathlib

# PyBullet
import pybullet_api

# OpTaS
import optas
from optas.spatialmath import *

from scipy.spatial.transform import Rotation as Rot
import matplotlib.pyplot as plt


class TrackingController:
    def __init__(self, dt):
        cwd = pathlib.Path(
            __file__
        ).parent.resolve()  # path to current working directory
        pi = optas.np.pi  # 3.141...
        T = 1  # no. time steps in trajectory
        link_ee = "lbr_link_ee"  # end-effector link name

        # Setup robot
        urdf_filename = os.path.join(cwd, "robots", "kuka_lbr", "med7.urdf")
        kuka = optas.RobotModel(
            urdf_filename=urdf_filename,
            time_derivs=[1],  # i.e. joint velocity
        )
        kuka_name = kuka.get_name()
        self.kuka = kuka
        # setup model
        X_dim = 6  # x,y,z, Rx, Ry, Rz dimensions
        dim = {0: [-1.5, -1.5]}

        T = 1
        builder = optas.OptimizationBuilder(T, robots=[kuka], derivs_align=True)
        # Setup parameters
        qc = builder.add_parameter("qc", kuka.ndof)  # current robot joint configuration
        pg = builder.add_parameter("pg", 7)

        # Get joint velocity
        dq = builder.get_model_state(kuka_name, t=0, time_deriv=1)

        # Get next joint state
        q = qc + dt * dq

        # Get jacobian
        J = kuka.get_global_link_geometric_jacobian(link_ee, qc)

        # Get end-effector velocity
        dp = J @ dq

        # Get current end-effector position
        pc = kuka.get_global_link_position(link_ee, qc)
        Rc = kuka.get_global_link_rotation(link_ee, qc)

        print("dp = {0}".format(dp.size()))
        Om = skew(dp[3:])

        # Get next end-effector position (Global)
        p = pc + dt * dp[:3]
        R = (Om * dt + I3()) @ Rc

        # Get next end-effector position (Current end-effector position)

        p = Rc.T @ dt @ dp[:3]
        R = Rc.T @ (Om * dt + I3())

        # Cost: match end-effector position
        Rotq = Quaternion(pg[3], pg[4], pg[5], pg[6])
        Rg = Rotq.getrotm()

        pg_ee = -Rc.T @ pc + Rc.T @ pg[:3]
        Rg_ee = Rc.T @ Rg

        diffp = p - pg_ee[:3]
        diffR = Rg_ee.T @ R

        W_p = optas.diag([1e3, 1e3, 1e3])
        builder.add_cost_term("match_p", diffp.T @ W_p @ diffp)

        w_dq = 0.01
        builder.add_cost_term("min_dq", w_dq * optas.sumsqr(dq))

        w_ori = 1e1
        builder.add_cost_term("match_r", w_ori * optas.sumsqr(diffR - I3()))

        builder.add_leq_inequality_constraint("eff_x", diffp[0] * diffp[0], 1e-6)
        builder.add_leq_inequality_constraint("eff_y", diffp[1] * diffp[1], 1e-8)
        builder.add_leq_inequality_constraint("eff_z", diffp[2] * diffp[2], 1e-8)

        optimization = builder.build()
        self.solver = optas.CasADiSolver(optimization).setup("sqpmethod")
        # Setup variables required later
        self.kuka_name = kuka_name

    def compute_target_velocity(self, qc, pg):
        self.solver.reset_parameters({"qc": optas.DM(qc), "pg": optas.DM(pg)})
        solution = self.solver.solve()
        return solution[f"{self.kuka_name}/dq"].toarray().flatten()


def main(gui=True):
    # Setup PyBullet
    qc = optas.np.deg2rad([0, 30, 0, -90, 0, 60, 0])
    # qc = optas.np.array([-0.03826181,  0.69341676, -0.02385922,  4.34133172, -0.01004864, -1.21899326,-0.02354659])
    q = qc.copy()

    hz = 500
    dt = 1.0 / float(hz)
    pb = pybullet_api.PyBullet(
        dt,
        camera_distance=0.5,
        camera_target_position=[0.3, 0.2, 0.0],
        camera_yaw=135,
        gui=gui,
    )
    # kuka = pybullet_api.KukaLWR()
    cwd = pathlib.Path(__file__).parent.resolve()
    path = os.path.join(cwd, "robots", "kuka_lbr", "med7.urdf")
    kuka = pybullet_api.FixedBaseRobot(path)
    #urdf_filename = path
    kuka.reset(qc)

    box_base_position = [0.0, 0.0, 0.06]
    pb.start()

    start_time = pybullet_api.time.time()

    # Move robot to start position
    Tmax_start = 10.0
    transl = kuka.robot.get_global_link_position("lbr_link_ee", q)
    pginit = optas.np.array([float(transl[0]), float(transl[1]), float(transl[2]), 0.0, 1.00, 0.0, 0.0])
    ik = TrackingController(dt)

    box_half_extents = [0.01, 0.01, 0.01]
    constdist = optas.np.array([0.0, 0.0, -0.04])
    box = pybullet_api.VisualBox(
        base_position=pginit[:3] + constdist,
        half_extents=box_half_extents,
    )

    ts = []
    xs = []
    xgs = []
    ys = []
    ygs = []
    zs = []
    zgs = []
    dqgoal_last = optas.np.zeros(7)
    nv = 1.0

    pybullet_api.p.setRealTimeSimulation(0)

    while True:
        t = pybullet_api.time.time() - start_time
        if t > Tmax_start:
            break
        q = kuka.q()
        transl = kuka.robot.get_global_link_position("lbr_link_ee", q)
        pginit1 = pginit + optas.np.array(
                [
                    0.03 * optas.np.sin(2 * pi * nv * t),
                    0.03 * optas.np.cos(2 * pi * nv * t),
                    0.0002 * pi * t,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            )
        
        box.reset(base_position=constdist + pginit1[:3], base_orientation=pginit1[3:])
        print("pginit1 = {0}".format(pginit1))
        # dqgoal_last = dqgoal
        dqgoal = ik.compute_target_velocity(q, pginit1)

        qdd = (dqgoal-dqgoal_last)/dt

        q += dt * dqgoal  #+ optas.np.random.uniform(-0.01, 0.01, kuka.ndof)  # noise

        qd = dqgoal

        print("q = {0}".format(q))
        # print("qd = {0}".format(qd))
        # print("qdd = {0}".format(qdd))

        # q = ik.kuka.get_random_joint_positions().toarray().flatten()
        # qd = ik.kuka.get_random_joint_positions().toarray().flatten()
        # qdd = ik.kuka.get_random_joint_positions().toarray().flatten()

        # q += dt * dqgoal #ideal control
        # taus =optas.np.zeros(7)
        taus = ik.kuka.rnea(optas.np.asarray(q), 
                            optas.np.asarray(qd), 
                            optas.np.asarray(qdd)).toarray().flatten()  +1.0 *optas.np.eye(7)@(dqgoal*dt) - 0.01*optas.np.eye(7)@(dqgoal)
        
        print("taus = {0}".format(taus))
        kuka.cmd_torque(taus)
        for i in range(1):
            pybullet_api.p.stepSimulation()
            pybullet_api.time.sleep(dt * float(gui)/1.0)
            
        # q = kuka.q()
        # kuka.reset(q)

        # transl = kuka.robot.get_global_link_position("lbr_link_ee", q)
        # print(T)
        ts.append(t)
        xs.append(float(transl[0]))
        xgs.append(float(pginit1[0]))

        ys.append(float(transl[1]))
        ygs.append(float(pginit1[1]))

        zs.append(float(transl[2]))
        zgs.append(float(pginit1[2]))

    # pybullet_api.time.sleep(5.0)
    pb.stop()
    pb.close()
    fig = plt.figure()
    xgs_np = optas.np.array(xgs)
    xs_np = optas.np.array(xs)

    print("X error mean = {0} mm".format(1000.0 * optas.np.mean(xs_np - xgs_np)))
    # print("X error mean = {0} mm".format(1000.0 * optas.np.mean(ys_np - xgs_np)))
    # print("X error mean = {0} mm".format(1000.0 * optas.np.mean(zs_np - xgs_np)))

    plt.subplot(311)
    plt.plot(ts, xs, label="Actual x")
    plt.plot(ts, xgs, label="Goal x")
    plt.title("Positional following re x-axis")
    plt.subplot(312)
    plt.plot(ts, ys, label="Actual y")
    plt.plot(ts, ygs, label="Goal y")
    plt.title("Positional following re y-axis")
    plt.subplot(313)
    plt.plot(ts, zs, label="Actual z")
    plt.plot(ts, zgs, label="Goal z")
    plt.title("Positional following re z-axis")
    plt.show()


if __name__ == "__main__":
    sys.exit(main())

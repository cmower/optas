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
    def __init__(self,dt):
        cwd = pathlib.Path(
            __file__
        ).parent.resolve()  # path to current working directory
        pi = optas.np.pi  # 3.141...
        T = 1  # no. time steps in trajectory
        link_ee = "end_effector_ball"  # end-effector link name

        # Setup robot
        urdf_filename = os.path.join(cwd, "robots", "kuka_lwr", "kuka_lwr.urdf")
        kuka = optas.RobotModel(
            urdf_filename=urdf_filename,
            time_derivs=[1],  # i.e. joint velocity
        )
        kuka_name = kuka.get_name()

        

        # setup model
        X_dim = 6 # x,y,z, Rx, Ry, Rz dimensions
        dim =  {0: [-1.5, -1.5]}

        T = 1
        builder = optas.OptimizationBuilder(T, robots=[kuka], derivs_align=True)
        # Setup parameters
        qc = builder.add_parameter("qc", kuka.ndof)  # current robot joint configuration
        pg = builder.add_parameter("pg", 7) # current robot joint configuration
        fh = builder.add_parameter("fh", 6)  

        # Get joint velocity
        dq = builder.get_model_state(kuka_name, t=0, time_deriv=1)

        # Get next joint state
        q = qc + dt * dq

        # Get jacobian
        J = kuka.get_link_geometric_jacobian(link_ee, qc,link_ee)
        # J = kuka.get_global_link_geometric_jacobian(link_ee, qc)


        # Get end-effector velocity
        dp = J @ dq

        # Get current end-effector position
        pc = kuka.get_global_link_position(link_ee, qc)
        Rc = kuka.get_global_link_rotation(link_ee, qc)
        
        print("dp = {0}".format(dp.size()))
        Om = skew(dp[3:]) 

        #TODO
        # 分解dp 并将其加到位置上和姿态上
        # Get next end-effector position (Global)
        # p = pc + dt * dp[:3]
        # R = (Om * dt + I3()) @ Rc
        
        # Get next end-effector position (Current end-effector position)
        
        p = Rc.T @ dt @ dp[:3]
        R = Rc.T @ (Om * dt + I3())

        # print("Rc.T = {0}".format(Rc.T.size()))
        # print("pc = {0}".format(pc.size()))
        # print("dp[:3] = {0}".format(dp[:3].size()))
        # print("pc = {0}".format(pc.size()))
        
        # Cost: match end-effector position
        Rotq = Quaternion(pg[3],pg[4],pg[5],pg[6])
        Rg = Rotq.getrotm()

        pg_ee = -Rc.T @ pc + Rc.T @ pg[:3]
        Rg_ee = Rc.T @ Rg

        # rgag = pg[3]*pg[3]+pg[4]*pg[4]+pg[5]*pg[5]
        # rgv = unit(pg[3:])


        # e = optas.DM([0, 0, -1.0])
        # z = R[:, 2]
        # print("e = {0}".format(e.size()))
        # print("e = {0}".format(e))
        # f = e.T @ z
        # print("e.T @ z = {0}".format(f))
        # # print("e.T @ z = {0}".format(e.T @ z))
        # print("e.T @ z = {0}".format(f.size()))
        # r2angvec()

        diffR = Rg_ee.T @ R
        # diffp = (I3() - nf @ nf.T) @ (p - pg_ee[:3])
        
        nf = optas.DM([1.0, 1.0, 1.0])
        W_nf = optas.diag([1.0, 1.0, 1.0])
        # diffFl = nf @ nf.T @Rc.T @  (fh[:3] -  dp[:3])
        diffFl =  nf @ nf.T @ (Rc.T @fh[:3] -  dp[:3])

        # diffFr = nf @ nf.T @ (fh[3:] - optas.diag([1e-3, 1e-3, 1e-3]) @ Rc.T @dq[3:])

        # W_p = optas.diag([1e3, 1e3, 1e3])
        # builder.add_cost_term("match_p", diffp.T @ W_p @ diffp)

        W_f = optas.diag([1e2, 1e2, 1e2])
        builder.add_cost_term("match_f", diffFl.T @ W_f @ diffFl)
        
        w_dq = 0.01
        builder.add_cost_term("min_dq", w_dq * optas.sumsqr(dq))

        # w_ori = 1e1
        # builder.add_cost_term("match_r", w_ori * optas.sumsqr(diffR - I3()))

        # builder.add_leq_inequality_constraint(
        #     "eff_x", diffp[0] * diffp[0], 1e-6
        # )
        # builder.add_leq_inequality_constraint(
        #     "eff_y", diffp[1] * diffp[1], 1e-8
        # )
        builder.add_leq_inequality_constraint(
            "dq1", dq[0] * dq[0], 1e-1
        )
        builder.add_leq_inequality_constraint(
            "dq2", dq[1] * dq[1], 1e-1
        )
        builder.add_leq_inequality_constraint(
            "dq3", dq[2] * dq[2], 1e-1
        )
        builder.add_leq_inequality_constraint(
            "dq4", dq[3] * dq[3], 1e-1
        )
        builder.add_leq_inequality_constraint(
            "dq5", dq[4] * dq[4], 1e-1
        )
        builder.add_leq_inequality_constraint(
            "dq6", dq[5] * dq[5], 1e-1
        )
        builder.add_leq_inequality_constraint(
            "dq7", dq[6] * dq[6], 1e-1
        )
        # builder.add_leq_inequality_constraint(
        #     "eff_z", diffp[2] * diffp[2], 1e-8
        # )

        optimization = builder.build()
        # self.solver = optas.CasADiSolver(optimization).setup("sqpmethod")
        self.solver = optas.ScipyMinimizeSolver(builder.build()).setup('SLSQP')
        # Setup variables required later
        self.kuka_name = kuka_name

    def compute_target_velocity(self, qc, pg, fh):
        self.solver.reset_parameters({"qc": optas.DM(qc), "pg": optas.DM(pg), "fh": optas.DM(fh)})
        solution = self.solver.solve()
        return solution[f"{self.kuka_name}/dq"].toarray().flatten()


def main(gui=True):
    # Setup PyBullet
    qc = -optas.np.deg2rad([20, 30, 10, 90, 30, 00, 00])
    q = qc.copy()

    hz = 50
    dt = 1.0/float(hz)
    pb = pybullet_api.PyBullet(
        dt,
        camera_distance=0.5,
        camera_target_position=[0.3, 0.2, 0.0],
        camera_yaw=135,
        gui=gui,
    )
    kuka = pybullet_api.KukaLWR()
    kuka.reset(qc)

    box_base_position = [0.0, 0.0, 0.06]
    pb.start()

    start_time = pybullet_api.time.time()

    

    # Move robot to start position
    Tmax_start = 6.0
    pginit = optas.np.array([0.5, 0.0, 0.16, 0.0, 1.00, 0.0, 0.0])
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
    ygs =[]
    zs =[]
    zgs =[]
    fhzs = []
    manipulatiys= []

    while True:
        t = pybullet_api.time.time() - start_time
        if t > Tmax_start:
            break
        # box.reset(
        #     base_position=constdist)
            # base_orientation=yaw2quat(state[2]).toarray().flatten(),
        #)
        nv = 0.5
        fh = optas.np.array([0.2*optas.np.cos(2*pi*nv*t),
                                  0.2*optas.np.cos(2*pi*nv*t),
                                  0.2*optas.np.cos(2*pi*nv*t), # changing force
                                  0.0,
                                  0.0,
                                  0.0]) 
        pginit1 = pginit + optas.np.array([0.03*optas.np.sin(2*pi*nv*t),
                                  0.03*optas.np.cos(2*pi*nv*t),
                                  0.02*pi*t,
                                  0.0,
                                  0.0,
                                  0.0,
                                  0.0]) + optas.np.random.uniform(-0.01, 0.01, kuka.ndof)
        box.reset(
            base_position=constdist+pginit1[:3],
            base_orientation= pginit1[3:])
        
        dqgoal = ik.compute_target_velocity(q, pginit1, fh)


        q += dt * dqgoal #+ optas.np.random.uniform(-0.01, 0.01, kuka.ndof) # noise

        # for v in dqgoal:
        #     print(v)
        #     if v>0.1:
        #         v = 0.1
        #     elif v<0.1:
        #         v = -0.1
        #     else:
        #         pass


        # q += dt * dqgoal #ideal control
        kuka.cmd(q)


        transl = kuka.robot.get_global_link_position("end_effector_ball", q)
        Jl = kuka.robot.get_global_link_linear_jacobian("end_effector_ball", q)
        Rc = kuka.robot.get_global_link_rotation("end_effector_ball", q)

        J = kuka.robot.get_global_link_geometric_jacobian("end_effector_ball", q)

        transl = Rc.T @ transl
        pginit_show = Rc.T @ pginit1[:3]
        fl = Jl @ dqgoal

        manipulatiy = optas.np.linalg.det(J @ J.T)
        # print(T)
        pybullet_api.time.sleep(dt * float(gui))
        ts.append(t)
        xs.append(float(fl[0]))
        xgs.append(float(fh[0]))

        ys.append(float(fl[1]))
        ygs.append(float(fh[1]))

        zs.append(float(fl[2]))
        zgs.append(float(fh[2]))
        manipulatiys.append(float(manipulatiy))

        


    # pybullet_api.time.sleep(5.0)
    pb.stop()
    pb.close()
    fig = plt.figure()
    xgs_np = optas.np.array(xgs)
    xs_np = optas.np.array(xs)

    print("X error mean = {0} mm".format(1000.0*optas.np.mean(xs_np-xgs_np)))
    
    plt.subplot(221)
    plt.plot(ts, xs, label ='Actual x')
    plt.plot(ts, xgs, label ='Goal x')
    plt.title("Positional following re x-axis")
    plt.subplot(222)
    plt.plot(ts, ys, label ='Actual y')
    plt.plot(ts, ygs, label ='Goal y')
    plt.title("Positional following re y-axis")
    plt.subplot(223)
    plt.plot(ts, zs, label ='Actual z')
    plt.plot(ts, zgs, label ='Goal z')
    plt.title("Positional following re z-axis")
    plt.subplot(224)
    plt.plot(ts, manipulatiys, label ='Mani')
    # plt.plot(ts, zgs, label ='Goal z')
    plt.title("Positional following re z-axis")
    plt.show()





if __name__ == "__main__":
    sys.exit(main())






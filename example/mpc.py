# Python standard lib
import os
import sys
import pathlib

# PyBullet
import pybullet_api

# OpTaS
import optas


######################################
# Task space planner and controller
#
# This is an implementation of [1].
#
# References
#
#   1. J. Moura, T. Stouraitis, and S. Vijayakumar, Non-prehensile
#      Planar Manipulation via Trajectory Optimization with
#      Complementarity Constraints, ICRA, 2022.
#


class TOMPCC:

    def __init__(self):
        self.L = optas.diag([1., 1., 1.])


class TOMPCCPlanner(TOMPCC):

    def __init__(self):

        # Setup
        super().__init__()
        T = 100

    def plan(self):
        pass


class TOMPCCController(TOMPCC):

    def __init__(self):

        # Setup
        super().__init__()
        T = 10

    def compute_target_velocity(self):
        pass


######################################
# Joint space IK
#


class IK:

    def __init__(self, dt, thresh_angle):

        cwd = pathlib.Path(__file__).parent.resolve() # path to current working directory
        pi = optas.np.pi  # 3.141...
        T = 1 # no. time steps in trajectory
        link_ee = 'end_effector_ball'  # end-effector link name

        # Setup robot
        urdf_filename = os.path.join(cwd, 'robots', 'kuka_lwr.urdf')
        kuka = optas.RobotModel(
            urdf_filename,
            time_derivs=[1],  # i.e. joint velocity
        )
        kuka_name = kuka.get_name()

        # Setup optimization builder
        builder = optas.OptimizationBuilder(T=T, robots=[kuka], derivs_align=True)

        # Setup parameters
        qc = builder.add_parameter('qc', kuka.ndof)  # current robot joint configuration
        pg = builder.add_parameter('pg', 3)  # goal end-effector position

        # Get joint velocity
        dq = builder.get_model_state(kuka_name, t=0, time_deriv=1)

        # Get next joint state
        q = qc + dt*dq

        # Get jacobian
        Jl = kuka.get_global_linear_geometric_jacobian(link_ee, qc)

        # Get end-effector velocity
        dp = Jl @ dq

        # Get current end-effector position
        pc = kuka.get_global_link_position(link_ee, qc)

        # Get next end-effector position
        p = pc + dt*dp

        # Cost: match end-effector position
        diffp = p - pg
        W_p = optas.diag([20., 20., 1.])
        builder.add_cost_term('match_p', diffp.T @ W_p @ diffp)

        # Cost: min joint velocity
        w_dq = 0.01
        builder.add_cost_term('min_dq', w_dq*optas.sumsqr(dq))

        # Get global z-axis of end-effector
        T = kuka.get_global_link_transform(link_ee, q)
        z = T[:3, 2]

        # Constraint: eff orientation
        e = optas.DM([0, 0, -1.])
        builder.add_leq_inequality_constraint('eff_orien', optas.cos(thresh_angle), e.T @ z)

        # Cost: align eff
        w_ori = 10.
        builder.add_cost_term('eff_orien', w_ori*optas.sumsqr(e.T @ z - 1))

        # Setup solver
        optimization = builder.build()
        self.solver = optas.CasADiSolver(optimization).setup('sqpmethod')

        # Setup variables required later
        self.kuka_name = kuka_name

        # Setup functions for later
        self._pos = kuka.get_global_link_position_function(link_ee)

    def ee_pos(self, q):
        return self._pos(q).toarray().flatten()

    def compute_target_velocity(self, qc, pg):
        self.solver.reset_parameters({'qc': optas.DM(qc), 'pg': optas.DM(pg)})
        solution = self.solver.solve()
        return solution[f'{self.kuka_name}/dq'].toarray().flatten()

def main():

    # Setup PyBullet
    qc = -optas.np.deg2rad([0, 30, 0, -90, 0, 60, 0])
    q = qc.copy()
    hz = 50
    dt = 1.0/float(hz)
    pb = pybullet_api.PyBullet(dt)
    kuka = pybullet_api.Kuka()
    kuka.reset(qc)
    box_base_position = [0.4, 0.065, 0.06]
    box_half_extents = [0.1, 0.05, 0.06]
    box = pybullet_api.DynamicBox(
        base_position=box_base_position,
        half_extents=box_half_extents,
    )
    pybullet_api.VisualBox(
        base_position=[0.6, -0.2, 0.06],
        half_extents=box_half_extents,
        rgba_color=[1., 0., 0., 0.5],
    )

    # Setup TO MPCC planner
    planner = TOMPCCPlanner()

    # Setup TO MPCC controller
    controller = TOMPCCController()

    # Setup IK
    thresh_angle = optas.np.deg2rad(30.)
    ik = IK(dt, thresh_angle)

    # Start pybullet
    pb.start()
    start_time = pybullet_api.time.time()

    # Move robot to start position
    Tmax_start = 6.
    pginit = optas.np.array([0.4, 0., 0.06])
    while True:
        t = pybullet_api.time.time() - start_time
        if t > Tmax_start:
            break
        dqgoal = ik.compute_target_velocity(q, pginit)
        q += dt*dqgoal
        kuka.cmd(q)
        pybullet_api.time.sleep(dt)

    # Main loop
    p = pginit.copy()
    start_time = pybullet_api.time.time()
    while True:
        t = pybullet_api.time.time() - start_time
        boxpose = box.get_pose()
        dqgoal = ik.compute_target_velocity(q, p)
        q += dt*dqgoal
        kuka.cmd(q)
        pybullet_api.time.sleep(dt)

    pb.stop()
    pb.close()

    return 0


if __name__ == '__main__':
    sys.exit(main())

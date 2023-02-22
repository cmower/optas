# Python standard lib
import os
import sys
import pathlib

# OpTaS
import optas
from optas.templates import Manager

# PyBullet
import pybullet_api

kukal_base_position = [0.0, -0.25, 0.0]
kukar_base_position = [0.0, 0.25, 0.0]


class DualKukaPlanner(Manager):
    def setup_solver(self):
        # Parameters
        T = 50
        Tmax = 10.0
        link_ee = "end_effector_ball"
        t = optas.linspace(0, Tmax, T)
        dt = float((t[1] - t[0]).toarray()[0, 0])

        # Setup robot models
        kukal = self._setup_kuka_model("kukal", kukal_base_position)
        kukar = self._setup_kuka_model("kukar", kukar_base_position)

        # Get robot names
        kukal_name = kukal.get_name()
        kukar_name = kukar.get_name()

        # Setup optimization builder
        builder = optas.OptimizationBuilder(T=T, robots=[kukal, kukar])

        # Setup parameters
        qcl = builder.add_parameter("qcl", kukal.ndof)
        qcr = builder.add_parameter("qcr", kukar.ndof)

        # Constraint: initial configuration
        builder.fix_configuration(kukal_name, qcl)
        builder.fix_configuration(kukar_name, qcr)

        # Constraint: dynamics
        builder.integrate_model_states(
            kukal_name,
            time_deriv=1,  # i.e. integrate velocities to positions
            dt=dt,
        )
        builder.integrate_model_states(
            kukar_name,
            time_deriv=1,  # i.e. integrate velocities to positions
            dt=dt,
        )

        # Get position FK function
        posl_ee = kukal.get_global_link_position_function(link_ee, n=T)
        posr_ee = kukar.get_global_link_position_function(link_ee, n=T)

        # Get joint trajectory
        Ql = builder.get_model_states(
            kukal_name
        )  # ndof-by-T sym array for robot trajectory
        Qr = builder.get_model_states(
            kukar_name
        )  # ndof-by-T sym array for robot trajectory

        # Get end-effector position trajectories
        ee_pos_pathl = posl_ee(Ql)
        ee_pos_pathr = posr_ee(Qr)

        # Get joint velocity trajectory
        dQl = builder.get_model_states(kukal_name, time_deriv=1)
        dQr = builder.get_model_states(kukar_name, time_deriv=1)

        # Cost: minimize joint velocity
        w_dq = 0.01
        builder.add_cost_term("kukal_min_join_vel", w_dq * optas.sumsqr(dQl))
        builder.add_cost_term("kukar_min_join_vel", w_dq * optas.sumsqr(dQr))

        # Get start position for each robot
        pos0l = kukal.get_global_link_position(link_ee, qcl)
        pos0r = kukar.get_global_link_position(link_ee, qcr)

        # Find first goal positions
        pos1l = pos0l + optas.DM([-0.1, 0.1, -0.2])
        pos1r = pos0r + optas.DM([-0.1, -0.1, -0.2])

        # Find second goal positions
        pos2l = pos1l + optas.DM([0.0, 0.0, 0.3])
        pos2r = pos1r + optas.DM([0.0, 0.0, 0.3])

        # Find ee path
        path_eel = optas.SX.zeros(3, T)
        path_eer = optas.SX.zeros(3, T)
        for i in range(T):
            alpha_ = float(i) / float(T - 1)

            if alpha_ < 0.4:
                alpha = alpha_ / 0.4

                path_eel[:, i] = alpha * pos1l + (1.0 - alpha) * pos0l
                path_eer[:, i] = alpha * pos1r + (1.0 - alpha) * pos0r

            elif 0.4 <= alpha_ < 0.5:
                path_eel[:, i] = pos1l
                path_eer[:, i] = pos1r

            else:
                alpha = (alpha_ - 0.5) / 0.5

                path_eel[:, i] = alpha * pos2l + (1.0 - alpha) * pos1l
                path_eer[:, i] = alpha * pos2r + (1.0 - alpha) * pos1r

        builder.add_cost_term("ee_pos_pathl", optas.sumsqr(ee_pos_pathl - path_eel))
        builder.add_cost_term("ee_pos_pathr", optas.sumsqr(ee_pos_pathr - path_eer))

        # Setup solver
        optimization = builder.build()
        solver = optas.CasADiSolver(optimization).setup("ipopt")

        # Save variables for later
        self.kukal_name = kukal_name
        self.kukar_name = kukar_name
        self.Tmax = Tmax

        return solver

    def _setup_kuka_model(self, name, base_position):
        cwd = pathlib.Path(
            __file__
        ).parent.resolve()  # path to current working directory
        urdf_filename = os.path.join(cwd, "robots", "kuka_lwr", "kuka_lwr.urdf")
        model = optas.RobotModel(
            urdf_filename=urdf_filename,
            name=name,
            time_derivs=[0, 1],  # i.e. joint position/velocity trajectory
        )

        model.add_base_frame("global_world", xyz=base_position)

        return model

    def is_ready(self):
        return True

    def reset(self, qcl, qcr):
        # Set parameters
        self.solver.reset_parameters(
            {
                "qcl": optas.DM(qcl),
                "qcr": optas.DM(qcr),
            }
        )

    def get_target(self):
        return self.solution

    def plan(self):
        self.solve()
        solution = self.get_target()

        # Interpolate
        planl = self.solver.interpolate(solution[f"{self.kukal_name}/q"], self.Tmax)
        planr = self.solver.interpolate(solution[f"{self.kukar_name}/q"], self.Tmax)

        return planl, planr


def main(gui=True):
    dual_kuka_planner = DualKukaPlanner()

    hz = 50
    dt = 1.0 / float(hz)
    pb = pybullet_api.PyBullet(dt, gui=gui)

    box = pybullet_api.DynamicBox(
        base_position=[0.75, 0, 0.15], half_extents=[0.15, 0.15, 0.15]
    )

    kukal = pybullet_api.KukaLWR(base_position=kukal_base_position)
    kukar = pybullet_api.KukaLWR(base_position=kukar_base_position)

    qc = optas.np.deg2rad([0, -30, 0, 90, 0, 30, 0])
    kukal.reset(qc)
    kukar.reset(qc)

    dual_kuka_planner.reset(qc, qc)
    planl, planr = dual_kuka_planner.plan()

    pb.start()
    pybullet_api.time.sleep(2.0)
    start_time = pybullet_api.time.time()

    while True:
        t = pybullet_api.time.time() - start_time
        if t > dual_kuka_planner.Tmax:
            break

        kukal.cmd(planl(t))
        kukar.cmd(planr(t))

        pybullet_api.time.sleep(dt*float(gui))

    pybullet_api.time.sleep(10.0*float(gui))

    pb.stop()
    pb.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())

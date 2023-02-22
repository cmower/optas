import optas
import numpy as np
from pybullet_api import *
from optas.templates import Manager


class SimpleJointSpacePlanner(Manager):
    def __init__(self, urdf_string, ee_link, duration):
        self.duration = duration
        self.ee_link = ee_link
        self.urdf_string = urdf_string
        super().__init__()

    def setup_solver(self):
        T = 20  # number of time steps
        dt = self.duration / float(T - 1)

        self.robot = optas.RobotModel(urdf_string=self.urdf_string, time_derivs=[0, 1])
        self.name = self.robot.get_name()
        builder = optas.OptimizationBuilder(T=T, robots=self.robot, derivs_align=True)

        qn = builder.add_parameter("nominal_joint_state", self.robot.ndof)
        qc = builder.add_parameter("current_joint_state", self.robot.ndof)
        pg = builder.add_parameter("position_goal", 3)
        og = builder.add_parameter("orientation_goal", 4)

        # Constraint: initial configuration
        builder.fix_configuration(self.name, config=qc)

        # Constraint: final pose
        qF = builder.get_model_state(self.name, -1)
        pF = self.robot.get_global_link_position(self.ee_link, qF)
        oF = self.robot.get_global_link_quaternion(self.ee_link, qF)
        builder.add_equality_constraint("final_position", pF, pg)
        builder.add_equality_constraint("final_orientation", oF, og)

        # Constraint: dynamics
        builder.integrate_model_states(self.name, time_deriv=1, dt=dt)

        # Constraint: keep end-effector above zero
        zpad = 0.05  # cm
        for t in range(T):
            q = builder.get_model_state(self.name, t)

            # Cost: nominal pose
            builder.add_cost_term(f"nominal_{t}", 0.1 * optas.sumsqr(q - qn))

            p = self.robot.get_global_link_position(self.ee_link, q)
            z = p[2]
            zsafe = z + zpad
            builder.add_geq_inequality_constraint(f"eff_safe_{t}", zsafe)

            p = self.robot.get_global_link_position("lbr_link_3", q)
            z = p[2]
            zsafe = z + zpad
            builder.add_geq_inequality_constraint(f"elbow_safe_{t}", zsafe)

        # Cost: minimize joint velocity
        dQ = builder.get_model_states(self.name, time_deriv=1)
        w_min_vel = 0.1
        builder.add_cost_term("minimize_velocity", w_min_vel * optas.sumsqr(dQ))

        # Cost: minmize joint acceleration
        ddQ = (dQ[:, 1:] - dQ[:, :-1]) / dt
        w_min_acc = 10
        builder.add_cost_term("minimize_acceleration", w_min_acc * optas.sumsqr(ddQ))

        # Constraint: final velocity is zero
        builder.fix_configuration(self.name, t=-1, time_deriv=1)

        solver = optas.CasADiSolver(builder.build()).setup("ipopt")
        return solver

    def is_ready(self):
        return True

    def reset(self, qc, pg, og, qn):
        self.solver.reset_parameters(
            {
                "current_joint_state": qc,
                "position_goal": pg,
                "orientation_goal": og,
                "nominal_joint_state": qn,
            }
        )

    def get_target(self):
        return self.solution

    def plan(self):
        self.solve()
        solution = self.get_target()
        plan = self.solver.interpolate(solution[f"{self.name}/q"], self.duration)
        return plan


def main():
    hz = 250
    dt = 1.0 / float(hz)
    pb = PyBullet(dt)
    kuka = KukaLBR()

    q0 = np.deg2rad([0, 45, 0, -90, 0, -45, 0])

    kuka.reset(q0)

    duration = 4.0  # seconds
    planner = SimpleJointSpacePlanner(kuka.urdf_string, "lbr_link_ee", duration)

    qc = kuka.q()
    pg = [0.4, 0.3, 0.4]
    og = [0, 1, 0, 0]

    planner.reset(qc, pg, og, q0)
    plan = planner.plan()

    pb.start()

    start = time.time()
    while True:
        t = time.time() - start
        if t < duration:
            kuka.cmd(plan(t))
        else:
            print("Completed motion")
            break

    while True:
        pass

    pb.stop()
    pb.close()

    return 0


if __name__ == "__main__":
    main()

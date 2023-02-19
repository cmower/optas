from . import *


class SimpleJointMotionPlanner:

    """

    Simple motion planner in the joint space
    ----------------------------------------

    This method plans a joint space trajectory from a initial
    configuration to a target configuration. Optionally, joint
    velocity/acceleration can be minimized.

    t          - time
    q(t)       - joint position
    q*(t)      - ideal joint position
    T          - final time
    dq(t)/dt   - joint velocity
    d2q(t)/dt2 - joint acceleration
    q0         - initial joint state
    qT         - target joint state at time T
    q-, q+     - lower, upper joint position limits
    dq-, dq+   - lower, upper joint velocity limits

      q*(t) = arg min ||q(T) - qT||^2 + integrate_0^T  ||dq(t)/dt||^2 + ||d2q(t)/dt2||^2  dt
                q(t)

          subject to

                              q(0) = q0
                              q(T) = qT        (optional)
                      q-  <=  q(t)    <= q+    (optional)
                      dq- <= dq(t)/dt <= dq+   (optional)

    """

    def __init__(
        self,
        urdf_filename,
        T=100,
        min_q_vel_w=0.0,
        min_q_acc_w=0.0,
        final_q_w=1.0,
        final_q_eq_con=True,
        qlim=True,
        qdlim=None,
        xlim=None,
        ylim=None,
        zlim=None,
        straight_line_warm_start=False,
    ):
        # Setup class attributes
        self.T = T
        self.straight_line_warm_start = straight_line_warm_start

        # Setup robot
        self.robot_model = RobotModel(urdf_filename=urdf_filename)
        self.name = self.robot_model.get_name()

        # Setup optimization builder
        builder = OptimizationBuilder(T, robots=[self.robot_model])

        # Get model states
        Q = builder.get_model_states(self.name)
        dQ = Q[:, 1:] - Q[:, :-1]
        ddQ = dQ[:, 1:] - dQ[:, :-1]

        # Add parameters
        q0 = builder.add_parameter("q0", self.robot_model.ndof)
        qT = builder.add_parameter("qT", self.robot_model.ndof)

        # Add equality constraints
        builder.add_equality_constraint("init_q", Q[:, 0], q0)
        if final_q_eq_con:
            builder.add_equality_constraint("final_q", Q[:, -1], qT)

        # Add cost terms
        if final_q_w > 0.0:
            builder.add_cost_term("final_q", final_q_w * sumsqr(Q[:, -1] - qT))
        if min_q_vel_w > 0.0:
            builder.add_cost_term("min_vel", min_q_vel_w * sumsqr(dQ))
        if min_q_acc_w > 0.0:
            builder.add_cost_term("min_acc", min_q_acc_w * sumsqr(ddQ))

        # Add limit constraints
        if qlim:
            builder.add_bound_inequality_constraint(
                "qlim",
                self.robot_model.lower_actuated_joint_limits,
                Q,
                self.robot_model.upper_actuated_joint_limits,
            )
        if qdlim is not None:
            builder.add_bound_inequality_constraint(
                "qdlim", -fabs(qdlim), dQ, fabs(qdlim)
            )
        for link in self.robot_model.link_names:
            if link == self.robot_model.get_root_link():
                continue
            P = self.robot_model.get_global_link_position(link, Q)
            if isinstance(xlim, (list, tuple)):
                builder.add_bound_inequality_constraint(
                    f"xlim_{link}", xlim[0], P[0, :], xlim[1]
                )
            if isinstance(ylim, (list, tuple)):
                builder.add_bound_inequality_constraint(
                    f"ylim_{link}", ylim[0], P[1, :], ylim[1]
                )
            if isinstance(zlim, (list, tuple)):
                builder.add_bound_inequality_constraint(
                    f"zlim_{link}", zlim[0], P[2, :], zlim[1]
                )

        # Setup solver
        self.solver = CasADiSolver(builder.build()).setup("ipopt")

    @arrayify_args
    def plan(self, qa, qb):
        self.solver.reset_parameters({"q0": qa, "qT": qb})
        if self.straight_line_warm_start:
            Q0 = DM.zeros(self.robot_model.ndof, self.T)
            for i in range(self.T):
                alpha = float(i) / float(self.T - 1)
                Q0[:, i] = qa * (1 - alpha) + qb * alpha
            self.solver.reset_initial_seed({f"{self.name}/q/x": Q0})
        solution = self.solver.solve()
        success = self.solver.did_solve()
        plan = self.solver.interpolate(
            solution[f"{self.name}/q/x"], 1.0, fill_value="extrapolate"
        )
        return success, plan

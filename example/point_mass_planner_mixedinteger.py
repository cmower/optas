import optas
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class Planner:
    def __init__(self):
        # Planner attributes
        dt = 0.1  # time step
        obs = [0, 0]  # obstacle position
        obs_rad = 0.2  # obstacle radii

        # Setup point mass model
        pm_radius = 0.1  # point mass radii
        pm_dim = 2  # x, y dimensions
        dlim = {0: [-1.5, 1.5], 1: [-5.0, 5.0]}  # pos/vel limits
        point_mass = optas.TaskModel(
            "point_mass", pm_dim, time_derivs=[0, 1], dlim=dlim
        )
        pm_name = point_mass.get_name()
        max_control = 10.0
        point_mass_control = optas.TaskModel("control", pm_dim, time_derivs=[0], dlim={0: [-1, 1]}, is_discrete=True)
        pm_control_name = point_mass_control.get_name()

        # Setup optimization builder
        T = 15  # number of time steps
        builder = optas.OptimizationBuilder(T, tasks=[point_mass, point_mass_control], derivs_align=True)

        # Add parameters
        init = builder.add_parameter("init", 2)  # initial point mass position
        goal = builder.add_parameter("goal", 2)  # goal point mass position

        # Constraint: limits
        builder.enforce_model_limits(pm_name, time_deriv=0)
        builder.enforce_model_limits(pm_name, time_deriv=1)
        builder.enforce_model_limits(pm_control_name, time_deriv=0)

        # Constraint: dynamics
        builder.integrate_model_states(pm_name, time_deriv=1, dt=dt)

        for t in range(T-1):
            prv = builder.get_model_state(pm_name, t, time_deriv=1)
            nxt = builder.get_model_state(pm_name, t+1, time_deriv=1)
            ctl = builder.get_model_state(pm_control_name, t, time_deriv=0)
            builder.add_equality_constraint(f"dynamics_ctl_integrated_{t}", nxt, prv + dt * max_control * ctl)

        # Constraint: initial state
        builder.fix_configuration(pm_name, config=init)
        builder.fix_configuration(pm_name, time_deriv=1)
        builder.fix_configuration(pm_control_name)

        # Constraint: final velocity
        dxF = builder.get_model_state(pm_name, -1, time_deriv=1)
        builder.add_equality_constraint("final_velocity", dxF)

        # Constraint: obstacle avoidance
        X = builder.get_model_states(pm_name)
        safe_dist_sq = (obs_rad + pm_radius) ** 2
        for i in range(T):
            dist_sq = optas.sumsqr(obs - X[:, i])
            builder.add_geq_inequality_constraint(
                f"obs_avoid_{i}", dist_sq, safe_dist_sq
            )

        # Cost: final state
        builder.add_equality_constraint("final_state", optas.sumsqr(goal - X[:, -1]))

        # Cost: minimize velocity
        w = 1e-6 / float(T)  # weight on cost term
        dX = builder.get_model_states(pm_name, time_deriv=1)
        builder.add_cost_term("minimize_velocity", w * optas.sumsqr(dX))

        # Create solver
        self.solver = optas.CasADiSolver(builder.build()).setup("bonmin")

        # Save variables
        self.T = T
        self.dt = dt
        self.pm_name = pm_name
        self.pm_control_name = pm_control_name
        self.pm_radius = pm_radius
        self.obs = obs
        self.max_control = max_control
        self.obs_rad = obs_rad
        self.duration = float(T - 1) * dt  # task duration
        self.point_mass = point_mass
        self.point_mass_control = point_mass_control

    def plan(self, init, goal):
        self.solver.reset_parameters({"init": init, "goal": goal})
        solution = self.solver.solve()

        if not self.solver.did_solve():
            raise RuntimeError("solver failed")
        
        plan_x = self.solver.interpolate(solution[f"{self.pm_name}/y"], self.duration)
        plan_dx = self.solver.interpolate(solution[f"{self.pm_name}/dy"], self.duration)
        return plan_x, plan_dx, solution[f"{self.pm_control_name}/y"].toarray()


class Animate:
    def __init__(self, animate):
        # Setup planner
        self.planner = Planner()
        self.init = [-1, -1]
        self.goal = [1, 1]
        self.plan_x, self.plan_dx, self.plan_u = self.planner.plan(self.init, self.goal)

        # Setup figure
        self.t = optas.np.linspace(0, self.planner.duration, self.planner.T)
        self.X = self.plan_x(self.t)
        self.dX = self.plan_dx(self.t)
        self.U = self.plan_u

        self.fig, self.ax = plt.subplot_mosaic(
            [["birdseye", "position"], ["birdseye", "velocity"], ["birdseye", "control"]],
            layout="constrained",
            figsize=(10, 5),
        )

        self.ax["birdseye"].plot(self.X[0, :], self.X[1, :], "-kx", label="plan")
        self.ax["birdseye"].add_patch(
            plt.Circle(
                self.init, radius=self.planner.pm_radius, color="green", alpha=0.5
            )
        )
        self.ax["birdseye"].add_patch(
            plt.Circle(self.goal, radius=self.planner.pm_radius, color="red", alpha=0.5)
        )
        self.ax["birdseye"].add_patch(
            plt.Circle(self.planner.obs, radius=self.planner.obs_rad, color="black")
        )
        self.ax["birdseye"].set_aspect("equal")
        self.ax["birdseye"].set_xlim(*self.planner.point_mass.dlim[0])
        self.ax["birdseye"].set_ylim(*self.planner.point_mass.dlim[0])
        self.ax["birdseye"].set_title("Birdseye View")
        self.ax["birdseye"].set_xlabel("x")
        self.ax["birdseye"].set_ylabel("y")

        self.ax["position"].plot(self.t, self.X[0, :], "-rx", label="plan-x")
        self.ax["position"].plot(self.t, self.X[1, :], "-bx", label="plan-y")
        self.ax["position"].set_ylabel("Position")
        self.ax["position"].set_xlim(0, self.planner.duration)

        axlim = max([abs(l) for l in self.planner.point_mass.dlim[0]])
        self.ax["position"].set_ylim(-axlim, axlim)

        self.ax["velocity"].plot(self.t, self.dX[0, :], "-rx", label="plan-dx")
        self.ax["velocity"].plot(self.t, self.dX[1, :], "-bx", label="plan-dy")
        self.ax["velocity"].axhline(
            self.planner.point_mass.dlim[1][0], color="red", linestyle="--"
        )
        self.ax["velocity"].axhline(
            self.planner.point_mass.dlim[1][1],
            color="red",
            linestyle="--",
            label="limit",
        )
        self.ax["velocity"].set_ylabel("Velocity")
        self.ax["velocity"].set_xlabel("Time")

        self.ax["velocity"].set_xlim(0, self.planner.duration)
        axlim = max([abs(1.5 * l) for l in self.planner.point_mass.dlim[1]])
        self.ax["velocity"].set_ylim(-axlim, axlim)

        self.ax["control"].plot(self.t, self.planner.max_control*self.U[0, :], "-rx", label="U-x")
        self.ax["control"].plot(self.t, self.planner.max_control*self.U[1, :], "-bx", label="U-y")
        self.ax["control"].set_ylabel("Control")
        self.ax["control"].set_xlim(0, self.planner.duration)

        self.ax["control"].axhline(
            self.planner.max_control*self.planner.point_mass_control.dlim[0][0], color="red", linestyle="--"
        )
        self.ax["control"].axhline(
            self.planner.max_control*self.planner.point_mass_control.dlim[0][1],
            color="red",
            linestyle="--",
            label="limit",
        )        

        axlim = 1.05*self.planner.max_control*max([abs(l) for l in self.planner.point_mass.dlim[0]])
        self.ax["control"].set_ylim(-axlim, axlim)
        
        for a in self.ax.values():
            a.legend(ncol=3, loc="lower right")
            a.grid()

        # Animate
        if not animate:
            return
        self.pos_line = self.ax["position"].axvline(color="blue", alpha=0.5)
        self.vel_line = self.ax["velocity"].axvline(color="blue", alpha=0.5)
        self.ctl_line = self.ax["control"].axvline(color="blue", alpha=0.5)
        self.pm_visual = plt.Circle(
            self.init, radius=self.planner.pm_radius, color="blue", alpha=0.5
        )
        self.ani = FuncAnimation(self.fig, self.update, frames=self.t, blit=True)

    def update(self, frame):
        # Udpate position/velocity indicator line
        self.pos_line.set_xdata([frame, frame])
        self.vel_line.set_xdata([frame, frame])
        self.ctl_line.set_xdata([frame, frame])

        # Update point mass
        self.pm_visual.set_center(self.plan_x(frame))
        self.ax["birdseye"].add_patch(self.pm_visual)

        return (self.pm_visual, self.pos_line, self.vel_line, self.ctl_line)

    @staticmethod
    def show():
        plt.show()


def main(show=True):
    from sys import argv

    animate = "--noanimate" not in argv
    anim = Animate(animate)
    if show:
        anim.show()

    return 0


if __name__ == "__main__":
    main()

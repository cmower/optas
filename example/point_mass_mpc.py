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
        pm_dim = 2 # x, y dimensions
        dlim = {0: [-1.5, 1.5], 1: [-1, 1]}  # pos/vel limits
        point_mass = optas.TaskModel('point_mass', pm_dim, time_derivs=[0, 1], dlim=dlim)
        pm_name = point_mass.get_name()

        # Setup optimization builder
        T = 45 # number of time steps
        builder = optas.OptimizationBuilder(T, tasks=point_mass, derivs_align=True)

        # Add parameters
        init = builder.add_parameter('init', 2)  # initial point mass position
        goal = builder.add_parameter('goal', 2)  # goal point mass position

        # Constraint: limits
        builder.enforce_model_limits(pm_name, time_deriv=0)
        builder.enforce_model_limits(pm_name, time_deriv=1)

        # Constraint: dynamics
        builder.integrate_model_states(pm_name, time_deriv=1, dt=dt)

        # Constraint: initial state
        builder.initial_configuration(pm_name, init=init)
        builder.initial_configuration(pm_name, time_deriv=1)

        # Constraint: final velocity
        dxF = builder.get_model_state(pm_name, -1, time_deriv=1)
        builder.add_equality_constraint('final_velocity', dxF)

        # Constraint: obstacle avoidance
        X = builder.get_model_states(pm_name)
        safe_dist_sq = (obs_rad + pm_radius)**2
        for i in range(T):
            dist_sq = optas.sumsqr(obs - X[:, i])
            builder.add_geq_inequality_constraint(f'obs_avoid_{i}', dist_sq, safe_dist_sq)

        # Cost: final state
        builder.add_cost_term('final_state', optas.sumsqr(goal - X[:, -1]))

        # Cost: minimize velocity
        w = 0.01/float(T)  # weight on cost term
        dX = builder.get_model_states(pm_name, time_deriv=1)
        builder.add_cost_term('minimize_velocity', w*optas.sumsqr(dX))

        # Cost: minimize acceleration
        w = 0.005/float(T)  # weight on cost term
        ddX = (dX[:, 1:] - dX[:, :-1])/dt
        builder.add_cost_term('minimize_acceleration', w*optas.sumsqr(ddX))

        # Create solver
        self.solver = optas.CasADiSolver(builder.build()).setup('ipopt')

        # Save variables
        self.T = T
        self.dt = dt
        self.pm_name = pm_name
        self.pm_radius = pm_radius
        self.obs = obs
        self.obs_rad = obs_rad
        self.duration = float(T-1)*dt  # task duration
        self.point_mass = point_mass

    def plan(self, init, goal):
        self.solver.reset_parameters({'init': init, 'goal': goal})
        solution = self.solver.solve()
        plan_x = self.solver.interpolate(solution[f'{self.pm_name}/x'], self.duration)
        plan_dx = self.solver.interpolate(solution[f'{self.pm_name}/dx'], self.duration)
        return plan_x, plan_dx

class Controller:

    def __init__(self):

        # Planner attributes
        dt = 0.05  # time step
        obs_rad = 0.2  # obstacle radii

        # Setup point mass model
        pm_radius = 0.1  # point mass radii
        pm_dim = 2 # x, y dimensions
        dlim = {0: [-1.5, 1.5], 1: [-1, 1]}  # pos/vel limits
        point_mass = optas.TaskModel('point_mass', pm_dim, time_derivs=[0, 1], dlim=dlim)
        pm_name = point_mass.get_name()

        # Setup optimization builder
        T = 20 # number of time steps
        builder = optas.OptimizationBuilder(T, tasks=point_mass, derivs_align=True)

        # Add parameters
        curr = builder.add_parameter('curr', 2)  # current point mass position
        dcurr = builder.add_parameter('dcurr', 2)  # current point mass velocity
        goal = builder.add_parameter('goal', 2, T)  # goal point mass positions
        obs = builder.add_parameter('obs', 2, T)  # obstacle position model

        # Constraint: limits
        builder.enforce_model_limits(pm_name, time_deriv=0)
        builder.enforce_model_limits(pm_name, time_deriv=1)

        # Constraint: dynamics
        builder.integrate_model_states(pm_name, time_deriv=1, dt=dt)

        # Constraint: initial state
        builder.initial_configuration(pm_name, init=curr)
        builder.initial_configuration(pm_name, init=dcurr, time_deriv=1)

        # Constraint: obstacle avoidance
        X = builder.get_model_states(pm_name)
        safe_dist_sq = (obs_rad + pm_radius)**2
        for i in range(T):
            dist_sq = optas.sumsqr(obs[:, i] - X[:, i])
            builder.add_geq_inequality_constraint(f'obs_avoid_{i}', dist_sq, safe_dist_sq)

        # Cost: optimal path
        builder.add_cost_term('optimal_path', optas.sumsqr(goal - X))

        # Cost: minimize acceleration
        dX = builder.get_model_states(pm_name, time_deriv=1)
        w = 0.0025/float(T)  # weight on cost term
        ddX = (dX[:, 1:] - dX[:, :-1])/dt
        builder.add_cost_term('minimize_acceleration', w*optas.sumsqr(ddX))

        # Create solver
        self.solver = optas.CasADiSolver(builder.build()).setup('ipopt')

        # Save variables
        self.T = T
        self.dt = dt
        self.pm_name = pm_name
        self.pm_radius = pm_radius
        self.obs = obs
        self.obs_rad = obs_rad
        self.duration = float(T-1)*dt  # task duration
        self.point_mass = point_mass
        self.solution = None

    def next_state(self, curr, dcurr, goal, obs):
        if self.solution is not None:
            self.solver.reset_initial_seed(self.solution)
        params = {'curr': curr, 'dcurr': dcurr, 'goal': goal, 'obs': obs}
        self.solver.reset_parameters(params)
        self.solution = self.solver.solve()
        if not self.solver.did_solve():
            for vc_collection in self.solver.violated_constraints(self.solution, params):
                for vc in vc_collection:
                    print(vc)
            raise RuntimeError("solver failed")
        plan_x = self.solver.interpolate(self.solution[f'{self.pm_name}/x'], self.duration)
        plan_dx = self.solver.interpolate(self.solution[f'{self.pm_name}/dx'], self.duration)
        return plan_x(2*self.dt), plan_dx(2*self.dt), plan_x, plan_dx

class Animate:

    def __init__(self, animate):

        # Setup planner
        self.planner = Planner()
        self.init = [-1, -1]
        self.goal = [1, 1]
        self.plan_x, self.plan_dx = self.planner.plan(self.init, self.goal)

        # Setup current state and controller
        self.curr = self.init
        self.dcurr = [0., 0.]
        self.controller = Controller()

        # Setup figure
        self.t = optas.np.linspace(0, self.planner.duration, self.planner.T)
        self.X = self.plan_x(self.t)
        self.dX = self.plan_dx(self.t)

        self.fig, self.ax = plt.subplot_mosaic([['birdseye', 'position'],
                                                ['birdseye', 'velocity']],
                                               layout='constrained',
                                               figsize=(10, 5),
        )

        self.mpc_line, = self.ax['birdseye'].plot([], [], '-x', color='yellow', label='mpc')
        self.ax['birdseye'].plot(self.X[0, :], self.X[1, :], '-kx', label='plan')
        self.ax['birdseye'].add_patch(plt.Circle(self.init, radius=self.planner.pm_radius, color='green', alpha=0.5))
        self.ax['birdseye'].add_patch(plt.Circle(self.goal, radius=self.planner.pm_radius, color='red', alpha=0.5))
        self.dt = self.planner.dt
        self.obs_pos = optas.np.array(self.planner.obs)
        self.obs_visual = plt.Circle(self.obs_pos, radius=self.planner.obs_rad, color='black')
        self.ax['birdseye'].add_patch(self.obs_visual)
        self.ax['birdseye'].set_aspect('equal')
        self.ax['birdseye'].set_xlim(*self.planner.point_mass.dlim[0])
        self.ax['birdseye'].set_ylim(*self.planner.point_mass.dlim[0])
        self.ax['birdseye'].set_title('Birdseye View')
        self.ax['birdseye'].set_xlabel('x')
        self.ax['birdseye'].set_ylabel('y')

        self.ax['position'].plot(self.t, self.X[0,:], '-rx', label='plan-x')
        self.ax['position'].plot(self.t, self.X[1,:], '-bx', label='plan-y')
        self.pm_pos_curr_x, = self.ax['position'].plot([], [], 'or', label='curr-x')
        self.pm_pos_curr_y, = self.ax['position'].plot([], [], 'ob', label='curr-y')
        self.ax['position'].set_ylabel('Position')
        self.ax['position'].set_xlim(0, self.planner.duration)

        axlim = max([abs(l) for l in self.planner.point_mass.dlim[0]])
        self.ax['position'].set_ylim(-axlim, axlim)

        self.ax['velocity'].plot(self.t, self.dX[0,:], '-rx', label='plan-dx')
        self.ax['velocity'].plot(self.t, self.dX[1,:], '-bx', label='plan-dy')
        self.pm_vel_curr_x, = self.ax['velocity'].plot([], [], 'or', label='curr-dx')
        self.pm_vel_curr_y, = self.ax['velocity'].plot([], [], 'ob', label='curr-dy')
        self.ax['velocity'].axhline(self.planner.point_mass.dlim[1][0], color='red', linestyle='--')
        self.ax['velocity'].axhline(self.planner.point_mass.dlim[1][1], color='red', linestyle='--', label='limit')
        self.ax['velocity'].set_ylabel('Velocity')
        self.ax['velocity'].set_xlabel('Time')

        self.ax['velocity'].set_xlim(0, self.planner.duration)
        axlim = max([abs(1.5*l) for l in self.planner.point_mass.dlim[1]])
        self.ax['velocity'].set_ylim(-axlim, axlim)

        for a in self.ax.values():
            a.legend(ncol=3, loc='lower right')
            a.grid()

        # Animate
        if not animate: return
        self.pos_line = self.ax['position'].axvline(color='blue', alpha=0.5)
        self.vel_line = self.ax['velocity'].axvline(color='blue', alpha=0.5)
        self.pm_visual = plt.Circle(self.init, radius=self.planner.pm_radius, color='blue', alpha=0.5)
        self.frames = list(range(self.planner.T))
        self.ani = FuncAnimation(self.fig, self.update, frames=self.frames, blit=True)

    def update(self, frame):

        if frame == 0:
            self.curr = self.init
            self.dcurr = [0., 0.]

        t = self.t[frame]

        # Udpate position/velocity indicator line
        self.pos_line.set_xdata([t, t])
        self.vel_line.set_xdata([t, t])
        self.pm_pos_curr_x.set_xdata([t])
        self.pm_pos_curr_x.set_ydata([self.curr[0]])
        self.pm_pos_curr_y.set_xdata([t])
        self.pm_pos_curr_y.set_ydata([self.curr[1]])

        self.pm_vel_curr_x.set_xdata([t])
        self.pm_vel_curr_x.set_ydata([self.dcurr[0]])
        self.pm_vel_curr_y.set_xdata([t])
        self.pm_vel_curr_y.set_ydata([self.dcurr[1]])

        # Update point mass and obstacle
        alpha = t*optas.np.pi - optas.np.pi
        temp = 0.15
        ox = temp*optas.np.sin(alpha)
        oy = temp*optas.np.cos(alpha) + temp
        obs = (ox, oy)
        self.obs_visual.set_center(obs)

        obs = []
        for i in range(self.controller.T):
            ti = t + self.controller.dt*i
            alpha = ti*optas.np.pi - optas.np.pi
            ox = temp*optas.np.sin(alpha)
            oy = temp*optas.np.cos(alpha) + temp
            obs.append([ox, oy])

        obs = optas.np.array(obs).T

        goal = []
        for i in range(self.controller.T):
            ti = t + self.controller.dt*i
            try:
                g = self.plan_x(ti).flatten()
                goal.append(g.tolist())
            except ValueError:
                goal.append(g.tolist()) # i.e. previous goal

        goal = optas.np.array(goal).T

        self.curr, self.dcurr, plan_x, plan_dx = self.controller.next_state(self.curr, self.dcurr, goal, obs)

        mpc_plan = []
        for i in range(self.controller.T):
            xmpc = plan_x(i*self.controller.dt)
            mpc_plan.append(xmpc.flatten().tolist())
        mpc_plan = optas.np.array(mpc_plan).T

        self.mpc_line.set_xdata(mpc_plan[0,:])
        self.mpc_line.set_ydata(mpc_plan[1,:])

        self.pm_visual.set_center(mpc_plan[:, 0])
        self.ax['birdseye'].add_patch(self.pm_visual)

        return (self.pm_visual, self.pos_line, self.vel_line, self.obs_visual, self.mpc_line, self.pm_pos_curr_x, self.pm_pos_curr_y, self.pm_vel_curr_x, self.pm_vel_curr_y)

    @staticmethod
    def show():
        plt.show()

def main():
    from sys import argv
    animate = '--noanimate' not in argv
    Animate(animate).show()

if __name__ == '__main__':
    main()

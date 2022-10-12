# Python standard lib
import os
import sys
import abc
import math
import pathlib

# Plotting
import matplotlib.pyplot as plt

# Scipy
from scipy.integrate import dblquad

# PyBullet
import pybullet_api

# OpTaS
import optas
optas.np.set_printoptions(precision=3, suppress=True)
from optas.spatialmath import *

pi = math.pi

def debug(val, pause=False, kill=False):
    print("---------------------------")
    print(val)
    print("---------------------------")

    if pause:
        input(">>>CLICK ANY KEY TO CONTINUE<<<")
    if kill:
        sys.exit(0)

class TOMPCC(abc.ABC):

    nX = 4  # state dimension
    nU = 4  # control dimension
    g = 9.81  # gravitational acceleration
    m = 0.5 # mass of box
    mu = 0.5  # cooeficient of friction between slider and surface

    def __init__(self, T, time_horizon, Lx, Ly, we):

        # Setup constants
        self.T = T  # number of steps
        self.time_horizon = time_horizon  # duration of the trajectory (secs)
        self.dt = time_horizon/float(T-1) # time step duration
        self.Lx = Lx  # length of box along x-axis of slider frame
        self.Ly = Ly  # length of box along y-axis of slider frame
        self.A = Lx*Ly  # area of slider
        self.fmax = self.mu*self.m*self.g # semi-principle axes for ellipsoid (xy)
        self.mmax = (self.mu*self.m*self.g/self.A)*dblquad( # semi-principle axes for ellipsoid (theta)
            lambda y, x: optas.np.sqrt(x**2+y**2),
            -0.5*Lx, 0.5*Lx, -0.5*Ly, 0.5*Ly,
        )[0]
        self.L = optas.diag([  # limit surface model
            1./self.fmax**2, 1./self.fmax**2, 1./self.mmax**2
        ])
        self.we = we # slack term weights

        # Setup task models
        self.state = optas.TaskModel('state', dim=self.nX, time_derivs=[0])
        self.control = optas.TaskModel('control', dim=self.nU, time_derivs=[0], T=T-1, symbol='u')

        # Create builder
        self.builder = optas.OptimizationBuilder(T, tasks=[self.state, self.control])

        # Setup slack variables
        self.eps = self.builder.add_decision_variables('eps', T-1)

        # Extract X, U trajectories
        self.X = self.builder.get_model_states('state')
        self.U = self.builder.get_model_states('control')

        # Constraint: integrate dynamics
        integrate = self.integrate_dynamics_function()
        self.builder.add_equality_constraint(
            'dynamics', integrate(self.X[:, :-1], self.U, self.X[:, 1:]),
        )

        # Constraint: complementary constraints
        for i in range(self.T-1):
            self.cc(i)

        # Constraint: bound phi
        _, _, _, Phi = optas.vertsplit(self.X)
        phi_max = optas.atan2(0.5*Ly, 0.5*Lx)
        self.builder.add_bound_inequality_constraint('bound_phi', -phi_max, Phi, phi_max)

        # Constraint: bound dphi
        _, _, dPhi_plus, dPhi_minus = optas.vertsplit(self.U)
        dPhi = dPhi_plus - dPhi_minus
        dphi_max = optas.deg2rad(30)
        self.builder.add_bound_inequality_constraint('bound_dphi', -dphi_max, dPhi, dphi_max)

        # Cost: minimize slack
        self.builder.add_cost_term('minimize_slack', we*optas.sumsqr(self.eps))

    @abc.abstractmethod
    def setup(self):
        pass  # ensure return self

    def phi2xy(self, phi):
        xc = 0.5*self.Lx
        yc = xc*optas.tan(phi)
        return optas.vertcat(xc, yc)

    def f(self, x, u):
        _, _, theta, phi = optas.vertsplit(x)
        R = rotz(theta)
        xc, yc = optas.vertsplit(self.phi2xy(phi))
        J = optas.horzcat(optas.DM.eye(2), optas.vertcat(-yc, xc))
        K = optas.vertcat(
            optas.horzcat(R @ self.L @ J.T, optas.DM.zeros(3, 2)),
            optas.DM([[0, 0, 1, -1]]),
        )
        return K @ u

    def integrate_dynamics_function(self):
        xcurr = optas.SX.sym('xcurr', self.nX)
        xnext = optas.SX.sym('xnext', self.nX)
        ucurr = optas.SX.sym('ucurr', self.nU)
        fun = optas.Function(
            'integrate_dynamics',
            [xcurr, ucurr, xnext],
            [xcurr + self.dt*self.f(xcurr, ucurr) - xnext]
        )
        return fun.map(self.T-1)

    def cc(self, i):

        u = self.U[:, i]

        fn, ft, dphi_plus, dphi_minus = optas.vertsplit(u)

        lambda_minus = self.mu*fn - ft
        lambda_plus = self.mu*fn + ft

        self.builder.add_geq_inequality_constraint(f'positive_fn_{i}', -fn)
        self.builder.add_geq_inequality_constraint(f'positive_dphi_plus_{i}', -dphi_plus)
        self.builder.add_geq_inequality_constraint(f'positive_dphi_minus_{i}', -dphi_minus)
        self.builder.add_geq_inequality_constraint(f'positive_lambda_minus_{i}', -lambda_minus)
        self.builder.add_geq_inequality_constraint(f'positive_lambda_plus_{i}', -lambda_plus)

        lambda_v = optas.vertcat(lambda_minus, lambda_plus)
        dphi_v = optas.vertcat(dphi_plus, dphi_minus)
        e = self.eps[i]

        self.builder.add_equality_constraint(f'cc_{i}', lambda_v.T@dphi_v + e)

class TOMPCCPlanner(TOMPCC):

    def setup(self, x0, xF, wx, wu):

        x0 = optas.vec(x0)
        xF = optas.vec(xF)

        # Constraint: Initial state
        self.builder.add_equality_constraint('initial_state', self.X[:, 0], x0)

        # Cost: final state
        xfinal = self.X[:, -1]
        Wx = optas.diag(wx)
        xbar = xfinal - xF
        self.builder.add_cost_term('final_state', xbar.T @ Wx @ xbar)

        # Cost: minimize controls
        Wu = optas.diag(wu)
        for i in range(self.T-1):
            u = self.U[:, i]
            self.builder.add_cost_term(f'minimize_control_{i}', u.T @ Wu @ u)

        # Setup solver
        opt = self.builder.build()
        self.solver = optas.CasADiSolver(self.builder.build()).setup('ipopt')

        # Set initial seed
        X0 = optas.DM.zeros(self.nX, self.T)
        for i in range(self.T):
            alpha = float(i)/float(self.T-1)
            X0[:, i] = (1.-alpha)*x0 + alpha*xF
        U0 = 0.01*optas.DM.zeros(self.nU, self.T-1)
        self.solver.reset_initial_seed({'state/x': X0, 'control/u': U0})

        # Save for later
        self.x0 = x0
        self.xF = xF

        return self

    def plan(self):

        # Compute plan
        solution = self.solver.solve()
        if not self.solver.did_solve():
            print("[ERROR] planning solver did not converge!")
            sys.exit(0)
        X = self.solver.interpolate(solution['state/x'], self.time_horizon, fill_value='extrapolate')
        U = self.solver.interpolate(solution['control/u'], self.time_horizon-self.dt, fill_value='extrapolate')

        # Plot plan
        plot_plan = False
        if plot_plan:
            fig, ax = plt.subplots(2, 1, sharex=True)
            T_ = optas.np.linspace(0, self.time_horizon, 50)
            X_ = X(T_)
            U_ = U(T_)
            for i in range(self.nX):
                ax[0].plot(T_, X_[i, :], label=str(i))


                x0 = self.x0[i].toarray()[0, 0]
                ax[0].plot([0.], [x0], 'o', label=str(i)+'_init')

                xF = self.xF[i].toarray()[0, 0]
                ax[0].plot([self.time_horizon], [xF], 'o', label=str(i)+'_goal')

            for i in range(self.nU):
                ax[1].plot(T_, U_[i, :], label=str(i))

            for a in ax.flatten():
                a.legend(ncol=4)
                a.grid()

            plt.show()

        class Plan:

            def __init__(self, plan, time_horizon):
                self.plan = plan
                self.time_horizon = time_horizon
                self.final_state = plan(time_horizon)

            def __call__(self, t):

                if t < self.time_horizon:
                    return self.plan(t)
                else:
                    return self.final_state


        return Plan(X, self.time_horizon), Plan(U, self.time_horizon)

class TOMPCCController(TOMPCC):

    def setup(self, planx, planu, wx, wu):

        # Setup
        self.planx = planx
        self.planu = planu

        # Add parameters
        Xnom = self.builder.add_parameter('Xnom', self.nX, self.T)  # nominal trajectory
        xc = self.builder.add_parameter('xc', self.nX) # current state

        # Constraint: initial state
        self.builder.add_equality_constraint('initial_state', self.X[:, 0], xc)

        # Cost: match nominal plan
        Wx = optas.diag(wx)
        Xbar = self.X - Xnom
        for i in range(self.T):
            xbar = Xbar[:, i]
            w = 1.  # 10.*float(self.T-i)
            self.builder.add_cost_term(f'match_nominal_plan_{i}', w*xbar.T@Wx@xbar)

        # Cost: minimize controls
        Wu = optas.diag(wu)
        for i in range(self.T-1):
            u = self.U[:, i]
            self.builder.add_cost_term(f'minimize_control_{i}', u.T @ Wu @ u)

        # Setup solver
        opt = self.builder.build()
        self.solution = None
        self.solver = optas.CasADiSolver(self.builder.build()).setup('ipopt')
        # self.solver = optas.ScipyMinimizeSolver(self.builder.build()).setup('SLSQP')

        return self

    def compute_next_state(self, t, xc, dt):

        # Setup solver
        xc = optas.DM(xc)
        tt = optas.np.linspace(t, t + self.time_horizon, self.T)
        Xnom = optas.DM.zeros(self.nX, self.T)
        for i in range(self.T):
            Xnom[:, i] = optas.DM(self.planx(tt[i]))

        # debug(Xnom.toarray(), pause=True)

        if self.solution is not None:
            self.solver.reset_initial_seed(self.solution) # reset with previous solution
        else:
            X0 = Xnom
            U0 = optas.DM.zeros(self.nU, self.T-1)
            for i in range(self.T-1):
                U0[:,i] = self.planu(tt[i])
            self.solver.reset_initial_seed({'state/x': X0, 'control/u': U0})

        self.solver.reset_parameters({'Xnom': Xnom, 'xc': xc})

        # Solve problem
        self.solution = self.solver.solve()
        if not self.solver.did_solve():
            print("[ERROR] MPC controller did not solve")
            sys.exit(0)

        # Interpolate solution
        mpc_x = self.solver.interpolate(self.solution['state/x'], self.time_horizon)
        mpc_u = self.solver.interpolate(self.solution['control/u'], self.time_horizon-self.dt, fill_value='extrapolate')
        xtarget = mpc_x(dt)

        # Plot mpc
        plot_mpc = True
        if plot_mpc:
            fig, ax = plt.subplots(2, 1, sharex=True)
            T_ = optas.np.linspace(0, self.time_horizon, 50)
            X_ = mpc_x(T_)
            U_ = mpc_u(T_)
            for i in range(self.nX):
                ax[0].plot(t+T_, X_[i, :], label=str(i))
                ax[0].plot(tt, Xnom[i,:].toarray().flatten(), label=str(i)+'_nom')


                x0 = xc[i].toarray()[0, 0]
                ax[0].plot([t], [x0], 'o', label=str(i)+'_c')

                err = optas.np.sqrt((self.solution['state/x'][i,:] - Xnom[i,:]).toarray().flatten()**2)
                ax[1].plot(tt, err, label=str(i))

            for a in ax.flatten():
                a.legend(ncol=4)
                a.grid()

            plt.show()

        return xtarget

class IK:

    def __init__(self, dt):

        # Setup
        T = 1 # no. time steps in trajectory
        link_ee = 'end_effector_ball'  # end-effector link name

        # Setup robot
        cwd = pathlib.Path(__file__).parent.resolve() # path to current working directory
        urdf_filename = os.path.join(cwd, 'robots', 'kuka_lwr.urdf')
        kuka = optas.RobotModel(
            urdf_filename=urdf_filename,
            time_derivs=[1],  # i.e. joint velocity
        )

        # Setup optimization builder
        builder = optas.OptimizationBuilder(T=T, robots=[kuka], derivs_align=True)

        # Get joint velocity
        dq = builder.get_model_state('kuka', t=0, time_deriv=1)

        # Setup parameters
        qc = builder.add_parameter('qc', kuka.ndof)  # current robot joint configuration
        pg = builder.add_parameter('pg', 3)  # goal end-effector position

        # Get Jacobian
        J = kuka.get_global_geometric_jacobian(link_ee, qc)
        Jp = J[:3, :]
        Ja = J[3:, :]

        # Constraint: no motion in end joint
        builder.add_equality_constraint('no_eff_motion', dq[-1])

        # Compute current end-effector position
        pc = kuka.get_global_link_position(link_ee, qc)

        # Cost: minimize joint velocity
        Wdq = 0.01*optas.DM.eye(7)
        builder.add_cost_term('minimize_joint_velocity', dq.T@Wdq@dq)

        # Cost: goal position
        p = pc + dt*Jp@dq
        pdiff = p - pg
        Wp = optas.diag([100000., 100000., 50000.])
        builder.add_cost_term('goal_position', pdiff.T@Wp@pdiff)

        # Constraint: bound angular motion
        da = Ja@dq
        alpha = 0.01
        da_lo = optas.DM([-alpha, -alpha, -100])
        da_up = optas.DM([alpha, alpha, 100])
        builder.add_bound_inequality_constraint('bound_angular_motion', da_lo, da, da_up)

        # Joint limits
        q = qc + dt*dq
        builder.add_bound_inequality_constraint('joint_position_limits', kuka.lower_actuated_joint_limits, q, kuka.upper_actuated_joint_limits)

        # Constraint: bound joint velocity
        dq_max = optas.deg2rad(30.)*optas.DM.ones(7)
        builder.add_bound_inequality_constraint('bound_joint_velocity', -dq_max, dq, dq_max)

        # Setup solver
        self.solution = None
        self.solver = optas.OSQPSolver(builder.build()).setup(settings={'verbose': False})

        # Save for later
        self.dt = dt
        self._eff_pos = kuka.get_global_link_position_function(link_ee)
        self.eff_pos = lambda q: self._eff_pos(q).toarray().flatten()

    def next_joint_state(self, qc, pg):

        # Setup
        qc = optas.DM(qc)
        pg = optas.DM(pg)

        # Setup solver
        if self.solution is not None:
            self.solver.reset_initial_seed(self.solution)  # reset with previous solution
        else:
            self.solver.reset_initial_seed({'kuka/dq': optas.DM.zeros(7)})

        self.solver.reset_parameters({'qc': qc, 'pg': pg})

        # Solve IK
        self.solution = self.solver.solve()
        if not self.solver.did_solve():
            print("[ERROR] IK did not solve:", self.solver.stats().info.status)
            sys.exit(0)

        # Compute next joint state
        dq = self.solution['kuka/dq']
        qn = qc + self.dt*dq

        return qn.toarray().flatten()

def get_state(box, kuka, Lx, x_plan, pg):

    # Extract box pose and end-effector position
    boxpos, boxori = box.get_pose()
    effpos = kuka.eff_pos(kuka.q())

    # Compute orientation of contact point
    R = rot2(boxori[2])
    e = optas.DM(pg[:2])
    b = optas.DM(boxpos[:2])
    c = (R.T@(e - b)).toarray().flatten().tolist()
    phi = math.atan2(c[1], c[0])

    # Build xc
    xc = [
        boxpos[0],
        boxpos[1],
        boxori[2],
        phi,
    ]

    return xc


def main():

    # Constants
    use_mpc = True  # True: robot uses MPC, False: robot follows plan
    eff_ball_radius = 0.015
    box_position0 = [0.4, 0.1, 0.05]   # initial box position
    box_positionF = [0.65, 0.4, 0.05]  # goal box position
    box_theta0 = optas.deg2rad(-90)
    box_thetaF = optas.deg2rad(-180)
    Lx, Ly = 0.1, 0.2
    q0 = optas.deg2rad([0, -30, 0, 90, 0, -60, 0])
    q = q0

    # Setup PyBullet
    pb_dt = 0.02
    pb = pybullet_api.PyBullet(
        pb_dt,
        camera_distance=0.6,
        camera_yaw=45,
        camera_pitch=-50,
        camera_target_position=[0.5, 0.2, 0.05],
        real_time=False,

    )
    kuka = pybullet_api.Kuka()
    kuka.reset(q0)
    box = pybullet_api.DynamicBox(
        box_position0,
        [0.5*Lx, 0.5*Ly, 0.05],
        base_orientation=[0, 0, box_theta0],
        base_mass=TOMPCC.m,
    )
    visual_box = pybullet_api.VisualBox(
        box_position0,
        [0.5*Lx, 0.5*Ly, 0.05],
        base_orientation=[0, 0, box_theta0],
        rgba_color=[0., 0., 1., 0.5],
    )
    pybullet_api.VisualBox(
        box_positionF,
        [0.5*Lx, 0.5*Ly, 0.05],
        base_orientation=[0, 0, box_thetaF],
        rgba_color=[1., 0., 0., 0.5],
    )

    # Setup IK
    ik = IK(pb_dt)

    # Setup planner
    T_plan = 40
    time_horizon_plan = 10.
    we_plan = 50.
    x0 = optas.vertcat(box_position0[:2], box_theta0, 0.)
    xF = optas.vertcat(box_positionF[:2], box_thetaF, 0.)
    planner = TOMPCCPlanner(T_plan, time_horizon_plan, Lx, Ly, we_plan).setup(x0, xF, wx=[100, 100, 1, 0.01], wu=[0.01, 0.01, 0.0, 0.0])

    # Plan trajectory
    planx, planu = planner.plan()

    # Setup controller
    T_mpc = 10
    time_horizon_mpc = 4.
    we_mpc = 50.
    controller = TOMPCCController(T_mpc, time_horizon_mpc, Lx, Ly, we=we_mpc).setup(planx, planu, wx=[100., 100., 100., 0.], wu=[0.0, 0.0, 0., 0.])

    # Compute initial goal position for kuka
    pg0 = optas.DM(box_position0) + rotz(box_theta0)@optas.vec([0.5*Lx+eff_ball_radius, 0, -0.03])
    error_tol = 1e-7

    # Start pybullet
    pb.start()

    # Move robot to initial configuration
    done = False
    while not done:
        q = ik.next_joint_state(q, pg0)
        kuka.cmd(q)
        error = optas.np.linalg.norm(ik.eff_pos(q) - pg0)
        if error < error_tol:
            done = True
        pb.step()

    # Start pushing
    pg = pg0
    delay = 10.
    t = 0.
    done = False
    while not done:

        # Update plan
        x_plan = planx(t)

        # Get position goal for robot
        if use_mpc:

            # Get current state
            xc = get_state(box, kuka, Lx, x_plan, pg)

            # Get target state
            xtarget = controller.compute_next_state(t, xc, pb_dt)

            # Get position goal from MPC
            pg = optas.vertcat(xtarget[:2], 0.05) + rotz(xtarget[2])@(optas.vertcat(planner.phi2xy(xtarget[3]), 0.) + optas.DM([eff_ball_radius, 0., -0.03]))

        else:

            # Get position goal from plan
            pg = optas.vertcat(x_plan[:2], 0.05) + rotz(x_plan[2])@(optas.vertcat(planner.phi2xy(x_plan[3]), 0.) + optas.DM([eff_ball_radius, 0., -0.03]))

        # Reset visual box/sphere
        b = optas.DM(x_plan[:2].tolist() + [0.05])
        visual_box.reset(
            base_position=b.toarray().flatten().tolist(),
            base_orientation=[0., 0., x_plan[2]],
        )
        # visual_sphere.reset(pg)

        # Update kuka
        q = ik.next_joint_state(q, pg)
        kuka.cmd(q)

        # Step PyBullet and update time
        pb.step(sleep=not use_mpc)
        t += pb_dt

        # Check if need to close
        if (not use_mpc) and t > time_horizon_plan + delay:
            done = True

    pb.stop()
    pb.close()

if __name__ == '__main__':
    main()

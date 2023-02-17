# Python standard lib
import os
import sys
import pathlib

# PyBullet
import pybullet_api

# OpTaS
import optas
from optas.templates import Manager

cwd = pathlib.Path(__file__).parent.resolve() # path to current working directory

class Planner(Manager):


    def setup_solver(self):

        # Setup robot  ========================

        # Kuka LWR
        # link_ee = 'end_effector_ball'  # end-effector link name
        # filename = os.path.join(cwd, 'robots', 'kuka_lwr', 'kuka_lwr.urdf')
        # use_xacro = False

        # Kuka LBR
        link_ee = 'lbr_link_ee'
        filename = os.path.join(cwd, 'robots', 'kuka_lbr', 'med7.urdf.xacro')
        use_xacro = True

        # =====================================


        # Setup
        pi = optas.np.pi  # 3.141...
        self.T = 50 # no. time steps in trajectory
        self.Tmax = 10.  # trajectory of 5 secs
        t = optas.linspace(0, self.Tmax, self.T)
        self.dt = float((t[1] - t[0]).toarray()[0, 0])  # time step

        # Setup robot
        robot_model_input = {}
        robot_model_input['time_derivs'] = [0, 1] # i.e. joint position/velocity trajectory
        if use_xacro:
            robot_model_input['xacro_filename'] = filename
        else:
            robot_model_input['urdf_filename'] = filename

        self.kuka = optas.RobotModel(**robot_model_input)
        self.kuka_name = self.kuka.get_name()
        print("Using robot named", self.kuka_name)

        # Setup optimization builder
        builder = optas.OptimizationBuilder(T=self.T, robots=[self.kuka])

        # Setup parameters
        qc = builder.add_parameter('qc', self.kuka.ndof)  # current robot joint configuration

        # Constraint: initial configuration
        builder.fix_configuration(self.kuka_name, config=qc)
        builder.fix_configuration(self.kuka_name, time_deriv=1) # initial joint vel is zero

        # Constraint: dynamics
        builder.integrate_model_states(
            self.kuka_name,
            time_deriv=1, # i.e. integrate velocities to positions
            dt=self.dt,
        )

        # Get joint trajectory
        Q = builder.get_model_states(self.kuka_name)  # ndof-by-T symbolic array for robot trajectory

        # End effector position trajectory
        pos = self.kuka.get_global_link_position_function(link_ee, n=self.T)
        pos_ee = pos(Q) # 3-by-T position trajectory for end-effector (FK)

        # Get current position of end-effector
        pc = self.kuka.get_global_link_position(link_ee, qc)
        Rc = self.kuka.get_global_link_rotation(link_ee, qc)
        quatc = self.kuka.get_global_link_quaternion(link_ee, qc)

        # Generate figure-of-eight path for end-effector in end-effector frame
        path = optas.SX.zeros(3, self.T)
        path[0, :] = 0.2*optas.sin(t*pi*0.5).T  # need .T since t is col vec
        path[1, :] = 0.1*optas.sin(t*pi).T  # need .T since t is col vec

        # Put path in global frame
        for k in range(self.T):
            path[:,k] = pc + Rc @ path[:,k]

        # Cost: figure eight
        builder.add_cost_term('ee_path', 1000.*optas.sumsqr(path - pos_ee))

        # Cost: minimize joint velocity
        dQ = builder.get_model_states(self.kuka_name, time_deriv=1)
        builder.add_cost_term('min_join_vel', 0.01*optas.sumsqr(dQ))

        # Prevent rotation in end-effector
        quat = self.kuka.get_global_link_quaternion_function(link_ee, n=self.T)
        builder.add_equality_constraint('no_eff_rot', quat(Q), quatc)

        # Setup solver
        optimization = builder.build()
        solver = optas.CasADiSolver(optimization).setup('ipopt')

        return solver

    def is_ready(self):
        return True

    def reset(self, qc):
        # Set parameters
        self.solver.reset_parameters({'qc': optas.DM(qc)})

        # Set initial seed, note joint velocity will be set to zero
        Q0 = optas.diag(qc) @ optas.DM.ones(self.kuka.ndof, self.T)
        self.solver.reset_initial_seed({f'{self.kuka_name}/q/x': Q0})

    def get_target(self):
        return self.solution

    def plan(self):

        # Solve problem
        self.solve()

        solution = self.get_target()

        # Interpolate
        plan = self.solver.interpolate(solution[f'{self.kuka_name}/q'], self.Tmax)

        class Plan:

            def __init__(self, robot, plan_function):
                self.robot = robot
                self.plan_function = plan_function

            def __call__(self, t):
                q = self.plan_function(t)
                return q

        return Plan(self.kuka, plan)

def main():

    # Initialize planner
    planner = Planner()

    # Plan trajectory
    qc = optas.np.deg2rad([0, 30, 0, -90, 0, -30, 0])
    planner.reset(qc)
    plan = planner.plan()

    # Setup PyBullet
    hz = 50
    dt = 1.0/float(hz)
    pb = pybullet_api.PyBullet(dt)
    if planner.kuka_name == 'med7':
        kuka = pybullet_api.KukaLBR()
    else:
        kuka = pybullet_api.KukaLWR()
    kuka.reset(plan(0.))
    pb.start()

    start_time = pybullet_api.time.time()

    # Main loop
    while True:
        t = pybullet_api.time.time() - start_time
        if t > planner.Tmax:
            break
        kuka.cmd(plan(t))
        pybullet_api.time.sleep(dt)

    pb.stop()
    pb.close()

    return 0


if __name__ == '__main__':
    sys.exit(main())

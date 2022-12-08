# Python standard lib
import os
import sys
import pathlib

# ROS
# import rospy
# from sensor_msgs.msg import JointState

# PyBullet
import pybullet_api

# OpTaS
import optas


class Planner:


    def __init__(self):

        cwd = pathlib.Path(__file__).parent.resolve() # path to current working directory
        pi = optas.np.pi  # 3.141...
        self.T = 50 # no. time steps in trajectory
        link_ee = 'end_effector_ball'  # end-effector link name
        self.Tmax = 10.  # trajectory of 5 secs
        t = optas.linspace(0, self.Tmax, self.T)
        self.dt = float((t[1] - t[0]).toarray()[0, 0])  # time step

        # Setup robot
        urdf_filename = os.path.join(cwd, 'robots', 'kuka_lwr.urdf')
        self.kuka = optas.RobotModel(
            urdf_filename=urdf_filename,
            time_derivs=[0, 1],  # i.e. joint position/velocity trajectory
            param_joints='lwr_arm_0_joint',
        )
        self.kuka_name = self.kuka.get_name()

        # Setup optimization builder
        builder = optas.OptimizationBuilder(T=self.T, robots=[self.kuka])

        # Setup parameters
        qc = builder.add_parameter('qc', self.kuka.ndof)  # current robot joint configuration

        # Constraint: initial configuration
        builder.initial_configuration(self.kuka_name, qc[self.kuka.opt_joint_indexes])
        builder.initial_configuration(self.kuka_name, time_deriv=1) # initial joint vel is zero

        # Constraint: dynamics
        builder.integrate_model_states(
            self.kuka_name,
            time_deriv=1, # i.e. integrate velocities to positions
            dt=self.dt,
        )

        # Get joint trajectory
        Q = builder.get_model_states_and_parameters(self.kuka_name)  # ndof-by-T symbolic array for robot trajectory

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
        dQ = builder.get_model_states_and_parameters(self.kuka_name, time_deriv=1)
        builder.add_cost_term('min_join_vel', 0.01*optas.sumsqr(dQ))

        # Prevent rotation in end-effector
        quat = self.kuka.get_global_link_quaternion_function(link_ee, n=self.T)
        builder.add_equality_constraint('no_eff_rot', quat(Q), quatc)

        # Setup solver
        optimization = builder.build()
        self.solver = optas.CasADiSolver(optimization).setup('ipopt')

    def plan(self, qc):


        # Set initial seed, note joint velocity will be set to zero
        Q0 = optas.diag(qc) @ optas.DM.ones(self.kuka.ndof, self.T)
        self.solver.reset_initial_seed({f'{self.kuka_name}/q': Q0[self.kuka.opt_joint_indexes, :]})

        # Set parameters
        self.solver.reset_parameters({
            'qc': optas.DM(qc),
            f'{self.kuka_name}/P': Q0[self.kuka.param_joint_indexes, :],
            f'{self.kuka_name}/dP': optas.DM.zeros(len(self.kuka.param_joint_indexes), self.T-1)
        })

        # Solve problem
        solution = self.solver.solve()

        # Merge solution with parameterized joint values
        Q = optas.DM.zeros(self.kuka.ndof, self.T)
        Q[self.kuka.opt_joint_indexes, :] = solution[f'{self.kuka_name}/q']
        Q[self.kuka.param_joint_indexes, :] = Q0[self.kuka.param_joint_indexes, :]

        # Interpolate
        plan = self.solver.interpolate(Q, self.Tmax)

        class Plan:

            def __init__(self, robot, plan_function):
                self.robot = robot
                self.plan_function = plan_function

            def __call__(self, t):
                q = self.plan_function(t)
                # msg = JointState(name=self.robot.actuated_joint_names, position=q.tolist())
                # msg.header.stamp = rospy.Time.now()
                # return msg
                return q

        return Plan(self.kuka, plan)

def main():

    # Initialize planner
    planner = Planner()

    # Plan trajectory
    qc = optas.np.deg2rad([0, 30, 0, -90, 0, -30, 0])
    plan = planner.plan(qc)

    # Setup PyBullet
    hz = 50
    dt = 1.0/float(hz)
    pb = pybullet_api.PyBullet(dt)
    kuka = pybullet_api.Kuka()
    kuka.reset(plan(0.))
    pb.start()

    # Connect to ROS and publish
    # rospy.init_node('figure_eight_plan_node')
    # js_pub = rospy.Publisher('rpbi/kuka_lwr/joint_states/target', JointState, queue_size=10)
    # rate = rospy.Rate(50)
    start_time = pybullet_api.time.time()

    # Main loop
    while True:
        t = pybullet_api.time.time() - start_time
        if t > planner.Tmax:
            break
        kuka.cmd(plan(t))
        # js_pub.publish(plan(t))
        # rate.sleep()
        pybullet_api.time.sleep(dt)

    pb.stop()
    pb.close()

    return 0


if __name__ == '__main__':
    sys.exit(main())

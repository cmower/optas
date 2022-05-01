import sys
import time
import rospy
import math
import random
import casadi as cs
import numpy as np
from math import radians
from pprint import pprint
import matplotlib.pyplot as plt
from sensor_msgs.msg import JointState
from ros_pybullet_interface.srv import ResetJointState
from custom_ros_tools.ros_comm import get_srv_handler
from custom_ros_tools.robot import resolve_joint_order

from pyinvk.builder import OptimizationBuilder
from pyinvk.robot_model import RobotModel
from pyinvk.solver import CasadiSolver, ScipySolver

from trac_ik_python.trac_ik_wrap import TRAC_IK

# Setup ros
rospy.init_node('test_pyinvk')
pub = rospy.Publisher('rpbi/kuka_lwr/joint_states/target', JointState, queue_size=1)
start_pos = np.array([-0.53775, -1.11219e-13, 0.40391])
Ntraj = 400
dt = 0.02
eff_b_tol = 0.001

fig_time, ax_time = plt.subplots(tight_layout=True)
ax_time.set_xlabel('Iteration')
ax_time.set_ylabel('CPU time (millisec)')

fig_jdiff, ax_jdiff = plt.subplots(tight_layout=True)
ax_jdiff.set_xlabel('Iteration')
ax_jdiff.set_ylabel('Joint diff (rad^2)')

fig_err, ax_err = plt.subplots(tight_layout=True)
ax_err.set_xlabel('Iteration')
ax_err.set_ylabel('Eff Error (m)')

def figure_eight(t):

    if t < 0.5*float(Ntraj)*dt:
        out = start_pos+np.array([
            0,
            math.sin(t * 2.0 * math.pi * 0.5) * 0.2,
            math.sin(t * math.pi * 0.5) * 0.3
        ])
    else:
        out = start_pos+np.array([
            0,
            math.sin(t * math.pi * 0.5) * 0.3,
            math.sin(t * 2.0 * math.pi * 0.5) * 0.2,
        ])

    return out

def run_expr_trac_ik():
    
    # Setup constants
    qstart = cs.DM([0, radians(30), 0, -radians(90), 0, radians(60), 0])
    N = 1
    t = 0.
    urdf = 'kuka_lwr.urdf'
    tip = 'lwr_arm_7_link'
    root = 'base'    
    robot_model = RobotModel(urdf, root, tip)    

    # Load solver and get joint names
    with open(urdf, 'r') as urdf:
        urdf_string = urdf.read()
        ik_solver = TRAC_IK(root, tip, urdf_string, 0.005, 1e-5, 'Speed')
        joint_names = ik_solver.getJointNamesInChain(urdf_string)

    eff_evolution = []
    comp_time = []
    eff_err = []
    joint_diff = []

    rate = rospy.Rate(50)
    for i in range(Ntraj):

        # Setup parameters
        eff_goal_ = figure_eight(t)
        xg, yg, zg = eff_goal_
        qseed_ = qstart.toarray().flatten().tolist()

        t0 = time.time_ns()
        solution = ik_solver.CartToJnt(
            qseed_,
            xg, yg, zg,
            0, 0, 0, 1,
            eff_b_tol, eff_b_tol, eff_b_tol, 100.0, 100.0, 100.0,
        )        
        t1 = time.time_ns()


        joint_diff.append(cs.norm_fro(cs.DM(solution) - cs.DM(qseed_)).toarray().flatten()[0])
        comp_time.append(1e-6*float(t1-t0))
        effpost = robot_model.get_end_effector_position(cs.DM(solution)).toarray().flatten().tolist()
        eff_evolution.append(effpost)
        eff_err.append(np.linalg.norm(np.array(effpost) - eff_goal_))

        # Publish joint state
        js = JointState(name=joint_names, position=solution)
        js.header.stamp = rospy.Time.now()
        pub.publish(js)
        print('trak_ik', "published", i+1, "of", Ntraj)
        t += dt
        qstart = cs.DM(solution)
        rate.sleep()

    # Plot
    ax_time.plot(np.arange(Ntraj), comp_time, 'x', label='trac_ik')
    ax_err.plot(np.arange(Ntraj)[1:], eff_err[1:], 'x', label='trak_ik')
    ax_jdiff.plot(np.arange(Ntraj), joint_diff, 'x', label='trac_ik')

    return eff_evolution, comp_time    

def run_expr(solver_interface, solver_name, solver_options):

    # Setup constants
    qstart = cs.DM([0, radians(30), 0, -radians(90), 0, radians(60), 0])
    N = 1
    t = 0.
    urdf = 'kuka_lwr.urdf'
    tip = 'lwr_arm_7_link'
    root = 'base'    

    # Setup robot model and optimization builder
    robot_model = RobotModel(urdf, root, tip)
    builder = OptimizationBuilder(robot_model, N)
    qseed = builder.add_parameter('qseed', robot_model.ndof)
    eff_goal = builder.add_parameter('eff_goal', 3)
    eff_pos = robot_model.get_end_effector_position(builder.get_q(0))

    # Setup cost function
    qdiff = qseed - builder.get_q(0)
    builder.add_cost_term('seed', qdiff.T@qdiff)

    # Setup constriants
    builder.enforce_joint_limits()

    for i in range(3):
        builder.add_ineq_constraint(f'euclidian_dist_err_{i}', eff_b_tol**2 - (eff_goal[i] - eff_pos[i])**2)

    # Build optimization
    optimization = builder.build()

    # Create solver
    if solver_interface == 'scipy':
        Solver = ScipySolver
    elif solver_interface == 'casadi':
        Solver = CasadiSolver

    solver = Solver(optimization)
    solver.setup(solver_name, solver_options)

    eff_evolution = []
    comp_time = []
    eff_err = []
    joint_diff = []

    rate = rospy.Rate(50)
    for i in range(Ntraj):

        # Setup parameters
        params = {
            'eff_goal': figure_eight(t),
            'qseed': qstart,
        }

        solver.reset(qstart, params)

        t0 = time.time_ns()
        solution = solver.solve()
        t1 = time.time_ns()

        joint_diff.append(cs.norm_fro(qstart - solution).toarray().flatten()[0])
        comp_time.append(1e-6*float(t1-t0))
        effpost = robot_model.get_end_effector_position(solution).toarray().flatten().tolist()
        eff_evolution.append(effpost)
        eff_err.append(np.linalg.norm(np.array(effpost) - params['eff_goal']))

        # Publish joint state
        jointmsgs = solver.solution_to_ros_joint_state_msgs(solution)
        js = jointmsgs[0]
        js.header.stamp = rospy.Time.now()
        pub.publish(js)
        print(solver_interface+'-'+solver_name, "published", i+1, "of", Ntraj)
        t += dt
        qstart = solution
        rate.sleep()

    # Plot
    ax_time.plot(np.arange(Ntraj), comp_time, 'x', label=solver_interface+'-'+solver_name)
    ax_err.plot(np.arange(Ntraj)[1:], eff_err[1:], 'x', label=solver_interface+'-'+solver_name)
    ax_jdiff.plot(np.arange(Ntraj), joint_diff, 'x', label=solver_interface+'-'+solver_name)

    return eff_evolution, comp_time

def main():

    # Setup experiments
    exprs = [
        # ('casadi', 'snopt', {}),
        # ('casadi', 'knitro', {}),
        ('casadi', 'ipopt', {'ipopt.print_level': 0, 'print_time': 0}),
        ('scipy', 'SLSQP', None),
        ('scipy', 'COBYLA', None),
        # ('scipy', 'trust-constr', None),
        'trac_ik',
    ]

    # Shuffle expr order
    random.shuffle(exprs)

    # Run experiments
    for e in exprs:
        if e == 'trac_ik':
            run_expr_trac_ik()
        else:
            run_expr(*e)
    
    # Plot results
    dt_ = 1.0/float(100)    
    ax_time.plot([0, Ntraj], [1000*dt_, 1000*dt_], '--r', label='100Hz')    
    ax_time.legend()
    ax_time.grid()
    ax_time.set_ylim(bottom=0)
    ax_time.set_xlim(0, Ntraj)
    ax_err.legend()
    ax_err.grid()
    ax_err.set_xlim(0, Ntraj)
    ax_err.set_ylim(bottom=0)
    ax_jdiff.legend()
    ax_jdiff.grid()
    ax_jdiff.set_ylim(bottom=0)
    ax_jdiff.set_xlim(0, Ntraj)
    plt.show()

if __name__ == '__main__':
    main()

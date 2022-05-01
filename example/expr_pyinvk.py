import os
import sys
import time
import rospy
import math
import random
import casadi as cs
import numpy as np
import tf_conversions
from math import radians
from pprint import pprint
import exotica_scipy_solver
import pyexotica as exo
import exotica_core_task_maps_py
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

# Setup constants
start_pos = np.array([-0.53775, -1.11219e-13, 0.40391])
Ntraj = 400
dt = 0.02
eff_b_tol = 0.001

eul_goal = np.array([radians(-135), 0, radians(90)])
quat_goal = tf_conversions.transformations.quaternion_from_euler(*eul_goal.tolist())

# Setup figure
fig_time, ax_time = plt.subplots(tight_layout=True)
ax_time.set_title('Solver duration')
ax_time.set_xlabel('Iteration')
ax_time.set_ylabel('CPU time (millisec)')

fig_jdiff, ax_jdiff = plt.subplots(tight_layout=True)
ax_jdiff.set_title('Joint difference')
ax_jdiff.set_xlabel('Iteration')
ax_jdiff.set_ylabel('Joint diff (rad^2)')

fig_err, ax_err = plt.subplots(tight_layout=True)
ax_err.set_title('Position error')
ax_err.set_xlabel('Iteration')
ax_err.set_ylabel('Eff Error (m)')

fig_err_eul, ax_err_eul = plt.subplots(tight_layout=True)
ax_err_eul.set_title('Rotation error')
ax_err_eul.set_xlabel('Iteration')
ax_err_eul.set_ylabel('Eff Error (rad)')

marker = 'o'
marker_size = 2

# Methods

def figure_eight(t):

    if t < 0.5*float(Ntraj)*dt:
        out = start_pos+np.array([
            0,
            math.sin(t * 2.0 * math.pi * 0.5) * 0.2,
            math.sin(t * math.pi * 0.5) * 0.2
        ])
    else:
        out = start_pos+np.array([
            0,
            math.sin(t * math.pi * 0.5) * 0.3,
            math.sin(t * 2.0 * math.pi * 0.5) * 0.2,
        ])

    return out

def run_expr_exotica_scipy(method):

    t = 0.
    urdf = 'kuka_lwr.urdf'
    tip = 'lwr_arm_7_link'
    root = 'base'    
    robot_model = RobotModel(urdf, root, tip)        

    qstart = np.array([0, radians(30), 0, -radians(90), 0, radians(60), 0])

    xml = os.getcwd()+'/exotica.xml'
    assert os.path.exists(xml), "xml file not found!"
    problem = exo.Setup.load_problem(xml)
    # solver = exo.Setup.load_solver('exotica.xml')
    # problem = solver.get_problem()
    solver = exotica_scipy_solver.SciPyEndPoseSolver(problem=problem, method=method, debug=False)
    scene = problem.get_scene()
    task_maps = problem.get_task_maps()

    eff_evolution = []
    comp_time = []
    eff_err = []
    joint_diff = []
    eff_eul_err = []
    
    rate = rospy.Rate(50)
    for i in range(Ntraj):

        # Setup parameters
        eff_goal_ = figure_eight(t)
        xg, yg, zg = eff_goal_
        qseed_ = qstart.copy()

        # problem.set_goal('Position', eff_goal_)
        scene.attach_object_local('Target', '', eff_goal_.tolist()+quat_goal.tolist())
        problem.start_state = qseed_
        jp = task_maps['JointPose']
        jp.joint_ref = qseed_
        # print(jp)
        # import pprint
        # pprint.pprint(dir(jp))
        # quit()
        # .set_joint_ref(qseed_)

        t0 = time.time_ns()
        solution = solver.solve()[0]
        t1 = time.time_ns()

        joint_diff.append(cs.norm_fro(cs.DM(solution) - cs.DM(qseed_)).toarray().flatten()[0])
        comp_time.append(1e-6*float(t1-t0))
        effpost = robot_model.get_end_effector_position(cs.DM(solution)).toarray().flatten().tolist()
        eff_evolution.append(effpost)
        eff_err.append(np.linalg.norm(np.array(effpost) - eff_goal_))
        eff_eul_err.append(np.linalg.norm(eul_goal - robot_model.get_end_effector_euler(solution).toarray().flatten()))
        
        # Publish joint state
        js = JointState(name=robot_model.joint_names, position=solution)
        js.header.stamp = rospy.Time.now()
        pub.publish(js)
        print('exotica-scipy-'+method, "published", i+1, "of", Ntraj)
        t += dt
        qstart = solution.copy()
        rate.sleep()

    # Plot
    ax_time.plot(np.arange(Ntraj), comp_time, marker, ms=marker_size, label='exotica-scipy-'+method)
    ax_err.plot(np.arange(Ntraj)[1:], eff_err[1:], marker, ms=marker_size, label='exotica-scipy-'+method)
    ax_jdiff.plot(np.arange(Ntraj), joint_diff, marker, ms=marker_size, label='exotica-scipy-'+method)
    ax_err_eul.plot(np.arange(Ntraj), eff_eul_err, marker, ms=marker_size, label='exotica-scipy-'+method)

    return eff_evolution, comp_time

def run_expr_exotica_snopt():

    t = 0.
    urdf = 'kuka_lwr.urdf'
    tip = 'lwr_arm_7_link'
    root = 'base'    
    robot_model = RobotModel(urdf, root, tip)        

    qstart = np.array([0, radians(30), 0, -radians(90), 0, radians(60), 0])

    xml = os.getcwd()+'/exotica_snopt.xml'
    assert os.path.exists(xml), "xml file not found!"
    # problem = exo.Setup.load_problem(xml)
    solver = exo.Setup.load_solver(xml)
    problem = solver.get_problem()
    # solver = exotica_scipy_solver.SciPyEndPoseSolver(problem=problem, method=method, debug=False)
    scene = problem.get_scene()
    task_maps = problem.get_task_maps()

    eff_evolution = []
    comp_time = []
    eff_err = []
    joint_diff = []
    eff_eul_err = []
    
    rate = rospy.Rate(50)
    for i in range(Ntraj):

        # Setup parameters
        eff_goal_ = figure_eight(t)
        xg, yg, zg = eff_goal_
        qseed_ = qstart.copy()

        # problem.set_goal('Position', eff_goal_)
        scene.attach_object_local('Target', '', eff_goal_.tolist()+quat_goal.tolist())
        problem.start_state = qseed_
        jp = task_maps['JointPose']
        jp.joint_ref = qseed_
        # print(jp)
        # import pprint
        # pprint.pprint(dir(jp))
        # quit()
        # .set_joint_ref(qseed_)

        t0 = time.time_ns()
        solution = solver.solve()[0]
        t1 = time.time_ns()

        joint_diff.append(cs.norm_fro(cs.DM(solution) - cs.DM(qseed_)).toarray().flatten()[0])
        comp_time.append(1e-6*float(t1-t0))
        effpost = robot_model.get_end_effector_position(cs.DM(solution)).toarray().flatten().tolist()
        eff_evolution.append(effpost)
        eff_err.append(np.linalg.norm(np.array(effpost) - eff_goal_))
        eff_eul_err.append(np.linalg.norm(eul_goal - robot_model.get_end_effector_euler(solution).toarray().flatten()))        

        # Publish joint state
        js = JointState(name=robot_model.joint_names, position=solution)
        js.header.stamp = rospy.Time.now()
        pub.publish(js)
        print('exotica-snopt', "published", i+1, "of", Ntraj)
        t += dt
        qstart = solution.copy()
        rate.sleep()

    # Plot
    ax_time.plot(np.arange(Ntraj), comp_time, marker, ms=marker_size, label='exotica-snopt')
    ax_err.plot(np.arange(Ntraj)[1:], eff_err[1:], marker, ms=marker_size, label='exotica-snopt')
    ax_jdiff.plot(np.arange(Ntraj), joint_diff, marker, ms=marker_size, label='exotica-snopt')
    ax_err_eul.plot(np.arange(Ntraj), eff_eul_err, marker, ms=marker_size, label='exotica-snopt')
    
    return eff_evolution, comp_time        
    

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
    eff_eul_err = []

    rate = rospy.Rate(50)
    for i in range(Ntraj):

        # Setup parameters
        eff_goal_ = figure_eight(t)
        xg, yg, zg = eff_goal_
        qseed_ = qstart.toarray().flatten().tolist()
        rx, ry, rz, rq = quat_goal.copy()

        t0 = time.time_ns()
        solution = ik_solver.CartToJnt(
            qseed_,
            xg, yg, zg,
            rx, ry, rz, rq,
            eff_b_tol, eff_b_tol, eff_b_tol, eff_b_tol, eff_b_tol, eff_b_tol,
        )        
        t1 = time.time_ns()


        joint_diff.append(cs.norm_fro(cs.DM(solution) - cs.DM(qseed_)).toarray().flatten()[0])
        comp_time.append(1e-6*float(t1-t0))
        effpost = robot_model.get_end_effector_position(cs.DM(solution)).toarray().flatten().tolist()
        eff_evolution.append(effpost)
        eff_err.append(np.linalg.norm(np.array(effpost) - eff_goal_))
        eff_eul_err.append(np.linalg.norm(eul_goal - robot_model.get_end_effector_euler(solution).toarray().flatten()))        

        # Publish joint state
        js = JointState(name=joint_names, position=solution)
        js.header.stamp = rospy.Time.now()
        pub.publish(js)
        print('trak_ik', "published", i+1, "of", Ntraj)
        t += dt
        qstart = cs.DM(solution)
        rate.sleep()

    # Plot
    ax_time.plot(np.arange(Ntraj), comp_time, marker, ms=marker_size, label='trac_ik')
    ax_err.plot(np.arange(Ntraj)[1:], eff_err[1:], marker, ms=marker_size, label='trak_ik')
    ax_jdiff.plot(np.arange(Ntraj), joint_diff, marker, ms=marker_size, label='trac_ik')
    ax_err_eul.plot(np.arange(Ntraj), eff_eul_err, marker, ms=marker_size, label='trac_ik')
    
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
    eff_quat = robot_model.get_end_effector_quaternion(builder.get_q(0))
    eff_eul = robot_model.get_end_effector_euler(builder.get_q(0))

    # Setup cost function
    qdiff = qseed - builder.get_q(0)
    builder.add_cost_term('seed', qdiff.T@qdiff)

    # Setup constriants
    builder.enforce_joint_limits()

    for i in range(3):
        builder.add_ineq_constraint(f'rot_dist_err_{i}', eff_b_tol**2 - (eff_eul[i] - eul_goal[i])**2)
        builder.add_ineq_constraint(f'euclidian_dist_err_{i}', eff_b_tol**2 - (eff_goal[i] - eff_pos[i])**2)
    # builder.add_ineq_constraint(f'rot_err', eff_b_tol**2 - cs.sumsqr(eff_quat))
    
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
    eff_eul_err = []
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
        eff_eul_err.append(np.linalg.norm(eul_goal - robot_model.get_end_effector_euler(solution).toarray().flatten()))

        # Publish joint state
        jointmsgs = solver.solution_to_ros_joint_state_msgs(solution)
        js = jointmsgs[0]
        js.header.stamp = rospy.Time.now()
        pub.publish(js)
        print('pyinvk-'+solver_interface+'-'+solver_name, "published", i+1, "of", Ntraj)
        t += dt
        qstart = solution
        rate.sleep()

    # Plot
    ax_time.plot(np.arange(Ntraj), comp_time, marker, ms=marker_size, label='pyinvk-'+solver_interface+'-'+solver_name)
    ax_err.plot(np.arange(Ntraj)[1:], eff_err[1:], marker, ms=marker_size, label='pyinvk-'+solver_interface+'-'+solver_name)
    ax_jdiff.plot(np.arange(Ntraj), joint_diff, marker, ms=marker_size, label='pyinvk-'+solver_interface+'-'+solver_name)
    ax_err_eul.plot(np.arange(Ntraj), eff_eul_err, marker, ms=marker_size, label='pyinvk-'+solver_interface+'-'+solver_name)
    
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
        ('exotica_scipy', 'SLSQP'),
        'exotica_snopt',
    ]

    # Shuffle expr order
    random.shuffle(exprs)

    # Run experiments    
    for e in exprs:
        if e == 'trac_ik':
            run_expr_trac_ik()
        elif e == 'exotica_snopt':
            run_expr_exotica_snopt()            
        elif e[0] == 'exotica_scipy':
            run_expr_exotica_scipy(e[1])
        else:
            run_expr(*e)
    
    # Plot results
    dt_ = 1.0/float(100)    
    ax_time.plot([0, Ntraj], [1000*dt_, 1000*dt_], '--r', label='100Hz')    
    ax_time.legend(loc='upper left')
    ax_time.grid()
    ax_time.set_ylim(bottom=0)
    ax_time.set_xlim(0, Ntraj)
    ax_err.legend(loc='upper left')
    ax_err.grid()
    ax_err.set_xlim(0, Ntraj)
    ax_err.set_ylim(bottom=0)
    ax_err_eul.legend(loc='upper left')
    ax_err_eul.grid()
    ax_err_eul.set_xlim(0, Ntraj)
    ax_err_eul.set_ylim(bottom=0)    
    ax_jdiff.legend(loc='upper left')
    ax_jdiff.grid()
    ax_jdiff.set_ylim(bottom=0)
    ax_jdiff.set_xlim(0, Ntraj)

    fig_time.savefig('./fig/time.pdf')
    fig_jdiff.savefig('./fig/jdiff.pdf')
    fig_err.savefig('./fig/err.pdf')
    fig_err_eul.savefig('./fig/err_eul.pdf')
    
    plt.show()

if __name__ == '__main__':
    main()

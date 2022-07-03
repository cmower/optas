# PyInvK: examples

Several examples are provided that highlight the different features of the PyInvK library.
The examples require ROS and the [ROS-PyBullet Interface](https://github.com/cmower/ros_pybullet_interface).

## Examples overview

### `ex_dual_arm.py`

This examples highlights the libraries ability to solve problems with more than one robot included in the problem formulation.

To run this example execute the following.
```
  # Open a terminal
  $ roslaunch rpbi_examples run_dual_kukas.launch
  
  # Open a second terminal
  $ cd /path/to/pyinvk/examples
  $ python ex_dual_arm.py
```

### `ex_lin_constrained_nlp.py`

This example shows how to setup a linearly constrained NLP. 

To run this example execute the following.
```
  # Open a terminal
  $ roslaunch rpbi_examples run_kuka.launch
  
  # Open a second terminal
  $ cd /path/to/pyinvk/examples
  $ python ex_lin_constrained_nlp.py
```

### `ex_lin_constrained_qp.py`

This example shows how to setup a linearly constrained QP. 

To run this example execute the following.
```
  # Open a terminal
  $ roslaunch rpbi_examples run_kuka.launch
  
  # Open a second terminal
  $ cd /path/to/pyinvk/examples
  $ python ex_lin_constrained_qp.py
```

### `ex_nonlin_constrained_nlp.py`

This example shows how to setup an NLP with nonlinear constraints.

To run this example execute the following.
```
  # Open a terminal
  $ roslaunch rpbi_examples run_kuka.launch
  
  # Open a second terminal
  $ cd /path/to/pyinvk/examples
  $ python ex_nonlin_constrained_nlp.py
```

### `ex_nonlin_constrained_qp.py`

This example shows how to setup a QP with nonlinear constraints.

To run this example execute the following.
```
  # Open a terminal
  $ roslaunch rpbi_examples run_kuka.launch
  
  # Open a second terminal
  $ cd /path/to/pyinvk/examples
  $ python ex_nonlin_constrained_qp.py
```

### `ex_unconstrained_nlp.py`

This example shows how to setup an unconstrained NLP.

To run this example execute the following.
```
  # Open a terminal
  $ roslaunch rpbi_examples run_kuka.launch
  
  # Open a second terminal
  $ cd /path/to/pyinvk/examples
  $ python ex_unconstrained_nlp.py
```

### `ex_unconstrained_qp.py`

This example shows how to setup an unconstrained QP.

To run this example execute the following.
```
  # Open a terminal
  $ roslaunch rpbi_examples run_kuka.launch
  
  # Open a second terminal
  $ cd /path/to/pyinvk/examples
  $ python ex_unconstrained_qp.py
```

### `mpc_for_shared_autonomy.py`

This example demonstrates an implementation of the controller described in the following paper.

1. [M. Rubagotti, T. Taunyazov, B. Omarali and A. Shintemirov, "Semi-Autonomous Robot Teleoperation With Obstacle Avoidance via Model Predictive Control," in IEEE Robotics and Automation Letters, vol. 4, no. 3, pp. 2746-2753, July 2019, doi: 10.1109/LRA.2019.2917707.](https://ieeexplore.ieee.org/document/8718327)

To run this example execute the following.
```
  # Open a terminal
  $ roslaunch rpbi_examples run_kuka.launch
  
  # Open a second terminal
  $ cd /path/to/pyinvk/examples
  $ roslaunch mpc_for_shared_autonomy.launch
  
  # Open a third terminal
  $ cd /path/to/pyinvk/examples
  $ python mpc_for_shared_autonomy.py
```

Use the LEFT/RIGHT/UP/DOWN and f/b keys to control the robot end-effector (note, there is a blank window that is started - this must be in focus for keyboard commands to be sent to the robot).

# Examples

Several examples are contained in this directory.
In addition, two experiments are also included.

The majority of these examples require [PyBullet](https://pybullet.org/wordpress/).
If you installed OpTaS without modifying `setup.py` this should have been installed by default.
Otherwise, you can install PyBullet using `$ pip install pybullet`.

## Experiments

### Experiment 1

Assuming PyBullet is installed, this experiment should run out of the box.

### Experiment 2

This experiment requires some additional setup to run.
Ultimately, it requires [TracIK](https://traclabs.com/projects/trac-ik/) and [EXOTica](https://github.com/ipab-slmc/exotica).
The easiest way to install these and run the experiment is to follow these instructions.

1. [Install ROS Noetic (full install)](http://wiki.ros.org/noetic/Installation/Ubuntu)
2. [Install catkin_tools](https://catkin-tools.readthedocs.io/en/latest/)
3. Create catkin workspace and `src` directory: `$ mkdir -p optas_expr_ws/src && cd optas_expr_ws`
4. Initialize catkin workspace: `$ catkin init`
5. Change directory: `$ cd src`
6. Clone EXOTica: `$ git clone git@github.com:ipab-slmc/exotica.git`
7. Clone TracIK: `$ git clone https://bitbucket.org/traclabs/trac_ik.git`
8. Install dependencies: `$ rosdep update ; rosdep install --from-paths ./ -iry`
9. Build catkin workspace: `$ catkin build`
10. Source catkin workspace: `$ source ../devel/setup.bash`
11. Change directory: `$ cd /path/to/optas/example`
12. Run experiment: `$ python experiment2.py` (**note**: since EXOTica loads URDF/SRDF files using relative filenames to where the load methods are called, you must ensure to run this script from inside the `example/` directory - this doesn't apply for the other examples)

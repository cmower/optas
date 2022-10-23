import unittest
import optas
import optas
import numpy as np
from scipy.spatial.transform import Rotation as R
np.random.seed(10)  # ensure random numbers are consistent

pi = np.pi

@optas.arrayify_args
def isclose(A, B):
    """Returns a boolean array where two arrays are element-wise equal within a tolerance."""
    A = A.toarray().flatten()
    B = B.toarray().flatten()
    return np.allclose(A, B)

urdf_string = """<?xml version="1.0"?>
<robot name="planar_3dof" xmlns:xacro="http://ros.org/wiki/xacro">

  <link name="world" />

  <link name="base_link">
  </link>

  <joint name="fixed" type="fixed">
    <parent link="world" />
    <child link="base_link" />
  </joint>

  <link name="link_1">
    <inertial>
      <origin xyz="0 0 .25" rpy="0 0 0" />
      <mass value="1" />
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0" />
    </inertial>
  </link>
  <joint name="joint_1" type="revolute">
    <parent link="base_link" />
    <child link="link_1" />
    <origin xyz="0 0 0" rpy="0 0 0" />
    <axis xyz="0 0 1" />
    <limit lower="-1.57" upper="1.57" effort="0" velocity="0.5" />
  </joint>

  <link name="link_2">
    <inertial>
      <origin xyz="0 0 0.25" rpy="0 0 0" />
      <mass value="1" />
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0" />
    </inertial>
  </link>

  <joint name="joint_2" type="revolute">
    <parent link="link_1" />
    <child link="link_2" />
    <origin xyz="1.5 0.0 0.0" rpy="0.0 0 0.0" />
    <axis xyz="0 0 1" />
    <limit lower="-1.57" upper="1.57" effort="0" velocity="0.5" />
  </joint>

  <link name="gripper">
    <inertial>
      <origin xyz="0 0 0.1" rpy="0 0 0" />
      <mass value="1" />
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0" />
    </inertial>
  </link>

  <joint name="joint_3" type="revolute">
    <parent link="link_2" />
    <child link="gripper" />
    <origin xyz="1.5 0.0 0.0" rpy="0.0 0 0.0" />
    <axis xyz="0 0 1" />
    <limit lower="-1.57" upper="1.57" effort="0" velocity="0.5" />
  </joint>
  <link name="end" />

  <joint name="gripper_end" type="fixed">
    <parent link="gripper" />
    <child link="end" />
    <origin xyz="1.0 0 0" rpy="0 0 0" />
  </joint>

</robot>

"""

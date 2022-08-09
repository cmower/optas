import casadi as cs
import numpy as np

from .transformations import euler_matrix
from .spatialmath import *
    
@vectorize_args
def quaternion_product(quata, quatb):
    """Compute the quaternion product quata.quatb."""
    x0, y0, z0, w0 = quata[0], quata[1], quata[2], quata[3]
    x1, y1, z1, w1 = quatb[0], quatb[1], quatb[2], quatb[3]
    return cs.vertcat(
        w0*x1 + x0*w1 + y0*z1 - z0*y1,
        w0*y1 - x0*z1 + y0*w1 + z0*x1,
        w0*z1 + x0*y1 - y0*x1 + z0*w1,
        w0*w1 - x0*x1 - y0*y1 - z0*z1
    )

@vectorize_args
def quaternion_fixed(rpy):
    """Quaternion for a fixed joint."""

    r = rpy[0]
    p = rpy[1]
    y = rpy[2]

    cr = cs.cos(0.5*r)
    sr = cs.sin(0.5*r)

    cp = cs.cos(0.5*p)
    sp = cs.sin(0.5*p)

    cy = cs.cos(0.5*y)
    sy = cs.sin(0.5*y)

    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    w = cr * cp * cy + sr * sp * sy

    n = cs.sqrt(x*x + y*y + z*z + w*w)

    return cs.vertcat(x, y, z, w)/n

@vectorize_args
def quaternion_revolute(xyz, rpy, axis, qi):
    """Quaternion for a revolute joint."""

    axis = cs.vec(axis)

    rpy = cs.vec(rpy)

    r = rpy[0]
    p = rpy[1]
    y = rpy[2]

    cr = cs.cos(0.5*r)
    sr = cs.sin(0.5*r)

    cp = cs.cos(0.5*p)
    sp = cs.sin(0.5*p)

    cy = cs.cos(0.5*y)
    sy = cs.sin(0.5*y)

    x_or = cy*sr*cp - sy*cr*sp
    y_or = cy*cr*sp + sy*sr*cp
    z_or = sy*cr*cp - cy*sr*sp
    w_or = cy*cr*cp + sy*sr*sp
    q_or = cs.vertcat(x_or, y_or, z_or, w_or)

    cqi = cs.cos(0.5*qi)
    sqi = cs.sin(0.5*qi)
    x_j = axis[0]*sqi
    y_j = axis[1]*sqi
    z_j = axis[2]*sqi
    w_j = cqi
    q_j = cs.vertcat(x_j, y_j, z_j, w_j)

    return quaternion_product(q_or, q_j)

@vectorize_args
def rotation_matrix_fixed(rpy):
    """Rotation matrix for a fixed joint."""
    return rpy2r(rpy[0], rpy[1], rpy[2])

@vectorize_args
def transformation_matrix_fixed(xyz, rpy):
    """Transformation matrix for a fixed joint."""
    return rt2tr(rpy2r(rpy[0], rpy[1], rpy[2]), xyz)

@vectorize_args
def transformation_matrix_prismatic(xyz, rpy, axis, qi):
    """Transformation matrix for a prismatic joint."""
    return tr2tr(R, cs.vec(xyz) + qi*R @ cs.vec(axis))

@vectorize_args
def transformation_matrix_revolute(xyz, rpy, axis, q):
    """Transformation matrix for a revolute joint."""
    return rt2tr(euler_matrix(rpy[0], rpy[1], rpy[2]) @ angvec2r(q, axis), xyz)

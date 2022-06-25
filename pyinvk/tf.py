import casadi as cs
import numpy as np
from typing import Union, List

ArrayLike = Union[cs.casadi.SX, cs.casadi.DM, np.ndarray, List]
CasADiArray = Union[cs.casadi.SX, cs.casadi.DM]

def quaternion_product(quata: ArrayLike, quatb: ArrayLike) -> CasADiArray:
    """Compute the quaternion product quata.quatb."""
    quata_ = cs.vec(quata)
    quatb_ = cs.vec(quatb)
    x0, y0, z0, w0 = quata_[0], quata_[1], quata_[2], quata_[3]
    x1, y1, z1, w1 = quatb_[0], quatb_[1], quatb_[2], quatb_[3]
    return cs.vertcat(
        w0*x1 + x0*w1 + y0*z1 - z0*y1,
        w0*y1 - x0*z1 + y0*w1 + z0*x1,
        w0*z1 + x0*y1 - y0*x1 + z0*w1,
        w0*w1 - x0*x1 - y0*y1 - z0*z1
    )

def quaternion_fixed(rpy: ArrayLike) -> CasADiArray:
    """Quaternion for a fixed joint."""

    rpy_ = cs.vec(rpy)

    r = rpy_[0]
    p = rpy_[1]
    y = rpy_[2]

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

def quaternion_revolute(xyz: ArrayLike, rpy: ArrayLike, axis: ArrayLike, qi: ArrayLike) -> CasADiArray:
    """Quaternion for a revolute joint."""

    axis_ = cs.vec(axis)

    rpy_ = cs.vec(rpy)

    r = rpy_[0]
    p = rpy_[1]
    y = rpy_[2]

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
    x_j = axis_[0]*sqi
    y_j = axis_[1]*sqi
    z_j = axis_[2]*sqi
    w_j = cqi
    q_j = cs.vertcat(x_j, y_j, z_j, w_j)

    return quaternion_product(q_or, q_j)

def euler_from_transformation_matrix(T: ArrayLike) -> CasADiArray:
    """Get the Euler angles from a transformation matrix."""
    if isinstance(T, list):
        T = cs.DM(T)
    R = T[:3, :3]
    eps = float(cs.np.finfo(float).eps * 4.0)
    i, j, k = 0, 1, 2
    cy = cs.sqrt(T[i, i]*T[i, i] + T[j, i]*T[j, i])
    return cs.if_else(
        cy >= eps,
        cs.vertcat(
            cs.arctan2( T[k, j], T[k, k]),
            cs.arctan2(-T[k, i], cy),
            cs.arctan2( T[j, i], T[i, i]),
        ),
        cs.vertcat(
            cs.arctan2(-T[j, k], T[j, j]),
            cs.arctan2(-T[k, i], cy),
            0.0,
        ),
    )

def rotation_matrix_fixed(rpy: ArrayLike) -> CasADiArray:
    """Rotation matrix for a fixed joint."""

    rpy_ = cs.vec(rpy)

    r = rpy_[0]
    p = rpy_[1]
    y = rpy_[2]

    cr = cs.cos(r)
    sr = cs.sin(r)

    cp = cs.cos(p)
    sp = cs.sin(p)

    cy = cs.cos(y)
    sy = cs.sin(y)

    return cs.vertcat(
        cs.horzcat(cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr),
        cs.horzcat(sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr),
        cs.horzcat(  -sp,             cp*sr,             cp*cr),
    )

def transformation_matrix_fixed(xyz: ArrayLike, rpy: ArrayLike) -> CasADiArray:
    """Transformation matrix for a fixed joint."""
    return cs.vertcat(
        cs.horzcat(rotation_matrix_fixed(rpy), cs.vec(xyz)),
        cs.DM([[0.0, 0.0, 0.0, 1.0]]),
    )

def transformation_matrix_prismatic(xyz: ArrayLike, rpy: ArrayLike, axis: ArrayLike, qi: ArrayLike) -> CasADiArray:
    """Transformation matrix for a prismatic joint."""
    R = rotation_matrix_fixed(rpy)
    return cs.vertcat(
        cs.horzcat(R, cs.vec(xyz) + qi*R @ cs.vec(axis)),
        cs.DM([[0.0, 0.0, 0.0, 1.0]]),
    )

def transformation_matrix_revolute(xyz: ArrayLike, rpy: ArrayLike, axis: ArrayLike, qi: ArrayLike) -> CasADiArray:
    """Transformation matrix for a revolute joint."""

    # Setup
    xyz_ = cs.vec(xyz)
    R = rotation_matrix_fixed(rpy)
    axis_ = cs.vec(axis)

    # joint rotation from skew-symmetric axis angle
    cqi = cs.cos(qi)
    sqi = cs.sin(qi)
    s00 = (1.0 - cqi)*axis_[0]*axis_[0] + cqi
    s11 = (1.0 - cqi)*axis_[1]*axis_[1] + cqi
    s22 = (1.0 - cqi)*axis_[2]*axis_[2] + cqi
    s01 = (1.0 - cqi)*axis_[0]*axis_[1] - axis_[2]*sqi
    s10 = (1.0 - cqi)*axis_[0]*axis_[1] + axis_[2]*sqi
    s12 = (1.0 - cqi)*axis_[1]*axis_[2] - axis_[0]*sqi
    s21 = (1.0 - cqi)*axis_[1]*axis_[2] + axis_[0]*sqi
    s20 = (1.0 - cqi)*axis_[0]*axis_[2] - axis_[1]*sqi
    s02 = (1.0 - cqi)*axis_[0]*axis_[2] + axis_[1]*sqi

    return cs.vertcat(
        cs.horzcat(R[0,0]*s00 + R[0,1]*s10 + R[0,2]*s20, R[0,0]*s01 + R[0,1]*s11 + R[0,2]*s21, R[0,0]*s02 + R[0,1]*s12 + R[0,2]*s22, xyz_[0]),
        cs.horzcat(R[1,0]*s00 + R[1,1]*s10 + R[1,2]*s20, R[1,0]*s01 + R[1,1]*s11 + R[1,2]*s21, R[1,0]*s02 + R[1,1]*s12 + R[1,2]*s22, xyz_[1]),
        cs.horzcat(R[2,0]*s00 + R[2,1]*s10 + R[2,2]*s20, R[2,0]*s01 + R[2,1]*s11 + R[2,2]*s21, R[2,0]*s02 + R[2,1]*s12 + R[2,2]*s22, xyz_[2]),
        cs.DM([[0.0, 0.0, 0.0, 1.0]]),
    )

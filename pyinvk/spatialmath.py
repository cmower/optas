import casadi as cs
from casadi import sin, cos, vec

"""

This is a partial port to Python/CasADi, with some modifications, of
the Spatial Math Toolbox for MATLAB. See the following.


https://github.com/petercorke/spatialmath-matlab

"""

eps = cs.np.finfo(float).eps

_arraylike_types = (cs.DM, cs.SX, list, tuple, cs.np.ndarray, float, int)

def _handle_arraylike_args(args, handle):
    args_out = []
    for a in args:
        if isinstance(a, _arraylike_types):
            args_out.append(handle(a))
        else:
            args_out.append(a)
    return args_out

def arrayify_args(fun):
    """Decorator that ensures all input arguments are casadi arrays (i.e. either DM or SX)"""

    def wrap(*args, **kwargs):
        args_use = _handle_arraylike_args(args, cs.horzcat)
        return fun(*args_use, **kwargs)

    return wrap

def vectorize_args(fun):
    """Decorator that vectorizes all input arguments."""

    def wrap(*args, **kwargs):
        args_use = _handle_arraylike_args(args, vec)
        return fun(*args_use, **kwargs)

    return wrap

def _is_shape(M, s1, s2):
    return M.shape[0] == s1 and M.shape[1] == s2

def is_2x2(M):
    return _is_shape(M, 2, 2)

def is_3x3(M):
    return _is_shape(M, 3, 3)

def I3():
    """3-by-3 identity matrix"""
    return cs.DM.eye(3)

def I4():
    """4-by-4 identity matrix"""
    return cs.DM.eye(4)

@vectorize_args
def angvec2r(theta, v):
    """Convert angle and vector orientation to a rotation matrix"""
    sk = skew(unit(v))
    R = I3() + sin(theta)*sk + (1.-cos(theta))*sk@sk # Rodrigue's equation
    return R

@vectorize_args
def angvec2tr(theta, v):
    """Convert angle and vector orientation to a homogeneous transform"""
    return r2t(angvec2r(theta, v))

@vectorize_args
def delta2tr(d):
    """Convert differential motion  to SE(3) homogeneous transform"""
    up = cs.horzcat(skew(d[3:6]), d[:3])
    lo = cs.DM.zeros(1, 4)
    return I4() + cs.vertcat(up, lo)

@arrayify_args
def e2h(e):
    """Euclidean to homogeneous"""
    ones = cs.DM.ones(1, e.shape[1])
    return cs.vertcat(e, ones)

@vectorize_args
def eul2jac(phi, theta, psi):
    """Euler angle rate Jacobian"""
    sphi, cphi = sin(phi), cos(phi)
    stheta, ctheta = sin(theta), cos(theta)
    return cs.vertcat(
        cs.horzcat(0., -sphi, cphi*stheta),
        cs.horzcat(0.,  cphi, sphi*stheta),
        cs.horzcat(1.,    0.,      ctheta),
    )

@vectorize_args
def eul2r(phi, theta, psi):
    """Convert Euler angles to rotation matrix"""
    return rotz(phi) @ roty(theta) @ rotz(psi)

@vectorize_args
def eul2tr(phi, theta, psi):
    """Convert Euler angles to homogeneous transform"""
    return r2t(eul2r(phi, theta, psi))

@arrayify_args
def h2e(h):
    """Homogeneous to Euclidean"""
    return h[:-1, :] / cs.repmat(h[-1, :], h.shape[0]-1, 1)

@vectorize_args
def oa2r(o, a):
    """Convert orientation and approach vectors to rotation matrix"""
    n = cs.cross(o, a)
    o = cs.cross(a, n)
    return cs.horzcat(unit(vec(n)), unit(vec(o)), unit(vec(a)))

@arrayify_args
def oa2tr(o, a):
    """Convert orientation and approach vectors to homogeneous transformation"""
    n = cs.cross(o, a)
    o = cs.cross(a, n)
    return cs.vertcat(
        cs.horzcat(unit(vec(n)), unit(vec(o)), unit(vec(a)), cs.DM.zeros(3)),
        cs.DM([[0., 0., 0., 1]]),
    )

@arrayify_args
def r2t(R):
    """Convert rotation matrix to a homogeneous transform"""
    return cs.vertcat(
        cs.horzcat(R, cs.DM.zeros(3)),
        cs.DM([[0., 0., 0., 1]]),
    )

@vectorize_args
def rot2(theta):
    """SO(2) rotation matrix"""
    ct, st = cos(theta), sin(theta)
    return cs.vertcat(
        cs.horzcat(ct, -st),
        cs.horzcat(st,  ct),
    )

@vectorize_args
def rotx(theta):
    """SO(3) rotation about X axis"""
    ct, st = cos(theta), sin(theta)
    return cs.vertcat(
        cs.DM([[1.0, 0.0, 0.0]]),
        cs.horzcat(0., ct, -st),
        cs.horzcat(0,  st,  ct),
    )

@vectorize_args
def roty(theta):
    """SO(3) rotation about Y axis"""
    ct, st = cos(theta), sin(theta)
    return cs.vertcat(
        cs.horzcat(ct, 0., st),
        cs.DM([[0., 1., 0.]]),
        cs.horzcat(-st, 0., ct),
    )

@vectorize_args
def rotz(theta):
    """SO(3) rotation about Z axis"""
    ct, st = cos(theta), sin(theta)
    return cs.vertcat(
        cs.horzcat(ct, -st, 0.),
        cs.horzcat(st, ct, 0.),
        cs.DM([[0., 0., 1.]]),
    )

@vectorize_args
def rpy2jac(rpy, opt='zyx'):
    """Jacobian from RPY angle rates to angular velocity"""

    order = ['zyx', 'xyz', 'yxz']
    r, p, y = cs.vertsplit(rpy)

    sr, cr = sin(r), cos(r)
    sp, cp = sin(p), cos(p)
    sy, cy = sin(y), cos(y)

    if opt == 'xyz':
        return cs.vertcat(
            cs.horzcat(sp, 0., 1.),
            cs.horzcat(-cp*sy, cy, 0.),
            cs.horzcat(cp*cy, sy, 0.),
        )
    elif opt == 'zyx':
        return cs.vertcat(
            cs.horzcat(cp*cy, -sy, 0.),
            cs.horzcat(cp*sy, cy, 0.),
            cs.horzcat(-sp, 0., 1.),
        )
    elif opt == 'yxz':
        return cs.vertcat(
            cs.horzcat(cp*sy, cy, 0.),
            cs.horzcat(-sp, 0., 1.),
            cs.horzcat(cp*cy, -sy, 0.),
        )
    else:
        raise ValueError(f"didn't recognize given option {opt=}, only allowed {order}")

@vectorize_args
def rpy2r(rpy, opt='zyx'):
    """Roll-pitch-yaw angles to SO(3) rotation matrix"""

    order = ['zyx', 'xyz', 'yxz', 'arm', 'vehicle', 'camera']
    r, p, y = cs.vertsplit(rpy)

    if opt in {'xyz', 'arm'}:
        return rotx(y) @ roty(p) @ rotz(r)
    elif opt in {'zyx', 'vehicle'}:
        return rotz(y) @ roty(p) @ rotx(r)
    elif opt in {'yxz', 'camera'}:
        return roty(y) @ rotx(p) @ rotz(r)
    else:
        raise ValueError(f"didn't recognize given option {opt=}, only allowed {order}")

@vectorize_args
def rpy2tr(rpy, opt='zyx'):
    return r2t(rpy2r(rpy, opt=opt))

@arrayify_args
def rt2tr(R, t):
    """Convert rotation and translation to homogeneous transform"""
    return cs.vertcat(
        cs.horzcat(R, vec(t)),
        cs.DM([[0., 0., 0., 1.]]),
    )

@vectorize_args
def skew(v):
    """Create skew-symmetric matrix"""
    if v.shape[0] == 1:
        return cs.vertcat(
            cs.horzcat(0., -v),
            cs.horzcat(v, 0.),
        )
    elif v.shape[0] == 3:
        return cs.vertcat(
            cs.horzcat(    0., -v[2],  v[1]),
            cs.horzcat( v[2],     0., -v[0]),
            cs.horzcat(-v[1],   v[0],    0.),
        )
    else:
        raise ValueError(f"expecting a scalar or 3-vector")

@vectorize_args
def skewa(v):
    """Create augmented skew-symmetric matrix"""
    if v.shape[0] == 3:
        return cs.vertcat(
            cs.horzcat(skew(s[2]), s[:2]),
            cs.DM.zeros(1, 3),
        )
    elif v.shape[0] == 6:
        return cs.vertcat(
            cs.horzcat(skew(s[3:6]), s[:3]),
            cs.DM.zeros(1, 4),
        )
    else:
        raise ValueError(f"expecting a 3- or 6-vector")

@arrayify_args
def t2r(T):
    """Rotational submatrix"""
    return T[:3, :3]

@vectorize_args
def tr2angvec(T):
    """Convert rotation matrix to angle-vector form"""
    return trlog(t2r(T))

@arrayify_args
def tr2delta(T0, T1):
    """Convert SE(3) homogeneous transform to differential motion"""
    TD = cs.inv(T0) @ T1
    return cs.vertcat(
        transl(TD),
        vex(t2r(TD) - I3()),
    )

@arrayify_args
def tr2eul(R, flip=False):
    """Convert SO(3) or SE(3) matrix to Euler angles"""

    cond = cs.logic_and(
        cs.abs(R[0, 2]) < eps,
        cs.abs(R[1, 2]) < eps,
    )

    eul_true = cs.vertcat(
        0.,
        cs.atan2(R[0, 2], R[2, 2]),
        cs.atan2(R[1, 0], R[1, 1]),
    )

    eul0_false = cs.atan2(-R[1, 2], -R[0, 2]) if flip else cs.atan2(R[1, 2], R[0, 2])
    sp, cp = sin(eul0_false), cos(eul0_false)

    eul_false = cs.vertcat(
        eul0_false,
        cs.atan2(cp*R[0,2] + sp*R[1,2], R[2,2]),
        cs.atan2(-sp * R[0,0] + cp * R[1,0], -sp*R[0,1] + cp*R[1,1]),
    )

    return cs.if_else(cond, eul_true, eul_false)

@arrayify_args
def tr2jac(T, samebody=False):
    """Jacobian for differential motion"""
    R = t2r(T)
    if samebody:
        return cs.vertcat(
            cs.horzcat(R.T, (skew(transl(T))@R).T),
            cs.horzcat(cs.DM.zeros(3, 3), R.T),
        )
    else:
        return cs.vertcat(
            cs.horzcat(R.T, cs.DM.zeros(3, 3)),
            cs.horzcat(cs.DM.zeros(3, 3), R.T),
        )

@arrayify_args
def tr2rpy(T):
    """Convert SO(3) or SE(3) matrix to roll-pitch-yaw angles"""
    raise NotImplementedError()

@arrayify_args
def tr2rt(T):
    """Convert homogeneous transform to rotation and translation"""
    return t2r(T), transl(T)

@arrayify_args
def transl(T):
    """SE(3) translational homogeneous transform"""
    return T[:3, 3]

@arrayify_args
def transl2(T):
    """SE(2) translational homogeneous transform"""
    return T[:2, 2]

@vectorize_args
def trexp(S, theta):
    """Matrix exponential for so(3) and se(3)"""
    raise NotImplementedError()

@arrayify_args
def trlog(R):
    """Logarithm of SO(3) or SE(3) matrix"""
    theta = cs.acos(0.5*(cs.trace(R) - 1.))
    return theta, vex((R-R.T)/2./sin(theta))

@arrayify_args
def trotx(theta):
    """SE(3) rotation about X axis"""
    return r2t(rotx(theta))

@arrayify_args
def troty(theta):
    """SE(3) rotation about Y axis"""
    return r2t(roty(theta))

@arrayify_args
def trotz(theta):
    """SE(3) rotation about Z axis"""
    return r2t(rotz(theta))

@vectorize_args
def unit(v):
    """Unitize a vector"""
    return v/cs.norm_fro(v)

@arrayify_args
def vex(S):
    if is_2x2(S):
        return 0.5*(S[1,0]-S[0,1])
    elif is_3x3(S):
        return 0.5*cs.vertcat(S[2,1]-S[1,2], S[0,2]-S[2,0], S[1,0]-S[0,1])
    else:
        raise ValueError(f'input must be a 2-by-2 or 3-by-3 matrix, not {S.shape[0]}-by-{S.shape[1]}')


@arrayify_args
def vexa(S):
    """Convert augmented skew-symmetric matrix to vector"""
    if S.shape == [3, 3]:
        return cs.vertcat(transl(S), vex(S[:3, :3]))
    elif S.shape == [4, 4]:
        return cs.vertcat(trasnl2(S), vex(S[:2, :2]))
    else:
        raise ValueError("input must be a 3-by-3 or 4-by-4 matrix")

class Quaternion:

    def __init__(self, x, y, z, w):
        self._q = cs.vertcat(x, y, z, w)

    def split(self):
        return cs.vertsplit(self._q)

    def __mul__(self, quat):
        assert isinstance(quat, Quaternion), "unsupported type"
        x0, y0, z0, w0 = self.split()
        x1, y1, z1, w1 = quat.split()
        return Quaternion(
            x1*w0 + y1*z0 - z1*y0 + w1*x0,
            -x1*z0 + y1*w0 + z1*x0 + w1*y0,
            x1*y0 - y1*x0 + z1*w0 + w1*z0,
            -x1*x0 - y1*y0 - z1*z0 + w1*w0
        )

    @staticmethod
    def fromrpy(rpy):
        r, p, y = cs.vertsplit(vec(rpy))
        cr, sr = cs.cos(0.5*r), cs.sin(0.5*r)
        cp, sp = cs.cos(0.5*p), cs.sin(0.5*p)
        cy, sy = cs.cos(0.5*y), cs.sin(0.5*y)

        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        w = cr * cp * cy + sr * sp * sy

        n = cs.sqrt(x*x + y*y + z*z + w*w)
        return Quaternion(x/n, y/n, z/n, w/n)

    @staticmethod
    def fromangvec(theta, v):
        w = cos(0.5*theta)
        xyz = sin(0.5*theta)*unit(vec(v))
        x, y, z = cs.vertsplit(xyz)
        return Quaternion(x, y, z, w)

    def getquat(self):
        return self._q

    def getrpy(self):
        qx, qy, qz, qw = self.split()

        sinr_cosp = 2.*(qw * qx + qy * qz)
        cosr_cosp = 1. - 2. * (qx * qx + qy * qy)
        roll = cs.atan2(sinr_cosp, cosr_cosp)

        sinp = 2. * (qw * qy - qz * qx)

        pitch = cs.if_else(cs.fabs(sinp) >= 1., cs.np.pi/2., cs.asin(sinp))

        siny_cosp = 2. * (qw * qz + qx * qy)
        cosy_cosp = 1. - 2. * (qy * qy + qz * qz)

        yaw = cs.atan2(siny_cosp, cosy_cosp)

        return cs.vertcat(roll, pitch, yaw)

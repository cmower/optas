import pyinvk
import pyinvk.spatialmath as sm
pyinvk.cs.np.set_printoptions(precision=3, suppress=True)


pi = pyinvk.cs.np.pi

thetar = pyinvk.cs.np.random.uniform(-pi, pi)  # random angle
print(f"{thetar = :.3f}", end='\n\n')

vr = pyinvk.cs.np.random.rand(3)  # random vector
vr /= pyinvk.cs.np.linalg.norm(vr)  # normalize

print(f"{vr = }", end='\n\n')

Rr = sm.angvec2r(thetar, vr) #  random rotation matrix
Rr = Rr.toarray()

print(f"Rr =\n{Rr}", end='\n\n')

tr = pyinvk.cs.np.random.rand(3)  # random translation

print(f"{tr = }", end='\n\n')

Tr = sm.rt2tr(Rr, tr)  # parse rotation matrix + translation to homogenous transform matrix
Tr = Tr.toarray()

print(f"Tr =\n{Tr}", end='\n\n')

rpy = pyinvk.cs.np.random.uniform(-pi, pi, size=(3,))
Qr = sm.Quaternion.fromrpy(rpy)  # random quaternion
qr = Qr.getquat().toarray().flatten()

print(f"{qr = }")

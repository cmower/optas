import time
import pybullet as p
import pybullet_data
import numpy as np

class PyBullet:

    def __init__(self, dt, add_floor=True):
        self.client_id = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        p.setGravity(gravX=0., gravY=0., gravZ=-9.81)
        p.setTimeStep(dt)
        p.configureDebugVisualizer(flag=p.COV_ENABLE_GUI, enable=0)
        p.resetDebugVisualizerCamera(
            cameraDistance=1.5,
            cameraYaw=45,
            cameraPitch=-40,
            cameraTargetPosition=[0, 0, 0.5],
        )
        if add_floor:
            self.add_floor()

    def add_floor(self):
        colid = p.createCollisionShape(p.GEOM_PLANE)
        visid = p.createVisualShape(p.GEOM_PLANE, rgbaColor=[0, 1, 0, 1.], planeNormal=[0, 0, 1])
        p.createMultiBody(baseMass=0.0, basePosition=[0.]*3,baseCollisionShapeIndex=colid, baseVisualShapeIndex=visid)

    def start(self):
        p.setRealTimeSimulation(1)

    def stop(self):
        p.setRealTimeSimulation(0)

    def close(self):
        p.disconnect(self.client_id)

class Kuka:

    def __init__(self, base_position=[0.]*3):
        self._id = p.loadURDF(fileName="robots/kuka_lwr.urdf", useFixedBase=1, basePosition=base_position)
        self.num_joints = p.getNumJoints(self._id)
        self._actuated_joints = []
        for j in range(self.num_joints):
            info = p.getJointInfo(self._id, j)
            if info[2] in {p.JOINT_REVOLUTE, p.JOINT_PRISMATIC}:
                self._actuated_joints.append(j)
        self.ndof = len(self._actuated_joints)

    def reset(self, q):
        for j, idx in enumerate(self._actuated_joints):
            qj = q[j]
            p.resetJointState(self._id, idx, qj)

    def cmd(self, q):
        p.setJointMotorControlArray(
            self._id,
            self._actuated_joints,
            p.POSITION_CONTROL,
            targetPositions=np.asarray(q).tolist(),
        )


def main():

    hz = 250
    dt = 1.0/float(hz)
    pb = PyBullet(dt)
    kuka = Kuka()

    q0 = np.zeros(7)
    qF = np.random.uniform(-np.pi, np.pi, size=(7,))

    alpha = 0.

    pb.start()

    while alpha < 1.:
        q = (1.-alpha)*q0 + alpha*qF
        kuka.cmd(q)
        time.sleep(dt)
        alpha += 0.05*dt

    pb.stop()
    pb.close()

if __name__ == '__main__':
    main()

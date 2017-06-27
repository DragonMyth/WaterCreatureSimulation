import numpy as np


class BaseCreatureController(object):
    """
    Add control forces to the back fin to one skeleton
    """

    def __init__(self, skel,dt):
        self.skel = skel

        # The first joint(root joint) is for the model to move. Not considered as a controllable dof

        num_of_dofs = len(self.skel.dofs) - len(self.skel.joints[0].dofs)
        self.dt = dt
        self.joint_max = np.zeros(num_of_dofs)
        self.joint_min = np.zeros(num_of_dofs)
        self.phi = np.zeros(num_of_dofs)
        self.omega = np.ones(num_of_dofs)
        # self.Kp = np.zeros(num_of_dofs)
        # self.Kd = np.zeros(num_of_dofs)
        self.Kp = np.diagflat([0.0] * len(self.skel.joints[0].dofs) + [400.0] * num_of_dofs)
        self.Kd = np.diagflat([0.0] * len(self.skel.joints[0].dofs) + [15.0] * num_of_dofs)
        self.invM = np.linalg.inv(skel.M+self.Kd*self.dt)

    def pd_controller_target_compute(self, t):
        amplitude = (self.joint_max - self.joint_min) * 0.5
        target_pos = amplitude * np.sin(self.omega * t + self.phi) + (self.joint_max - amplitude)
        return np.concatenate(([0.0]*len(self.skel.joints[0].dofs),target_pos))

    def compute(self, ):
        skel = self.skel
        time_step = skel.world.time()
        target_pos = self.pd_controller_target_compute(time_step+self.dt)

        ##SPD Controller
        invM = self.invM
        p = -self.Kp.dot(skel.q+skel.dq*self.dt-target_pos)
        d = -self.Kd.dot(skel.dq)
        qddot = invM.dot(-skel.c+p+d+skel.constraint_forces())
        tau = p+d-self.Kd.dot(qddot)*self.dt
        tau[0:len(self.skel.joints[0].dofs)] = 0
        return tau


class CaudalFinFishController(BaseCreatureController):
    def __init__(self, skel,dt):
        BaseCreatureController.__init__(self, skel,dt)
        self.omega[:] = 25


class PectoralFinFishController(BaseCreatureController):
    def __init__(self, skel,dt):
        BaseCreatureController.__init__(self, skel,dt)
        self.omega[:] = 5


class EelController(BaseCreatureController):
    def __init__(self, skel,dt):
        BaseCreatureController.__init__(self, skel,dt)
        self.omega[:] = 25
class FlatCreatureController(BaseCreatureController):
    def __init__(self, skel,dt):
        BaseCreatureController.__init__(self, skel,dt)
        self.omega[:] = 25
class ConstraintTestModelController(BaseCreatureController):
    def __init__(self, skel,dt):
        BaseCreatureController.__init__(self, skel,dt)
        self.omega[:] = 25

class TurtleController(BaseCreatureController):
    def __init__(self, skel,dt):
        BaseCreatureController.__init__(self, skel,dt)
        self.omega[:] = 25

    def pd_controller_target_compute(self, t):
        amplitude = (self.joint_max - self.joint_min) * 0.5
        target_pos = amplitude * np.sin(self.omega * t + self.phi) + (self.joint_max - amplitude)

        #Mirror the target pos to the other side
        target_pos[4] = -target_pos[0]
        target_pos[5] = target_pos[1]
        target_pos[6] = -target_pos[2]
        target_pos[7] = target_pos[3]

        return np.concatenate(([0.0]*len(self.skel.joints[0].dofs),target_pos))
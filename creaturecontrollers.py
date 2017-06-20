import numpy as np


class BaseCreatureController(object):
    """
    Add control forces to the back fin to one skeleton
    """

    def __init__(self, skel):
        self.skel = skel

        # The first joint(root joint) is for the model to move. Not considered as a controllable dof

        num_of_dofs = len(self.skel.dofs) - len(self.skel.joints[0].dofs)
        self.joint_max = np.zeros(num_of_dofs)
        self.joint_min = np.zeros(num_of_dofs)
        self.phi = np.zeros(num_of_dofs)
        self.omega = np.ones(num_of_dofs)
        self.Kp = np.zeros(num_of_dofs)
        self.Kd = np.zeros(num_of_dofs)

    def pd_controller_target_compute(self, t):
        amplitude = (self.joint_max - self.joint_min) * 0.5
        target_pos = amplitude * np.sin(self.omega * t + self.phi) + (self.joint_max - amplitude)
        return target_pos

    def compute(self, ):
        time_step = self.skel.world.time()
        root_dofs_num = len(self.skel.joints[0].dofs)
        target_pos = self.pd_controller_target_compute(time_step)
        curr_pos = self.skel.q[root_dofs_num:]
        curr_velocity = self.skel.dq[root_dofs_num:]

        tau = np.zeros(len(self.skel.dofs))
        tau[root_dofs_num::] = self.Kp * (target_pos - curr_pos) - self.Kd * curr_velocity

        return tau


class CaudalFinFishController(BaseCreatureController):
    def __init__(self, skel):
        BaseCreatureController.__init__(self, skel)
        self.Kp[:] = 5
        self.Kd[:] = 0.0005
        self.omega[:] = 25


class PectoralFinFishController(BaseCreatureController):
    def __init__(self, skel):
        BaseCreatureController.__init__(self, skel)
        self.Kp[:] = 5
        self.Kd[:] = 0.0005
        self.omega[:] = 5


class EelController(BaseCreatureController):
    def __init__(self, skel):
        BaseCreatureController.__init__(self, skel)
        self.Kp[:] = 0.5
        self.Kd[:] = 0.000005
        self.omega[:] = 25


class TurtleController(BaseCreatureController):
    def __init__(self, skel):
        BaseCreatureController.__init__(self, skel)
        self.Kp[:] = 0.05
        self.Kd[:] = 0.00005
        self.omega[:] = 25

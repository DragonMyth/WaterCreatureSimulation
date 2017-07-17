import numpy as np
import pydart2 as pydart
import creatureParamSettings


class BaseCreatureController(object):
    """
    Add control forces to the back fin to one skeleton
    """

    def __init__(self, skel, dt):
        self.skel = skel

        # The first joint(root joint) is for the model to move. Not considered as a controllable dof

        num_of_dofs = len(self.skel.dofs) - len(self.skel.joints[0].dofs)

        self.dt = dt
        self.joint_max = (np.random.rand(num_of_dofs)) * np.pi / 16
        self.joint_min = (np.random.rand(num_of_dofs) - 1) * np.pi / 16
        self.phi = (np.random.rand(num_of_dofs) - 0.5) * np.pi / 8
        self.omega = 25 * np.ones(num_of_dofs)

        self.Kp = np.diagflat([0.0] * len(self.skel.joints[0].dofs) + [4000000.0] * num_of_dofs)
        self.Kd = self.dt * self.Kp
        self.invM = np.linalg.inv(skel.M + self.Kd * self.dt)


        # Initialize position
        # q = (np.random.rand(skel.ndofs)-0.5) *np.pi/4
        # q[0:len(self.skel.joints[0].dofs)] = 0
        # skel.set_positions(q)

    def pd_controller_target_compute(self, t):
        amplitude = (self.joint_max - self.joint_min) * 0.5
        target_pos = amplitude * np.sin(self.omega * t + self.phi) + (self.joint_max - amplitude)
        return self.build_target_pos(target_pos)

    def build_target_pos(self, calculated_target_pos):
        return np.concatenate(([0.0] * len(self.skel.joints[0].dofs), calculated_target_pos))

    def compute(self, ):
        skel = self.skel
        time_step = skel.world.time()
        target_pos = self.pd_controller_target_compute(time_step + self.dt)

        ##SPD Controller
        invM = self.invM
        p = -self.Kp.dot(skel.q + skel.dq * self.dt - target_pos)
        d = -self.Kd.dot(skel.dq)
        qddot = invM.dot(-skel.c + p + d + skel.constraint_forces())
        tau = p + d - self.Kd.dot(qddot) * self.dt
        tau[0:len(self.skel.joints[0].dofs)] = 0
        return tau


class CaudalFinFishController(BaseCreatureController):
    def __init__(self, skel, dt):
        BaseCreatureController.__init__(self, skel, dt)
        num_of_dofs = len(self.skel.dofs) - len(self.skel.joints[0].dofs)

        # kp = world.d
        # # self.Kp = np.diagflat([0.0] * len(self.skel.joints[0].dofs) + [1000.0] * num_of_dofs)
        # # self.Kd = np.diagflat([0.0] * len(self.skel.joints[0].dofs) + [15.0] * num_of_dofs)


class PectoralFinFishController(BaseCreatureController):
    def __init__(self, skel, dt):
        BaseCreatureController.__init__(self, skel, dt)


class EelController(BaseCreatureController):
    def __init__(self, skel, dt):
        BaseCreatureController.__init__(self, skel, dt)


class FlatCreatureController(BaseCreatureController):
    def __init__(self, skel, dt):
        BaseCreatureController.__init__(self, skel, dt)
        num_of_dofs = len(self.skel.dofs) - len(self.skel.joints[0].dofs)

        # self.Kp = np.diagflat([0.0] * len(self.skel.joints[0].dofs) + [1000.0] * num_of_dofs)
        # self.Kd = np.diagflat([0.0] * len(self.skel.joints[0].dofs) + [15.0] * num_of_dofs)


class TurtleController(BaseCreatureController):
    def __init__(self, skel, dt):
        BaseCreatureController.__init__(self, skel, dt)

        effective_len = 4
        num_of_dofs = len(self.skel.dofs) - len(
            self.skel.joints[0].dofs)  # 2 for spine 4 for wings of first spine segment
        self.dt = dt
        self.joint_max = (np.random.rand(effective_len)) * np.pi / 4
        self.joint_min = (np.random.rand(effective_len) - 1) * np.pi / 4
        self.phi = (np.random.rand(effective_len) - 0.5) * np.pi / 2
        self.omega = 25 * np.ones(effective_len)

        self.Kp = np.diagflat([0.0] * len(self.skel.joints[0].dofs) + [400000.0] * num_of_dofs)
        self.Kd = self.dt * self.Kp
        self.invM = np.linalg.inv(skel.M + self.Kd * self.dt)

    def build_target_pos(self, calculated_target_pos):
        num_of_dofs = len(self.skel.dofs) - len(
            self.skel.joints[0].dofs)
        res = np.zeros(num_of_dofs)

        res[0:2] = calculated_target_pos[0:2]
        res[2:4] = calculated_target_pos[2::]

        res[4] = -res[0]
        res[5] = res[1]
        res[6] = -res[2]
        res[7] = res[3]
        res = np.concatenate(([0.0] * len(self.skel.joints[0].dofs), res))
        return res

class TurtleCircController(BaseCreatureController):
    def __init__(self, skel, dt):
        BaseCreatureController.__init__(self, skel, dt)

        #Mirror front leg but free back leg
        effective_len = 6
        num_of_dofs = len(self.skel.dofs) - len(
            self.skel.joints[0].dofs)  # 2 for spine 4 for wings of first spine segment
        self.dt = dt
        self.joint_max = (np.random.rand(effective_len)) * np.pi / 4
        self.joint_min = (np.random.rand(effective_len) - 1) * np.pi / 4
        self.phi = (np.random.rand(effective_len) - 0.5) * np.pi / 2
        self.omega = 25 * np.ones(effective_len)

        self.Kp = np.diagflat([0.0] * len(self.skel.joints[0].dofs) + [400000.0] * num_of_dofs)
        self.Kd = self.dt * self.Kp
        self.invM = np.linalg.inv(skel.M + self.Kd * self.dt)

    def build_target_pos(self, calculated_target_pos):
        num_of_dofs = len(self.skel.dofs) - len(
            self.skel.joints[0].dofs)
        res = np.zeros(num_of_dofs)

        res[0:2] = calculated_target_pos[0:2]
        res[2:4] = calculated_target_pos[2:4]
        res[6:8] = calculated_target_pos[4:6]

        res[4] = -res[0]
        res[5] = res[1]
        res = np.concatenate(([0.0] * len(self.skel.joints[0].dofs), res))
        return res

class SimplifiedFlatwormController(BaseCreatureController):
    def __init__(self, skel, dt):
        BaseCreatureController.__init__(self, skel, dt)
        effective_len = 6
        num_of_dofs = len(self.skel.dofs) - len(
            self.skel.joints[0].dofs)  # 2 for spine 4 for wings of first spine segment
        self.dt = dt
        self.joint_max = (np.random.rand(effective_len)) * np.pi / 4
        self.joint_min = (np.random.rand(effective_len) - 1) * np.pi / 4
        self.phi = (np.random.rand(effective_len) - 0.5) * np.pi / 2
        self.omega = 25 * np.ones(effective_len)

        self.Kp = np.diagflat([0.0] * len(self.skel.joints[0].dofs) + [400000.0] * num_of_dofs)
        self.Kd = self.dt * self.Kp
        self.invM = np.linalg.inv(skel.M + self.Kd * self.dt)

    def build_target_pos(self, calculated_target_pos):
        num_of_dofs = len(self.skel.dofs) - len(
            self.skel.joints[0].dofs)
        res = np.zeros(num_of_dofs)

        res[2:4] = calculated_target_pos[2:4]
        res[10:12] = calculated_target_pos[4::]

        res[4] = -res[2]
        res[5] = res[3]
        res[12] = -res[10]
        res[13] = res[11]
        res = np.concatenate(([0.0] * len(self.skel.joints[0].dofs), res))
        return res


class FlatwormController(BaseCreatureController):
    def __init__(self, skel, dt):
        BaseCreatureController.__init__(self, skel, dt)
        num_of_dofs = len(self.skel.dofs) - len(
            self.skel.joints[0].dofs)  # 2 for spine 4 for wings of first spine segment
        effective_len = 18
        self.dt = dt
        self.joint_max = (np.random.rand(effective_len)) * np.pi / 8
        self.joint_min = (np.random.rand(effective_len) - 1) * np.pi / 8
        self.phi = (np.random.rand(effective_len) - 0.5) * np.pi / 4
        self.omega = 25 * np.ones(effective_len)

        self.Kp = np.diagflat([0.0] * len(self.skel.joints[0].dofs) + [400000.0] * num_of_dofs)
        self.Kd = self.dt * self.Kp
        self.invM = np.linalg.inv(skel.M + self.Kd * self.dt)

    def build_target_pos(self, calculated_target_pos):
        num_of_dofs = len(self.skel.dofs) - len(
            self.skel.joints[0].dofs)
        res = np.zeros(num_of_dofs)

        res[5:11] = calculated_target_pos[0:6]
        res[29:35] = calculated_target_pos[6:12]
        res[53:59] = calculated_target_pos[12:18]
        # Mirroring the first row of wing segments
        res[11:17:2] = -res[5:11:2]
        res[12:18:2] = res[6:12:2]

        res[35:41:2] = -res[29:35:2]
        res[36:42:2] = res[30:36:2]

        res[59:65:2] = -res[53:59:2]
        res[60:66:2] = res[54:60:2]

        res = np.concatenate(([0.0] * len(self.skel.joints[0].dofs), res))
        return res


class FlatwormHardBackboneController(FlatwormController):
    def build_target_pos(self, calculated_target_pos):
        num_of_dofs = len(self.skel.dofs) - len(
            self.skel.joints[0].dofs)
        res = np.zeros(num_of_dofs)

        res[0:6] = calculated_target_pos[0:6]
        res[24:30] = calculated_target_pos[6:12]
        res[48:54] = calculated_target_pos[12:18]
        # Mirroring the first row of wing segments
        res[6:11:2] = -res[0:6:2]
        res[7:13:2] = res[1:7:2]

        res[30:36:2] = -res[24:30:2]
        res[31:37:2] = res[25:31:2]

        res[54:60:2] = -res[48:54:2]
        res[55:61:2] = res[59:65:2]

        res = np.concatenate(([0.0] * len(self.skel.joints[0].dofs), res))
        return res

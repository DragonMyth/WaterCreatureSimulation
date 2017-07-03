import numpy as np
import pydart2 as pydart
import creatureParamSettings
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
        self.omega = 15*np.ones(num_of_dofs)

        self.Kp = np.diagflat([0.0] * len(self.skel.joints[0].dofs) + [400000.0] * num_of_dofs)
        self.Kd = self.dt*self.Kp
        self.invM = np.linalg.inv(skel.M+self.Kd*self.dt)


        #Initialize position
        # q = (np.random.rand(skel.ndofs)-0.5) *np.pi/4
        # q[0:len(self.skel.joints[0].dofs)] = 0
        # skel.set_positions(q)


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
    def __init__(self, skel,dt,world):
        BaseCreatureController.__init__(self, skel,dt)
        num_of_dofs = len(self.skel.dofs) - len(self.skel.joints[0].dofs)

        # kp = world.d
        # # self.Kp = np.diagflat([0.0] * len(self.skel.joints[0].dofs) + [1000.0] * num_of_dofs)
        # # self.Kd = np.diagflat([0.0] * len(self.skel.joints[0].dofs) + [15.0] * num_of_dofs)


class PectoralFinFishController(BaseCreatureController):
    def __init__(self, skel,dt,world):
        BaseCreatureController.__init__(self, skel,dt)


class EelController(BaseCreatureController):
    def __init__(self, skel,dt,world):
        BaseCreatureController.__init__(self, skel,dt)
class FlatCreatureController(BaseCreatureController):
    def __init__(self, skel,dt,world):
        BaseCreatureController.__init__(self, skel,dt)
        num_of_dofs = len(self.skel.dofs) - len(self.skel.joints[0].dofs)

        # self.Kp = np.diagflat([0.0] * len(self.skel.joints[0].dofs) + [1000.0] * num_of_dofs)
        # self.Kd = np.diagflat([0.0] * len(self.skel.joints[0].dofs) + [15.0] * num_of_dofs)
class ConstraintTestModelController(BaseCreatureController):
    def __init__(self, skel,dt,world):
        BaseCreatureController.__init__(self, skel,dt)

        # q = skel.q
        # q[8]= 0.2
        # q[9] = -0.2
        # skel.set_positions(q)
        self.setConstrains(skel,world)

        creatureParamSettings.loopCreatureParam(self)
    def compute(self, ):
        tau = BaseCreatureController.compute(self)
        tau[:] = 0

        return tau
    def setConstrains(self,skel,world):
        bd1 = skel.bodynodes[3]
        bd2 = skel.bodynodes[4]
        jointPos = 0.5*(bd1.C+bd2.C)#np.array([-0.21,0,0])
        joint = pydart.constraints.BallJointConstraint(bd1,bd2,jointPos)
        # joint = pydart.constraints.WeldJointConstraint(bd1,bd2)
        joint.add_to_world(world)
class TurtleController(BaseCreatureController):
    def __init__(self, skel,dt,world):
        BaseCreatureController.__init__(self, skel,dt)

    def pd_controller_target_compute(self, t):
        amplitude = (self.joint_max - self.joint_min) * 0.5
        target_pos = amplitude * np.sin(self.omega * t + self.phi) + (self.joint_max - amplitude)

        #Mirror the target pos to the other side
        target_pos[4] = -target_pos[0]
        target_pos[5] = target_pos[1]
        target_pos[6] = -target_pos[2]
        target_pos[7] = target_pos[3]

        return np.concatenate(([0.0]*len(self.skel.joints[0].dofs),target_pos))
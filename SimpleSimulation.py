import pydart2 as pydart
import numpy as np
import scipy as sp

boxNormals = {
    'up': [0, 1, 0],
    'down': [0, -1, 0],
    'inward': [0, 0, -1],
    'outward': [0, 0, 1],
    'left': [-1, 0, 0],
    'right': [1, 0, 0]
}

class SimpleWaterCreatureController(object):
    """
    Add control forces to the back fin
    """

    def __init__(self, skel):
        self.skel = skel
        # Set all target positions to zero
        num_of_dofs = len(self.skel.dofs)
        self.joint_amplitudes =np.zeros(num_of_dofs - 1)
        self.omega = np.ones(num_of_dofs - 1)
        self.phi = np.zeros(num_of_dofs - 1)
        # Randomly Initialize K's between 0 to 10
        self.Kp = np.zeros(num_of_dofs - 1)
        self.Kd = np.zeros(num_of_dofs - 1)

    def computeTargetPos(self, t):
        target_pos = self.joint_amplitudes * np.sin(self.omega*t+self.phi)
        return target_pos

    def compute(self, ):
        time_step = self.skel.world.time()
        target_pos = self.computeTargetPos(time_step)
        curr_pos = self.skel.q[1]
        curr_velocity = self.skel.dq[1]
        tau = np.zeros(len(self.skel.dofs))
        tau[1::] = self.Kp * (target_pos - curr_pos) - self.Kd * curr_velocity
        return tau


class MyWorld(pydart.World):
    def __init__(self, ):
        pydart.World.__init__(self, 1.0 / 2000.0, './simpleSwimingCreature.skel')
        self.robot = self.skeletons[0]
        self.duration = 0
        self.controller = SimpleWaterCreatureController(self.robot)

        ##Set values to the params in the controller
        self.controller.joint_amplitudes[0] = np.pi / 4
        self.controller.Kp[0] = 10
        self.controller.Kd[0] = 0.5
        self.controller.omega[0] = 1.2

        self.robot.set_controller(self.controller)
        print(self.robot.num_dofs())

    def step(self, ):
        for i in range(len(self.robot.bodynodes)):
            force = 10 * self.calcFluidForce(self.robot.bodynodes[i])
            # print(force)
            self.robot.bodynodes[i].add_ext_force(force)
        super(MyWorld, self).step()

    def calcFluidForce(self, bodynode):
        dq = self.robot.dq

        shape = bodynode.shapenodes[0]
        worldCenterPoint = bodynode.to_world([0, 0, 0])
        bodyGeomety = shape.shape.size()
        J_out = bodynode.linear_jacobian(offset=[0, 0, bodyGeomety[2] / 2])
        node_linear_velocity_out = J_out.dot(dq)
        world_out_normal = bodynode.to_world(boxNormals['outward']) - worldCenterPoint
        unit_force_out = world_out_normal.dot(node_linear_velocity_out)
        unit_force_out = 0 if unit_force_out < 0 else unit_force_out
        force_out = -bodyGeomety[0] * bodyGeomety[1] * unit_force_out * world_out_normal

        J_in = bodynode.linear_jacobian(offset=[0, 0, -bodyGeomety[2] / 2])
        node_linear_velocity_in = J_in.dot(dq)
        world_in_normal = bodynode.to_world(boxNormals['inward']) - worldCenterPoint
        unit_force_in = world_in_normal.dot(node_linear_velocity_in)
        unit_force_in = 0 if unit_force_in < 0 else unit_force_in
        force_in = -bodyGeomety[0] * bodyGeomety[1] * unit_force_in * world_in_normal

        J_up = bodynode.linear_jacobian(offset=[0, bodyGeomety[1] / 2, 0])
        node_linear_velocity_up = J_up.dot(dq)
        world_up_normal = bodynode.to_world(boxNormals['up']) - worldCenterPoint
        unit_force_up = world_up_normal.dot(node_linear_velocity_up)
        unit_force_up = 0 if unit_force_up < 0 else unit_force_up
        force_up = -bodyGeomety[0] * bodyGeomety[2] * unit_force_up * world_up_normal

        J_down = bodynode.linear_jacobian(offset=[0, -bodyGeomety[1] / 2, 0])
        node_linear_velocity_down = J_down.dot(dq)
        world_down_normal = bodynode.to_world(boxNormals['down']) - worldCenterPoint
        unit_force_down = world_down_normal.dot(node_linear_velocity_down)
        unit_force_down = 0 if unit_force_down < 0 else unit_force_down
        force_down = -bodyGeomety[0] * bodyGeomety[2] * unit_force_down * world_down_normal

        J_left = bodynode.linear_jacobian(offset=[-bodyGeomety[0] / 2, 0, 0])
        node_linear_velocity_left = J_left.dot(dq)
        world_left_normal = bodynode.to_world(boxNormals['left']) - worldCenterPoint
        unit_force_left = world_left_normal.dot(node_linear_velocity_left)
        unit_force_left = 0 if unit_force_left < 0 else unit_force_left
        force_left = -bodyGeomety[1] * bodyGeomety[2] * unit_force_left * world_left_normal

        J_right = bodynode.linear_jacobian(offset=[bodyGeomety[0] / 2, 0, 0])
        node_linear_velocity_right = J_right.dot(dq)
        world_right_normal = bodynode.to_world(boxNormals['right']) - worldCenterPoint
        unit_force_right = world_right_normal.dot(node_linear_velocity_right)
        unit_force_right = 0 if unit_force_right < 0 else unit_force_right
        force_right = -bodyGeomety[1] * bodyGeomety[2] * unit_force_right * world_right_normal

        return force_in + force_out + force_up + force_down + force_left + force_right


if __name__ == '__main__':
    pydart.init()

    world = MyWorld()
    pydart.gui.viewer.launch_pyqt5(world)

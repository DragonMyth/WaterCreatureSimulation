import pydart2 as pydart
import numpy as np
import scipy as sp
import creatureParamSettings as param_settings
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
        num_of_dofs = len(self.skel.dofs)-len(self.skel.joints[0].dofs)
        self.joint_amplitudes = np.zeros(num_of_dofs)
        self.omega = np.ones(num_of_dofs)
        self.phi = np.zeros(num_of_dofs)
        # Randomly Initialize K's between 0 to 10
        self.Kp = np.zeros(num_of_dofs)
        self.Kd = np.zeros(num_of_dofs)

    def pd_controller_target_compute(self, t):
        target_pos = self.joint_amplitudes * np.sin(self.omega * t + self.phi)
        return target_pos

    def compute(self, ):
        time_step = self.skel.world.time()
        root_dofs_num = len(self.skel.joints[0].dofs)
        target_pos = self.pd_controller_target_compute(time_step)
        print("Target Pos", target_pos)
        curr_pos = self.skel.q[root_dofs_num:]
        print("Current Pos", curr_pos)
        curr_velocity = self.skel.dq[root_dofs_num:]

        print("Current Velocity", curr_velocity)
        tau = np.zeros(len(self.skel.dofs))
        tau[root_dofs_num::] = self.Kp * (target_pos - curr_pos) - self.Kd * curr_velocity

        # if(time_step>10.1):
        #     tau[:] = 0
        print("Current Torque", tau)

        return tau


class MyWorld(pydart.World):
    def __init__(self, ):
        skel_direction = './skeletons/FishWithPectoralFins.skel'
        pydart.World.__init__(self, 1.0 / 2000.0, skel_direction)
        self.robot = self.skeletons[0]
        self.controller = SimpleWaterCreatureController(self.robot)

        ##Set values to the params in the controller
        param_settings.simpleFishParam(self.controller)
        self.robot.set_controller(self.controller)
        self.forces = np.zeros((len(self.robot.bodynodes), 3))
        print(self.robot.num_dofs())

    def step(self, ):
        for i in range(len(self.robot.bodynodes)):
            self.forces[i] = self.calcFluidForce(self.robot.bodynodes[i])
            # print(force)
            self.forces[i][1] = 0
            self.robot.bodynodes[i].add_ext_force(self.forces[i])

        super(MyWorld, self).step()

    def render_with_ri(self, ri):
        for i in range(len(self.robot.bodynodes)):
            p0 = self.robot.bodynodes[i].C
            p1 = p0 - 2 * self.forces[i]
            ri.set_color(0.0, 1.0, 0.0)
            ri.render_arrow(p1, p0, r_base=0.03, head_width=0.03, head_len=0.1)

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

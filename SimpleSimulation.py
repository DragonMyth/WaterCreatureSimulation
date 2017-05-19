import pydart2 as pydart
import numpy as np
import scipy as sp


boxNormals={
    'up':[0,1,0],
    'down':[0,-1,0],
    'inward':[0,0,-1],
    'outward':[0,0,1],
    'left':[-1,0,0],
    'right':[1,0,0]
}

class SimpleWaterCreatureController(object):
    """
    Add control forces to the back fin
    """
    def __init__(self,skel):
        self.skel = skel
    def compute(self, ):

        finPart = self.skel.bodynodes[1]
        J = finPart.linear_jacobian()
        hintchRot = self.skel.q[1]
        tau = J.transpose().dot(np.array([0,0,10])*(-2*hintchRot))
        tau[0] = 0
        return tau

class MyWorld(pydart.World):
    def __init__(self, ):
        pydart.World.__init__(self, 1.0 / 2000.0,'./simpleSwimingCreature.skel')
        self.robot = self.skeletons[0]
        self.robot.q = [-1.5,0.7]
        self.duration = 0
        self.controller = SimpleWaterCreatureController(self.robot)
        self.robot.set_controller(self.controller)
        print(self.robot.num_dofs())

    def step(self, ):

        dq = self.robot.dq
        for i in range(len(self.robot.bodynodes)):
            J = self.robot.bodynodes[i].linear_jacobian()
            linear_velocities = J.dot(dq)
            force = self.calcFluidForce(linear_velocities,self.robot.bodynodes[i])
            self.robot.bodynodes[i].add_ext_force(force)
        super(MyWorld, self).step()

    def calcFluidForce(self,node_linear_velocity,bodynode):
        shape = bodynode.shapenodes[0]

        bodyGeomety = shape.shape.size()
        world_out_normal = bodynode.to_world(boxNormals['outward'])
        unit_force_out = world_out_normal.dot(node_linear_velocity)
        unit_force_out = 0 if unit_force_out<0 else unit_force_out
        force_out = -bodyGeomety[0]*bodyGeomety[1]*unit_force_out*world_out_normal

        world_in_normal = bodynode.to_world(boxNormals['inward'])
        unit_force_in = world_in_normal.dot(node_linear_velocity)
        unit_force_in = 0 if unit_force_in<0 else unit_force_in
        force_in = -bodyGeomety[0]*bodyGeomety[1]*unit_force_in*world_in_normal

        world_up_normal = bodynode.to_world(boxNormals['up'])
        unit_force_up = world_up_normal.dot(node_linear_velocity)
        unit_force_up = 0 if unit_force_up<0 else unit_force_up
        force_up = -bodyGeomety[0]*bodyGeomety[2]*unit_force_up*world_up_normal

        world_down_normal = bodynode.to_world(boxNormals['down'])
        unit_force_down = world_down_normal.dot(node_linear_velocity)
        unit_force_down = 0 if unit_force_down<0 else unit_force_down
        force_down = -bodyGeomety[0]*bodyGeomety[2]*unit_force_down*world_down_normal

        world_left_normal = bodynode.to_world(boxNormals['left'])
        unit_force_left = world_left_normal.dot(node_linear_velocity)
        unit_force_left = 0 if unit_force_left<0 else unit_force_left
        force_left = -bodyGeomety[1]*bodyGeomety[2]*unit_force_left*world_left_normal

        world_right_normal = bodynode.to_world(boxNormals['right'])
        unit_force_right = world_right_normal.dot(node_linear_velocity)
        unit_force_right = 0 if unit_force_right<0 else unit_force_right
        force_right = -bodyGeomety[1]*bodyGeomety[2]*unit_force_right*world_right_normal

        return force_in + force_out + force_up + force_down + force_left + force_right


if __name__ == '__main__':
    pydart.init()

    world = MyWorld()
    pydart.gui.viewer.launch_pyqt5(world)
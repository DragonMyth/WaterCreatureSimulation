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
        tau = J.transpose().dot(np.array([0,0,5])*(-2*hintchRot))
        tau[0] = 0
        return tau

class MyWorld(pydart.World):
    def __init__(self, ):
        pydart.World.__init__(self, 1.0 / 2000.0,'./simpleSwimingCreature.skel')
        self.robot = self.skeletons[0]
        self.robot.q = [-1,0.5]
        self.duration = 0
        self.controller = SimpleWaterCreatureController(self.robot)
        self.robot.set_controller(self.controller)
        print(self.robot.num_dofs())
    def step(self, ):

        # for bodynode in self.robot.bodynodes:
        force = self.calcFluidForce(self.robot.bodynodes[1])
        self.robot.bodynodes[1].add_ext_force(force)
        super(MyWorld, self).step()

    def calcFluidForce(self,bodynode):
        shape = bodynode.shapenodes[0]

        bodyGeomety = shape.shape.size()
        print("Linear Velocity",bodynode.com_linear_velocity())
        print("Spatial Velocity",bodynode.com_spatial_velocity())

        velocity = bodynode.com_linear_velocity()+(bodynode.com_spatial_velocity()[1]*bodyGeomety[0]/2)*bodynode.to_world(boxNormals['outward'])
        unitForce = bodynode.to_world(boxNormals['outward']).dot(velocity)
        print(unitForce)
        unitForce = 0 if unitForce<0 else unitForce
        force1 = -bodyGeomety[0]*bodyGeomety[1]*unitForce*bodynode.to_world(boxNormals['outward'])
        velocity = bodynode.com_linear_velocity()+(bodynode.com_spatial_velocity()[1]*bodyGeomety[0]/2)*bodynode.to_world(boxNormals['inward'])
        unitForce = bodynode.to_world(boxNormals['inward']).dot(velocity)
        unitForce = 0 if unitForce < 0 else unitForce
        force2 = -bodyGeomety[0]*bodyGeomety[1]*unitForce*bodynode.to_world(boxNormals['inward'])

        return force1+force2
        # print(shape.size())



if __name__ == '__main__':
    pydart.init()

    world = MyWorld()
    pydart.gui.viewer.launch_pyqt5(world)
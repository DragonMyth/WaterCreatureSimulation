import pydart2 as pydart
import numpy as np
import scipy as sp





class SimpleWaterCreatureController(object):
    """
    Add control forces to the back fin
    """
    def __init__(self,skel):
        self.skel = skel
    def compute(self, ):

        finPart = self.skel.bodynodes[1]
        J = finPart.linear_jacobian()
        tau = J.transpose().dot(np.array([-0.5,0,0]))

        #Set main body part force =0
        # tau[0] = 0
        return tau


class MyWorld(pydart.World):
    def __init__(self, ):
        pydart.World.__init__(self, 1.0 / 2000.0,'./simpleSwimingCreature.skel')
        self.robot = self.skeletons[0]
        self.robot.q = [1,0.5]
        self.duration = 0
        self.controller = SimpleWaterCreatureController(self.robot)
        self.robot.set_controller(self.controller)
        print(self.robot.num_dofs())
    def step(self, ):


        print('[Position]')
        print('\tpositions = %s' % str(self.robot.q))
        print('\tvelocities = %s' % str(self.robot.dq))
        print('\tstates = %s' % str(self.robot.x))
        # for body in self.robot.bodynodes:
            # force = self.calcFluidForce(body)
        super(MyWorld, self).step()
    def calcFluidForce(self,body):
        shape = body.shapenodes[0]

        normalOut = np.array([0,0,1])
        rotation = shape.getRelativeRotation()

        # print(shape.size())



if __name__ == '__main__':
    pydart.init()

    world = MyWorld()
    pydart.gui.viewer.launch_pyqt5(world)
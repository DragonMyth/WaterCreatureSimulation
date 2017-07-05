import pydart2 as pydart
import numpy as np
import creature_controllers
import creatureParamSettings
import jointconstraints as constraints
from sklearn.preprocessing import normalize

BOXNORMALS = {
    'up': [0, 1, 0],
    'down': [0, -1, 0],
    'inward': [0, 0, -1],
    'outward': [0, 0, 1],
    'left': [-1, 0, 0],
    'right': [1, 0, 0]
}


class BaseFluidSimulator(pydart.World):
    def __init__(self, skel_path):
        super(BaseFluidSimulator, self).__init__(1.0 / 2000.0, skel_path=skel_path)
        self.forces = np.zeros((len(self.skeletons[0].bodynodes), 3))
        self.debug = False

    def step(self, ):
        for i in range(len(self.skeletons[0].bodynodes)):
            self.forces[i] = self.calcFluidForce(self.skeletons[0].bodynodes[i])
            self.skeletons[0].bodynodes[i].add_ext_force(self.forces[i])
        super(BaseFluidSimulator, self).step()

    def calcFluidForce(self, bodynode):
        dq = self.skeletons[0].dq

        shape = bodynode.shapenodes[0]
        worldCenterPoint = bodynode.to_world([0, 0, 0])
        bodyGeomety = shape.shape.size()
        J_out = bodynode.linear_jacobian(offset=[0, 0, bodyGeomety[2] / 2])
        node_linear_velocity_out = J_out.dot(dq)
        world_out_normal = bodynode.to_world(BOXNORMALS['outward']) - worldCenterPoint
        unit_force_out = world_out_normal.dot(node_linear_velocity_out)
        unit_force_out = 0 if unit_force_out < 0 else unit_force_out
        force_out = -bodyGeomety[0] * bodyGeomety[1] * unit_force_out * world_out_normal

        J_in = bodynode.linear_jacobian(offset=[0, 0, -bodyGeomety[2] / 2])
        node_linear_velocity_in = J_in.dot(dq)
        world_in_normal = bodynode.to_world(BOXNORMALS['inward']) - worldCenterPoint
        unit_force_in = world_in_normal.dot(node_linear_velocity_in)
        unit_force_in = 0 if unit_force_in < 0 else unit_force_in
        force_in = -bodyGeomety[0] * bodyGeomety[1] * unit_force_in * world_in_normal

        J_up = bodynode.linear_jacobian(offset=[0, bodyGeomety[1] / 2, 0])
        node_linear_velocity_up = J_up.dot(dq)
        world_up_normal = bodynode.to_world(BOXNORMALS['up']) - worldCenterPoint
        unit_force_up = world_up_normal.dot(node_linear_velocity_up)
        unit_force_up = 0 if unit_force_up < 0 else unit_force_up
        force_up = -bodyGeomety[0] * bodyGeomety[2] * unit_force_up * world_up_normal

        J_down = bodynode.linear_jacobian(offset=[0, -bodyGeomety[1] / 2, 0])
        node_linear_velocity_down = J_down.dot(dq)
        world_down_normal = bodynode.to_world(BOXNORMALS['down']) - worldCenterPoint
        unit_force_down = world_down_normal.dot(node_linear_velocity_down)
        unit_force_down = 0 if unit_force_down < 0 else unit_force_down
        force_down = -bodyGeomety[0] * bodyGeomety[2] * unit_force_down * world_down_normal

        J_left = bodynode.linear_jacobian(offset=[-bodyGeomety[0] / 2, 0, 0])
        node_linear_velocity_left = J_left.dot(dq)
        world_left_normal = bodynode.to_world(BOXNORMALS['left']) - worldCenterPoint
        unit_force_left = world_left_normal.dot(node_linear_velocity_left)
        unit_force_left = 0 if unit_force_left < 0 else unit_force_left
        force_left = -bodyGeomety[1] * bodyGeomety[2] * unit_force_left * world_left_normal

        J_right = bodynode.linear_jacobian(offset=[bodyGeomety[0] / 2, 0, 0])
        node_linear_velocity_right = J_right.dot(dq)
        world_right_normal = bodynode.to_world(BOXNORMALS['right']) - worldCenterPoint
        unit_force_right = world_right_normal.dot(node_linear_velocity_right)
        unit_force_right = 0 if unit_force_right < 0 else unit_force_right
        force_right = -bodyGeomety[1] * bodyGeomety[2] * unit_force_right * world_right_normal

        return force_in + force_out + force_up + force_down + force_left + force_right

    def render_with_ri(self, ri):
        ri.set_color(0.2, 1.0, 0.6)
        if self.debug:
            for i in range(len(self.skeletons[0].bodynodes)):
                p0 = self.skeletons[0].bodynodes[i].C
                p1 = p0 - 2 * self.forces[i]
                ri.set_color(0.0, 1.0, 0.0)
                ri.render_arrow(p1, p0, r_base=0.03, head_width=0.03, head_len=0.1)


class SimpleFishWithCaudalFinSimulator(BaseFluidSimulator):
    def __init__(self, ):
        BaseFluidSimulator.__init__(self, './skeletons/SimpleFishWithCaudalFin.skel')
        # dt is the time difference. Here we set it to 1/100 s
        self.controller = creature_controllers.CaudalFinFishController(self.skeletons[0], 1.0 / 100)
        self.skeletons[0].set_controller(self.controller)


class SimpleFishWithPectoralFinSimulator(BaseFluidSimulator):
    def __init__(self, ):
        BaseFluidSimulator.__init__(self, './skeletons/FishWithPectoralFins.skel')
        # dt is the time difference. Here we set it to 1/100 s
        self.controller = creature_controllers.PectoralFinFishController(self.skeletons[0], 1.0 / 100)
        self.skeletons[0].set_controller(self.controller)


class SimpleEelSimulator(BaseFluidSimulator):
    def __init__(self, ):
        BaseFluidSimulator.__init__(self, './skeletons/SimpleEel.skel')
        # dt is the time difference. Here we set it to 1/100 s
        self.controller = creature_controllers.EelController(self.skeletons[0], 1.0 / 100)
        self.skeletons[0].set_controller(self.controller)


class SimpleSeaTurtleSimulator(BaseFluidSimulator):
    def __init__(self, ):
        BaseFluidSimulator.__init__(self, './skeletons/SimpleTurtle.skel')
        # dt is the time difference. Here we set it to 1/100 s
        self.controller = creature_controllers.TurtleController(self.skeletons[0], 1.0 / 100)
        self.skeletons[0].set_controller(self.controller)


class SimpleFlatWormSimulator(BaseFluidSimulator):
    def __init__(self, ):
        BaseFluidSimulator.__init__(self, './skeletons/SimplifiedFlatWorm.skel')
        # dt is the time difference. Here we set it to 1/100 s
        self.controller = creature_controllers.SimplifiedFlatwormController(self.skeletons[0], 1.0 / 100)
        self.skeletons[0].set_controller(self.controller)

    def step(self):
        bodynodes_dict = self.construct_skel_dict()
        comb = []
        import itertools
        for i in itertools.product(['l', 'r'], [1]):
            comb.append(i)
        for segIdx in range(2):
            for side, idx in comb:
                offset1_dir = np.array([-1, 0, 0])
                offset2_dir = np.array([1, 0, 0])
                curr_key = 'wing_' + str(side) + '_' + str(segIdx) + str(idx)
                next_key = 'wing_' + str(side) + '_' + str(segIdx + 1) + str(idx)
                curr_body = bodynodes_dict[curr_key]
                next_body = bodynodes_dict[next_key]

                constraint_force, offset1, offset2 = self.calc_constraint_force(curr_body, offset1_dir, next_body,
                                                                                offset2_dir, strength=100)

                curr_body.add_ext_force(constraint_force, _offset=offset1)
                next_body.add_ext_force(-constraint_force, _offset=offset2)
        super(SimpleFlatWormSimulator, self).step()


    def calc_constraint_force(self, bodynode1, offset1_dir, bodynode2, offset2_dir, strength=1.0):

        shape1 = bodynode1.shapenodes[0]
        body1_geometry = shape1.shape.size()
        shape2 = bodynode2.shapenodes[0]
        body2_geometry = shape2.shape.size()

        offset1 = offset1_dir * body1_geometry / 2
        offset2 = offset2_dir * body2_geometry / 2

        body1_link_pos_to_world = bodynode1.to_world(offset1)
        body2_link_pos_to_world = bodynode2.to_world(offset2)
        constraint_force_dir = body2_link_pos_to_world - body1_link_pos_to_world
        constraint_force = constraint_force_dir * strength
        return constraint_force, offset1, offset2

    def construct_skel_dict(self):
        node_dict = {}
        bodynodes = self.skeletons[0].bodynodes
        for i in range(len(bodynodes)):
            node_dict[bodynodes[i].name] = bodynodes[i]
        return node_dict

class FlatWormSimulator(BaseFluidSimulator):
    def __init__(self, ):
        BaseFluidSimulator.__init__(self, './skeletons/FlatWorm.skel')
        # dt is the time difference. Here we set it to 1/100 s
        self.controller = creature_controllers.FlatwormController(self.skeletons[0], 1.0 / 100)
        self.skeletons[0].set_controller(self.controller)
        # self.constraint_force = []

    def step(self):
        bodynodes_dict = self.construct_skel_dict()
        comb = []
        import itertools
        for i in itertools.product(['l', 'r'], [1,2,3]):
            comb.append(i)
        for segIdx in range(5):
            for side, idx in comb:
                offset1_dir = np.array([-1, 0, 0])
                offset2_dir = np.array([1, 0, 0])
                curr_key = 'wing_' + str(side) + '_' + str(segIdx) + str(idx)
                next_key = 'wing_' + str(side) + '_' + str(segIdx + 1) + str(idx)
                curr_body = bodynodes_dict[curr_key]
                next_body = bodynodes_dict[next_key]

                constraint_force, offset1, offset2 = self.calc_constraint_force(curr_body, offset1_dir, next_body,
                                                                                offset2_dir, strength=5)

                curr_body.add_ext_force(constraint_force, _offset=offset1)
                next_body.add_ext_force(-constraint_force, _offset=offset2)
        super(FlatWormSimulator, self).step()


    def calc_constraint_force(self, bodynode1, offset1_dir, bodynode2, offset2_dir, strength=1.0):

        shape1 = bodynode1.shapenodes[0]
        body1_geometry = shape1.shape.size()
        shape2 = bodynode2.shapenodes[0]
        body2_geometry = shape2.shape.size()

        offset1 = offset1_dir * body1_geometry / 2
        offset2 = offset2_dir * body2_geometry / 2

        body1_link_pos_to_world = bodynode1.to_world(offset1)
        body2_link_pos_to_world = bodynode2.to_world(offset2)
        constraint_force_dir = body2_link_pos_to_world - body1_link_pos_to_world
        constraint_force = constraint_force_dir * strength
        return constraint_force, offset1, offset2

    def construct_skel_dict(self):
        node_dict = {}
        bodynodes = self.skeletons[0].bodynodes
        for i in range(len(bodynodes)):
            node_dict[bodynodes[i].name] = bodynodes[i]
        return node_dict
if __name__ == '__main__':
    pydart.init()

    simulator = SimpleFlatWormSimulator()
    # simulator.debug = True
    creatureParamSettings.testFlatCreatureParam(simulator.controller)
    pydart.gui.viewer.launch_pyqt5(simulator)

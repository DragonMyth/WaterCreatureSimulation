import simplesimulation
import pydart2 as pydart
import cma
import numpy as np
import creaturecontrollers
import multiprocessing as mp
from functools import partial
import threading

fish_with_caudal_directory = './skeletons/SimpleFishWithCaudalFin.skel'
fish_with_pectoral_directory = './skeletons/FishWithPectoralFins.skel'
eel_directory = './skeletons/SimpleEel.skel'
turtle_directory = './skeletons/SimpleTurtle.skel'

general_option = {'maxiter': 5, 'popsize': 10, 'tolx': 1e-5}
turtle_straight_option = {'maxiter': 30, 'popsize': 30, 'fixed_variables': {2: 0, 3: 0, 6: 0, 7: 0,
                                                                            10: 0, 11: 0, 14: 0, 15: 0,
                                                                            18: 0, 19: 0, 22: 0, 23: 0}}

DURATION = 1
DIRECTORY = fish_with_caudal_directory
CONTROLLER = creaturecontrollers.CaudalFinFishController
OPTIONS = cma.CMAOptions.defaults()


class WorldPool(object):
    def __init__(self, population):
        self.worldPool = []
        for _ in range(population):
            world = simplesimulation.MyWorld(DIRECTORY, CONTROLLER)
            self.worldPool.append(world)


def episode(world):
    while world.t < DURATION:
        world.step()
    res = world.skeletons[0].q
    world.reset()
    return res


def general_fitness_func(worldPool, specFitnessFunc, x):
    # Set the parameter for this model
    thisWorld = worldPool.pop(0)
    parse_param_for_optimization(x, thisWorld.controller)

    origin_q = thisWorld.skeletons[0].q
    final_q = episode(thisWorld)
    root_joint_dof = len(thisWorld.skeletons[0].joint[0].dofs)
    worldPool.append(thisWorld)

    res = specFitnessFunc(origin_q, final_q, root_joint_dof)
    return res


def straight_fitness_func(origin_q, final_q, root_joint_dof):
    cost = -0.7 * (final_q[0] - origin_q[0])
    for i in range(1, root_joint_dof):
        cost += (final_q[i] - origin_q[i]) ** 2
    return cost


def parse_param_for_optimization(x, controller, mirror=False):
    """
    :param x: The result of X after one iteration
    :param controller: The controller we want to set
    :param modelType: Type of the model. Decide if we can reduce some dof for optimization
    :return: 
    """
    x_split = np.split(x, 3)
    controller.joint_max = x_split[0]
    controller.joint_min = x_split[1]
    controller.phi = x_split[2]
    if mirror:
        controller.joint_max[2:4] = controller.joint_max[0:2]
        controller.joint_max[6:8] = controller.joint_max[4:6]
        controller.joint_min[2:4] = controller.joint_min[0:2]
        controller.joint_min[6:8] = controller.joint_min[4:6]
        controller.phi[3] = controller.phi[1]
        controller.phi[2] = controller.phi[0] + np.pi
        controller.phi[7] = controller.phi[5]
        controller.phi[6] = controller.phi[4] + np.pi


if __name__ == '__main__':
    pydart.init()

    testWorld = simplesimulation.MyWorld(eel_directory, creaturecontrollers.EelController)
    # param_settings.simpleFishParam(world.controller)
    joint_max = testWorld.controller.joint_max
    joint_min = testWorld.controller.joint_min
    phi = testWorld.controller.phi

    x0 = np.concatenate((joint_max, joint_min, phi))
    es = cma.CMAEvolutionStrategy(x0, 0.3, general_option)

    pool = mp.Pool(es.popsize)
    worldPool = WorldPool(es.popsize).worldPool

    while not es.stop():
        X = es.ask()
        partial_fitness = partial(general_fitness_func, worldPool, straight_fitness_func)
        fit = pool.map(partial_fitness, X)
        print("MIAOMIAO")

        es.tell(X, fit)
        es.disp()
    res = es.result()
    print ("The final Result is: ", res[0])

    testWorld.reset()
    parse_param_for_optimization(res[0], testWorld.controller)
    pydart.gui.viewer.launch_pyqt5(testWorld)

import SimpleSimulation
import pydart2 as pydart
import cma
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import multiprocessing as mp
import creatureParamSettings as param_settings
from functools import partial

fish_with_caudal_directory = './skeletons/SimpleFishWithCaudalFin.skel'
fish_with_pectoral_directory = './skeletons/FishWithPectoralFins.skel'
eel_directory = './skeletons/SimpleEel.skel'
turtle_directory = './skeletons/SimpleTurtle.skel'

general_option = {'maxiter': 100, 'popsize': 35, 'tolx': 1e-6}

turtle_straight_option = {'maxiter': 30, 'popsize': 30, 'fixed_variables': {2: 0, 3: 0, 6: 0, 7: 0,
                                                                            10: 0, 11: 0, 14: 0, 15: 0,
                                                                            18: 0, 19: 0, 22: 0, 23: 0}}


def episode(world):
    while world.t < 2:
        world.step()
    return world.skeletons[0].q


def straight_fitness_func(x):
    temp_world = SimpleSimulation.MyWorld(eel_directory, SimpleSimulation.SimpleWaterCreatureController)
    # Set the parameter for this model
    set_parameter_for_opt(x, temp_world.controller, 'eel')

    origin_q = temp_world.skeletons[0].q
    final_q = episode(temp_world)
    temp_world.destroy()
    res = -(final_q[1] - origin_q[1] - (((origin_q[0] - final_q[0]) ** 2) + (origin_q[2] - final_q[2]) ** 2))
    return res


def set_parameter_for_opt(x, controller, modelType):
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
    if modelType == 'turtle':
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
    world = SimpleSimulation.MyWorld(eel_directory, SimpleSimulation.SimpleWaterCreatureController)

    # param_settings.simpleFishParam(world.controller)
    joint_max = world.controller.joint_max
    joint_min = world.controller.joint_min
    phi = world.controller.phi

    x0 = np.concatenate((joint_max, joint_min, phi))
    es = cma.CMAEvolutionStrategy(x0, 0.3, general_option)

    pool = mp.Pool(es.popsize)
    a = cma.CMAOptions.defaults()
    while not es.stop():
        X = es.ask()
        # print(X)
        fit = pool.map(straight_fitness_func, X)
        # print(fit)
        es.tell(X, fit)
        es.disp()
        es.logger.add()

    res = es.result()
    print ("The final Result is: ", res[0])

    world.reset()
    set_parameter_for_opt(res[0], world.controller, 'eel')
    pydart.gui.viewer.launch_pyqt5(world)

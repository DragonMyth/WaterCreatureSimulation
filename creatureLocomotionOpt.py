import SimpleSimulation
import pydart2 as pydart
import cma
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import multiprocessing as mp
import creatureParamSettings as param_settings
from functools import partial

skel_directory = './skeletons/SimpleFishWithCaudalFin.skel'


def episode(world):
    while world.t < 3:
        world.step()
    return world.skeletons[0].q
def straight_fitness_func(world,x):
    # world = args[0]

    new_x = np.split(x, 3)
    joint_max = new_x[0]
    joint_min = new_x[1]
    phi = new_x[2]
    world.reset()
    world.controller.joint_max = joint_max
    world.controller.joint_min = joint_min
    world.controller.phi = phi

    origin_q = world.skeletons[0].q
    final_q = episode(world)
    return -(final_q[1]-origin_q[1]-(((origin_q[0]-final_q[0])**2)+(origin_q[2]-final_q[2])**2))

if __name__ == '__main__':
    pydart.init()

    world = SimpleSimulation.MyWorld(skel_directory, SimpleSimulation.SimpleWaterCreatureController)

    param_settings.simpleFishParam(world.controller)
    joint_max = world.controller.joint_max
    joint_min = world.controller.joint_min
    phi = world.controller.phi
    # pydart.gui.viewer.launch_pyqt5(world)
    x0=np.concatenate((joint_max,joint_min,phi))
    # print(x0)
    # res = cma.fmin(straight_fitness_func,x0,0.1)
    es = cma.CMAEvolutionStrategy(x0, 0.1, {'maxiter': 10, 'popsize': 40})
    pool = mp.Pool(es.popsize)
    while not es.stop():
        X = es.ask()
        # print(X)
        func = partial(straight_fitness_func,world)
        fit = pool.map(func,X)
        es.tell(X,fit)
        es.disp()
        es.logger.add()

    # es.optimize(straight_fitness_func,iterations=10,args=[world])
    res = es.result()
    print (res[0])

    world.reset()
    opt_param = np.split(res[0],3)
    world.controller.joint_max = opt_param[0]
    world.controller.joint_min = opt_param[1]
    world.controller.phi = opt_param[2]
    pydart.gui.viewer.launch_pyqt5(world)
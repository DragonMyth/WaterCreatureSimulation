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
flat_creature_directory = './skeletons/FlatShapeCreature.skel'

general_option = {'maxiter': 80, 'popsize': 35, 'tolx': 1e-4, 'tolfun': 1e-1}
turtle_straight_option = {'maxiter': 120, 'popsize': 30,
                          'fixed_variables': {4: 0, 5: 0, 6: 0, 7: 0, 12: 0, 13: 0, 14: 0, 15: 0, 20: 0, 21: 0,
                                              22: 0, 23: 0}}

DURATION = 0.5
DIRECTORY = eel_directory
CONTROLLER = creaturecontrollers.EelController
OPTIONS = general_option
NUM_RESTART = 10


class WorldPool(object):
    def __init__(self, population):
        self.worldPool = mp.Manager().Queue()
        for _ in range(population):
            world = simplesimulation.MyWorld(DIRECTORY, CONTROLLER)
            skel = world.skeletons[0]
            q = skel.q
            q[2] = np.pi / 2
            q[6] = np.pi / 3 * 2
            q[7] = -np.pi / 3 * 2
            skel.set_positions(q)
            bd1 = skel.bodynodes[0]
            bd2 = skel.bodynodes[2]
            jointPos = np.array([np.sqrt(3) / 2 * 0.2, 0.1, 0])
            joint = pydart.constraints.BallJointConstraint(bd1, bd2, jointPos)
            joint.add_to_world(world)
            self.worldPool.put(world)


def _init(queue):
    global currentWorld
    currentWorld = queue.get()


def setBoundary(length, joint_max_l, joint_max_u, joint_min_l, joint_min_u, phi_l, phi_h):
    lower_bounds = [0 for _ in range(length)]
    upper_bounds = [0 for _ in range(length)]
    for i in range(length / 3):
        lower_bounds[i] = joint_max_l
        lower_bounds[i + length / 3] = joint_min_l
        lower_bounds[i + length * 2 / 3] = phi_l

        upper_bounds[i] = joint_max_u
        upper_bounds[i + length / 3] = joint_min_u
        upper_bounds[i + length * 2 / 3] = phi_h

    return lower_bounds, upper_bounds


def episode():
    global currentWorld
    while currentWorld.t < DURATION:
        currentWorld.step()
    res = currentWorld.skeletons[0].q
    return res


def general_fitness_func(specFitnessFunc, x):
    # Set the parameter for this model
    global currentWorld
    parse_param_for_optimization(x, currentWorld.controller)

    origin_q = currentWorld.skeletons[0].q

    final_q = episode()
    currentWorld.reset()
    root_joint_dof = len(currentWorld.skeletons[0].joints[0].dofs)

    res = specFitnessFunc(origin_q, final_q, root_joint_dof)
    return res


def straight_fitness_func(origin_q, final_q, root_joint_dof):
    cost = - 3 * (final_q[3] - origin_q[3]) ** 2 * ((final_q[3] - origin_q[3]) / abs(final_q[3] - origin_q[3]))
    for i in range(root_joint_dof):
        if (i == 3): continue
        cost += 6 * (final_q[i] - origin_q[i]) ** 2
    return cost


def parse_param_for_optimization(x, controller):
    """
    :param x: The result of X after one iteration
    :param controller: The controller we want to set
    :return: 
    """
    x_split = np.split(x, 3)
    controller.joint_max = x_split[0]
    controller.joint_min = x_split[1]
    controller.phi = x_split[2]

def restraintRadians(rad):
    if rad > np.pi * 2:
        return rad - np.pi * 2
    elif rad < -np.pi * 2:
        return rad + np.pi * 2
    else:
        return rad


def runCMA(x0, pool, processCnt):
    es = cma.CMAEvolutionStrategy(x0, 0.7
                                  , OPTIONS)
    while not es.stop():
        X = es.ask()
        partial_fitness = partial(general_fitness_func, straight_fitness_func)
        fit = []
        for i in range(int(np.ceil(es.popsize / processCnt)) + 1):
            if (len(X[i * processCnt:]) < processCnt):
                batchFit = pool.map(partial_fitness, X[i * processCnt:])
            else:
                batchFit = pool.map(partial_fitness, X[i * processCnt:(i + 1) * processCnt])
            fit.extend(batchFit)

        es.tell(X, fit)
        es.disp()
        es.logger.add()
    res = es.result()
    return res


def writeOptimaToFile(optima):
    exportFile = open('LatestOptimaResult.txt', 'w')
    for optimum in optima:
        exportFile.write(str(optimum[1]) + ' , ' + str(optimum[0].tolist()) + '\n')
    exportFile.close()


if __name__ == '__main__':
    pydart.init()

    testWorld = simplesimulation.MyWorld(DIRECTORY, CONTROLLER)





    joint_max = testWorld.controller.joint_max
    joint_min = testWorld.controller.joint_min
    phi = testWorld.controller.phi

    x0 = np.concatenate((joint_max, joint_min, phi))

    lb, hb = setBoundary(len(x0), 0, np.pi / 3, -np.pi / 3, 0, -np.pi, np.pi)
    OPTIONS['boundary_handling'] = cma.BoundPenalty
    OPTIONS['bounds'] = [lb, hb]

    allRes = []
    bestEval = np.inf
    processCnt = mp.cpu_count()
    worldPool = WorldPool(processCnt).worldPool
    pool = mp.Pool(processCnt, _init, (worldPool,))

    res = 0
    exportFile = open('LatestOptimaResult.txt', 'w')
    exportFile.write('')
    exportFile.close()
    for i in range(NUM_RESTART):
        exportFile = open('LatestOptimaResult.txt', 'a')

        res = runCMA(x0, pool, processCnt)
        exportFile.write(str(res[1]) + ' , ' + str(res[0].tolist())[1:-1] + '\n')
        # writeOptimaToFile(allRes)
        exportFile.close()

    res = res[0]
    print ("The final Result is: ", res)

    testWorld.reset()
    parse_param_for_optimization(res, testWorld.controller)
    pydart.gui.viewer.launch_pyqt5(testWorld)

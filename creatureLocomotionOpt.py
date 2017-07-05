import creature_simulators
import pydart2 as pydart
import cma
import numpy as np
import creature_controllers

import multiprocessing as mp
from functools import partial
import threading

general_option = {'maxiter': 30, 'popsize': 40}
turtle_straight_option = {'maxiter': 100, 'popsize': 30,
                          'fixed_variables': {4: 0, 5: 0, 6: 0, 7: 0, 12: 0, 13: 0, 14: 0, 15: 0, 20: 0, 21: 0,
                                              22: 0, 23: 0}}

DURATION = 0.5
SIMULATOR = creature_simulators.SimpleFlatWormSimulator
OPTIONS = general_option
CMA_STEP_SIZE = 0.7
NUM_RESTART = 1

class SimulatorPool(object):
    def __init__(self, population):
        self.simulatorPool = mp.Manager().Queue()
        for _ in range(population):
            simulator = SIMULATOR()
            self.simulatorPool.put(simulator)
def _init(queue):
    global current_simulator
    current_simulator = queue.get()


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
    global current_simulator
    terminal_flag = False
    while current_simulator.t < DURATION and not terminal_flag:
        current_simulator.step()
        curr_q = current_simulator.skeletons[0].q
        if np.isnan(curr_q).any():
            terminal_flag = True
            print("NAN")
    res = current_simulator.skeletons[0].q
    if terminal_flag:
        for i in range(len(res)):
            res[i] = 0
    return res


def general_fitness_func(spec_fitness_func, x):
    # Set the parameter for this model
    global current_simulator
    parse_param_for_optimization(x, current_simulator.controller)

    origin_q = current_simulator.skeletons[0].q

    final_q = episode()
    current_simulator.reset()
    root_joint_dof = len(current_simulator.skeletons[0].joints[0].dofs)

    res = spec_fitness_func(origin_q, final_q, root_joint_dof)
    return res


def straight_fitness_func(origin_q, final_q, root_joint_dof):
    norm = abs(final_q[3] - origin_q[3])
    if norm != 0:
        cost = - 3 * (final_q[3] - origin_q[3]) ** 2 * ((final_q[3] - origin_q[3]) / norm)
    else:
        cost = np.inf
    for i in range(root_joint_dof):
        if i == 3: continue
        cost += 5 * (final_q[i] - origin_q[i]) ** 2
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


def run_CMA(x0, pool, processCnt):
    es = cma.CMAEvolutionStrategy(x0, CMA_STEP_SIZE
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

    testSimulator = SIMULATOR()

    joint_max = testSimulator.controller.joint_max
    joint_min = testSimulator.controller.joint_min
    phi = testSimulator.controller.phi

    x0 = np.concatenate((joint_max, joint_min, phi))

    lb, hb = setBoundary(len(x0), 0, np.pi / 2, -np.pi / 2, 0, -np.pi, np.pi)
    OPTIONS['boundary_handling'] = cma.BoundPenalty
    OPTIONS['bounds'] = [lb, hb]

    allRes = []
    bestEval = np.inf
    processCnt = mp.cpu_count()
    simulatorPool = SimulatorPool(processCnt).simulatorPool
    pool = mp.Pool(processCnt, _init, (simulatorPool,))

    res = 0
    exportFile = open('LatestOptimaResult.txt', 'w')
    exportFile.write('')
    exportFile.close()
    for i in range(NUM_RESTART):
        exportFile = open('LatestOptimaResult.txt', 'a')

        res = run_CMA(x0, pool, processCnt)
        exportFile.write(str(res[1]) + ' , ' + str(res[0].tolist())[1:-1] + '\n')
        # writeOptimaToFile(allRes)
        exportFile.close()

    res = res[0]
    print ("The final Result is: ", res)

    testSimulator.reset()
    parse_param_for_optimization(res, testSimulator.controller)
    pydart.gui.viewer.launch_pyqt5(testSimulator)

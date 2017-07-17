import creature_simulators
import pydart2 as pydart
import cma
import numpy as np
from utils import least_square_circle
import copy
import multiprocessing as mp
from functools import partial

general_option = {'maxiter': 70, 'popsize': 35}

DURATION = 5
SIMULATOR = creature_simulators.TurtleCircSimulator
OPTIONS = general_option
CMA_STEP_SIZE = 0.6
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
        if curr_q.any() > 10 ** 3:
            print(curr_q)
            print(current_simulator.skeletons[0].controller.compute())
            print(current_simulator.skeletons[0].controller.pd_controller_target_compute())
            terminal_flag = True
            print("NAN")
    res = current_simulator.skeletons[0].q
    if terminal_flag:
        res = -1
    return res


def general_fitness_func(spec_fitness_func, x):
    # Set the parameter for this model
    global current_simulator
    origin_q = current_simulator.skeletons[0].q
    parse_param_for_optimization(x, current_simulator.controller)

    final_q = episode()
    if type(final_q) != int:
        trail = copy.deepcopy(current_simulator.trail)
        current_simulator.reset()
        root_joint_dof = len(current_simulator.skeletons[0].joints[0].dofs)

        res = spec_fitness_func(origin_q, final_q, root_joint_dof, trail)
    else:
        res = 10
    return res


def circular_fitness_func(origin_q, final_q, root_joint_dof, head_trail):
    target_radius = 1.5
    X = np.array(head_trail[0])
    Y = np.array(head_trail[2])
    xc, yc, radius = least_square_circle(X, Y)

    distance = 0
    for i in range(1, len(X)):
        distance += np.sqrt((X[i] - X[i - 1]) ** 2 + (Y[i] - Y[i - 1]) ** 2)
    w0 = 20
    w1 = 3
    return w0 * (((radius - target_radius) ** 2) / (1.5 ** 2)) - w1 * distance


def straight_fitness_func(origin_q, final_q, root_joint_dof, trail):
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
        partial_fitness = partial(general_fitness_func, FITNESS_FUNC)
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


FITNESS_FUNC = circular_fitness_func

if __name__ == '__main__':
    pydart.init()

    testSimulator = SIMULATOR()

    joint_max = testSimulator.controller.joint_max
    joint_min = testSimulator.controller.joint_min
    phi = testSimulator.controller.phi

    x0 = np.concatenate((joint_max, joint_min, phi))

    lb, hb = setBoundary(len(x0), 0, np.pi / 3, -np.pi / 3, 0, -np.pi, np.pi)
    OPTIONS['boundary_handling'] = cma.BoundTransform
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

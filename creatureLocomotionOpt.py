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

general_option = {'maxiter': 120, 'popsize': 35, 'tolx': 1e-4, 'tolfun': 1e-1}
turtle_straight_option = {'maxiter': 35, 'popsize': 25,
                          'fixed_variables': {4: 0, 5: 0, 6: 0, 7: 0, 12: 0, 13: 0, 14: 0, 15: 0, 20: 0, 21: 0,
                                              22: 0, 23: 0}}

DURATION = 1
DIRECTORY = eel_directory
CONTROLLER = creaturecontrollers.EelController

OPTIONS = general_option
NUM_RESTART = 1
MIRROR = False

class WorldPool(object):
    def __init__(self, population):
        self.worldPool = mp.Manager().Queue()
        for _ in range(population):
            world = simplesimulation.MyWorld(DIRECTORY, CONTROLLER)
            self.worldPool.put(world)


def _init(queue):
    global currentWorld
    currentWorld = queue.get()


def setBoundary(length, joint_max_l, joint_max_u, joint_min_l, joint_min_u, phi_l, phi_h):
    lower_bounds = [0 for _ in range(length)]
    upper_bounds = [0 for _ in range(length)]
    for i in range(length / 3):
        lower_bounds[3 * i] = joint_max_l
        lower_bounds[3 * i + 1] = joint_min_l
        lower_bounds[3 * i + 2] = phi_l
        upper_bounds[3 * i] = joint_max_u
        upper_bounds[3 * i + 1] = joint_min_u
        upper_bounds[3 * i + 2] = phi_h

    return lower_bounds, upper_bounds


def episode():
    global currentWorld
    while currentWorld.t < DURATION:
        # print("processIdent",mp.current_process().pid,'ID2', currentWorld.id)
        currentWorld.step()
    res = currentWorld.skeletons[0].q
    return res


def general_fitness_func(specFitnessFunc, x):
    # Set the parameter for this model
    # thisWorld = worldPool.pop(0)
    global currentWorld
    parse_param_for_optimization(x, currentWorld.controller, mirror=MIRROR)

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
        cost += 5 * (final_q[i] - origin_q[i]) ** 2
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
        controller.joint_max[4::] = controller.joint_max[0:4]
        controller.joint_min[4::] = controller.joint_min[0:4]
        # controller.phi[4::] = controller.phi[0:4]+np.pi
        controller.phi[5] = controller.phi[1]
        controller.phi[4] = controller.phi[0] + np.pi
        controller.phi[7] = controller.phi[3]
        controller.phi[6] = controller.phi[2] + np.pi


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
        # fit = pool.map(partial_fitness, X)

        es.tell(X, fit)
        es.disp()
        es.logger.add()
    res = es.result()
    return res


def writeOptimaToFile(optima):
    exportFile = open('optima.txt', 'w')
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

    lb, hb = setBoundary(len(x0), 0, np.pi / 2, -np.pi / 2, 0, -np.pi, np.pi)
    OPTIONS['boundary_handling'] = cma.BoundPenalty
    OPTIONS['bounds'] = [lb, hb]

    allRes = []
    bestEval = np.inf
    processCnt = 2 * mp.cpu_count()
    worldPool = WorldPool(processCnt).worldPool
    pool = mp.Pool(processCnt, _init, (worldPool,))
    for i in range(NUM_RESTART):
        res = runCMA(x0, pool, processCnt)
        allRes.append(res)
    writeOptimaToFile(allRes)

    # res = allRes[0][0]
    res = allRes[0][0]



    # res = np.array(
    #     [0.0, -1.4224849745340211, 0.7057698816961092, 0.6146904679326392, -0.3870720440782131, 0.7580191899163806,
    #      0.9806520839194101, -0.8000682998945848, -0.7000274918445064, 0.393923839745565, -1.5707963267948966,
    #      -2.530883539669202, 0.0, -1.0001673112601475, 1.0536236557171075])
    # res = np.array(
    #     [1.5707963267948966, 0.0, 1.1046112000754285, 0.7835113416271448, -0.6026744399986707, -1.002522405308755, 0.0,
    #      -0.9520845078052966, -0.9130757705385262, 0.4900470445430839, 0.0, -0.23071548043181905, 0.0,
    #      -1.361826622709664,
    #      0.4875886599513184])
    # res = np.array(
    #     [1.5707963267948966, -0.9141843296860563, -0.5471013452437721, 0.0, -0.4724276807906173, -0.7457052224641255,
    #      1.5707963267948966, 0.0, 0.6583854587558936, 0.2562671634279459, -1.5707963267948966, 1.030051719294236,
    #      0.5650676611581056, -0.03475189484395323, -0.6239401292010973])
    # res = np.array(
    #     [0.0, -1.5707963267948966, 0.6286299387890962, 0.6504742527912812, -0.3605727151100941, 0.7245509919372788,
    #      0.9514229394691246, -0.862902609024752, -0.5720849946016655, 0.38939463981048267, -1.479263394681494,
    #      -2.3205098110070446, 0.0, -0.8423680778598122, 1.1831139847909957])
    # res = np.array(
    #     [1.5707963267948966, -0.9436446643052514, -0.5378847180124914, 0.0, -0.45407605756024966, -0.7277153533098101,
    #      1.5707963267948966, 0.0, 0.6644884007146518, 0.23239039074215062, -1.5707963267948966, 1.0339772801859646,
    #      0.5519752557424235, -0.0549073303529046, -0.6056218370660087])
    # res = np.array(
    #     [1.5707963267948966, -0.9222312021426817, -0.54248721499089, 0.0, -0.4802517376557276, -0.7343211267794951,
    #      1.5707963267948966, 0.0, 0.6628064345253312, 0.2515476216847905, -1.5707963267948966, 1.0338874031087744,
    #      0.5414244842898892, -0.006599776116775577, -0.6518578704525275])
    # res = np.array(
    #     [1.5707963267948966, -0.9340766628190289, -0.545031744787835, 0.0, -0.46702069457976353, -0.7330258799289101,
    #      1.5707963267948966, 0.0, 0.6640947500124772, 0.25973278415388174, -1.5707963267948966, 1.0368562271770094,
    #      0.5643060059419477, -0.016527202169759878, -0.6666220196982827])
    # res = np.array(
    #     [0.0, -1.5707963267948966, 0.5032437008911052, 0.5696174102799634, -0.402988642719676, 0.436662942306962,
    #      1.0586944448268598, -0.7437307107136528, -0.612293793796137, 0.5169325219186756, -0.8239233701121581,
    #      -2.0521828914492697, 0.2837094954720273, -0.33297451325329414, 1.5487073660288186])
    # res = np.array(
    #     [0.0, -1.45744065835128, 0.6776077434994641, 0.6405265036927875, -0.3855095677767568, 0.7279080626002453,
    #      0.9623472927471843, -0.8000683856929355, -0.6930560210648284, 0.43065038901077185, -1.5455299600230266,
    #      -2.4821930527615916, 0.0, -0.9141593374010224, 1.0386009653666757])
    # res = np.array(
    #     [0.0, -1.4087479800529528, 0.6803115264632309, 0.6500410986412195, -0.39455594437126573, 0.744778323029786,
    #      0.9903095793066098, -0.846482039836443, -0.7008902206013475, 0.4107369146693624, -1.4342831662242335,
    #      -2.485041600257863, 0.0, -0.9230375552072599, 1.0617024402831579])
    print ("The final Result is: ", res)

    testWorld.reset()
    parse_param_for_optimization(res, testWorld.controller, mirror=MIRROR)
    pydart.gui.viewer.launch_pyqt5(testWorld)

import simplesimulation
import pydart2 as pydart
import numpy as np
import creaturecontrollers

fish_with_caudal_directory = './skeletons/SimpleFishWithCaudalFin.skel'
fish_with_pectoral_directory = './skeletons/FishWithPectoralFins.skel'
eel_directory = './skeletons/SimpleEel.skel'
turtle_directory = './skeletons/SimpleTurtle.skel'


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


def eelMultiResultsenario():
    pydart.init()

    testWorld = simplesimulation.MyWorld(eel_directory, creaturecontrollers.EelController)

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
    res = np.array(
        [1.5707963267948966, -0.9340766628190289, -0.545031744787835, 0.0, -0.46702069457976353, -0.7330258799289101,
         1.5707963267948966, 0.0, 0.6640947500124772, 0.25973278415388174, -1.5707963267948966, 1.0368562271770094,
         0.5643060059419477, -0.016527202169759878, -0.6666220196982827])
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
    parse_param_for_optimization(res, testWorld.controller, mirror=False)
    pydart.gui.viewer.launch_pyqt5(testWorld)


def turtleFrontLimpTrack():
    pydart.init()
    testWorld = simplesimulation.MyWorld(turtle_directory, creaturecontrollers.TurtleController)
    testWorld.reset()
    res = np.array([1.5707963267948966, -0.3308522849896008, 0.18453346454873287, 1.5707963267948966, 0.0, 0.0, 0.0, 0.0, -2.070887352096422, 0.5078136968877673, 0.0, 0.8994943601154446, 0.0, 0.0, 0.0, 0.0, -1.038568025388365, -0.13477440819880682, 1.2907824168398736, 0.0, 0.0, 0.0, 0.0, 0.0])
    parse_param_for_optimization(res,testWorld.controller,mirror=True)
    taus = []
    while(testWorld.t<1):
        testWorld.step()
        tauCp = []
        for ele in testWorld.skeletons[0].tau[6::]:
            tauCp.append(ele)
        taus.append(tauCp)
    taus = np.asarray(taus)

    T = np.linspace(0,2,len(taus))
    return taus, T


def turtleTest():
    pydart.init()
    testWorld = simplesimulation.MyWorld(turtle_directory, creaturecontrollers.TurtleController)
    testWorld.reset()
    res = np.array([1.5707963267948966, -0.3308522849896008, 0.18453346454873287, 1.5707963267948966, 0.0, 0.0, 0.0, 0.0, -2.070887352096422, 0.5078136968877673, 0.0, 0.8994943601154446, 0.0, 0.0, 0.0, 0.0, -1.038568025388365, -0.13477440819880682, 1.2907824168398736, 0.0, 0.0, 0.0, 0.0, 0.0])
    parse_param_for_optimization(res,testWorld.controller,mirror=True)
    pydart.gui.viewer.launch_pyqt5(testWorld)

if __name__ == '__main__':
    eelMultiResultsenario()
    # turtleFrontLimpTrack()
    # turtleTest()
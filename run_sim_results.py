import creature_simulators
import pydart2 as pydart
import numpy as np
import creature_controllers

fish_with_caudal_directory = './skeletons/SimpleFishWithCaudalFin.skel'
fish_with_pectoral_directory = './skeletons/FishWithPectoralFins.skel'
eel_directory = './skeletons/SimpleEel.skel'
turtle_directory = './skeletons/SimpleTurtle.skel'
flat_creature_directory = './skeletons/FlatShapeCreature.skel'


def parse_param_for_optimization(x, controller):
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


def restraintRadians(rad):
    if rad > np.pi:
        return rad - np.pi * 2
    elif rad < -np.pi:
        return rad + np.pi * 2
    else:
        return rad


def eelMultiResultsenario():
    pydart.init()

    testWorld = creature_simulators.SimpleEelSimulator()
    # res = np.array([0.865800698946, -0.194346939603, 0.197102257761, 1.0594520005, 0.0, -1.13235956076,
    #                 0.833331413606, -0.334648774595, -0.0214682856467, 0.162269855443, -0.52118503726, -0.984741673114,
    #                 0.572946877556, 0.0, 0.224657728364, 0.558109311755, 0.0, -0.407858856304, 0.0, -0.00887840121136,
    #                 0.993795062808, 0.0, 0.0, 1.20457487205, 0.0, -0.313108383912, -0.884193850598, 0.0, 0.0,
    #                 -0.662197488814, 0.0, -1.16065393284, 0.750792483552, 0.130154853371, -0.237845587115,
    #                 0.0622343542702, 0.0, -0.789138170157, 1.18652286169, 0.404065663997, 0.0, -0.37872940787, 0.0,
    #                 -0.80099910133, 0.49102745224, 0.970537981247, -1.11282332379, -0.799196623624, 0.454687710012, 0.0,
    #                 -0.84788853308, 1.57079632679, 0.0, 1.49220125961, 1.42565228888, -0.210353649049, -1.09260718216,
    #                 0.442969800238, 0.0, 0.473205009435, 0.0, -0.174169562181, -0.609629991704, 0.0, 0.0, 1.16290666193,
    #                 0.0, 0.0, -0.342331417775, 1.54976601245, -0.174877305971, 0.44174145415, 0.654404529489, 0.0,
    #                 2.24887770565, 0.0, 0.0, -0.00769640945576, 0.800446971405, -0.0425060670261, 0.194379485045, 0.0,
    #                 0.0, -0.863460320338, 0.0, 0.0, -0.464364975736, 0.0, -0.936700796897, 0.304650077326,
    #                 1.35142100717, 0.0, 0.178962910486, 0.25471523241, -0.796583440319, -0.718088417278, 0.0,
    #                 -0.557752449687, 0.38277474962, 0.0, -0.587990795142, -0.282388379209, 1.57079632679,
    #                 -0.237551277775, -0.0210916382682, 1.03600923673, -1.57079632679, 0.437586766996, 0.0, 0.0,
    #                 0.255025682551, 0.0, -0.0864280324999, -0.17792108792, 1.57079632679, -0.312777558852,
    #                 0.957192481596, 0.847946084013, -0.558056991406, -1.22906994478, 0.702431337145, 0.0,
    #                 -0.582107242156]
    #
    #                )

    # print(res.shape)
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
    res = np.array(
        [0.0, -1.4087479800529528, 0.6803115264632309, 0.6500410986412195, -0.39455594437126573, 0.744778323029786,
         0.9903095793066098, -0.846482039836443, -0.7008902206013475, 0.4107369146693624, -1.4342831662242335,
         -2.485041600257863, 0.0, -0.9230375552072599, 1.0617024402831579])
    # -----------------------------------------------------------------------------------------------------------------------------------
    res = np.array(
        [0.25668119929879385, 0.9266550007393575, 0.7529615004628528, 0.33918886684452365, 0.4860584031082612,
         -0.09949557970074752, -0.6491974757832526, -0.716295986495328, -0.3693374258889609, -0.3651949934544141,
         -2.225814524318039, -2.9097140274478788, -1.4410324083965937, -0.7212207280715586, 0.3054951072133603])
    print ("The final Result is: ", res)

    testWorld.reset()
    parse_param_for_optimization(res, testWorld.controller)
    pydart.gui.viewer.launch_pyqt5(testWorld)


def turtleFrontLimpTrack():
    pydart.init()
    testWorld = creature_simulators.SimpleSeaTurtleSimulator()
    testWorld.reset()
    res = np.array(
        [0.0, -0.7078904102036592, 1.0418840941151293, 0.0, 0.0, 0.0, 0.0, 0.0, 0.10663599329888622, 1.0471975511965976,
         -1.0471975511965976, 1.7879390624673452, 0.0, 0.0, 0.0, 0.0, -0.7555759584293742, -2.1196360599388986,
         0.4654608688951542, -1.0471975511965976, 0.0, 0.0, 0.0, 0.0])
    parse_param_for_optimization(res, testWorld.controller)
    taus = []
    while (testWorld.t < 1):
        testWorld.step()
        tauCp = []
        for ele in testWorld.skeletons[0].tau[6::]:
            tauCp.append(ele)
        taus.append(tauCp)
    taus = np.asarray(taus)

    T = np.linspace(0, 2, len(taus))
    return taus, T


def turtleTest():
    pydart.init()
    testWorld = creature_simulators.SimpleSeaTurtleSimulator()
    testWorld.reset()
    res = np.array(
        [1.5707963267948966, -0.3308522849896008, 0.18453346454873287, 1.5707963267948966, 0.0, 0.0, 0.0, 0.0,
         -2.070887352096422, 0.5078136968877673, 0.0, 0.8994943601154446, 0.0, 0.0, 0.0, 0.0, -1.038568025388365,
         -0.13477440819880682, 1.2907824168398736, 0.0, 0.0, 0.0, 0.0, 0.0])
    res = np.array(
        [1.5707963267948966, 0.9961768713996245, 1.5707963267948966, 0.0, -1.5707963267948966,
         -0.7521668135857681, -1.5707963267948966, -1.5707963267948966, 0.7209630882534658,
         -0.84154544186923, 1.8160795318017378, 0.285419779891251])
    parse_param_for_optimization(res, testWorld.controller)
    pydart.gui.viewer.launch_pyqt5(testWorld)


def turtleCircleTest():
    pydart.init()
    testWorld = creature_simulators.TurtleCircSimulator()
    testWorld.reset()
    res = np.array(
        [1.0471975511965976, 0.9986565548373585, 0.0, 1.0471975511965976, 0.8914101568082288, 0.0, -1.0471975511965976,
         -0.8459011127674524, -0.9985262902467856, -0.3971157289782413, -0.9646710848414166, -1.0471975511965976,
         -0.07590327371994275, -1.5411200183879052, -1.4159266839321565, -0.3356806222899789, -0.7809463558381204,
         0.9296038447176289
         ])
    parse_param_for_optimization(res, testWorld.controller)
    pydart.gui.viewer.launch_pyqt5(testWorld)


def simplified_flatwormTest():
    pydart.init()
    testWorld = creature_simulators.SimpleFlatWormSimulator()
    testWorld.reset()
    res = np.array(
        [1.5707963267948966, 1.1725515389131442, 1.5707963267948966, 0.476874647996004, 0.0, 0.07711606745163911,
         -1.5707963267948966, -1.244981271404276, -1.5707963267948966, -0.2126363288060239, -0.056124372193900374, 0.0,
         1.137030665369484, 0.584456864548018, -2.0108484907883613, -3.141592653589793, 0.5257032384928889,
         1.9878978493069974
         ])
    parse_param_for_optimization(res, testWorld.controller)
    pydart.gui.viewer.launch_pyqt5(testWorld)


def flatwormTest():
    pydart.init()
    testWorld = creature_simulators.FlatWormSimulator()
    testWorld.reset()
    res = np.array(
        [0.0, 0.0, 0.0, 0.31165818849512983, 0.0, 0.4830137470807886, 0.5235987755982988, 0.19090576900795259,
         0.5235987755982988, 0.5235987755982988, 0.41740002048951796, 0.5235987755982988, 0.0, 0.5235987755982988,
         0.5235987755982988, 0.13396627466038177, 0.5235987755982988, 0.011179447075155169, -0.5235987755982988,
         -0.5235987755982988, 0.0, -0.5235987755982988, 0.0, -0.28666133239593916, -0.5235987755982988,
         -0.5235987755982988, -0.5235987755982988, -0.4067177987907987, -0.5235987755982988, -0.5235987755982988, 0.0,
         0.0, -0.5235987755982988, 0.0, -0.47718928935557886, 0.0, -2.6156361018421257, 2.610084404755959,
         0.31552753918999565, 0.9948195894232679, 0.17503829923323136, 1.0936975439321552, 0.5160581391964534,
         -0.16652173865272113, 0.24859287468597668, -0.6240957597714036, 0.600421943226074, -0.5855285709499185,
         0.4107999513115864, -2.2472872219747124, -1.031743567503235, -0.9188256090758944, -0.9639423717811846,
         0.03973576654331373
         ])
    parse_param_for_optimization(res, testWorld.controller)
    pydart.gui.viewer.launch_pyqt5(testWorld)


def flatworm_hard_backboneTest():
    pydart.init()
    testWorld = creature_simulators.FlatWormHardBackboneSimulator()
    testWorld.reset()
    res = np.array(
        [0.6562605961876525, 1.0471975511965976, 1.0471975511965976, 1.0471975511965976, 0.0, 1.0471975511965976,
         1.0471975511965976, 1.0471975511965976, 0.0, 1.0471975511965976, 1.0471975511965976, 1.0471975511965976, 0.0,
         1.0471975511965976, 1.0471975511965976, 0.0, 0.0, 0.9502349790277378, -1.0471975511965976, -1.0471975511965976,
         -0.052812457866623586, -1.0471975511965976, -1.0471975511965976, -1.0471975511965976, -0.9188914799022845,
         -1.0471975511965976, -1.0471975511965976, -1.0471975511965976, 0.0, -1.0471975511965976, 0.0,
         -1.0471975511965976, -1.0471975511965976, -1.0471975511965976, 0.0, -0.06384292343869931, -1.3849769254213469,
         -2.467087406779596, 2.254422141251807, -2.4511781617647364, -0.8885368555107817, -2.8736451895515582,
         3.141592653589793, 2.3851045143109593, 3.141592653589793, 1.3036415379546094, -0.8707266491264466,
         2.004086381932325, 0.40108587541984153, -1.0117670581289566, 1.783211939323375, -0.4164127143744856,
         1.549842361221608, -0.6608019856240022
         ])
    parse_param_for_optimization(res, testWorld.controller)
    pydart.gui.viewer.launch_pyqt5(testWorld)


if __name__ == '__main__':
    eelMultiResultsenario()
    # turtleFrontLimpTrack()`
    # turtleTest()
    # turtleCircleTest()
    # flatwormTest()
    # flatworm_hard_backboneTest()
    # simplified_flatwormTest()
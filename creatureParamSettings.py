import numpy as np


def simpleFishParam(controller):
    controller.joint_max[0] = np.pi / 3
    controller.joint_min[0] = -np.pi / 3

    controller.joint_max[1] = np.pi / 6
    controller.joint_min[1] = -np.pi / 6
    controller.omega[:] = 25
    # controller.phi[2] = 25*np.pi/2


def simpleFishWithPectoralParamRight(controller):
    controller.joint_amplitudes[0] = np.pi / 3
    controller.joint_amplitudes[1] = np.pi / 6
    controller.joint_amplitudes[2:] = np.pi / 3
    # controller.joint_amplitudes[4] = np.pi / 3
    controller.Kp[:] = 5
    controller.Kd[:] = 0.0005
    controller.omega[:] = 5
    controller.phi[2] = np.pi


def simpleFishWithPectoralParamLeft(controller):
    controller.joint_amplitudes[0] = np.pi / 3
    controller.joint_amplitudes[1] = np.pi / 6
    controller.joint_amplitudes[2:] = np.pi / 3
    # controller.joint_amplitudes[4] = np.pi / 3
    controller.Kp[:] = 5
    controller.Kd[:] = 0.0005
    controller.omega[:] = 5
    controller.phi[4] = np.pi


def simpleEelParam(controller):
    controller.joint_max[:] = np.pi / 2
    controller.joint_min[:] = -np.pi / 2
    controller.Kp[:] = 0.5
    controller.Kd[:] = 0.000005
    controller.omega[:] = 10
    for i in range(len(controller.phi)):
        controller.phi[i] -= (-1) ** i * i * np.pi / 4


def simpleTurtleParam(controller):
    controller.joint_max[:] = np.pi / 3
    controller.joint_min[:] = -np.pi / 3

    controller.Kp[:] = 0.05

    controller.Kd[:] = 0.00005

    controller.omega[:] = 25

    controller.phi[2] += np.pi
    controller.phi[6] += np.pi

def turtleCircParam(controller):
    controller.joint_max[:] = np.pi / 6
    controller.joint_min[:] = -np.pi / 6

    controller.joint_max[4] = np.pi
    controller.joint_min[4] = -np.pi
    controller.joint_max[5] = np.pi
    controller.joint_min[5] = -np.pi



def testFlatCreatureParam(controller):
    controller.joint_max[:] = 0
    controller.joint_max[0:2] = np.pi / 2
    controller.joint_max[2::] = -np.pi / 3

    controller.joint_min[:] = 0
    controller.joint_min[0:2] = -np.pi / 2
    controller.joint_min[2::] = np.pi / 3

def FlatCreatureParam(controller):
    controller.joint_max[:] = 0
    controller.joint_max[0:6] = np.pi / 12
    controller.joint_max[6:12] =-np.pi / 12
    controller.joint_max[12:18] =np.pi / 12

    controller.joint_min[:] = 0
    controller.joint_min[0:6] = -np.pi / 12
    controller.joint_min[6:12] = np.pi / 12
    controller.joint_min[12:18] = -np.pi / 12
def loopCreatureParam(controller):
    controller.joint_max[2] = -np.pi / 2
    controller.joint_min[2] = np.pi / 2
    controller.joint_max[3] = np.pi / 2
    controller.joint_min[3] = -np.pi / 2

    controller.omega[:] = 25

    # pass

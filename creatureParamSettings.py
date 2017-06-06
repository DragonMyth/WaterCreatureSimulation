import numpy as np


def simpleFishParam(controller):
    controller.joint_max[0] = np.pi / 3
    controller.joint_min[0] = -np.pi / 3

    controller.joint_max[1] = np.pi / 6
    controller.joint_min[1] = -np.pi / 6
    # controller.Kp[:] = 5
    # controller.Kd[:] = 0.0005
    # controller.omega[:] = 25
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
    controller.joint_amplitudes[:] = np.pi/2
    controller.Kp[:] = 0.5
    controller.Kd[:] = 0.000005
    controller.omega[:] = 10
    for i in range(len(controller.phi)):
        controller.phi[i] -= (-1)**i*i*np.pi/4

def simpleTurtleParam(controller):
    controller.joint_amplitudes[:] = np.pi/3
    controller.joint_amplitudes[0] = 0

    controller.Kp[:] = 0.5
    controller.Kp[0] = 0

    controller.Kd[:] = 0.0005
    controller.Kd[0] = 0

    controller.omega[:] = 10

    controller.phi[1] += np.pi
    controller.phi[5] += np.pi


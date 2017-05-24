import numpy as np


def simpleFishParam(controller):
    controller.joint_amplitudes[0] = np.pi / 3
    controller.joint_amplitudes[1] = np.pi / 6
    controller.Kp[:] = 5
    controller.Kd[:] = 0.0005
    controller.omega[:] = 25
    # controller.phi[2] = 25*np.pi/2

def simpleFishWithPectoralParam(controller):
    controller.joint_amplitudes[0] = np.pi / 3
    controller.joint_amplitudes[1] = np.pi / 6
    controller.joint_amplitudes[2] = np.pi / 3
    controller.joint_amplitudes[4] = -np.pi / 3
    controller.Kp[:] = 5
    controller.Kd[:] = 0.0005
    controller.omega[:] = 25
    # controller.phi[2] = 25*np.pi/2

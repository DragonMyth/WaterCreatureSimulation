import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import testSimResults

def initData(filename):
    res = []
    with open(filename) as input_file:
        for line in input_file:
            a = line.replace(' ', '').split(',')
            res.append(a[1::])
    res = np.asarray(res)
    return res


def split_parameter(x):
    """
    :param x: Setting of a parameter set
    :return:  Splitted version of the array
    """
    # x_split = np.split(x, 3)
    # input_shape = x.shape()

    joint_max = np.ndarray([len(x), len(x[0]) / 3])
    joint_min = np.ndarray([len(x), len(x[0]) / 3])
    phi = np.ndarray([len(x), len(x[0]) / 3])

    for i in range(len(x)):
        joint_max[i] = np.split(x[i], 3)[0]
        joint_min[i] = np.split(x[i], 3)[1]
        phi[i] = np.split(x[i], 3)[2]

    # phi = x_split[2]
    return joint_max, joint_min, phi


def plot3dPhi(phi_list):
    X = np.linspace(-4, 4, len(phi_list[0]))
    Z = np.linspace(-2, 3, len(phi_list))
    fig, ax2 = plt.subplots(1, 1, figsize=(8, 8), subplot_kw={'projection': '3d'})

    for i in range(len(phi_list)):
        ax2.plot(X, phi_list[i], Z[i], zdir='y', label='NO. ' + str(i+1))
    ax2.set_title("Phi of All Joints in All Results")
    ax2.legend()

    plt.show()
def plotTurtleLimpT():
    taus,T = testSimResults.turtleFrontLimpTrack()

    fig, ax = plt.subplots(1,1,figsize=(10,10),subplot_kw={'projection':'3d'})

    Z = np.linspace(-0.2,0.2,len(taus[0]))
    for j in range(len(taus[0])):
        ax.plot(T,taus[:,j],zs=Z[j],zdir = 'y',label='Joint No. ' + str(j+1))


    ax.set_title("Tau of All Joints vs Time")
    ax.legend()

    plt.show()


if __name__ == '__main__':
    res = initData('optimaEel.txt')

    joint_max, joint_min, phi = split_parameter(res)
    # plot3dPhi(phi)
    plotTurtleLimpT()
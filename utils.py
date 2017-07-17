import scipy as scp
import numpy as np
from scipy import optimize

def least_square_circle(X, Y):
    def calc_R(xc, yc):
        """ calculate the distance of each 2D points from the center c=(xc, yc) """
        return np.sqrt((X - xc) ** 2 + (Y - yc) ** 2)

    def f(c):
        """ calculate the algebraic distance between the 2D points and the mean circle centered at c=(xc, yc) """
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    def Df(c):
        """ Jacobian of f_2b
            The axis corresponding to derivatives must be coherent with the col_deriv option of leastsq"""
        xc, yc = c
        df_dc = scp.empty((len(c), X.size))

        Ri = calc_R(xc, yc)
        df_dc[0] = (xc - X) / Ri  # dR/dxc
        df_dc[1] = (yc - Y) / Ri  # dR/dyc
        df_dc = df_dc - df_dc.mean(axis=1)[:, scp.newaxis]
        return df_dc

    center_estimate = np.mean(X), np.mean(Y)
    center, ier = optimize.leastsq(f,center_estimate,Dfun=Df,col_deriv=True,maxfev=10000)

    xc, yc = center
    Ri = calc_R(xc,yc)
    R = Ri.mean()

    return xc,yc,R


# X = np.array([1,0,-1])
# Y = np.array([0,1,0])
#
# xc,yc,R = least_square_circle(X,Y)
# print(xc,yc)
# print(R)


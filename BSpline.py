from __future__ import division 
import numpy as np
from sympy import *
import math
import scipy as sp
from scipy.optimize import fmin
from numpy import ma
import scipy.integrate as integrate
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import rc

#Define spline basis of order k with nots t_i
#in the variable x.
#x and t_i share the same initial and final values.
#Returns shape (len(x), len(t)-k-1).

def Bsplined2(x, t, k):
    if isinstance(x, float) or isinstance(x, int):
        x = [x]
    t = np.array(t)
    kmax = k
    if kmax > len(t)-2:
        raise Exception("Input error in Nbspl: require that k < len(t)-2")
    x = np.array(x)[:, np.newaxis]
    N = 1.0 * ((x >= t[: -1]) & (x < t[1:]))
    dN = np.zeros_like(N)
    d2N = np.zeros_like(N)
    for k in range(1, kmax+1):
        dt = t[k:] - t[:-k]
        _dt = dt.copy()
        _dt[dt != 0] = 1./dt[dt != 0]
        d2N = d2N[:,:-1]*(x-t[:-k-1])*_dt[:-1] - d2N[:,1:]*(x-t[k+1:])*_dt[1:] \
              + 2*dN[:,:-1]*_dt[:-1] - 2*dN[:,1:]*_dt[1:]
        dN = dN[:,:-1]*(x-t[:-k-1])*_dt[:-1] - dN[:,1:]*(x-t[k+1:])*_dt[1:] \
             + N[:,:-1]*_dt[:-1] - N[:,1:]*_dt[1:]
        N = N[:,:-1]*(x-t[:-k-1])*_dt[:-1] - N[:,1:]*(x-t[k+1:])*_dt[1:]
    return N, dN, d2N


x = np.linspace(0.0, 50, 100)
t = [0.1, 0.1, 0.1, 10, 15, 40, 50, 50, 50]
parameters = np.array([[1, 10, 5, 5, 1], [3,4,5,6,7], [5,7,8,3,6]])


#Define the field Phi as
#a superposition of Bsplines plus
#plus a linear contribution
#in order to have Phi(0)=Phi_true
#and Phi(x_bar)=Phi_false

def Phi(x, parameters):
    res = 0
    for basis, coeff in zip(Bsplined2(x,t,3)[0].T, parameters):
        res += coeff * basis
    if isinstance(x, int) or isinstance(x, float):
        if x != 0:
            res += 1 - x/np.abs(x)
        else :
            res += 1
    else:
        if x[-1] != 0:
            res += 1 - x/x[-1]
        else:
            res += 1
    return res


#Make the plot.
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
plt.plot(x, Phi(x, parameters[0]), label = '$coeff: {}$'.format(parameters[0]))
plt.ylabel('$\phi(x)$')
plt.xlabel('$x$')
plt.legend()
plt.show()

def integral(parameters):
    integral = integrate.quad(Phi, 0, 10, args = parameters)
    return integral[0]

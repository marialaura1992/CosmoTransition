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
t = [0.1, 0.1, 10, 20, 50, 50]

#Define the field Phi as
#a superposition of Bsplines plus
#plus a linear contribution
#in order to have Phi(0)=Phi_true
#and Phi(x_bar)=Phi_false

#print(len(Bsplined2(x,t,3)[0][1]))

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


#Try to solve the equation for the bouble.

def DPhi(x, parameters):
    res = 0
    for basis, coeff in zip(Bsplined2(x,t,3)[1].T, parameters):
        res += coeff * basis
    return res


def D2Phi(x, parameters):
    res = 0
    for basis, coeff in zip(Bsplined2(x,t,3)[2].T, parameters):
        res += coeff * basis
    return res

v = 1


def DV(x, parameters):
    return(Phi(x, parameters) * (Phi(x, parameters)**2 - v**2))


def EOM2(x, parameters):
    res = D2Phi(x, parameters) + (2/x) * DPhi(x, parameters) - DV(x, parameters)
    return(4*np.pi * x**2 * np.abs(res)**2)


def Functional(parameters):
    res = np.array([])
    result = np.array([])
    for parameters in parameters:
        res = np.append(res, integrate.quad(EOM2, 0.1, 50, args = parameters))
    for i in range(len(res)):
        if i % 2 == 0:
            result = np.append(result, res[i])
    return(result)


X = np.zeros((len(t)-4, 50))
for i in range(len(x)):
    X[i] = np.linspace(0, 10, 50)


cartesian_prod = [np.array([a,b]) for a in X[0] for b in X[1]]
y = Functional(cartesian_prod)
pol = np.polyfit(cartesian_prod, y, 6)
print(pol)


#opt = sp.optimize.minimize(Functional, [1, 1, 1, 1, 1])
#print(opt)

#print(integral([*opt['x']]))


#parameters = np.array([1,1,1,1,1])

#print(Functional(parameters))



'''
plt.plot(x, Phi(x, [*opt['x']]))
plt.plot(x, DPhi(x, [*opt['x']]))
plt.plot(x, D2Phi(x, [*opt['x']]))
plt.show()
'''



'''
#Make the plot.
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
plt.plot(x, Phi(x, [*opt['x']]), label = '$coeff: {}$'.format(parameters[0]))
plt.ylabel('$\phi(x)$')
plt.xlabel('$x$')
plt.legend()
plt.show()


def integral(parameters):
    integral = integrate.quad(Phi, 0, 10, args = parameters)
    return integral[0]


opt = sp.optimize.minimize(integral, [1, 1, 1, 1, 1])


print(integral([*opt['x']]))
'''



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

#Supposing a Z2 symmetric potential
#V(\phi)= \lambda/4 * (\phi**2 - v**2)**2
# with v = \mu/\sqrt(\lambda)

v = 1

def Phi(rho, alpha):
    return v * (1 - np.tanh(rho/alpha)) 


def DPhi(rho, alpha):
    return (v * (-1/alpha) * np.cosh(rho/alpha)**(-2))


def D2Phi(rho, alpha):
    return(2 * v/(alpha**2) * np.tanh(rho/alpha) * np.cosh(rho/alpha)**(-2))


def DV(rho, alpha):
    return(Phi(rho, alpha) * (Phi(rho, alpha)**2 - v**2))


def EOM2(rho, alpha):
    res = D2Phi(rho, alpha) + (2/rho) * DPhi(rho, alpha) - DV(rho, alpha)
    return(4*np.pi * rho**2 * np.abs(res)**2)


def Functional(alpha):
    res = np.array([])
    result = np.array([])
    for alpha in alpha:
        res = np.append(res, integrate.quad(EOM2, 0.1, 50, args = alpha))
    for i in range(len(res)):
        if i % 2 == 0:
            result = np.append(result, res[i])
    return(result)


alpha = np.linspace(-0.01, 10, 100)
fa = Functional(alpha)


#Analytic function to be fitted to the Functional

def analytic(alpha, a, b, c, d, e):
    poly = a*alpha**4 + b*alpha**3 +\
           c*alpha**2 + d*alpha + e
    return poly


p0 = np.array([0,0,0,0,0])
#fit = sp.optimize.curve_fit(analytic, alpha, fa, p0)


def f(alpha):
    return analytic(alpha, *fit[0])

#opt = sp.optimize.minimize(f, [10])
#print(opt)

plt.plot(alpha, Functional(alpha), label = 'analytic')
#plt.plot(alpha, analytic(alpha, *fit[0]), label = 'fit')
plt.xlabel('alpha')
plt.ylabel('F(alpha)')
plt.legend()
plt.show()


'''
#Armonic oscillator with frequency**2 = 4
#Ansatz solution is Phi = cos(alpha * rho)
#where alpha is the parameter to be optimized

omega = 4

def Phi(rho, alpha):
    return np.cos(alpha * rho)


def D2Phi(rho, alpha):
    return  -alpha**2 * np.cos(alpha * rho)


def EOM(rho, alpha):
    res = D2Phi(rho, alpha) + omega * Phi(rho, alpha)
    return (np.abs(res)**2)


def Functional(alpha):
    res = np.array([])
    result = np.array([])
    for alpha in alpha:
        res = np.append(res, integrate.quad(EOM, 0., 50, args = alpha))
    for i in range(len(res)):
        if i % 2 == 0:
            result = np.append(result, res[i])
    return(result)

#Derivative of the functional with respect
#to alpha.

def DalphaIntegrand(rho, alpha):
    deriv = (2 * ( D2Phi(rho, alpha) + omega * Phi(rho, alpha))) *\
          (- 2 * alpha * Phi(rho, alpha) + rho *\
           (alpha**2 * np.sin(rho * alpha) -\
            omega * np.sin(rho * alpha)))
    return deriv


def DF(alpha):
    res = np.array([])
    result = np.array([])
    for alpha in alpha:
        res = np.append(res, integrate.quad(DalphaIntegrand,\
                                            0, 100, args = alpha))
    for i in range(len(res)):
        if i % 2 == 0:
            result = np.append(result, res[i])
    return(result)


alpha = np.linspace(-10, 10, 1000)
fa = Functional(alpha)


#Analytic function to be fitted to the Functional

def analytic(alpha, a, b, c, d, e):
    poly = a*alpha**4 + b*alpha**3 +\
           c*alpha**2 + d*alpha + e
    return poly


p0 = np.array([0,0,0,0,0])
fit = sp.optimize.curve_fit(analytic, alpha, fa, p0)


def f(alpha):
    return analytic(alpha, *fit[0])

opt = sp.optimize.minimize(f, [10])
print(opt)

plt.plot(alpha, Functional(alpha))
plt.plot(alpha, analytic(alpha, *fit[0]))
plt.show()


'''


# 3D PLOTS

'''
fig = plt.figure()
ax = fig.gca(projection='3d')


X = np.arange(0, 100, 0.5)
Y = np.arange(-100, 100, 2)
X, Y = np.meshgrid(X, Y)

#R = np.sqrt(X**2 + Y**2)

#Z = integrate.quad(R)
#Z = np.sin(R)

# Plot the surface.
surf = ax.plot_surface(X, Y, EOM2(X,Y), cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
#ax.set_zlim(-6, 6)
#ax.zaxis.set_major_locator(LinearLocator(10))
#ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.xlabel('rho')
plt.ylabel('alpha')
plt.show()

'''

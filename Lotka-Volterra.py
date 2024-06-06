import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate

alpha = 1
beta = 2
delta = 1.5
gamma = 1
x0 = 5
y0 = 2

def derivative(X, t, alpha, beta, delta, gamma):
    x, y = X
    dotx = x * (alpha - beta * y)
    doty = y * (-delta + gamma * x)
    return np.array([dotx, doty])

Nt = 1000
tmax = 30.
t = np.linspace(0.,tmax, Nt)
X0 = [x0, y0]
res = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
x, y = res.T

def RK4(func, X0, t, alpha,  beta, delta, gamma):
    """
    Runge Kutta 4 solver.
    """
    dt = t[1] - t[0]
    nt = len(t)
    X  = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    return X

Xrk4 = RK4(derivative, X0, t, alpha,  beta, delta, gamma)
plt.figure()
plt.title("Мeтод RK4")
plt.plot(t, Xrk4[:, 0], '-b', label = 'Зайці')
plt.plot(t, Xrk4[:, 1], '-r', label = "Вовки")
plt.grid()
plt.xlabel("Час")
plt.ylabel('Чисельність')
plt.legend(loc = "best")

plt.show();
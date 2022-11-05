import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
from mpl_toolkits import mplot3d
from matplotlib import cm

## Problem 1

def rhsfunc1(t, y, eps):
    f1 = y[1]
    f2 = (t**2 - eps)*y[0]
    return np.array([f1, f2])

xp = [-4, 4]
tol = 1e-6
A = 1

x_evals = np.linspace(-4, 4, 81)
eps_start = 0
eigenvalues1 = np.zeros(5)
eigenfunctions1 = np.zeros([81, 5])
phi = np.zeros([81, 5])
for mode in range(5):
    eps = eps_start
    deps = 1
    
    for i in range(1000):
        y0 = np.array([A, A*np.sqrt(4**2 - eps)])
        sol = scipy.integrate.solve_ivp(lambda x,y: rhsfunc1(x, y, eps), xp, y0, t_eval = x_evals)

        if (np.abs(sol.y[1, -1] + np.sqrt(4**2 - eps)*sol.y[0, -1])) < tol:
            eigenfunctions1[:, mode] = sol.y[0, :]
            eigenvalues1[mode] = eps
            break

        if (-1)**(mode)*(sol.y[1, -1] + np.sqrt(4**2 - eps)*sol.y[0, -1]) > 0:
            eps = eps + deps
        else:
            eps = eps - deps/2
            deps = deps/2
    
    eps_start = eps + 0.1
    eig_norm = np.trapz(eigenfunctions1[:, mode]**2, x = x_evals)
    eigenfunctions1[:, mode] = eigenfunctions1[:, mode]/np.sqrt(eig_norm)
    
    plt.plot(sol.t, eigenfunctions1[:, mode], linewidth=2)
    plt.plot(sol.t, 0*sol.t, 'k')

    phi[:, mode] = eigenfunctions1[:, mode]
    eigenfunctions1[:, mode] = abs(eigenfunctions1[:, mode])

A1 = eigenfunctions1[:, 0].reshape(-1, 1)
A2 = eigenfunctions1[:, 1].reshape(-1, 1)
A3 = eigenfunctions1[:, 2].reshape(-1, 1)
A4 = eigenfunctions1[:, 3].reshape(-1, 1)
A5 = eigenfunctions1[:, 4].reshape(-1, 1)
A6 = eigenvalues1.reshape(1, -1)

# 3D Plot
x = np.linspace(-4, 4, 81)
t = np.linspace(0, 5, 100)

fig = plt.figure()
ax = plt.axes(projection = '3d')
density = (phi[:, 1].reshape(-1, 1) * np.cos(eigenvalues1[1]*t/2)).T

X, T = np.meshgrid(x, t)
surf = ax.plot_surface(X, T, density, cmap = cm.hsv, rstride=1, cstride=1)
fig.colorbar(surf, pad = 0.2)
ax.contour(X, T, density, cmap = cm.hsv, offset = -0.6)

ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel(r'$\psi_2(x, t)$')
plt.title(r'Time Evolution of Probability Density Function $\psi_2(x, t)$')

## Problem 2

xp = [-4, 4]
x_evals = np.linspace(-4, 4, 81)

D1 = -2*np.ones(79)
U1 = np.ones(78)
L1 = np.ones(78)
matrix1 = (np.diag(D1) + np.diag(U1, 1) + np.diag(L1, -1))

matrix1[0, 0] += 4/3
matrix1[0, 1] += -1/3
matrix1[-1, -1] += 4/3
matrix1[-1, -2] += -1/3
matrix1 = -matrix1/(0.1**2)

D2 = x_evals[1:-1]
matrix2 = np.diag(D2**2)

matrix = matrix1 + matrix2
eigenvals, eigenfuncs = np.linalg.eig(matrix)

sol2 = np.zeros([81, 5])
indices = np.argsort(eigenvals)
for i, index in enumerate(indices[0:5]):
    temp = np.append([(4*eigenfuncs[0, index] - eigenfuncs[1, index])/(3 + 2*0.1*np.sqrt(4**2 - eigenvals[index]))], eigenfuncs[:, index])
    sol2[:, i] = np.append(temp, [(4*temp[79] - temp[78])/(3 + 2*0.1*np.sqrt(4**2 - eigenvals[index]))])
    sol2_norm = np.trapz((sol2[:, i])**2, x = x_evals)
    sol2[:, i] = sol2[:, i]/np.sqrt(sol2_norm)
    sol2[:, i] = abs(sol2[:, i])

A7 = sol2[:, 0].reshape(-1, 1)
A8 = sol2[:, 1].reshape(-1, 1)
A9 = sol2[:, 2].reshape(-1, 1)
A10 = sol2[:, 3].reshape(-1, 1)
A11 = sol2[:, 4].reshape(-1, 1)
A12 = np.array([eigenvals[indices[0]], eigenvals[indices[1]], eigenvals[indices[2]], eigenvals[indices[3]], eigenvals[indices[4]]]).reshape(1, -1)

## Problem 3

def rhsfunc2(t, y, eps, gam):
    f1 = y[1]
    f2 = (gam*abs(y[0])**2 + t**2 - eps)*y[0]
    return np.array([f1, f2])

xp = [-3, 3]
tol = 1e-5
A = 1e-3

x_evals3 = np.linspace(-3, 3, 61)
eigenvalues3 = np.zeros(4)
eigenfunctions3 = np.zeros([61, 4])

for i, gamma in enumerate([0.05, -0.05]):
    eps_start = 0
    
    for mode in range(2):
        eps = eps_start
        deps = 0.01
        
        for j in range(1000):
            y0 = np.array([A, A*np.sqrt(3**2 - eps)])
            sol = scipy.integrate.solve_ivp(lambda x,y: rhsfunc2(x, y, eps, gamma), xp, y0, t_eval = x_evals3)
            
            eig_norm3 = np.trapz(sol.y[0, :]**2, x = x_evals3)
            if ((np.abs(sol.y[1, -1] + np.sqrt(3**2 - eps)*sol.y[0, -1]) < tol) and (np.abs(eig_norm3 - 1) < tol)):
                eigenfunctions3[:, mode + 2*i] = abs(sol.y[0, :])
                eigenvalues3[mode + 2*i] = eps
                break
            else:
                A = A/np.sqrt(eig_norm3)
            
            y0 = np.array([A, A*np.sqrt(3**2 - eps)])
            sol = scipy.integrate.solve_ivp(lambda x,y: rhsfunc2(x, y, eps, gamma), xp, y0, t_eval = x_evals3)
            
            eig_norm3 = np.trapz(sol.y[0, :]**2, x = x_evals3)
            if ((np.abs(sol.y[1, -1] + np.sqrt(3**2 - eps)*sol.y[0, -1]) < tol) and (np.abs(eig_norm3 - 1) < tol)):
                eigenfunctions3[:, mode + 2*i] = abs(sol.y[0, :])
                eigenvalues3[mode + 2*i] = eps
                break           
            elif (-1)**(mode)*(sol.y[1, -1] + np.sqrt(3**2 - eps)*sol.y[0, -1]) > 0:
                eps = eps + deps
            else:
                eps = eps - deps/2
                deps = deps/2
            
        eps_start = eps + 0.1

A13 = eigenfunctions3[:, 0].reshape(-1, 1)
A14 = eigenfunctions3[:, 1].reshape(-1, 1)
A15 = np.array([eigenvalues3[0], eigenvalues3[1]]).reshape(1, -1)
A16 = eigenfunctions3[:, 2].reshape(-1, 1)
A17 = eigenfunctions3[:, 3].reshape(-1, 1)
A18 = np.array([eigenvalues3[2], eigenvalues3[3]]).reshape(1, -1)

# print("A1:", A1)
# print("A2:", A2)
# print("A3:", A3)
# print("A4:", A4)
# print("A5:", A5)
# print("A6:", A6)
# print("A7:", A7)
# print("A8:", A8)
# print("A9:", A9)
# print("A10:", A10)
# print("A11:", A11)
# print("A12:", A12)
# print("A13:", A13)
# print("A14:", A14)
# print("A15:", A15)
# print("A16:", A16)
# print("A17:", A17)
# print("A18:", A18)
# plt.show()
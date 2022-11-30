import numpy as np
import scipy.integrate
import scipy.sparse
import matplotlib.pyplot as plt
import time

# Problem 1

## a

x_evals = np.arange(-10, 10, 0.1)
n = 200
b = np.ones((n))
Bin = np.array([-b, b, -b, b])
d = np.array([-1,1, n-1, 1-n])
matrix1 = scipy.sparse.spdiags(Bin, d, n, n, format='csc')/(2*0.1)

A1 = matrix1.todense()

## b

u0 = np.exp(-(x_evals-5)**2)
def advection1(t, u, A, c):
    return -c*(A@u)

sol1 = scipy.integrate.solve_ivp(advection1, [0, 10], u0, t_eval = np.arange(0, 10 + 0.5, 0.5), args = (matrix1, -0.5))

A2 = sol1.y

## c

def advection2(t, u, A, cFunc, x):
    c = cFunc(t, x)
    return c*(A@u)

def cFunc(t, x):
    return (1 + 2*np.sin(5*t) - np.heaviside(x - 4, 0))

sol2 = scipy.integrate.solve_ivp(advection2, [0, 10], u0, t_eval = np.arange(0, 10 + 0.5, 0.5), args = (matrix1, cFunc, x_evals))

A3 = sol2.y

## 3D Plot

X, T = np.meshgrid(x_evals, sol1.t)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, T, sol1.y.T, cmap='magma')
ax.plot3D(x_evals, 0*x_evals, u0,'-r',linewidth=5)
ax.set_xlabel('x')
ax.set_ylabel('time')
ax.set_zlabel('height')
ax.set_title('Constant Velocity Field')

X, T = np.meshgrid(x_evals, sol2.t)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, T, sol2.y.T, cmap='magma')
ax.plot3D(x_evals, 0*x_evals, u0,'-r',linewidth=5)
ax.set_xlabel('x')
ax.set_ylabel('time')
ax.set_zlabel('height')
ax.set_title('Variable (Nonconstant) Velocity Field')

# Problem 2

## a

m = 64
n = m**2
e1 = np.ones(n)
e0 = e1.copy()
e0[0] = -0.5
Low1 = np.tile(np.concatenate((np.ones(m-1), [0])), (m,))
Low2 = np.tile(np.concatenate(([1], np.zeros(m-1))), (m,))
Up1 = np.roll(Low1, 1)
Up2 = np.roll(Low2, m-1)
matrix2 = scipy.sparse.spdiags([e1, e1, Low2, Low1, -4*e0, Up1, Up2, e1, e1], [-(n-m), -m, -m+1, -1, 0, 1, m-1, m, (n-m)], n, n, format='csc')/((20/64)**2)

A4 = matrix2.todense()

m = 64
n = m**2
e1 = np.ones(n)
matrix3 = scipy.sparse.spdiags([e1, -e1, e1, -e1], [-(n-m), -m, m, (n-m)], n, n, format='csc')/(2*(20/64))

A5 = matrix3.todense()

m = 64
n = m**2
e1 = np.ones(n)
Lower1 = np.tile(np.concatenate((np.ones(m-1), [0])), (m,))
Lower2 = np.tile(np.concatenate(([1], np.zeros(m-1))), (m,))
Upper1 = np.roll(Lower1, 1)
Upper2 = np.roll(Lower2, m-1)
matrix4 = scipy.sparse.spdiags([Lower2, -Lower1, Upper1, -Upper2], [1-m, -1, 1, m-1], n, n, format='csc')/(2*(20/64))

A6 = matrix4.todense()

## b

x_span = np.linspace(-10, 10, num=64, endpoint=False)
y_span = np.linspace(-10, 10, num=64, endpoint=False)
X, Y = np.meshgrid(x_span, y_span)
w0 = np.exp(-2*X**2 - ((Y**2)/20))
w0_temp = (w0.T).flatten()
t_span = np.arange(0, 4 + 0.5, 0.5)

def omegaFunc1(t, omega):
    psi = scipy.sparse.linalg.spsolve(matrix2, omega)
    w_t = ((0.001*matrix2@omega) - (matrix3@psi)*(matrix4@omega) + (matrix4@psi)*(matrix3@omega))
    return w_t

start1 = time.time()
sol3 = scipy.integrate.solve_ivp(omegaFunc1, [0, 4], w0_temp, t_eval = t_span)
end1 = time.time()
# print("Gaussian elimination computation time:", end1 - start1)

A7 = sol3.y.T

LU = scipy.sparse.linalg.splu(matrix2)
def omegaFunc2(t, omega):
    psi = LU.solve(omega)
    w_t = ((0.001*matrix2@omega) - (matrix3@psi)*(matrix4@omega) + (matrix4@psi)*(matrix3@omega))
    return w_t

start2 = time.time()
sol4 = scipy.integrate.solve_ivp(omegaFunc2, [0, 4], w0_temp, t_eval = t_span)
end2 = time.time()
# print("LU decomposition computation time:", end2 - start2)
# print("Ratio between the methods:", (end1 - start1)/(end2 - start2))

A8 = sol4.y.T
A9 = np.zeros((9, 64, 64))
for i in range(9):
    A9[i] = A8[i].reshape(64, 64)

# print("A1:", A1)
# print("A2:", A2)
# print("A3:", A3)
# print("A4:", A4)
# print("A5:", A5)
# print("A6:", A6)
# print("A7:", A7)
# print("A8:", A8)
# print("A9:", A9)
# plt.show()
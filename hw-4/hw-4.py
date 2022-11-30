import numpy as np
import scipy.integrate
import scipy.sparse
import scipy.optimize

# Problem 1

x = np.linspace(-10, 10, 128, endpoint=False)
dx = 20/128
t = np.linspace(0, 2, 501)
dt = 2/500
CFL = (2*dt)/(dx)**2
g1 = lambda z: 1 + (CFL/12)*(32*np.cos(z) - 2*np.cos(2*z) - 30)
A1 = abs(g1(1))

min_index1 = scipy.optimize.fminbound(lambda z: -abs(g1(z)), -np.pi, np.pi)
A2 = g1(min_index1)

n = 128
e = np.ones(n)
matrix1 = scipy.sparse.spdiags([16*e, -e, -e, 16*e, -30*e, 16*e, -e, -e, 16*e], [1-n, 2-n, -2, -1, 0, 1, 2, n-2, n-1], n, n, format='csc')/12

A3 = matrix1.todense()
A4 = 0

sol1 = np.zeros((len(x), len(t)))
u0 = 10*np.cos(2*np.pi*x/10) + 30*np.cos(8*np.pi*x/10)
sol1[:, 0] = u0
for i in range(int(2/dt)):
    u1 = u0 + CFL*(matrix1@u0)
    u0 = u1
    sol1[:, i+1] = u1

A5 = sol1[:, -1].reshape(-1, 1)

# Problem 2

g2 = lambda z: (1 + CFL*(np.cos(z) - 1))/(1 - CFL*(np.cos(z) - 1))
min_index2 = scipy.optimize.fminbound(lambda z: -abs(g2(z)), -np.pi, np.pi)

A6 = g2(min_index2)

matrix2 = scipy.sparse.eye(128, format='csc') - (CFL/2)*scipy.sparse.spdiags([e, e, -2*e, e, e], [1-n, -1, 0, 1, n-1], n, n, format='csc')
matrix3 = scipy.sparse.eye(128, format='csc') + (CFL/2)*scipy.sparse.spdiags([e, e, -2*e, e, e], [1-n, -1, 0, 1, n-1], n, n, format='csc')

A7 = matrix2.todense()
A8 = matrix3.todense()

sol2 = np.zeros((len(x), len(t)))
v0 = 10*np.cos(2*np.pi*x/10) + 30*np.cos(8*np.pi*x/10)
sol2[:, 0] = v0
PLU = scipy.sparse.linalg.splu(matrix2)
for i in range(int(2/dt)):
    v1 = PLU.solve(matrix3@v0)
    v0 = v1
    sol2[:, i+1] = v1

A9 = sol2[:, -1].reshape(-1, 1)

sol3 = np.zeros((len(x), len(t)))
w0 = 10*np.cos(2*np.pi*x/10) + 30*np.cos(8*np.pi*x/10)
sol3[:, 0] = w0
for i in range(int(2/dt)):
    w1 = scipy.sparse.linalg.bicgstab(matrix2, matrix3@w0)[0]
    w0 = w1
    sol3[:, i+1] = w1

A10 = sol3[:, -1].reshape(-1, 1)

# Problem 3

exact1 = np.loadtxt('hw-4\exact_128.csv').reshape(-1, 1)
exact2 = np.loadtxt('hw-4\exact_256.csv').reshape(-1, 1)

A11 = np.linalg.norm(exact1 - A5)
A12 = np.linalg.norm(exact1 - A9)

x = np.linspace(-10, 10, 256, endpoint=False)
dx = 20/256
t = np.linspace(0, 2, 2001)
dt = (2/500)/4
CFL = (2*dt)/(dx)**2

n = 256
e = np.ones(n)
matrix1 = scipy.sparse.spdiags([16*e, -e, -e, 16*e, -30*e, 16*e, -e, -e, 16*e], [1-n, 2-n, -2, -1, 0, 1, 2, n-2, n-1], n, n, format='csc')/12

sol1 = np.zeros((len(x), len(t)))
u0 = 10*np.cos(2*np.pi*x/10) + 30*np.cos(8*np.pi*x/10)
sol1[:, 0] = u0
for i in range(int(2/dt)):
    u1 = u0 + CFL*(matrix1@u0)
    u0 = u1
    sol1[:, i+1] = u1

A13 = np.linalg.norm(exact2 - sol1[:, -1].reshape(-1, 1))

matrix2 = scipy.sparse.eye(256, format='csc') - (CFL/2)*scipy.sparse.spdiags([e, e, -2*e, e, e], [1-n, -1, 0, 1, n-1], n, n, format='csc')
matrix3 = scipy.sparse.eye(256, format='csc') + (CFL/2)*scipy.sparse.spdiags([e, e, -2*e, e, e], [1-n, -1, 0, 1, n-1], n, n, format='csc')

sol2 = np.zeros((len(x), len(t)))
v0 = 10*np.cos(2*np.pi*x/10) + 30*np.cos(8*np.pi*x/10)
sol2[:, 0] = v0
PLU = scipy.sparse.linalg.splu(matrix2)
for i in range(int(2/dt)):
    v1 = PLU.solve(matrix3@v0)
    v0 = v1
    sol2[:, i+1] = v1

A14 = np.linalg.norm(exact2 - sol2[:, -1].reshape(-1, 1))

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
# plt.show()
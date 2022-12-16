import numpy as np
import scipy.integrate
from scipy.fft import fft2, ifft2
from cheb import *

# Problem 1

x = np.linspace(-10, 10, 64, endpoint=False)
y = np.linspace(-10, 10, 64, endpoint=False)
X, Y = np.meshgrid(x, y)
m = 3
alpha = 0
n = 64
u = (np.tanh(np.sqrt(X**2 + Y**2)) - alpha)*np.cos(m*np.angle(X + 1j*Y) - np.sqrt(X**2 + Y**2))
v = (np.tanh(np.sqrt(X**2 + Y**2)) - alpha)*np.sin(m*np.angle(X + 1j*Y) - np.sqrt(X**2 + Y**2))
A1 = X
A2 = u

u0 = fft2(u)
v0 = fft2(v)
A3 = u0.real

u0 = u0.reshape(-1, 1, order='F')
v0 = v0.reshape(-1, 1, order='F')
vec0 = np.concatenate((u0, v0))
A4 = vec0.imag

def rhs1(t, n, vec, beta, KX, KY):
    u_hat = vec[:4096].reshape(n, n, order='F')
    v_hat = vec[4096:].reshape(n, n, order='F')

    u = ifft2(u_hat)
    v = ifft2(v_hat)

    u_nl = u - u**3 - u*v**2 + beta*(v*u**2 + v**3)
    v_nl = -beta*(u**3 + u*v**2) - v + v*u**2 + v**3

    u_t = fft2(u_nl) - 0.1*((KX**2)*u_hat + (KY**2)*u_hat)
    v_t = fft2(v_nl) - 0.1*((KX**2)*v_hat + (KY**2)*v_hat)

    u_t = u_t.reshape(n**2, order='F')
    v_t = v_t.reshape(n**2, order='F')
    rhs = np.concatenate((u_t, v_t), axis=0)

    return rhs

t_span = np.linspace(0, 25, 51)
r1 = np.arange(0, n/2, 1)
r2 = np.arange(-n/2, 0, 1)
kx = (2*np.pi/20)*np.concatenate((r1, r2))
ky = kx.copy()
KX, KY = np.meshgrid(kx, ky)
beta = 1

sol1 = scipy.integrate.solve_ivp(lambda t, vec: rhs1(t, n, vec, beta, KX, KY), [0, 25], np.squeeze(vec0), t_eval = t_span)
A5 = sol1.y.real
A6 = sol1.y.imag
A7 = sol1.y.real[:4096][:, 4].reshape(-1, 1)
A8 = sol1.y.real[:4096][:, 4].reshape(n, n, order='F')
A9 = ifft2(sol1.y[:4096][:, 4].reshape(n, n, order='F')).real

# Problem 2

m = 2
alpha = 1
n = 30

[D, x] = cheb(n)
x = 10*x.reshape(n + 1)
D2 = D@D
D2 = D2[1:-1, 1:-1]/100

I = np.eye(len(D2))
Lap = np.kron(D2, I) + np.kron(I, D2)
A10 = Lap

x2 = x[1:-1]
y2 = x2.copy()
[X, Y] = np.meshgrid(x2, y2)
A11 = Y

u = (np.tanh(np.sqrt(X**2 + Y**2)) - alpha)*np.cos(m*np.angle(X + 1j*Y) - np.sqrt(X**2 + Y**2))
v = (np.tanh(np.sqrt(X**2 + Y**2)) - alpha)*np.sin(m*np.angle(X + 1j*Y) - np.sqrt(X**2 + Y**2))
A12 = v

u = u.reshape(-1, 1, order='F')
v = v.reshape(-1, 1, order='F')
vec1 = np.concatenate((u, v), axis=0)
A13 = vec1

def rhs2(t, vec, beta, Lap):
    u = vec[:841]
    v = vec[841:]

    u_nl = u - u**3 - u*v**2 + beta*(v*u**2 + v**3)
    v_nl = -beta*(u**3 + u*v**2) - v + v*u**2 + v**3

    u_t = u_nl + 0.1*(Lap@u)
    v_t = v_nl + 0.1*(Lap@v)

    rhs = np.concatenate((u_t, v_t), axis=0)

    return rhs

sol2 = scipy.integrate.solve_ivp(lambda t, vec: rhs2(t, vec, beta, Lap), [0, 25], np.squeeze(vec1), t_eval = t_span)
A14 = sol2.y.T
A15 = sol2.y.real[841:][:, 4].reshape(-1, 1)
A16 = np.pad(A15.reshape(29, 29).T, [1, 1])

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
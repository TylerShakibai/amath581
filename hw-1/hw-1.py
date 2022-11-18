import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt

# Problem 1

dydt = lambda t, y: -3*y*np.sin(t)
y0 = np.pi/np.sqrt(2)
ytrue = lambda t: np.pi*np.exp(3*(np.cos(t) - 1))/np.sqrt(2)

## a - Forward Euler

def forward_euler(f, t, y0):
	dt = t[2] - t[1]
	y = np.zeros(len(t))
	y[0] = y0
	for k in range(len(y)-1):
		y[k+1] = y[k] + dt*f(t[k], y[k])
	return y

dt_vals = 2**(-np.linspace(2, 8, 7))
err = np.zeros(len(dt_vals))
for k, dt in enumerate(dt_vals):
	N = int(5/dt)
	t = np.arange(0, 5+dt, dt)
	y = forward_euler(dydt, t, y0)
	err[k] = np.abs(y[-1] - ytrue(5))

A1 = y.reshape(-1, 1)
A2 = err.reshape(1, -1)

pfit = np.polyfit(np.log(dt_vals), np.log(err), 1)
A3 = pfit[0]

fig, ax = plt.subplots()
ax.loglog(dt_vals, err, 'k.', markersize=20, label='Forward Euler Error')
ax.loglog(dt_vals, 2.8*dt_vals, 'k--', linewidth=2, label=r'O($\Delta t$) trend line')

## b - Heun's method

def heun(f, t, y0):
	dt = t[2]-t[1]
	y = np.zeros(len(t))
	y[0] = y0
	for k in range(len(y)-1):
		y[k+1] = y[k] + 0.5*dt*(f(t[k], y[k]) + f(t[k+1], y[k]+dt*f(t[k], y[k])))
	return y

err2 = np.zeros(len(dt_vals))
for k, dt in enumerate(dt_vals):
	t = np.arange(0, 5+dt, dt)
	yh = heun(dydt, t, y0)
	err2[k] = np.abs(yh[-1] - ytrue(5))

					
A4 = yh.reshape(-1, 1)
A5 = err2.reshape(1, -1)

pfit2 = np.polyfit(np.log(dt_vals), np.log(err2), 1)
A6 = pfit2[0]

ax.loglog(dt_vals, err2, 'bd', markersize=10, markerfacecolor='b', label="Heun's Error")
ax.loglog(dt_vals, 0.75*dt_vals**2, 'b--', linewidth=2, label=r'O($\Delta t^2$) trend line')

## c - Adams predictor-corrector method

def adam(f, t, y0):
	dt = t[2]-t[1]
	y = np.zeros(len(t))
	y[0] = y0
	y[1] = y0 + dt*f(t[0] + dt/2, y0 + 0.5*dt*f(t[0], y0))
	for k in range(1,len(y)-1):
		yp = y[k] + 0.5*dt*(3*f(t[k], y[k]) - f(t[k-1], y[k-1]))
		y[k+1] = y[k] + 0.5*dt*( \
			f(t[k+1], yp) + f(t[k], y[k]) )
	return y

err = np.zeros(len(dt_vals))
for k, dt in enumerate(dt_vals):
	t = np.arange(0, 5+dt, dt)
	y = adam(dydt, t, y0)
	err[k] = np.abs(y[-1] - ytrue(5))

A7 = y.reshape(-1, 1)
A8 = err.reshape(1, -1)

pfit = np.polyfit(np.log(dt_vals), np.log(err), 1)
A9 = pfit[0]

### 2D Plot
					
ax.loglog(dt_vals, err, 'gs', markersize=10, markerfacecolor='g', \
			label="Adam's Predictor-Corrector Error")
ax.loglog(dt_vals, 8*dt_vals**3, 'g--', linewidth=2, \
		label=r'O($\Delta t^3$) trend line')
ax.legend(loc='lower right')
ax.set_xlabel(r'$\Delta t$')
ax.set_ylabel(r'Global error at t=5: $|y_{true}(5) - y_N|$')
ax.set_title('Global error trends for three methods')

# Problem 2

dydt = lambda t, y, eps: np.array([y[1], -eps*(y[0]**2 -1)*y[1] - y[0]])

## a

y0 = np.array([np.sqrt(3), 1])
trange = np.arange(0, 32.5, 0.5)
out1 = scipy.integrate.solve_ivp(lambda t,y: dydt(t, y, 0.1), [0, 32], y0, t_eval = trange)
out2 = scipy.integrate.solve_ivp(lambda t,y: dydt(t, y, 1), [0, 32], y0, t_eval = trange)
out3 = scipy.integrate.solve_ivp(lambda t,y: dydt(t, y, 20), [0, 32], y0, t_eval = trange)

A10 = np.array([out1.y[0], out2.y[0], out3.y[0]]).T

## b

y0 = np.array([2, np.pi**2])
tolerances = np.logspace(-4, -10, 7)
t1_diff = np.zeros(len(tolerances))
t2_diff = np.zeros(len(tolerances))
t3_diff = np.zeros(len(tolerances))

for iter, tol in enumerate(np.logspace(-4, -10, 7)):
	out1 = scipy.integrate.solve_ivp(lambda t, y: dydt(t, y, 1), [0, 32], y0, method='RK45', atol=tol, rtol=tol)
	t1 = out1.t
	t1_diff[iter] = np.mean(np.diff(t1))

	out2 = scipy.integrate.solve_ivp(lambda t,y: dydt(t, y, 1), [0, 32], y0, method='RK23', atol = tol, rtol = tol)
	t2 = out2.t
	t2_diff[iter]  = np.mean(np.diff(t2))

	out3 = scipy.integrate.solve_ivp(lambda t,y: dydt(t, y, 1), [0, 32], y0, method='BDF', atol = tol, rtol = tol)
	t3 = out3.t
	t3_diff[iter] = np.mean(np.diff(t3))

plt.figure()
plt.loglog(t1_diff, tolerances, 'ko')
plt.loglog(t2_diff, tolerances, 'bd')
plt.loglog(t3_diff, tolerances, 'gs')

plt.figure()
plt.plot(out1.t, out1.y[0,:] )

A11 = np.polyfit(np.log(t1_diff), np.log(tolerances), 1)[0]
A12 = np.polyfit(np.log(t2_diff), np.log(tolerances), 1)[0]
A13 = np.polyfit(np.log(t3_diff), np.log(tolerances), 1)[0]

# Problem 3

a1, a2, b, c, I = 0.05, 0.25, 0.1, 0.1, 0.1
tvals = np.arange(0, 100+0.5, 0.5)

dv1dt = lambda v1, w1, v2, w2, d12: -v1**3 + (1+a1)*v1**2 - a1*v1 - w1 + I + d12*v2
dw1dt = lambda v1, w1, v2, w2: b*v1 - c*w1
dv2dt = lambda v1, w1, v2, w2, d21: -v2**3 + (1+a2)*v2**2 - a2*v2 - w2 + I + d21*v1
dw2dt = lambda v1, w1, v2, w2: b*v2 - c*w2

dydt = lambda t, y, d12, d21: np.array([dv1dt(y[0], y[1], y[2], y[3], d12),
                          dw1dt(y[0], y[1], y[2], y[3]),
                          dv2dt(y[0], y[1], y[2], y[3], d21),
                          dw2dt(y[0], y[1], y[2], y[3])
                         ])

y0 = np.array([0.1, 0, 0.1, 0])

sol1 = scipy.integrate.solve_ivp(dydt, [tvals[0], tvals[-1]], y0, method='BDF', args=[0, 0], t_eval=tvals)
sol2 = scipy.integrate.solve_ivp(dydt, [tvals[0], tvals[-1]], y0, method='BDF', args=[0, 0.2], t_eval=tvals)
sol3 = scipy.integrate.solve_ivp(dydt, [tvals[0], tvals[-1]], y0, method='BDF', args=[-0.1, 0.2], t_eval=tvals)
sol4 = scipy.integrate.solve_ivp(dydt, [tvals[0], tvals[-1]], y0, method='BDF', args=[-0.3, 0.2], t_eval=tvals)
sol5 = scipy.integrate.solve_ivp(dydt, [tvals[0], tvals[-1]], y0, method='BDF', args=[-0.5, 0.2], t_eval=tvals)

A14 = (sol1.y[(0, 2, 1, 3), :]).T
A15 = (sol2.y[(0, 2, 1, 3), :]).T
A16 = (sol3.y[(0, 2, 1, 3), :]).T
A17 = (sol4.y[(0, 2, 1, 3), :]).T
A18 = (sol5.y[(0, 2, 1, 3), :]).T

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
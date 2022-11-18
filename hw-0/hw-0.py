import numpy as np
import matplotlib.pyplot as plt

# Problem 1

A = np.array([[12, 37], [-9, 0]])
A1 = np.copy(A)

# Problem 2

def f(x): # define function f(x)
   return -x - np.cos(x)

def fp(x): # define function f'(x)
   return np.sin(x) - 1

x = np.empty([1,1])
x[0] = -3
for j in range(1000):
   x = np.append(x, [x[j] - f(x[j]) / fp(x[j]) ])
   fc = f(x[j + 1])

   if np.abs(fc) < 10**(-6):
      break

A2 = x.reshape([-1, 1])

xl,xr = -3, 1
mids = []
mids = np.array(mids)
for j in range(1000):
   xc = (xl+xr)/2
   mids = np.append(mids, [xc])
   fc = f(xc)
   
   if fc > 0:
      xl=xc
   else:
      xr=xc

   if np.abs(fc)<10**(-6):
      break

A3 = mids.reshape([1, -1])
A4 = np.array([[len(A2), len(mids)]])

# Problem 3

x = np.array([1, 3, 4, 8, 9])
y = np.array([3, 4, 5, 7, 12])
z = np.polyfit(x, y, 1)
A5 = z[0]

## 2D Plot

fig, ax = plt.subplots()
ax.plot(x, y, "ko", linewidth=3, label="Data")
xplotting = np.arange(0, 10, 0.1)
ax.plot(xplotting, np.polyval(z, xplotting), "b", linewidth=3, label="Line of best fit")
ax.legend(loc="upper left", fontsize=13 )
plt.title("Plot of data and best-fit line", fontsize=15)

# Problem 4

A = np.array([[-0.1, 3], [3, -0.1]])
b = np.array([[-0.2], [0.2]])
A6 = np.linalg.solve(A, b)

# print("A1:", A1)
# print("A2:", A2)
# print("A3:", A3)
# print("A4:", A4)
# print("A5:", A5)
# print("A6:", A6)
# plt.show()
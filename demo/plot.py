import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

x = np.linspace(0, 1, 200)
y = - x * np.log2(x) - (1 - x) * np.log2(1 - x)
g = 1 - x * x - (1 - x) * (1 - x)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(x, 0.5 * y, label='1/2 * entropy')
ax.plot(x, g, c='r', label='gini')
plt.legend(loc='upper left')
plt.show()

fig = plt.figure()
ax = Axes3D(fig)
X = np.arange(-2, 2, 0.1)
Y = np.arange(-2, 2, 0.1)
X, Y = np.meshgrid(X, Y)
Z = X ** 2 + Y ** 2
ax.plot_surface(X, Y, Z)
plt.show()

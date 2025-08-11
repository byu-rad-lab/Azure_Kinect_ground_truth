import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import UnivariateSpline

data = np.loadtxt('/home/radlab/ros2_ws/src/azure_listener/azure_listener/filtered_data2.txt', delimiter = ',')
data1 = data.T
# x, y, z = data

x = []
y = []
z = []

for point in data:
    x1, y1, z1 = point
    if z1 < 1:
        x.append(x1)
        y.append(y1)
        z.append(z1)


fig = plt.figure()
ax = fig.add_subplot(111, projection= '3d')
ax.scatter(x, y, z, label = 'Data Points', color = 'red')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.tight_layout()
plt.show()
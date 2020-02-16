import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

# Own Libraries/Classes
from regression import regression

####################################################### Start ##########################################################


# Testing out Regression Class
x = [1, 2, 2]
y = [2, 1, 3]
O = regression(x, 0, y, 0, 0, 0)
m, b = O.linear()
print("The slope is: ", m, "The y-intercept is: ", b)


# Testing for 3 defined points in 3 dimensions

ran = 100
x = [25, 15, 67]
y = [36, 48, 16]
z = [49, 15, 37]

O = regression(x, y, z, 0, 0, 0)
m1, m2, b = O.linear_3D()
print("Slopes for 3 points in 3 dimensions: Slope m1 ", m1, "Slope M2: ", m2, " and the Y-intercept is: ", b)

fig = plt.figure()
ax = plt.axes(projection='3d')

x1 = [0, ran]
x2 = [0, ran]
zline = [m1 *x1[0] + m2 *x2[0]+ b, m1 * x1[1] + m2 * x2[1] + b]
ax.plot3D(x1, x2, zline, 'gray')
ax.scatter3D(x, y, z,  c=None, cmap='Greens')
plt.show()


# Testing for multiple points in 3 dimensions
ran = 20
total_points = 19
x = random.sample(range(0, ran), total_points)
y = random.sample(range(0, ran), total_points)
z = random.sample(range(0, ran), total_points)

O = regression(x, y, z, 0, 0, 0)
m1, m2, b = O.linear_3D()
print("Slopes for multiple points in 3 dimensions are Slope M1: ", m1, "Slope M2: ", m2, "and the Y-intercept is: ", b)

fig1 = plt.figure()
ax = plt.axes(projection='3d')
x1 = [0, ran]
x2 = [0, ran]
zline = [m1*x1[0] + m2*x2[0]+ b, m1 * x1[1] + m2 * x2[1] + b]
ax.plot3D(x1, x2, zline, 'gray')
ax.scatter3D(x, y, z, c=None, cmap='Greens')
plt.show()


# Finding and plotting regression plane for 3 dimensions
ran = 10
total_points = 10
plane_size = 5
x = random.sample(range(0, ran), total_points)
y = random.sample(range(0, ran), total_points)
z = random.sample(range(0, ran), total_points)

O = regression(x, y, z, 0, 0, 0)
m1, m2, b = O.linear_3D()
print("Slopes for multiple points in 3 dimensions are Slope M1", m1, "Slope M2: ", m2, "and the Y-intercept is: ", b)

fig2 = plt.figure()
ax = plt.axes(projection='3d')
x1 = [-plane_size * ran, plane_size * ran]
x2 = [-plane_size* ran, plane_size * ran]
x1 = np.array(x1)
x2 = np.array(x2)
zline = [[m1*x1[0] + m2*x2[0]+ b, m1*x1[1] + m2*x2[0]+ b], [m1*x1[0] + m2*x2[1]+ b, m1*x1[1] + m2*x2[1]+ b]]
zline = np.array(zline)
ax.plot_surface(x1, x2, zline, alpha=0.2)
ax.scatter3D(x, y, z, c=None, cmap='Greens');
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')
plt.show()



# Test out regression class method mlinear for all dimensions
data = pd.read_csv("MLRtestEx.csv")
df = pd.DataFrame(data)
x = df[["X1", "X2"]]
y = df[["Y"]]

O = regression(0, 0, 0, x, y, df)
A = O.mlinear()
m1, m2, b = A[0], A[1], A[2]
print("Slopes for 3 points in 3 or (multiple) dimensions are Slope M1: ", m1, "Slope M2 is: ", m2, "Y-intercept is: ", b)

ran = 100
x = df["X1"].tolist()
y = df["X2"].tolist()
z = df["Y"].tolist()
fig3 = plt.figure()
ax = plt.axes(projection='3d')
x1 = [0, ran]
x2 = [0, ran]
zline = [m1 *x1[0] + m2 *x2[0]+ b, m1 * x1[1] + m2 * x2[1] + b]
ax.plot3D(x1, x2, zline, 'gray')
ax.scatter3D(x, y, z, c=None, cmap='Greens')
plt.show()
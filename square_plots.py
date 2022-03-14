
# CUBE
# import numpy as np
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
# import matplotlib.pyplot as plt
# points = np.array([[-1, -1, -1],
#                   [1, -1, -1 ],
#                   [1, 1, -1],
#                   [-1, 1, -1],
#                   [-1, -1, 1],
#                   [1, -1, 1 ],
#                   [1, 1, 1],
#                   [-1, 1, 1]])
# Z = points
# Z = 10.0*Z
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# r = [-1,1]
# X, Y = np.meshgrid(r, r)
# ax.scatter3D(Z[:, 0], Z[:, 1], Z[:, 2])
# verts = [[Z[0],Z[1],Z[2],Z[3]],
#  [Z[4],Z[5],Z[6],Z[7]],
#  [Z[0],Z[1],Z[5],Z[4]],
#  [Z[2],Z[3],Z[7],Z[6]],
#  [Z[1],Z[2],Z[6],Z[5]],
#  [Z[4],Z[7],Z[3],Z[0]]]
# ax.add_collection3d(Poly3DCollection(verts, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.20))
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.show()



# SQUARE
import matplotlib.pyplot as plt
#from matplotlib.patches import Rectangle
fig, ax = plt.subplots()
ax.plot([None,None])

square1 = plt.Rectangle((1, 0), 4, 4, fc = 'white', ec = 'blue')

square2 = plt.Rectangle((6, 0), 2, 2, fc = 'white', ec = 'red')


ax.add_patch(square1)
ax.add_patch(square2)

plt.xticks([i for i in range(15)])
plt.yticks([i for i in range(15)])
plt.xlabel("X-AXIS")
plt.ylabel("Y-AXIS")

plt.show()


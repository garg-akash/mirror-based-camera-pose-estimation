import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def getEuler(Rmat):
  rotation = R.from_matrix(Rmat)
  euler_angles_rad = rotation.as_euler('zyx', degrees=False)
  euler_angles_deg = rotation.as_euler('zyx', degrees=True)
  print("Euler angles (radians):", euler_angles_rad)
  print("Euler angles (degrees):", euler_angles_deg)

def plotFrame(ax, R, T, frame_name, color):
  # Define the origin
  origin = T
  
  # Define the axis of the frame
  x_axis = T + R[:, 0] * 1
  y_axis = T + R[:, 1] * 1
  z_axis = T + R[:, 2] * 1

  # Plot the origin
  ax.scatter(origin[0], origin[1], origin[2], color=color)
  ax.text(origin[0], origin[1], origin[2], frame_name, color=color)

  # Plot the axes
  ax.plot([origin[0], x_axis[0]], [origin[1], x_axis[1]], [origin[2], x_axis[2]], color=color)
  ax.plot([origin[0], y_axis[0]], [origin[1], y_axis[1]], [origin[2], y_axis[2]], color=color)
  ax.plot([origin[0], z_axis[0]], [origin[1], z_axis[1]], [origin[2], z_axis[2]], color=color)

def createPlot(ax, R1, t1, R2, t2, num):
  # Create a plot
  # fig = plt.figure()
  # ax = fig.add_subplot(111, projection='3d')

  # Plot the frames
  plotFrame(ax, np.eye(3), np.zeros(3), 'Chessboard ' + num, 'k')  # Chessboard frame
  plotFrame(ax, R1, t1, 'Camera 1 ' + num, 'r')  # Camera 1 frame
  plotFrame(ax, R2, t2, 'Camera 2 ' + num, 'b')  # Camera 2 frame
  # Set labels
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')

  # Set equal scaling
  ax.set_box_aspect([1, 1, 1])

  # plt.show()

pre = "data/96/"

R1x1 = np.loadtxt(pre + "cam1x1R.txt")
t1x1 = np.loadtxt(pre + "cam1x1T.txt")
R2x1 = np.loadtxt(pre + "cam2x1R.txt")
t2x1 = np.loadtxt(pre + "cam2x1T.txt")

# print(f"r1x1 : {R1x1}")
# print(f"t1x1 : {t1x1}")
# print(f"r2x1 : {R2x1}")
# print(f"t2x1 : {t2x1}")

# corners_cam1 = np.loadtxt("data/cam1corners.txt")
# corners_cam2 = np.loadtxt("data/cam2corners.txt")

# num1 = corners_cam1.shape[0]
# num2 = corners_cam2.shape[0]
# assert num1 == num2, "Number of corners are not same in two cameras"
# corners_cam1 = np.hstack([corners_cam1,np.zeros((num1,1)),np.ones((num1,1))])
# corners_cam2 = np.hstack([corners_cam2,np.zeros((num2,1)),np.ones((num2,1))])
# print(corners_cam1)
# # print(f"p1 : {corners_cam1[0]}")

# trans_world_in_cam1 = np.eye(4)
# trans_world_in_cam1[:3,:3] = R1
# trans_world_in_cam1[:3,3] = t1
# print(f"trans_world_in_cam1 : {trans_world_in_cam1}")

# trans_world_in_cam2 = np.eye(4)
# trans_world_in_cam2[:3,:3] = R2
# trans_world_in_cam2[:3,3] = t2
# print(f"trans_world_in_cam2 : {trans_world_in_cam2}")

# trans_cam2_in_world = np.linalg.inv(trans_world_in_cam2)
 
# trans_cam2_in_cam1 = trans_world_in_cam1 @ trans_cam2_in_world
# print(f"trans_cam2_in_cam1 : {trans_cam2_in_cam1}")

# corners_cam1_computed = (trans_cam2_in_cam1 @ corners_cam1.T).T
# print(corners_cam1_computed)

R1x2 = np.loadtxt(pre + "cam1x2R.txt")
t1x2 = np.loadtxt(pre + "cam1x2T.txt")
R2x2 = np.loadtxt(pre + "cam2x2R.txt")
t2x2 = np.loadtxt(pre + "cam2x2T.txt")

print(f"r1x2 : {R1x2}")
print(f"t1x2 : {t1x2}")
print(f"r2x2 : {R2x2}")
print(f"t2x2 : {t2x2}")

R1_val = R1x2 @ R2x2.T @ R2x1                   # validation data 
t1_val = R1x2 @ R2x2.T @ (t2x1 - t2x2) + t1x2

print("Printing eulers for R1_val")
getEuler(R1_val)

print("Printing eulers for R1_comp")
getEuler(R1x1)

print("Val R1 : ")
print(R1_val)

print()

print("Comp R1 : ")
print(R1x1)

temp1 = np.dot(R1x1.T, R2x1)
print("Printing eulers for temp1")
getEuler(temp1)

temp2 = np.dot(R1x2.T, R2x2)
print("Printing eulers for temp2")
getEuler(temp2)

oR1x2 = np.loadtxt(pre + "old_cam1x2R.txt")
oR2x2 = np.loadtxt(pre + "old_cam2x2R.txt")
temp3 = np.dot(oR1x2.T, oR2x2)
print("Printing eulers for temp3")
getEuler(temp3)

# distance betweeen t1 and t2 in X1 frame
temp1 = -np.dot(R1x1,t1x1)
temp_val1 = -np.dot(R1_val,t1_val)
temp2 = -np.dot(R2x1,t2x1)
del_t1 = np.linalg.norm(temp1 - temp2)
del_t1_val = np.linalg.norm(temp_val1 - temp2)
print(f"del : {del_t1} ; del_val : {del_t1_val}")

## plotting begins
fig = plt.figure(figsize=(18, 6))
ax1 = fig.add_subplot(131, projection='3d')
createPlot(ax1, R1x1, t1x1, R2x1, t2x1, 'x1')

ax2 = fig.add_subplot(132, projection='3d')
createPlot(ax2, R1x2, t1x2, R2x2, t2x2, 'x2')

ax3 = fig.add_subplot(133, projection='3d')
createPlot(ax3, R1_val, t1_val, R2x1, t2x1, 'x1val')


plt.show()

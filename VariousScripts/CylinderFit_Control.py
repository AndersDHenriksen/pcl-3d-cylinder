import numpy as np


# Generate cylinder data and test methods in bab_perp
height = 76
radius = 330 // 2
axis = np.array([-0.1, 0.05, 1])
axis /= np.linalg.norm(axis)
offset = np.array([5, 8, 0])

angles = np.linspace(-np.pi, np.pi, int(2 * np.pi * radius))
angles_std = np.diff(angles)[0] / 8
z_std = 1 / 2

points = []
for h in range(height):
    disc_center = h * axis + offset
    disc_angles = angles + angles_std * np.random.randn(*angles.shape)  # Add noise to angles
    disc_xyz = np.array([radius * np.cos(disc_angles), radius * np.sin(disc_angles), z_std * np.random.randn(*angles.shape)])
    points.append(disc_center + disc_xyz.T)

cloud = np.vstack(points)


import open3d as o3d
from open3d_helper import perp, cloud_to_nparray, nparray_to_cloud, visualize_cloud, icp_multiple, vectors_to_transfor, \
    get_rotation_matrix, normalize, get_normals, plane_fitting, cylinder_fitting

cloud = nparray_to_cloud(cloud)
fit_axis = cylinder_fitting(cloud, height)

print(f"{axis} =?= {fit_axis}")

_ = 'bp'




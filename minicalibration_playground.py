import numpy as np
import open3d as o3d

from open3d_helper import perp, cloud_to_nparray, nparray_to_cloud, visualize_cloud, icp_multiple, vectors_to_transfor, \
    get_rotation_matrix, normalize, get_normals, plane_fitting, cylinder_fitting, draw_registration_result

diameter = 200
height = 70

top, side = 4 * [None], 4 * [None]
for i in range(4):
    side[i] = o3d.io.read_point_cloud(rf"C:\Projects\Umicore_SW\inspection_sw\out\build\RelWithDebInfo\cloud_side_{i}.ply").voxel_down_sample(1)
    top[i] = o3d.io.read_point_cloud(rf"C:\Projects\Umicore_SW\inspection_sw\out\build\RelWithDebInfo\cloud_top_{i}.ply").voxel_down_sample(1)
    # side[i] = o3d.io.read_point_cloud(rf"C:\Users\ahe\Desktop\par10\cloud_side_{i}.ply").voxel_down_sample(1)
    # top[i] = o3d.io.read_point_cloud(rf"C:\Users\ahe\Desktop\par10\cloud_top_{i}.ply").voxel_down_sample(1)

top_normals = 4 * [None]
for i, cloud_t in enumerate(top):
    top_normals[i] = plane_fitting(cloud_t, diameter)
normal = normalize(np.array(top_normals[:4]).mean(axis=0))

cyl_axis = cylinder_fitting(side[0] + side[1] + side[2] + side[3], height)

print(f"Top perp is: {perp(normal, cyl_axis, diameter):.2f} mm")
print(f"Bottom perp is: {perp([0,0,1], cyl_axis, diameter):.2f} mm")
print(f"Parallelism is: {perp(normal, [0,0,1], diameter):.2f} mm")

_ = 'bp'

###

nz = [float(t) for t in "-0.000659463 -0.000514578 0.999999702".split(" ")]
T = "   -0.484798 -1.37235e-05    -0.874626      785.727 0.00728254     0.999965   -0.0040525      4.02378 0.874596  -0.00833414    -0.484781      637.813           0            0            0            1"
T_Side2Rot = np.array([float(t) for t in T.split(" ") if len(t)]).reshape(4, 4)  # np.array(T)
ny = normalize(np.cross(nz, T_Side2Rot[:3, 3]))
nx = normalize(np.cross(ny, nz))
T_Calib2Rot = vectors_to_transfor(nx, ny, nz, [0,0,0])
T_Side2Calib = np.dot(np.linalg.inv(T_Calib2Rot), T_Side2Rot)
print(T_Side2Calib)

# Alternative
M = get_rotation_matrix(nz, [0,0,1])
T_Rot2Calib = vectors_to_transfor(M[:, 0], M[:, 1], M[:, 2], [0,0,0])
print(np.dot(T_Rot2Calib, T_Side2Rot))

# Extra check
nz_side = np.dot(np.linalg.inv(T_Side2Rot), nz + [1])
print(np.dot(T_Side2Calib, nz_side))
import copy
import numpy as np
import open3d as o3d
import os

from open3d_helper import perp, cloud_to_nparray, nparray_to_cloud, visualize_cloud, icp_multiple, vectors_to_transfor, \
    get_rotation_matrix, normalize, get_normals, plane_fitting, cylinder_fitting, draw_registration_result

diameter = 330
height = 76

top, side = 4 * [None], 4 * [None]
for i in range(4):
    top[i] = o3d.io.read_point_cloud(rf"D:\Data\2802. Umicore\BadPerp\cloud_top_{i}.ply").voxel_down_sample(1)
    side[i] = o3d.io.read_point_cloud(rf"D:\Data\2802. Umicore\BadPerp\cloud_side_{i}.ply").voxel_down_sample(1)

top_normals = 4 * [None]
for i, cloud_t in enumerate(top):
    top_normals[i] = plane_fitting(cloud_t, diameter)
normal = normalize(np.array(top_normals).mean(axis=0))

cyl_axis = cylinder_fitting(side[0] + side[1] + side[2] + side[3], height)

print(f"Top perp is: {perp(normal, cyl_axis, diameter):.2f} mm")
print(f"Top CutSt is: {perp(normal, cyl_axis, height):.2f} mm")
_ = 'bp'
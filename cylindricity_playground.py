import copy
import numpy as np
import open3d as o3d
import os

from open3d_helper import perp, cloud_to_nparray, nparray_to_cloud, visualize_cloud, icp_multiple, vectors_to_transfor, \
    get_rotation_matrix, normalize, get_normals, plane_fitting, cylinder_fitting, draw_registration_result

diameter = 330
height = 76

top, side = 5 * [None], 5 * [None]
for i in range(5):
    side[i] = o3d.io.read_point_cloud(rf"C:\Projects\Umicore_SW\inspection_sw\out\build\RelWithDebInfo\cloud_side_{i}.ply").voxel_down_sample(1)
    top[i] = o3d.io.read_point_cloud(rf"C:\Projects\Umicore_SW\inspection_sw\out\build\RelWithDebInfo\cloud_top_{i}.ply").voxel_down_sample(1)

combined_cloud = icp_multiple(side[0], side[1:])

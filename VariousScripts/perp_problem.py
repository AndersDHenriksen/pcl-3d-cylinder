import copy
import numpy as np
import open3d as o3d
import os

from open3d_helper import perp, cloud_to_nparray, nparray_to_cloud, visualize_cloud, icp_multiple, vectors_to_transfor, get_rotation_matrix, normalize
from Open3d_PlaneFitting import plane_fitting
from Open3d_CylinderFitting import cylinder_fitting

t000 = o3d.io.read_point_cloud(r"E:\BadPerp\cloud_top_0.ply").voxel_down_sample(1)
t090 = o3d.io.read_point_cloud(r"E:\BadPerp\cloud_top_1.ply").voxel_down_sample(1)
t180 = o3d.io.read_point_cloud(r"E:\BadPerp\cloud_top_2.ply").voxel_down_sample(1)
t270 = o3d.io.read_point_cloud(r"E:\BadPerp\cloud_top_3.ply").voxel_down_sample(1)

b000 = o3d.io.read_point_cloud(r"E:\BadPerp\cloud_top_0.ply").voxel_down_sample(1)
b090 = o3d.io.read_point_cloud(r"E:\BadPerp\cloud_top_1.ply").voxel_down_sample(1)
b180 = o3d.io.read_point_cloud(r"E:\BadPerp\cloud_top_2.ply").voxel_down_sample(1)
b270 = o3d.io.read_point_cloud(r"E:\BadPerp\cloud_top_3.ply").voxel_down_sample(1)

top_normals = []
for cloud in [t000, t090, t180, t270]:
    top_normals.append(plane_fitting(cloud, 268))
top_normals = np.array(top_normals)
top_average_true = top_normals[1:].mean(axis=0)

bot_normals = []
for cloud in [b000, b090, b180, b270]:
    bot_normals.append(plane_fitting(cloud, 268))
bot_normals = np.array(bot_normals)
bot_average_true = bot_normals[1:].mean(axis=0)

perp(top_normals[0], top_normals[1])

bot_normals[0] - top_normals[1] + top_average_true

_ = 'bp'

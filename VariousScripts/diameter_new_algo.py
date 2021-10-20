import copy
import numpy as np
import open3d as o3d
import os

from open3d_helper import perp, cloud_to_nparray, nparray_to_cloud, visualize_cloud, icp_multiple, vectors_to_transfor, get_rotation_matrix, normalize
from Open3d_PlaneFitting import plane_fitting
from Open3d_CylinderFitting import cylinder_fitting

CloudS, CloudT = 6*[None],6*[None]
for i in range(6):
    CloudS[i] = o3d.io.read_point_cloud(rf"C:\Projects\Umicore_SW\inspection_sw\out\build\RelWithDebInfo\cloud_side_{i}.ply").voxel_down_sample(1)
    CloudT[i] = o3d.io.read_point_cloud(rf"C:\Projects\Umicore_SW\inspection_sw\out\build\RelWithDebInfo\cloud_top_{i}.ply").voxel_down_sample(1)

# CloudT.append(o3d.io.read_point_cloud(rf"E:\TOP.ply").voxel_down_sample(1))
# CloudT.append(o3d.io.read_point_cloud(rf"E:\top_22.ply").voxel_down_sample(1))

# CloudS_combined = CloudS[0] + CloudS[1] + CloudS[2] + CloudS[3] + CloudS[4] + CloudS[5]
# CloudS_combined = cloud_to_nparray(CloudS_combined)
# CloudS_centered = CloudS_combined - CloudS_combined.mean(axis=0)

for t in range(6):
    CloudT[t].estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.5, max_nn=30))
    good_idx = np.abs(np.asarray(CloudT[t].normals)[:, 2]) > 0.93
    cloud_filt = cloud_to_nparray(CloudT[t])[good_idx, :]
    good_idx = np.abs(cloud_filt[:, 2] - np.median(cloud_filt[:,2])) < 10
    cloud_filt = cloud_filt[good_idx, :]
    cloud_cent = cloud_filt - cloud_filt.mean(axis=0)
    cloud_angles = np.arctan2(cloud_cent[:,0], cloud_cent[:,1])
    angles_delim = np.linspace(-np.pi, np.pi, 101)
    radii = []
    for i in range(angles_delim.size - 1):  # TODO maybe faster to loop through cloud
        angle_lower = angles_delim[i]
        angle_upper = angles_delim[i + 1]
        angle_idx = (cloud_angles > angle_lower) & (cloud_angles < angle_upper)
        local_radius = np.linalg.norm(cloud_cent[angle_idx, :2], axis=1).max()
        radii.append(local_radius)
    print(np.array(radii).mean() * 2)

_ = 'bp'
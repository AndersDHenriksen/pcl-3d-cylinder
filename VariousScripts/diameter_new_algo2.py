from pathlib import Path
import copy
import numpy as np
import open3d as o3d
import os

from open3d_helper import cloud_to_nparray, nparray_to_cloud, visualize_cloud, fit_circle

CloudS = 4 * [None]
for i in range(4):
    CloudS[i] = cloud = o3d.io.read_point_cloud(rf"D:\User\Desktop\20210621 15.20.21\cloud_side_{i}.ply").voxel_down_sample(1)
    cloud = cloud_to_nparray(cloud)
    mean_z = cloud[:, 2].mean()
    good_idx = np.abs(cloud[:, 2] - mean_z) < 1
    xs, ys = cloud[good_idx, 0], cloud[good_idx, 1]
    r = fit_circle(xs, ys)[2]
    print(2 * r)
    # visualize_cloud(cloud[good_idx])

cloud = CloudS[0] + CloudS[1] + CloudS[2] + CloudS[3]
cloud = cloud_to_nparray(cloud)
mean_z = cloud[:, 2].mean()
good_idx = np.abs(cloud[:, 2] - mean_z) < 1
xs, ys = cloud[good_idx, 0], cloud[good_idx, 1]
r = fit_circle(xs, ys)[2]
print(2 * r)

###

CloudT = 4 * [None]
for t in range(4):
    CloudT[t] = o3d.io.read_point_cloud(rf"D:\User\Desktop\20210621 15.20.21\cloud_top_{t}.ply", format='ply').voxel_down_sample(1)
    CloudT[t].estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.5, max_nn=30))
    good_idx = np.abs(np.asarray(CloudT[t].normals)[:, 2]) > 0.93
    cloud_filt = cloud_to_nparray(CloudT[t])[good_idx, :]
    good_idx = np.abs(cloud_filt[:, 2] - np.median(cloud_filt[:, 2])) < 10
    cloud_filt = cloud_filt[good_idx, :]
    cloud_cent = cloud_filt - cloud_filt.mean(axis=0)
    # visualize_cloud(cloud_cent)

    cloud_angles = np.arctan2(cloud_cent[:, 0], cloud_cent[:, 1])
    angles_delim = np.linspace(-np.pi, np.pi, 101)
    radii = []
    for i in range(angles_delim.size - 1):
        angle_lower = angles_delim[i]
        angle_upper = angles_delim[i + 1]
        angle_idx = (cloud_angles > angle_lower) & (cloud_angles < angle_upper)
        local_radius = np.linalg.norm(cloud_cent[angle_idx, :2], axis=1).max()
        radii.append(local_radius)
    print(np.array(radii).mean() * 2)

###

expected_diameter = 267

for fp in Path.cwd().glob("*.ply"):
    CloudT = o3d.io.read_point_cloud(str(fp), format='ply').voxel_down_sample(1)
    CloudT.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.5, max_nn=30))
    good_idx = (np.abs(np.asarray(CloudT.normals)[:, 2]) > 0.93) & (np.abs(CloudT[:,:2]).max(axis=1) < expected_diameter/2 + 10)
    cloud_filt = cloud_to_nparray(CloudT)[good_idx, :]
    good_idx = np.abs(cloud_filt[:, 2] - np.median(cloud_filt[:, 2])) < 10
    cloud_filt = cloud_filt[good_idx, :]
    cloud_cent = cloud_filt - cloud_filt.mean(axis=0)
    # visualize_cloud(cloud_cent)

    cloud_angles = np.arctan2(cloud_cent[:, 0], cloud_cent[:, 1])
    angles_delim = np.linspace(-np.pi, np.pi, 101)
    radii = []
    for i in range(angles_delim.size - 1):
        angle_lower = angles_delim[i]
        angle_upper = angles_delim[i + 1]
        angle_idx = (cloud_angles > angle_lower) & (cloud_angles < angle_upper)
        local_radius = np.linalg.norm(cloud_cent[angle_idx, :2], axis=1).max()
        radii.append(local_radius)
    print(f"{fp.name}: {np.array(radii).mean() * 2}")
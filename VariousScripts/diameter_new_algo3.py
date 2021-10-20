from pathlib import Path
import copy
import numpy as np
import open3d as o3d
import os

from open3d_helper import cloud_to_nparray, nparray_to_cloud, visualize_cloud, fit_circle

expected_diameter = 267

for fp in Path(r"D:\User\Desktop\20210623 15.02.39").glob("*.ply"):
    CloudT = o3d.io.read_point_cloud(str(fp), format='ply').voxel_down_sample(1)
    CloudT.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.5, max_nn=30))
    good_idx = np.abs(np.asarray(CloudT.normals)[:, 2]) > 0.96
    cloud_filt = cloud_to_nparray(CloudT)[good_idx, :]
    good_idx = (np.abs(cloud_filt[:, 2] - np.median(cloud_filt[:, 2])) < 10) & (np.abs(cloud_filt[:, :2]).max(axis=1) < expected_diameter/2 + 10)
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
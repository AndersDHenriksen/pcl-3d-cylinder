from pathlib import Path
import copy
import numpy as np
import open3d as o3d
import os

from open3d_helper import cloud_to_nparray, nparray_to_cloud, visualize_cloud

for fp in Path(r"D:\User\Desktop\Warmup").glob("*.ply"):
    cloud = o3d.io.read_point_cloud(str(fp)).voxel_down_sample(1)
    cloud = cloud_to_nparray(cloud)
    heights = cloud[:, 2]
    median_height = np.median(heights)
    good_idx = np.abs(heights - median_height) < 20
    cloud_filt = cloud[good_idx, :]
    visualize_cloud(cloud_filt)
    print(f"{fp.name}: {heights[good_idx].mean()}")
import copy

import numpy as np
import open3d as o3d

def fit_circle(xs, ys):
    xs = xs.reshape(-1, 1)
    ys = ys.reshape(-1, 1)
    A = np.hstack((2 * xs, 2 * ys, np.ones_like(xs)))
    b = xs ** 2 + ys ** 2
    c = np.linalg.lstsq(A, b)[0]
    c = np.squeeze(c)
    x_c = c[0]
    y_c = c[1]
    r = np.sqrt(c[2] + c[0] ** 2 + c[1] ** 2)
    return x_c, y_c, r


clouds = [None, None, None, None]
clouds[0] = o3d.io.read_point_cloud(r"D:\Data\2802. Umicore\Series7 - new calibration\small_ref_element\cloud_0.ply")
clouds[1] = o3d.io.read_point_cloud(r"D:\Data\2802. Umicore\Series7 - new calibration\small_ref_element\cloud_1.ply")
clouds[2] = o3d.io.read_point_cloud(r"D:\Data\2802. Umicore\Series7 - new calibration\small_ref_element\cloud_2.ply")
clouds[3] = o3d.io.read_point_cloud(r"D:\Data\2802. Umicore\Series7 - new calibration\small_ref_element\cloud_3.ply")
clouds_top = o3d.io.read_point_cloud(r"D:\Data\2802. Umicore\Series7 - new calibration\small_ref_element\cloud_top.ply")

#U, S, VT = np.linalg.svd(np.asarray(clouds_top.points))
#top_normal = VT[2, :]

clouds_top.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=3, max_nn=30))
clouds_top_normal = np.asarray(clouds_top.normals)
top_normal = clouds_top_normal.mean(axis=0)
top_normal /= np.linalg.norm(top_normal)

normals = [None, None, None, None]
for i, cloud in enumerate(clouds):
    cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=3, max_nn=30))
    cloud_normal = np.asarray(cloud.normals)
    normals[i] = cloud_normal.mean(axis=0)
    normals[i] /= np.linalg.norm(normals[i])




clouds_top_down = clouds_top.voxel_down_sample(voxel_size=3)
down_np = np.asarray(clouds_top_down.points)
U, S, VT = np.linalg.svd(down_np - down_np.mean(axis=0))
top_normal = VT[2, :]

258 * np.sqrt(1 - top_normal[2]**2) / np.abs(top_normal[2])

_ = 'bp'


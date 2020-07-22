import copy
import json
import numpy as np
import open3d as o3d

def perp(ndn):
    return 258 * np.sqrt(1 - min(1, ndn) ** 2) / np.abs(ndn)

def cloud_to_nparray(cloud):
    return np.asarray(cloud.points)

def nparray_to_cloud(data):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(data)
    return cloud

def visualize_cloud(cloud):
    if isinstance(cloud, np.ndarray):
        cloud = nparray_to_cloud(cloud)
    if cloud.is_empty():
        raise Exception("Cannot visualize empty cloud")
    failed = True
    while failed:
        try:
            o3d.visualization.draw_geometries([cloud])
        except:
            pass
        else:
            failed = False

def plane_fitting(cloud_top, diameter):
    cloud_top.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.5, max_nn=30))
    cloud_top_npa = cloud_to_nparray(cloud_top)
    good_idx = np.abs(np.asarray(cloud_top.normals)[:, 2]) > 0.95
    cloud_r2 = cloud_top_npa[:,0]**2 + cloud_top_npa[:,1]**2
    inside_idx = (cloud_r2 < (diameter / 2 - 10) ** 2) & (cloud_r2 > 10 ** 2)

    cloud_top_npa = cloud_top_npa[good_idx & inside_idx, :]

    visualize_cloud(cloud_top_npa)
    downsampled_cloud = cloud_to_nparray(nparray_to_cloud(cloud_top_npa).voxel_down_sample(3))
    normal = np.linalg.svd(downsampled_cloud - downsampled_cloud.mean(axis=0))[2][2,:]
    if normal[2] < 0:
        normal = - normal
    return normal

if __name__ == "__main__":
    diameter = 268
    cloud_top = o3d.io.read_point_cloud(r"C:\Projects\Umicore_SW\inspection_sw\out\build\RelWithDebInfo\cloud_top3.ply")
    normal = plane_fitting(cloud_top, diameter)
    print(normal)

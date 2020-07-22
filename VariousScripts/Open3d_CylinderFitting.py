import copy

import numpy as np
import open3d as o3d

perp = lambda ndn: 258 * np.sqrt(1 - ndn ** 2) / np.abs(ndn)

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

def fit_circle2(xs, ys):
    p = 0
    x_c = np.percentile(xs, 100 - p)/2 + np.percentile(xs, p)/2
    y_c = np.percentile(ys, 100 - p)/2 + np.percentile(ys, p)/2
    return x_c, y_c, 0

def cloud_to_nparray(cloud):
    return np.asarray(cloud.points)

def nparray_to_cloud(data):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(data)
    return cloud

def visualize_cloud(cloud):
    if isinstance(cloud, np.ndarray):
        cloud = nparray_to_cloud(cloud)
    failed = True
    while failed:
        try:
            o3d.visualization.draw_geometries([cloud])
        except:
            pass
        else:
            failed = False

def cylinder_fitting(cloud, height):
    cloud = cloud_to_nparray(cloud)
    # cloud = cloud_to_nparray(nparray_to_cloud(cloud).voxel_down_sample(1))
    visualize_cloud(nparray_to_cloud(cloud))

    xc, yc, zc = [], [], []
    for z in range(20, height - 20, 2):
        zc.append(z + 1)
        good_idx = (cloud[:, 2] > z) & (cloud[:, 2] < z + 2)
        xs = cloud[good_idx, 0]
        ys = cloud[good_idx, 1]
        x_c, y_c, r = fit_circle(xs, ys)
        xc.append(x_c)
        yc.append(y_c)

    circle_points = np.array([xc, yc, zc]).T
    # print(circle_points)

    U, S, VT = np.linalg.svd(circle_points - circle_points.mean(axis=0))
    cyl_axis = VT[0, :]
    return cyl_axis

    # mesh_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=134.5, height=268)
    # o3d.visualization.draw_geometries([mesh_cylinder, nparray_to_cloud(cloud - cloud.mean(axis=0))])

    # top_cloud = o3d.io.read_point_cloud(r"E:\Measurements\200628 Yellow 5\Ply\cloud_top.ply")
    # top_cloud = top_cloud.voxel_down_sample(voxel_size=2)
    # top_cloud = cloud_to_nparray(top_cloud)
    #
    # U, S, VT = np.linalg.svd(top_cloud - top_cloud.mean(axis=0))
    # surface_normal = VT[2, :]

if __name__ == "__main__":
    cloud = cloud_to_nparray(o3d.io.read_point_cloud(r"C:\Projects\Umicore_SW\inspection_sw\out\build\RelWithDebInfo\cloud_sides3.ply"))
    height = 268
    cylinder_axis = cylinder_fitting(cloud, height)
    print(cylinder_axis)
    _ = 'bp'

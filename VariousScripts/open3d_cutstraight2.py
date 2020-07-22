import copy

import numpy as np
import open3d as o3d

def cloud_to_nparray(cloud):
    return np.asarray(cloud.points)

def nparray_to_cloud(data):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(data)
    return cloud

def visualize_cloud(cloud):
    failed = True
    while failed:
        try:
            o3d.visualization.draw_geometries([cloud])
        except:
            pass
        else:
            failed = False


clouds, elements = 2 * [None], 2 * [None]
clouds[0] = o3d.io.read_point_cloud(r"E:\ply_files\s0.ply")
elements[0] = 'small'
clouds[1] = o3d.io.read_point_cloud(r"E:\ply_files\c0.ply")
elements[1] = 'cylinder'


# T = np.load(r"C:\Users\ahe\Google Drive\ProInvent\10. Umicore 3d\T_stage_to_scanner.txt.npy")
# cloud = clouds[0].transform(np.linalg.inv(T))
# o3d.io.write_point_cloud(r"E:\ply_files\m0_t2.ply", cloud)


centers = {'small': np.array([151, 20, 1089]),
           'medium': np.array([150, 12, 1089]),
           'large': np.array([150, 12, 1089]),
           'cylinder': np.array([11, 4, 712])}
filter_radii = {'small': 90, 'medium': 110, 'large': 140, 'cylinder': 120}

# clouds[1] = o3d.io.read_point_cloud(r"E:\200625 NoCylinder\s1.ply")
# centers = [np.array([151, 20, 1089]), np.array([151, 20, 1089])]
# filter_radii = [90, 90]

filtered_clouds = []
normals = []
for cloud, element in zip(clouds, elements):

    # Downsample
    cloud_down = cloud.voxel_down_sample(voxel_size=3)

    # Filter
    cloud_points = np.asarray(cloud_down.points) - centers[element]
    distances = np.linalg.norm(cloud_points, axis=1)
    good_idx = (distances < filter_radii[element]) & (distances > 28)
    filtered_cloud = cloud_points[good_idx, :]
    filtered_clouds.append(filtered_cloud)

    # Visualize
    # cloud_filtered = nparray_to_cloud(filtered_cloud)
    # visualize_cloud(cloud_filtered)

    # Estimate normal
    down_np = cloud_to_nparray(cloud_down)
    U, S, VT = np.linalg.svd(filtered_cloud - filtered_cloud.mean(axis=0))
    normal = VT[2, :]
    if normal[2] < 0:
        normal = - normal
    normals.append(normal)
    print(f"{element}: {normal}")

ndn = np.dot(normals[0], normals[1])
perpendicularity = 258 * np.sqrt(1 - ndn ** 2) / np.abs(ndn)
print(f"n . n : {ndn}")
print(f"Perpendicularity: {perpendicularity} mm")  # 0 for plane-plane measurements

_ = 'bp'
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


# cloud = o3d.io.read_point_cloud(r"E:\ply_files\s0.ply")
# cloud = o3d.io.read_point_cloud(r"E:\ply_files\m0.ply")
# cloud = o3d.io.read_point_cloud(r"E:\200627 BigRef - LongCalibration\TopEmpty\NoCylinder.ply")
# cloud = o3d.io.read_point_cloud(r"E:\200627 Empty Calibrated Locked Closed\scan_side.ply")
# cloud = o3d.io.read_point_cloud(r"E:\200627 Empty Calibrated Locked Closed\scan_top.ply")
cloud = o3d.io.read_point_cloud(r"E:\200627 USB\270_side.ply")

# Transform cloud to correct coordinate system
# T = np.load(r"C:\Users\ahe\Google Drive\ProInvent\10. Umicore 3d\T_stage_to_scanner.txt.npy")
# cloud = cloud.transform(np.linalg.inv(T))

# Filter away bad points
cloud_npa = cloud_to_nparray(cloud)
good_idx = (np.abs(cloud_npa[:, 2] + 158) < 10) & (np.abs(cloud_npa[:, 0]) < 160) & (np.abs(cloud_npa[:, 1]) < 160)
cloud_npa = cloud_npa[good_idx, :]

cloud_r2 = cloud_npa[:, 0] ** 2 + cloud_npa[:, 1] ** 2

r_upper = 1.3 * np.sqrt(np.mean(cloud_r2))
r_lower = 50 # 30

# Filter away more points
inside_idx = (cloud_r2 > r_lower**2) & (cloud_r2 < r_upper**2)
cloud_npa = cloud_npa[inside_idx, :]

visualize_cloud(nparray_to_cloud(cloud_npa))

# Calc centroid and fit plane
cloud = nparray_to_cloud(cloud_npa)
clouds_down = cloud.voxel_down_sample(voxel_size=3)
down_np = np.asarray(clouds_down.points)
U, S, VT = np.linalg.svd(down_np - down_np.mean(axis=0))
normal = VT[2, :]
if normal[2] < 0:
    normal = -normal
offset = cloud_npa[:,2].mean()

perp = lambda ndn: 258 * np.sqrt(1 - ndn**2) / np.abs(ndn)  # small reference cylinder

print(f"Offset: {offset}\n Normal: {normal}\n Perp with z: {perp(normal[2])}\n")

# visualize_cloud(nparray_to_cloud(cloud_npa))

# CylinderAxis = np.array([0.000525696378, -0.000121592871, -0.99999981])
# nda = np.dot(CylinderAxis, normal)
# perpendicularity = 258 * np.sqrt(1 - nda**2) / np.abs(nda)
# print(f"For axis {CylinderAxis}\nPerpendicularity: {perpendicularity}")
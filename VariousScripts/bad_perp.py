import copy
import numpy as np
import open3d as o3d
import os

from open3d_helper import perp, cloud_to_nparray, nparray_to_cloud, visualize_cloud, icp_multiple, vectors_to_transfor, \
    get_rotation_matrix, normalize, get_normals, plane_fitting, cylinder_fitting, draw_registration_result

diameter = 330
height = 76

top, side = 4 * [None], 4 * [None]
for i in range(4):
    top[i] = o3d.io.read_point_cloud(rf"D:\Data\2802. Umicore\BadPerp\cloud_top_{i}.ply").voxel_down_sample(1)
    side[i] = o3d.io.read_point_cloud(rf"D:\Data\2802. Umicore\BadPerp\cloud_side_{i}.ply").voxel_down_sample(1)

top_normals = 4 * [None]
for i, cloud_t in enumerate(top):
    top_normals[i] = plane_fitting(cloud_t, diameter)
normal = normalize(np.array(top_normals).mean(axis=0))

cyl_axis = cylinder_fitting(side[0] + side[1] + side[2] + side[3], height)

print(f"Top perp is: {perp(normal, cyl_axis, diameter):.2f} mm")
print(f"Top CutSt is: {perp(normal, cyl_axis, height):.2f} mm")
_ = 'bp'

# New attempt align with icp because of tops - Works better but not sure why - Also seems incorrect to do
combined = [top[i] + side[i] for i in range(4)]
total_cloud = icp_multiple(combined[0], combined[1:])
# total_cloud = icp_multiple(side[0], side[1:])  # Not working
# total_cloud = combined[0] + combined[1] + combined[2] + combined[3]  # Not working
top_normal = plane_fitting(total_cloud, diameter, True)
cylinder_axis = cylinder_fitting(total_cloud, height, True)
perp(top_normal, cylinder_axis, diameter)

# New attempt - Use side for everything - Same result even though circle traced by top normal is different
# top_normals = 4 * [None]
# for i, cloud_t in enumerate(side):
#     top_normals[i] = plane_fitting(cloud_t, diameter)
# normal = normalize(np.array(top_normals).mean(axis=0))
#
# cyl_axis = cylinder_fitting(side[0] + side[1] + side[2] + side[3], height)
#
# print(f"Top perp is: {perp(normal, cyl_axis, diameter):.2f} mm")
# print(f"Top CutSt is: {perp(normal, cyl_axis, height):.2f} mm")

# New attempt fit 4 cylinders - Same result
single_normal, single_axis = 4 * [None], 4 * [None]
for i in range(4):
    single_normal[i] = plane_fitting(side[i], diameter)
    single_axis[i] = cylinder_fitting(side[i], height)
    print(f"Side {i} perp: {perp(single_normal[i], single_axis[i], diameter):.2f} mm")
combined_normal = normalize(np.array(single_normal).mean(axis=0))
combined_axis = normalize(np.array(single_axis).mean(axis=0))
print(f"Combined perp: {perp(combined_normal, combined_axis, diameter):.2f} mm")


# # New idea/attempt - use normals
# normal_s0_top = plane_fitting(side[0], diameter)
# side0 = cloud_to_nparray(side[0])
# good_idx = side0[:, 2] < height - 5
# side0 = side0[go
# od_idx, :]
# side0_normals = get_normals(side0)
# ndn = np.dot(side0_normals, normal_s0_top)
# ndn2 = 1 - ndn.mean()
# perp_new = 330 * np.sqrt(1 - ndn2 ** 2) / ndn2
# print(perp_new)

# next attempt - outside lib
from cylinder_analysis_o3d import fit
cyl_all = cloud_to_nparray(side[0] + side[1] + side[2] + side[3])
w_fit, C_fit, r_fit, fit_err = fit(cyl_all)

# next attempt, fit lines to 100 side segments
cyl_all = cloud_to_nparray(side[0] + side[1] + side[2] + side[3])
good_idx = (cyl_all[:,2] < height - 10) & (cyl_all[:,2] > 10)
cyl_filt = cyl_all[good_idx, :]
cyl_angle = np.arctan2(cyl_filt[:, 1], cyl_filt[:, 0])
angle_delimeter = np.linspace(-np.pi, np.pi, 101)
angle_axis = (angle_delimeter.size - 1) * [0]
for i in range(angle_delimeter.size - 1):
    print(i)
    angle_lower = angle_delimeter[i]
    angle_higher = angle_delimeter[i + 1]
    angle_idx = (cyl_angle > angle_lower) & (cyl_angle < angle_higher)
    cyl_reduced = cyl_filt[angle_idx, :]
    U, S, VT = np.linalg.svd(cyl_reduced - cyl_reduced.mean(axis=0))
    angle_axis[i] = VT[0, :] if VT[0, 2] > 0 else -VT[0, :]
cyl_axis2 = normalize(np.mean(angle_axis, axis=0))

print(f"Top perp is: {perp(normal, cyl_axis2, diameter):.2f} mm")
print(f"Top CutSt is: {perp(normal, cyl_axis2, height):.2f} mm")
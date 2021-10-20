import copy
import numpy as np
import open3d as o3d
import os

from open3d_helper import perp, cloud_to_nparray, nparray_to_cloud, visualize_cloud, icp_multiple, vectors_to_transfor, \
    get_rotation_matrix, normalize, get_normals, plane_fitting, cylinder_fitting, draw_registration_result, transform_cloud

diameter = 200
height = 70

top, side = 4 * [None], 4 * [None]
for i in range(4):
    top[i] = o3d.io.read_point_cloud(rf"D:\Data\2802. Umicore\BlackRefCyl\cloud_top_{i}.ply").voxel_down_sample(1)
    side[i] = o3d.io.read_point_cloud(rf"D:\Data\2802. Umicore\BlackRefCyl\cloud_side_{i}.ply").voxel_down_sample(1)

top_normals = 4 * [None]
for i, cloud_t in enumerate(top):
    top_normals[i] = plane_fitting(cloud_t, diameter)
normal = normalize(np.array(top_normals).mean(axis=0))

cyl_axis = cylinder_fitting(side[0] + side[1] + side[2] + side[3], height)

print(f"Top perp is: {perp(normal, cyl_axis, diameter):.2f} mm")
print(f"Top CutSt is: {perp(normal, cyl_axis, height):.2f} mm")
_ = 'bp'


# New attempt - Use side for everything - Same result even though circle traced by top normal is different
top_normals = 4 * [None]
for i, cloud_t in enumerate(side):
    top_normals[i] = plane_fitting(cloud_t, diameter)
normal = normalize(np.array(top_normals).mean(axis=0))

cyl_axis = cylinder_fitting(side[0] + side[1] + side[2] + side[3], height)

print(f"Top perp is: {perp(normal, cyl_axis, diameter):.2f} mm")
print(f"Top CutSt is: {perp(normal, cyl_axis, height):.2f} mm")

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
# side0 = side0[good_idx, :]
# side0_normals = get_normals(side0)
# ndn = np.dot(side0_normals, normal_s0_top)
# ndn2 = 1 - ndn.mean()
# perp_new = 330 * np.sqrt(1 - ndn2 ** 2) / ndn2
# print(perp_new)

# next attempt - outside lib
# from cylinder_analysis_o3d import fit
# cyl_all = cloud_to_nparray(side[0] + side[1] + side[2] + side[3])
# w_fit, C_fit, r_fit, fit_err = fit(cyl_all)

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

# next attempt - correct side slope by icp to top surface first - side scanner must measure top surface incorrectly?
new_side, new_side2 = 4 * [None], 4 * [None]
for i in range(4):
    # new_side[i] = icp_multiple(top[i], [side[i]])
    max_iterations = 100
    threshold = 4
    trans_init = np.eye(4)
    reg_p2p = o3d.registration.registration_icp(side[i], top[i], threshold, trans_init,
                                                o3d.registration.TransformationEstimationPointToPoint(),
                                                o3d.registration.ICPConvergenceCriteria(max_iteration=max_iterations))
    new_side[i] = transform_cloud(side[i], reg_p2p.transformation)
    reg_p2p = o3d.registration.registration_icp(new_side[i], top[0], threshold, trans_init,
                                                o3d.registration.TransformationEstimationPointToPoint(),
                                                o3d.registration.ICPConvergenceCriteria(max_iteration=max_iterations))
    new_side2[i] = transform_cloud(new_side[i], reg_p2p.transformation)
    # check that it worked
    print(perp(plane_fitting(top[i], diameter), plane_fitting(new_side[i], diameter), diameter))
cyl_axis = cylinder_fitting(new_side[0] + new_side[1] + new_side[2] + new_side[3], height)
print(f"Top perp is: {perp(normal, cyl_axis, diameter):.2f} mm")
combined_cloud = icp_multiple(new_side[0], new_side[1:])
cyl_axis = cylinder_fitting(combined_cloud, height)
normal0 = plane_fitting(top[0], diameter)
print(f"Top perp is: {perp(normal0, cyl_axis, diameter):.2f} mm")

# next attempt break each section into angles, works somewhat
angle_delimeter = np.linspace(-np.pi, np.pi, 101)
axes = np.zeros((4, angle_delimeter.size - 1, 3))
for s in range(4):
    print(f"S{s}")
    scan = cloud_to_nparray(side[s])
    good_idx = (scan[:, 2] < height - 10) & (scan[:, 2] > 10)
    side_filt = scan[good_idx, :]
    side_angles = np.arctan2(side_filt[:, 1], side_filt[:, 0])
    for i in range(angle_delimeter.size - 1):
        angle_lower = angle_delimeter[i]
        angle_higher = angle_delimeter[i + 1]
        angle_idx = (side_angles > angle_lower) & (side_angles < angle_higher)
        if angle_idx.sum() < 500:
            continue
        cyl_reduced = side_filt[angle_idx, :]
        U, S, VT = np.linalg.svd(cyl_reduced - cyl_reduced.mean(axis=0))
        angle_axis = VT[0, :] if VT[0, 2] > 0 else -VT[0, :]
        axes[s, i, :] = angle_axis
        # print(f"Top perp is: {perp([0, 0, 1], angle_axis, diameter):.2f} mm")
perp(normal, normalize(np.array([normalize(axes[s, axes[s,:,:].any(axis=-1), :].mean(axis=0)) for s in range(4)]).mean(axis=0)), diameter)

# next attempt, do 4 cylinder fittings on best piece of cloud
cyl_axes = 4 * [None]
for i in range(4):
    scan = cloud_to_nparray(side[i])
    good_idx = (scan[:, 2] < height - 10) & (scan[:, 2] > 10)
    side_filt = scan[good_idx, :]
    side_angles = np.arctan2(side_filt[:, 1], side_filt[:, 0])
    mean_angle = np.arctan2(side_filt[:, 1].mean(), side_filt[:, 0].mean()) #side_angles.mean()
    if min(np.abs(mean_angle - np.pi), np.abs(mean_angle + np.pi)) < np.pi/8:
        offset = np.pi if mean_angle < 0 else -np.pi
        mean_angle += offset
        side_angles += offset
        side_angles[side_angles < -np.pi] += 2 * np.pi
        side_angles[side_angles > np.pi] -= 2 * np.pi
    good_idx2 = np.abs(side_angles - mean_angle) < np.pi/4
    side_filt2 = side_filt[good_idx2, :]
    side_angles2 = side_angles[good_idx2]
    # print(side_filt2.size)
    # cyl_axes[i] = cylinder_fitting(nparray_to_cloud(side_filt2), height)  # Code below is fitting with angle delimeter method
    angle_delimeter = np.linspace(-np.pi/4, np.pi/4, 21) + mean_angle
    angle_axes = []
    for j in range(angle_delimeter.size - 1):
        angle_lower = angle_delimeter[j]
        angle_higher = angle_delimeter[j + 1]
        angle_idx = (side_angles2 > angle_lower) & (side_angles2 < angle_higher)
        cyl_reduced = side_filt2[angle_idx, :]
        U, S, VT = np.linalg.svd(cyl_reduced - cyl_reduced.mean(axis=0))
        angle_axes.append(VT[0, :] if VT[0, 2] > 0 else -VT[0, :])
    cyl_axes[i] = normalize(np.array(angle_axes).mean(axis=0))

cyl_axis = normalize(np.array(cyl_axes).mean(axis=0))
print(f"Top perp is: {perp(normal, cyl_axis, diameter):.2f} mm")

# Next attempt
side_clouds = [o3d.io.read_point_cloud(rf"D:\Data\2802. Umicore\BadPerp\cloud_side_{i}.ply") for i in range(4)]
combined_side_cloud = (side_clouds[0] + side_clouds[1] + side_clouds[2] + side_clouds[3]).voxel_down_sample(1)
combined_side_cloud = cloud_to_nparray(combined_side_cloud)
z_range = np.arange(8, height - 5, 5)
centroids = np.zeros((z_range.size - 1, 3))
for z in range(z_range.size - 1):
    z_lower = z_range[z]
    z_upper = z_range[z + 1]
    z_idx = (combined_side_cloud[:, 2] > z_lower) & (combined_side_cloud[:, 2] < z_upper)
    centroids[z] = combined_side_cloud[z_idx, :].max(axis=0)/2 + combined_side_cloud[z_idx, :].min(axis=0)/2
centroids = centroids[:-1, :]
U, S, VT = np.linalg.svd(centroids - centroids.mean(axis=0))
cyl_axis = VT[0, :]
print(f"Top perp is: {perp(normal, cyl_axis, diameter):.2f} mm")
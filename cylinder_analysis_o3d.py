import copy
import numpy as np
import open3d as o3d
import os

from open3d_helper import perp, cloud_to_nparray, nparray_to_cloud, visualize_cloud, icp_multiple, vectors_to_transfor, \
    get_rotation_matrix, normalize, get_normals, plane_fitting, cylinder_fitting, draw_registration_result


def flip_normal(normal):
    theta = np.deg2rad(14.75 - 270)
    s, c = np.sin(2 * theta), np.cos(2 * theta)
    flipped_normal = [c * normal[0] + s * normal[1], s * normal[0] - c * normal[1], -normal[2]]
    return flipped_normal


def rotation_matrix_from_coordinates(x,y,z,o=None):
    if o is None:
        o = np.array([0, 0, 0])
    rot_matrix = np.eye(4)
    for i, v in enumerate([x, y, z, o]):
        rot_matrix[:3, i] = v
    return rot_matrix


def get_rotation_matrix(start, target):
    X = normalize(np.cross(start, target))
    Y = normalize(start)
    Z = normalize(np.cross(X, Y))
    A = rotation_matrix_from_coordinates(X, Y, Z)
    Y = normalize(target)
    Z = normalize(np.cross(X, Y))
    B = rotation_matrix_from_coordinates(X, Y, Z)
    M = np.dot(B, np.linalg.inv(A))
    return M


diameter = 317
height = 76

top, side = 6 * [None], 6 * [None]
for i in range(6):
    side[i] = o3d.io.read_point_cloud(rf"C:\Projects\Umicore_SW\inspection_sw\out\build\RelWithDebInfo\cloud_side_{i}.ply").voxel_down_sample(1)
    top[i] = o3d.io.read_point_cloud(rf"C:\Projects\Umicore_SW\inspection_sw\out\build\RelWithDebInfo\cloud_top_{i}.ply").voxel_down_sample(1)
    # side[i] = o3d.io.read_point_cloud(rf"C:\Users\ahe\Desktop\par10\cloud_side_{i}.ply").voxel_down_sample(1)
    # top[i] = o3d.io.read_point_cloud(rf"C:\Users\ahe\Desktop\par10\cloud_top_{i}.ply").voxel_down_sample(1)

top_normals = 6 * [None]
for i, cloud_t in enumerate(top):
    top_normals[i] = plane_fitting(cloud_t, diameter)
normal = normalize(np.array(top_normals[:4]).mean(axis=0))

cyl_axis = cylinder_fitting(side[0] + side[1] + side[2] + side[3], height)

print(f"Top perp is: {perp(normal, cyl_axis, diameter):.2f} mm")
print(f"Top CutSt is: {perp(normal, cyl_axis, height):.2f} mm")

normal2 = normalize(np.array(top_normals[4:]).mean(axis=0))
cyl_axis2 = cylinder_fitting(side[4] + side[5], height)

print(f"Bottom perp is: {perp(normal2, cyl_axis2, diameter):.2f} mm")
print(f"Bottom CutSt is: {perp(normal2, cyl_axis2, height):.2f} mm")

# New parallelism
# print(f"New Parallelism is: {perp(normal, [0,0,1], diameter)/2 + perp(normal2, [0,0,1], diameter)/2:.2f} mm")
print(f"New Parallelism is: {perp(normal, [0,0,1], diameter):.2f} mm")

# Old Parallelism
# theta = np.deg2rad(14.75 - 270)
# s, c = np.sin(2*theta), np.cos(2*theta)
# flipped_normal = [c * normal2[0] + s * normal2[1], s * normal2[0] - c * normal2[1], -normal2[2]]
#
# print(f"Parallelism is: {perp(normal, flipped_normal, diameter):.2f} mm")
#
# # Old old Parallelism
# normal_top = normalize(normal + ([0, 0, 1] - cyl_axis))
# normal_bottom = normalize(normal2 + ([0, 0, 1] - cyl_axis2))
# flipped_normal = [c * normal_bottom[0] + s * normal_bottom[1], s * normal_bottom[0] - c * normal_bottom[1], -normal_bottom[2]]
# print(f"Old Parallelism is: {perp(normal_top, flipped_normal, diameter):.2f} mm")

# Bottom perp revisited 1
cyl_axis3 = cylinder_fitting(side[1] + side[3], height)
# cyl_axis4 = normalize(cyl_axis + cyl_axis2 - cyl_axis3)  # flip cyl_axis3 and cyl_axis
R = get_rotation_matrix(cyl_axis2, cyl_axis3)  # flip cyl_axis3 and cyl_axis
cyl_axis5 = R.dot(np.append(cyl_axis, 1))[:3]
print(f"New Bottom perp is: {perp(normal2, cyl_axis5, diameter):.2f} mm")

# Bottom perp revisited 2
cyl_axis3 = cylinder_fitting(side[1] + side[3], height)
R = get_rotation_matrix(cyl_axis3, flip_normal(cyl_axis2))
print(f"New Bottom perp is: {perp(flip_normal(normal2), R.dot(np.append(cyl_axis, 1))[:3], diameter):.2f} mm")
# print(f"New Bottom perp is: {perp(normal, R.dot(np.append(cyl_axis, 1))[:3], diameter):.2f} mm")
_ = 'bp'

# Debug code to identiy side skewness
for i in range(6):
    print(f"Side perp is: {perp([0,0,1], cylinder_fitting(side[i], height), diameter):.2f} mm")

# Bottom perp revisited 3
cyl_axis3 = cylinder_fitting(side[1] + side[3], height)
R = get_rotation_matrix(flip_normal(cyl_axis3), flip_normal(cyl_axis))
cyl_axis5 = R.dot(np.append(cyl_axis2, 1))[:3]
print(f"New Bottom perp is: {perp(normal2, cyl_axis5, diameter):.2f} mm")

# Bottom perp revisited 4
print(f"Bottom perp is: {perp([0,0,1], cyl_axis, diameter):.2f} mm")
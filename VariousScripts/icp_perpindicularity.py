import copy
import numpy as np
import open3d as o3d
import os

from open3d_helper import perp, cloud_to_nparray, nparray_to_cloud, visualize_cloud, icp_multiple, vectors_to_transfor, get_rotation_matrix, normalize
from Open3d_PlaneFitting import plane_fitting
from Open3d_CylinderFitting import cylinder_fitting

max_iterations = 30
threshold = 2
trans_init = np.eye(4)

def draw_registration_result(source, target, transformation=None):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    if transformation is not None:
        source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

if "Loading clouds":
    clouds_s, clouds_t = 5 * [None], 5 * [None]
    clouds_s[0] = o3d.io.read_point_cloud(r"data/cloud_side_0.ply").voxel_down_sample(1)
    clouds_s[1] = o3d.io.read_point_cloud(r"data/cloud_side_1.ply").voxel_down_sample(1)
    clouds_s[2] = o3d.io.read_point_cloud(r"data/cloud_side_2.ply").voxel_down_sample(1)
    clouds_s[3] = o3d.io.read_point_cloud(r"data/cloud_side_3.ply").voxel_down_sample(1)
    clouds_s[4] = o3d.io.read_point_cloud(r"data/cloud_side_4.ply").voxel_down_sample(1)
    clouds_t[0] = o3d.io.read_point_cloud(r"data/cloud_top_0.ply").voxel_down_sample(1)
    clouds_t[1] = o3d.io.read_point_cloud(r"data/cloud_top_1.ply").voxel_down_sample(1)
    clouds_t[2] = o3d.io.read_point_cloud(r"data/cloud_top_2.ply").voxel_down_sample(1)
    clouds_t[3] = o3d.io.read_point_cloud(r"data/cloud_top_3.ply").voxel_down_sample(1)
    clouds_t[4] = o3d.io.read_point_cloud(r"data/cloud_top_4.ply").voxel_down_sample(1)

# Working prototype
last_axis = cylinder_fitting(clouds_s[0], 268)
current_axis = cylinder_fitting(clouds_s[4], 268)
average_axis = cylinder_fitting(clouds_s[0] + clouds_s[1] + clouds_s[2] + clouds_s[3], 268)
corrected_axis = np.dot(get_rotation_matrix(last_axis, average_axis), current_axis)
# corrected_axis = normalize(current_axis - last_axis + average_axis)


last_normal = plane_fitting(clouds_t[0], 268)
current_normal = plane_fitting(clouds_t[4], 268)
surface_normals = np.array([plane_fitting(c, 268) for c in clouds_t[:4]])
average_normal = normalize(surface_normals.mean(axis=0))
corrected_normal = np.dot(get_rotation_matrix(last_normal, average_normal), current_normal)

print(f"Top perp: {perp(normalize(average_axis), normalize(average_normal))}")
print(f"Bottom perp: {perp(normalize(corrected_axis), normalize(corrected_normal))}")

rot_bottom = np.dot(get_rotation_matrix(corrected_axis, [0,0,1]), corrected_normal)
rot_top = np.dot(get_rotation_matrix(average_axis, [0,0,1]), average_normal)
theta = np.deg2rad(14.75)  # rad
R = np.array([[np.cos(2*theta), np.sin(2*theta), 0], [np.sin(2*theta), -np.cos(2*theta), 0], [0, 0, -1]])
print(f"Parallelism: {perp(rot_top, np.dot(R, rot_bottom))}")


# Attempt at cylinder
cyl_axis0 = cylinder_fitting(clouds_s[0] + clouds_s[1] + clouds_s[2] + clouds_s[3], 268)
theta = 0 # rad
T = np.array([[np.cos(2*theta), np.sin(2*theta), 0, 0], [np.sin(2*theta), -np.cos(2*theta), 0, 0], [0, 0, -1, 268], [0, 0, 0, 1]])
clouds_s[4] = clouds_s[4].transform(T)
reg_p2p = o3d.registration.registration_icp(clouds_s[0], clouds_s[4], threshold, trans_init,
                                             o3d.registration.TransformationEstimationPointToPoint(),
                                             o3d.registration.ICPConvergenceCriteria(max_iteration=max_iterations))
new_cyl_axis = np.dot(reg_p2p.transformation[:3, :3], cyl_axis0)

# Attempt af surface normal
surface_normals = np.array([plane_fitting(c, 268) for c in clouds_t[:4]])
surface_normal_average = normalize(surface_normals.mean(axis=0))
rotate_to_avereage = get_rotation_matrix(surface_normals[0], surface_normal_average)
bottom_normal = plane_fitting(clouds_t[4], 268)
surface_normal_new = np.dot(rotate_to_avereage, bottom_normal)
# Maybe switch x axis sign?

if not os.path.exists("data/cloud_side_4_aligned.ply"):
    # Align top 4 and s4 to s0 and its cylinder axis
    theta = 0 # rad
    T = np.array([[np.cos(2*theta), np.sin(2*theta), 0, 0], [np.sin(2*theta), -np.cos(2*theta), 0, 0], [0, 0, -1, 268], [0, 0, 0, 1]])

    #draw_registration_result(clouds_s[4], clouds_s[0])

    clouds_s[4] = clouds_s[4].transform(T)
    clouds_t[4] = clouds_t[4].transform(T)

    # draw_registration_result(clouds_s[4], clouds_s[0])

    trans_init = np.eye(4)
    reg_p2p = o3d.registration.registration_icp(clouds_s[4], clouds_s[0], threshold, trans_init,
                                                o3d.registration.TransformationEstimationPointToPoint(),
                                                o3d.registration.ICPConvergenceCriteria(max_iteration=max_iterations))



    # draw_registration_result(clouds_s[4], clouds_s[0], reg_p2p.transformation)

    clouds_s[4] = clouds_s[4].transform(reg_p2p.transformation)
    clouds_t[4] = clouds_t[4].transform(reg_p2p.transformation)  # TODO combine earlier and this transform into one

    o3d.io.write_point_cloud("data/cloud_side_4_aligned.ply", clouds_s[4])
    o3d.io.write_point_cloud("data/cloud_top_4_aligned.ply", clouds_t[4])
else:
    # clouds_s[4] = o3d.io.read_point_cloud(r"data/cloud_side_4_aligned.ply")
    clouds_t[4] = o3d.io.read_point_cloud(r"data/cloud_top_4_aligned.ply").voxel_down_sample(1)


clouds_t[4] = clouds_t[4]
surface_normal = plane_fitting(clouds_t[4], diameter=268)

surface_normal_top = plane_fitting(clouds_t[0], diameter=268)

combined_cloud = icp_multiple(clouds_s[0], clouds_s[1:4])

cyl_axis = cylinder_fitting(combined_cloud, height=268)

print(f"Perp of top: {perp(cyl_axis, surface_normal_top)} mm.")  # Align everything with icp not good enough
print(f"Perp of bottom: {perp(cyl_axis, surface_normal)} mm.")

# Check current method
clouds_s[0] = o3d.io.read_point_cloud(r"data/cloud_side_0.ply").voxel_down_sample(1)
clouds_s[1] = o3d.io.read_point_cloud(r"data/cloud_side_1.ply").voxel_down_sample(1)
clouds_s[2] = o3d.io.read_point_cloud(r"data/cloud_side_2.ply").voxel_down_sample(1)
clouds_s[3] = o3d.io.read_point_cloud(r"data/cloud_side_3.ply").voxel_down_sample(1)
cyl_axis = cylinder_fitting(clouds_s[0] + clouds_s[1] + clouds_s[2] + clouds_s[3], 268)
clouds_t[0] = o3d.io.read_point_cloud(r"data/cloud_top_0.ply").voxel_down_sample(1)
clouds_t[1] = o3d.io.read_point_cloud(r"data/cloud_top_1.ply").voxel_down_sample(1)
clouds_t[2] = o3d.io.read_point_cloud(r"data/cloud_top_2.ply").voxel_down_sample(1)
clouds_t[3] = o3d.io.read_point_cloud(r"data/cloud_top_3.ply").voxel_down_sample(1)
surface_normals = np.array([plane_fitting(c, 268) for c in clouds_t[:4]])
sn = surface_normals.mean(axis=0)
sn /= np.linalg.norm(sn)
perp(cyl_axis, sn)

_ = 'bp'

# Align everything to s0
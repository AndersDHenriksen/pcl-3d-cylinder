from copy import deepcopy
import numpy as np
import open3d as o3d

from open3d_helper import perp, cloud_to_nparray, nparray_to_cloud, visualize_cloud
from Open3d_CylinderFitting import cylinder_fitting
from Open3d_PlaneFitting import plane_fitting


ply_folder = r"D:\Data\2802. Umicore\Measurements1 - Repeats\200628 Yellow 7\ply"


max_iterations = 30
threshold = 2

try_icp = False

def draw_registration_result(source, target, transformation=None):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    if transformation is not None:
        source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


clouds = [o3d.io.read_point_cloud(rf"{ply_folder}\cloud_side_{i}.ply") for i in range(4)]
top_cloud = o3d.io.read_point_cloud(rf"{ply_folder}\cloud_top.ply")

normal = plane_fitting(top_cloud, 268)
if try_icp:
    combined_cloud = deepcopy(clouds[3])
    for i in range(3)[::-1]:
        reg_p2p = o3d.registration.registration_icp(clouds[i], combined_cloud, threshold, np.eye(4),
                o3d.registration.TransformationEstimationPointToPoint(),
                o3d.registration.ICPConvergenceCriteria(max_iteration=max_iterations))
        print(reg_p2p)
        print("Transformation is:")
        print(reg_p2p.transformation)
        # draw_registration_result(clouds[i], combined_cloud, reg_p2p.transformation)
        # trans_init = reg_p2p.transformation

        icp_cloud = deepcopy(clouds[i])
        combined_cloud += icp_cloud.transform(reg_p2p.transformation)
    cyl_axis = cylinder_fitting(combined_cloud, 268)
    print(perp(normal, cyl_axis))

combined_cloud_raw = clouds[0] + clouds[1] + clouds[2] + clouds[3]
cyl_axis = cylinder_fitting(combined_cloud_raw, 268)
print(perp(normal, cyl_axis))
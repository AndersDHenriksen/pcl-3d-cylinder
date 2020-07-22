import copy

import numpy as np
import open3d as o3d

max_iterations = 30
threshold = 2


def draw_registration_result(source, target, transformation=None):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    if transformation is not None:
        source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


clouds = [None, None, None]
combined_cloud = o3d.io.read_point_cloud(r"D:\Data\2802. Umicore\Series7 - new calibration\big_ref_element\after_icp_installed\cloud_0.ply")
clouds[0] = o3d.io.read_point_cloud(r"D:\Data\2802. Umicore\Series7 - new calibration\big_ref_element\after_icp_installed\cloud_1.ply")
clouds[1] = o3d.io.read_point_cloud(r"D:\Data\2802. Umicore\Series7 - new calibration\big_ref_element\after_icp_installed\cloud_2.ply")
clouds[2] = o3d.io.read_point_cloud(r"D:\Data\2802. Umicore\Series7 - new calibration\big_ref_element\after_icp_installed\cloud_3.ply")

trans_init = np.eye(4)

for cloud in clouds:

    # print("Initial alignment")
    # evaluation = o3d.registration.evaluate_registration(source, target, threshold, trans_init)
    # print(evaluation)

    reg_p2p = o3d.registration.registration_icp(cloud, combined_cloud, threshold, trans_init,
            o3d.registration.TransformationEstimationPointToPoint(),
            o3d.registration.ICPConvergenceCriteria(max_iteration=max_iterations))
    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation)
    # draw_registration_result(cloud, combined_cloud, reg_p2p.transformation)

    # trans_init = reg_p2p.transformation

    combined_cloud += cloud.transform(reg_p2p.transformation)

_ = 'bp'
o3d.io.write_point_cloud(r"D:\Data\2802. Umicore\Series7 - new calibration\big_ref_element\after_icp_installed\icp_o3d_python.ply", combined_cloud)

# icp_pcl = o3d.io.read_point_cloud(r"D:\Data\2802. Umicore\Series7 - new calibration\big_ref_element\icp_pcl.ply")

# draw_registration_result(icp_pcl, combined_cloud)
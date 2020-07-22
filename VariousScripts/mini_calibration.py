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
    failed = True
    while failed:
        try:
            o3d.visualization.draw_geometries([cloud])
        except:
            pass
        else:
            failed = False

def vectors_to_transfor(x_axis, y_axis, normal, translation):
    return np.vstack((np.array([x_axis, y_axis, normal, translation]).T, [0, 0, 0, 1]))

def calc_normal_and_displacement(cloud_path, is_stone_present=False):
    cloud = o3d.io.read_point_cloud(cloud_path)
    cloud_npa = cloud_to_nparray(cloud)

    cloud_normals = np.asarray(cloud.normals)
    z_est = np.median(cloud_npa[np.abs(cloud_normals[:,2]) > 0.95, 2])

    good_idx = (np.abs(cloud_npa[:, 2] - z_est) < 10) & (np.abs(cloud_npa[:, 0]) < 160) & (np.abs(cloud_npa[:, 1]) < 160)
    cloud_npa = cloud_npa[good_idx, :]

    cloud_r2 = cloud_npa[:, 0] ** 2 + cloud_npa[:, 1] ** 2

    r_upper = 1.25 * np.sqrt(np.mean(cloud_r2))
    if is_stone_present:
        r_upper = 1.1 * np.sqrt(np.mean(cloud_r2))
    r_lower = 50  # 30

    # Filter away more points
    inside_idx = (cloud_r2 > r_lower ** 2) & (cloud_r2 < r_upper ** 2)
    cloud_surface = cloud_npa[inside_idx, :]

    visualize_cloud(nparray_to_cloud(cloud_surface))

    # Calc centroid and fit plane
    cloud_surface = nparray_to_cloud(cloud_surface)
    clouds_down = cloud_surface.voxel_down_sample(voxel_size=3)
    down_np = np.asarray(clouds_down.points)
    U, S, VT = np.linalg.svd(down_np - down_np.mean(axis=0))
    normal = VT[2, :]
    if normal[2] < 0:
        normal = -normal
    # offset = cloud_surface[:, 2].mean()

    # cloud.normals
    good_normal_idx = np.dot(np.asarray(cloud.normals)[good_idx,:], normal) > 0.98
    inside_idx2 = (cloud_r2 > r_lower ** 2)
    cloud_top = cloud_npa[good_normal_idx & inside_idx2, :]
    # offset = cloud_top.mean(axis=0)
    offset = nparray_to_cloud(cloud_top).voxel_down_sample(3).get_center()

    if is_stone_present:
        offset[2] -= 51.54


    return offset, normal

if __name__ == "__main__":
    offset_side, normal_side = calc_normal_and_displacement(r"I:\Stone-config12\270\side.ply", True)
    offset_top, normal_top = calc_normal_and_displacement(r"I:\Stone-config12\270\top.ply", True)
    # offset_side, _ = calc_normal_and_displacement(r"E:\CalibrationDebug\200627 USB\270_side.ply")
    # offset_top, _ = calc_normal_and_displacement(r"E:\CalibrationDebug\200627 USB\270_top.ply")
    #json_54 = [-0.939357340335846, 0.004611530341207981, -0.34290894865989685, 0, 0, 0.9999096393585205, 0.013447050005197525, 0, 0.34293994307518005, 0.012631584890186787, -0.9392724633216858, 0, -233.57696533203125, -29.67249298095703, 916.7695922851562, 1]
    #json_19 = [-0.3304727375507355, -0.018310150131583214, 0.9436378479003906, 0, 0, 0.9998117685317993, 0.019400134682655334, 0, -0.9438155889511108, 0.006411215756088495, -0.33041051030158997, 0, 862.0155029296875, -18.695938110351562, 426.70465087890625, 1]
    json_54 = [0.9141826026832479, 0.21953118977673644, -0.3406996504143366, 0.0, 0.2373479327044896, -0.971362905389262, 0.01096243178158713, 0.0, -0.32853637596142626, -0.09088601437456754, -0.9401083468988407, 0.0, 218.02315400908617, 81.07482706108144, 1075.5786662539788, 1.0]
    json_19 = [-0.330267847572644, -0.023838913469322865, 0.9435861036050365, 0.0, -0.0209545806818147, 0.9996197831127275, 0.017920169600264504, 0.0, -0.9436546775930857, -0.013853998142853383, -0.3306417867489047, 0.0, 862.0620769716616, 0.23388706183223107, 585.5058507574187, 1.0]

    T_TopToRot = np.array(json_54).reshape((4, 4)).T
    T_SideToRot = np.array(json_19).reshape((4, 4)).T

    # # Top
    # nz = normal_top
    # nx = np.cross([0, 1, 0], nz)
    # ny = np.cross(nz, nx)
    # origin = np.array([0, 0, offset_top[2]])
    # T_CalibToRotT = vectors_to_transfor(nx, ny, nz, origin)
    # T_RotTToCalib = np.linalg.inv(T_CalibToRotT)
    # T_TopToCalib = np.dot(T_RotTToCalib, T_TopToRot)
    #
    # # Side
    # nz = normal_side
    # nx = np.cross([0, 1, 0], nz)
    # ny = np.cross(nz, nx)
    # origin = np.array([0, 0, offset_side[2]])
    # T_CalibToRotS = vectors_to_transfor(nx, ny, nz, origin)
    # T_SideToCalib = np.dot(np.linalg.inv(T_CalibToRotS), T_SideToRot)
    #
    # better_54 = T_TopToCalib.T.ravel()
    # better_19 = T_SideToCalib.T.ravel()
    #
    # import json
    # print(json.dumps(better_54.tolist()))
    # print(json.dumps(better_19.tolist()))

    # Try to fix alignments
    # Side
    nz = normal_side
    towards_side = T_SideToRot[:3, 3]
    ny = np.cross(nz, towards_side)
    ny /= np.linalg.norm(ny)
    nx = np.cross(ny, nz)
    origin = np.array([0, 0, offset_side[2]])
    T_CalibToRotS = vectors_to_transfor(nx, ny, nz, origin)
    T_SideToCalib = np.dot(np.linalg.inv(T_CalibToRotS), T_SideToRot)
    np.save("T_SideToCalib_latest.npy", T_SideToCalib)
    better_19 = T_SideToCalib.T.ravel()
    print(json.dumps(better_19.tolist()))

    # Top
    # T_pose = np.array([  0.902302, -0.0918253,   0.421211,    -451.64,
    #                     -0.116627,  -0.992613,  0.0334415,     114.13,
    #                      0.415029, -0.0792989,  -0.906346,    844.273,
    #                             0,          0,          0,          1]).reshape(4,4)
    # S_pose = np.array([   -0.8898,  0.222777, -0.398279,   302.141,
    #                     0.0910382,   0.94186,  0.323439,   -190.02,
    #                      0.447178,  0.251537, -0.858348,   899.937,
    #                             0,         0,         0,         1]).reshape(4,4)
    # T_SideToTop = np.dot(np.linalg.inv(T_pose), S_pose)


    # T_SideToTop = np.array([[-6.28224e-01,  1.90450e-01, -7.54363e-01,  7.39663e+02],
    #                         [-4.24434e-02, -9.76523e-01, -2.11191e-01,  2.23462e+02],
    #                         [-7.76874e-01, -1.00658e-01,  6.21558e-01,  2.56588e+02],
    #                         [ 0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e+00]])
    T_SideToTop = np.load(r'T_SideToTop_SweetSpotCalib.npy')


    SideLocationSeenFromTop = T_SideToTop[:3, 3]
    towards_side = np.dot(T_TopToRot, np.append(SideLocationSeenFromTop,1))[:3]
    nz = normal_top
    ny = np.cross(nz, towards_side)
    ny /= np.linalg.norm(ny)
    nx = np.cross(ny, nz)
    origin = np.array([0, 0, offset_top[2]])
    T_CalibToRotT = vectors_to_transfor(nx, ny, nz, origin)
    T_TopToCalib = np.dot(np.linalg.inv(T_CalibToRotT), T_TopToRot)
    np.save("T_TopToCalib_latest.npy", T_TopToCalib)
    better_54 = T_TopToCalib.T.ravel()
    print(json.dumps(better_54.tolist()))

    _ = 'bp'

    # Verify result
    # stone_normal, stone_offset = calc_normal_and_displacement(r"E:\Stone270-config12\stone_top.ply", 1)
    _ = 'bp'
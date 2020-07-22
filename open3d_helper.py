import copy
import json
import numpy as np
import open3d as o3d

def copy_cloud(cloud):
    return copy.deepcopy(cloud)

def transform_cloud(cloud, transformation):
    return copy_cloud(cloud).transform(transformation)

def perp(n1, n2, diameter=258):
    ndn = np.dot(n1, n2)
    return diameter * np.sqrt(1 - min(1, ndn) ** 2) / np.abs(ndn)

def cloud_to_nparray(cloud):
    return np.asarray(cloud.points)

def nparray_to_cloud(data):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(data)
    return cloud

def visualize_cloud(cloud):
    if isinstance(cloud, np.ndarray):
        cloud = nparray_to_cloud(cloud)
    if cloud.is_empty():
        raise Exception("Cannot visualize empty cloud")
    failed = True
    while failed:
        try:
            o3d.visualization.draw_geometries([cloud])
        except:
            pass
        else:
            failed = False

def icp_multiple(start_cloud, append_clouds):
    max_iterations = 30
    threshold = 2
    trans_init = np.eye(4)
    combined_cloud = copy_cloud(start_cloud)

    for cloud in append_clouds:

        # print("Initial alignment")
        # evaluation = o3d.registration.evaluate_registration(source, target, threshold, trans_init)
        # print(evaluation)

        reg_p2p = o3d.registration.registration_icp(cloud, combined_cloud, threshold, trans_init,
                o3d.registration.TransformationEstimationPointToPoint(),
                o3d.registration.ICPConvergenceCriteria(max_iteration=max_iterations))

        # print("Transformation is:")
        # print(reg_p2p.transformation)

        # draw_registration_result(cloud, combined_cloud, reg_p2p.transformation)

        # trans_init = reg_p2p.transformation

        combined_cloud += transform_cloud(cloud, reg_p2p.transformation)
    return combined_cloud

def vectors_to_transfor(x_axis, y_axis, normal, translation):
    return np.vstack((np.array([x_axis, y_axis, normal, translation]).T, [0, 0, 0, 1]))

def get_rotation_matrix(start_vector, target_vector, return_4d=False):
    X = np.cross(start_vector, target_vector)
    Y = start_vector
    Z = np.cross(X,Y)
    X /= np.linalg.norm(X)
    Y /= np.linalg.norm(Y)
    Z /= np.linalg.norm(Z)
    A = vectors_to_transfor(X, Y, Z, np.array([0, 0, 0]))

    Y2 = target_vector / np.linalg.norm(target_vector)
    Z2 = np.cross(X, Y2)
    Z2 /= np.linalg.norm(Z2)
    B = vectors_to_transfor(X, Y2, Z2, np.array([0, 0, 0]))
    T = np.dot(B, np.linalg.inv(A))
    if return_4d:
        return T
    return T[:3, :3]

def normalize(a):
    return a / np.linalg.norm(a)

def draw_registration_result(source, target, transformation=None):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    if transformation is not None:
        source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

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

def cylinder_fitting(cloud, height, plot=False):
    cloud = cloud_to_nparray(cloud)
    # cloud = cloud_to_nparray(nparray_to_cloud(cloud).voxel_down_sample(1))
    if plot:
        visualize_cloud(nparray_to_cloud(cloud))

    xc, yc, zc = [], [], []
    offset = 20 if height > 100 else 10
    for z in range(offset, height - offset, 2):
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
    if cyl_axis[2] < 0:
        cyl_axis = - cyl_axis
    return cyl_axis

def plane_fitting(cloud_top, diameter, plot=False):
    cloud_top.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.5, max_nn=30))
    cloud_top_npa = cloud_to_nparray(cloud_top)
    good_idx = np.abs(np.asarray(cloud_top.normals)[:, 2]) > 0.95
    cloud_r2 = cloud_top_npa[:,0]**2 + cloud_top_npa[:,1]**2
    inside_idx = (cloud_r2 < (diameter / 2 - 10) ** 2) & (cloud_r2 > 10 ** 2)

    cloud_top_npa = cloud_top_npa[good_idx & inside_idx, :]

    if plot:
        visualize_cloud(cloud_top_npa)
    downsampled_cloud = cloud_to_nparray(nparray_to_cloud(cloud_top_npa).voxel_down_sample(3))
    normal = np.linalg.svd(downsampled_cloud - downsampled_cloud.mean(axis=0))[2][2,:]
    if normal[2] < 0:
        normal = - normal
    return normal

def is_np(cloud):
    return isinstance(cloud, np.ndarray)

def enforce_cloud(cloud):
    if is_np(cloud):
        cloud = nparray_to_cloud(cloud)
    return cloud

def enforce_numpy(cloud):
    if not is_np(cloud):
        cloud = cloud_to_nparray(cloud)
    return cloud

def get_normals(cloud):
    cloud = enforce_cloud(cloud)
    cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.5, max_nn=30))
    return np.asarray(cloud.normals)
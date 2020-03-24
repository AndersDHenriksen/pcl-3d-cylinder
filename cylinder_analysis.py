import numpy as np
import pcl
import pcl.pcl_visualization

def cloud_from_array(array):
    cloud = pcl.PointCloud()
    cloud.from_array(array.astype(np.float32))
    return cloud


def center_cloud(cloud):
    return cloud_from_array(cloud - np.mean(cloud, 0))


def visualize_cloud(cloud):
    # visual = pcl.pcl_visualization.CloudViewing()
    # visual.ShowMonochromeCloud(cloud, b'cloud')
    #
    # v = True
    # while v:
    #     v = not (visual.WasStopped())

    viewer = pcl.pcl_visualization.PCLVisualizering("vis")
    viewer.AddPointCloud(cloud)
    viewer.AddCoordinateSystem(100)
    # viewer.AddCube(-100, 100, -100, 100, -1, 1, 200, 100, 100, b"xy-plane")
    # viewer.AddPlane()  # not implemented yet
    viewer.Spin()
    v = True
    while v:
        v = not (viewer.WasStopped())


def filter_cloud(cloud, direction, lower, upper):
    # assert direction in "xyz"  # It seems c++ plc has a fields member, but here the only way to get fields is to_file
    fil = cloud.make_passthrough_filter()
    fil.set_filter_field_name(direction)
    fil.set_filter_limits(lower, upper)
    cloud_filtered = fil.filter()
    return cloud_filtered


def surface_normals(cloud, r_search=0.5, kdtree=None):
    if kdtree is None:
        kdtree = cloud.make_kdtree()
    ne = cloud.make_NormalEstimation()
    ne.set_SearchMethod(kdtree)
    ne.set_RadiusSearch(r_search)
    cloud_normals = ne.compute()
    return cloud_normals  # returns X, Y, Z, curvature

def rotation_matrix_from_angle(angle, axis):
    assert axis in 'xyz' or axis in [0,1,2]  #TODO not working for int
    c, s = np.cos(angle), np.sin(angle)
    if axis == 0 or axis == 'x':
        m = [[1, 0, 0], [0, c, -s], [0, s, c]]
    elif axis == 1 or axis == 'y':
        m = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    elif axis == 2 or axis == 'z':
        m = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    return np.array(m)


def rotate_cloud(cloud, angle, axis):
    convert_to_cloud = False
    if isinstance(cloud, pcl.PointCloud):
        convert_to_cloud = True
        cloud = cloud.to_array()
    rot_mat = rotation_matrix_from_angle(angle, axis)
    rot_cloud = np.dot(rot_mat, cloud.T).T  #TODO is this the correct way?
    if convert_to_cloud:
        cloud_from_array(rot_cloud)
    return rot_cloud




cloud = center_cloud(pcl.load('combined_point_cloud_test.ply'))
# visualize_cloud(cloud)
normals = surface_normals(cloud)

# Get elements with low curvature
# curvature = normals.to_array()[:, 3]
# flat_idx = normals.to_array()[:, 3] == 0 # Didn't work as intended
normals_array = normals.to_array()
flat_idx = (normals_array[:, 0] < .5) & (normals_array[:, 2] > .5)
flat_cloud = cloud.extract(np.flatnonzero(flat_idx))
# visualize_cloud(flat_cloud)

# Find plane
# cloud_filtered = filter_cloud(cloud, "z", -0.01, 0.01)
seg = flat_cloud.make_segmenter_normals(ksearch=50)
seg.set_optimize_coefficients(True)
seg.set_model_type(pcl.SACMODEL_NORMAL_PLANE)
seg.set_normal_distance_weight(0.1)
seg.set_method_type(pcl.SAC_RANSAC)
seg.set_max_iterations(100)
seg.set_distance_threshold(0.03)
indices, plane_model = seg.segment()

angle_x = np.arctan2(np.sqrt(plane_model[1]**2 + plane_model[2]**2), plane_model[0])
new_cloud = rotate_cloud(flat_cloud, angle_x, 'x')


_ = 'breakpoint'
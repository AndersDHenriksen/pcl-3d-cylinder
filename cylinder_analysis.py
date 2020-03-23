import numpy as np
import pcl
import pcl.pcl_visualization


def center_cloud(cloud):
    centred = cloud - np.mean(cloud, 0)
    centered_cloud = pcl.PointCloud()
    centered_cloud.from_array(centred)
    return centered_cloud


def visualize_cloud(cloud):
    # visual = pcl.pcl_visualization.CloudViewing()
    # visual.ShowMonochromeCloud(cloud, b'cloud')
    #
    # v = True
    # while v:
    #     v = not (visual.WasStopped())

    viewer = pcl.pcl_visualization.PCLVisualizering("vis")
    viewer.AddPointCloud(cloud)
    viewer.AddCoordinateSystem()
    viewer.AddCube(-100, 100, -100, 100, -1, 1, 200, 100, 100, b"xy-plane")
    # viewer.AddPlane()  # not implemented yet
    viewer.Spin()
    v = True
    while v:
        v = not (viewer.WasStopped())


def filter_cloud(cloud, direction, lower, upper):
    assert direction in "xyz"
    fil = cloud.make_passthrough_filter()
    fil.set_filter_field_name("z")
    fil.set_filter_limits(lower, upper)
    cloud_filtered = fil.filter()
    return cloud_filtered


def surface_normals(cloud, r_search=0.03, kdtree=None):
    if kdtree is None:
        kdtree = cloud.make_kdtree()
    ne = cloud.make_NormalEstimation()
    ne.set_SearchMethod(kdtree)
    ne.set_RadiusSearch(r_search) #3 cm search radius?
    cloud_normals = ne.compute()
    return cloud_normals  # returns X, Y, Z, curvature


cloud = center_cloud(pcl.load('combined_point_cloud_test.ply'))

normals = surface_normals(cloud)

# visualize_cloud(cloud)

# Find plane
cloud_filtered = filter_cloud(cloud, "z", -0.01, 0.01)
seg = cloud_filtered.make_segmenter_normals(ksearch=50)
seg.set_optimize_coefficients(True)
seg.set_model_type(pcl.SACMODEL_NORMAL_PLANE)
seg.set_normal_distance_weight(0.1)
seg.set_method_type(pcl.SAC_RANSAC)
seg.set_max_iterations(100)
seg.set_distance_threshold(0.03)
indices, model = seg.segment()



_ = 'breakpoint'
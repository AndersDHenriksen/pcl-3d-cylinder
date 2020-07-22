import numpy as np


def vectors_to_transfor(x_axis, y_axis, normal, translation):
    return np.vstack((np.array([x_axis, y_axis, normal, translation]).T, [0, 0, 0, 1]))


def generate_data():
    origo = np.array([4, 5, 6])
    normal = np.array([1.5, 1, 0.5])
    normal /= np.linalg.norm(normal)
    some_vector = [1, 0, 0]  # this should not be important
    x_axis = np.cross(normal, some_vector)
    x_axis /= np.linalg.norm(normal)
    y_axis = np.cross(normal, x_axis)
    T_plane_to_world = vectors_to_transfor(x_axis, y_axis, normal, origo)

    r = 3
    angles = np.arange(90, 240, 10)
    angles_rad = np.deg2rad(angles)
    plane_points = np.array([r * np.cos(angles_rad), r * np.sin(angles_rad), np.zeros_like(angles), np.ones_like(angles)])
    world_points = np.dot(T_plane_to_world, plane_points)[:3, :].T
    return world_points


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


def load_data():
    data = np.loadtxt(r"C:\Users\ahe\Desktop\sensor_poses2.txt", delimiter=',')
    world_points = []
    for data_vector in data:
        # Invert transformation matrices
        T_marker_to_scanner = data_vector.reshape((4, 4))
        t_scanner_to_marker = np.linalg.inv(T_marker_to_scanner)
        # Take translation column as world point
        world_points.append(t_scanner_to_marker[:3, 3])
    return np.array(world_points)


def fit_data():
    # world_points = generate_data()  # Positions of calibration board corner seen from top scanner
    world_points = load_data()  # Positions of calibration board corner seen from top scanner
    centroid = world_points.mean(axis=0)
    U, S, VT = np.linalg.svd(world_points - centroid)
    # assert S[2] < 1e-8
    zp = normal = VT[2, :]
    if zp[2] > 0:  # Make sure normal points towards top scanner
        zp = - zp
    some_vector = [1, -2, 0]  # TODO chose this more wisely, this will define x, y axis
    xp = np.cross(some_vector, zp)
    xp /= np.linalg.norm(xp)
    yp = np.cross(zp, xp)

    # All vectors and origi of plane known, define transformation to/from this plane.
    T_plane_to_scanner = vectors_to_transfor(xp, yp, zp, centroid)
    t_scanner_to_plane = np.linalg.inv(T_plane_to_scanner)

    # Project world points onto plane and fit circle in this plane
    world_points_extended = np.hstack((world_points, np.ones((world_points.shape[0], 1))))
    plane_points = np.dot(t_scanner_to_plane, world_points_extended.T).T
    x_c, y_c, r = fit_circle(plane_points[:, 0], plane_points[:, 1])

    # Reporject circle center back to scanner coordinate system then define final transformation matrix
    rotatation_origo = np.dot(T_plane_to_scanner, np.array([x_c, y_c, 0, 1]))[:3]
    T_stage_to_scanner = vectors_to_transfor(xp, yp, zp, rotatation_origo)
    t_scanner_to_stage = np.linalg.inv(T_plane_to_scanner)
    return T_stage_to_scanner

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(world_points[:,0], world_points[:,1], world_points[:,2], marker='o')
    ax.quiver(*rotatation_origo, *zp, length=100, color='b')
    ax.quiver(*rotatation_origo, *xp, length=100, color='r')
    ax.quiver(*rotatation_origo, *yp, length=100, color='g')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    # ax.set_aspect('equal')
    set_axes_equal(ax)

    # Align coordinate system so axis is towards side scanner
    side_scanner_position_word = np.array([100, 20, 5])
    side_scanner_position_stage = np.dot(t_scanner_to_stage, np.append(side_scanner_position_word, 1))
    xp_better = side_scanner_position_stage[0] * xp + side_scanner_position_stage[1] * yp
    xp_better /= np.linalg.norm(xp_better)
    yp_better = np.cross(zp, xp_better)
    T_stage_to_scanner = vectors_to_transfor(xp_better, yp_better, zp, rotatation_origo)
    return T_stage_to_scanner


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


if __name__ == "__main__":
    T = fit_data()
    print(T)
    # np.savetxt(r"C:\Users\ahe\Desktop\T_stage_to_scanner.txt", T, delimiter=",")
    np.save("T_stage_to_scanner.txt", T)

import numpy as np
import open3d as o3d

from open3d_helper import perp, cloud_to_nparray, nparray_to_cloud, visualize_cloud, icp_multiple, vectors_to_transfor, \
    get_rotation_matrix, normalize, get_normals, plane_fitting, cylinder_fitting, draw_registration_result


def rot90(normal, n=1):
    if n == 0:
        return [normal[0], normal[1], normal[2]]
    new_normal = [-normal[1], normal[0], normal[2]]
    if n > 1:
        return rot90(new_normal, n-1)
    return new_normal


# diameter = 240
# height = 76
#
# top, top2 = 4 * [None], 4 * [None]
# for i in range(4):
#     top[i] = o3d.io.read_point_cloud(rf"C:\Users\ahe\Desktop\par02\cloud_top_{i}.ply").voxel_down_sample(1)
#     top2[i] = o3d.io.read_point_cloud(rf"C:\Users\ahe\Desktop\par07\cloud_top_{i}.ply").voxel_down_sample(1)
#
# top_normals = 4 * [None]
# for i, cloud_t in enumerate(top):
#     top_normals[i] = plane_fitting(cloud_t, diameter, True)
# normal1 = normalize(np.array(top_normals).mean(axis=0))
#
# top_normals2 = 4 * [None]
# for i, cloud_t in enumerate(top2):
#     top_normals2[i] = plane_fitting(cloud_t, diameter, True)
# normal2 = normalize(np.array(top_normals2).mean(axis=0))
#
# _ = 'bp'
# perp(normal1, [0,0,1], diameter)
# perp(normal2, [0,0,1], diameter)
# perp(normal1, normal2, diameter)

### Analyze empty stage

diameter = 240
height = 0

top, normals = 20 * [None], 20 * [None]
for i in range(20):
    top[i] = o3d.io.read_point_cloud(rf"C:\Users\ahe\Desktop\EmptyStage\{i:03}.ply").voxel_down_sample(1)

    top[i] = cloud_to_nparray(top[i])
    good_idx = np.abs(top[i][:,2]) < 5
    top[i] = nparray_to_cloud(top[i][good_idx, :])

    normals[i] = plane_fitting(top[i], diameter, False)

for i, normal in enumerate(normals):
    normals[i] = rot90(normal, i)

normals = np.array(normals)
average_normal = normalize(normals.mean(axis=0))
print(perp(average_normal, [0,0,1], 250))
angle_normals = []
for i in range(4):
    angle_normals.append(normalize(normals[i::4].mean(axis=0)))
    print(f"{i}, {90*i:3}: {perp(angle_normals[-1], average_normal)}")

### Analyze cylinder with flat hat

diameter = 180
height = 86

orig, flip = 4 * [None], 4 * [None]
for i in range(4):
    orig[i] = o3d.io.read_point_cloud(rf"C:\Users\ahe\Desktop\CylHat\High{i}.ply").voxel_down_sample(1)
    flip[i] = o3d.io.read_point_cloud(rf"C:\Users\ahe\Desktop\CylHat\Low{i}.ply").voxel_down_sample(1)

clouds, normals = [], []
for cloud in orig + flip:
    np_cloud = cloud_to_nparray(cloud)
    good_idx = (np.abs(np_cloud[:, 2] - height) < 5) & (np_cloud[:,0]**2 + np_cloud[:,1]**2 < (diameter/2)**2)
    cloud = np_cloud[good_idx, :]
    normals.append(plane_fitting(nparray_to_cloud(cloud), diameter, True))
    # visualize_cloud(cloud)

for i, normal in enumerate(normals):
    normals[i] = rot90(normal, i)

normals = np.array(normals)
orig_normal = normalize(normals[:4].mean(axis=0))
print(perp(orig_normal, [0,0,1], 250))
flip_normal = normalize(normals[4:].mean(axis=0))
print(perp(flip_normal, [0,0,1], 250))

# visualize_cloud(orig)
# visualize_cloud(flip)

_ = 'bp'

diameter = 180  # 240
height = 86

top_normals = 4 * [None]
for i in range(4):
    top = o3d.io.read_point_cloud(rf"C:\Users\ahe\Desktop\CylHat\cloud_top_{i}.ply").voxel_down_sample(1)
    np_cloud = cloud_to_nparray(top)
    good_idx = (np.abs(np_cloud[:, 2] - height) < 5) & (np_cloud[:, 0] ** 2 + np_cloud[:, 1] ** 2 < (diameter / 2) ** 2)
    cloud = np_cloud[good_idx, :]
    top_normals[i] = plane_fitting(nparray_to_cloud(cloud), diameter, True)

top_normals = np.array(top_normals)
top_normal = normalize(top_normals.mean(axis=0))
print(perp(top_normal, [0,0,1], 250))

## NORMAL ALGORITHM - EXCEPT NO CORRECTION

diameter = 177
height = 330

top_normals = 4 * [None]
for i in range(4):
    top = o3d.io.read_point_cloud(rf"C:\Projects\Umicore_SW\inspection_sw\out\build\RelWithDebInfo\cloud_top_0.ply").voxel_down_sample(1)
    top_normals[i] = plane_fitting(top, diameter, plot=False)
top_normal = normalize(np.array(top_normals).mean(axis=0))
print(f"Parallelism: {perp(top_normal, [0,0,1], diameter)}")


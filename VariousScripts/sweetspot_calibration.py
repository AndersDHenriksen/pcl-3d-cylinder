import numpy as np

T_TopToScanner = np.array([0.898734, 0.0882083, 0.42953, -354.808,
                           0.121389, -0.991324, -0.0504118, 171.936,
                           0.421357, 0.097447, -0.901644, 538.684,
                           0, 0, 0, 1]).reshape(4,4)

T_SideToScanner = np.array([-0.902649, 0.0456862, -0.427945, 435.879,
                            0.00839961, 0.99603, 0.0886163, 24.1736,
                            0.430295, 0.0763949, -0.89945, 638.535,
                            0, 0, 0, 1]).reshape(4,4)

# T_SideToTop_old = np.array([[-6.28224e-01, 1.90450e-01, -7.54363e-01, 7.39663e+02],
#                         [-4.24434e-02, -9.76523e-01, -2.11191e-01, 2.23462e+02],
#                         [-7.76874e-01, -1.00658e-01, 6.21558e-01, 2.56588e+02],
#                         [0.00000e+00, 0.00000e+00, 0.00000e+00, 1.00000e+00]])  # This was basis for x,y orientation only

T_SideToTop_old = np.array(
       [[-6.30770954e-01,  1.89125504e-01, -7.52568654e-01,  7.36956288e+02],
        [-4.23032590e-02, -9.76782421e-01, -2.10015167e-01,  2.22558423e+02],
        [-7.74814948e-01, -1.00635365e-01,  6.24126660e-01,  2.54884297e+02],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

#T_TopToSide_old =
#[[-6.28223754e-01 -4.24430508e-02 -7.76874156e-01  6.73494862e+02]
# [ 1.90449834e-01 -9.76522667e-01 -1.00657525e-01  1.03174526e+02]
# [-7.54363071e-01 -2.11191427e-01  6.21558538e-01  4.45683249e+02]
# [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]


T_SideToTop_new = np.dot(np.linalg.inv(T_TopToScanner), T_SideToScanner)
print(T_SideToTop_new)

# To convert measurement series to new data transform top data by using:
T_TopOldToTopNew = np.dot(T_SideToTop_new, np.linalg.inv(T_SideToTop_old))
print(T_TopOldToTopNew)

R = T_TopOldToTopNew[:3, :3]
perp = lambda ndn : 258 * np.sqrt(1 - ndn ** 2) / np.abs(ndn)

# # Yellow 5 - Worst perpendicularity
# n = np.array([0.00229116972, -0.0010739945, 0.999996781])
# a = np.array([0.00108886266, -1.5977457e-5, 0.999999404])
# before = np.dot(n, a)
# after = np.dot(np.dot(R,n), a)
# print(f"Yellow 5. Before: {perp(before)}. After {perp(after)}")
#
# # Yellow 2 - Best perpendicularity
#
# n = np.array([-0.000561124296, -0.00114244339, 0.999999225])
# a = np.array([0.000676405791, 0.00065730210, -0.999999583])
# before = np.dot(n, a)
# after = np.dot(np.dot(R,n), a)
# print(f"Yellow 2. Before: {perp(before)}. After {perp(after)}")

a, n = 8 * [None], 8 * [None]

# Yellow 0
a[0] = np.array([0.000586045, -0.00135584, 0.999999])
n[0] = np.array([0.000864198, -0.00236637, 0.999997])

# Yellow 1
a[1] = np.array([0.000258605, 0.00105618, -0.999999])
n[1] = np.array([0.000230175, 0.00200755, -0.999998])

# Yellow 2 - Best perpendicularity
a[2] = np.array([0.000676405791, 0.00065730210, -0.999999583])
n[2] = np.array([-0.000561124296, -0.00114244339, 0.999999225])

# Yellow 3
a[3] = np.array([0.000678592, -0.000473216, -1])
n[3] = np.array([0.000361148, 0.000399501, 1])

# Yellow 4
a[4] = np.array([0.00010926, 0.000830805, 1])
n[4] = np.array([0.00155045, 0.000297879, 0.999999])

# Yellow 5 - Worst perpendicularity
a[5] = np.array([0.00108886266, -1.5977457e-5, 0.999999404])
n[5] = np.array([0.00229116972, -0.0010739945, 0.999996781])

# Yellow 6
a[6] = np.array([0.000565318, -0.00104927, 0.999999])
n[6] = np.array([0.00108369, -0.00219197, 0.999997])

# Yellow 7
a[7] = np.array([0.000644802, 0.000851719, -0.999999])
n[7] = np.array([0.000563741, 0.00151637, -0.999999])

for i, axis, normal in zip(range(8), a, n):
    before = np.dot(normal, axis)
    after = np.dot(np.dot(R, normal), axis)
    print(f"Yellow {i}. Before: {perp(before)}. After {perp(after)}")

def perp_span(R):
    after = 8 * [None]
    for i, axis, normal in zip(range(8), a, n):
        after[i] = perp(min(1, np.dot(np.dot(R, normal), axis)))
    return np.array(after).ptp() + (np.array(after).mean() - .4)/1.5

def perp_span2(x,y,z):
    from scipy.spatial.transform import Rotation as R
    r = R.from_rotvec(np.array([x, y, z])).as_matrix()
    return perp_span(r)

best = 1
for x in np.arange(-3, 3, 1):
    print(x)
    for y in np.arange(-3, 3, 1):
        # for z in np.arange(-3, 3, 0.1):
        for z in np.arange(-20, 20, .1):
            current = perp_span2(np.deg2rad(x), np.deg2rad(y), np.deg2rad(z))
            if current < best:
                best = current
                # print(f"New best {(x,y,z)} : {best}")
                x_best, y_best, z_best = x,y,z

print("done")

from scipy.optimize import minimize
out = minimize(lambda x: perp_span2(x[0], x[1], x[2]), np.array([np.deg2rad(x_best), np.deg2rad(y_best), np.deg2rad(z_best)]))
from scipy.spatial.transform import Rotation
R_best = Rotation.from_rotvec(out.x).as_matrix()

after = 8 * [None]
for i, axis, normal in zip(range(8), a, n):
    after[i] = perp(min(1, np.dot(np.dot(R_best, normal), axis)))

print(after)

_ = 'bp'
import numpy as np
import matplotlib.pyplot as plt

real_heights = [np.array([532.3, 532.3, 532.2]),
                np.array([227.6, 227.6, 227.4]),
                np.array([265.8, 265.7, 265.6, 265.8, 265.9, 265.9, 265.8, 265.6]),
                np.array([532.4, 532.1, 532.4, 532.3, 532.5, 532.6, 532.8, 532.4, 532.3]),
                np.array([62.4, 66.5, 67.6, 62.7, 62.2]),
                np.array([64.7, 64.2, 63]),
                np.array([84.7, 84.7, 84.8, 84.9, 84.7]),
                np.array([423.8, 423.9, 424.3, 424.2, 424.3, 424, 423.5])]

# mean_heights = np.array([h.mean() for h in real_heights])
mean_heights = np.array([h.min() + h.ptp()/2 for h in real_heights])

diameters = [241, 305, 268, 267, 330, 190, 240, 242]

scan_heights = [534.789, 228.699, 267.16, 534.953, 64.6, 63.6363, 85.0134, 426.244]

offset = scan_heights - mean_heights

for i, letter in enumerate('ABCDEFGH'):
    print(f"{letter}: {mean_heights[i]:6.2f}, {scan_heights[i]:6.2f}, {offset[i]:.2f}")

plt.plot(mean_heights, offset, 'xr')
plt.show()

# Try to fit polynomial
p_coef = np.polyfit(mean_heights, offset, 2)
plt.plot(mean_heights, offset - np.polyval(p_coef, mean_heights), 'xr')
plt.show()

_ = 'bp'

### ATTEMPT 2

real_heights = np.array([532.3, 227.8, 265.75, 400.1, 63.4, 131, 84.8, 271])
scan_heights = np.array([535.89, 229.82, 267.93, 402.89, 64.3, 132.41, 85.84, 273.23])
scan_heights = np.array([535.97, 230.03, 268.35, 403.05, 64.92, 132.89, 86.41, 273.641])

offset = scan_heights - real_heights
plt.plot(real_heights, offset, 'xr')
plt.show()

p_coef = np.polyfit(real_heights, offset, 2)
plt.plot(real_heights, offset - np.polyval(p_coef, real_heights), 'xr')
plt.show()

### ATTEMPT - DIAMETER

real_diam = np.array([258.1, 317.15, 240.6, 304.8, 190.4, 330])
scan_diam = np.array([258.53, 317.814, 241.294, 305.486, 192.5, 332.233])  #icp results
# scan_diam = np.array([258.485, 317.2, 240.128, , , 332.865])  # no icp result

real_diam = np.array([240.6, 304.8, 190.4, 330])
scan_diam = np.array([241.294, 305.486, 192.5, 332.233])

offset = scan_diam - real_diam
plt.plot(real_diam, offset, 'xr')
plt.show()

p_coef = np.polyfit(real_diam, offset, 2)
plt.plot(real_diam, offset - np.polyval(p_coef, real_diam), 'xr')
plt.show()

print(p_coef)

### ATTEMPT - HEIGHT

current_pcoef = np.array([-1.40266196e-06,  5.30839375e-03,  1.19468097e+00])
scan_heights = np.array([227.11, 129.3, 531.86, 63.08, 265.5, 265.41, 84.55])
real_heights = np.array([227.9, 130.2, 532.5, 64.1, 266.15, 266.15, 85.5])
offset = scan_heights - real_heights
p_coef = np.polyfit(real_heights, offset, 2)
print(p_coef)
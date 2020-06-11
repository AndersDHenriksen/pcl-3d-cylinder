import numpy as np
import cv2
import Vision.VisionTools as vt
try:
    from skimage import io
except ImportError:
    print("Scikit image not available")

# Convert tiff to npy
# conda activate playground
# import numpy as np
# from skimage import io
# img = io.imread(r"data.tiff")
# np.save(r"img.npy", img)

pixelsize_in_mm = 1

radial_map_float = io.imread("radius_image.tiff")

# Fix holes
radial_map_holes = radial_map_float == 0
radial_map_closed = vt.morph("close", radial_map_float, (3, 3))
radial_map_float[radial_map_holes] = radial_map_closed[radial_map_holes]
for hole_mask in vt.cc_masks(radial_map_float == 0)[0]:
    if hole_mask.sum() > 100:
        print("Large hole in radial map. Something is wrong")
    perimeter = vt.morph("dilate", hole_mask, (3, 3)) > hole_mask
    radial_map_float[hole_mask] = radial_map_float[perimeter].mean()

# Create uint8 image
radial_lower = radial_map_float.min()
radial_range = radial_map_float.ptp()
float_to_uint8 = lambda i: np.uint8(np.round(255 * np.clip(i - radial_lower, a_min=0, a_max=radial_range) / radial_range))
uint8_to_float = lambda i: i / 255 * radial_range + radial_lower
radial_map = float_to_uint8(radial_map_float)

intensity_per_mm = 256 / radial_range

# Inpaint alternative
# radial_map = cv2.inpaint(radial_map_raw, (radial_map_raw == 0).astype(np.uint8), 3, cv2.INPAINT_TELEA)

# Calculate volume
real_radial_map = uint8_to_float(radial_map) #intensity_to_radius(radial_map)
cylinder_volume = np.mean(real_radial_map ** 2) * np.pi * radial_map.shape[0]
print(f"Cylinder volume: {cylinder_volume:.0f} mm^3")

# Calculate Wrinkles
indent_mask = vt.morph("blackhat", radial_map, (15, 1)) > intensity_per_mm / 2
indent_mask = vt.morph("open", indent_mask, (1, 3))
# indent_mask = vt.bw_area_filter(indent_mask, n=500, area_range=(5, 1e6), output="mask")
indents_closed = vt.morph("close", indent_mask, (3, 15))
wrinkles_mask = vt.bw_area_filter(indents_closed, n=50, area_range=(20, 1e6), output="mask")
wrinkles_mask = vt.morph("close", wrinkles_mask, (15, 1))

# Handle wrinkles at wrap point
if np.any(wrinkles_mask[:, 0] * wrinkles_mask[:, -1]):
    warp_marker = 0 * wrinkles_mask
    warp_marker[:, 0] = True
    wrinkles_mask_part_2 = vt.bw_reconstruct(warp_marker, wrinkles_mask)
    wrinkles_mask_part_1 = wrinkles_mask.copy()
    wrinkles_mask_part_1[wrinkles_mask_part_2] = False
    wrinkles_mask2 = np.hstack((wrinkles_mask_part_1, wrinkles_mask_part_2))
    radial_map2 = np.hstack((radial_map, radial_map))
else:
    wrinkles_mask2 = wrinkles_mask
    radial_map2 = radial_map

for wrinkle_mask in vt.cc_masks(wrinkles_mask2)[0]:
    perimeter_mask = vt.morph("dilate", wrinkle_mask, (11, 11)) ^ wrinkle_mask
    wrinkle_radius = radial_map2[wrinkle_mask].mean()
    perimeter_radius = radial_map2[perimeter_mask].mean()
    wrinkle_depth = perimeter_radius - wrinkle_radius
    wrinkle_mask_reduced = vt.bw_remove_empty_lines(wrinkle_mask)
    if wrinkle_depth < intensity_per_mm / 2 or wrinkle_mask_reduced.shape[0] < 40 / pixelsize_in_mm:
        wrinkles_mask2[wrinkles_mask] = 0
        continue
    wrinkle_length = max(wrinkle_mask_reduced.shape)
    wrinkle_width = wrinkle_mask_reduced.sum() / wrinkle_length
    wringle_length_width_depth_mm = (wrinkle_length * pixelsize_in_mm, wrinkle_width * pixelsize_in_mm, wrinkle_depth / intensity_per_mm)
    print(f"Wrinkle (length, width, depth): {wrinkle_length:.0f}, {wrinkle_width:.0f}, {wrinkle_depth:.0f} px")
    # vt.showimg(radial_map2, wrinkle_mask)
    # _ = 'bp'

# Calculate diameters
diameter_A = np.median(real_radial_map[0, :])
diameter_B = np.median(real_radial_map[-1, :])
diameters_X = 7 * [None]
for i, h in enumerate(np.arange(100, min(701, real_radial_map.shape[0]), 100)):
    diameters_X[i] = np.median(real_radial_map[h, :])

# Calculate cylindricity
radial_vector = real_radial_map[~wrinkles_mask]
cylindricity = np.percentile(radial_vector, 95) - np.percentile(radial_vector, 5)  # Would be faster with np.bincount
print(f"Cylindricity: {cylindricity:.1f} mm")

if wrinkles_mask2.shape[1] > wrinkles_mask.shape[1]:
    wrinkles_mask = wrinkles_mask2[:, :wrinkles_mask.shape[1]] | wrinkles_mask2[:, wrinkles_mask.shape[1]:]
else:
    wrinkles_mask = wrinkles_mask2

vt.showimg(radial_map, overlay_mask=wrinkles_mask)
_ = 'bp'
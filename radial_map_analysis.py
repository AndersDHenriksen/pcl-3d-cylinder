import numpy as np
import cv2
import Vision.VisionTools as vt

radial_map_file_path = r"C:\Projects\Umicore_CylinderAnalysis\radius_image_r_119_pp_4.bmp"
pixelsize_in_mm = 1
radial_mean = 119
radial_range = 4

intensity_to_radius = lambda i: radial_mean + radial_range * (i/255 - .5)
intensity_per_mm = 256 / radial_range
radial_map = cv2.imread(radial_map_file_path)[:, :, 0]

# Fix holes
for hole_mask in vt.cc_masks(radial_map == 0)[0]:
    if hole_mask.sum() > 100:
        print("Large hole in radial map. Something is wrong")
    perimeter = vt.bw_edge(vt.morph("dilate", hole_mask, (3, 3)))
    radial_map[hole_mask] = radial_map[perimeter].mean()

# Inpaint alternative
# radial_map = cv2.inpaint(radial_map_raw, (radial_map_raw == 0).astype(np.uint8), 3, cv2.INPAINT_TELEA)

# Calculate volume
real_radial_map = intensity_to_radius(radial_map)
cylinder_volume = np.mean(real_radial_map ** 2) * np.pi * radial_map.shape[0]

# Calculate Wrinkles
indent_mask = vt.morph("blackhat", radial_map, (15, 1)) > intensity_per_mm / 4
# indent_mask = vt.bw_area_filter(indent_mask, n=500, area_range=(5, 1e6), output="mask")
indents_closed = vt.morph("close", indent_mask, (3, 15))
wrinkles_mask = vt.bw_area_filter(indents_closed, n=50, area_range=(20, 1e6), output="mask")
wrinkles_mask = vt.morph("close", wrinkles_mask, (15, 1))

# Handle wrinkles at wrap point
if np.any(wrinkles_mask[:, 0] * wrinkles_mask[:, -1]):
    warp_marker = 0 * wrinkles_mask
    warp_marker[:, 0] = True
    wrinkles_mask_reduced = vt.bw_reconstruct(warp_marker, wrinkles_mask)
    wrinkles_mask[wrinkles_mask_reduced] = False
    wrinkles_mask2 = np.hstack((wrinkles_mask, wrinkles_mask_reduced))
    radial_map2 = np.hstack((radial_map, radial_map))
else:
    wrinkles_mask2 = wrinkles_mask
    radial_map2 = radial_map

for wrinkle_mask in vt.cc_masks(wrinkles_mask2)[0]:
    perimeter_mask = vt.morph("dilate", wrinkle_mask, (11, 11)) ^ wrinkle_mask
    wrinkle_radius = radial_map2[wrinkle_mask].mean()
    perimeter_radius = radial_map2[perimeter_mask].mean()
    wrinkle_depth = perimeter_radius - wrinkle_radius
    if wrinkle_depth < 0:
        continue
    wrinkle_mask_reduced = vt.bw_remove_empty_lines(wrinkle_mask)
    wrinkle_length = max(wrinkle_mask_reduced.shape)
    wrinkle_width = wrinkle_mask_reduced.sum() / wrinkle_length
    wringle_length_width_depth_mm = (wrinkle_length * pixelsize_in_mm, wrinkle_width * pixelsize_in_mm, wrinkle_depth / intensity_per_mm)
    print(f"wrinkle (length, width, depth): {wrinkle_length:.0f}, {wrinkle_width:.0f}, {wrinkle_depth:.0f} px")
    # vt.showimg(radial_map2, wrinkle_mask)
    # _ = 'bp'

# Calculate cylindricity
radial_vector = real_radial_map[~wrinkles_mask]
cylindricity = np.percentile(radial_vector, 95) - np.percentile(radial_vector, 5)


vt.showimg(radial_map, overlay_mask=wrinkles_mask)

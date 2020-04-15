import numpy as np
import cv2
import Vision.VisionTools as vt

radial_map_file_path = r"C:\Projects\Umicore_CylinderAnalysis\radius_image_r_119_pp_4.bmp"
radial_mean = 119
radial_range = 4


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
real_radial_map = radial_mean + radial_range * (radial_map/255 - .5)
cylinder_volume = np.mean(real_radial_map ** 2) * np.pi * radial_map.shape[0]

# Wrinkles
indent_mask = vt.morph("blackhat", radial_map, (15, 1)) > intensity_per_mm / 4
# indent_mask = vt.bw_area_filter(indent_mask, n=500, area_range=(5, 1e6), output="mask")
indents_closed = vt.morph("close", indent_mask, (3, 15))
wringles_mask = vt.bw_area_filter(indents_closed, n=50, area_range=(20, 1e6), output="mask")

for wingle_mask in vt.cc_masks(wringles_mask)[0]:



vt.showimg(radial_map, overlay_mask=wringles_mask)

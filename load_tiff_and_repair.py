import numpy as np
import cv2
import Vision.VisionTools as vt
from skimage import io

radial_map_float = io.imread(r"C:\Projects\Umicore_SW_Old1\out\build\x64-Release\radius_image.tiff")
radial_map_holes = radial_map_float == 0
radial_map_closed = vt.morph("close", radial_map_float, (3, 3))
radial_map_float[radial_map_holes] = radial_map_closed[radial_map_holes]
for hole_mask in vt.cc_masks(radial_map_float == 0)[0]:
    if hole_mask.sum() > 100:
        print("Large hole in radial map. Something is wrong")
    perimeter = vt.morph("dilate", hole_mask, (3, 3)) > hole_mask
    radial_map_float[hole_mask] = radial_map_float[perimeter].mean()

_ = 'bp'

print(radial_map_float[:100,:].mean())

print(radial_map_float[-100:,:].mean())
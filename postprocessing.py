# -*- coding: utf-8 -*-
"""
postprocess segmentation results by

1) separating touching nuclei
 (watershed on distance transform)
 
2) filtering segmented components by size

3) filling holes

"""

import numpy as np

from scipy.ndimage import measurements 
from scipy.ndimage import distance_transform_edt

from skimage.segmentation import watershed
from skimage import feature

from scipy.ndimage.morphology import binary_fill_holes


def separate_touching_nuclei(mask):
    # minimum distance between two watershed seed
    region_width = 15

    distance = distance_transform_edt(mask)
    coords = feature.peak_local_max(distance, min_distance=region_width, footprint=np.ones((3, 3)))
    seed_mask = np.zeros(distance.shape, dtype=bool)
    seed_mask[tuple(coords.T)] = True
    seeds, _ = measurements.label(seed_mask)
    watershed_result = watershed(-distance, seeds, mask=mask, watershed_line=True)
    segmentation_result = mask.copy()
    segmentation_result[watershed_result==0] = 0

    return segmentation_result

def filter_by_size(mask, min_size=0, max_size=np.inf):
    filtered_mask = np.zeros_like(mask)
    components, n_components = measurements.label(mask)
    for c_idx in range(1,n_components+1):    
        size = np.count_nonzero(components==c_idx)
        if size > min_size and size < max_size:
            filtered_mask[components == c_idx] = mask[components == c_idx]
    
    return filtered_mask
        

def fill_holes(mask):
    filled_mask = binary_fill_holes(mask).astype(np.uint8)
    return filled_mask


        
        
        
        
        
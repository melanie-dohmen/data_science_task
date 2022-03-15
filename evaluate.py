# -*- coding: utf-8 -*-
"""
Contains code to evaluate segmentations
"""
import numpy as np
#import os

from skimage import measure
from scipy.ndimage import measurements 



def new_evaluation_dictionary():
    eval_dict = {
        "Accuracy": [],
        "AP 50": [],
        "mAP": [],
        "gt counts": [],
        "pred counts": [],
        "IoU": [],
    }
    return eval_dict

def append_evaluation(prediction, ground_truth, eval_dict):
    eval_dict["Accuracy"].append(np.mean(prediction==ground_truth))
    eval_dict["AP 50"].append(average_precision(prediction, ground_truth, overlap=0.5))
    eval_dict["mAP"].append(mean_average_precision(prediction, ground_truth))
    eval_dict["gt counts"].append(count_instances(ground_truth))
    eval_dict["pred counts"].append(count_instances(prediction))
    eval_dict["IoU"].append(IoU(prediction, ground_truth))
    return eval_dict

def extend_evaluation(eval_dict1, eval_dict2):
    result_dict = eval_dict1.copy()
    for key in eval_dict1.keys():
        result_dict[key].extend(eval_dict2[key])
    return result_dict

def insert_space_in_evaluation(eval_dict):
    result_dict = eval_dict.copy()
    for key in eval_dict.keys():
        result_dict[key].append(np.nan)
    return result_dict

def count_instances(mask):
    
    _,n = measurements.label(mask)

    return n

def component_sizes(mask, sort=False):
    components, n = measurements.label(mask)
    
    component_sizes = []
    for c_idx in range(1,n):    
        size = np.count_nonzero(components==c_idx)
        component_sizes.append(size)
    
    if sort:
        component_sizes = sorted(component_sizes)
        
    return np.array(component_sizes)


def average_precision(pred_mask, gt_mask, overlap=0.5):
   
    pred_instances = measure.label(pred_mask)
    gt_instances = measure.label(gt_mask)
    
    TP = 0
    FN = 0
    FP = 0    
    
    matched_pred_labels = []
    # calculate TP and FN (for recall)
    for gt_label in np.unique(gt_instances):

        if gt_label != 0: # assume 0 is background
            pred_region = pred_instances[(gt_instances == gt_label)]
        
            matched = False
            for pred_label in np.unique(pred_region):  
                if pred_label != 0:
                    #calculate IoU
                    intersection = (gt_instances == gt_label) & (pred_instances == pred_label)        
                    pixels_intersection = np.count_nonzero(intersection)  
                    
                    union = (gt_instances == gt_label) | (pred_instances == pred_label)
                    pixels_union = np.count_nonzero(union)  
                    
                    if pixels_union != 0:
                        #print("IoU: ", (pixels_intersection / pixels_union) )
                        if (pixels_intersection / pixels_union) > overlap:
                            TP +=  1
                            matched = True
                            matched_pred_labels.append(pred_label)
                            break
            if not matched:
                FN += 1 
                
    for pred_label in np.unique(pred_instances):
        if pred_label != 0 and  pred_label not in matched_pred_labels:
            FP += 1
    
    #print("TP: ", TP, " FP: ", FP, " FN: ", FN)
    average_precision = TP / (TP + 0.5*FN + 0.5*FP)

    return average_precision

    
def mean_average_precision(prediction, ground_truth):
        
    ap = np.zeros((10))
    thresholds = [ 0.5 + i*0.05 for i in range(0,10)]
    for i, overlap in enumerate(thresholds):
        ap[i] = average_precision(prediction, ground_truth, overlap=overlap)
        #print("AP for threshold ", overlap, " is ", ap[i])
    return np.mean(ap)
        
            
def IoU(prediction, ground_truth):
    
    intersection = (prediction > 0) & (ground_truth > 0)        
    pixels_intersection = np.count_nonzero(intersection)  
    
    union = (prediction > 0) | (ground_truth > 0)   
    pixels_union = np.count_nonzero(union)  
    
    if pixels_union != 0:
        return (pixels_intersection / pixels_union)
    else:
        return 0
 
        
        
        
        
        

if __name__=="__main__":
    
    from data_utils import load_data
    import os
    from postprocessing import separate_touching_nuclei

    # load data
    tissue_data = load_data(os.path.join("data", "tissue_images"))
    mask_data = load_data(os.path.join("data", "mask binary"))
    
    for mask_idx in range(mask_data.shape[0]):
        
        mask = mask_data[mask_idx]
        refined_mask = separate_touching_nuclei(mask)
        print("average_precision: ", average_precision(mask, refined_mask))


# -*- coding: utf-8 -*-
"""
 cross validate random forest classifier with different parameters
 
 keep test set fixed and create different splits of train and val data
 
"""
import os

from rf_pixel_classifier import train
from evaluate import new_evaluation_dictionary, extend_evaluation

from data_utils import get_split_indices, get_filename_prefixes

import pickle



params = {
    "experiment_name": "rfc_onlyRGB_test0",
    "sampling_factor": 0.01,
    "data_path": "data",
    "test_split_idx": 0,
    "validation_split_idx": 1,
    "seed": 42,
    "norm_stain": True,
    "include_hessian_1": False,
    "include_DoG_1": False,
    "include_hessian_2": False,
    "include_DoG_2": False,
    
    "separate_touching_nuclei": True,
    "fill_holes": True,
    "filter_min_size": 10,
    }

eval_dict = new_evaluation_dictionary()
eval_dict_pp = new_evaluation_dictionary()

for i in range(1,10):
    params["validation_split_idx"] = i
    print("running cross validation with validation set nr ", i)
    print("image idxs: ", get_split_indices(get_filename_prefixes("data//mask binary", split="all"), split="validation_"+str(i)))
    eval_dict_i, eval_dict_pp_i = train(params)
    eval_dict = extend_evaluation(eval_dict, eval_dict_i)
    eval_dict_pp = extend_evaluation(eval_dict_pp, eval_dict_pp_i)
    
filename = os.path.join("results_rf",params["experiment_name"],"cross_eval.dict")
pickle.dump(eval_dict, open(filename, 'wb'))

filename = os.path.join("results_rf",params["experiment_name"],"cross_eval_pp.dict")
pickle.dump(eval_dict_pp, open(filename, 'wb'))


    
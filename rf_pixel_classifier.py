# -*- coding: utf-8 -*-
"""
Random Forest to classify pixels
"""
import os

from data_utils import load_data, add_hessian, add_DoG, get_filename_prefixes
import numpy as np

import pickle

from sklearn.ensemble import RandomForestClassifier

from datetime import datetime

# export predicted images
from PIL import Image

from norm_stain import normalizeData
from postprocessing import separate_touching_nuclei, filter_by_size, fill_holes

from evaluate import new_evaluation_dictionary, append_evaluation, insert_space_in_evaluation

default_params = {
    # name of savend model and folder with results
    "experiment_name": "TEST",
    
    # randomly sample a subset of annotated pixels
    "sampling_factor": 0.01,
    
    # path were to find "mask binary" and "tissue_images" folders    
    "data_path": "data",
    
    # tissue type index, that is excluded from training
    "test_split_idx": 0,
    # tissue type index, that is excluded from training
    # can be varied for cross-validation
    "validation_split_idx": 1,
    
    # make random decisions reproducible
    "seed": 42,
    
    # normalize stain in tissue images
    "norm_stain": True, 
    
    # add further features to detect
    # lines and edges on different scales:
    # Hessian Matrix with sigma=1 or 2
    # Difference of Gaussians with low_sigma=1 or 2
    # high_sigma = 1.6*low_sigma
    "include_hessian_1": True,
    "include_DoG_1": True,
    "include_hessian_2": True,
    "include_DoG_2": True,
    
    # post-processing
    "separate_touching_nuclei": True,
    "fill_holes": True,
    "filter_min_size": 10,
    }

def train_and_test(params):

    np.random.seed(params["seed"])
    
    # create "results_rf" directory to save all plots
    os.makedirs("results_rf", exist_ok=True)
    
    train_split = "train_"+str(params["validation_split_idx"]) + "_" + str(params["test_split_idx"])
    
    outdir = os.path.join("results_rf", params["experiment_name"],"predictions_"+train_split)

    os.makedirs(outdir, exist_ok=True)
    
   
    
    # load data
    tissue_train_data = load_data(os.path.join("data", "tissue_images"), train_split)
    mask_train_data = load_data(os.path.join("data", "mask binary"), train_split)    
     
       
    n_train,w,h,c = tissue_train_data.shape
    feature_dims = c
    
    if params["norm_stain"]:   
        tissue_train_data = normalizeData(tissue_train_data, method="Macenko")

    
    if params["include_hessian_1"]:    
        tissue_train_data = add_hessian(tissue_train_data, sigma=1)
        feature_dims = feature_dims + c*3
    
    if params["include_hessian_2"]:
        tissue_train_data = add_hessian(tissue_train_data, sigma=2)
        feature_dims = feature_dims + c*3
    
    if params["include_DoG_1"]:
        tissue_train_data = add_DoG(tissue_train_data, sigma=1)
        feature_dims = feature_dims + 3
        
    if params["include_DoG_2"]:
        tissue_train_data = add_DoG(tissue_train_data, sigma=2)
        feature_dims = feature_dims + 3
    

    # reshape (n,h,w,feature_dims) for learning:
    full_X = tissue_train_data.reshape(-1,feature_dims)
    full_y = mask_train_data.ravel()/255
    
    
    random_sampling = np.random.rand(n_train*w*h)
     
    X = full_X[random_sampling < params["sampling_factor"],:]    
    y = full_y[random_sampling < params["sampling_factor"]]
    
    # load or train model
    model_filename = os.path.join(outdir, "fitted_rfc.sav")
    try:
        # load the model from disk
        rfc = pickle.load(open(model_filename, 'rb'))
        print("loaded model from disk")
    except:
        rfc = RandomForestClassifier(n_estimators=100, n_jobs=4, random_state=params["seed"])
        
        starttime = datetime.now()
        print("starting to fit random forest classifier....")
        rfc.fit(X, y)
        endtime = datetime.now()
        print("done after....", endtime-starttime)
        
        # save the model to disk
        pickle.dump(rfc, open(model_filename, 'wb'))
    
    
    val_split = "validation_"+str(params["validation_split_idx"])
    tissue_val_data = load_data(os.path.join("data", "tissue_images"), val_split)
    mask_val_data = load_data(os.path.join("data", "mask binary"), val_split)

    test_split = "test_"+str(params["test_split_idx"])
    tissue_test_data = load_data(os.path.join("data", "tissue_images"), test_split)
    mask_test_data = load_data(os.path.join("data", "mask binary"), test_split)

    n_val,_,_,_ = tissue_val_data.shape
    n_test,_,_,_ = tissue_test_data.shape
    #feature_dims = c
    
    val_filename_prefixes = get_filename_prefixes(os.path.join("data", "tissue_images"), split=val_split)
    test_filename_prefixes = get_filename_prefixes(os.path.join("data", "tissue_images"), split=test_split)
    train_filename_prefixes = get_filename_prefixes(os.path.join("data", "tissue_images"), split=train_split)
    
    if params["norm_stain"]:   
        tissue_val_data = normalizeData(tissue_val_data, method="Macenko")
        tissue_test_data = normalizeData(tissue_test_data, method="Macenko")
    
    if params["include_hessian_1"]:    
        tissue_val_data = add_hessian(tissue_val_data, sigma=1)
        tissue_test_data = add_hessian(tissue_test_data, sigma=1)
        #feature_dims = feature_dims + c*3
    
    if params["include_hessian_2"]:
        tissue_val_data = add_hessian(tissue_val_data, sigma=2)
        tissue_test_data = add_hessian(tissue_test_data, sigma=2)
        #feature_dims = feature_dims + c*3
    
    if params["include_DoG_1"]:
        tissue_val_data = add_DoG(tissue_val_data, sigma=1)
        tissue_test_data = add_DoG(tissue_test_data, sigma=1)
        #feature_dims = feature_dims + 3
        
    if params["include_DoG_2"]:
        tissue_val_data = add_DoG(tissue_val_data, sigma=2)
        tissue_test_data = add_DoG(tissue_test_data, sigma=2)
        #feature_dims = feature_dims + 3
     
    X_val = tissue_val_data.reshape(-1,feature_dims)
    X_test = tissue_test_data.reshape(-1,feature_dims)
         
    X_predict_val = rfc.predict(X_val)    
    X_predict_test = rfc.predict(X_test)    
    X_predict_train = rfc.predict(full_X)    
    
    predicted_val_data = X_predict_val.reshape(tissue_val_data.shape[0],w,h)
    predicted_test_data = X_predict_test.reshape(tissue_test_data.shape[0],w,h)
    predicted_train_data = X_predict_train.reshape(tissue_train_data.shape[0],w,h)
      
    eval_dict = new_evaluation_dictionary()
    eval_dict_pp = new_evaluation_dictionary()  
    
    predicted_mask_data = {"train": predicted_train_data,
                      "validation": predicted_val_data,
                      "test": predicted_test_data}
    
    gt_mask_data = {"train": mask_train_data,
                    "validation": mask_val_data,
                      "test": mask_test_data}
    
    filename_prefixes = {"train": train_filename_prefixes,
                  "validation": val_filename_prefixes,
                  "test": test_filename_prefixes}

    
    for subset in ["train", "validation", "test"]:
        print("predicting ", subset, " set...")

        for image_nr in range(predicted_mask_data[subset].shape[0]):
                       
            gt_mask = gt_mask_data[subset][image_nr]
            prediction = predicted_mask_data[subset][image_nr]
                        
            # make binary:
            prediction = (prediction>0.5).astype(np.uint8)
            
            # save prediction images
            pred_PIL = Image.fromarray(prediction)
            pred_PIL.save(os.path.join(outdir, filename_prefixes[subset][image_nr]+".png"))

            
            # calculate metrics before post-processing
            append_evaluation(prediction, gt_mask, eval_dict)
            
            # post-processing:
            if params["fill_holes"]:
                prediction = fill_holes(prediction)
            prediction = filter_by_size(prediction, min_size=params["filter_min_size"])
            if params["separate_touching_nuclei"]:
                prediction = separate_touching_nuclei(prediction)
                gt_mask = separate_touching_nuclei(gt_mask)
                
            # save prediction images after post-processing
            pred_PIL = Image.fromarray(prediction)
            pred_PIL.save(os.path.join(outdir, "pp_"+filename_prefixes[subset][image_nr]+".png"))

            
            # calculate metrics after post-processing:
            append_evaluation(prediction, gt_mask, eval_dict_pp)
            
        # insert spaces between values of subsets
        eval_dict = insert_space_in_evaluation(eval_dict)
        eval_dict_pp = insert_space_in_evaluation(eval_dict_pp)

    
    # save evaluation metrics to file:
    filename = os.path.join(outdir,"eval.dict")
    pickle.dump(eval_dict, open(filename, 'wb'))

    filename = os.path.join(outdir,"eval_pp.dict")
    pickle.dump(eval_dict_pp, open(filename, 'wb'))
    
    return eval_dict, eval_dict_pp

if __name__=="__main__":

    eval_dict, eval_dict_pp = train_and_test(default_params)
    print("eval:", eval_dict)
    print("eval_pp:", eval_dict_pp)


# -*- coding: utf-8 -*-
"""
Generate segmentation results
 
 Choose to 
   
 1) Predict with a Fine-Tuned CNN

 2) Train a new CNN and use it for predictions

 3) Train and predict with a Random Forest Classifier


Creates figures of the evaluation metrics
 
"""

from train_NN import train
from test_NN import test
from rf_pixel_classifier import train_and_test

from plot_figures import plot_evaluation_results



CNN_train_params = {
    "experiment_name": "res-net_new",
    "epochs": 5,
    "start_epoch": 0, 
    "resume_model": "",
    "data_path": "data",
    "patch_size": 256,
    "batch_size": 4,
    "test_split_idx": 0,
    "validation_split_idx": 1,
    "seed": 42,
    "sem_loss": "BCEWithLogitsLoss",
    "norm_stain": True,
    "random_crop": True,
    "flip": True,
    "rotate90": True,
    "color_aug": False,
    }

CNN_test_params = {
  "model_name": "test_train_0_1_NoColorAug", 
  "model_path": "models",
  "checkpoint": "best",
  "data_path": "data",
  "batch_size": 4,
  "test_split_idx": 0,
  "validation_split_idx": 1,
  "seed": 42,
  "patch_size": 512,
  "norm_stain": True,
  "random_crop": False,
  "flip": False,
  "rotate90": False, 
  "color_aug": False,
  "separate_touching_nuclei": True,
  "fill_holes": True,
  "filter_min_size": 10,
}

rfc_params = {
    "experiment_name": "rfc_new",
    "sampling_factor": 0.01,
    "data_path": "data",
    "test_split_idx": 0,
    "validation_split_idx": 0,
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

if __name__=="__main__":

    '''
    test the fine-tuned CNN:
    '''
    # call test routine for CNN model
    # evaluation metrics are returned as dictionaries
    eval_dict_ftNN, eval_dict_pp_ftNN = test(CNN_test_params)
    
    # plot evaluation metrics for unprocessed predictions:
    plot_evaluation_results(eval_dict_ftNN, "Evaluation of ft-CNN model")
    
    # plot evaluation metrics after post-processing predictions:
    plot_evaluation_results(eval_dict_pp_ftNN, "Evaluation of ft-CNN model after pp")
    
    
    '''
    train and test a new CNN model:
    '''
    # # call train routine for CNN model (see parameters)
    # train(CNN_train_params)
    
    # # get model name:
    # CNN_test_params["model_name"] = CNN_train_params["experiment_name"]
    
    # # run test routine:
    # eval_dict_NN, eval_dict_pp_NN = test(CNN_test_params)
    
    # # plot results with and without post-processing:
    # plot_evaluation_results(eval_dict_NN, "Evaluation of newly trained CNN model")
    # plot_evaluation_results(eval_dict_pp_NN, "Evaluation of newly trained CNN model after pp")
    
    
    '''
    train and evaluate a random forest classifier
    '''   
    # # call train and test routine for random forest classifier 
    # eval_dict_rfc, eval_dict_pp_rfc = train_and_test(rfc_params)
 
    # # plot results with and without post-processing:
    # plot_evaluation_results(eval_dict_rfc, "Evaluation of RFC Model")
    # plot_evaluation_results(eval_dict_pp_rfc, "Evaluation of RFC Model after pp")
          
    
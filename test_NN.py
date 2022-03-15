# -*- coding: utf-8 -*-
"""
test a neural network 
- export predicted segmentation masks
- calculate evaluation metrics

"""
import os
import torch
from torch.utils.data import DataLoader
from data_utils import TissueDataset

import numpy as np


from postprocessing import separate_touching_nuclei, fill_holes, filter_by_size
from evaluate import new_evaluation_dictionary, append_evaluation,insert_space_in_evaluation

from plot_figures import plot_losses

# export predictions
from PIL import Image

# export evaluation metrics
import pickle

default_params = {
    
    
    # <checkpoint>_<model_name>.pytorch will be loaded from folder <models>
    "model_name": "test_train_0_1_NoColorAug",  
    "model_path": "models",
    "checkpoint": "last",
    # path were to find "mask binary" and "tissue_images" folders
    "data_path": "data",
    
    "batch_size": 1,
    
    # tissue type index, that is excluded from training
    "test_split_idx": 0,
    # tissue type index, that is excluded from training,
    # but for which the loss is tracked to select best model
    "validation_split_idx": 1,
    
    # make random decisions reproducible
    "seed": 42,

    # images will be cropped to sqared patches
    "patch_size": 512,
    
    # normalize stain in tissue images
    "norm_stain": True,
    
    # perform different kinds of data augmentation
    # (not desired in testing setup)
    "random_crop": False,
    "flip": False,
    "rotate90": False, 
    "color_aug": False,
    
    # post-processing:
    "separate_touching_nuclei": True,
    "fill_holes": True,
    "filter_min_size": 10,
    }

def test(params):

    torch.manual_seed(params["seed"])
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
        
    # create "results_NN" directory to save predictions
    os.makedirs("results_NN", exist_ok=True)
    
    
    # create directory for predictions
    outdir = os.path.join("results_NN",params["model_name"],
                          "predictions_split"+str(params["test_split_idx"])+"_"+str(params["validation_split_idx"]))
    os.makedirs(outdir, exist_ok=True)
    
    eval_dict = new_evaluation_dictionary()
    eval_dict_pp = new_evaluation_dictionary()  
    
        
    model = torch.load(os.path.join("models",
                        params["checkpoint"]+"_"+params["model_name"]+".pytorch"),
                        map_location=device)
        
    plot_losses(params["model_name"]) 


    model.eval()
    
    
    # test data:
    params["split"] = "test_"+str(params["test_split_idx"])
    test_dataset = TissueDataset(params)
    test_dataloader = DataLoader(test_dataset, batch_size=params["batch_size"])
    test_filename_prefixes = test_dataset.get_filename_prefixes()
    
    # val data:
    params["split"] = "validation_"+str(params["validation_split_idx"])
    val_dataset = TissueDataset(params)
    val_dataloader = DataLoader(val_dataset, batch_size=params["batch_size"])
    val_filename_prefixes = val_dataset.get_filename_prefixes()
    
    # train data:
    params["split"] = "train_"+str(params["test_split_idx"])+"_"+str(params["validation_split_idx"])
    train_dataset = TissueDataset(params)
    train_dataloader = DataLoader(train_dataset, batch_size=params["batch_size"])
    train_filename_prefixes = train_dataset.get_filename_prefixes()
    
    dataloader = {"train": train_dataloader,
                  "validation": val_dataloader,
                  "test": test_dataloader}
    
    filename_prefixes = {"train": train_filename_prefixes,
                  "validation": val_filename_prefixes,
                  "test": test_filename_prefixes}
    
    
    
    for subset in ["train", "validation", "test"]:
        print("running through ", subset, " set...")
        for i, data in enumerate(dataloader[subset]):
        
            print("predicting batch nr ", i, "/", len(dataloader[subset]))
            tissue_image, mask_image = data
            
            tissue_image = tissue_image.to(device)
            mask_image = mask_image.to(device)
        
            with torch.no_grad():
                output = torch.sigmoid(model(tissue_image)["out"])
                
                # loop over samples of one batch:
                for s in range(tissue_image.shape[0]):
                    prediction = np.zeros((params["patch_size"], params["patch_size"]), dtype = np.uint8)
                    prediction[output[s,0,:,:]>0.5] = 255
                    
                    # save prediction images
                    pred_PIL = Image.fromarray(prediction)
                    pred_PIL.save(os.path.join(outdir, filename_prefixes[subset][i]+".png"))
                    
                    gt_mask = mask_image[s].detach().cpu().squeeze().numpy()
                    
    
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
                    pred_PIL.save(os.path.join(outdir, "pp_"+filename_prefixes[subset][i]+".png"))

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

    eval_dict, eval_dict_pp = test(default_params)
    print("eval:", eval_dict)
    print("eval_pp:", eval_dict_pp)
# -*- coding: utf-8 -*-
"""
train a convolutional neural network 

"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_utils import TissueDataset
import numpy as np

from tifffile import imsave

from datetime import datetime

default_params = {
    
    # name of savend model and folder with results
    "experiment_name": "TEST", 
    "epochs": 200,
    
    # to resume a model at a later checkpoint
    "start_epoch": 0, 
    "resume_model": "",
    
    # path were to find "mask binary" and "tissue_images" folders
    "data_path": "data",
    
    # images will be cropped to sqared patches
    "patch_size": 256,
    
    "batch_size": 4,
    
    # tissue type index, that is excluded from training
    "test_split_idx": 0,
    # tissue type index, that is excluded from training,
    # but for which the loss is tracked to select best model
    "validation_split_idx": 1,
    
    # make random decisions reproducible
    "seed": 42,
    
    # Only foreground /background segmentation
    # needs only Binary Cross Entropy Loss
    "sem_loss": "BCEWithLogitsLoss",
    
    # normalize stain in tissue images
    "norm_stain": True,
    
    # perform different kinds of data augmentation
    "random_crop": True,
    "flip": True,
    "rotate90": True,
    "color_aug": True,
    }

def train(params):
    
    # make random choices reproducible:
    torch.manual_seed(params["seed"])
    
    # select cpu or gpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
            
    
    outdir = os.path.join("results_NN",params["experiment_name"])
    os.makedirs(outdir, exist_ok=True)
    
    params["split"] = "train_"+str(params["validation_split_idx"])+"_"+str(params["test_split_idx"])
    
    dataset = TissueDataset(params)
    dataloader = DataLoader(dataset, batch_size=params["batch_size"], shuffle=True)
    
    val_params = params.copy()
    val_params["split"] = "validation_"+str(params["validation_split_idx"])
    # no augmentation:
    val_params["flip"] = False
    val_params["rotate"] = False
    val_params["blur"] = False
    val_params["color_aug"] = False
    
    val_dataset = TissueDataset(params)
    val_dataloader = DataLoader(val_dataset, batch_size=params["batch_size"], shuffle=True)
    
    dataloader = {"train": dataloader, "val": val_dataloader}
    
    if params["start_epoch"] == 0:
        try:
            model = torch.load(os.path.join("models", "fcn_resnet50.pytorch"),
                               map_location=device)
        except FileNotFoundError:
            print("Downloading FCN ResNet50 Model")
            model = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet50',
                               pretrained=True)
            torch.save(model, os.path.join("models","fcn_resnet50.pytorch"))
    
        # exchange output layers to predict 2 classes
        model.classifier[4] = nn.Conv2d(512, 1, kernel_size=(1,1), stride=(1,1))
        model.aux_classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
    
    else:
        model = torch.load(os.path.join("models", params["resume_model"]),
                           map_location=device)
    
    
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters())
    
    if params["sem_loss"] == "BCEWithLogitsLoss":
        loss_fn = nn.BCEWithLogitsLoss().to(device)
    else:
        print("Unknown loss function")
    
    loss_history = {"train": [], "val": []}
    min_val_loss = np.inf
    if params["start_epoch"] > 0:
        loss_history["train"] = np.load(os.path.join("results_NN", params["resume_model"][4:-8], "train_loss.npy"))
        loss_history["val"] = np.save(os.path.join("results_NN", params["resume_model"][4:-8], "val_loss.npy"))
        min_val_loss = np.min(np.array(loss_history["val"]))
        
    
    total_starttime = datetime.now()
    
    for epoch in range(params["start_epoch"],params["epochs"]):
        print("running epoch ", epoch, "...")
        epoch_starttime = datetime.now()
        epoch_loss = {"train": 0, "val": 0}  
        
        for phase in ["train", "val"]:
    
            for i, data in enumerate(dataloader[phase]):
        
                tissue_image, mask_image = data
                
                tissue_image = tissue_image.to(device)
                mask_image = mask_image.to(device)
                
                # Forward pass
                output = model(tissue_image)
                
                # Compute loss
                loss = loss_fn(output['out'], mask_image)
                if phase == "train":
                    # Zero out gradients
                    optimizer.zero_grad()
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                elif i==0:
                    # save one validation image per epoch to check training success
                    imsave(os.path.join(outdir, "val_epoch_in_"+str(epoch)+".tif"), tissue_image[0].detach().cpu().moveaxis(0,2).numpy())
                    imsave(os.path.join(outdir, "val_epoch_out_"+str(epoch)+".tif"), output['out'][0].detach().cpu().numpy())
                    imsave(os.path.join(outdir, "val_epoch_gt_"+str(epoch)+".tif"), mask_image[0].detach().cpu().numpy())
    
                
                epoch_loss[phase] = epoch_loss[phase] + loss.item()
            
            # Done with an epoch, print stats
            loss_history[phase].append(epoch_loss[phase]/len(dataloader[phase]))
        endtime = datetime.now()
        print('[', epoch, '] train loss: ', epoch_loss["train"]/len(dataloader["train"]))
        print('[', epoch, '] val loss: ', epoch_loss["val"]/len(dataloader["val"]))
        print("epoch time: ",(endtime-epoch_starttime),", total time: ", (endtime-total_starttime))
    
        np.save(os.path.join("results_NN", params["experiment_name"], "train_loss.npy"), np.array(loss_history["train"]))
        np.save(os.path.join("results_NN", params["experiment_name"], "val_loss.npy"), np.array(loss_history["val"]))
        torch.save(model, os.path.join("models", "last_"+params["experiment_name"]+".pytorch"))
        if epoch_loss["val"] < min_val_loss:
            min_val_loss = epoch_loss["val"]
            torch.save(model, os.path.join("models", "best_"+params["experiment_name"]+".pytorch"))
            
if __name__=="__main__":
    
    train(default_params)
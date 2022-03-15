# -*- coding: utf-8 -*-
"""
Contains method to read and transform data
"""
import numpy as np
import os

from PIL import Image
import torch
from torch.utils.data import Dataset

from norm_stain import normalizeStaining_Macenko, augmentColor

from skimage.feature import hessian_matrix
from skimage.filters import difference_of_gaussians

from postprocessing import separate_touching_nuclei, filter_by_size, fill_holes

from tifffile import imsave


def load_data(path, split="all"):
    filenames = sorted(os.listdir(path))
    image_shape = np.array(Image.open(os.path.join(path,filenames[0]))).shape
    data = np.zeros((len(filenames), *image_shape))
    
    for i, filename in enumerate(filenames):
        data[i] = np.asarray(Image.open(os.path.join(path,filename)))
        
    split_indices = get_split_indices(get_filename_prefixes(path), split)
    
    data = data[split_indices]
       
    return data

def get_filename_prefixes(path, split="all"):
    filenames = sorted(os.listdir(path))
    prefixes = []
    
    for filename in filenames:
        prefix, suffix = os.path.basename(filename).rsplit(".")
        if suffix in ["png","tif"]:
            prefixes.append(prefix)
    
    split_indices = get_split_indices(prefixes, split)

    prefixes = [prefixes[idx] for idx in split_indices]

            
    return prefixes    

def get_split_indices(filename_prefixes, split="all"):
    '''
    Parameters
    ----------
    filename_prefixes : list
        List contains filenames without filetype ending, e.g.
        ["Human_Larynx_01", "Human_Larynx_02", "Human_Mediastinum_03"]
        Tissue types are extracted from the filename prefix and ordered
        alphabetically to get a tissue type index
    split : string, optional
        The default is "all". Which selects the complete data set.
        Otherwise it should have the form
        "test_<idx>": for a test set consisting of tissue type with index idx
        "val_<idx>": for a validation set consisting of tissue type with index idx
        "train_<idx>_<idx2>": for a training set without tissue types with indices idx1 or idx2

    Returns
    -------
    np.array
        The array contains the indices to directly select the subset
        from the list of given filename_prefixes

    '''
    if split == "all":
        return np.arange(len(filename_prefixes))
    
    
    # get tissue types from filename_prefix:
    tissue_types = []
    image_nrs_per_tissue = []
    for prefix in filename_prefixes:
        # prefix have pattern: "Human_<tissue_type>_<index>"
        tissue_type = prefix.split("_",2)[1]
        image_nr_per_tissue = prefix.split("_",2)[2]
        tissue_types.append(tissue_type)
        image_nrs_per_tissue.append(image_nr_per_tissue)
    
    unique_tissue_types = sorted(list(set(tissue_types)))
    
    if split.startswith("test") or split.startswith("validation"):        
        subset, tissue_idx = split.split("_")
    
    elif split.startswith("train"):        
        subset, tissue_idx, tissue_idx2 = split.split("_",2)

    else:
        print("Tissue split must begin with train, validation or test!")
        
    if int(tissue_idx) >= len(unique_tissue_types):
        print("Tissue type ", tissue_idx, " for split does not exist")
        print("Maximum tissue type index is ", len(unique_tissue_types))
        
    if subset == "train":
        if int(tissue_idx) == int(tissue_idx2):
            print("Training without Tissue type", tissue_idx)
        if int(tissue_idx2) >= len(unique_tissue_types):
            print("Tissue type ", tissue_idx, " for split does not exist")
            print("Maximum tissue type index is ", len(unique_tissue_types))
         

    list_of_indices = []
    for idx, filename in enumerate(filename_prefixes):
        if unique_tissue_types[int(tissue_idx)] in filename:
            if subset == "test" or subset == "validation":
                list_of_indices.append(idx)
        elif subset == "train" and not unique_tissue_types[int(tissue_idx2)] in filename:
                list_of_indices.append(idx)                     
                    
    return np.array(list_of_indices)

    
    
class TissueDataset(Dataset):

    def __init__(self, params):
        self.path = params["data_path"]
        self.image_data = load_data(os.path.join(params["data_path"],
                                                 "tissue_images"))
        self.mask_data = load_data(os.path.join(params["data_path"],
                                                "mask binary"))
        self.filename_prefixes = get_filename_prefixes(os.path.join(self.path,
                                                 "tissue_images"))
        
        split_indices = get_split_indices(self.filename_prefixes, params["split"])
        self.mask_data = self.mask_data[split_indices]
        self.image_data = self.image_data[split_indices]
        self.filename_prefixes = [self.filename_prefixes[idx] for idx in split_indices]
        

        self.norm_stain = params["norm_stain"]

        # data augmentation parameters:
        self.rotate90 = params["rotate90"]
        self.patch_size = params["patch_size"]
        self.flip = params["flip"]
        self.random_crop = params["random_crop"]
        self.color_aug = params["color_aug"]
        

    def __getitem__(self, index):    
        image = self.image_data[index]
        mask = self.mask_data[index]/255.0
        
        if self.norm_stain:
            image = normalizeStaining_Macenko(image)
        
       
        image,mask = self.augment(image,mask)

        # reorder/expand dimensions to C x W x H:
        image = np.moveaxis(image,2,0)
        mask = np.expand_dims(mask,0)
 
        imageT = torch.Tensor(image.copy())
        maskT = torch.Tensor(mask.copy())
        return imageT, maskT

    def __len__(self):
        return self.image_data.shape[0]
    
    def get_filename_prefixes(self):
        return self.filename_prefixes

    def augment(self, image, mask):
        
        if self.random_crop:
            cropx = np.random.randint(0, image.shape[0]-self.patch_size+1)
            cropy = np.random.randint(0, image.shape[1]-self.patch_size+1)
        else:
            cropx = 0
            cropy = 0
        image = image[cropx:cropx+self.patch_size, cropy:cropy+self.patch_size]
        mask = mask[cropx:cropx+self.patch_size, cropy:cropy+self.patch_size]
        
        if self.flip:
            if np.random.rand() > 0.5:
                np.fliplr(image)
                np.fliplr(mask)
            if np.random.rand() > 0.5:
                np.flipud(image)
                np.flipud(mask)
        if self.rotate90:
            for i in range(np.random.randint(0,4)):
                image = np.rot90(image)
                mask = np.rot90(mask)
        if self.color_aug:
            image = augmentColor(image)
        return image, mask
    
    
def add_hessian(data, sigma=1):
    n,w,h,c = data.shape
    c = 3 # only use first 3 channels
    hessian_data = np.zeros((n,w,h,c*3))
    for image_idx in range(n):
        for channel_idx in range(c):
        
            hessian_features = np.array(hessian_matrix(data[image_idx,:,:,channel_idx], sigma=sigma, mode='reflect'))
            # 3, w, h -> w, h, 3
            hessian_features = np.moveaxis(hessian_features,0,2)
            hessian_data[image_idx,:,:,c*channel_idx:c*channel_idx+3] = hessian_features
        
    return np.concatenate([data, hessian_data],axis=3)


def add_DoG(data, sigma=1):
    n,w,h,c = data.shape
    c = 3  # only use first 3 channels
    diff_gauss_data = np.zeros((n,w,h,c))
    for image_idx in range(n):
        for channel_idx in range(c):
            # has shape x, h
            diff_gauss_features = difference_of_gaussians(data[image_idx,:,:,channel_idx], low_sigma=sigma, mode='reflect')
            diff_gauss_data[image_idx,:,:,channel_idx] = diff_gauss_features
    
    return np.concatenate([data, diff_gauss_data],axis=3)    






if __name__=="__main__":
    
    import matplotlib.pyplot as plt
    
    print("Testing data utils...")
    params = {
        "data_path": "data",
        "patch_size": 512,
        "split": "train_0_1",
        "norm_stain": False,
        "flip": True,
        "rotate90": True, 
        "random_crop": True, 
        "color_aug": True,
        }
    dataset = TissueDataset(params)
    for j in range(10):
        for i in range(1):
            image, mask = dataset[i]
            fig, ax =  plt.subplots(1,2)

            refined_mask = mask[0].detach().cpu().numpy().astype(np.uint8)
            imsave("mask_before_pp.tif", refined_mask)
            print("pixels !=0 before pp fill: ", np.count_nonzero(refined_mask))
            refined_mask = fill_holes(refined_mask)
            print("pixels !=0 after fill: ", np.count_nonzero(refined_mask))
            refined_mask = filter_by_size(refined_mask, 10)
            print("pixels !=0 after filter size: ", np.count_nonzero(refined_mask))
            refined_mask = separate_touching_nuclei(refined_mask)
            print("pixels !=0 after separate: ", np.count_nonzero(refined_mask))
            imsave("mask_after_pp.tif", refined_mask)
            #ax[0].imshow(image.moveaxis(0,2)/255)
            ax[0].imshow(mask[0], cmap="gray")
            ax[1].imshow(refined_mask, cmap = "gray")
            fig.suptitle("Train Dataset Item "+str(i))
            plt.show()
            fig.clf()

    # params["split"] = "validation_1"
    # dataset = TissueDataset(params)
    # print("validation set has length: ", len(dataset))

    # params["split"] = "test_0"
    # dataset = TissueDataset(params)
    # print("test set has length: ", len(dataset))
    



    
    
    
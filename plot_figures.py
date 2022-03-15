# -*- coding: utf-8 -*-
"""
Creates plots to explore and analyze
data and results

"""
import numpy as np
import os
import pickle

from PIL import Image
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from scipy.ndimage import measurements
from skimage import measure


from norm_stain import normalizeStaining_Macenko
from data_utils import load_data, add_hessian, add_DoG

from postprocessing import separate_touching_nuclei
from evaluate import count_instances
from tifffile import imsave

def plot_colordistribution_imagewise(tissue_data, mask_data, filename):
    
    (n, w, h, c) = tissue_data.shape
    tissue_data = tissue_data.reshape(-1,c)
    
    # create PCA decomposition of color distribution of all images:
    pca = PCA(n_components=1, whiten=True)
    pca.fit(tissue_data)
    
    # create colored legend on the bottom:
    steps = 50
    transformed_colors = pca.transform(tissue_data)
    range_start = np.min(transformed_colors)
    range_end = np.max(transformed_colors)
    array_of_steps = np.arange(range_start,range_end, (range_end-range_start)/50)
    transformed_colors = pca.inverse_transform(np.expand_dims(array_of_steps,1))
    transformed_colors = np.clip(transformed_colors/255.0,0,1)

    tissue_data = tissue_data.reshape(n,w,h,c)

    # plot composition for each image separately:
    for i in range(tissue_data.shape[0]):
        
        nuclei = tissue_data[i,mask_data[i]>0]
        background = tissue_data[i,mask_data[i]==0]
    
        transformed_background_colors = pca.transform(background.reshape(-1,c))
        transformed_nuclei_colors = pca.transform(nuclei.reshape(-1,c))
    
        # Plot data in PCA - transformed space:
        plt.hist(transformed_background_colors, bins=steps, histtype="step", color=(0.5,0.5,0.5), label="background")
        plt.hist(transformed_nuclei_colors, bins=steps, histtype="step", color = (0.5,0,1.0), label="tissue")
        for j in range(steps):
            plt.plot(array_of_steps[j], -0.02,"s", color = (transformed_colors[j,0],transformed_colors[j,1],transformed_colors[j,2]))
        
        plt.title('Color distribution in image '+str(i) )
        plt.show()
        fig = plt.gcf()
        fig.savefig(filename)

        
        
def plot_colordistribution(tissue_data, mask_data, filename):
    
    (n, w, h, c) = tissue_data.shape
    tissue_data = tissue_data.reshape(-1,c)
    
    # create PCA decomposition of color distribution of all images:
    pca = PCA(n_components=1, whiten=True)
    pca.fit(tissue_data)
    
    # create colored legend on the bottom:
    steps = 50
    transformed_colors = pca.transform(tissue_data)
    range_start = np.min(transformed_colors)
    range_end = np.max(transformed_colors)
    array_of_steps = np.arange(range_start,range_end, (range_end-range_start)/50)
    transformed_colors = pca.inverse_transform(np.expand_dims(array_of_steps,1))
    transformed_colors = np.clip(transformed_colors/255.0,0,1)

    tissue_data = tissue_data.reshape(n,w,h,c)

    nuclei = tissue_data[mask_data>0]
    background = tissue_data[mask_data==0]
    
    transformed_background_colors = pca.transform(background.reshape(-1,c))
    transformed_nuclei_colors = pca.transform(nuclei.reshape(-1,c))
    
    # Plot data in PCA - transformed space:
    plt.hist(transformed_background_colors, bins=steps, histtype="step", color=(0.5,0.5,0.5), label="background")
    plt.hist(transformed_nuclei_colors, bins=steps, histtype="step", color = (0.5,0,1.0), label="tissue")
    for j in range(steps):
        plt.plot(array_of_steps[j], -0.02,"s", color = (transformed_colors[j,0],transformed_colors[j,1],transformed_colors[j,2]))
    
    plt.title('Color distribution over all images')
    plt.show()
    fig = plt.gcf()
    fig.savefig(filename)

    
def plot_images(data, n=None):
    if n is None:
        n = data.shape[0]
    for image_nr in range(n):
        plt.imshow(data[image_nr]/255)
        plt.title("Image Nr: "+ str(image_nr))
        plt.show()

def plot_size_distribution(mask_data):
    sizes_by_image = []
    for mask_idx in range(mask_data.shape[0]):
        components, n_components = measurements.label(mask_data[mask_idx])
        sizes_in_image = []
        for c_idx in range(1,n_components+1):    
            sizes_in_image.append(np.count_nonzero(components==c_idx))
        sizes_by_image.append(sizes_in_image)
    plt.hist(sizes_by_image, bins=100, stacked=True)
    plt.title("Distribution of Cell Nuclei Sizes")
    plt.show()
    
    n = 20
    plt.hist(sizes_by_image, bins=n, range=(1,n+1), align="left", stacked=True )
    plt.title("Distribution of Small Cell Nuclei Areas (1-"+str(n)+"px)")
    plt.xticks(ticks=range(1,n+1))
    plt.yticks(ticks=range(1,n+1))
    plt.show()
    
def plot_boxplot_size_distribution(mask_data):
    sizes_by_image = []
    average_size_by_image = []
    count_per_image = []
    for mask_idx in range(mask_data.shape[0]):
        components, n_components = measurements.label(mask_data[mask_idx])
        sizes_in_image = []
        for c_idx in range(1,n_components+1):    
            sizes_in_image.append(np.count_nonzero(components==c_idx))
        sizes_by_image.append(sizes_in_image)
        average_size_by_image.append(np.count_nonzero(mask_data[mask_idx])/n_components)
        count_per_image.append(n_components)
    plt.boxplot(sizes_by_image)
    plt.plot(range(mask_data.shape[0]), average_size_by_image, label="average size per cell")
    plt.plot(range(mask_data.shape[0]), count_per_image, label="# cells per image")
    plt.ylim(0,2000)
    plt.legend()
    plt.title("Distribution of Cell Nuclei Sizes")
    plt.show()
    


def show_smallest_nuclei(mask_data, tissue_data, n_samples=30):

    list_of_tissueROIs = []
    list_of_maskROIs = []
    sizes_of_nuclei = np.ones((n_samples))*np.inf
    for mask_idx in range(mask_data.shape[0]):
        components, n_components = measurements.label(mask_data[mask_idx])
        current_max = sizes_of_nuclei.max()
        for c_idx in range(1,n_components+1):    
            size = np.count_nonzero(components==c_idx)
            if size < current_max:
                # get coordinates of this component + 1 px padding
                x_indices = np.where(np.max(components==c_idx, axis=1))
                startx = np.maximum(0,np.min(x_indices)-1)
                endx = np.minimum(np.max(x_indices)+2,mask_data[mask_idx].shape[0])
                y_indices = np.where(np.max(components==c_idx, axis=0))
                starty = np.maximum(0,np.min(y_indices)-1)
                endy = np.minimum(np.max(y_indices)+2,mask_data[mask_idx].shape[1])
                # create ROI from images
                tissueROI = Image.fromarray(tissue_data[mask_idx,startx:endx,starty:endy,:].astype(np.uint8))
                component_mask = np.zeros_like(mask_data[mask_idx,startx:endx,starty:endy])
                component_mask[components[startx:endx,starty:endy] ==c_idx] = 255
                maskROI = Image.fromarray(component_mask)
                if len(list_of_tissueROIs) < n_samples:
                    list_of_tissueROIs.append(tissueROI)
                    list_of_maskROIs.append(maskROI)
                    sizes_of_nuclei[len(list_of_tissueROIs)-1] = size
                else:
                    max_position = np.argmax(sizes_of_nuclei)
                    sizes_of_nuclei[max_position] = size
                    list_of_tissueROIs[max_position] = tissueROI
                    list_of_maskROIs[max_position] = maskROI
                current_max = sizes_of_nuclei.max()

    rows = 5 #int(np.ceil(np.sqrt(n)))
    cols = int(np.ceil(n_samples//5 ))#int(np.ceil(n / rows))
    fig, axs = plt.subplots(rows, cols)
    for idx, tissueROI in enumerate(list_of_tissueROIs):
        tissue = np.asarray(tissueROI)/255
        #print("plotting tissue with value range: ", tissue.min(), "-", tissue.max())
        axs[idx // cols, idx % cols].imshow(tissue)
        #print("plotting small nuclei nr ", idx)
        mask = np.asarray(list_of_maskROIs[idx])
        for x in range(0,mask.shape[0]):
            for y in range(0,mask.shape[1]):
                if x<mask.shape[0]-1 and mask[x,y] != mask[x+1,y]:
                    axs[idx // cols, idx % cols].plot([y-0.5,y+0.5],[x+0.5, x+0.5], color="yellow")
                if x>1 and mask[x,y] != mask[x-1,y]:
                    axs[idx // cols, idx % cols].plot([y-0.5,y+0.5],[x-0.5, x-0.5], color="yellow")
                if y<mask.shape[1]-1 and mask[x,y] != mask[x,y+1]:
                    axs[idx // cols, idx % cols].plot([y+0.5,y+0.5],[x-0.5, x+0.5], color="yellow")
                if y>1 and mask[x,y] != mask[x,y-1]:
                    axs[idx // cols, idx % cols].plot([y-0.5,y-0.5],[x-0.5, x+0.5], color="yellow") 
        axs[idx // cols, idx % cols].axis('off')
    fig.savefig("figures/SmallNucleiExamples.png")

def show_stain_normalization(tissue_data):
    
    fig, axs = plt.subplots(1, 2)
    
    for image_nr in range(tissue_data.shape[0]):

        Macenko = normalizeStaining_Macenko(tissue_data[image_nr])

        axs[0].imshow(tissue_data[image_nr]/255.)
        axs[0].set_title("Original")
        axs[0].axis("off")
        axs[1].imshow(Macenko)
        axs[1].set_title("Macenko")
        axs[1].axis("off")
    
        fig.savefig("StainNormalization_Image"+str(image_nr)+".png")





def compare_nuclei_counts_after_refine(tissue_data, mask_data):

    n = mask_data.shape[0]
    nuclei_counts = [ count_instances(mask_data[i]) for i in range(n)]
    refined_counts = [ count_instances(separate_touching_nuclei(mask_data[i], i)) for i in range(n)]

    plt.plot(range(n), nuclei_counts, label="count")
    plt.plot(range(n), refined_counts, label="refined counts")
    plt.legend()
    plt.title("nuclei counts")
    plt.show()
    fig = plt.gcf()
    fig.savefig("NucleiCountsRefined.pdf")


def export_feature_images(tissue_data):
   
    for image_nr in range(tissue_data.shape[0]):
        
        folder_name = "features"
        feature_names = ["R","G","B",
                         "Hess1_R_rc","Hess1_R_rr","Hess1_R_cc",
                         "Hess1_G_rc","Hess1_G_rr","Hess1_G_cc",
                         "Hess1_B_rc","Hess1_B_rr","Hess1_B_cc",
                         "Hess2_R_rc","Hess2_R_rr","Hess2_R_cc",
                         "Hess2_G_rc","Hess2_G_rr","Hess2_G_cc",
                         "Hess2_B_rc","Hess2_B_rr","Hess2_B_cc",
                         "LoG1_R","LoG1_G","LoG1_B",
                         "LoG2_R","LoG2_G","LoG2_B"]
        os.makedirs(folder_name, exist_ok=True)
        
        imsave(os.path.join(folder_name,"Original"+str(image_nr)+".tif"), tissue_data[image_nr])

        tissue_data[image_nr] = add_hessian(tissue_data[image_nr], sigma=1)
        tissue_data[image_nr] = add_hessian(tissue_data[image_nr], sigma=2)
        tissue_data[image_nr] = add_DoG(tissue_data[image_nr], sigma=1)
        tissue_data[image_nr] = add_DoG(tissue_data[image_nr], sigma=2)

        for d in range(3,tissue_data.shape[2]):
            imsave(os.path.join(folder_name,"Imagenr_"+str(image_nr)+"_"+feature_names[image_nr]+".tif"), tissue_data[image_nr,:,:,d])



def visualize_average_precision(pred_mask, gt_mask, overlap=0.5):
   
    pred_instances = measure.label(pred_mask)
    gt_instances = measure.label(gt_mask)
    
    gt_matching_image = np.zeros((*pred_mask.shape, 3))
    pred_matching_image = np.zeros((*pred_mask.shape, 3))
    
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
                            pred_matching_image[pred_instances == pred_label,:] = [0,255,0]
                            gt_matching_image[gt_instances == gt_label,:] = [0,255,0]
                            break
            if not matched:
                gt_matching_image[gt_instances == gt_label,:] = [255,0,0]
                FN += 1 
                
    for pred_label in np.unique(pred_instances):
        if pred_label != 0 and  pred_label not in matched_pred_labels:
            FP += 1
            pred_matching_image[pred_instances == pred_label,:] = [255,0,0]
    
    #print("TP: ", TP, " FP: ", FP, " FN: ", FN)
    average_precision = TP / (TP + 0.5*FN + 0.5*FP)
    
    fig, ax =  plt.subplots(2,2)
    
    ax[0,0].imshow(gt_mask)
    ax[0,0].set_title("GT Mask")
    ax[0,0].set_axis_off()
    ax[0,1].imshow(pred_mask)
    ax[0,1].set_title("Pred Mask")
    ax[0,1].set_axis_off()   
    ax[1,0].imshow(gt_matching_image)
    ax[1,0].set_title("GT Match")
    ax[1,0].set_axis_off()  
    ax[1,1].imshow(pred_matching_image)
    ax[1,1].set_title("Pred Match")
    ax[1,1].set_axis_off() 
    fig.suptitle("AP: TP="+str(TP)+", FP="+str(FP)+", FN="+str(FN))

    return average_precision

 ## plot losses from training a CNN model
def plot_losses(model_name):   
    train_loss = np.load(os.path.join("results_NN",model_name,"train_loss.npy"))
    val_loss = np.load(os.path.join("results_NN",model_name,"val_loss.npy"))
    plt.plot(range(train_loss.shape[0]), train_loss, label="train")
    plt.plot(range(val_loss.shape[0]), val_loss, label="val")
    plt.legend()
    plt.show()
    plt.savefig( os.path.join("results_NN",model_name,"loss.png") )
    
def plot_evaluation_results_from_file(eval_filename, title, filename_indices = None):
    
    eval_dict = pickle.load(open(eval_filename, 'rb'))
    plot_evaluation_results(eval_dict, title, filename_indices = None)

def plot_evaluation_results(eval_dict, title, filename_indices = None):
    
    for key in eval_dict.keys():
        if key == "gt counts":
            count_error = np.divide(np.abs(np.array(eval_dict["gt counts"])- np.array(eval_dict["pred counts"])),np.array(eval_dict["gt counts"]) )
            plt.plot(range(len(eval_dict[key])), count_error/100, label="APE")
        elif key == "pred counts":
            pass
        else:
            plt.plot(range(len(eval_dict[key])), eval_dict[key], label=key)

        
    plt.legend()
    plt.title(title)
    if filename_indices is None:
        xticklabels = range(len(eval_dict["Accuracy"]))
        xticklocations = range(len(eval_dict["Accuracy"]))
        if np.nan in eval_dict["Accuracy"]:
            xticklocations = [xticklocations[x] for x in range(len(eval_dict["Accuracy"])) if np.isnan(eval_dict["Accuracy"][x])]
            xticklabels = ["" for x in xticklocations]
            plt.text(0.5 *xticklocations[0], -0.05, "train")
            plt.text(xticklocations[0]+0.5 *(xticklocations[1]-xticklocations[0]), -0.3, "val")
            plt.text(xticklocations[1]+0.5 *(xticklocations[2]-xticklocations[1]), -0.3, "test")
        plt.xticks(xticklocations, xticklabels)

    else:
        plt.xticks(range(len(filename_indices)), filename_indices)
    plt.show()
    fig = plt.gcf()
    os.makedirs("figures", exist_ok=True)
    fig.savefig(os.path.join("figures",title+".pdf"))
    plt.clf()


if __name__=="__main__":
    
    # create "figures" directory to save all plots
    os.makedirs("figures", exist_ok=True)
    
    # load data
    tissue_data = load_data(os.path.join("data", "tissue_images"))
    mask_data = load_data(os.path.join("data", "mask binary"))
    
    # plot all images
    #plot_images(tissue_data)
    
    # plot histogram of annotated mask sizes
    #plot_size_distribution(mask_data)
    #plot_boxplot_size_distribution(mask_data)
    
    # plot ROIs with small mask sizes
    #show_smallest_nuclei(mask_data, tissue_data, 30)
    
    # plot color distributions of nuclei and background
    # in RGB -> 1D projected PCA-space 
    #plot_colordistribution(tissue_data, mask_data)

    #plot_colordistribution(normalizeData(tissue_data, method="Macenko"), mask_data)
    
    # show HE images before and after stain normalization
    #show_stain_normalization(tissue_data)
    
    #compare_nuclei_counts_after_refine(tissue_data, mask_data)
    
    #compare_errors_with_estimate(mask_data)
    
    #plot_features(tissue_data)
    
    
    #plot_results_from_file("results_rf/rfc_all_test0/eval.dict", title="Evaluation for RF model with All Features")
    #plot_results_from_file(("results_rf/rfc_onlyRGB_test0/eval.dict", title="Evaluation for RF model without additional Features")

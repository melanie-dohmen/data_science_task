# data_science_task


This repository solves a cell nuclei segmentation and cell nuclei counting task.

Two approaches were developed.

1. Convolutional Neural Network
2. Random Forest Pixel Classifier

## Getting Started

git clone https://github.com/melaniedohmen/data_science_task

Unpack data/data.zip into data/

```
conda env create -f environment.yml
conda activate data_science_task
```
Then run
```
python segment.py
```

By default runs prediction on fine-tuned CNN model
(parameters documented in `testNN.py`)

```
 # call test routine for CNN model
 # evaluation metrics are returned as dictionaries
 eval_dict_ftNN`, eval_dict_pp_ftNN = test(CNN_test_params)
 
 # plot evaluation metrics for unprocessed predictions:
 plot_evaluation_results(eval_dict_ftNN, "Evaluation of ft-CNN model")
 
 # plot evaluation metrics after post-processing predictions:
 plot_evaluation_results(eval_dict_pp_ftNN, "Evaluation of ft-CNN model after pp")
```

To train a new CNN with different parameters do:
(parameters documented in `train_NN.py`)

```
 # call train routine for CNN model (see parameters)
 train(CNN_train_params)
 
 # get model name:
 CNN_test_params["model_name"] = CNN_train_params["experiment_name"]
 
 # run test routine:
 eval_dict_NN, eval_dict_pp_NN = test(CNN_test_params)
 
 # plot results with and without post-processing:
 plot_evaluation_results(eval_dict_NN, "Evaluation of newly trained CNN model")
 plot_evaluation_results(eval_dict_pp_NN, "Evaluation of newly trained CNN model after pp")
```

To train a random forest classifier do:
(parameters documented in `rf_pixel_classifier.py`)

```
 # call train and test routine for random forest classifier 
 eval_dict_rfc, eval_dict_pp_rfc = train_and_test(rfc_params)
 
 # plot results with and without post-processing:
 plot_evaluation_results(eval_dict_rfc, "Evaluation of RFC Model")
 plot_evaluation_results(eval_dict_pp_rfc, "Evaluation of RFC Model after pp")

```

## Data

Two folders of data are available: 

1. data/tissue_images" with 28 HE stained images (512x512px, TIFF-Format)
2. data/binary mask 28 binary images (512x512px, PNG-Format)

There are 10 human tissue types with (samples) available:
- AdrenalGland (3)
- Larynx (3)
- LymphNodes (2)
- Mediastinum (3)
- Pancreas (3)
- Pleura (3)
- Skin (3)
- Testis (3)
- Thymus (3)
- ThyroidGland (2)

To ensure best generalabilty, the samples of AdrenalGland (a complete tissue type)
 was selected as test set and excluded from training.

By default, Larynx is selected as validation set.
 
To better compare models based on the small dataset, a cross-validation approach was implemented
(for the random forest classifier).


## Evaluation

The following evaluation metrics were implemented

-Accuracy
-Average Precision for IoU threshold 0.5
-Mean Average Precision mAP for IoU thresholds in range(0.5;1.0;0.05)
-Absolute Percentage Error (on cell nuclei counts)
-Intersection over Union 

## Pre-processing

HE Stain Normalization is performed according to 

https://github.com/schaugf/HEnorm_python/blob/master/normalizeStaining.py

[1] A method for normalizing histology slides for quantitative analysis,
 M Macenko, M Niethammer, JS Marron, D Borland, JT Woosley,
 G Xiaojun, C Schmitt, NE Thomas,
 IEEE ISBI, 2009. dx.doi.org/10.1109/ISBI.2009.5193250
 

Data aumgmentation was performed for CNN training including:
- horizontal/vertical flip
- rotation by 0,90,180 or 270 degrees
- random crop

- color augmentation was implemented, but did not yield better results


## Post-processing

Types of postprocessing that are applied by default in this order:
(see postprocessing.py)

- filling holes
- filtering out all segments with area \< 11px 
- separate touching nuclei
  (watershed on distance transform)
  
## CNN model

Using a pretrained fully convolutional Res-Net-50 model from:

https://pytorch.org/hub/pytorch_vision_fcn_resnet101/

Modified the output layers (classifier and aux_classifier) to have only 1
output channel for foreground/background segmentation


## Random Forest Classifier

See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

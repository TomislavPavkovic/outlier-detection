# Outlier detection

This is the repository of an interdisciplinary project on the topic of automated outlier detection in organ and bone segmentation with shape reconstruction.

Student: Tomislav Pavković
Advisor: Robert Graf
Supervisor: Jan Kirschke

All subfolders (preprocessing, evaluation, if-net) have their own README file with instructions for running the program.

## Motivation

Segmentation in medicine refers to the process of identifying and delineating specific structures or regions of interest within medical images, such as CT scans, MRI images, or X-rays. It involves the precise outlining or labeling of anatomical structures, tumors, organs, or tissues to separate them from the background or other structures. Medical image segmentation is crucial for accurate diagnosis, treatment planning, and quantitative analysis in various medical applications.

For efficient biomarker selection automated segmentation of large epidemiological cohorts is essential since it’s difficult to manually verify a huge number of scans.

To solve this problem as a part of the interdisciplinary project I tryed to enhance the automated segmentation process through the utilization of Neural Implicit Representation techniques to learn the shape prior.

## Dataset

The dataset used for this project consists of TotalSegmentator project’s ground truth and its predictions, and the German national cohort’s predicted Abdomen segmentation.

The dataset contains 4739 CT scans. In order to solve the inconsistent organ selection in the scans, and since it’s easier and more accurate to work with each organ independently the model was trained on individual organs chosen based on size, complexity and tissue type.

There are pretrained models available for right kindey, liver, pancreas and right hip [here](https://drive.google.com/drive/folders/1G4yvbw-ClqmgoQ3VOxddSx0gK_nTQLlo?usp=sharing).

## Solution

To solve the outlier detection problem in this project, shape priors were used to incorporate prior knowledge about typical shapes of organs or structures.
Shape reconstruction was used to obtain ideal object shape. After the reconstruction, chosen metrics are calculated to compare the input and the reconstruction and determine the outliers based on the metrics score. This way it is possible to locate the defect by visually comparing input and the reconstruction.

In order to reconstruct the ideal organ shape, Neural Implicit Representation model was used.
Software for automatic analysis of CT scans compares the segmentation and the reconstruction using metrics like Dice score, Hausdorff distance, Average Surface Distance or Maximal Distance and setting a threshold for the outliers.

## Data preprocessing

To start the training some preprocessing is needed. Extraction of the single organ requires knowing the segmentation label of that organ. That information is contained in excel spreadsheet as the organs in column names are ordered the same way as segmentation labels. After getting the segmentation label for chosen organ in each group the segmentation is extracted, the object is centered and image is resized as the model input resolution is 128 voxels in each dimension. 

Incomplete organ segmentation are filtered out.

Then the format needs to be converted from nifti to numpy array and the boundary sampling needs to be done.

## Data augmentation

First augmentation implemented is a so called online or on the fly data augmentation method which adds a cube of random size and crops a cube of random size out of the input shape during each training iteration. 
Since it is very fast process it is beneficial to do it on the fly because that way it can add different deformations to the same input in different epochs of the training.

To add a more random shape, because a cube is not a natural and expected shape in the organ, second augmentation uses perlin noise by creating a perlin noise in a space the same size as the input image and keeping the values above the certain threshold, that way creating a random shapes in the space. Then created random shapes are added and deleted from the original organ shape.

Because the organ size varies it is more effective to add shapes only to the close proximity of the organ and that way the deletion also works better since there is a higher probability of the random shapes actually overlapping with the organ shape. Third augmentation restricts the area for shape generation to the size of the object plus a few voxels on all sides.

Third augmentation showed the best results and is recommended.

Since the perlin noise generation is a time consuming process it is done offline, meaning deformed image is generated for each scan before the training.

## Metrics and outliers

For the comparison of input and output Dice score, Average surface distance, 95% Hausdorff distance and Maximum distance were calculated but the Dice score showed the best differentiation between good and bad reconstructions which is why it is recommended for calculating the outlier threshold based on the certain percentage of worst dice scores. 

## Conclusions

After the inspection of detected outliers, the model seems successful. With a cleaner dataset the model could be more successful and more precise both in the reconstruction and the outlier detection. 

During the hyperparameter tuning process, data augmentation proved to be very important and made the biggest difference on the results. In case of multi organ segmentation it was shown that current resolution of 128 voxels in all dimensions is not sufficient for a successful reconstruction. 

## Next steps

Some promising steps for the future work would include training the model on the cleaner dataset. It would also be interesting to try some other reconstruction models.
It could also be useful to find a faster algorithm for noise creation to be able to use in on the fly.

For the multiorgan reconstruction, adapting the model for a higher input resolution would most likely be beneficial.
Lastly, training the model on all other organs to be able to provide pretrained model, ready for use.

# Optimal hyperparameters

The optimal hyperparameters are set in the config files.
For training the model following following hyperparameters were used:

Learning rate: 1e-4
Optimizer: Adam
Adam weight decay: 1e-5
Pointcloud samples: 3000
Sample distribution: [0.5, 0.5]
Sample sigmas: [0.1, 0.01]
Batch size: the biggest that can fit in GPU (10 was used)
Augemnted: True

Preprocessing parameters:

Boundary sampling sigmas: [0.1, 0.01]
Boundary sampling sample number: 100000
Perlin noise augmentation (deform_input.py):
    threshold minimum: 0.65
    threshold maximim: 1

Outlier detection (analyse_image.py) parameters:

metrics: ['dice score']

Evaluation parameters:

outlier_percentage: 0.005

## Installation
Create a new conda environment by executing the following command:
```shell
conda env create --name evaluation --file=env.yml
```

Now, activate the new environment:
```shell
conda activate outlier-detection
```

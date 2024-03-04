# Preprocessing

This package includes scripts for extraction of selected organs from a scan, size adaptation of the images, file counting and file removing.

These scripts are used to prepare dataset for the model training.

The model used in this project requires input images size to be 128 voxels in each dimension.

## Installation

If outlier-detection environment was installed from the env.yml in the root directory no additional installations are needed.
In the extract_label.py file, change the path to your preprocessing folder in line sys.path.append('').

## Preprocessing

To extract an organ or multiple organs segmentations from the input image, run the following command
```
python extract_label.py
```

To resize the segmentations that are bigger than desired size, center the organs in the image space, and filter out incomplete organ segmentations (the whole organ is not visible in the scan), run the following command
```
python adapt_size.py
```

remove_file.py and count_files.py can be used if removing or counting files with specified file ending is needed.

Parameters for the scripts should be specified in preprocessing_config.yaml file. Parameters are explained in the comments.

The scripts are dataset specific because of the folder structure and naming. It is adapted to the dataset used for this project containg TotalSegmentator project's ground truth and it's predictions and German national cohort's predicted Abdomen segmentation, for other dataset main functions might need adaptation while other functions are universal. 


import nibabel as nib
import numpy as np
import pandas as pd
import os
from scipy.ndimage import zoom
from nilearn.image import resample_img
import hydra
from omegaconf import DictConfig

def adapt_size(res, file_path, new_file_path):
    ct_scan = nib.load(file_path)
    image_data = ct_scan.get_fdata()

    nonzero_voxels = np.array(np.where(image_data > 0))
    min_coords = np.min(nonzero_voxels, axis=1)
    max_coords = np.max(nonzero_voxels, axis=1)

    if (min_coords==0).any() or (max_coords+1==image_data.shape).any():
        print('Skipping (cut object)', file_path)
        return 1
    
    if ((max_coords - min_coords) > 128).any():
        print('zooming', file_path)
        zoom_multipliers = [(max_coord - min_coord) / 124 * 1.5
                            if max_coord - min_coord > 128 else 1.5 
                            for max_coord, min_coord in zip(max_coords, min_coords)]
        ct_scan = resample_img(ct_scan, target_affine = nib.affines.rescale_affine(ct_scan.affine, ct_scan.shape, (max(zoom_multipliers),)*3), interpolation='nearest')
        image_data = ct_scan.get_fdata()

        nonzero_voxels = np.array(np.where(image_data > 0))
        min_coords = np.min(nonzero_voxels, axis=1)
        max_coords = np.max(nonzero_voxels, axis=1)

    center = (max_coords + min_coords) // 2

    # Define the desired new size for the cropped image
    new_size = (res, res, res)  # Replace with the desired dimensions

    # Calculate the new crop boundaries around the centered object
    start = np.maximum(center - np.array(new_size) // 2, 0)
    end = np.minimum(start + new_size, np.array(image_data.shape))

    # Crop the image around the object and resize it to the desired dimensions
    cropped_image = image_data[start[0]:end[0], start[1]:end[1], start[2]:end[2]]

    current_shape = cropped_image.shape

    # Define the desired minimum dimensions
    desired_shape = (128, 128, 128)

    if any(cs < ds for cs, ds in zip(current_shape, desired_shape)):
            # Calculate the amount of padding needed
            pad_widths = [(max(ds - cs, 0), 0) for cs, ds in zip(current_shape, desired_shape)]
            pad_widths = [(0, pw[0]) for pw in pad_widths]  # Ensure padding is positive
            
            # Pad the image with zeros to reach the desired shape
            cropped_image = np.pad(cropped_image, pad_width=pad_widths, mode='constant')
            #print(str(current_shape) + " to " + str(cropped_image.shape))
    # Create a new NIfTI image with the cropped and resized data, using the original image's affine
    new_affine = ct_scan.affine
    new_nifti_img = nib.Nifti1Image(cropped_image, new_affine)
    os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
    nib.save(new_nifti_img, new_file_path)

@hydra.main(version_base=None, config_path='.', config_name='preprocessing_config')
def main(cfg: DictConfig):
    root_directory = cfg.adapt_size.root_directory
    dataset_name =  cfg.adapt_size.dataset_name
    new_dataset_name =  cfg.adapt_size.new_dataset_name
    res =  cfg.adapt_size.resolution
    new_root_directory = root_directory.replace(dataset_name, new_dataset_name)
    if not os.path.exists(new_root_directory):
        os.makedirs(new_root_directory)
    res=128
    for root, dirs, files in os.walk(root_directory):
        for file in files:
            file_path = os.path.join(root, file)
            new_root = root.replace(dataset_name, new_dataset_name)
            base, extension = os.path.splitext(file)
            base2, extension2 = os.path.splitext(base)
            new_filename = f"{base2}_sized{extension2}{extension}"
            new_file_path = os.path.join(new_root, new_filename)
            adapt_size(res, file_path, new_file_path)
                
if __name__ == "__main__":
    main()
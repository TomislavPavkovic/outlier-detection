import nibabel as nib
import numpy as np
import os

def extract_label(label, file_path, new_file_path):
    ct_scan = nib.load(file_path)
    label_data = ct_scan.get_fdata()
    label_value = 5
    label_mask = (label_data == label_value)
    if np.any(label_mask):
        os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
        new_image = nib.Nifti1Image(label_mask, ct_scan.affine, ct_scan.header)
        nib.save(new_image, new_file_path)

if __name__ == "__main__":
    root_directory = "/home/tomislav/datasets-ct/dataset-CACTS-generated"
    dataset_name = "datasets-ct"
    new_dataset_name = "datasets-ct-extracted"
    new_root_directory = root_directory.replace(dataset_name, new_dataset_name)
    if not os.path.exists(new_root_directory):
        os.makedirs(new_root_directory)
    label = 5
    for root, dirs, files in os.walk(root_directory):
        for file in files:
            if file.endswith("_part_251.nii.gz"):
                file_path = os.path.join(root, file)
                new_root = root.replace(dataset_name, new_dataset_name)
                base, extension = os.path.splitext(file)
                base2, extension2 = os.path.splitext(base)
                new_filename = f"{base2}_{label}{extension2}{extension}"
                new_file_path = os.path.join(new_root, new_filename)
                extract_label(label, file_path, new_file_path)
import nibabel as nib
import numpy as np
import pandas as pd
import os

def extract_label(label, file_path, new_file_path):
    ct_scan = nib.load(file_path)
    label_data = ct_scan.get_fdata()

    label_mask = (label_data == label)

    if np.any(label_mask):
        os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
        new_image = nib.Nifti1Image(label_mask, ct_scan.affine, ct_scan.header)
        nib.save(new_image, new_file_path)

if __name__ == "__main__":
    root_directory = "/home/tomislav/datasets-ct/dataset-CACTS"
    dataset_name = "datasets-ct"
    new_dataset_name = "datasets-ct-extracted"
    new_root_directory = root_directory.replace(dataset_name, new_dataset_name)
    if not os.path.exists(new_root_directory):
        os.makedirs(new_root_directory)
    label = 5
    organ_name = 'liver'
    for root, dirs, files in os.walk(root_directory):
        for dir in dirs:
            if dir.startswith("00"):
                print(dir)
                summary_filename = root + '/' + dir + '/dataset_summary.xlsx'
                if not os.path.exists(summary_filename):
                    continue
                summary_excel = pd.read_excel(summary_filename, engine='openpyxl')
                if 'background' in summary_excel.columns:
                    background_col_num = summary_excel.columns.get_loc('background')
                    end_index = summary_excel.columns.get_loc('organs_sum')
                    for organ in summary_excel.columns[background_col_num + 1 : end_index]:
                        print(organ)
                        organ_name = organ.lower()
                        organ_col_num = summary_excel.columns.get_loc(organ)
                        organ_label = organ_col_num - background_col_num
                        organ_col = summary_excel[organ]
                        for index, id in enumerate(summary_excel['ids']):
                            if organ_col[index] == 1:
                                if id.startswith('LITS_test-volume'):
                                    file = dir + '/labels/' + id + '_0000.nii.gz'
                                else:
                                    file = dir + '/labels/' + id + '.nii.gz'
                                file_path = os.path.join(root, file)
                                new_root = root.replace(dataset_name, new_dataset_name)
                                new_root = new_root.replace('CACTS', 'CACTS/'+organ_name)
                                base, extension = os.path.splitext(file)
                                base2, extension2 = os.path.splitext(base)
                                new_filename = f"{base2}_{organ_name}{extension2}{extension}"
                                new_file_path = os.path.join(new_root, new_filename)
                                if os.path.exists(new_file_path):
                                    print(new_file_path)
                                else:
                                    extract_label(organ_label, file_path, new_file_path)

        '''for file in files:
            if file.endswith("_part_251.nii.gz"):
                file_path = os.path.join(root, file)
                new_root = root.replace(dataset_name, new_dataset_name)
                base, extension = os.path.splitext(file)
                base2, extension2 = os.path.splitext(base)
                new_filename = f"{base2}_{label}{extension2}{extension}"
                new_file_path = os.path.join(new_root, new_filename)
                extract_label(label, file_path, new_file_path)
                '''
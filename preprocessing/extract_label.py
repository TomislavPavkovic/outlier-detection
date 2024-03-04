import nibabel as nib
import numpy as np
import pandas as pd
import os
import hydra
from omegaconf import DictConfig
import sys
sys.path.append('/home/tomislav/outlier-detection/preprocessing')
from map_to_binary import class_map

def extract_label(label, file_path, new_file_path):
    ct_scan = nib.load(file_path)
    label_data = ct_scan.get_fdata()

    label_mask = np.isin(label_data, label)
    if np.any(label_mask):
        os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
        new_image = nib.Nifti1Image(label_mask, ct_scan.affine, ct_scan.header)
        nib.save(new_image, new_file_path)
        print(new_file_path)

@hydra.main(version_base=None, config_path='.', config_name='preprocessing_config')
def main(cfg: DictConfig):
    root_directory = cfg.extract_label.root_directory
    dataset_name =  cfg.extract_label.dataset_name
    new_dataset_name =  cfg.extract_label.new_dataset_name
    label = np.array(cfg.extract_label.get('labels'))
    organ_name = cfg.extract_label.organ_name
    generated = cfg.extract_label.generated
    multi_organ = cfg.extract_label.multi_organ
    new_root_directory = root_directory.replace(dataset_name, new_dataset_name)
    file_ending = cfg.extract_label.file_ending

    if not os.path.exists(new_root_directory):
        os.makedirs(new_root_directory)
    
    for root, dirs, files in os.walk(root_directory):
        if generated:
            for file in files:
                subsection = cfg.extract_label.get('dataset_subsection')
                if subsection is not None and subsection not in root:
                    continue
                if file.endswith(file_ending):
                    if multi_organ:
                        file_path = os.path.join(root, file)
                        new_root = root.replace(dataset_name, new_dataset_name)
                        new_root = new_root.replace('dataset-CACTS-generated', 'dataset-CACTS-generated/' + organ_name)
                        new_root = new_root.replace('images', 'labels')
                        new_root = '/'.join(new_root.split('/')[:-1])
                        new_filename = file.replace('combined', organ_name)
                        new_file_path = os.path.join(new_root, new_filename)
                        extract_label(label, file_path, new_file_path)
                    else:
                        for organ_label in label:
                            generated_organ_name = class_map.get(organ_label)
                            file_path = os.path.join(root, file)
                            new_root = root.replace(dataset_name, new_dataset_name)
                            new_root = new_root.replace('dataset-CACTS-generated', 'dataset-CACTS-generated/' + generated_organ_name)
                            new_root = new_root.replace('images', 'labels')
                            new_root = '/'.join(new_root.split('/')[:-1])
                            new_filename = file.replace('combined', generated_organ_name)
                            new_file_path = os.path.join(new_root, new_filename)
                            extract_label(organ_label, file_path, new_file_path)
        else:
            if multi_organ:
                for file in files:
                    subsection = cfg.extract_label.get('dataset_subsection')
                    if subsection is not None and subsection not in root:
                        continue
                    if file.endswith(file_ending):
                        file_path = os.path.join(root, file)
                        new_root = root.replace(dataset_name, new_dataset_name)
                        new_root = new_root.replace('dataset-CACTS', 'dataset-CACTS/' + organ_name)
                        base, extension = os.path.splitext(file)
                        base2, extension2 = os.path.splitext(base)
                        new_filename = f"{base2}_{organ_name}{extension2}{extension}"
                        new_file_path = os.path.join(new_root, new_filename)
                        extract_label(label, file_path, new_file_path)
            else:
                for dir in dirs:
                    if dir.startswith("00"):
                        print(dir)
                        subsection = cfg.extract_label.get('dataset_subsection')
                        if subsection is not None and subsection not in dir:
                            continue
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
                                if len(label) > 0 and organ_label not in label:
                                    continue
                                for index, id in enumerate(summary_excel['ids']):
                                    if organ_col[index] == 1:
                                        if id.startswith('LITS_test-volume'):
                                            file = dir + '/labels/' + id + '_0000.nii.gz'
                                        else:
                                            file = dir + '/labels/' + id + '.nii.gz'
                                        file_path = os.path.join(root, file)
                                        new_root = root.replace(dataset_name, new_dataset_name)
                                        new_root = new_root.replace('CACTS', 'CACTS/' + organ_name)
                                        base, extension = os.path.splitext(file)
                                        base2, extension2 = os.path.splitext(base)
                                        new_filename = f"{base2}_{organ_name}{extension2}{extension}"
                                        new_file_path = os.path.join(new_root, new_filename)
                                        if os.path.exists(new_file_path):
                                            print(new_file_path)
                                        else:
                                            extract_label(np.array([organ_label]), file_path, new_file_path)

if __name__ == "__main__":
    main()
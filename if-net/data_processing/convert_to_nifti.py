import hydra
import numpy as np
import nibabel as nib
import os
import glob

from omegaconf import DictConfig


def convert_to_nifti(res, input_path, output_path, ground_truth):
        np_data = np.load(input_path)
        np_data = np.unpackbits(np_data)
        np_data = np.reshape(np_data, (res,)*3)
        try:
            ni_ground_truth = nib.load(ground_truth)
        except:
            print("skip: "+input_path)
            return
        new_nifti = nib.Nifti1Image(np_data, ni_ground_truth.affine, ni_ground_truth.header)
        nib.save(new_nifti, output_path)

def find_gt_file(gt_path):
    gt_paths = glob.glob(gt_path)
    if not gt_paths:
        print("ground truth not found: " + gt_path)
        return None
    elif len(gt_paths) > 1:
        print("multiple groundtruths found: " + gt_paths)
        return None
    else:
        return gt_paths[0]


@hydra.main(version_base=None, config_path='..', config_name='ifnet_config')
def main(cfg: DictConfig):
    cfg_convert_to_nifti = cfg.convert_to_nifti
    res = cfg.general.resolution

    if cfg_convert_to_nifti.generate_all:
        inputs_regex = cfg_convert_to_nifti.input + '/*/evaluation*/generation'
        inputs = glob.glob(inputs_regex)
    else:
        inputs = [cfg_convert_to_nifti.input]
    
    for input in inputs:
        print('-------------------------------------')
        print(input)
        print('-------------------------------------')
        
        if cfg_convert_to_nifti.generate_all:
            output = input + '/nifties'
            if os.path.exists(output):
                print("Nifties already created.")
                continue
        else:
            output = cfg_convert_to_nifti.output
        for folder in os.listdir(input):
            if folder == 'nifties':
                continue
            sub_verse = os.path.join(input, folder)
            sub_verse_output = os.path.join(output, folder)
            out_dir_exists = os.path.exists(sub_verse_output)
            if not out_dir_exists:
                os.makedirs(sub_verse_output)
            sub_verse_gt = os.path.join(cfg_convert_to_nifti.ground_truth, '*/')
            sub_verse_gt = os.path.join(sub_verse_gt, folder)
            if os.path.isdir(sub_verse):
                for filename in os.listdir(sub_verse):
                    crop = os.path.join(sub_verse, filename)
                    crop_output = os.path.join(sub_verse_output, filename)
                    out_dir_exists = os.path.exists(crop_output)
                    if not out_dir_exists:
                        os.makedirs(crop_output)
                    if cfg_convert_to_nifti.is_prediction:
                        filename = filename.replace("aligned-false", "prediction-lowres_aligned-true")
                    crop_gt = os.path.join(sub_verse_gt, "%s.nii.gz" % filename)
                    if os.path.isdir(crop):
                        voxel_input = os.path.join(crop, "voxelization_{}.npy".format(res))
                        nifti_output = os.path.join(crop_output, "surface_reconstruction.nii.gz")
                        crop_gt = find_gt_file(crop_gt)
                        if crop_gt and os.path.exists(voxel_input):
                            convert_to_nifti(cfg.general.resolution, voxel_input, nifti_output, crop_gt)


if __name__ == '__main__':
    main()

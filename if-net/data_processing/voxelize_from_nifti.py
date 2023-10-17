import os

import hydra
import numpy as np
import nibabel as nib
from pathlib import Path

from omegaconf import DictConfig


def voxelize_from_nifti(input_path, output_path):
        try:
                filename = os.path.join(output_path, 'voxelization_128.npy')

                if os.path.exists(filename):
                        return
                
                img = nib.load(input_path)
                occupancies = np.array(img.dataobj)
                print(occupancies.shape)
                if not occupancies.any():
                        raise ValueError('No empty voxel grids allowed.')

                occupancies = np.packbits(occupancies)
                print(occupancies.shape)
                np.save(filename, occupancies)

        except Exception as err:
                path = os.path.normpath(input_path)
                print('Error with {}: {}'.format(path, traceback.format_exc()))
        print('finished {}'.format(input_path))


@hydra.main(version_base=None, config_path='..', config_name='ifnet_config')
def main(cfg: DictConfig):
    input_path = Path(cfg.voxelize_from_nifti.input)
    output_path = Path(cfg.voxelize_from_nifti.output)
    out_dir_exists = os.path.exists(output_path)
    if not out_dir_exists:
        os.makedirs(output_path)
        print("Directory %s created!" % output_path)

    for folder in os.listdir(input_path):
        sub_verse = os.path.join(input_path, folder)
        if os.path.isdir(sub_verse):
            sub_verse_dir = os.path.join(output_path, folder)
            exists = os.path.exists(sub_verse_dir)
            if not exists:
                os.makedirs(sub_verse_dir)
                print("Directory %s created!" % sub_verse_dir)
            for filename in os.listdir(sub_verse):
                crop = os.path.join(sub_verse, filename)
                if os.path.isfile(crop):
                    crop_dir_name = os.path.splitext(filename)[0]
                    crop_dir_name = os.path.splitext(crop_dir_name)[0]
                    crop_dir = os.path.join(sub_verse_dir, crop_dir_name)
                    exists = os.path.exists(crop_dir)
                    if not exists:
                        os.makedirs(crop_dir)
                        print("Directory %s created!" % crop_dir)
                    voxelize_from_nifti(crop, crop_dir)


if __name__ == '__main__':
    main()

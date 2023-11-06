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
                occupancies = np.array(img.dataobj).astype(int)
                if not occupancies.any():
                        raise ValueError('No empty voxel grids allowed.')

                occupancies = np.packbits(occupancies)
                print(occupancies.shape)
                np.save(filename, occupancies)

        except Exception as err:
                path = os.path.normpath(input_path)
                print('Error with {}: {}'.format(path, err))
        print('finished {}'.format(input_path))


@hydra.main(version_base=None, config_path='..', config_name='ifnet_config')
def main(cfg: DictConfig):
    input_path = Path(cfg.voxelize_from_nifti.input)
    output_path = Path(cfg.voxelize_from_nifti.output)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if file.endswith(".nii.gz"):
                file_path = os.path.join(root, file)
                new_root = file_path.replace(str(input_path), str(output_path))
                new_root, extension = os.path.splitext(new_root)
                new_root, extension2 = os.path.splitext(new_root)
                os.makedirs(new_root, exist_ok=True)
                voxelize_from_nifti(file_path, new_root)


if __name__ == '__main__':
    main()

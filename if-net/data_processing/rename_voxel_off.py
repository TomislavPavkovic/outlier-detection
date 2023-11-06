import os
import numpy as np
import nibabel as nib
from pathlib import Path
import hydra

from omegaconf import DictConfig

@hydra.main(version_base=None, config_path='..', config_name='ifnet_config')
def main(cfg: DictConfig):
    input_path=Path(cfg.rename_voxel_off.root)
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if file.endswith('voxelization_128.off'):
                file_path = os.path.join(root, file)
                new_file_path = file_path.replace('voxelization_128.off', 'isosurf_scaled.off')
                os.rename(file_path, new_file_path)

if __name__=="__main__":
    main()
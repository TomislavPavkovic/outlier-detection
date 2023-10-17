import os
import numpy as np
import nibabel as nib
from pathlib import Path
import hydra

from omegaconf import DictConfig

@hydra.main(version_base=None, config_path='..', config_name='ifnet_config')
def main(cfg: DictConfig):
    input_path=Path(cfg.rename_voxel_off.root)
    for folder in os.listdir(input_path):
        sub_verse = os.path.join(input_path, folder)
        if os.path.isdir(sub_verse):
            for filename in os.listdir(sub_verse):
                crop = os.path.join(sub_verse, filename)
                if os.path.isdir(crop):
                    voxel_off = os.path.join(crop, "voxelization_128.off")
                    isosurf_scaled = os.path.join(crop, "isosurf_scaled.off")
                    os.rename(voxel_off, isosurf_scaled)

if __name__=="__main__":
    main()
import os
import numpy as np
from data_augmentation import Deform_with_perlin_noise
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path='..', config_name='ifnet_config')
def main(cfg: DictConfig):
    root_directory = cfg.deform_input.root
    threshold_min = cfg.deform_input.threshold_min
    threshold_max = cfg.deform_input.threshold_max
    threshold_avg = (threshold_min + threshold_max) / 2
    res = cfg.general.resolution
    
    aug1 = Deform_with_perlin_noise((threshold_min, threshold_max), '+', padding=10, hills=3)
    aug2 = Deform_with_perlin_noise((threshold_min, threshold_max), '-', padding=2, hills=3)
    aug1_fixed = Deform_with_perlin_noise((threshold_avg, threshold_avg + 0.001), '+', padding=10, hills=3)
    aug2_fixed = Deform_with_perlin_noise((threshold_avg, threshold_avg + 0.001), '-', padding=2, hills=3)
    
    for root, dirs, files in os.walk(root_directory):
        for file in files:
            if file.endswith('voxelization_{}.npy'.format(res)):
                file_path = os.path.join(root, file)
                new_path = os.path.join(root, 'voxelization_{}_def_p_sized.npy'.format(res))
                new_path_fixed = os.path.join(root, 'voxelization_{}_def_fixed.npy'.format(res))
                ct_scan = np.load(file_path)
                occupancies = np.unpackbits(ct_scan)
                input = np.reshape(occupancies, (res,)*3)
                # Deform the cube using Perlin noise
                print(file_path)
                deformed_grid = aug1(np.copy(input))
                deformed_grid = aug2(np.copy(deformed_grid)).astype('uint8')
                
                occupancies = np.packbits(deformed_grid)
                np.save(new_path, occupancies)

                deformed_grid_fixed = aug1_fixed(np.copy(input))
                deformed_grid_fixed = aug2_fixed(np.copy(deformed_grid_fixed)).astype('uint8')

                occupancies_fixed = np.packbits(deformed_grid_fixed)
                np.save(new_path_fixed, occupancies_fixed)
                
if __name__ == '__main__':
    main()
import os
import numpy as np
from data_augmentation import Deform_with_perlin_noise

root_directory = "/home/tomislav/datasets-ct-sized/dataset-CACTS/right kidney-ifnet"
aug1 = Deform_with_perlin_noise((0.75, 1), '+', padding=10, hills=3)
aug2 = Deform_with_perlin_noise((0.75, 1), '-', padding=2, hills=3)
for root, dirs, files in os.walk(root_directory):
    for file in files:
        if file.endswith('voxelization_128.npy'):
            file_path = os.path.join(root, file)
            new_path = os.path.join(root, 'voxelization_128_def_p_sized.npy')
            ct_scan = np.load(file_path)
            occupancies = np.unpackbits(ct_scan)
            input = np.reshape(occupancies, (128,)*3)
            # Deform the cube using Perlin noise
            print(new_path)
            deformed_grid = aug1(np.copy(input))
            deformed_grid = aug2(np.copy(deformed_grid)).astype('uint8')
            
            occupancies = np.packbits(deformed_grid)
            np.save(new_path, occupancies)
            
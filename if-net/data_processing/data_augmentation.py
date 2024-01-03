import numpy as np
import noise
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from data_processing.perlin import rand_perlin_3d_mask

class Delete_random_patch():
    def __init__(self, min_patch_size, max_patch_size):
        self.min_patch_size = min_patch_size
        self.max_patch_size = max_patch_size

    def __call__(self, input):
        image_data = (np.rint(input)).astype(int)
        nonzero_voxels = np.array(np.where(image_data > 0))
        min_coords = np.min(nonzero_voxels, axis=1)
        max_coords = np.max(nonzero_voxels, axis=1)
        shape_size = max_coords - min_coords
        # Get the shape of the image data
        image_shape = image_data.shape
        
        patch_size = [np.random.randint(self.min_patch_size, self.max_patch_size),]*3

        for i in range(0, 3):
            if patch_size[i] > shape_size[i]:
                patch_size[i] = shape_size[i]
        # Generate random indices for the patch
        random_indices = [np.random.randint(min_c, max_c - patch_size + 1) for min_c, max_c, patch_size in zip(min_coords, max_coords, patch_size)]
        
        # Create a mask to zero out the random patch
        mask = np.zeros_like(image_data)
        mask[random_indices[0]:random_indices[0] + patch_size[0],
            random_indices[1]:random_indices[1] + patch_size[1],
            random_indices[2]:random_indices[2] + patch_size[2]] = 1

        # Apply the mask to delete the random patch
        modified_data = image_data * (1 - mask)

        return modified_data

class Add_random_patch():
    def __init__(self, min_patch_size, max_patch_size):
        self.min_patch_size = min_patch_size
        self.max_patch_size = max_patch_size

    def __call__(self, input):
        image_data = (np.rint(input)).astype(int)
        nonzero_voxels = np.array(np.where(image_data > 0))
        min_coords = np.min(nonzero_voxels, axis=1)
        max_coords = np.max(nonzero_voxels, axis=1)
        shape_size = max_coords - min_coords
        # Get the shape of the image data
        image_shape = image_data.shape
        
        patch_size = [np.random.randint(self.min_patch_size, self.max_patch_size),]*3

        for i in range(0, 3):
            if patch_size[i] > shape_size[i]:
                patch_size[i] = shape_size[i]
        # Generate random indices for the patch
        random_indices = [np.random.randint(min_c, max_c - patch_size + 1) for min_c, max_c, patch_size in zip(min_coords, max_coords, patch_size)]
        
        # Create a mask to zero out the random patch
        mask = np.zeros_like(image_data)
        mask[random_indices[0]:random_indices[0] + patch_size[0],
            random_indices[1]:random_indices[1] + patch_size[1],
            random_indices[2]:random_indices[2] + patch_size[2]] = 1

        # Apply the mask to delete the random patch
        modified_data = image_data + mask

        return modified_data
'''
class Deform_with_perlin_noise():
    def __init__(self, min_scale=10.0, max_scale=30.0, octaves=6, persistence=0.5, lacunarity=2.0, seed=None, freq=16):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity
        self.seed = seed
        self.freq = freq

    def __call__(self, obj):
        scale = np.random.uniform(self.min_scale, self.max_scale)
        print(scale)
        shape = obj.shape
        
        for x in range(shape[0]):
            for y in range(shape[1]):
                for z in range(shape[2]):
                    oldv = obj[x][y][z]
                    obj[x][y][z] += noise.pnoise3(x/scale,
                                                y/scale,
                                                z/scale,
                                                octaves=self.octaves,
                                                persistence=self.persistence,
                                                lacunarity=self.lacunarity,
                                                repeatx=self.freq,
                                                repeaty=self.freq,
                                                repeatz=self.freq,
                                                base=self.seed)
                    
        return obj
'''
class Deform_with_perlin_noise():
    def __init__(self, percent_range, mode):
        self.percent_range = percent_range
        self.mode = mode

    def __call__(self, obj):
        if self.mode == '-':
            return obj * (1 - rand_perlin_3d_mask(obj.shape, 4, self.percent_range).numpy())
        else:
            return obj + rand_perlin_3d_mask(obj.shape, 4, self.percent_range).numpy()
    
ct_scan = np.load('/home/tomislav/datasets-ct-sized/dataset-CACTS/right kidney-ifnet/0001_visceral_gc/labels/10000011_right kidney_sized/voxelization_128.npy')
occupancies = np.unpackbits(ct_scan)
input = np.reshape(occupancies, (128,)*3)
# Deform the cube using Perlin noise
#aug = Add_random_patch(29, 30)
aug = Deform_with_perlin_noise((0.8, 0.85), '+')
deformed_grid = aug(np.copy(input))

fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
ax.set_title("Original Object")
ax.voxels(input, edgecolor='k')

ax = fig.add_subplot(122, projection='3d')
ax.set_title("Deformed Object with Perlin Noise")
ax.voxels(deformed_grid, edgecolor='k')

plt.show()

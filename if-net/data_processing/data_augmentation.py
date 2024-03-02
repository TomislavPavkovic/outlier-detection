import numpy as np
import noise
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from perlin import rand_perlin_3d_mask

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

class Deform_with_perlin_noise():
    def __init__(self, percent_range, mode, padding=12, hills=4):
        self.percent_range = percent_range
        self.mode = mode
        self.padding = padding
        self.hills = hills

    def __call__(self, obj):
        nonzero_voxels = np.array(np.where(obj > 0))
        min_coords = np.min(nonzero_voxels, axis=1)
        max_coords = np.max(nonzero_voxels, axis=1)
        shape = max_coords - min_coords
        for i in range(0,3):
            if shape[i] > 128 - self.padding:
                shape[i] = 128
                min_coords[i] = 0
            else:
                shape[i] += self.padding
                min_coords[i] -= self.padding // 2
                if min_coords[i] < 0:
                    min_coords[i] = 0
        shape = shape // self.hills * self.hills
        if (shape == 0).any():
            return obj
        perlin_shape = rand_perlin_3d_mask(shape, self.hills, self.percent_range).numpy()
        expanded_perlin_shape = np.zeros(obj.shape, dtype=obj.dtype)
        expanded_perlin_shape[min_coords[0]:min_coords[0] + shape[0],
                              min_coords[1]:min_coords[1] + shape[1],
                              min_coords[2]:min_coords[2] + shape[2]] = perlin_shape
        if self.mode == '-':
            return obj * (1 - expanded_perlin_shape)
        else:
            return obj + expanded_perlin_shape


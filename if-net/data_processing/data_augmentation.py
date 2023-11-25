import numpy as np

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

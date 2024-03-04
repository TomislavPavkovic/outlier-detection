import os
from pathlib import Path

input_path = Path('/data1/practical-sose23/dataset-verse19/prediction-lowres_crops-aligned_voxels-128_zoom-1-1-1-ifnet/')

for folder in os.listdir(input_path):
    sub_verse = os.path.join(input_path, folder)
    if os.path.isdir(sub_verse):
        for filename in os.listdir(sub_verse):
            new_filename = filename.replace("prediction-lowres_aligned-true", "aligned-false")
            crop = os.path.join(sub_verse, filename)
            crop_new = os.path.join(sub_verse, new_filename)
            os.rename(crop, crop_new)

                
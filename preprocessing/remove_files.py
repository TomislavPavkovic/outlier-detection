import os
root_directory = "/home/tomislav/datasets-ct/dataset-CACTS"
for root, dirs, files in os.walk(root_directory):
    for file in files:
        if file.endswith('_liver.nii.gz'):
            file_path = os.path.join(root, file)
            os.remove(file_path)
            print(file_path)
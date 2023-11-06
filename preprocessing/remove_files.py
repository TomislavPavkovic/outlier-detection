import os
root_directory = "/home/tomislav/datasets-ct-sized/dataset-CACTS/right kidney-ifnet"
for root, dirs, files in os.walk(root_directory):
    for file in files:
        if file.endswith('voxelization_128.off'):
            file_path = os.path.join(root, file)
            os.remove(file_path)
            print(file_path)
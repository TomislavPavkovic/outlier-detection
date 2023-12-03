import os
root_directory = "/home/tomislav/datasets-ct-sized/dataset-CACTS/right kidney-ifnet"
counter = 0
for root, dirs, files in os.walk(root_directory):
    for file in files:
        if file.endswith('.off'):
            counter += 1

print(counter)
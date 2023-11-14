import numpy as np
import glob
from sklearn.model_selection import train_test_split
import re
import argparse

# Example split.npz can be viewed here

# data = np.load('/u/home/cev/adlm/shape-prior/if-net/shapenet/split.npz', allow_pickle=True)
# print(data.files['train'][0])
# example input point in train set: /04401088/1f565ab552dc89727e51366b0cf77473

# Read paths to the folders
if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description='Run conversion to nifti'
    )
    parser.add_argument('-root', type=str)
    parser.add_argument('-target', type=str)

    args = parser.parse_args()
    root = args.root
    target = args.target

    data_folders = glob.glob(root+'/*/*/*/', recursive = True)
    #data_folders = [i.replace(root, '').rstrip('/') for i in data_folders] 
    data_folders = [folder.replace(root, '').rstrip('/') for folder in data_folders]

    # Create train, test, val splits
    X_train_objs, X_test_objs = train_test_split(data_folders, test_size=0.20, random_state=42, shuffle=True)
    X_val_objs, X_test_objs = train_test_split(X_test_objs, test_size=0.50, random_state=42)

    ## Save split.npz file
    np.savez_compressed(target, train=np.array(X_train_objs), test=np.array(X_test_objs), val=np.array(X_val_objs))
    data = np.load(target, allow_pickle=True)
    print('Train: ', data['train'][0],len(data['train']))
    print('Test: ',data['test'][0],len(data['test']))
    print('Val: ',data['val'][0],len(data['val']))
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
    parser.add_argument('-region', type=str, help="cervical, thoracic, lumbar, all", default="all")

    args = parser.parse_args()
    root = args.root
    target = args.target
    region = args.region
    verse_folders = glob.glob(root+'/*', recursive = True)
    unique_subjects = list(set([i.split('/')[-1] for i in verse_folders]))
    # collect test subject samples
    data_folders = glob.glob(root+'/*/*/', recursive = True)
    #data_folders = [i.replace(root, '').rstrip('/') for i in data_folders] 

    # Create train, test, val splits
    X_train_objs, X_test_objs = train_test_split(unique_subjects, test_size=0.20, random_state=42, shuffle=True)
    X_val_objs, X_test_objs = train_test_split(X_test_objs, test_size=0.50, random_state=42)

    X_train , X_test, X_val = [], [], []
    # collect train subject samples
    #regex for cervical: "_label-[1-7]_", thoracic: "_label-([8-9]|1[0-9])_", lumbar: "_label-2[0-4]_"
    if region == "all":
        regex = ".*"
    elif region == "cervical":
        regex = "_label-[1-7]_"
    elif region == thoracic:
        regex = "_label-([8-9]|1[0-9])_"
    elif region == lumbar:
        regex = "_label-2[0-4]_"
    else:
        print("Unknown spinal region")
        exit(0)
    
    for i in X_train_objs:
        for j in data_folders:
            if i in j and re.search(regex, j):
                X_train.append(j.replace(root, '').rstrip('/'))

    # collect test subject samples
    for i in X_test_objs:
        for j in data_folders:
            if i in j and re.search(regex, j):
                X_test.append(j.replace(root, '').rstrip('/'))

    # collect val subject samples
    for i in X_val_objs:
        for j in data_folders:
            if i in j and re.search(regex, j):
                X_val.append(j.replace(root, '').rstrip('/'))


    ## Save split.npz file
    np.savez_compressed(target, train=np.array(X_train), test=np.array(X_test), val=np.array(X_val))
    data = np.load(target, allow_pickle=True)
    print('Train: ', data['train'][0],len(data['train']))
    print('Test: ',data['test'][0],len(data['test']))
    print('Val: ',data['val'][0],len(data['val']))
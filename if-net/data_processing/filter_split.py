import numpy as np
import os

if __name__=="__main__":
    split_path = '/home/tomislav/datasets-ct-sized/dataset-CACTS/right kidney-ifnet/split.npz'
    root = '/home/tomislav/datasets-ct-sized-clean/dataset-CACTS/right kidney-ifnet'
    out_path = '/home/tomislav/outlier-detection/clean_split.npz'
    split = np.load(split_path)
    train_set = split['train']
    val_set = split['val']
    test_set = split['test']
    new_train_set = []
    new_val_set = []
    new_test_set = []
    data = np.load(split_path, allow_pickle=True)
    print('Train: ', data['train'][0],len(data['train']))
    print('Test: ',data['test'][0],len(data['test']))
    print('Val: ',data['val'][0],len(data['val']))

    for item in train_set:
        path = os.path.join(root + item, 'voxelization_128_def_p_sized.npy')
        if os.path.exists(path):
            new_train_set.append(item)
    for item in val_set:
        path = os.path.join(root + item, 'voxelization_128_def_p_sized.npy')
        if os.path.exists(path):
            new_val_set.append(item)
    for item in test_set:
        path = os.path.join(root + item, 'voxelization_128_def_p_sized.npy')
        if os.path.exists(path):
            new_test_set.append(item)

    np.savez_compressed(out_path, train=np.array(new_train_set), test=np.array(new_test_set), val=np.array(new_val_set))
    data = np.load(out_path, allow_pickle=True)
    print('Train: ', data['train'][0],len(data['train']))
    print('Test: ',data['test'][0],len(data['test']))
    print('Val: ',data['val'][0],len(data['val']))
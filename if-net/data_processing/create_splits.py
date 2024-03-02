import numpy as np
import glob
from sklearn.model_selection import train_test_split
import re
import hydra
from omegaconf import DictConfig

# Read paths to the folders
@hydra.main(version_base=None, config_path='..', config_name='ifnet_config')
def main(cfg: DictConfig):
    root = cfg.create_splits.root
    target = cfg.create_splits.target
    splits_ratio = cfg.create_splits.splits_ratio

    data_folders = glob.glob(root+'/*/*/*/', recursive = True)
    #data_folders = [i.replace(root, '').rstrip('/') for i in data_folders] 
    data_folders = [folder.replace(root, '').rstrip('/') for folder in data_folders]

    # Create train, test, val splits
    test_total = splits_ratio[1]+splits_ratio[2]
    X_train_objs, X_test_objs = train_test_split(data_folders, test_size=test_total, random_state=42, shuffle=True)
    test_ratio = splits_ratio[2] / test_total
    X_val_objs, X_test_objs = train_test_split(X_test_objs, test_size=test_ratio, random_state=42)

    ## Save split.npz file
    np.savez_compressed(target, train=np.array(X_train_objs), test=np.array(X_test_objs), val=np.array(X_val_objs))
    data = np.load(target, allow_pickle=True)
    print('Train: ', len(data['train']))
    print('Test: ', len(data['test']))
    print('Val: ', len(data['val']))

if __name__=="__main__":
    main()
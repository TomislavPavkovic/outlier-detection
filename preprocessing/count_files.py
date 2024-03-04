import os
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path='.', config_name='preprocessing_config')
def main(cfg: DictConfig):
    root_directory = cfg.count_files.root
    file_ending = cfg.count_files.file_ending
    counter = 0
    for root, dirs, files in os.walk(root_directory):
        for file in files:
            if file.endswith(file_ending):
                counter += 1

    print(counter)

if __name__ == '__main__':
    main()
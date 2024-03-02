import os
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path='.', config_name='preprocessing_config')
def main(cfg: DictConfig):
    root_directory = cfg.remove_files.root_directory
    file_ending = cfg.remove_files.file_ending

    for root, dirs, files in os.walk(root_directory):
        for file in files:
            if file.endswith(file_ending):
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(file_path)

if __name__ == '__main__':
    main()
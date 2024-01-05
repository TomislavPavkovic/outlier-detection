import hydra
import torch
from omegaconf import DictConfig
import glob
import numpy as np

import models.local_model as model
import models.data.voxelized_data_shapenet as voxelized_data
from models.generation import Generator
from models import training
from generation_iterator import gen_iterator
from generation_iterator_with_training import gen_iterator_training


@hydra.main(version_base=None, config_path='.', config_name='ifnet_config')
def main(cfg: DictConfig):
    cfg_general = cfg.general
    cfg_generate = cfg.generate
    if cfg_general.model == 'ShapeNet32Vox':
        net = model.ShapeNet32Vox()
    elif cfg_general.model == 'ShapeNet128Vox':
        net = model.ShapeNet128Vox()
    elif cfg_general.model == 'ShapeNetPoints':
        net = model.ShapeNetPoints()
    elif cfg_general.model == 'SVR':
        net = model.SVR()

    dataset = voxelized_data.VoxelizedDataset(
        cfg_generate.mode,
        data_path=cfg_general.data_path,
        split_file=cfg_general.split_file,
        voxelized_pointcloud=cfg_general.pointcloud,
        pointcloud_samples=cfg_general.pointcloud_samples,
        res=cfg_general.resolution,
        sample_distribution=cfg_general.sample_distribution,
        sample_sigmas=cfg_general.sample_sigmas,
        num_sample_points=100,
        batch_size=1,
        num_workers=0
    )

    exp_name = 'i{}_dist-{}sigmas-{}v{}_m{}'.format(
        'PC' + str(cfg_general.pointcloud_samples) if cfg_general.pointcloud else 'Voxels',
        ''.join(str(e) + '_' for e in cfg_general.sample_distribution),
        ''.join(str(e) + '_' for e in cfg_general.sample_sigmas),
        cfg_general.resolution,
        cfg_general.model
    )
    if cfg_generate.generate_all:
        exp_name_regex = './experiments/' + exp_name + '*'
        exp_names = glob.glob(exp_name_regex)
        exp_names = [name.split('/')[-1] for name in exp_names]
    else:
        exp_names = [exp_name]
    
    for exp_name in exp_names:
        print('-------------------------------------')
        print(exp_name)
        print('-------------------------------------')
        trainer = training.Trainer(net, torch.device("cuda"), None, None, exp_name, optimizer='Adam')
        val_mins = glob.glob('./experiments/' + exp_name + '/val_min=*.npy')
        if cfg_generate.get('checkpoint') is not None:
            checkpoint_value = cfg_generate.checkpoint
        else:
            if len(val_mins) != 1:
                print('Could not find minimum validation value!')
                continue
            else:
                checkpoint_value = int(np.load(val_mins[0])[0])
                print('Checkpoint value:', checkpoint_value)
        gen = Generator(
            net,
            0.5,
            exp_name,
            checkpoint=checkpoint_value,
            resolution=cfg_generate.retrieval_resolution,
            batch_points=cfg_generate.batch_points,
            trainer=trainer
        )

        if not cfg_generate.training_during_inference:
            out_path = 'experiments/{}/evaluation_{}_@{}/'.format(exp_name, checkpoint_value, cfg_generate.retrieval_resolution)
            gen_iterator(out_path, dataset, gen)
        else:
            out_path = 'experiments/{}/evaluation_{}_@{}_unoptimized/'.format(exp_name, checkpoint_value, cfg_generate.retrieval_resolution)
            out_path_optimized = 'experiments/{}/evaluation_{}_@{}_optimized/'.format(exp_name, checkpoint_value, cfg_generate.retrieval_resolution)
            gen_iterator_training(out_path, out_path_optimized, dataset, gen)


if __name__ == '__main__':
    main()

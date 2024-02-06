import hydra
from omegaconf import DictConfig

import models.local_model as model
import models.data.voxelized_data_shapenet as voxelized_data
from models import training
import torch
#from data_processing.data_augmentation import Delete_random_patch, Deform_with_perlin_noise, Add_random_patch

@hydra.main(version_base=None, config_path='.', config_name='ifnet_config')
def main(cfg: DictConfig):
    cfg_general = cfg.general
    cfg_train = cfg.train
    if cfg_general.model == 'ShapeNet32Vox':
        net = model.ShapeNet32Vox()
    elif cfg_general.model == 'ShapeNet128Vox':
        net = model.ShapeNet128Vox(dropout_p=cfg_train.dropout_probability)
    elif cfg_general.model == 'ShapeNetPoints':
        net = model.ShapeNetPoints()
    elif cfg_general.model == 'SVR':
        net = model.SVR()

    train_dataset = voxelized_data.VoxelizedDataset(
        'train',
        data_path=cfg_general.data_path,
        split_file=cfg_general.split_file,
        voxelized_pointcloud=cfg_general.pointcloud,
        pointcloud_samples=cfg_general.pointcloud_samples,
        res=cfg_general.resolution,
        sample_distribution=cfg_general.sample_distribution,
        sample_sigmas=cfg_general.sample_sigmas,
        num_sample_points=50000,
        batch_size=cfg_train.batch_size,
        num_workers=30,
        augmented_extension='_def_p_sized'
        #transforms=[Add_random_patch(cfg_train.deleted_patch_min_size, cfg_train.deleted_patch_max_size),
        #           Delete_random_patch(cfg_train.deleted_patch_min_size, cfg_train.deleted_patch_max_size)]
    )

    val_dataset = voxelized_data.VoxelizedDataset(
        'val',
        data_path=cfg_general.data_path,
        split_file=cfg_general.split_file,
        voxelized_pointcloud=cfg_general.pointcloud,
        pointcloud_samples=cfg_general.pointcloud_samples,
        res=cfg_general.resolution,
        sample_distribution=cfg_general.sample_distribution,
        sample_sigmas=cfg_general.sample_sigmas,
        num_sample_points=50000,
        batch_size=cfg_train.batch_size,
        num_workers=30,
        augmented_extension='_def_fixed'
    )

    exp_name = 'i{}_dist-{}sigmas-{}v{}_m{}'.format(
        'PC' + str(cfg_general.pointcloud_samples) if cfg_general.pointcloud else 'Voxels',
        ''.join(str(e) + '_' for e in cfg_general.sample_distribution),
        ''.join(str(e) + '_' for e in cfg_general.sample_sigmas),
        cfg_general.resolution,
        cfg_general.model
    )

    trainer = training.Trainer(
        net,
        torch.device("cuda"),
        train_dataset,
        val_dataset,
        exp_name,
        optimizer=cfg_train.optimizer,
        adam_weight_decay=cfg_train.adam_weight_decay
    )

    trainer.evaluate_all_checkpoints(cfg_train.epochs)


if __name__ == '__main__':
    main()

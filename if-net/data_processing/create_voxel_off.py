from functools import partial

import hydra
from omegaconf import DictConfig

from voxels import VoxelGrid
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
import glob


def create_voxel_off(path, unpackbits, resolution, min, max):
    voxel_path = path + '/voxelization_{}.npy'.format(resolution)
    off_path = path + '/voxelization_{}.off'.format(resolution)

    if unpackbits:
        occ = np.unpackbits(np.load(voxel_path))
        voxels = np.reshape(occ, (resolution,)*3)
    else:
        voxels = np.reshape(np.load(voxel_path)['occupancies'], (resolution,)*3)

    loc = ((min+max)/2, )*3
    scale = max - min

    VoxelGrid(voxels, loc, scale).to_mesh().export(off_path)
    print('Finished: {}'.format(path))


@hydra.main(version_base=None, config_path='..', config_name='ifnet_config')
def main(cfg: DictConfig):
    cfg_create_voxel_off = cfg.create_voxel_off

    p = Pool(mp.cpu_count())
    p.map(
        partial(
            create_voxel_off,
            unpackbits=cfg_create_voxel_off.unpackbits,
            resolution=cfg.general.resolution,
            min=cfg_create_voxel_off.min,
            max=cfg_create_voxel_off.max
        ),
        glob.glob(cfg_create_voxel_off.root + '/*/*/*/')
    )


if __name__ == '__main__':
    main()

import hydra
import trimesh
import numpy as np
import os
import glob
import multiprocessing as mp
from multiprocessing import Pool
from functools import partial
import traceback

from omegaconf import DictConfig

import voxels
from pathlib import Path

def voxelize(in_path, res, input_filename):
    try:

        filename = os.path.join(in_path, 'voxelization_{}.npy'.format(res))

        if os.path.exists(filename):
            return
        mesh = trimesh.load(Path(in_path + input_filename), process=False)
        occupancies = voxels.VoxelGrid.from_mesh(mesh, res, loc=[0, 0, 0], scale=1).data
        occupancies = np.reshape(occupancies, -1)

        if not occupancies.any():
            raise ValueError('No empty voxel grids allowed.')

        occupancies = np.packbits(occupancies)
        np.save(filename, occupancies)

    except Exception as err:
        path = os.path.normpath(in_path)
        print('Error with {}: {}'.format(path, traceback.format_exc()))
    print('finished {}'.format(in_path))


@hydra.main(version_base=None, config_path='..', config_name='ifnet_config')
def main(cfg: DictConfig):
    p = Pool(mp.cpu_count())
    p.map(
        partial(
            voxelize,
            res=cfg.general.resolution,
            input_filename=cfg.voxelize.filename
        ),
        glob.glob(cfg.voxelize.root + '/*/*/')
    )


if __name__ == '__main__':
    main()

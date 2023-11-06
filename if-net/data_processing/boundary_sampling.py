from functools import partial

import hydra
import trimesh
import numpy as np
from omegaconf import DictConfig

import implicit_waterproofing as iw
import glob
import multiprocessing as mp
from multiprocessing import Pool
import os
import traceback


def boundary_sampling(path, sigma, sample_num):
    try:

        if os.path.exists(path +'/boundary_{}_samples.npz'.format(sigma)):
            return

        off_path = path + '/isosurf_scaled.off'
        out_file = path +'/boundary_{}_samples.npz'.format(sigma)

        mesh = trimesh.load(off_path)
        points = mesh.sample(sample_num)

        boundary_points = points + sigma * np.random.randn(sample_num, 3)
        grid_coords = boundary_points.copy()
        grid_coords[:, 0], grid_coords[:, 2] = boundary_points[:, 2], boundary_points[:, 0]

        grid_coords = 2 * grid_coords

        occupancies = iw.implicit_waterproofing(mesh, boundary_points)[0]

        np.savez(out_file, points=boundary_points, occupancies = occupancies, grid_coords= grid_coords)
        print('Finished {}'.format(path))
    except:
        print('Error with {}: {}'.format(path, traceback.format_exc()))


@hydra.main(version_base=None, config_path='..', config_name='ifnet_config')
def main(cfg: DictConfig):
    cfg_boundary_sampling = cfg.boundary_sampling

    for sigma in cfg_boundary_sampling.sigmas:
        p = Pool(mp.cpu_count())
        p.map(
            partial(
                boundary_sampling,
                sigma=sigma,
                sample_num=cfg_boundary_sampling.sample_number
            ),
            glob.glob(cfg_boundary_sampling.root + '/*/*/*/')
        )


if __name__ == '__main__':
    main()

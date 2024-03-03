import os
import sys
sys.path.append('..')
import shutil
import numpy as np
import hydra
import torch
from omegaconf import DictConfig
import glob
import nibabel as nib
import csv

from preprocessing.extract_label import extract_label
from preprocessing.map_to_binary import class_map
from preprocessing.adapt_size import adapt_size
import models.local_model as model
import models.data.voxelized_data_shapenet as voxelized_data
from models.generation import Generator
from models import training
from generation_iterator import gen_iterator

from data_processing.voxelize_from_nifti import voxelize_from_nifti
from data_processing.create_voxel_off import create_voxel_off
from data_processing.boundary_sampling import boundary_sampling
from data_processing.voxelize import voxelize
from data_processing.convert_to_nifti import convert_to_nifti

from evaluation.utils.surface_distance import metrics

@hydra.main(version_base=None, config_path='.', config_name='ifnet_config')
def main(cfg: DictConfig):
    cfg_general = cfg.general
    cfg_generate = cfg.generate
    cfg_analyse = cfg.analyse

    if os.path.isfile(cfg_analyse.input):
        inputs = [cfg_analyse.input]
    elif os.path.isdir(cfg_analyse.input):
        inputs = glob.glob(cfg_analyse.input + '/**/*_combined.nii.gz', recursive=True)

    output_file = open(cfg_analyse.output_path, 'w')
    writer = csv.writer(output_file)
    writer.writerow(['file path', 'organ', 'metric', 'score'])
    log_file = open(cfg_analyse.log_path, 'w')
    log_writer = csv.writer(log_file)
    log_writer.writerow(['file path', 'processing step', 'organ', 'error'])

    for input in inputs:
        base, extension = os.path.splitext(input)
        new_folder_name, extension2 = os.path.splitext(base)

        for label in os.listdir('./experiments/'):
            try:
                if not label.isdigit():
                    continue
                label = int(label)
                print(label)
                new_path = f"{new_folder_name}/{label}{extension2}{extension}"
                label_subfolder = os.path.join(new_folder_name, str(label))
                extract_label(label=label, file_path=input, new_file_path=new_path)
                if not os.path.exists(new_path):
                    continue
                if adapt_size(res=cfg_general.resolution, file_path=new_path, new_file_path=new_path) is not None:
                    os.remove(new_path)
                    continue
                voxelize_from_nifti(input_path=new_path, output_path=label_subfolder)
                create_voxel_off(path=label_subfolder, 
                                unpackbits=cfg.create_voxel_off.unpackbits,
                                resolution=cfg_general.resolution,
                                min=cfg.create_voxel_off.min,
                                max=cfg.create_voxel_off.max)
                os.rename(label_subfolder + '/voxelization_{}.off'.format(cfg_general.resolution), 
                        label_subfolder + '/isosurf_scaled.off')
                for sigma in cfg.boundary_sampling.sigmas:
                    boundary_sampling(path=label_subfolder, sigma=sigma, sample_num= cfg.boundary_sampling.sample_number)
            except Exception as error:
                print('Error during data preprocessing on file: {}: {}'.format(input, error))
                log_writer.writerow([input, 'data preprocessing', class_map.get(label), error])
                label_subfolder = os.path.join(new_folder_name, str(label))
                shutil.rmtree(label_subfolder)
        
        if not os.path.isdir(new_folder_name):
            continue

        for organ_scan in os.listdir(new_folder_name):
            try:
                if not organ_scan.isdigit():
                    continue
                organ_scan = int(organ_scan)
                print(organ_scan)
                net = model.ShapeNetPoints()
                dataset = voxelized_data.VoxelizedDataset(
                    cfg_generate.mode,
                    data_path=os.path.join(new_folder_name, str(organ_scan)),
                    voxelized_pointcloud=cfg_general.pointcloud,
                    pointcloud_samples=cfg_general.pointcloud_samples,
                    res=cfg_general.resolution,
                    sample_distribution=cfg_general.sample_distribution,
                    sample_sigmas=cfg_general.sample_sigmas,
                    num_sample_points=100,
                    batch_size=1,
                    num_workers=0,
                    single_image=True,
                    augmented=False
                )

                exp_name = str(organ_scan)
                trainer = training.Trainer(net, torch.device("cuda"), None, None, exp_name, optimizer='Adam')
                
                val_mins = glob.glob('./experiments/' + exp_name + '/val_min=*.npy')
                if cfg_analyse.get('checkpoint') is not None:
                    checkpoint_value = cfg_analyse.checkpoint
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

                out_path = f"{new_folder_name}/{organ_scan}_prediction"
                input_label = f"{new_folder_name}/{organ_scan}.nii.gz"
                gen_iterator(out_path, dataset, gen, analysis=True)
                voxelize(in_path=out_path, res=cfg_general.resolution, input_filename=cfg.voxelize.filename)
                convert_to_nifti(res=cfg_general.resolution,
                                input_path=out_path + '/voxelization_{}.npy'.format(cfg_general.resolution),
                                output_path=out_path + '/surface_reconstruction.nii.gz',
                                ground_truth=input_label)
            except Exception as error:
                print('Error during organ reconstruction on file: {}: {}'.format(input, error))
                log_writer.writerow([input, 'organ reconstruction', class_map.get(organ_scan), error])
                continue

            try:
                label_gt_path = f"{new_folder_name}/{organ_scan}.nii.gz"
                label_pred_path = out_path + '/surface_reconstruction.nii.gz'
                label_gt = nib.load(label_gt_path)
                label_pred = nib.load(label_pred_path)
                spacing = nib.affines.voxel_sizes(label_pred.affine)
                label_pred = np.array(label_pred.get_fdata(), dtype=bool)
                label_gt = np.array(label_gt.get_fdata(), dtype=bool)
                dice_score = metrics.compute_dice_coefficient(label_gt, label_pred)
                surf_distances = metrics.compute_surface_distances(label_gt, label_pred, spacing, True)
                avg_distance_gt_to_pred, avg_distance_pred_to_gt = \
                    metrics.compute_average_surface_distance(surf_distances)
                asd = (avg_distance_gt_to_pred + avg_distance_pred_to_gt) / 2
                max_dist = metrics.compute_robust_hausdorff(surf_distances, 100)
                hausdorff95 = metrics.compute_robust_hausdorff(surf_distances, 95)
                print(dice_score, asd, hausdorff95, max_dist)

                score_types = ['dice score', 'average surface distance', '95% hausdorff distance', 'maximal surface distance']

                treshold_scores = np.load('./experiments/' + exp_name + '/outlier_threshold.npy')
                scores = [dice_score, asd, hausdorff95, max_dist]
                
                for metric in cfg_analyse.metrics:
                    i = score_types.index(metric)
                    if scores[i] <= treshold_scores[i]:
                        print('Outlier detected: {}, organ: {}, score type: {}, score: {}'.format(input, class_map[organ_scan], metric, scores[i]))
                        writer.writerow([input, class_map[organ_scan], metric, scores[i]])
            except Exception as error:
                print('Error during evaluation on file: {}: {}'.format(input, error))
                log_writer.writerow([input, 'evaluation', class_map.get(organ_scan), error])

        if cfg_analyse.delete_reconstructions:
            shutil.rmtree(new_folder_name)
    output_file.close()
    log_file.close()

if __name__ == '__main__':
    main()
import csv
import glob
import os
from typing import List

import hydra
import nibabel as nb
import numpy as np
from omegaconf import DictConfig

from utils.surface_distance import metrics


def eval_single(label_pred: np.ndarray, label_gt: np.ndarray, spacing: np.ndarray,
               dices: List[float], asds: List[float], hd95s: List[float],
               max_distances: List[float]):
    """Evaluate a single volumetric mask."""
    if label_pred.shape != label_gt.shape:
        raise ValueError(f'Batch evaluation not possible: predicted shape {label_pred.shape} '
                         f'is different from GT shape {label_gt.shape}.')

    # Convert integer array to boolean
    label_pred = np.array(label_pred, dtype=bool)
    label_gt = np.array(label_gt, dtype=bool)

    # Empty GT/prediction mess up metrics calculations...
    if label_gt.sum() == 0:
        print('Warning: empty GT occured!')
    if label_pred.sum() == 0:
        print('Warning: empty prediciton occured!')

    # Compute metrics
    dice = metrics.compute_dice_coefficient(label_gt, label_pred)
    surf_distances = metrics.compute_surface_distances(label_gt, label_pred, spacing, True)
    avg_distance_gt_to_pred, avg_distance_pred_to_gt = \
        metrics.compute_average_surface_distance(surf_distances)
    asd = (avg_distance_gt_to_pred + avg_distance_pred_to_gt) / 2
    hausdorff = metrics.compute_robust_hausdorff(surf_distances, 100)
    hausdorff95 = metrics.compute_robust_hausdorff(surf_distances, 95)

    # Append metrics to lists
    dices.append(dice)
    asds.append(asd)
    hd95s.append(hausdorff95)
    max_distances.append(hausdorff)


@hydra.main(version_base=None, config_path='.', config_name='config')
def main(cfg: DictConfig):
    # Load config
    pred_source_dir = cfg.source_dir.pred
    gt_source_dir = cfg.source_dir.gt
    results_file = cfg.results.file
    overwrite_results_file_if_exists = cfg.results.overwrite_if_exists

    if cfg.evaluate_all:
        pred_source_dir_regex = pred_source_dir + '/*/evaluation*/generation/nifties/labels'
        pred_source_dirs = glob.glob(pred_source_dir_regex)
    else:
        pred_source_dirs = [pred_source_dir]
    
    for pred_source_dir in pred_source_dirs:
        print('-------------------------------------')
        print(pred_source_dir)
        print('-------------------------------------')
        
        if cfg.evaluate_all:
            results_file = '/'.join(pred_source_dir.split('/')[:-3]) + '/scores.csv'

        if os.path.exists(results_file):
            continue

        casenames: List[str] = []
        dices: List[float] = []
        asds: List[float] = []
        hd95s: List[float] = []
        max_distances: List[float] = []

        for dirpath, dirnames, filenames in os.walk(pred_source_dir):
            # Nifti files are only at the leaf directory
            if len(filenames) == 0:
                continue

            filename: str
            for filename in filenames:
                # Skip if prediction is actually a ground truth (can be the case for recon-siren)
                if filename.endswith('gt.nii.gz'):
                    continue

                if_net = False
                if filename == 'surface_reconstruction.nii.gz':
                    if_net = True
                    filename = dirpath.split('/')[-1]
                    dirpath = '/'.join(dirpath.split('/')[:-1])
                print(filename)
                # Get subject name

                # Load prediction
                nifti_pred: nb.nifti1.Nifti1Image
                if if_net:
                    nifti_pred = nb.load(dirpath + '/' + filename + '/surface_reconstruction.nii.gz')
                else:
                    nifti_pred = nb.load(dirpath + '/' + filename)
                shape_pred = nifti_pred.shape
                spacing_pred = nb.affines.voxel_sizes(nifti_pred.affine)

                # Load ground truth
                path_gt_regex = gt_source_dir + '/**/' + filename + '.nii.gz'
                paths_gt = glob.glob(path_gt_regex, recursive=True)
                if len(paths_gt) != 1:
                    print('Found not exactly one one matching ground truth path for', path_gt_regex)
                    continue
                path_gt = paths_gt[0]
                nifti_gt: nb.nifti1.Nifti1Image = nb.load(path_gt)
                shape_gt = nifti_gt.shape
                spacing_gt = nb.affines.voxel_sizes(nifti_gt.affine)

                # Make sure shape and spacing is the same for prediction and ground truth
                assert shape_pred == shape_gt
                #assert np.allclose(spacing_pred, spacing_gt)

                # Compute scores
                print('Computing scores for', path_gt)
                casenames.append(path_gt)
                eval_single(nifti_pred.get_fdata(), nifti_gt.get_fdata(), spacing_pred, dices, asds, hd95s, max_distances)

        # Compute means and standard deviations
        dice_mean = np.mean(dices)
        dice_std = np.std(dices)
        asd_mean = np.mean(asds)
        asd_std = np.std(asds)
        hd95_mean = np.mean(hd95s)
        hd95_std = np.std(hd95s)
        max_distance_mean = np.mean(max_distances)
        max_distance_std = np.std(max_distances)
        dice_sorted_indexes = np.argsort(dices)

        # Print results
        print('Dice:           Mean:', dice_mean, '  ;  Std:', dice_std)
        print('ASD:            Mean:', asd_mean, '  ;  Std:', asd_std)
        print('HD95:           Mean:', hd95_mean, '  ;  Std:', hd95_std)
        print('Max Distance:   Mean:', max_distance_mean, '  ;  Std:', max_distance_std)

        # Check if we are allowed to write the results to a file
        if overwrite_results_file_if_exists is False and os.path.isfile(results_file) is True:
            print('WARNING: Does not write results to file as file already exists!')
            return

        # Save results in csv file
        with open(results_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['casename', 'dice', 'asds', 'hd95s', 'max_distances'])
            for row in zip(casenames, dices, asds, hd95s, max_distances):
                writer.writerow(row)
            writer.writerow([])
            writer.writerow(['MEAN', dice_mean, asd_mean, hd95_mean, max_distance_mean])
            writer.writerow(['STD', dice_std, asd_std, hd95_std, max_distance_std])
            writer.writerow([])
            for index in dice_sorted_indexes[0:int(len(dice_sorted_indexes)*0.1)]:
                writer.writerow([casenames[index], dices[index]])


if __name__ == '__main__':
    main()

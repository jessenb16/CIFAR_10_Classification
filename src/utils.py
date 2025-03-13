import os
import glob
import torch


def is_kaggle():
    return 'KAGGLE_KERNEL_RUN_TYPE' in os.environ


def get_paths():
    if is_kaggle():
        inference_dataset_path = '/kaggle/input/deep-learning-spring-2025-project-1/cifar_test_nolabel.pkl'
        saved_models_path = '/kaggle/working/saved_models'
        saved_predictions_path = '/kaggle/working/saved_predictions'
        saved_data_path = '/kaggle/working/saved_data'
        
    else:
        inference_dataset_path = 'cifar_test_nolabel.pkl'
        saved_models_path = 'saved_models'
        saved_predictions_path = 'saved_predictions'
        saved_data_path = 'saved_data'
    
    return inference_dataset_path, saved_models_path, saved_predictions_path, saved_data_path


def find_best_checkpoint(models_path=None, kaggle_dataset_path=None):
    """Find the best checkpoint by accuracy."""
    if models_path is None:
        _, models_path, _, _ = get_paths()
    
    checkpoints = []
    
    # Check local path
    checkpoint_files = glob.glob(os.path.join(models_path, '*.pth'))
    for file in checkpoint_files:
        filename = os.path.basename(file)
        if 'acc' in filename:
            try:
                acc_str = filename.split('acc')[-1].split('.pth')[0].replace('_', '.')
                acc = float(acc_str)
                checkpoints.append((acc, file, False))
            except:
                continue
    
    # Check Kaggle dataset path
    if kaggle_dataset_path and is_kaggle():
        kaggle_files = glob.glob(os.path.join(kaggle_dataset_path, '*.pth'))
        for file in kaggle_files:
            filename = os.path.basename(file)
            if 'acc' in filename:
                try:
                    acc_str = filename.split('acc')[-1].split('.pth')[0].replace('_', '.')
                    acc = float(acc_str)
                    checkpoints.append((acc, file, True))
                except:
                    continue
    
    if checkpoints:
        checkpoints.sort(reverse=True)
        return checkpoints[0][1], checkpoints[0][2]
        
    return None, False
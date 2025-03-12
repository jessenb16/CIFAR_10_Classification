import os


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
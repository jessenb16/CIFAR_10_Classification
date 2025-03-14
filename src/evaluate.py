"""
This module evaluates a trained model on the CIFAR-10 hidden test dataset.
Functions:
    load_cifar_batch(file):
        Loads a CIFAR-10 batch from a file.
    load_hidden_test():
        Loads and preprocesses the hidden test dataset.
    load_model(file, device):
        Loads a trained model from a file.
    save_predictions(predictions, file):
        Saves model predictions to a CSV file.
    evaluate(model, testloader, device, batch_size=100):
        Evaluates the model on the test dataset and returns predictions.
    main(filename=None):
        Main function to load data, load model(s), run inference, and save predictions.
"""
import os
import pickle
import torch
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import glob
from src.utils import get_paths


HIDDEN_PATH, SAVED_MODELS_PATH, SAVED_PREDICTIONS_PATH, SAVED_DATA_PATH = get_paths()

def load_cifar_batch(file):
    with open(file, 'rb') as fo:
        batch = pickle.load(fo, encoding='bytes')
    return batch

def load_hidden_test():
    # Load the hidden test batch.
    cifar10_batch = load_cifar_batch(HIDDEN_PATH)
    
    # Extract images; the test data is in (N x W x H x C) format.
    images = cifar10_batch[b'data']


    #Test transform
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Reshape from (N, H, W, C) → (N, C, H, W) for PyTorch
    images = images.reshape(-1, 32, 32, 3).astype(np.uint8)  # Ensure correct shape and data type

    # Convert to PyTorch tensor
    test_images = torch.stack([test_transform(img) for img in images])

    return test_images

def load_model(file, device):
    # Ensure saved_models directory exist
    os.makedirs(SAVED_MODELS_PATH, exist_ok=True)

    # Ensure file exists in saved_models directory
    if not os.path.exists(f'{SAVED_MODELS_PATH}/{file}'):
        raise FileNotFoundError(f'{file} not found in {SAVED_MODELS_PATH}')
    
    # Load the checkpoint
    print(f'Loading model from {SAVED_MODELS_PATH}/{file}')
    checkpoint = torch.load(f'{SAVED_MODELS_PATH}/{file}', map_location=device)
    
    # Extract model from checkpoint
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            # New format - complete model is stored
            model = checkpoint['model']
            print(f"Loaded model from checkpoint (epoch {checkpoint.get('epoch', 'unknown')}, "
                  f"accuracy {checkpoint.get('accuracy', 'unknown')}%)")
        else:
            # Old format or unexpected format
            raise ValueError(f"Expected 'model' key in checkpoint, found: {list(checkpoint.keys())}")
    else:
        # Direct model object
        model = checkpoint
        print("Loaded model (direct model format)")
    
    model.to(device)
    model.eval()  # Set to evaluation mode for inference
    
    return model

def save_predictions(predictions, file):
    # Ensure saved_predictions directory exists
    os.makedirs(SAVED_PREDICTIONS_PATH, exist_ok=True)

    # Save predictions to CSV in saved_predictions directory
    output_filename = file.replace('.pth', '.csv')
    df = pd.DataFrame({'ID': range(len(predictions)), 'Label': predictions})
    df.to_csv(f'{SAVED_PREDICTIONS_PATH}/{output_filename}', index=False)
    print(f'Predictions saved as {output_filename}')

def evaluate(model, testloader, device, batch_size=100):
    model.eval()
    predictions = []
    images = testloader.to(device)

    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            outputs = model(batch)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())

    return np.array(predictions)
        

def main(filename=None):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    test_images = load_hidden_test()

    # Get model filename(s)
    filenames = []
    if filename:
        filenames.append(filename)
    else:
        # Find saved models
        model_paths = glob.glob(f'{SAVED_MODELS_PATH}/*.pth')
        if not model_paths:
            raise FileNotFoundError(f'No models found in {SAVED_MODELS_PATH}')
        
        for model_path in model_paths:
            filenames.append(os.path.basename(model_path))

    # Run inference and save predictions
    for filename in filenames:
        # Load model
        model = load_model(filename, device)

        # Run inference
        predictions = evaluate(model, test_images, device)

        # Save predictions
        save_predictions(predictions, filename)


if __name__ == '__main__':
    try:
        main()
    except (FileNotFoundError, NameError) as e:
        print(f'ERROR: {e}')

















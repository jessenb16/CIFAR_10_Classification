import torch
import torch.optim as optim
import torch.nn as nn
import csv
import os
from src.dataset import get_cifar10_loaders
from src.model import ResNet18, ResNetTiny

def log_experiment(model_name, num_params, train_acc, test_acc, notes=""):
    """Logs experiment results to logs/experiments.csv and commits them to GitHub."""
    log_path = "logs/experiments.csv"
    commit_message = f"Logged experiment: {model_name}"

    # Append results to CSV
    with open(log_path, "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([model_name, num_params, train_acc, test_acc, notes])

    # Commit and push to GitHub
    os.system(f'git add {log_path}')
    os.system(f'git commit -m "{commit_message}"')
    os.system('git push origin main')

def train_model(epochs=20, batch_size=128, learning_rate=0.001, model_name="ResNet18"):
    """Trains ResNet model on CIFAR-10 dataset and logs results."""
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    trainloader, testloader = get_cifar10_loaders(batch_size)

    # Select model dynamically
    model_dict = {
        "ResNet18": ResNet18(),
        "ResNetTiny": ResNetTiny(),  # Smaller version
    }
    model = model_dict[model_name].to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_acc = 0.0
    num_params = sum(p.numel() for p in model.parameters())

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        train_acc = 100. * correct / total
        print(f"Epoch {epoch+1}/{epochs}: Loss: {running_loss/len(trainloader):.4f} | Train Acc: {train_acc:.2f}%")

        # Evaluate model
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

        test_acc = 100. * correct / total
        print(f"Test Acc: {test_acc:.2f}%")

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), f"models/{model_name}.pth")
            print(f"✅ Best model saved as {model_name}.pth")

    # Log experiment results
    log_experiment(model_name, num_params, train_acc, test_acc, "Baseline model")

if __name__ == "__main__":
    train_model()

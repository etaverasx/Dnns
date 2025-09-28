# train.py
# Task 1 training script
# Handles model training, evaluation, checkpoint saving, and plotting
# Uses the simple DataLoader (no augmentations) from dataset.py

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import csv
from datetime import datetime

# Import models
from models.lenet import LeNet
from models.resnet18 import get_resnet18
from models.vit import get_vit

# Import Task 1 DataLoader (no augmentations)
from dataset import get_dataloader_task1


# ======================
# Build model function
# ======================
def build_model(model_name, dataset, device):
    """Select and return the correct model based on name and dataset."""
    # MNIST = grayscale (1 channel), CIFAR10 = RGB (3 channels)
    in_channels = 1 if dataset.upper() == "MNIST" else 3
    num_classes = 10  # both datasets have 10 classes

    if model_name.lower() == "lenet":
        return LeNet(num_classes=num_classes, in_channels=in_channels).to(device)
    elif model_name.lower() == "resnet18":
        return get_resnet18(num_classes=num_classes, in_channels=in_channels).to(device)
    elif model_name.lower() == "vit":
        return get_vit(num_classes=num_classes, in_channels=in_channels).to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ======================
# Evaluate function
# ======================
def evaluate(model, loader, criterion, device):
    """Evaluate model on validation/test data."""
    model.eval()  # set model to eval mode
    loss_total, correct, total = 0.0, 0, 0
    with torch.no_grad():  # no gradients during evaluation
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss_total += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return loss_total / len(loader), 100.0 * correct / total


# ======================
# Train model
# ======================
def train_model(model_name="lenet", dataset="MNIST", epochs=10,
                batch_size=64, lr=0.001, device="cpu", exp_num=1, run_id=1):

    # Create directories to save results
    task_dir = os.path.join("reports", "task1")
    plots_dir = os.path.join(task_dir, "plots")
    models_dir = os.path.join(task_dir, "models", "checkpoints")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # Choose input image size
    # ViT expects 224x224, while LeNet/ResNet18 use 32x32
    img_size = 224 if model_name.lower() == "vit" else 32

    # Load train and test data using Task 1 dataloader
    train_loader = get_dataloader_task1(dataset, batch_size=batch_size, train=True, img_size=img_size)
    test_loader = get_dataloader_task1(dataset, batch_size=batch_size, train=False, img_size=img_size)

    # Initialize model, loss function, and optimizer
    model = build_model(model_name, dataset, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Metric storage for plotting
    epochs_recorded = []
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []

    # === Training loop ===
    for epoch in range(1, epochs + 1):
        model.train()  # set model to training mode
        running_loss, correct, total = 0.0, 0, 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()          # clear gradients
            outputs = model(inputs)        # forward pass
            loss = criterion(outputs, labels)  # compute loss
            loss.backward()                # backward pass
            optimizer.step()               # update weights

            # Track metrics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        # Compute training accuracy/loss for this epoch
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        print(f"Epoch {epoch}/{epochs} "
              f"Train Loss: {train_loss:.4f} "
              f"Train Acc: {train_acc:.2f}%")

        # Every 5 epochs (and final) evaluate on test set
        if epoch % 5 == 0 or epoch == epochs:
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)

            epochs_recorded.append(epoch)
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            test_losses.append(test_loss)
            test_accuracies.append(test_acc)

            print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

            # Save model checkpoint
            checkpoint_name = f"exp{exp_num}_{model_name}_{dataset}_epoch{epoch}.pth"
            checkpoint_path = os.path.join(models_dir, checkpoint_name)
            torch.save(model.state_dict(), checkpoint_path)

    # === Save plots ===
    timestamp = datetime.now().strftime("%I:%M%p")  # e.g., "12:30PM"

    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_recorded, train_losses, label="Train Loss", marker='o')
    plt.plot(epochs_recorded, test_losses, label="Test Loss", marker='o')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"{model_name.upper()} - Loss")
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_recorded, train_accuracies, label="Train Acc", marker='o')
    plt.plot(epochs_recorded, test_accuracies, label="Test Acc", marker='o')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title(f"{model_name.upper()} - Accuracy")
    plt.legend()

    plt.tight_layout()
    plot_path = os.path.join(plots_dir, f"exp{exp_num}_{model_name}_{dataset}_{timestamp}.png")
    plt.savefig(plot_path)
    plt.close()


    # === Save results to CSV ===
    csv_path = os.path.join(task_dir, "results.csv")
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "Run_ID", "Experiment", "Model", "Dataset", "Epoch",
                "Train Loss", "Train Acc", "Test Loss", "Test Acc", "Timestamp"
            ])
        for i, epoch in enumerate(epochs_recorded):
            writer.writerow([
                run_id, exp_num, model_name, dataset, epoch,
                train_losses[i], train_accuracies[i],
                test_losses[i], test_accuracies[i],
                timestamp
            ])


def evaluate(model, dataloader, criterion, device="cpu"):
    model.eval()
    loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss += criterion(outputs, labels).item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return loss / len(dataloader), 100. * correct / total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="lenet", help="Model: lenet | resnet18 | vit")
    parser.add_argument("--dataset", type=str, default="MNIST", help="Dataset: MNIST | CIFAR10")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--exp_num", type=int, default=1, help="Experiment number")
    parser.add_argument("--run_id", type=int, default=1, help="Global run ID")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_model(
        model_name=args.model,
        dataset=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        exp_num=args.exp_num,
        run_id=args.run_id
    )
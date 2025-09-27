import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import csv
from datetime import datetime

from models.lenet import LeNet
from models.resnet18 import get_resnet18
from dataset import get_dataloader


def build_model(model_name, dataset, device):
    in_channels = 1 if dataset.upper() == "MNIST" else 3
    num_classes = 10

    if model_name.lower() == "lenet":
        return LeNet(num_classes=num_classes, in_channels=in_channels).to(device)
    elif model_name.lower() == "resnet18":
        return get_resnet18(num_classes=num_classes, in_channels=in_channels).to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(dataloader), 100. * correct / total


def train_model(model_name, dataset, epochs, batch_size, lr,
                optimizer_choice, augment, exp_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Task 2 dirs ===
    task_dir = os.path.join("reports", "task2")
    plots_dir = os.path.join(task_dir, "plots-task2")
    models_dir = os.path.join(task_dir, "models-task2")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # === Transformations ===
    img_size = 32
    transform_aug = None
    if augment == "rotation":
        transform_aug = "rotation"
    elif augment == "flip":
        transform_aug = "flip"

    train_loader = get_dataloader(dataset, batch_size=batch_size, train=True, img_size=img_size)
    test_loader = get_dataloader(dataset, batch_size=batch_size, train=False, img_size=img_size)

    model = build_model(model_name, dataset, device)
    criterion = nn.CrossEntropyLoss()

    if optimizer_choice.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    epochs_recorded, train_losses, train_accs = [], [], []
    test_losses, test_accs = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, labels in tqdm(train_loader, desc=f"{exp_name} | Epoch {epoch}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        epochs_recorded.append(epoch)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

    # === Save plot ===
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_recorded, train_losses, label="Train Loss")
    plt.plot(epochs_recorded, test_losses, label="Test Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"{model_name.upper()} Loss ({dataset})")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_recorded, train_accs, label="Train Acc")
    plt.plot(epochs_recorded, test_accs, label="Test Acc")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title(f"{model_name.upper()} Accuracy ({dataset})")
    plt.legend()

    plot_path = os.path.join(plots_dir, f"{exp_name}.png")
    plt.savefig(plot_path)
    plt.close()

    # === Save checkpoint ===
    checkpoint_path = os.path.join(models_dir, f"{exp_name}.pth")
    torch.save(model.state_dict(), checkpoint_path)

    # === Log results to CSV ===
    csv_path = os.path.join(task_dir, "task2-results.csv")
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "Experiment", "Model", "Dataset", "Optimizer",
                "LearningRate", "BatchSize", "Augmentation",
                "FinalTrainLoss", "FinalTrainAcc",
                "FinalTestLoss", "FinalTestAcc"
            ])
        writer.writerow([
            exp_name, model_name, dataset, optimizer_choice,
            lr, batch_size, augment,
            round(train_losses[-1], 4), round(train_accs[-1], 2),
            round(test_losses[-1], 4), round(test_accs[-1], 2)
        ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--augment", type=str, default="none",
                        choices=["none", "rotation", "flip"])
    parser.add_argument("--exp_name", type=str, required=True)

    args = parser.parse_args()
    train_model(
        args.model, args.dataset, args.epochs,
        args.batch_size, args.lr, args.optimizer,
        args.augment, args.exp_name
    )

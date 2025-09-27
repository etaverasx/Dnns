import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import csv

from models.lenet import LeNet
from models.resnet18 import get_resnet18
from dataset import get_dataloader


# ============================
# Model builder
# ============================
def build_model(model_name, dataset, device):
    in_channels = 1 if dataset.upper() == "MNIST" else 3
    num_classes = 10

    if model_name.lower() == "lenet":
        return LeNet(num_classes=num_classes, in_channels=in_channels).to(device)
    elif model_name.lower() == "resnet18":
        return get_resnet18(num_classes=num_classes, in_channels=in_channels).to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ============================
# Evaluation
# ============================
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_total, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss_total += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return loss_total / len(loader), 100.0 * correct / total


# ============================
# Training
# ============================
def train_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Task 2 directories ===
    task2_dir = os.path.join("reports", "task2")
    plots_dir = os.path.join(task2_dir, "plots-task2")
    models_dir = os.path.join(task2_dir, "models-task2")
    csv_path = os.path.join(task2_dir, "task2-results.csv")

    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # === Data ===
    img_size = 32
    train_loader = get_dataloader(args.dataset, batch_size=args.batch_size, train=True, img_size=img_size)
    test_loader = get_dataloader(args.dataset, batch_size=args.batch_size, train=False, img_size=img_size)

    # === Model, loss, optimizer ===
    model = build_model(args.model, args.dataset, device)
    criterion = nn.CrossEntropyLoss()
    if args.optimizer.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    # === Metric storage ===
    epochs_recorded = []
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []

    # === Training loop ===
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, labels in tqdm(train_loader, desc=f"{args.exp_name} | Epoch {epoch}/{args.epochs}"):
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

        # Compute training metrics
        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total

        # === Record every 5 epochs ===
        if epoch % 5 == 0 or epoch == args.epochs:
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)

            epochs_recorded.append(epoch)
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            test_losses.append(test_loss)
            test_accuracies.append(test_acc)

            print(f"[Epoch {epoch}] Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
                  f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.2f}%")

            # === Save checkpoint ===
            ckpt_name = f"{args.exp_name}_epoch{epoch}.pth"
            torch.save(model.state_dict(), os.path.join(models_dir, ckpt_name))

            # === Append to CSV ===
            file_exists = os.path.isfile(csv_path)
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow([
                        "exp_name", "model", "dataset", "optimizer",
                        "learning_rate", "batch_size", "augmentation", "epoch",
                        "train_loss", "train_acc", "test_loss", "test_acc"
                    ])
                writer.writerow([
                    args.exp_name, args.model, args.dataset, args.optimizer,
                    args.lr, args.batch_size, args.augment, epoch,
                    f"{train_loss:.4f}", f"{train_acc:.2f}",
                    f"{test_loss:.4f}", f"{test_acc:.2f}"
                ])

    # === Save plots ===
    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_recorded, train_losses, label="Train Loss", marker="o")
    plt.plot(epochs_recorded, test_losses, label="Test Loss", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"{args.exp_name} Loss")
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_recorded, train_accuracies, label="Train Acc", marker="o")
    plt.plot(epochs_recorded, test_accuracies, label="Test Acc", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title(f"{args.exp_name} Accuracy")
    plt.legend()

    plt.tight_layout()
    plot_path = os.path.join(plots_dir, f"{args.exp_name}.png")
    plt.savefig(plot_path)
    plt.close()


# ================================
# CLI
# ================================
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
    train_model(args)   # ✅ call correct function
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.lenet import LeNet
from models.resnet18 import get_resnet18
from models.vit import get_vit
from dataset import get_dataloader


def build_model(model_name, dataset, device):
    """Return the requested model with correct input channels."""
    in_channels = 1 if dataset.upper() == "MNIST" else 3
    num_classes = 10

    if model_name.lower() == "lenet":
        return LeNet(num_classes=num_classes, in_channels=in_channels).to(device)
    elif model_name.lower() == "resnet18":
        return get_resnet18(num_classes=num_classes, in_channels=in_channels).to(device)
    elif model_name.lower() == "vit":
        return get_vit(num_classes=num_classes, in_channels=in_channels).to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_model(model_name="lenet", dataset="MNIST", epochs=10, batch_size=64, lr=0.001, device="cpu"):
    # Choose image size depending on model
    img_size = 224 if model_name.lower() == "vit" else 32

    # Load data
    train_loader = get_dataloader(dataset, batch_size=batch_size, train=True, img_size=img_size)
    test_loader = get_dataloader(dataset, batch_size=batch_size, train=False, img_size=img_size)

    # Model, loss, optimizer
    model = build_model(model_name, dataset, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Metric storage (sampled every 5 epochs)
    epochs_recorded = []
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
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
        train_acc = 100. * correct / total

        print(f"Epoch {epoch}/{epochs} "
              f"Train Loss: {train_loss:.4f} "
              f"Train Acc: {train_acc:.2f}%")

        # Record every 5 epochs (and final)
        if epoch % 5 == 0 or epoch == epochs:
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)

            epochs_recorded.append(epoch)
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            test_losses.append(test_loss)
            test_accuracies.append(test_acc)

            print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

    # Plot metrics (2 plots)
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
    plt.show()


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
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_model(
        model_name=args.model,
        dataset=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device
    )

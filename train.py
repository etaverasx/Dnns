import torch
import torch.nn as nn
import torch.optim as optim
from models.lenet import LeNet
from dataset import get_dataloader

def train_model(epochs=10, batch_size=64, lr=0.001, device="cpu"):
    # Load data
    train_loader = get_dataloader("MNIST", batch_size=batch_size, train=True)
    test_loader = get_dataloader("MNIST", batch_size=batch_size, train=False)

    # Model, loss, optimizer
    model = LeNet(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for i, (inputs, labels) in enumerate(train_loader):
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

        print(f"Epoch {epoch}/{epochs} "
              f"Train Loss: {running_loss/len(train_loader):.4f} "
              f"Train Acc: {100.*correct/total:.2f}%")

        # Evaluate on test set every 5 epochs
        if epoch % 5 == 0:
            evaluate(model, test_loader, criterion, device)

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
    print(f"Test Loss: {loss/len(dataloader):.4f}, "
          f"Test Acc: {100.*correct/total:.2f}%")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_model(epochs=10, batch_size=64, lr=0.001, device=device)

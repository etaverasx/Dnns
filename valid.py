# valid.py - Validate Task 1 checkpoints on MNIST and CIFAR-10

import torch
import torch.nn as nn
import os
import pandas as pd
from tqdm import tqdm

from models.lenet import LeNet
from models.resnet18 import get_resnet18
from models.vit import get_vit
from dataset import get_dataloader_task1   #  Task 1 dataloader only


# --------------------------
# Load model weights
# --------------------------
def load_checkpoint(model, checkpoint_path, device):
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)  # strict=False safe for ViT
    return model


# --------------------------
# Run evaluation
# --------------------------
def validate_model(model, dataloader, device):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    model.to(device)

    total_loss, total_correct, total_samples = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples * 100
    return avg_loss, avg_acc


# --------------------------
# Parse checkpoint filename
# Example: exp3_resnet18_MNIST_epoch25.pth
# --------------------------
def parse_checkpoint_name(ckpt):
    parts = ckpt.replace(".pth", "").split("_")
    exp_num = int(parts[0].replace("exp", ""))
    model = parts[1]
    dataset = parts[2]
    epoch = int(parts[3].replace("epoch", ""))
    return exp_num, model, dataset, epoch


# --------------------------
# Validate all checkpoints in Task 1
# --------------------------
def validate_all():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths
    checkpoints_dir = "./reports/task1/models/checkpoints/"
    results_csv = "./reports/validation.csv"

    checkpoints = sorted(os.listdir(checkpoints_dir))
    results = []

    for ckpt in tqdm(checkpoints, desc="Validating"):
        ckpt_path = os.path.join(checkpoints_dir, ckpt)

        try:
            exp_num, model_name, dataset, epoch = parse_checkpoint_name(ckpt)
        except Exception:
            print(f" Skipping malformed checkpoint name: {ckpt}")
            continue

        # dataset & preprocessing
        if dataset.upper() == "MNIST":
            in_channels = 1
            img_size = 32 if model_name in ["lenet", "resnet18"] else 224
        elif dataset.upper() == "CIFAR10":
            in_channels = 3
            img_size = 32 if model_name in ["lenet", "resnet18"] else 224
        else:
            continue

        # build model
        if model_name == "lenet":
            model = LeNet(num_classes=10, in_channels=in_channels)
        elif model_name == "resnet18":
            model = get_resnet18(num_classes=10, in_channels=in_channels)
        elif model_name == "vit":
            model = get_vit(num_classes=10, in_channels=in_channels)
        else:
            continue

        # dataloader (Task 1 pipeline only)
        dataloader = get_dataloader_task1(dataset, batch_size=64, train=False, img_size=img_size)

        # load weights & eval
        model = load_checkpoint(model, ckpt_path, device)
        loss, acc = validate_model(model, dataloader, device)

        print(f"[{ckpt}] Loss={loss:.4f}, Acc={acc:.2f}%")

        # append results (match results.csv schema where possible)
        results.append({
            "run_id": 1,  # fixed since validation doesn't track multiple runs
            "exp_num": exp_num,
            "model": model_name,
            "dataset": dataset,
            "epoch": epoch,
            "train_loss": None,   # not available during validation
            "train_acc": None,    # not available during validation
            "test_loss": round(loss, 4),
            "test_acc": round(acc, 2),
            "timestamp": None     # not re-logged here
        })

    # save results
    df = pd.DataFrame(results)
    df.to_csv(results_csv, index=False)
    print(f"\n Validation complete. Results saved to {results_csv}")


if __name__ == "__main__":
    validate_all()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import pandas as pd
import numpy as np 
import pydicom
import cv2
import wandb
import argparse
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader, random_split

class Custom_CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=3):
        super(Custom_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class MRIDataset(Dataset):
    def __init__(self, image_dir, csv_path):
        all_paths = [
            os.path.join(root, file)
            for root, _, files in os.walk(image_dir)
            for file in files if file.endswith(".dcm")
        ]

        self.df = pd.read_csv(csv_path)
        self.image_paths = []

        print(f"🔍 Checking {len(all_paths)} DICOM files for 2D validity...")

        for path in all_paths:
            try:
                dcm = pydicom.dcmread(path)
                image = dcm.pixel_array
                if image.ndim == 2:
                    self.image_paths.append(path)
                else:
                    print(f"❌ Skipping non-2D image: {path} (shape: {image.shape})")
            except Exception as e:
                print(f"❌ Could not read file: {path}, error: {e}")

        print(f"✅ Final usable 2D DICOMs: {len(self.image_paths)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path = self.image_paths[index]
        dicom = pydicom.dcmread(path)
        image = dicom.pixel_array.astype(np.float32)

        image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-5)
        image = cv2.resize(image, (64, 64))

        tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)

        patient_id = dicom.PatientID
        row = self.df[self.df["Subject"] == patient_id]
        label = row.iloc[0]["Group"] if not row.empty else "CN"

        return tensor, label


def train(epoch, model, loader, optimizer, criterion, CONFIG):
    device = CONFIG["device"]
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    label_map = {"CN": 0, "MCI": 1, "AD": 2}

    progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]", leave=False)
    for i, (inputs, labels) in enumerate(progress_bar):
        inputs = inputs.to(device)
        labels = torch.tensor([label_map[label] for label in labels]).to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        progress_bar.set_postfix({"loss": running_loss / (i + 1), "acc": 100. * correct / total})

    return running_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    label_map = {"CN": 0, "MCI": 1, "AD": 2}

    with torch.no_grad():
        progress_bar = tqdm(loader, desc="[Validate]", leave=False)
        for i, (inputs, labels) in enumerate(progress_bar):
            inputs = inputs.to(device)
            labels = torch.tensor([label_map[label] for label in labels]).to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            progress_bar.set_postfix({"loss": running_loss / (i+1), "acc": 100. * correct / total})

    return running_loss / len(loader), 100. * correct / total

def main():
    print("🔥 Starting training script...")

    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    CONFIG = {
        "model": "Custom_CNN_AD_2D",
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "optimizer": args.optimizer,
        "epochs": args.epochs,
        "num_workers": 4,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "data_dir": "Midline Train Test/train",
        "test_dir": "Midline Train Test/test",
        "csv_path": "Project-Datase.csv",
        "wandb_project": "alzheimers-dx"
    }

    dataset = MRIDataset(CONFIG["data_dir"], CONFIG["csv_path"])
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    trainset, valset = random_split(dataset, [train_size, val_size])

    trainloader = DataLoader(trainset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"])
    valloader = DataLoader(valset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])

    test_dataset = MRIDataset(CONFIG["test_dir"], CONFIG["csv_path"])
    testloader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])

    model = Custom_CNN(in_channels=1, num_classes=3).to(CONFIG["device"])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"]) if CONFIG["optimizer"] == "adam" else \
                optim.SGD(model.parameters(), lr=CONFIG["learning_rate"], momentum=0.9)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    wandb.init(project=CONFIG["wandb_project"], config=CONFIG)
    wandb.watch(model)

    best_val_acc = 0.0
    for epoch in range(CONFIG["epochs"]):
        train_loss, train_acc = train(epoch, model, trainloader, optimizer, criterion, CONFIG)
        val_loss, val_acc = validate(model, valloader, criterion, CONFIG["device"])
        scheduler.step()

        wandb.log({"epoch": epoch+1, "train_loss": train_loss, "train_acc": train_acc,
                   "val_loss": val_loss, "val_acc": val_acc, "lr": optimizer.param_groups[0]["lr"]})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model_cnn2d.pth")
            wandb.save("best_model_cnn2d.pth")

    test_loss, test_acc = validate(model, testloader, criterion, CONFIG["device"])
    print(f"✅ Test Accuracy: {test_acc:.2f}%, Test Loss: {test_loss:.4f}")
    wandb.log({"test_acc": test_acc, "test_loss": test_loss})

    wandb.finish()

if __name__ == "__main__":
    main()
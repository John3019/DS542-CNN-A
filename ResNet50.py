import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import os
import pandas as pd
import numpy as np
import pydicom
import cv2
from tqdm.auto import tqdm
import wandb
from PIL import Image

class MRIDataset(Dataset):
    def __init__(self, image_dir, csv_path, transform=None):
        self.image_paths = [
            os.path.join(root, file)
            for root, _, files in os.walk(image_dir)
            for file in files if file.endswith(".dcm")
        ]
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)



    def __getitem__(self, idx):
        path = self.image_paths[idx]
        dcm = pydicom.dcmread(path)
        image = dcm.pixel_array.astype(np.float32)

        # Handle problematic shapes
        if image.ndim == 3 and image.shape[0] == 1:
            image = image[0]  # squeeze single channel
        elif image.ndim == 1:
            raise ValueError(f"Unexpected 1D image shape: {image.shape} in file {path}")
        elif image.ndim > 3:
            raise ValueError(f"Too many dimensions: {image.shape} in file {path}")

        # Resize
        image = cv2.resize(image, (224, 224))

        # Normalize
        image = image / np.max(image) if np.max(image) > 0 else image

        # Convert grayscale to RGB
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)

        # Convert to uint8 and to PIL image
        try:
            image = Image.fromarray((image * 255).astype(np.uint8))
        except Exception as e:
            raise ValueError(f"Failed to convert image from file {path} to PIL format. Shape: {image.shape}, Error: {e}")

        patient_id = dcm.PatientID
        row = self.df[self.df["Subject"] == patient_id]
        label_str = row.iloc[0]["Group"] if not row.empty else "CN"

        if self.transform:
            image = self.transform(image)

        return image, label_str




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
    CONFIG = {
        "model": "ResNet50-Alzheimers",
        "batch_size": 8,
        "learning_rate": 0.001,
        "epochs": 5,
        "num_workers": 4,
        "device": "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu",
        "data_dir": "Midline Train Test/train",
        "csv_path": "Project-Datase.csv",
        "wandb_project": "alzheimers-dx"
    }

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    dataset = MRIDataset(CONFIG["data_dir"], CONFIG["csv_path"], transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    trainset, valset = random_split(dataset, [train_size, val_size])

    trainloader = DataLoader(trainset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"])
    valloader = DataLoader(valset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])

    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 3)
    model = model.to(CONFIG["device"])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
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
            torch.save(model.state_dict(), "best_model.pth")
            wandb.save("best_model.pth")

    wandb.finish()

if __name__ == "__main__":
    main()

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
from PIL import Image
from tqdm.auto import tqdm
import wandb
import argparse

class AlzheimerDataset(Dataset):
    def __init__(self, image_dir, csv_path, transform=None):
        self.image_paths = [
            os.path.join(root, file)
            for root, _, files in os.walk(image_dir)
            for file in files if file.endswith(".dcm")
        ]
        self.df = pd.read_csv(csv_path)
        self.transform = transform
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        dcm = pydicom.dcmread(path)
        image = dcm.pixel_array.astype(np.float32)

        # Handle 3D volumes
        if image.ndim == 3 and image.shape[-1] > 1:
            image = image[:, :, image.shape[-1] // 2]
        elif image.ndim == 3 and image.shape[0] == 1:
            image = image[0]
        elif image.ndim == 1:
            raise ValueError(f"Unexpected 1D image shape: {image.shape} in {path}")
        elif image.ndim > 3:
            raise ValueError(f"Too many dimensions: {image.shape} in {path}")

        image = cv2.resize(image, (224, 224))
        image = image / np.max(image) if np.max(image) > 0 else image

        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)

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
    labels = labels.to(device)


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    CONFIG = {
        "model": "VGG16_Alzheimers",
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "optimizer": args.optimizer,
        "epochs": args.epochs,
        "num_workers": 4,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "data_dir": "/projectnb/dl4ds/students/atuladas/DS542-CNN-A/Midline Train Test/train",
        "csv_path": "Project-Datase.csv",
        "wandb_project": "Danke Thomas-Müller"
    }

    wandb.init(project=CONFIG["wandb_project"], config=CONFIG)

    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.40], std=[0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.40], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = AlzheimerDataset(CONFIG["data_dir"], CONFIG["csv_path"], transform=transform_train)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    val_dataset.dataset.transform = transform_test

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"])
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])

    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, 3)
    model = model.to(CONFIG["device"])

    if CONFIG["optimizer"] == "adam":
        optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    else:
        optimizer = optim.SGD(model.parameters(), lr=CONFIG["learning_rate"], momentum=0.9)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    for epoch in range(CONFIG["epochs"]):
        train_loss, train_acc = train(epoch, model, train_loader, optimizer, criterion, CONFIG)
        val_loss, val_acc = validate(model, val_loader, criterion, CONFIG["device"])
        
        wandb.log({
            "train/loss": train_loss,
            "train/acc": train_acc,
            "val/loss": val_loss,
            "val/acc": val_acc,
            "epoch": epoch + 1
        })

if __name__ == "__main__":
	main()

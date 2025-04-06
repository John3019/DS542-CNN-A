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
    def __init__(self,image_dir,csv_path,transform=None):

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
    parser=argparse.ArgumentParser()
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

    transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),  
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.40], std=[0.229, 0.224, 0.225])])
    
    transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.40], std=[0.229, 0.224, 0.225])])



    model=models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    model.fc=nn.Linear(vgg.fc.in_features,3)
    model = model.to(CONFIG["model"])

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)



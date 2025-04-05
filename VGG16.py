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
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
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset


#Custom CNN that analyzes 3D Images 
class Custom_CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=3):                                   #initialize model with 1 in_channel for grayscale and 3 for num_classes for CN, MCI, AD
        super(Custom_CNN, self).__init__()                                              #parent constructor
        self.conv_layer_1 = nn.Conv3d(in_channels, 32, kernel_size=3, padding=1)        #first convolutional layer 3D
        self.batch_normal_1 = nn.BatchNorm3d(32)                                        #normalize the output of the first convolutional layer
        self.pooling_layer_1 = nn.MaxPool3d(2)                                          #downsample by factor of 2

        self.conv_layer_2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)                 #second convolutional layer 3D
        self.batch_normal_2 = nn.BatchNorm3d(64)                                        #normalize the output of the second convolutional layer
        self.pooling_layer_2 = nn.MaxPool3d(2)                                          #downsample by factor of 2

        self.conv_layer_3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)                #third convolutional layer 3D
        self.batch_normal_3 = nn.BatchNorm3d(128)                                       #normalize the output of the third convolutional layer
        self.pooling_layer_3 = nn.AdaptiveAvgPool3d(1)                                  #global average pooling, has an output of 128x1x1x1

        self.fully_connected = nn.Linear(128, num_classes)                              #final fully connected layer used for classiciation

    def forward(self, x):
        x = self.pooling_layer_1(F.relu(self.batch_normal_1(self.conv_layer_1(x))))     #first convolutional layer -> batch normalization -> ReLU -> pooling 
        x = self.pooling_layer_2(F.relu(self.batch_normal_2(self.conv_layer_2(x))))     #second convolutional layer -> batch normalization -> ReLU -> pooling 
        x = self.pooling_layer_3(F.relu(self.batch_normal_3(self.conv_layer_3(x))))     #third convolutional layer -> batch normalization -> ReLU -> pooling 
        x = x.view(x.size(0), -1)                                                       #flatten
        return self.fully_connected(x)                                                  #fully connected layer 

class MRIDataset(Dataset):
    def __init__(self, image_dir, csv_path, target_size=(64,64)):
        self.df = pd.read_csv(csv_path)                                                 #load the csv file to pull Subject Information
        self.target_size = target_size                                                  #resize the 2d slices
        groups = {}                                                                     #dictionary for slices to be grouped by (subject, visit)
        for root, _, files in os.walk(image_dir):                                       #walk through all files in path
            for file_name in files:                                                     #for all files in the folder
                if not file_name.lower().endswith('.dcm'): continue                     #skip files that are not .dcm
                parts = file_name.split('_')                                            #extract Subject ID from file
                subject = '_'.join(parts[:3])                                           #Subject ID
                visit = parts[3]                                                        #Date for Visit
                try:
                    slice_num = int(parts[-1].replace('Slice_','').replace('.dcm',''))  #get te slice number 
                except:
                    slice_num = 0
                groups.setdefault((subject,visit), []).append((slice_num, os.path.join(root,file_name)))    #group together the slices
        self.keys = list(groups.keys())
        self.groups = {k: [p for _,p in sorted(v, key=lambda x:x[0])] for k,v in groups.items()}            #sorts the slices

    def __len__(self): 
        return len(self.keys)                                                                               #return the total number of visits per subject

    def __getitem__(self, idx):
        subject, visit = self.keys[idx]                                                         #get the the subject ID and visit date
        paths = self.groups[(subject,visit)]                                                    #get corresponding dicom slices
        slices = []                                                                             #list to store slice pixel data
        for path in paths:          
            dcm = pydicom.dcmread(path)                                                         #read the specific dicom file
            pix = dcm.pixel_array.astype(np.float32)                                            #get the array of the scan
            if pix.ndim == 2:
                slices.append(pix)                                                              #add the 2d slice
            elif pix.ndim == 3:
                for z in range(pix.shape[0]): slices.append(pix[z])                             #add the 3d slices
            else:
                raise ValueError(f"Unsupported dims={pix.ndim}")                                #handles files that have invalid shapes
        if not slices:
            raise RuntimeError(f"No slices for {subject} {visit}")                              #handle the empty cases
        volume = np.stack(slices, axis=0)                                                       #stack the slices to be 3D
        mn, mx = volume.min(), volume.max()                                                     #normalization min and max
        volume = (volume - mn)/(mx - mn + 1e-5)                                                 #normalize values to be between 0-1
        volume = np.stack([cv2.resize(s, self.target_size) for s in volume], axis=0)            #resize the slices
        tensor = torch.from_numpy(volume).unsqueeze(0)                                          #channel dimension
        row = self.df[self.df['Subject']==subject]                                              #pull the label from the CSV
        label = row.iloc[0]['Group'] if not row.empty else 'CN'                                 #if there is no group assign it as CN
        return tensor, label                                                                    #return the volume and the label

def pad_collate(batch):
    volumes, labels = zip(*batch)                                                               #unpack the batch
    depths = [v.shape[1] for v in volumes]                                                      #get the depth of each of the volumes
    maxD = max(depths)                                                                          #calculate the maximum depth of the batch
    padded = []                                                                                 #list to store the padded volumes
    for v in volumes:
        D = v.shape[1]
        pad_f = (maxD - D)//2                                                                   #pad the front
        pad_b = maxD - D - pad_f                                                                #pad the back
        v_p = F.pad(v, (0,0, 0,0, pad_f, pad_b))                                                #pad  the depth dimension only... no other dimensions
        padded.append(v_p)                                                                      #add the padded volume
    vol_batch = torch.stack(padded, dim=0)                                                      #stack all the padded tensors
    label_batch = torch.tensor([{'CN':0,'MCI':1,'AD':2}[l] for l in labels])                    #convert the label to an integer
    return vol_batch, label_batch                                                               #return batched tensors 


def train(epoch, model, loader, optimizer, criterion, CONFIG):
    model.train()                                               #set the model to traiing mode
    running_loss = correct = total = 0                          #reset the counteres
    for inputs, labels in tqdm(loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]", leave=False):
        inputs = inputs.to(CONFIG['device'])
        labels = labels.to(CONFIG['device'])
        optimizer.zero_grad()                                   #reset the gradient             
        outputs = model(inputs)                                 #forward pass
        loss = criterion(outputs, labels)                       #copmute the loss
        loss.backward()                                         #backpropagation
        optimizer.step()                                        #optimizer step
        running_loss += loss.item()                             #track the loss 
        _,pred = outputs.max(1)                                 #predicted labels
        total += labels.size(0)                                 #all samples seen
        correct += pred.eq(labels).sum().item()                 #accumulation of correct predictions
    return running_loss/len(loader), 100.*correct/total         #return the average loss and the accuracy 


def validate(model, loader, criterion, device):
    model.eval()                                                #set the model to evaluation mode
    running_loss = correct = total = 0                          #reset the counters
    with torch.no_grad():                                       #turn off the gradient
        for inputs, labels in tqdm(loader, desc='[Validate]', leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _,pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
    return running_loss/len(loader), 100.*correct/total       #return thte locc and accuracy 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--optimizer', choices=['adam','sgd'], default='adam')
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    CONFIG = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'optimizer': args.optimizer,
        'epochs': args.epochs,
        'num_workers': 4,
        'data_dir': '/projectnb/dl4ds/students/rsingh13/DS542_Final/Structure Dataset Train Test/train',
        'test_dir': '/projectnb/dl4ds/students/rsingh13/DS542_Final/Structure Dataset Train Test/test',
        'csv_path': '/projectnb/dl4ds/students/rsingh13/DS542_Final/Project-Datase.csv',
        'wandb_project': '3D_CNN'
    }

    # Prepare datasets & loaders
    full_ds = MRIDataset(CONFIG['data_dir'], CONFIG['csv_path'])                                                      #load full training dataset

    all_keys = full_ds.keys                                                                                           #extract all (subject, visit) pairs
    unique_subjects = list(set([k[0] for k in all_keys]))                                                              #extract unique subjects
    train_subjects, val_subjects = train_test_split(unique_subjects, test_size=0.2, random_state=42)                  #split by subject ID to prevent leakage
    train_keys = [k for k in all_keys if k[0] in train_subjects]                                                      #get all (subject, visit) pairs for train
    val_keys   = [k for k in all_keys if k[0] in val_subjects]                                                        #get all (subject, visit) pairs for val

    train_indices = [i for i, k in enumerate(full_ds.keys) if k in train_keys]                                        #find all indices in train split
    val_indices   = [i for i, k in enumerate(full_ds.keys) if k in val_keys]                                          #find all indices in val split

    train_ds = Subset(full_ds, train_indices)                                                                         #create subset for training
    val_ds   = Subset(full_ds, val_indices)                                                                           #create subset for validation

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'], collate_fn=pad_collate)      #train dataloader
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'], collate_fn=pad_collate)          #val dataloader
    test_loader = DataLoader(MRIDataset(CONFIG['test_dir'], CONFIG['csv_path']), batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'], collate_fn=pad_collate)  #test dataloader

    model = Custom_CNN().to(CONFIG['device'])                                                                          #instantiate model
    criterion = nn.CrossEntropyLoss()                                                                                  #define loss function
    if CONFIG["optimizer"] == "adam":
        optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])                                        #Adam optimizer
    else:
        optimizer = optim.SGD(model.parameters(), lr=CONFIG["learning_rate"], momentum=0.9)                           #SGD optimizer with momentum

    run = wandb.init(project=CONFIG['wandb_project'], config=CONFIG)                                                  #initiate wandb run
    wandb.watch(model)                                                                                                #track gradients

    best_acc = 0
    for epoch in range(CONFIG['epochs']):                                                                             #training loop
        print(f"[INFO] === Epoch {epoch+1}/{CONFIG['epochs']} ===")
        tr_loss, tr_acc = train(epoch, model, train_loader, optimizer, criterion, CONFIG)
        val_loss, val_acc = validate(model, val_loader, criterion, CONFIG['device'])
        wandb.log({'train_loss':tr_loss, 'train_acc':tr_acc, 'val_loss':val_loss,  'val_acc':val_acc,  'epoch':epoch+1})   #log metrics
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')                                                          #save best model

    test_loss, test_acc = validate(model, test_loader, criterion, CONFIG['device'])                                   #final evaluation
    print(f"Test Acc: {test_acc:.2f}% | Test Loss: {test_loss:.4f}")
    wandb.log({'test_acc':test_acc, 'test_loss':test_loss})                                                           #log final metrics
    wandb.finish()                                                                                                     #end wandb run

if __name__ == '__main__':
    main()
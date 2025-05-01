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
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

activations = {}

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach().cpu()
    return hook

def log_activation_map(model, loader, device, label='activation_map'):
    model.eval()
    inputs, _ = next(iter(loader))
    inputs = inputs.to(device)
    with torch.no_grad():
        _ = model(inputs)

    fmap = activations['conv_layer_1'][0]  # First sample in batch
    D = fmap.shape[1] // 2  # Middle depth slice
    channels_to_show = min(6, fmap.shape[0])

    fig, axes = plt.subplots(1, channels_to_show, figsize=(15, 4))
    for i in range(channels_to_show):
        axes[i].imshow(fmap[i, D], cmap='viridis')
        axes[i].axis('off')
        axes[i].set_title(f"Ch {i}")
    plt.tight_layout()
    wandb.log({label: wandb.Image(fig)})
    plt.close()

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
        subject, visit = self.keys[idx]                                                         # get subject ID and visit date
        paths = self.groups[(subject, visit)]                                                   # get paths to DICOM slices
        slices = []                                                                             # list to hold individual 2D slices

        for path in paths:
            dcm = pydicom.dcmread(path)                                                         # read DICOM file
            pix = dcm.pixel_array.astype(np.float32)                                            # get pixel array

            if pix.ndim == 2:
                slices.append(pix)
            elif pix.ndim == 3:
                for z in range(pix.shape[0]):
                    slices.append(pix[z])
            else:
                raise ValueError(f"Unsupported dims={pix.ndim}")

        if not slices:
            raise RuntimeError(f"No slices for {subject} {visit}")

        # Stack into a 3D volume
        volume = np.stack(slices, axis=0)

        # Normalize
        mn, mx = volume.min(), volume.max()
        volume = (volume - mn) / (mx - mn + 1e-5)

        # Crop 25% from each side and resize
        cropped_resized = []
        for s in volume:
            h, w = s.shape
            crop_h = int(h * 0.2)
            crop_w = int(w * 0.195)
            s_cropped = s[crop_h:h - crop_h, crop_w:w - crop_w]
            s_resized = cv2.resize(s_cropped, self.target_size)
            cropped_resized.append(s_resized)

        # Stack final volume
        volume = np.stack(cropped_resized, axis=0)
        tensor = torch.from_numpy(volume).unsqueeze(0)  # add channel dimension

        # Retrieve label
        row = self.df[self.df['Subject'] == subject]
        label = row.iloc[0]['Group'] if not row.empty else 'CN'

        return tensor, label                #return the volume and the label

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
        'wandb_project': '3D_CNN_V2'
    }

    full_ds = MRIDataset(CONFIG['data_dir'], CONFIG['csv_path'])
    label_list = [label for _, label in full_ds]
    print(f"[INFO] Label distribution: {Counter(label_list)}")

    depths_by_label = defaultdict(list)
    for volume, label in full_ds:
        depths_by_label[label].append(volume.shape[1])
    for label, depths in depths_by_label.items():
        print(f"[INFO] Depth for {label}: mean={np.mean(depths):.2f}, std={np.std(depths):.2f}, count={len(depths)}")

    val_size = int(0.2 * len(full_ds))
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'], collate_fn=pad_collate)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'], collate_fn=pad_collate)
    test_loader = DataLoader(MRIDataset(CONFIG['test_dir'], CONFIG['csv_path']), batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'], collate_fn=pad_collate)

    model = Custom_CNN().to(CONFIG['device'])
    model.conv_layer_1.register_forward_hook(get_activation('conv_layer_1'))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate']) if CONFIG['optimizer'] == 'adam' else optim.SGD(model.parameters(), lr=CONFIG['learning_rate'], momentum=0.9)

    run = wandb.init(project=CONFIG['wandb_project'], config=CONFIG)
    wandb.watch(model)

    best_acc = 0
    for epoch in range(CONFIG['epochs']):
        print(f"[INFO] === Epoch {epoch+1}/{CONFIG['epochs']} ===")
        tr_loss, tr_acc = train(epoch, model, train_loader, optimizer, criterion, CONFIG)
        val_loss, val_acc = validate(model, val_loader, criterion, CONFIG['device'])
        wandb.log({'train_loss': tr_loss, 'train_acc': tr_acc,
                   'val_loss': val_loss, 'val_acc': val_acc, 'epoch': epoch+1})

        log_activation_map(model, val_loader, CONFIG['device'], label='customcnn_activation_map')

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')

    test_loss, test_acc = validate(model, test_loader, criterion, CONFIG['device'])
    print(f"Test Acc: {test_acc:.2f}% | Test Loss: {test_loss:.4f}")
    wandb.log({'test_acc': test_acc, 'test_loss': test_loss})
    wandb.finish()

if __name__ == '__main__':
    main()

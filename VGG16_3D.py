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
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split

activation = {}

# Custom full 3D VGG16 model
class VGG16_3D(nn.Module):
    def __init__(self, num_classes=3):
        super(VGG16_3D, self).__init__()
        self.conv1 = nn.Conv3d(1, 64, kernel_size=(3,7,7), stride=(1,2,2), padding=(1,3,3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, blocks=3)
        self.layer2 = self._make_layer(64, 128, blocks=4, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=6, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))
        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False))
        layers.append(nn.BatchNorm3d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(1, blocks):
            layers.append(nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm3d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

class MRIDataset(Dataset):
    def __init__(self, image_dir, csv_path, target_size=(64,64)):
        self.df = pd.read_csv(csv_path)
        self.target_size = target_size
        groups = {}
        for root, _, files in os.walk(image_dir):
            for file_name in files:
                if not file_name.lower().endswith('.dcm'):
                    continue
                parts = file_name.split('_')
                subject = '_'.join(parts[:3])
                visit = parts[3]
                try:
                    slice_num = int(parts[-1].replace('Slice_','').replace('.dcm',''))
                except:
                    slice_num = 0
                groups.setdefault((subject, visit), []).append((slice_num, os.path.join(root, file_name)))
        self.keys = list(groups.keys())
        self.groups = {k: [p for _,p in sorted(v, key=lambda x:x[0])] for k,v in groups.items()}

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        subject, visit = self.keys[idx]
        paths = self.groups[(subject, visit)]
        slices = []
        for path in paths:
            dcm = pydicom.dcmread(path)
            pix = dcm.pixel_array.astype(np.float32)
            if pix.ndim == 2:
                slices.append(pix)
            elif pix.ndim == 3:
                for z in range(pix.shape[0]):
                    slices.append(pix[z])
            else:
                raise ValueError(f"Unsupported dims={pix.ndim}")
        if not slices:
            raise RuntimeError(f"No slices for {subject} {visit}")
        volume = np.stack(slices, axis=0)
        mn, mx = volume.min(), volume.max()
        volume = (volume - mn) / (mx - mn + 1e-5)
        volume = np.stack([cv2.resize(s, self.target_size) for s in volume], axis=0)
        tensor = torch.from_numpy(volume).unsqueeze(0)
        row = self.df[self.df['Subject'] == subject]
        label = row.iloc[0]['Group'] if not row.empty else 'CN'
        return tensor, label

def pad_collate(batch):
    volumes, labels = zip(*batch)
    depths = [v.shape[1] for v in volumes]
    maxD = max(depths)
    padded = []
    for v in volumes:
        D = v.shape[1]
        pad_f = (maxD - D)//2
        pad_b = maxD - D - pad_f
        v_p = F.pad(v, (0,0, 0,0, pad_f, pad_b))
        padded.append(v_p)
    vol_batch = torch.stack(padded, dim=0)
    label_batch = torch.tensor([{'CN':0,'MCI':1,'AD':2}[l] for l in labels])
    return vol_batch, label_batch

def train(epoch, model, loader, optimizer, criterion, CONFIG):
    model.train()
    running_loss = correct = total = 0
    for inputs, labels in tqdm(loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]", leave=False):
        inputs = inputs.to(CONFIG['device'])
        labels = labels.to(CONFIG['device'])
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, pred = outputs.max(1)
        total += labels.size(0)
        correct += pred.eq(labels).sum().item()
    return running_loss/len(loader), 100.*correct/total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = correct = total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc='[Validate]', leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
    return running_loss/len(loader), 100.*correct/total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--optimizer', choices=['adam','sgd'], default='adam')
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    CONFIG = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'batch_size': args.batch_size,
        'learning_rate': 0.0001,
        'optimizer': args.optimizer,
        'epochs': 20,
        'num_workers': 4,
        'train_dir': 'train',
        'test_dir': 'test',
        'csv_path': 'Project-Datase.csv',
        'wandb_project': 'alzheimers-vgg16-3d'
    }

    train_ds_full = MRIDataset(CONFIG['train_dir'], CONFIG['csv_path'])
    test_ds_full = MRIDataset(CONFIG['test_dir'], CONFIG['csv_path'])

    all_keys = train_ds_full.keys
    unique_subjects = list(set([k[0] for k in all_keys]))
    train_subjects, val_subjects = train_test_split(unique_subjects, test_size=0.2, random_state=42)
    train_keys = [k for k in all_keys if k[0] in train_subjects]
    val_keys = [k for k in all_keys if k[0] in val_subjects]

    train_indices = [i for i, k in enumerate(train_ds_full.keys) if k in train_keys]
    val_indices = [i for i, k in enumerate(train_ds_full.keys) if k in val_keys]

    train_ds = Subset(train_ds_full, train_indices)
    val_ds = Subset(train_ds_full, val_indices)

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'], collate_fn=pad_collate)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'], collate_fn=pad_collate)
    test_loader = DataLoader(test_ds_full, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'], collate_fn=pad_collate)

    model = VGG16_3D(num_classes=3).to(CONFIG['device'])
    model.layer4.register_forward_hook(get_activation('layer4'))

    criterion = nn.CrossEntropyLoss()
    if CONFIG['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    else:
        optimizer = optim.SGD(model.parameters(), lr=CONFIG['learning_rate'], momentum=0.9)

    wandb.init(project=CONFIG['wandb_project'], config=CONFIG)
    wandb.watch(model)

    best_acc = 0
    for epoch in range(CONFIG['epochs']):
        print(f"[INFO] === Epoch {epoch+1}/{CONFIG['epochs']} ===")
        tr_loss, tr_acc = train(epoch, model, train_loader, optimizer, criterion, CONFIG)
        val_loss, val_acc = validate(model, val_loader, criterion, CONFIG['device'])
        wandb.log({'train_loss': tr_loss, 'train_acc': tr_acc, 'val_loss': val_loss, 'val_acc': val_acc, 'epoch': epoch+1})
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')

    test_loss, test_acc = validate(model, test_loader, criterion, CONFIG['device'])
    print(f"Test Acc: {test_acc:.2f}% | Test Loss: {test_loss:.4f}")
    wandb.log({'test_acc': test_acc, 'test_loss': test_loss})
    wandb.finish()
    sample_input, _ = test_ds_full[0]
    sample_input = sample_input.unsqueeze(0).to(CONFIG['device'])  # Add batch dim
    model.eval()
    _ = model(sample_input)  # Trigger forward pass
    
    # Retrieve and process activations
    act = activation['layer4'].squeeze(0).cpu()  # shape: (C, D, H, W)
    
    # Average across channels to get a heatmap
    heatmap = act.mean(dim=0)  # shape: (D, H, W)
    
    # Pick middle slice
    mid_slice = heatmap.shape[0] // 2
    slice_img = heatmap[mid_slice]
    
    # Normalize for visualization
    slice_img = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min() + 1e-5)
    
    # Plot using matplotlib
    import matplotlib.pyplot as plt
    plt.imshow(slice_img.numpy(), cmap='inferno')
    plt.title('Activation Map from layer4 (Middle Slice)')
    plt.axis('off')
    plt.savefig("activation_map.png")
    plt.show()


if __name__ == '__main__':
    main()

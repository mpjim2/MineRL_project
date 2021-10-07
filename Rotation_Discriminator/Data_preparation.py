import torchvision.models as models
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import pickle
import numpy as np
import minerl

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

class MyDataset(Dataset):
    '''Custom Dataset for the Rotation Discrimination Task; x is the frame y is the (binary) label whether the frame is rotated or not'''
    def __init__(self, frames : np.ndarray, labels : int, transform=None):
        self.frames = frames
        self.transform = transform
        self.labels = labels

    def __getitem__(self, index):
        x = self.frames[index]
        x = self.transform(x)
        
        y = self.labels[index]
        return x,  y
    
    def __len__(self):
        return len(self.frames)

def prepare_data(dataset='MineRLTreechop-v0' : str, samplecount=1000 : int, batch_size=16 : int) -> (DataLoader, DataLoader):
    '''Itereates the given dataset and extracts the relevant data and returns pytorch dataloaders for training & validation'''
    data = minerl.data.make(dataset, data_dir='../Data/')
    iterator = minerl.data.BufferedBatchIter(data)

    frames = []
    rot_labels = []

    for current_state, action, _, _, _ in iterator.buffered_batch_iter(batch_size=1, num_epochs=1):

        #every frame is appended before AND after rotating, such that the dataset consists of 50% rotated
        #and 50 % original frames        
        
        frames.append(current_state['pov'][0])
        rot_labels.append(1)

        #rotate frame either by 90 180 or 270 degrees 
        k = np.random.choice([1, 2, 3])
        rotated = np.rot90(current_state['pov'][0], k).copy()
        frames.append(rotated)
        rot_labels.append(0)

        if len(frames) >= samplecount:
            break
            
    rot_labels  = np.asarray(rot_labels)
    dataset = MyDataset(frames, rot_labels, transform)

    #Split data into training and validation set; 90 / 10 split
    set_lengths = [int(samplecount * 0.9), int(samplecount * 0.1)]
    train_set, val_set = torch.utils.data.random_split(dataset, set_lengths)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=1,
        shuffle=True
    )
    validation_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        num_workers=1,
        shuffle=True
    )

    return train_loader, validation_loader
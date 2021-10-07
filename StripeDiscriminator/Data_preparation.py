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
import matplotlib.pyplot as plt

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

class MyDataset(Dataset):
    def __init__(self, frames, labels, transform=None):
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


def prepare_data(batchsize, dataset='MineRLTreechop-v0'):
    '''extract relevant Data from the given Dataset'''
    data = minerl.data.make(dataset, data_dir='../Data/')
    
    iterator = minerl.data.BufferedBatchIter(data)

    print(len(iterator.all_trajectories))    
    stripes = []
    labels = []

    for current_state, _, _, _, _ in iterator.buffered_batch_iter(batch_size=1, num_epochs=1):
        
        #divide the current frame into 4 stripes of size (64, 16, 3)
        for i in range(0, 64, 16):

            stripe = current_state['pov'][0][:, i:(i+16), :]
            stripes.append(stripe)
            
            labels.append(int(i/16))
            

    stripes = np.asarray(stripes)
    labels = np.asarray(labels)

    dataset = MyDataset(stripes, labels, transform)
    #training/validation split: 90 / 10
    set_lengths = [len(stripes)-int(0.1 * len(stripes)), int(0.1 * len(stripes))]
    
    
    train_set, val_set = torch.utils.data.random_split(dataset, set_lengths)

    train_loader = DataLoader(
        train_set,
        batch_size=batchsize,
        num_workers=1,
        shuffle=True
    )
    validation_loader = DataLoader(
        val_set,
        batch_size=batchsize,
        num_workers=1,
        shuffle=True
    )

    return train_loader, validation_loader



if __name__ == '__main__':
    
    x, y = prepare_data(4)

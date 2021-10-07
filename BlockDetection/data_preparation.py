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
from tqdm import tqdm 
import os

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

class MyDataset(Dataset):
    '''Custon pytorch dataset for the Block classification task'''
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


def load_sets(samplesPerclass):
    '''Loads the Datasets (one per class) from the disk'''
    path = './'
    keys = ['coal', 'diamond', 'misc', 'cobblestone', 'log','dirt', 'iron_ore']
    frames = []
    labels = []
    block2id = {}

    min_len = None
    smallest_set = None
    for key in keys: 
        filename = './' + key + '_frames.pickle'
        label_id = len(np.unique(labels))
        block2id[key] = label_id

        with open(filename, 'rb') as f:
            key_frames = pickle.load(f)
        
        if min_len is None:
            min_len = len(key_frames)
            smallest_set = label_id
        else:
            if min_len > len(key_frames):
                min_len = len(key_frames)
        
        print(len(key_frames), key)
        frames += key_frames
        labels += [label_id for x in range(len(key_frames))]

    assert len(frames) == len(labels)

    frames, labels = equalize(frames, labels, samplesPerclass)
    
    return frames, labels, block2id

def equalize(frames, labels, size): 
    '''equalize the distribution of classes in the dataset; if the given size is bigger than the number of samples
    for a specific class all samples of that class are used; else randomly select x=size samples'''
    keep_labels = []
    keep_frames = []
    uni, counts = np.unique(labels, return_counts=True)

    for val, count in zip(uni, counts):
        if count > size:
            ids = np.argwhere(labels == val).flatten()
            
            keep_ids = np.random.choice(ids, size)

            keep_labels += [val for x in range(size)]
            keep_frames += [frames[i] for i in keep_ids]
        else:
            ids = np.argwhere(labels == val).flatten()
            key_frames = [frames[i] for i in ids]
            keep_labels += [val for x in range(len(key_frames))]
            keep_frames +=  key_frames
    
    assert len(keep_frames) == len(keep_labels)
    
    keep_frames = np.asarray(keep_frames)
    keep_labels = np.asarray(keep_labels)

    print(keep_frames.shape)
    print(keep_labels.shape)

    return keep_frames, keep_labels

def make_datasets(frames, labels, batchsize): 
    '''Create pytorch dataloaders from the frames and labels'''

    dataset = MyDataset(frames, labels, transform)
    set_lengths = [frames.shape[0] - int(0.1 * frames.shape[0]), int(0.1 * frames.shape[0])]
    
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

def prepare_data(batchsize, samplesPerclass=10000):

    '''Prepare the data that was saved to the disk previously'''
    f, l, block2id = load_sets(samplesPerclass)

    t, v = make_datasets(f, l, batchsize)

    return t, v, block2id
    
if __name__ == '__main__':
    
    f, l, block2id = load_sets()

    t, v = make_datasets(f, l, 4)


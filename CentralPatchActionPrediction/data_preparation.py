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

def equalize(frames, labels): 

    positive = np.argwhere(labels==1.)
    negative = np.argwhere(labels==0.)

    pos_frames = frames[positive]
    neg_frames = frames[negative]
    
    pos_labels = labels[positive]
    neg_labels = labels[negative]

    diff = abs(len(positive) - len(negative))

    if len(positive) > len(negative):
        del_idx = np.random.choice([x for x in range(len(positive))], size=diff, replace=False)
        del_idx.sort()
        for i in reversed(del_idx):
            pos_labels = np.vstack([pos_labels[:i, :], pos_labels[i+1:, :]])
            pos_frames = np.vstack([pos_frames[:i, :, :, :], pos_frames[i+1:, :, :, :]])

    else:
        del_idx = np.random.choice([x for x in range(len(negative))], size=diff, replace=False)
        del_idx.sort()
        for i in reversed(del_idx):
            neg_labels = np.vstack([neg_labels[:i, :], neg_labels[i+1:, :]])
            neg_frames = np.vstack([neg_frames[:i, :, :, :], neg_frames[i+1:, :, :, :]])
    
    frames = np.vstack([pos_frames, neg_frames])
    
    labels = np.vstack([pos_labels, neg_labels])
    
    
    return np.reshape(frames, newshape=(frames.shape[0], 64, 64, 3)), np.reshape(labels, newshape=(labels.shape[0]))


def prepare_data(batchsize, dataset='MineRLTreechop-v0'):

    data = minerl.data.make(dataset, data_dir='../Data/')
    iterator = minerl.data.BufferedBatchIter(data, buffer_target_size=5)

    frames = []
    labels = []

    for current_state, action, _, _, _ in iterator.buffered_batch_iter(batch_size=1, num_epochs=1):
        
        frame = current_state['pov'][0]
        label = action['attack'][0]
    
        frames.append(frame)
        labels.append(label)
        if len(labels) == 100:
            break

    
    frames = np.asarray(frames, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)
    print(labels.shape)
    frames, labels = equalize(frames, labels)
    print(labels.shape)
    
    dataset = MyDataset(frames, labels, transform)
    set_lengths = [len(frames)-int(0.1 * len(frames)), int(0.1 * len(frames))]
    
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

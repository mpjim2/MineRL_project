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
    '''Custom Dataset for the supervised Angle Prediction task; x: frame, y: current viewangle fo the agent'''
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


def iterate_episode(d):
    '''Iterate over the frames of a given episode to commpute vertical view angles at every step'''
    current_states ,actions ,_ ,next_states,_ = d

    assert len(current_states['pov']) == len(actions['camera']) 

    num_frames = len(current_states['pov'])

    frames = []
    angles = []
    #assumption: view angle of the agent is 0 at the beginning of every episode
    cur_angle = 0
    for i in range(num_frames):
        angles.append(cur_angle)
        frames.append(current_states['pov'][i])
        #substract user input at this state from the current view angle
        cur_angle -= actions['camera'][i][0]

        #if the angle is 90 or -90 the maximum is reached and further turning in that direction has no effect
        if cur_angle > 90:
            cur_angle = 90
        if cur_angle < -90:
            cur_angle = -90

    frames = np.asarray(frames)
    angles = np.asarray(angles, dtype=np.float32)

    return frames, angles

def extract_data(dataset : str) -> np.ndarray, np.ndarray:
    '''extract the relevant data from the given dataset by iterating each episode indiviudally; returns frames and labels as numpy arrays'''
    data = minerl.data.make(dataset, data_dir='../Data/')
    trajectories = data.get_trajectory_names()
    
    frames = []
    angles = []
    
    for traj in tqdm(trajectories):
        d = data._load_data_pyfunc(os.path.join('../Data/MineRLTreechop-v0', traj), -1, None)
        f, a = iterate_episode(d)
        frames.append(f)
        angles.append(a)

    frames = np.vstack(frames)
    angles = np.hstack(angles)

    return frames, angles

def make_datasets(batchsize: int, dataset : str) -> DataLoader, DataLoader:
    '''creates and returns the data loaders for training and validation'''
    frames, angles = extract_data(dataset)

    dataset = MyDataset(frames, angles, transform)
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

if __name__ == '__main__':
    
    x, y = make_datasets(batchsize=4, dataset='MineRLTreechop-v0')
    
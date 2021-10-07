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
    '''Custom pytorch dataset for the Classification Task; x1 and x2 are the two successive frames y is the action taken between them'''
    def __init__(self, state0 : np.ndarray, state1 : np.ndarray, actions : np.ndarray, action_index : int, transform=None):
        self.state0 = state0
        self.state1 = state1
        self.transform = transform
        self.actions = torch.tensor(actions[:, action_index], dtype=torch.float32)
        self.actions = torch.unsqueeze(self.actions, 1)

    def __getitem__(self, index):
        x0 = self.state0[index]
        x0 = self.transform(x0)
        x1 = self.state1[index]
        x1 = self.transform(x1)
        
        y = self.actions[index]
        return x0, x1, y
    
    def __len__(self):
        return len(self.state0)


def prepare_data(dataset : str, samplecount : int, batch_size : int) -> DataLoader:
    '''Extract the relevant data from the given MineRL-Dataset and make pytorch dataloaders for training, validation and testing'''
    
    data = minerl.data.make(dataset, data_dir='../Data/')
    iterator = minerl.data.BufferedBatchIter(data)

    currs = []
    nexts = []
    #Only consider these actions : ['attack', 'camera_up', 'camera_down','camera_left', 'camera_right', 'forward']
    #The classification task only considered 1 action at a time; i.e. it was a binary classification task whether the chosen action
    #was active or not
    actions = []

    #iterate the Dataset and save the current state, the next state and the action taken between them to their respective arrays
    for current_state, action, _, next_state, _ in iterator.buffered_batch_iter(batch_size=1, num_epochs=1):
        
        currs.append(current_state['pov'][0])
        nexts.append(next_state['pov'][0])

        actions.append([action['attack'][0] == 1, 
                        action['camera'][0][0] > 0,
                        action['camera'][0][0] < 0,
                        action['camera'][0][1] > 0, 
                        action['camera'][0][1] < 0,
                        action['forward'][0] == 1])
        
        if len(currs) >= samplecount:
            break

    actions  = np.asarray(actions)
    dataset = MyDataset(currs, nexts, actions, 0, transform)

    set_lengths = [int(samplecount * 0.9), int(samplecount * 0.05), int(samplecount * 0.05)]
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, set_lengths)

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
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        num_workers=1,
        shuffle=True
    )

    return train_loader, validation_loader, test_loader


if __name__ == '__main__': 

    x, y, z = prepare_data(dataset='MineRLTreechop-v0', samplecount=100, batch_size=4)
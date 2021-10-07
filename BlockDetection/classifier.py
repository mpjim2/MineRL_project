import torchvision.models as models
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim


class BlockClassifier(nn.Module):
    '''Convolutional Neural Net to classify the Block the MineRL agent is currently looking at'''
    def __init__(self, n_categories):
        
        super(BlockClassifier, self).__init__()

        #three convolutional layers
        self.convolution = nn.Sequential(
            nn.Conv2d(3, 32, 8, 4), 
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv2d(32, 32, 4, 2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv2d(32, 64, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=False),
        )

        #computing the number of neurons after the convolution given a (1, 3, 64, 64) sized input
        shape = self.convolution(torch.rand(size=(1, 3, 64, 64))).flatten().shape[0]

        #Three fully connected layers with one output neuron per class
        self.fullyconnected = nn.Sequential(

            nn.Linear(shape, shape, bias=False),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.25, inplace=False),

            nn.Linear(shape, int(shape/2), bias=False),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.25, inplace=False),

            nn.Linear(int(shape/2), n_categories, bias=False),
            nn.Softmax(dim=1)
        )

    
    def forward(self, x):

        x = self.convolution(x)
        x = x.flatten(start_dim=1)
        x = self.fullyconnected(x)
        
        return x
    

if __name__ =='__main__':

    x = torch.rand(size=(1, 3, 64, 64))

    net = BlockClassifier(7)

    y = net(x)
    print(y.shape)
import torchvision.models as models
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim


class AnglePredictor(nn.Module):
    '''Very simple Convolutional network to predict the view angle of a given frame from the MineRL dataset'''
    def __init__(self):
        
        super(AnglePredictor, self).__init__()

        self.convolution = nn.Sequential(
            nn.Conv2d(3, 8, 8, 4), 
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv2d(8, 16, 4, 2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv2d(16, 32, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=False),
        )

        self.fullyconnected = nn.Sequential(

            nn.Linear(800, 512, bias=False),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.25, inplace=False),

            nn.Linear(512, 512, bias=False),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.25, inplace=False),

            nn.Linear(512, 1, bias=False),
        )

    def forward(self, x):

        x = self.convolution(x)
        x = x.flatten(start_dim=1)
        x = self.fullyconnected(x)
        return x
    


if __name__ =='__main__':

    x = torch.rand(size=(1, 3, 64, 64))

    net = AnglePredictor()

    y = net(x)
    print(y.shape)
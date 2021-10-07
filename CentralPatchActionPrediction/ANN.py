import torchvision.models as models
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import pickle
import numpy as np



class ActionPredictor(nn.Module):

    def __init__(self):

        super(ActionPredictor, self).__init__()
        
        self.convolution = nn.Sequential(
            nn.Conv2d(3, 8, 4, 4), 
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv2d(8, 16, 2, 2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv2d(16, 32, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=False),
        )

        self.fullyconnected = nn.Sequential(

            nn.Linear(128, 128, bias=False),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.25, inplace=False),

            nn.Linear(128, 128, bias=False),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.25, inplace=False),

            nn.Linear(128, 1, bias=False),
        )


    def forward(self, x):
        
        x = x[]
        x = self.convolution(x)
        x = x.flatten(start_dim=1)
        x = self.fullyconnected(x)
        return x

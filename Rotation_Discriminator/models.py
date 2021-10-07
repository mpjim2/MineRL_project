import torchvision.models as models
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

#DCGAN Discriminator implementation from : https://github.com/csinva/gan-vae-pretrained-pytorch/blob/master/cifar10_dcgan/dcgan.py
class DCGAN_Discriminator(nn.Module):
    '''Implements the discriminator from the DCGAN model; NOT PRETRAINED; Returns a probability whethter the given frame is rotated or not'''
    def __init__(self,  nc=3, ndf=64):
        super(DCGAN_Discriminator, self).__init__()

        self.convolution = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        
        x = self.convolution(x)
        x = x.view(x.size(0))
        return x

class VGG_Discriminator(nn.Module):
    '''Discriminator based on the VGG-16 convolutional network; Returns a probability whethter the given frame is rotated or not'''
    def __init__(self):
        super(VGG_Discriminator, self).__init__()
        vgg = models.vgg16(pretrained=True)
        self.features = vgg.features

        #no further training of VGG weights
        for layer in self.features:
          for param in layer.parameters():
            param.requires_grad = False
        
        #Add custom Fully connected network, with a single output neuron to the VGG convolution
        self.output = torch.nn.Sequential(
            torch.nn.Linear(2048, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 1),
            torch.nn.Sigmoid())
    
    def forward(self, x):
        x_feat = self.features(x)
        x_feat = torch.flatten(x_feat, start_dim=1)
        x = self.output(x_feat)
        x = x.view(x.size(0))
        return x
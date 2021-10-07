import torchvision.models as models
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim


class StripeClassifier(nn.Module):
    '''Simple Convolutional network to predict the location of a given stripe of a frame'''
    def __init__(self):
        
        super(StripeClassifier, self).__init__()

        self.convolution = nn.Sequential(
            nn.Conv2d(3, 8, 8, 2), 
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv2d(8, 16, 4, 1),
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

            nn.Linear(512, 4, bias=False),
            nn.Softmax(dim=1)
        )
        

    def forward(self, x):
        '''predicts the probabilities for a SINGLE stripe of the frame'''
        x = self.convolution(x)
        x = x.flatten(start_dim=1)
        x = self.fullyconnected(x)
        return x
    
    def predict(self, img):
        '''Takes a whole frame and predicts probabilities for each Stripe'''
        ps = []
        for i in range(0, 64, 16):
            ps.append(self.forward(img[:,:,:,i:(i+16)]))
        x = torch.stack(ps)

        return x


if __name__ =='__main__':

    x = torch.rand(size=(1, 3, 64, 16))

    net = StripeClassifier()

    y = net(x)
    print(y.shape)
    x1 = torch.rand(size=(1, 3, 64, 64))
    y1 = net.predict(x1)
    print(y1.shape)

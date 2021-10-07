import torchvision.models as models
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim


class action_classifier(nn.Module):
    '''Classifier that predicts whether a specific action was taken between two succsive states from their difference in VGG Space'''
    
    def __init__(self):
        super(action_classifier, self).__init__()
        
        #First part of the model is the convolutional part of the VGG-16 network
        vgg = models.vgg16(pretrained=True)
        self.features = vgg.features

        #Convolutional part should not be trained further, as it is already pretrained
        for layer in self.features:
          for param in layer.parameters():
            param.requires_grad = False
        
        #Fully connected network with one output neuron
        self.output = torch.nn.Sequential(
            torch.nn.Linear(2048*2, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 1),
            torch.nn.Sigmoid())
    
    def forward(self, x, y):
        x_feat = self.features(x)
        y_feat = self.features(y)
        x_feat = x_feat.view(x_feat.size(0), -1)
        y_feat = y_feat.view(y_feat.size(0), -1)
        
        diff = y_feat - x_feat
        
        x = torch.cat((x_feat, diff), 1)
        
        x = self.output(x)
        return x
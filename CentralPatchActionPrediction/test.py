import torchvision.models as models
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import ANN 
import data_preparation
import numpy as np
import pickle
import os
import glob
import argparse
from datetime import datetime


if __name__ == '__main__':
    
    device = torch.device('cpu')
    net = ANN.ActionPredictor()
    net.load_state_dict(torch.load('./batchsize=128_maxepochs=100/SavedModels/MostRecent100', map_location=device))

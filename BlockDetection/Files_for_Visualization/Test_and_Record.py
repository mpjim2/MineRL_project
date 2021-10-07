import torchvision.models as models
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import classifier 
import data_preparation
import numpy as np
import pickle
import os
import glob
import argparse
from datetime import datetime
import minerl
import matplotlib.pyplot as plt
import matplotlib.animation as animation

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

def get_data(EP_id):

    data = minerl.data.make('MineRLObtainDiamond-v0', data_dir='../TestsetBlockDetection/')
    trajectories = data.get_trajectory_names()
    
    print(trajectories)
    d = data._load_data_pyfunc(os.path.join('../TestsetBlockDetection/MineRLObtainDiamond-v0', trajectories[EP_id]), -1, None)
    states, _, _, _, _ = d

    frames = states['pov']
    return frames

def classify_frames(frames, net):

    predicitons = []
    for frame in frames: 
        prediction = net(frame)
        predicitons.append(prediciton)
    
    return predicitons


def visualize(frames, net, labels):

    def run2(c):
    
        l1.set_array(frames[c])

        output = net(torch.unsqueeze(transform(frames[c]), 0))
        output = output.detach().numpy()[0]
        for i, b in enumerate(h1):
            b.set_height(output[i])

        return l1, h1


    labels = labels
    x =  [1, 1, 1, 1, 1, 1, 1]

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)

    fig = plt.figure()

    p1 = fig.add_subplot(121)
    p1.axis('off')
    
    p2 = fig.add_subplot(122)
    p2.grid()
    p2.set_ylim([0, 1])

    
    plt.setp(p2.get_xticklabels(), rotation=45, horizontalalignment='right')

    h1 = p2.bar(labels, x)
    l1 = p1.imshow(frames[0])
    ani2 = animation.FuncAnimation(fig, run2 ,interval=30, frames=len(frames),  blit=False)

    ani2.save('TestVideo_3.mp4', writer=writer)

    plt.show()


if __name__ == '__main__':

    with open('./BlockIDdict.pickle', 'rb') as f:
        blockids = pickle.load(f)
    
    labels = []

    for x in blockids:
        labels.append(x)

    device = torch.device('cpu')
    net = classifier.BlockClassifier(7)
    net.load_state_dict(torch.load('./batchsize=128_maxepochs=100_learningrate=0.0001_SamplesPerClass=6000_19-09-2021_19:42:38/SavedModels/HighestAccuracy', map_location=device))

    test_frames = get_data(1)
    
    
    visualize(test_frames, net, labels)
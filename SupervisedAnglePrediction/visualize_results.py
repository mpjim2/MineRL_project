import torchvision.models as models
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import model 
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

def iterate_episode(d):
    current_states ,actions ,_ ,next_states,_ = d

    assert len(current_states['pov']) == len(actions['camera']) 

    num_frames = len(current_states['pov'])

    frames = []
    angles = []

    cur_angle = 0
    for i in range(num_frames):
        angles.append(cur_angle)
        frames.append(current_states['pov'][i])
        cur_angle -= actions['camera'][i][0]
        if cur_angle > 90:
            cur_angle = 90
        if cur_angle < -90:
            cur_angle = -90

    frames = np.asarray(frames)
    angles = np.asarray(angles, dtype=np.float32)

    return frames, angles


def get_data(EP_id):

    data = minerl.data.make('MineRLTreechop-v0', data_dir='../TestData/')
    trajectories = data.get_trajectory_names()

    d = data._load_data_pyfunc(os.path.join('../TestData/MineRLTreechop-v0', trajectories[EP_id]), -1, None)
    f, a = iterate_episode(d)

    return f, a 

def predict_angles(net, frames):

    predict_angles = []
    for frame in frames: 

        predicted_angle = net(torch.unsqueeze(transform(frame), axis=0)).detach().numpy()[0][0]
        predict_angles.append(predicted_angle)
    
    return predict_angles

def run1(c):
        l1.set_array(frames[c])


def visualize(frames, groundtruth, predictions):

    def run2(c, x, y1, y2, line):
    
        if c >= 10:
            if ((c+10) < len(x)):
                l2.set_data(x[(c-10):(c+10)], y1[(c-10):(c+10)])
                l2.axes.axis([x[c-10], x[c+10], -100, 100])

                l4.set_data(x[(c-10):(c+10)], y2[(c-10):(c+10)])
                l4.axes.axis([x[c-10], x[c+10], -100, 100])
            else:
                l2.set_data(x[(c-10):], y1[(c-10):])
                l2.axes.axis([x[c-10], x[-1], -100, 100]) 

                l4.set_data(x[(c-10):], y2[(c-10):])
                l4.axes.axis([x[c-10], x[-1], -100, 100]) 
        else:
            l2.set_data(x[:(c+10)], y1[0:(c+10)])
            l2.axes.axis([x[0], x[c+10], -100, 100])

            l4.set_data(x[:(c+10)], y2[0:(c+10)])
            l4.axes.axis([x[0], x[c+10], -100, 100])
        
        l5.set_xdata(c)
        asp = np.diff(l2.axes.get_xlim())[0] / np.diff(l2.axes.get_ylim())[0]
        l2.axes.set_aspect(asp)
        l1.set_array(frames[c])
        
        return l2, l1, l4, l5,

    x = [x for x in range(len(frames))]
    y1 = groundtruth
    y2 = predictions
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)

    fig = plt.figure()

    p1 = fig.add_subplot(121)
    p1.axis('off')
    
    p2 = fig.add_subplot(122)
    p2.grid()

    
    l1 = p1.imshow(frames[0])
    l2, = p2.plot(x, y1)
    l4, = p2.plot(x, y2)
    l5 = p2.axvline(0, c='black')

    p2.legend([l2, l4], ['Groundtruth', 'Model-Prediction'], bbox_to_anchor=(0,1.02), loc="lower left")

    ani2 = animation.FuncAnimation(fig, run2 ,interval=30, frames=len(frames), fargs=[x, y1, y2, l2],  blit=False)

    ani2.save('im.mp4', writer=writer)

    plt.show()


if __name__ == '__main__':

    device = torch.device('cpu')
    net = model.AnglePredictor()
    net.load_state_dict(torch.load('./batchsize=128_maxepochs=100/SavedModels/MostRecent100', map_location=device))

    test_frames, groundtruth = get_data(0)
    
    predictions = predict_angles(net, test_frames)

    
    visualize(test_frames, groundtruth, predictions)